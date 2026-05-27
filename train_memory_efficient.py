"""
TRULY MEMORY-EFFICIENT TRAINING
=================================
No CPU offloading - everything stays on GPU
Uses chunked loss computation to handle large vocab
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

print("="*80)
print("MEMORY-EFFICIENT TRAINING - NO OFFLOADING")
print("="*80)

device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()

# ============================================================================
# LOAD TEACHER IN FP16
# ============================================================================
print("\n[1/3] Loading TEACHER (FP16, CPU first)...")

teacher_path = Path("Qwen3.5-0.8B")
teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Load teacher to CPU first to save GPU memory during init
teacher_model = AutoModelForCausalLM.from_pretrained(
    str(teacher_path),
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()

teacher_vocab = len(teacher_tokenizer)
print(f"  Teacher vocab: {teacher_vocab:,}")
print(f"  Teacher on CPU (will move to GPU for inference only)")

# ============================================================================
# CREATE SMALL STUDENT MODEL
# ============================================================================
print("\n[2/3] Creating COMPACT STUDENT...")

# Very compact model to save memory
bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=128,      # Very small!
    n_layer=3,       # Very few layers!
    n_head=4,
    ffn_dim=512,
    dropout=0.0,
    max_seq_len=64,  # Short sequences!
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)
student_params = sum(p.numel() for p in student_model.parameters())

# Convert to FP16
student_model = student_model.half()
print(f"  Student: {student_params:,} params (FP16)")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[3/3] Loading data...")

data_file = Path("data/tinystories.txt")
with open(data_file, 'r', encoding='utf-8') as f:
    all_stories = [line.strip() for line in f if line.strip()]

print(f"  Total: {len(all_stories):,} stories")
stories = all_stories[:50000]

def collate_batch(batch_texts):
    encodings = teacher_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=64,  # Short!
        return_tensors='pt'
    )
    return encodings

dataloader = DataLoader(
    stories,
    batch_size=2,  # Very small batch!
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Batches: {len(dataloader):,}")
print(f"  Batch size: 2 (small for VRAM)")

# ============================================================================
# CHUNKED LOSS FUNCTION (handles large vocab without OOM)
# ============================================================================
def chunked_cross_entropy(logits, labels, ignore_index=-100, chunk_size=1000):
    """
    Compute cross-entropy in chunks to avoid OOM with large vocab.
    logits: [N, V] where V can be 248K
    labels: [N]
    """
    total_loss = 0.0
    total_tokens = 0

    N, V = logits.shape

    # Process in chunks of vocab
    for start_idx in range(0, V, chunk_size):
        end_idx = min(start_idx + chunk_size, V)

        # Logits for this vocab chunk
        chunk_logits = logits[:, start_idx:end_idx]  # [N, chunk_size]

        # Adjust labels for this chunk
        # Labels in range [start_idx, end_idx) map to [0, chunk_size)
        # Labels outside this range are set to ignore_index
        chunk_labels = labels.clone()
        mask = (chunk_labels >= start_idx) & (chunk_labels < end_idx)
        chunk_labels[~mask] = ignore_index
        chunk_labels[mask] -= start_idx

        # Compute loss for this chunk
        chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, ignore_index=ignore_index, reduction='sum')

        # Count valid tokens
        valid_tokens = (chunk_labels != ignore_index).sum().item()

        total_loss += chunk_loss.item()
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Expected initial loss: {torch.log(torch.tensor(float(teacher_vocab))):.2f}")
print("Using chunked loss computation to handle 248K vocab")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

session_start = time.time()
step = 0
last_checkpoint = session_start

student_model.train()

try:
    for epoch in range(20):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if time.time() - session_start >= 7200:
                print("\n[SESSION] Complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Move teacher to GPU briefly, then back to CPU
            with torch.no_grad():
                teacher_model.to(device)
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits
                teacher_model.to('cpu')  # Free GPU memory!
                torch.cuda.empty_cache()

            # Student forward (stays on GPU)
            student_logits, _ = student_model(input_ids)

            # Prepare labels
            shift_logits = student_logits[:, :-1, :].contiguous()  # [B, T-1, V]
            shift_labels = input_ids[:, 1:].contiguous()           # [B, T-1]
            shift_mask = attention_mask[:, 1:].contiguous()       # [B, T-1]

            # Flatten
            B, T, V = shift_logits.shape
            shift_logits = shift_logits.view(-1, V)  # [N, V]
            shift_labels = shift_labels.view(-1)     # [N]
            shift_mask = shift_mask.view(-1)         # [N]

            # Filter valid tokens
            valid = shift_mask == 1
            shift_logits = shift_logits[valid]      # [N_valid, V]
            shift_labels = shift_labels[valid]      # [N_valid]

            # Compute loss with chunking (handles 248K vocab!)
            loss = chunked_cross_entropy(shift_logits, shift_labels, chunk_size=1000)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss
            batches += 1

            # Progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                alloc = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                      f"VRAM: {alloc:.2f}GB alloc | Time: {elapsed/60:.0f}m")

                if alloc > 6.0:
                    print(f"       ⚠ High VRAM - consider reducing batch size")

                if avg_loss < 11.0:
                    print(f"       ↓ Learning!")
                if avg_loss < 8.0:
                    print(f"       ↓ Great progress!")

            # Checkpoint
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/efficient").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/efficient/latest.pt')
                print(f"\n[CHECKPOINT] Saved")
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

except StopIteration:
    pass

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Final loss: {avg_loss:.4f}")
print(f"Time: {(time.time() - session_start) / 60:.0f} minutes")
