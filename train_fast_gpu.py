"""
FAST GPU TRAINING - Both models on GPU
======================================
Uses chunked loss to handle 248K vocab without OOM
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
print("FAST GPU TRAINING - Both models on GPU")
print("="*80)

device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD TEACHER ON GPU
# ============================================================================
print("\n[1/2] Loading TEACHER on GPU (FP16)...")

teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-0.8B",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
).eval()

teacher_vocab = len(teacher_tokenizer)
print(f"  Teacher vocab: {teacher_vocab:,}")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# CREATE STUDENT ON GPU
# ============================================================================
print("\n[2/2] Creating STUDENT on GPU...")

bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=192,
    n_layer=4,
    n_head=4,
    ffn_dim=768,
    dropout=0.0,
    max_seq_len=96,
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)
student_model = student_model.half().train()  # FP16 for speed & memory

student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student: {student_params:,} params (FP16)")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"  VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[DATA] Loading TinyStories...")

data_file = Path("data/tinystories.txt")
with open(data_file, 'r', encoding='utf-8') as f:
    all_stories = [line.strip() for line in f if line.strip()]

stories = all_stories[:50000]

def collate_batch(batch_texts):
    encodings = teacher_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors='pt'
    )
    return encodings

dataloader = DataLoader(
    stories,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Stories: {len(stories):,}")
print(f"  Batches: {len(dataloader):,}")

# ============================================================================
# CHUNKED LOSS (handles 248K vocab efficiently)
# ============================================================================
def chunked_loss(student_logits, teacher_logits, labels, vocab_size=248077, chunk_size=500):
    """
    Compute KL divergence loss in chunks to avoid OOM.
    This is the KEY to handling large vocab on 8GB GPU!
    """
    B, T, V = student_logits.shape

    # Shift for next-token prediction: predict t+1 using info up to t
    shift_student = student_logits[:, :-1, :]  # [B, T-1, V]
    shift_teacher = teacher_logits[:, :-1, :]  # [B, T-1, V]
    shift_labels = labels[:, 1:]               # [B, T-1]

    # Create attention mask for valid positions (not padding)
    # We'll use labels != 0 to identify valid positions
    mask = (shift_labels != 0)  # [B, T-1]

    # Get valid token counts
    num_valid = mask.sum().item()
    if num_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Expand mask to vocab dimension
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, V)  # [B, T-1, V]

    # Apply mask - get only valid positions
    valid_student = shift_student[mask_expanded]  # [num_valid, V]
    valid_teacher = shift_teacher[mask_expanded]  # [num_valid, V]
    valid_labels = shift_labels[mask]              # [num_valid]

    # Now compute loss in vocab chunks
    temperature = 2.0
    total_loss = 0.0
    total_chunks = 0

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)

        # Get vocab chunk
        student_chunk = valid_student[:, start:end]  # [num_valid, chunk_size]
        teacher_chunk = valid_teacher[:, start:end]  # [num_valid, chunk_size]

        # Find labels that fall in this chunk
        in_chunk = (valid_labels >= start) & (valid_labels < end)
        if in_chunk.sum() == 0:
            continue  # Skip if no labels in this chunk

        # Get logits for labels in this chunk
        student_logits_chunk = student_chunk[in_chunk]  # [N_in_chunk, chunk_size]
        teacher_logits_chunk = teacher_chunk[in_chunk]  # [N_in_chunk, chunk_size]

        # Adjust labels to chunk-local indices
        chunk_labels = valid_labels[in_chunk] - start  # [N_in_chunk]

        # KL divergence
        student_log_probs = F.log_softmax(student_logits_chunk / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_chunk / temperature, dim=-1)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        total_loss += kl
        total_chunks += 1

    if total_chunks == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / total_chunks
        teacher_chunk = shift_teacher[:, start:end]  # [N_valid, chunk_size]

        # Labels in this chunk?
        chunk_labels = shift_labels.clone()
        mask = (chunk_labels >= start) & (chunk_labels < end)

        if mask.sum() == 0:
            continue  # Skip if no labels in this chunk

        chunk_labels = chunk_labels[mask] - start  # Reindex to [0, chunk_size)

        # Softmax over chunk (not full vocab, but approximates)
        student_log_probs = F.log_softmax(student_chunk[mask] / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_chunk[mask] / temperature, dim=-1)

        # KL divergence
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        total_loss += kl

    return total_loss

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING - Both models on GPU")
print("="*80)
print(f"Expected initial loss: ~12.4 (ln({teacher_vocab:,}))")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

session_start = time.time()
step = 0
last_checkpoint = session_start

teacher_model.eval()  # Keep teacher in eval mode
student_model.train()

try:
    for epoch in range(30):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if time.time() - session_start >= 7200:  # 2 hours
                print("\n[SESSION] Complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Both on GPU - fast!
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            student_logits, _ = student_model(input_ids)

            # Chunked loss (handles 248K vocab)
            loss = chunked_loss(
                student_logits,
                teacher_logits,
                input_ids,
                vocab_size=teacher_vocab,
                chunk_size=500
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            # Progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                alloc = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                      f"VRAM: {alloc:.1f}GB (res: {reserved:.1f}GB) | {elapsed/60:.0f}m")

                if avg_loss < 11.0:
                    print(f"       ↓ Learning!")
                if avg_loss < 9.0:
                    print(f"       ↓ Great progress!")
                if avg_loss < 6.0:
                    print(f"       ★ Almost there!")

            # Checkpoint
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/fast_gpu").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/fast_gpu/latest.pt')
                print(f"\n[CHECKPOINT] Saved at step {step}")
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
print(f"Checkpoint: checkpoints/fast_gpu/latest.pt")
