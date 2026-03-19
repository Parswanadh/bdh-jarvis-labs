"""
FINAL WORKING TRAINING SCRIPT - CPU VERSION
============================================
This version works on CPU when CUDA is not available.
Based on train_final.py but adapted for CPU.
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
print("FINAL WORKING TRAINING (CPU VERSION)")
print("="*80)

# Handle device - use CUDA if available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device('cpu')
    print("\n[DEVICE] Using CPU (CUDA not available)")

print(f"[PyTorch] {torch.__version__}")

torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================================
# LOAD TEACHER (FP16)
# ============================================================================
print("\n[1/3] Loading TEACHER (FP16)...")

teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Teacher in appropriate dtype
teacher_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-0.8B",
    torch_dtype=teacher_dtype,
    device_map="cuda:0" if torch.cuda.is_available() else None,
    trust_remote_code=True
).eval()

teacher_vocab = len(teacher_tokenizer)
print(f"  Vocab: {teacher_vocab:,}")
if torch.cuda.is_available():
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# CREATE STUDENT (FP32 for stability)
# ============================================================================
print("\n[2/3] Creating STUDENT (FP32)...")

bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=192,      # Small for VRAM
    n_layer=3,       # Small for VRAM
    n_head=4,
    ffn_dim=768,
    dropout=0.0,
    max_seq_len=80,  # Short for VRAM
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)
student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student: {student_params:,} params ({'FP32' if device.type == 'cpu' else 'FP32'})")
if torch.cuda.is_available():
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[3/3] Loading data...")

data_file = Path("data/tinystories.txt")
with open(data_file, 'r', encoding='utf-8') as f:
    all_stories = [line.strip() for line in f if line.strip()]

stories = all_stories[:50000]

def collate_batch(batch_texts):
    encodings = teacher_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=80,
        return_tensors='pt'
    )
    return encodings

dataloader = DataLoader(
    stories,
    batch_size=6,  # Safe batch size
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Stories: {len(stories):,}")
print(f"  Batches: {len(dataloader):,}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Expected initial loss: ~12.4 (ln vocab)")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

session_start = time.time()
step = 0
last_checkpoint = session_start

teacher_model.eval()
student_model.train()

try:
    for epoch in range(50):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if time.time() - session_start >= 7200:
                print("\n[SESSION] 2 hours complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)

            # Teacher forward
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            # Student forward
            student_logits, _ = student_model(input_ids)

            # Loss computation - STABLE
            shift_logits = student_logits[:, :-1, :]  # [B, T-1, V]
            shift_labels = input_ids[:, 1:]            # [B, T-1]

            B, T, V = shift_logits.shape

            # Reshape (using reshape not view for safety)
            shift_logits = shift_logits.reshape(B * T, V)
            shift_labels = shift_labels.reshape(B * T)

            # Simple CE loss (no masking needed for this approach)
            # Filter out zero labels (padding)
            valid = shift_labels != 0
            if valid.sum() > 0:
                shift_logits = shift_logits[valid]
                shift_labels = shift_labels[valid]
                loss = F.cross_entropy(shift_logits, shift_labels)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: Bad loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            batches += 1

            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                alloc = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                lr = optimizer.param_groups[0]['lr']

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                      f"{'VRAM: {alloc:.1f}GB | ' if torch.cuda.is_available() else ''}LR: {lr:.2e} | {elapsed/60:.0f}m")

                if avg_loss < 11.0:
                    print(f"       ↓ Learning!")
                if avg_loss < 8.0:
                    print(f"       ↓ Great progress!")
                if avg_loss < 5.0:
                    print(f"       ★ Almost there!")
                if avg_loss < 2.5:
                    print(f"       ★★ TARGET REACHED!")

            # Checkpoint
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/final/latest.pt')
                print(f"\n[CHECKPOINT] Saved")
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

except StopIteration:
    pass

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Time: {(time.time() - session_start) / 60:.0f} minutes")
print(f"Steps: {step}")
print(f"Final loss: {avg_loss:.4f}")
print(f"Checkpoint: checkpoints/final/latest.pt")