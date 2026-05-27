"""
MEMORY-EFFICIENT TRAINING - Fixed Version
========================================
Lower VRAM usage, proper loss tracking
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

print("="*80)
print("MEMORY-EFFICIENT BDH TRAINING")
print("="*80)

# Device
device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Clear any existing cache
torch.cuda.empty_cache()

# ============================================================================
# LOAD TEACHER - FP16 to save VRAM
# ============================================================================
print("\n[1/3] Loading TEACHER to GPU (FP16)...")

teacher_path = Path("Qwen3.5-0.8B")
if not teacher_path.exists():
    print(f"ERROR: Teacher model not found")
    sys.exit(1)

teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    str(teacher_path),
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()

teacher_params = sum(p.numel() for p in teacher_model.parameters())
teacher_vocab = len(teacher_tokenizer)

print(f"  OK - Teacher: {teacher_params:,} params")
print(f"  Allocated VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"  Reserved VRAM: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# ============================================================================
# CREATE SMALLER STUDENT MODEL
# ============================================================================
print("\n[2/3] Creating COMPACT STUDENT...")

bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=256,      # Reduced from 384
    n_layer=4,       # Reduced from 6
    n_head=4,
    ffn_dim=1024,    # Reduced from 1536
    dropout=0.0,
    max_seq_len=128, # Reduced from 256
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)
student_params = sum(p.numel() for p in student_model.parameters())

print(f"  OK - Student: {student_params:,} params")
print(f"  Allocated VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Convert student to FP16 to save memory
student_model = student_model.half()
print(f"  Converted to FP16")
print(f"  Allocated VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA - SMALLER SAMPLE
# ============================================================================
print("\n[3/3] Loading TinyStories data...")

data_file = Path("data/tinystories.txt")
if not data_file.exists():
    print("  WARNING: data/tinystories.txt not found")
    stories = ["Once upon a time " * 20] * 1000
else:
    with open(data_file, 'r', encoding='utf-8') as f:
        all_stories = [line.strip() for line in f if line.strip()]
    print(f"  Total: {len(all_stories):,} stories")
    stories = all_stories[:50000]  # 50K stories

print(f"  Using {len(stories):,} stories")

# ============================================================================
# DATALOADER
# ============================================================================
def collate_batch(batch_texts):
    encodings = teacher_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,  # Reduced from 256
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

print(f"\n[ batches: {len(dataloader):,} ]")
print(f"[ effective batch size: 4 ]")

# ============================================================================
# OPTIMIZER
# ============================================================================
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.01)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"\nExpected initial loss: ~{torch.log(torch.tensor(float(teacher_vocab))):.2f} (ln({teacher_vocab:,}))")
print("This is NORMAL for random initialization!\n")

session_start = time.time()
step = 0
checkpoint_freq = 1800  # 30 min
last_checkpoint = session_start

student_model.train()
teacher_model.eval()

try:
    for epoch in range(50):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Check 2-hour limit
            if time.time() - session_start >= 7200:
                print("\n[SESSION] 2 hours complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Teacher forward (no grad, FP16)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits.float()  # Cast to FP32 for stability

            # Student forward (FP16)
            student_logits, _ = student_model(input_ids.float() if input_ids.dtype == torch.float16 else input_ids)

            # Loss computation
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Set padding to ignore_index
            shift_labels[shift_mask == 0] = -100

            B, T, V = shift_logits.shape
            shift_logits = shift_logits.view(-1, V)
            shift_labels = shift_labels.view(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            batches += 1

            # Progress every 25 batches
            if (batch_idx + 1) % 25 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                      f"VRAM: {allocated:.2f}GB alloc, {reserved:.2f}GB res | Time: {elapsed/60:.0f}m")

                # Loss decrease indicator
                if avg_loss < 11.0:
                    print(f"       ↓ Loss decreasing! Good progress!")
                elif avg_loss < 10.0:
                    print(f"       ↓ Model learning!")
                elif avg_loss < 8.0:
                    print(f"       ↓ Excellent progress!")

            # Checkpoint
            if time.time() - last_checkpoint >= checkpoint_freq:
                Path("checkpoints/minimal").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/minimal/latest.pt')
                print(f"\n[CHECKPOINT] Saved at step {step}")
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} complete | Loss: {avg_loss:.4f}")

except StopIteration:
    pass
except KeyboardInterrupt:
    print("\n[INTERRUPTED]")

# Final
total_time = time.time() - session_start
final_loss = epoch_loss / max(batches, 1)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Time: {total_time/60:.1f} minutes")
print(f"Steps: {step}")
print(f"Final loss: {final_loss:.4f}")
print(f"Checkpoint: checkpoints/minimal/latest.pt")
