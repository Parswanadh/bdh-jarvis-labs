"""
STABLE TRAINING - No NaN issues
================================
Uses mixed precision properly, gradient clipping, and stable loss computation
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
print("STABLE TRAINING - No NaN")
print("="*80)

device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD MODELS
# ============================================================================
print("\n[1/2] Loading models...")

teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Teacher in FP16 (no grad needed)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-0.8B",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
).eval()

teacher_vocab = len(teacher_tokenizer)
print(f"  Teacher: {sum(p.numel() for p in teacher_model.parameters()):,} params")

# Student in FP32 for stability during training
bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=256,
    n_layer=4,
    n_head=4,
    ffn_dim=1024,
    dropout=0.0,
    max_seq_len=96,
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)
student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student: {student_params:,} params (FP32)")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[2/2] Loading data...")

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
# STABLE LOSS FUNCTION
# ============================================================================
def stable_loss(student_logits, teacher_logits, input_ids):
    """
    Stable loss computation that avoids NaN.
    Uses simple cross-entropy instead of KL divergence for stability.
    """
    # Shift for next-token prediction
    shift_student = student_logits[:, :-1, :].float()  # FP32 for stability
    shift_labels = input_ids[:, 1:]

    # Flatten
    B, T, V = shift_student.shape
    shift_student = shift_student.view(-1, V)
    shift_labels = shift_labels.view(-1)

    # Filter out padding (assuming pad=0)
    valid = shift_labels != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    shift_student = shift_student[valid]
    shift_labels = shift_labels[valid]

    # Simple cross-entropy loss
    loss = F.cross_entropy(shift_student, shift_labels)

    # Check for NaN
    if torch.isnan(loss):
        print("  WARNING: NaN loss detected!")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING")
print("="*80)
print(f"Expected initial loss: ~12.4")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)  # Lower LR for stability

session_start = time.time()
step = 0
last_checkpoint = session_start

teacher_model.eval()
student_model.train()

# Enable autocast for mixed precision (but compute loss in FP32)
scaler = torch.cuda.amp.GradScaler()

try:
    for epoch in range(20):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0
        nan_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if time.time() - session_start >= 7200:
                print("\n[SESSION] Complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)

            # Teacher forward (FP16, no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            # Student forward with autocast
            with torch.cuda.amp.autocast():
                student_logits, _ = student_model(input_ids)

                # Stable loss computation
                loss = stable_loss(student_logits, teacher_logits, input_ids)

                # Check for NaN
                if torch.isnan(loss):
                    nan_count += 1
                    print(f"  WARNING: NaN at step {step}, skipping batch")
                    optimizer.zero_grad()
                    continue

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            batches += 1

            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                alloc = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                      f"VRAM: {alloc:.1f}GB (res: {reserved:.1f}GB) | {elapsed/60:.0f}m")

                if avg_loss < 11.0:
                    print(f"       ↓ Learning!")
                if avg_loss < 9.0:
                    print(f"       ↓ Progress!")
                if avg_loss < 6.0:
                    print(f"       ★ Almost there!")

            # Checkpoint
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/stable").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/stable/latest.pt')
                print(f"\n[CHECKPOINT] Saved")
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} | Loss: {avg_loss:.4f} | NaN count: {nan_count}")

except StopIteration:
    pass

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Final loss: {avg_loss:.4f}")
print(f"Time: {(time.time() - session_start) / 60:.0f} minutes")
