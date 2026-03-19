"""
RESUME TRAINING FOR 2 MORE HOURS
Continues from checkpoint and trains for another 2 hours
Based on train_final.py but with checkpoint resuming
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
print("RESUMING BDH TRAINING FOR 2 MORE HOURS")
print("="*80)

device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
torch.cuda.empty_cache()

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================
print("\n[1/3] Loading checkpoint...")

checkpoint_path = Path("checkpoints/final/latest.pt")
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    print("Please run train_final.py first to generate a checkpoint.")
    sys.exit(1)

print(f"  Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract checkpoint info
if 'model' in checkpoint:
    model_state = checkpoint['model']
    start_step = checkpoint.get('step', 0)
    checkpoint_loss = checkpoint.get('loss', 0.0)
    print(f"  Resuming from step {start_step} with loss {checkpoint_loss:.4f}")
else:
    # Fallback - assume entire checkpoint is model state
    model_state = checkpoint
    start_step = 0
    checkpoint_loss = 12.4  # Default initial loss
    print(f"  Warning: Unexpected checkpoint format, starting from scratch")

# ============================================================================
# LOAD TEACHER (FP16)
# ============================================================================
print("\n[2/3] Loading TEACHER (FP16)...")

teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen3.5-0.8B",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
).eval()

teacher_vocab = len(teacher_tokenizer)
print(f"  Vocab: {teacher_vocab:,}")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# CREATE STUDENT (FP32 for stability)
# ============================================================================
print("\n[3/3] Creating STUDENT (FP32)...")

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

# Load model state from checkpoint
if model_state is not None:
    try:
        missing_keys, unexpected_keys = student_model.load_state_dict(model_state, strict=False)
        print(f"  Model state loaded from checkpoint")
        if missing_keys:
            print(f"  Info: {len(missing_keys)} missing keys (will be initialized)")
        if unexpected_keys:
            print(f"  Info: {len(unexpected_keys)} unexpected keys (will be ignored)")
    except Exception as e:
        print(f"  ERROR loading model state: {e}")
        print(f"  Continuing with randomly initialized model")
else:
    print(f"  No model state to load - starting with random initialization")

student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student: {student_params:,} params (FP32)")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[Data] Loading TinyStories...")

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
print("STARTING TRAINING (RESUMED)")
print("="*80)
print(f"Expected initial loss: ~12.4 (ln vocab)")
print(f"Resuming from step: {start_step}")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

session_start = time.time()
step = start_step  # CRITICAL: Start from checkpoint step
last_checkpoint = session_start

teacher_model.eval()
student_model.train()

# Calculate target end time (2 hours from NOW, not from original start)
target_end_time = session_start + 7200
print(f"Target completion: {time.ctime(target_end_time)}")

try:
    for epoch in range(50):  # Enough epochs for 2 hours
        # Check if we've reached our time limit
        if time.time() >= target_end_time:
            print("\n[SESSION] 2 hours complete!")
            break

        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Double-check time limit
            if time.time() >= target_end_time:
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
                elapsed_hours = elapsed / 3600
                remaining_hours = (target_end_time - time.time()) / 3600
                alloc = torch.cuda.memory_allocated(0) / 1e9
                lr = optimizer.param_groups[0]['lr']

                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"VRAM: {alloc:.1f}GB | LR: {lr:.2e} | "
                      f"Elapsed: {elapsed_hours:.1f}h | Remaining: {remaining_hours:.1f}h")

                if avg_loss < 11.0:
                    print(f"       ↓ Learning!")
                if avg_loss < 8.0:
                    print(f"       ↓ Great progress!")
                if avg_loss < 5.0:
                    print(f"       ↓ Excellent progress!")
                if avg_loss < 3.0:
                    print(f"       ↓ Getting close to target!")
                if avg_loss < 2.5:
                    print(f"       ★★ NEAR TARGET!")

            # Checkpoint every 30 minutes
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
                current_avg_loss = epoch_loss / max(batches, 1)

                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': current_avg_loss
                }, 'checkpoints/final/latest.pt')
                print(f"\n[CHECKPOINT] Saved at step {step}")
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

except StopIteration:
    pass

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
total_time = time.time() - session_start
print(f"Time: {total_time / 60:.0f} minutes")
print(f"Steps: {step}")
print(f"Steps in this session: {step - start_step}")
print(f"Final loss: {avg_loss:.4f}")
print(f"Checkpoint: checkpoints/final/latest.pt")

# Save final checkpoint
torch.save({
    'step': step,
    'model': student_model.state_dict(),
    'loss': avg_loss
}, 'checkpoints/final/latest.pt')
print(f"Final checkpoint saved.")