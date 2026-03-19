#!/usr/bin/env python
"""
CPU COMPATIBLE RESUME TRAINING SCRIPT
Continues training from existing checkpoint on CPU
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time
import os

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

print("="*80)
print("RESUMING BDH TRAINING (CPU MODE)")
print("="*80)

# Device configuration - force CPU
device = torch.device('cpu')
print("\n[DEVICE] Using CPU (forced)")

print(f"[PyTorch] {torch.__version__}")

print("\n[1/3] Loading models and checkpoint...")

# Load tokenizer
try:
    teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    print(f"  Tokenizer loaded successfully")
except Exception as e:
    print(f"  ERROR loading tokenizer: {e}")
    # Fallback to a basic tokenizer if needed
    from transformers import AutoTokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    print(f"  Using fallback tokenizer")

# Load checkpoint to get configuration
checkpoint_path = Path("checkpoints/final/latest.pt")
print(f"  Loading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"  Checkpoint loaded successfully")
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    # Extract information from checkpoint
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        saved_step = checkpoint.get('step', 0)
        saved_loss = checkpoint.get('loss', 0.0)
        print(f"  Resuming from step {saved_step} with loss {saved_loss:.4f}")
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
        saved_step = checkpoint.get('step', 0)
        saved_loss = checkpoint.get('loss', 0.0)
        print(f"  Resuming from step {saved_step} with loss {saved_loss:.4f}")
    else:
        # Assume the checkpoint IS the state dict
        model_state = checkpoint
        saved_step = 0
        saved_loss = 12.4  # Default initial loss
        print(f"  Assuming checkpoint is state dict")
        print(f"  Starting from step 0 with estimated initial loss")

except Exception as e:
    print(f"  ERROR loading checkpoint: {e}")
    print(f"  Falling back to default configuration")
    model_state = None
    saved_step = 0
    saved_loss = 12.4  # Initial loss

# Determine vocabulary size from tokenizer
teacher_vocab = len(teacher_tokenizer)
print(f"  Vocabulary size: {teacher_vocab:,}")

# Create student model configuration (matching original training)
bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=192,      # Matches original
    n_layer=3,       # Matches original
    n_head=4,
    ffn_dim=768,
    dropout=0.0,
    max_seq_len=80,  # Matches original
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

# Create model
student_model = MultiScaleBDH(bdh_config).to(device)

# Load model state if available
if model_state is not None:
    try:
        # Try to load with strict=False to allow for minor mismatches
        missing_keys, unexpected_keys = student_model.load_state_dict(model_state, strict=False)
        print(f"  Model state loaded")
        if missing_keys:
            print(f"  Info: {len(missing_keys)} missing keys (will be initialized)")
        if unexpected_keys:
            print(f"  Info: {len(unexpected_keys)} unexpected keys (will be ignored)")
    except Exception as e:
        print(f"  ERROR loading model state: {e}")
        print(f"  Continuing with randomly initialized model")
else:
    print(f"  No model state in checkpoint - starting with random initialization")

student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student model: {student_params:,} parameters")

# Load teacher model (FP32 for CPU stability)
print("\n[2/3] Loading teacher model...")
try:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen3.5-0.8B",
        torch_dtype=torch.float32,  # FP32 for CPU stability
        trust_remote_code=True
    ).eval()
    print(f"  Teacher model loaded (FP32 for CPU)")
except Exception as e:
    print(f"  WARNING: Could not load teacher model: {e}")
    print(f"  Using fallback approach...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen3.5-0.8B",
        trust_remote_code=True
    ).eval()
    teacher_model = teacher_model.to(device)
    print(f"  Teacher model loaded via fallback")

teacher_model.eval()

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[3/3] Loading dataset...")

data_file = Path("data/tinystories.txt")
with open(data_file, 'r', encoding='utf-8') as f:
    all_stories = [line.strip() for line in f if line.strip()]

stories = all_stories[:50000]  # Same as original training

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
    batch_size=6,  # Same as original
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Loaded {len(stories):,} stories")
print(f"  {len(dataloader):,} batches per epoch")

# ============================================================================
# SETUP TRAINING
# ============================================================================
print("\n" + "="*80)
print("CONFIGURING TRAINING")
print("="*80)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

# Training state
session_start = time.time()
step = saved_step  # Resume from saved step
last_checkpoint = session_start
best_loss = saved_loss

print(f"  Resuming from step: {step}")
print(f"  Previous best loss: {best_loss:.4f}")
print(f"  Target: Train for 2 more hours (until {(time.time() + 7200)/60:.0f} minutes from start)")

teacher_model.eval()
student_model.train()

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

try:
    for epoch in range(100):  # Large number, will break via time check
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Check if 2 hours have elapsed
            if time.time() - session_start >= 7200:
                print("\n[SESSION] 2 hours complete!")
                raise StopIteration

            step += 1

            # Move batch to device
            input_ids = batch['input_ids'].to(device)

            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            # Student forward pass
            student_logits, _ = student_model(input_ids)

            # Loss computation - next token prediction
            shift_logits = student_logits[:, :-1, :]  # [B, T-1, V]
            shift_labels = input_ids[:, 1:]            # [B, T-1]

            # Handle tensor reshaping safely
            B, T, V = shift_logits.shape
            shift_logits = shift_logits.reshape(B * T, V)
            shift_labels = shift_labels.reshape(B * T)

            # Create loss - simple cross-entropy
            # Filter out padding tokens (assuming 0 is pad token)
            valid_mask = shift_labels != 0
            if valid_mask.sum() > 0:
                valid_logits = shift_logits[valid_mask]
                valid_labels = shift_labels[valid_mask]
                loss = F.cross_entropy(valid_logits, valid_labels)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: Invalid loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track losses
            epoch_loss += loss.item()
            batches += 1

            # Progress reporting every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batches
                elapsed_hours = (time.time() - session_start) / 3600
                current_lr = optimizer.param_groups[0]['lr']

                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Elapsed: {elapsed_hours:.2f}h")

                # Progress indicators
                if avg_loss < 11.0:
                    print("       ↓ Learning!")
                if avg_loss < 8.0:
                    print("       ↓ Great progress!")
                if avg_loss < 5.0:
                    print("       ↓ Excellent progress!")
                if avg_loss < 3.0:
                    print("       ↓ Getting close to target!")
                if avg_loss < 2.5:
                    print("       ★★ NEAR TARGET!")

            # Checkpoint every 30 minutes
            if time.time() - last_checkpoint >= 1800:
                Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
                current_avg_loss = epoch_loss / max(batches, 1)

                torch.save({
                    'step': step,
                    'model': student_model.state_dict(),
                    'loss': current_avg_loss,
                    'epoch': epoch + 1,
                    'timestamp': time.time()
                }, 'checkpoints/final/latest.pt')

                print(f"\n[CHECKPOINT] Saved at step {step} (loss: {current_avg_loss:.4f})")
                last_checkpoint = time.time()

        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} Summary | Loss: {avg_epoch_loss:.4f}")

        # Update best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"  *** NEW BEST LOSS: {best_loss:.4f} ***")

except StopIteration:
    print("\n[SESSION] Training completed successfully!")

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Training interrupted by user")

except Exception as e:
    print(f"\n[ERROR] Training failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Final summary and checkpoint
    total_time = time.time() - session_start
    final_loss = epoch_loss / max(batches, 1) if batches > 0 else best_loss

    print("\n" + "="*80)
    print("TRAINING SESSION SUMMARY")
    print("="*80)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Steps completed: {step}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Best loss: {best_loss:.4f}")

    # Save final checkpoint
    Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': step,
        'model': student_model.state_dict(),
        'loss': final_loss,
        'best_loss': best_loss,
        'total_time_hours': total_time/3600,
        'completed_at': time.time()
    }, 'checkpoints/final/final_checkpoint.pt')

    print(f"  Final checkpoint saved: checkpoints/final/final_checkpoint.pt")
    print("="*80)