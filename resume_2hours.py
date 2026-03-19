#!/usr/bin/env python
"""
Quick script to resume training from latest checkpoint for 2 more hours
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time
import os

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

print("="*80)
print("RESUMING BDH TRAINING FOR 2 MORE HOURS")
print("="*80)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n[DEVICE] Using CPU")

print(f"[PyTorch] {torch.__version__}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n[1/3] Loading latest checkpoint...")

# Load latest checkpoint
checkpoint_path = Path("checkpoints/final/latest.pt")
if not checkpoint_path.exists():
    print(f"ERROR: No checkpoint found at {checkpoint_path}")
    print("Please run the initial training first.")
    sys.exit(1)

print(f"  Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract checkpoint info
if 'model' in checkpoint:
    model_state = checkpoint['model']
    start_step = checkpoint.get('step', 0)
    start_loss = checkpoint.get('loss', 0.0)
    print(f"  Resuming from step {start_step:,} with loss {start_loss:.4f}")
else:
    # Fallback
    model_state = checkpoint
    start_step = 0
    start_loss = 12.4
    print(f"  Warning: Unexpected checkpoint format")

# Load tokenizer
print("\n[2/3] Loading tokenizer...")
teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_vocab = len(teacher_tokenizer)
print(f"  Vocabulary size: {teacher_vocab:,}")

# Create student model (same config as original)
print("\n[3/3] Creating student model...")
bdh_config = MultiScaleBDHConfig(
    vocab_size=teacher_vocab,
    n_embd=192,
    n_layer=3,
    n_head=4,
    ffn_dim=768,
    dropout=0.0,
    max_seq_len=80,
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
        print(f"  Continuing with random initialization")
else:
    print(f"  No model state in checkpoint")

student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student model: {student_params:,} parameters")

# Load teacher model
print("\n[Teacher] Loading Qwen3.5-0.8B...")
try:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen3.5-0.8B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).eval()
    print(f"  Teacher model loaded")
except Exception as e:
    print(f"  ERROR loading teacher: {e}")
    # Fallback
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen3.5-0.8B",
        trust_remote_code=True
    ).eval()
    if torch.cuda.is_available():
        teacher_model = teacher_model.half().to(device)
    else:
        teacher_model = teacher_model.to(device)

teacher_model.eval()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load data (same as original)
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
    batch_size=6,
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Loaded {len(stories):,} stories")
print(f"  {len(dataloader):,} batches per epoch")

# Setup training
print("\n" + "="*80)
print("CONFIGURING TRAINING")
print("="*80)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

# Resume from checkpoint time
session_start = time.time() - (start_step * 0.1)  # Approximate start time adjustment
last_checkpoint = session_start

print(f"  Resuming from step: {start_step:,}")
print(f"  Previous loss: {start_loss:.4f}")
print(f"  Target: Train for 2 more hours")
print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

teacher_model.eval()
student_model.train()

print("\n" + "="*80)
print("STARTING TRAINING (2 MORE HOURS)")
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

            step = start_step + batch_idx + 1 + (epoch * len(dataloader)) + (batch_idx if epoch > 0 else 0)
            # Actually, let's track it properly
            # Better approach: increment step from start_step

            # Let's simplify: track total steps from start_step
            if batch_idx == 0 and epoch == 0:
                step = start_step
            else:
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

                mem_info = ""
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
                    mem_info = f" | VRAM: {mem_alloc:.1f}GB"

                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Elapsed: {elapsed_hours:.2f}h{mem_info}")

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
                    'loss': current_avg_loss
                }, 'checkpoints/final/latest.pt')

                print(f"\n[CHECKPOINT] Saved at step {step:,} (loss: {current_avg_loss:.4f})")
                last_checkpoint = time.time()

        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} Summary | Loss: {avg_epoch_loss:.4f}")

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
    final_loss = epoch_loss / max(batches, 1) if batches > 0 else start_loss
    total_steps = start_step + (batches if batches > 0 else 0)

    print("\n" + "="*80)
    print("TRAINING SESSION SUMMARY")
    print("="*80)
    print(f"  Total training time: {total_time/3600:.2f} hours")
    print(f"  Steps completed in this session: {batches:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Starting loss: {start_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {start_loss - final_loss:.4f}")

    # Save final checkpoint
    Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': total_steps,
        'model': student_model.state_dict(),
        'loss': final_loss
    }, 'checkpoints/final/latest.pt')

    print(f"  Final checkpoint saved: checkpoints/final/latest.pt")
    print("="*80)