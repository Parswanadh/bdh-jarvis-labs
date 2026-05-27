#!/usr/bin/env python
"""
Continue BDH training from checkpoint - handles CPU/GPU automatically
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

try:
    from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
except ImportError as e:
    print(f"Error importing multiscale_bdh: {e}")
    print("Make sure you're running from the BDH project directory")
    sys.exit(1)

print("="*80)
print("CONTINUING BDH TRAINING FROM CHECKPOINT")
print("="*80)

# Smart device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device('cpu')
    print("\n[DEVICE] Using CPU (CUDA not available)")
    # Warn about performance
    print("[WARNING] Training will be significantly slower on CPU")

print(f"[PyTorch] {torch.__version__}")

# Clear cache if CUDA available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n[1/3] Loading models and checkpoint...")

# Load tokenizer
try:
    teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Trying to load without trust_remote_code...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-0.8B")
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Load checkpoint
checkpoint_path = Path("checkpoints/final/latest.pt")
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    print("Available checkpoints:")
    checkpoint_dir = Path("checkpoints/final")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.pt"):
            print(f"  {f}")
    sys.exit(1)

print(f"  Loading checkpoint: {checkpoint_path}")

try:
    # Try different loading strategies
    checkpoint = None
    loading_errors = []

    # Strategy 1: weights_only=False (for newer PyTorch)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e1:
        loading_errors.append(f"weights_only=False: {e1}")

        # Strategy 2: weights_only=True (for security)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception as e2:
            loading_errors.append(f"weights_only=True: {e2}")

            # Strategy 3: No map_location
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
            except Exception as e3:
                loading_errors.append(f"No map_location: {e3}")

                # If all fail, raise the last error
                raise e3

    if checkpoint is None:
        raise RuntimeError("Failed to load checkpoint with all strategies")

    print(f"  Checkpoint loaded successfully")

    # Extract information
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            saved_step = checkpoint.get('step', 0)
            saved_loss = checkpoint.get('loss', 0.0)
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            saved_step = checkpoint.get('step', 0)
            saved_loss = checkpoint.get('loss', 0.0)
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            saved_step = checkpoint.get('step', 0)
            saved_loss = checkpoint.get('loss', 0.0)
        else:
            # Assume the checkpoint IS the state dict
            model_state = checkpoint
            saved_step = 0
            saved_loss = 0.0

        print(f"  Checkpoint type: dict with keys {list(checkpoint.keys())}")
        print(f"  Resuming from step {saved_step} with loss {saved_loss:.4f}")
    else:
        # Checkpoint is just the state dict
        model_state = checkpoint
        saved_step = 0
        saved_loss = 0.0
        print(f"  Checkpoint is raw state dict")

except Exception as e:
    print(f"  ERROR loading checkpoint: {e}")
    print("  Loading errors encountered:")
    for err in loading_errors:
        print(f"    - {err}")
    print("\n  Falling back to default configuration")
    model_state = None
    saved_step = 0
    saved_loss = 12.4  # Initial loss

# Determine vocabulary size
try:
    teacher_vocab = len(teacher_tokenizer)
    print(f"  Vocabulary size: {teacher_vocab:,}")
except Exception as e:
    print(f"  Error getting vocab size: {e}")
    teacher_vocab = 32000  # Fallback
    print(f"  Using fallback vocab size: {teacher_vocab}")

# Create student model configuration
print("\n[2/3] Creating student model...")
try:
    bdh_config = MultiScaleBDHConfig(
        vocab_size=teacher_vocab,
        n_embd=192,      # Matches original training
        n_layer=3,       # Matches original training
        n_head=4,
        ffn_dim=768,
        dropout=0.0,
        max_seq_len=80,  # Matches original training
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    student_model = MultiScaleBDH(bdh_config).to(device)

    # Load model state if available
    if model_state is not None:
        try:
            # Handle different possible state dict formats
            if isinstance(model_state, dict):
                # Try to load with strict=False to allow for minor mismatches
                missing_keys, unexpected_keys = student_model.load_state_dict(model_state, strict=False)
                print(f"  Model state loaded")
                if missing_keys:
                    print(f"  Info: {len(missing_keys)} missing keys (will be initialized)")
                if unexpected_keys:
                    print(f"  Info: {len(unexpected_keys)} unexpected keys (will be ignored)")
            else:
                print(f"  WARNING: model_state is not a dict, skipping load")
        except Exception as e:
            print(f"  WARNING: Could not load model state: {e}")
            print(f"  Continuing with randomly initialized model")
    else:
        print(f"  No model state in checkpoint - starting with random initialization")

except Exception as e:
    print(f"  ERROR creating model: {e}")
    sys.exit(1)

student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Student model: {student_params:,} parameters")

# Load teacher model
print("\n[3/3] Loading teacher model...")
try:
    # Determine appropriate dtype
    if torch.cuda.is_available():
        teacher_dtype = torch.float16  # Save memory on GPU
    else:
        teacher_dtype = torch.float32  # More stable on CPU

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen3.5-0.8B",
        torch_dtype=teacher_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Help with memory efficiency
    )

    # Move to device if needed
    if not teacher_model.device.type == str(device):
        teacher_model = teacher_model.to(device)

    teacher_model.eval()
    print(f"  Teacher model loaded ({teacher_dtype}) on {device}")

except Exception as e:
    print(f"  ERROR loading teacher model: {e}")
    print("  Trying fallback without dtype specification...")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "Qwen3.5-0.8B",
            trust_remote_code=True
        )
        if torch.cuda.is_available():
            teacher_model = teacher_model.half().to(device)
        else:
            teacher_model = teacher_model.to(device)
        teacher_model.eval()
        print(f"  Teacher model loaded via fallback on {device}")
    except Exception as e2:
        print(f"  FATAL: Could not load teacher model: {e2}")
        sys.exit(1)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[4/4] Loading dataset...")

data_file = Path("data/tinystories.txt")
if not data_file.exists():
    print(f"ERROR: Data file not found at {data_file}")
    sys.exit(1)

try:
    with open(data_file, 'r', encoding='utf-8') as f:
        all_stories = [line.strip() for line in f if line.strip()]

    stories = all_stories[:50000]  # Same as original training
    print(f"  Loaded {len(all_stories):,} total stories")
    print(f"  Using {len(stories):,} stories for training")

except Exception as e:
    print(f"  ERROR loading data: {e}")
    sys.exit(1)

def collate_batch(batch_texts):
    try:
        encodings = teacher_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=80,
            return_tensors='pt'
        )
        return encodings
    except Exception as e:
        print(f"  Warning: Error in collate_batch: {e}")
        # Return empty batch to avoid crashing
        return {
            'input_ids': torch.zeros((1, 1), dtype=torch.long),
            'attention_mask': torch.zeros((1, 1), dtype=torch.long)
        }

try:
    dataloader = DataLoader(
        stories,
        batch_size=6,  # Same as original
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=True,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    print(f"  Created dataloader with {len(dataloader):,} batches per epoch")
except Exception as e:
    print(f"  ERROR creating dataloader: {e}")
    sys.exit(1)

# ============================================================================
# SETUP TRAINING
# ============================================================================
print("\n" + "="*80)
print("CONFIGURING TRAINING")
print("="*80)

try:
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
except Exception as e:
    print(f"  ERROR setting up optimizer/scheduler: {e}")
    sys.exit(1)

# Training state
session_start = time.time()
step = saved_step  # Resume from saved step
last_checkpoint = session_start
best_loss = saved_loss if saved_step > 0 else float('inf')

print(f"  Resuming from step: {step:,}")
if saved_step > 0:
    print(f"  Previous best loss: {best_loss:.4f}")
else:
    print(f"  Starting fresh training")
print(f"  Target: Train for 2 more hours (until completion)")

teacher_model.eval()
student_model.train()

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

# Training loop variables
epoch_loss = 0.0
batches = 0
epoch_num = 0

try:
    # Calculate target end time
    target_end_time = session_start + 7200  # 2 hours from now

    for epoch in range(1000):  # Large number, will break via time check
        # Check if time's up
        if time.time() >= target_end_time:
            print("\n[SESSION] 2 hours complete!")
            break

        epoch_num = epoch + 1
        epoch_loss = 0.0
        batches = 0

        print(f"\n--- EPOCH {epoch_num} ---")

        for batch_idx, batch in enumerate(dataloader):
            # Double-check time
            if time.time() >= target_end_time:
                print("\n[SESSION] 2 hours complete!")
                raise StopIteration

            step += 1

            # Move batch to device
            try:
                input_ids = batch['input_ids'].to(device)
            except Exception as e:
                print(f"  Warning: Error moving batch to device: {e}")
                continue

            # Teacher forward pass (no gradients)
            try:
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids)
                    teacher_logits = teacher_outputs.logits
            except Exception as e:
                print(f"  Warning: Teacher forward failed: {e}")
                optimizer.zero_grad()
                continue

            # Student forward pass
            try:
                student_logits, _ = student_model(input_ids)
            except Exception as e:
                print(f"  Warning: Student forward failed: {e}")
                optimizer.zero_grad()
                continue

            # Loss computation - next token prediction
            try:
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

            except Exception as e:
                print(f"  Warning: Loss computation failed: {e}")
                optimizer.zero_grad()
                continue

            # Backward pass
            try:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            except Exception as e:
                print(f"  Warning: Backward pass failed: {e}")
                continue

            # Track losses
            epoch_loss += loss.item()
            batches += 1

            # Progress reporting every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batches
                elapsed_hours = (time.time() - session_start) / 3600
                remaining_hours = max(0, (target_end_time - time.time()) / 3600)
                current_lr = optimizer.param_groups[0]['lr']

                # Memory info (if available)
                mem_info = ""
                if torch.cuda.is_available():
                    try:
                        mem_alloc = torch.cuda.memory_allocated(0) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
                        mem_info = f" | VRAM: {mem_alloc:.1f}GB/{mem_reserved:.1f}GB"
                    except:
                        mem_info = " | VRAM: N/A"

                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Elapsed: {elapsed_hours:.2f}h | "
                      f"Remaining: {remaining_hours:.2f}h{mem_info}")

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
                try:
                    Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
                    current_avg_loss = epoch_loss / max(batches, 1)

                    checkpoint_data = {
                        'step': step,
                        'model': student_model.state_dict(),
                        'loss': current_avg_loss,
                        'epoch': epoch_num,
                        'timestamp': time.time(),
                        'best_loss': min(best_loss, current_avg_loss)
                    }

                    torch.save(checkpoint_data, 'checkpoints/final/latest.pt')

                    print(f"\n[CHECKPOINT] Saved at step {step:,} (loss: {current_avg_loss:.4f})")
                    last_checkpoint = time.time()

                except Exception as e:
                    print(f"  Warning: Checkpoint failed: {e}")

        # End of epoch summary
        if batches > 0:
            avg_epoch_loss = epoch_loss / batches
            print(f"  Epoch {epoch_num} Summary | Loss: {avg_epoch_loss:.4f}")

            # Update best loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                print(f"  *** NEW BEST LOSS: {best_loss:.4f} ***")
        else:
            print(f"  Epoch {epoch_num} Summary | No batches processed")

except StopIteration:
    print("\n[SESSION] Training completed successfully!")

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Training interrupted by user")

except Exception as e:
    print(f"\n[ERROR] Training failed with error: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()

finally:
    # Final summary and checkpoint
    total_time = time.time() - session_start
    if batches > 0:
        final_loss = epoch_loss / batches
    else:
        final_loss = best_loss if best_step > 0 else 0.0

    best_loss_actual = min(best_loss, final_loss) if 'best_loss' in locals() else final_loss

    print("\n" + "="*80)
    print("TRAINING SESSION SUMMARY")
    print("="*80)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Steps completed: {step:,}")
    if batches > 0:
        print(f"  Final loss: {final_loss:.4f}")
    print(f"  Best loss: {best_loss_actual:.4f}")
    print(f"  Epochs completed: {epoch_num}")

    # Save final checkpoint
    try:
        Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
        final_checkpoint = {
            'step': step,
            'model': student_model.state_dict(),
            'loss': final_loss if batches > 0 else 0.0,
            'best_loss': best_loss_actual,
            'total_time_hours': total_time/3600,
            'completed_at': time.time(),
            'finished_by_timeout': time.time() >= target_end_time if 'target_end_time' in locals() else False
        }

        torch.save(final_checkpoint, 'checkpoints/final/final_checkpoint.pt')
        torch.save(final_checkpoint, 'checkpoints/final/latest.pt')  # Also update latest

        print(f"  Final checkpoint saved: checkpoints/final/final_checkpoint.pt")
        print(f"  Latest checkpoint updated: checkpoints/final/latest.pt")
    except Exception as e:
        print(f"  Warning: Failed to save final checkpoint: {e}")

    print("="*80)