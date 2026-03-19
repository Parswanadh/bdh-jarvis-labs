#!/usr/bin/env python
"""
TRUE RESUME BDH DISTILLATION - MEMORY OPTIMIZED
Optimized for 8GB VRAM GPUs (Batch size reduction + Gradient accumulation).
"""

import torch
import torch.nn as nn
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

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div * (temperature ** 2)

def project_logits(student_logits, target_vocab_size):
    batch, seq, student_vocab = student_logits.shape
    if target_vocab_size > student_vocab:
        repeat_factor = (target_vocab_size // student_vocab) + 1
        projected = student_logits.repeat(1, 1, repeat_factor)[..., :target_vocab_size]
    else:
        projected = student_logits[..., :target_vocab_size]
    return projected

print("="*80)
print("BDH TRUE DISTILLATION RESUMPTION (VRAM OPTIMIZED)")
print("="*80)

# 1. SETUP DEVICE & MEMORY
if not torch.cuda.is_available():
    print("\n[ERROR] CUDA required for this script.")
    sys.exit(1)

device = torch.device('cuda')
torch.cuda.empty_cache()
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 2. LOCATE CHECKPOINT
checkpoint_path = Path("checkpoints/final/latest.pt")
if not checkpoint_path.exists():
    checkpoint_path = Path("checkpoints/working_distillation/model.pt")

print(f"\n[1/4] Loading checkpoint: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        saved_step = checkpoint.get('step', 0)
        saved_loss = checkpoint.get('loss', 0.0)
    else:
        model_state = checkpoint
        saved_step = 0
        saved_loss = 0.0
    print(f"  Successfully loaded (Step: {saved_step})")
except Exception as e:
    print(f"  [ERROR] Failed to load checkpoint: {e}")
    sys.exit(1)

# 3. LOAD MODELS
print("\n[2/4] Loading models...")
try:
    model_name = "Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    teacher_vocab_size = teacher_model.config.vocab_size
    print(f"  Teacher loaded ({teacher_vocab_size} vocab)")
except Exception as e:
    print(f"  [ERROR] Failed to load teacher: {e}")
    sys.exit(1)

config = MultiScaleBDHConfig(
    vocab_size=len(tokenizer),
    n_embd=192,
    n_layer=3,
    n_head=4,
    ffn_dim=768,
    dropout=0.0,
    max_seq_len=80,
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(config).to(device)
student_model.load_state_dict(model_state, strict=False)

# 4. LOAD DATA (REDUCED BATCH SIZE)
print("\n[3/4] Preparing dataset...")
data_path = Path("data/tinystories.txt")
with open(data_path, 'r', encoding='utf-8') as f:
    stories = [line.strip() for line in f if line.strip()][:50000]

def collate_fn(batch):
    return tokenizer(batch, padding=True, truncation=True, max_length=80, return_tensors='pt')

# Lower batch size to 4 to save memory
BATCH_SIZE = 4 
dataloader = DataLoader(stories, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print(f"  Using Batch Size: {BATCH_SIZE} (Optimized for VRAM)")

# 5. TRAINING LOOP
print("\n[4/4] Starting TRUE DISTILLATION...")
optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

step = saved_step
session_start = time.time()
last_save = session_start
target_duration = 7200 
grad_accum_steps = 2 # Update weights every 2 batches (Effective batch size = 8)

student_model.train()
torch.cuda.empty_cache()

try:
    for epoch in range(20):
        for i, batch in enumerate(dataloader):
            elapsed = time.time() - session_start
            if elapsed > target_duration:
                print("\n[SESSION] Time limit reached.")
                raise StopIteration

            input_ids = batch['input_ids'].to(device)
            
            # 1. Teacher forward (No gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits

            # 2. Student forward
            student_logits, _ = student_model(input_ids)
            
            # 3. Project and calculate Distillation Loss
            student_logits_proj = project_logits(student_logits, teacher_vocab_size)
            kl_loss = distillation_loss(student_logits_proj, teacher_logits)
            
            # 4. Cross Entropy Loss
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 5. Combined Loss (Normalized by accumulation steps)
            loss = (0.7 * kl_loss + 0.3 * ce_loss) / grad_accum_steps
            loss.backward()

            # 6. Step optimizer every 'grad_accum_steps'
            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            if step % 10 == 0 and (i + 1) % grad_accum_steps == 0:
                print(f"  Step {step:6d} | Loss: {loss.item()*grad_accum_steps:.4f} | KL: {kl_loss.item():.4f} | CE: {ce_loss.item():.4f} | {elapsed/60:.1f} min")

            # Save periodically
            if time.time() - last_save > 1800:
                print(f"\n[SAVING] Checkpoint...")
                torch.save({'step': step, 'model': student_model.state_dict(), 'loss': loss.item()}, checkpoint_path)
                last_save = time.time()

except (KeyboardInterrupt, StopIteration):
    print("\n[INFO] Stopped.")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n[CRITICAL] CUDA Out of Memory. Final save attempted...")
    else:
        raise e
finally:
    # Final cleanup and save
    Path("checkpoints/final").mkdir(parents=True, exist_ok=True)
    try:
        torch.save({
            'step': step, 
            'model': student_model.state_dict(), 
            'loss': loss.item() if 'loss' in locals() else saved_loss
        }, checkpoint_path)
        print(f"Final checkpoint saved to {checkpoint_path}")
    except:
        print("Failed to save final checkpoint due to memory error.")
    print("="*80)
