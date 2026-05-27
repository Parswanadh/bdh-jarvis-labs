#!/usr/bin/env python
"""
ULTIMATE SOTA BDH DISTILLATION (RESUME ENABLED)
- Architecture: 8 Layers | 256 Embedding | 256 Seq Len
- Teacher: Qwen 3.5 0.8B
- Features: Automatic Resumption, Mixed Precision, Vocab Alignment
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

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7, beta=0.3):
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    mask = shift_labels != -100
    if mask.sum() == 0: return torch.tensor(0.0, device=student_logits.device)
    
    s_logits = shift_student.view(-1, shift_student.size(-1))[mask.view(-1)]
    t_logits = shift_teacher.view(-1, shift_teacher.size(-1))[mask.view(-1)]
    target = shift_labels.view(-1)[mask.view(-1)]

    kl_div = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=-1),
        F.softmax(t_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    ce_loss = F.cross_entropy(s_logits, target)
    return alpha * kl_div + beta * ce_loss

print("="*80)
print("SOTA BDH DISTILLATION SYSTEM (RESUME ENABLED)")
print("="*80)

# 1. SETUP
device = torch.device('cuda')
scaler = torch.amp.GradScaler('cuda')
torch.cuda.empty_cache()

# 2. MODELS
model_name = "Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

actual_vocab_size = teacher_model.config.vocab_size 

config = MultiScaleBDHConfig(
    vocab_size=actual_vocab_size, 
    n_embd=256,
    n_layer=8,
    n_head=8,
    ffn_dim=1024,
    max_seq_len=256,
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.0008
)

student_model = MultiScaleBDH(config).to(device)

# --- RESUME LOGIC ---
checkpoint_dir = Path("checkpoints/sota")
checkpoint_path = checkpoint_dir / "sota_scaled_latest.pt"
start_step = 0

if checkpoint_path.exists():
    print(f"Found existing SOTA checkpoint. Loading...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        student_model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0)
    else:
        student_model.load_state_dict(ckpt)
    print(f"Resuming from Step: {start_step}")
else:
    print("No existing SOTA checkpoint found. Starting fresh.")

print(f"Student Model: {sum(p.numel() for p in student_model.parameters())/1e6:.1f}M Params")

# 3. DATA
data_path = Path("data/tinystories.txt")
with open(data_path, 'r', encoding='utf-8') as f:
    stories = [line.strip() for line in f if line.strip()][:50000]

def collate_fn(batch):
    enc = tokenizer(batch, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    ids = enc['input_ids']
    ids[ids >= actual_vocab_size] = tokenizer.eos_token_id 
    return ids, enc['attention_mask']

dataloader = DataLoader(stories, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 4. TRAINING
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
grad_accum_steps = 16 
session_start = time.time()

student_model.train()
try:
    for epoch in range(10):
        for i, (input_ids, mask) in enumerate(dataloader):
            # Skip batches already processed if resuming
            current_iter_step = (epoch * len(dataloader) + i) // grad_accum_steps
            if current_iter_step < start_step:
                continue

            input_ids, mask = input_ids.to(device), mask.to(device)
            labels = input_ids.clone()
            labels[mask == 0] = -100

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids, attention_mask=mask).logits
                student_logits, _ = student_model(input_ids)
                loss = distillation_loss(student_logits, teacher_logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                step = current_iter_step
                if step % 2 == 0:
                    elapsed = (time.time()-session_start)/60
                    print(f"Step {step:5d} | SOTA Loss: {loss.item()*grad_accum_steps:.4f} | Time: {elapsed:.1f} min")

                # Auto-save every 100 steps
                if step > 0 and step % 100 == 0:
                    torch.save({'model': student_model.state_dict(), 'step': step}, checkpoint_path)

except KeyboardInterrupt:
    print("\nInterrupted. Saving...")
finally:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': student_model.state_dict(),
        'config': config,
        'step': current_iter_step if 'current_iter_step' in locals() else start_step
    }, checkpoint_path)
    print(f"Progress saved to {checkpoint_path}")
