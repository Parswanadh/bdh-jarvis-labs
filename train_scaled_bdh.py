#!/usr/bin/env python
"""
SCALED BDH TRAINING (10M+ Params)
Increased layers, embedding, and sequence length for better coherence.
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
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def project_logits(student_logits, target_vocab_size):
    batch, seq, student_vocab = student_logits.shape
    repeat_factor = (target_vocab_size // student_vocab) + 1
    return student_logits.repeat(1, 1, repeat_factor)[..., :target_vocab_size]

print("="*80)
print("SCALED BDH TRAINING SYSTEM")
print("Architecture: 8 Layers | 256 Embedding | 256 Seq Len")
print("="*80)

# 1. SETUP
device = torch.device('cuda')
torch.cuda.empty_cache()

# 2. MODELS
model_name = "Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

# NEW SCALED CONFIGURATION
config = MultiScaleBDHConfig(
    vocab_size=len(tokenizer),
    n_embd=256,      # Increased from 192
    n_layer=8,       # Increased from 3
    n_head=8,        # Increased from 4
    ffn_dim=1024,    # Increased from 768
    max_seq_len=256, # Increased from 80
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.0008 # Slightly lower LR for larger model stability
)

student_model = MultiScaleBDH(config).to(device)
print(f"Total Student Params: {sum(p.numel() for p in student_model.parameters())/1e6:.1f}M")

# 3. DATA (INCREASED SEQ LEN)
data_path = Path("data/tinystories.txt")
with open(data_path, 'r', encoding='utf-8') as f:
    stories = [line.strip() for line in f if line.strip()][:50000]

def collate_fn(batch):
    return tokenizer(batch, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

# Lower batch size because 256 seq len takes MUCH more VRAM during distillation
BATCH_SIZE = 2 
dataloader = DataLoader(stories, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 4. TRAINING LOOP
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1.5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

step = 0
session_start = time.time()
grad_accum_steps = 4 # 2 * 4 = Effective Batch Size 8

print(f"\nStarting from scratch with NEW architecture...")
student_model.train()

try:
    for epoch in range(10):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits

            student_logits, _ = student_model(input_ids)
            student_logits_proj = project_logits(student_logits, teacher_model.config.vocab_size)
            
            kl_loss = distillation_loss(student_logits_proj, teacher_logits)
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss = (0.7 * kl_loss + 0.3 * ce_loss) / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0:
                    elapsed = (time.time() - session_start) / 60
                    print(f"  Step {step:5d} | Loss: {loss.item()*grad_accum_steps:.4f} | KL: {kl_loss.item():.4f} | CE: {ce_loss.item():.4f} | {elapsed:.1f} min")

            # Save every 500 steps
            if step > 0 and step % 500 == 0 and (i + 1) % grad_accum_steps == 0:
                print(f"\n[SAVING] Checkpoint step {step}...")
                Path("checkpoints/scaled").mkdir(parents=True, exist_ok=True)
                torch.save({'step': step, 'model': student_model.state_dict(), 'config': config}, f"checkpoints/scaled/model_step_{step}.pt")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    Path("checkpoints/scaled").mkdir(parents=True, exist_ok=True)
    torch.save({'step': step, 'model': student_model.state_dict(), 'config': config}, "checkpoints/scaled/latest_scaled.pt")
    print("Final checkpoint saved to checkpoints/scaled/latest_scaled.pt")
