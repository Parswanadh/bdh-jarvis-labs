#!/usr/bin/env python
"""
BDH HATCHLING - SOTA FINISHER
Goal: Reduce Perplexity from 60 down to <10.
Technique: Low-Temperature Distillation (T=1.5) + High Weight Decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from pathlib import Path
import sys
import time
import gc
import os

# System priority
if sys.platform == 'win32':
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except: pass

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- CONFIG ---
MODEL_NAME = "Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("checkpoints/safe/laptop_safe.pt")
DATA_PATH = Path("data/tinystories.txt")
SEQ_LEN = 192 
TOTAL_TARGET_STEPS = 25000 # Extended for SOTA polish

def distillation_loss(student_logits, teacher_logits, labels, temperature=1.5): # T=1.5 for precision
    shift_s = student_logits[..., :-1, :].contiguous()
    shift_t = teacher_logits[..., :-1, :].contiguous()
    shift_l = labels[..., 1:].contiguous()
    mask = shift_l != -100
    s_flat = shift_s.view(-1, shift_s.size(-1))[mask.view(-1)]
    t_flat = shift_t.view(-1, shift_t.size(-1))[mask.view(-1)]
    l_flat = shift_l.view(-1)[mask.view(-1)]
    
    s_log_probs = F.log_softmax(s_flat.float() / temperature, dim=-1)
    t_probs = F.softmax(t_flat.float() / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature**2)
    ce = F.cross_entropy(s_flat, l_flat)
    return 0.5 * kl + 0.5 * ce # 50/50 split for final word-accuracy

def safe_save(state, path):
    temp = str(path) + ".tmp"
    torch.save(state, temp)
    if os.path.exists(path):
        bak = str(path) + ".bak"
        if os.path.exists(bak): os.remove(bak)
        os.rename(path, bak)
    os.rename(temp, path)
    print(f"\n[SOTA SAVE] Step {state['step']} secured.")

def run_finish():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    print("Loading models for the SOTA Finisher...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    
    config = MultiScaleBDHConfig(
        vocab_size=teacher.config.vocab_size,
        n_embd=256, n_layer=8, n_head=8, ffn_dim=1024,
        max_seq_len=SEQ_LEN, decay_rates=[0.95, 0.99, 0.995], hebbian_lr=0.0005
    )
    
    student = MultiScaleBDH(config).to(device)
    
    start_step = 0
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        student.load_state_dict(ckpt['model'])
        start_step = ckpt['step']
        print(f"Resumed at Step {start_step}. Entering Final Polishing Phase.")

    dataset = DataLoader(IterableDataset.from_generator(lambda: (line.strip() for line in open(DATA_PATH, encoding='utf-8') if line.strip())), batch_size=1)

    # LOWER LR + HIGHER DECAY: This is the secret to low perplexity
    optimizer = torch.optim.AdamW(student.parameters(), lr=4.0e-5, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(TOTAL_TARGET_STEPS - start_step))
    scaler = torch.amp.GradScaler('cuda')

    grad_accum = 16 
    student.train()
    session_start = time.time()

    print(f"SOTA Finisher Active. Targeting Step {TOTAL_TARGET_STEPS}...")
    
    try:
        for i, batch in enumerate(dataset):
            current_step = (i // grad_accum) + start_step
            if current_step >= TOTAL_TARGET_STEPS: break
            if current_step < start_step: continue

            ids = tokenizer(batch, padding='max_length', truncation=True, max_length=SEQ_LEN, return_tensors='pt')['input_ids'].to(device)
            mask = (ids != tokenizer.pad_token_id).to(device)
            labels = ids.clone()
            labels[mask == 0] = -100

            with torch.amp.autocast('cuda'):
                with torch.no_grad(): t_logits = teacher(ids).logits
                s_logits, _ = student(ids)
                loss = distillation_loss(s_logits, t_logits, labels) / grad_accum

            scaler.scale(loss).backward()
            time.sleep(0.1)

            if (i + 1) % grad_accum == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
                if current_step % 5 == 0:
                    print(f"Step {current_step:5d} | Loss: {loss.item()*grad_accum:.4f} | Polishing...")
                if current_step % 100 == 0:
                    safe_save({'model': student.state_dict(), 'step': current_step}, CHECKPOINT_PATH)
                torch.cuda.empty_cache(); gc.collect()

    finally:
        safe_save({'model': student.state_dict(), 'step': current_step}, CHECKPOINT_PATH)

if __name__ == "__main__":
    run_finish()
