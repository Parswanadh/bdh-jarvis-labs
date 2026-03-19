#!/usr/bin/env python
"""
BDH HATCHLING - TURBO SESSION (2 HOURS)
- Teacher: Qwen3.5-0.8B (Standard)
- Speed: 0.1s pause, 8 Grad Accumulation
- Target: Run for 120 minutes exactly.
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

# Priority
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
TARGET_MINUTES = 120 # 2-hour target

class LazyTinyStories(IterableDataset):
    def __init__(self, file_path, limit=200000):
        self.file_path = file_path
        self.limit = limit
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line:
                    yield line
                    count += 1
                    if count >= self.limit: break

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
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
    return 0.7 * kl + 0.3 * ce

def safe_save(state, path):
    temp = str(path) + ".tmp"
    torch.save(state, temp)
    if os.path.exists(path):
        bak = str(path) + ".bak"
        if os.path.exists(bak): os.remove(bak)
        os.rename(path, bak)
    os.rename(temp, path)
    print(f"\n[TURBO SAVE] Step {state['step']} secured.")

def run_turbo():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    print("Loading models (Qwen3.5-0.8B Teacher)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    
    config = MultiScaleBDHConfig(
        vocab_size=teacher.config.vocab_size,
        n_embd=256, n_layer=8, n_head=8, ffn_dim=1024,
        max_seq_len=SEQ_LEN, decay_rates=[0.95, 0.99, 0.995], hebbian_lr=0.0005
    )
    
    student = MultiScaleBDH(config).to(device)
    
    # RESUME
    start_step = 0
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        student.load_state_dict(ckpt['model'])
        start_step = ckpt['step']
        print(f"Resuming from Step {start_step}")

    dataset = LazyTinyStories(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda b: tokenizer(b, padding='max_length', truncation=True, max_length=SEQ_LEN, return_tensors='pt'))

    optimizer = torch.optim.AdamW(student.parameters(), lr=1.0e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=25000)
    for _ in range(start_step): scheduler.step()
    scaler = torch.amp.GradScaler('cuda')

    grad_accum = 8 # Agile training
    student.train()
    session_start = time.time()

    print(f"\nTurbo Mode Active! Target: {TARGET_MINUTES} minutes.")
    
    try:
        for i, batch in enumerate(dataloader):
            # Timer Check
            elapsed_min = (time.time() - session_start) / 60
            if elapsed_min >= TARGET_MINUTES:
                print(f"\n[SESSION COMPLETE] 2 hours reached!")
                break

            current_step = (i // grad_accum) + start_step
            if current_step < start_step: continue

            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = ids.clone()
            labels[mask == 0] = -100

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    t_logits = teacher(ids, attention_mask=mask).logits
                s_logits, _ = student(ids)
                loss = distillation_loss(s_logits, t_logits, labels) / grad_accum

            scaler.scale(loss).backward()
            
            # Fast Cooling
            time.sleep(0.1) 

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                if current_step % 5 == 0:
                    print(f"Step {current_step:5d} | Loss: {loss.item()*grad_accum:.4f} | {elapsed_min:.1f}/{TARGET_MINUTES} min")

                if current_step % 50 == 0:
                    safe_save({'model': student.state_dict(), 'step': current_step}, CHECKPOINT_PATH)
                
                torch.cuda.empty_cache()
                gc.collect()

            del t_logits, s_logits, loss

    except KeyboardInterrupt:
        print("\nSaving and stopping...")
    finally:
        safe_save({'model': student.state_dict(), 'step': current_step}, CHECKPOINT_PATH)

if __name__ == "__main__":
    run_turbo()
