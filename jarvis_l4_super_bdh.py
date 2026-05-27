#!/usr/bin/env python
"""
JARVIS LABS L4 SUPER SCRIPT (24GB VRAM OPTIMIZED)
Hardware: 1x L4 GPU (24GB VRAM)
Time Limit: 5 Hours
Goal: Maximize Logit Generation & Distillation Training Parallelism.

HOW WE USE 24GB VRAM WITHOUT INCREASING SEQ_LEN:
Instead of loading multiple identical teacher models (which wastes VRAM on duplicate weights),
we load ONE 4-bit Quantized Teacher Model, and we exponentially increase the BATCH SIZE.
A batch size of 64 or 128 will process 64-128 stories SIMULTANEOUSLY in parallel,
saturating the 24GB VRAM perfectly while keeping the sequence length super fast (192).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from pathlib import Path
import sys
import time
import gc
import os
import random

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = Path("checkpoints/jarvis/l4_super_model.pt")
DATA_PATH = Path("data/tinystories.txt")
SEQ_LEN = 192 
TARGET_HOURS = 4.8 # Leave 12 minutes for safe shutdown and save

# Massive batching to utilize 24GB VRAM. 
# 4-bit teacher uses ~1GB. Student uses ~0.5GB. 
# Leaving ~22GB for activations. We can handle a massive batch size!
BATCH_SIZE = 64  
GRAD_ACCUM = 2    # Effective batch = 128 stories per step!

class LazyTinyStories(IterableDataset):
    def __iter__(self):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield line.strip()

def sota_distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    shift_s = student_logits[..., :-1, :].contiguous()
    shift_t = teacher_logits[..., :-1, :].contiguous()
    shift_l = labels[..., 1:].contiguous()
    mask = shift_l != -100
    
    s_flat = shift_s.view(-1, shift_s.size(-1))[mask.view(-1)]
    t_flat = shift_t.view(-1, shift_t.size(-1))[mask.view(-1)]
    l_flat = shift_l.view(-1)[mask.view(-1)]

    # Sub-Atomic Top-K
    tk_val, tk_idx = torch.topk(t_flat, 512, dim=-1)
    s_top = torch.gather(s_flat, -1, tk_idx)
    
    s_log_probs = F.log_softmax(s_top.float() / temperature, dim=-1)
    t_probs = F.softmax(tk_val.float() / temperature, dim=-1)
    
    kl_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(s_flat, l_flat)
    
    return (0.7 * kl_loss) + (0.3 * ce_loss), kl_loss, ce_loss

def atomic_save(state, path):
    temp = str(path) + ".tmp"
    torch.save(state, temp)
    if os.path.exists(path):
        bak = str(path) + ".bak"
        if os.path.exists(bak): os.remove(bak)
        os.rename(path, bak)
    os.rename(temp, path)
    print(f"\n[JARVIS SECURED] Step {state['step']} saved to disk.")

def run_jarvis():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    print("="*60)
    print("INITIATING JARVIS LABS L4 SUPER-DISTILLATION")
    print(f"Targeting Massive Parallelism. Batch Size: {BATCH_SIZE}")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load Teacher in 4-bit to save VRAM for massive batch sizes
    print("Loading 4-bit Quantized Teacher...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    config = MultiScaleBDHConfig(
        vocab_size=teacher.config.vocab_size,
        n_embd=256, n_layer=11, n_head=8, ffn_dim=1024, # 11-Layer RYS Architecture
        max_seq_len=SEQ_LEN, decay_rates=[0.95, 0.99, 0.995], hebbian_lr=0.0005
    )
    student = MultiScaleBDH(config).to(device)
    
    start_step = 0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in [CHECKPOINT_PATH, Path(str(CHECKPOINT_PATH)+".bak")]:
        if p.exists():
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
                student.load_state_dict(ckpt['model'], strict=False)
                start_step = ckpt.get('step', 0)
                print(f"Resumed from Step {start_step}")
                break
            except: continue

    def collate_fn(b):
        return tokenizer(b, padding='max_length', truncation=True, max_length=SEQ_LEN, return_tensors='pt')

    dataloader = DataLoader(LazyTinyStories(), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(student.parameters(), lr=8.0e-5, weight_decay=0.02)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=20000)
    for _ in range(start_step): scheduler.step()
    scaler = torch.amp.GradScaler('cuda')

    student.train()
    session_start = time.time()
    
    print(f"\n[SYSTEM ONLINE] VRAM Capacity Unleashed. Auto-shutdown in {TARGET_HOURS} hours.")

    try:
        for i, batch in enumerate(dataloader):
            elapsed_hours = (time.time() - session_start) / 3600
            if elapsed_hours >= TARGET_HOURS:
                print(f"\n[JARVIS TIME LIMIT] {TARGET_HOURS} hours reached. Commencing safe shutdown.")
                break

            current_step = (i // GRAD_ACCUM) + start_step

            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = ids.clone()
            labels[mask == 0] = -100

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    t_logits = teacher(ids, attention_mask=mask).logits
                s_logits, _ = student(ids)
                loss, kl, ce = sota_distillation_loss(s_logits, t_logits, labels)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                if current_step % 5 == 0:
                    vram_usage = torch.cuda.memory_allocated() / 1e9
                    print(f"Step {current_step:5d} | Loss: {loss.item()*GRAD_ACCUM:.4f} | KL: {kl.item():.2f} | VRAM: {vram_usage:.1f}GB | Time: {elapsed_hours:.2f}h")

                if current_step > 0 and current_step % 100 == 0:
                    atomic_save({'model': student.state_dict(), 'step': current_step}, CHECKPOINT_PATH)
                    
            del t_logits, s_logits, loss
            
            # Auto-Recovery GC if VRAM gets too close to 24GB
            if torch.cuda.memory_allocated() > 22e9:
                torch.cuda.empty_cache()
                gc.collect()

    except KeyboardInterrupt:
        print("\nInterrupt detected. Saving...")
    finally:
        try:
            atomic_save({'model': student.state_dict(), 'step': current_step if 'current_step' in locals() else start_step}, CHECKPOINT_PATH)
        except Exception as e: 
            print(f"Final save error: {e}")

if __name__ == "__main__":
    while True:
        try:
            run_jarvis()
            break
        except Exception as e:
            print(f"\n[JARVIS AUTO-HEAL] Incident: {e}. Restarting in 10s...")
            torch.cuda.empty_cache(); gc.collect(); time.sleep(10)
