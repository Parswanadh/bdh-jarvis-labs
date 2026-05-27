#!/usr/bin/env python
"""
BDH HATCHLING - ULTIMATE HACKS (NO SOFTMAX)
Features:
- RYS Architecture (11 Layers)
- Hidden-State Matching (MSE Deep-Layer Distillation)
- Dataset Pivot (Logic/Math Injection)
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
import random

if sys.platform == 'win32':
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except: pass

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- CONFIG ---
MODEL_NAME = "Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("checkpoints/safe/laptop_safe_11L.pt")
DATA_PATH = Path("data/tinystories.txt")
SEQ_LEN = 192 
TOTAL_STEPS = 15000 

# --- HACK 3: DATASET PIVOT (LOGIC INJECTION) ---
LOGIC_PROBLEMS = [
    "Q: John has 15 apples. He gives 5 to Mary and buys 3 more. How many apples does John have now? A: John has 13 apples",
    "Q: A school has 240 students. If 3/5 of them are girls, how many boys are there? A: There are 96 boys",
    "Q: If A is taller than B, and B is taller than C, who is the shortest? A: C is the shortest",
    "Q: What comes next in the sequence: 2, 4, 8, 16, ? A: 32",
    "Q: What is 15 × 14? A: 210",
    "Q: What is 144 ÷ 12? A: 12",
    "Q: What is 2⁸? A: 256",
    "Q: What is √625? A: 25",
    "Q: All cats are mammals. All mammals have kidneys. Does Fluffy the cat have kidneys? A: Yes, Fluffy has kidneys"
]

class MixedLogicDataset(IterableDataset):
    def __iter__(self):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                
                # 10% chance to inject a pure logic problem to teach reasoning
                if random.random() < 0.10:
                    yield random.choice(LOGIC_PROBLEMS)
                else:
                    yield line.strip()

def distillation_loss(student_logits, teacher_logits, labels, temperature=1.5):
    shift_s = student_logits[..., :-1, :].contiguous()
    shift_t = teacher_logits[..., :-1, :].contiguous()
    shift_l = labels[..., 1:].contiguous()
    mask = shift_l != -100
    
    s_flat = shift_s.view(-1, shift_s.size(-1))[mask.view(-1)]
    t_flat = shift_t.view(-1, shift_t.size(-1))[mask.view(-1)]
    l_flat = shift_l.view(-1)[mask.view(-1)]

    _, top_indices = torch.topk(t_flat, 512, dim=-1)
    s_top = torch.gather(s_flat, -1, top_indices)
    t_top = torch.gather(t_flat, -1, top_indices)

    s_log_probs = F.log_softmax(s_top.float() / temperature, dim=-1)
    t_probs = F.softmax(t_top.float() / temperature, dim=-1)
    
    kl = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    ce = F.cross_entropy(s_flat, l_flat)
    
    return 0.5 * kl + 0.5 * ce, kl, ce

def save_state(student, adapter, step, path):
    torch.cuda.empty_cache()
    gc.collect()
    state = {
        'model': {k: v.cpu() for k, v in student.state_dict().items()}, 
        'adapter': {k: v.cpu() for k, v in adapter.state_dict().items()},
        'step': step
    }
    temp = str(path) + ".tmp"
    torch.save(state, temp)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp, path)
    print(f"\n[HACKS SAVED] Step {step} secured.")

def run_hacks():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    
    # HACK 1: RYS SURGERY (11 Layers)
    config = MultiScaleBDHConfig(
        vocab_size=teacher.config.vocab_size,
        n_embd=256, n_layer=11, n_head=8, ffn_dim=1024,
        max_seq_len=SEQ_LEN, decay_rates=[0.95, 0.99, 0.995], hebbian_lr=0.0005
    )
    student = MultiScaleBDH(config).to(device)
    
    # HACK 2: HIDDEN STATE MATCHING
    # Adapter to match student 256-dim to teacher's hidden dim (usually 896 or 1024 or 1536)
    teacher_dim = teacher.config.hidden_size
    adapter = nn.Linear(256, teacher_dim, bias=False).to(device)

    # Hook to capture Student's hidden states before LM head
    student_hiddens = []
    def hook_fn(module, inp, out):
        student_hiddens.append(out)
    student.norm_f.register_forward_hook(hook_fn)

    start_step = 0
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        student.load_state_dict(ckpt['model'], strict=False)
        if 'adapter' in ckpt:
            adapter.load_state_dict(ckpt['adapter'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed 11-Layer RYS Model at Step {start_step}")
    else:
        print("Please run rys_surgery.py first to create the 11-layer checkpoint!")
        return

    dataset = DataLoader(MixedLogicDataset(), batch_size=1, collate_fn=lambda b: tokenizer(b, padding='max_length', truncation=True, max_length=SEQ_LEN, return_tensors='pt'))

    # Combine params for optimizer
    params = list(student.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=4.0e-5, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=TOTAL_STEPS)
    scaler = torch.amp.GradScaler('cuda')

    grad_accum = 16 
    student.train()
    adapter.train()

    print(f"Ultimate Hacks Active (No Softmax). Target: Step {TOTAL_STEPS}...")
    
    try:
        for i, batch in enumerate(dataset):
            current_step = (i // grad_accum) + start_step
            if current_step >= TOTAL_STEPS: break
            if current_step < start_step: continue

            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = ids.clone()
            labels[mask == 0] = -100
            
            student_hiddens.clear() # Reset hook

            with torch.amp.autocast('cuda'):
                # Get Teacher Hidden States & Logits
                with torch.no_grad():
                    t_outputs = teacher(ids, attention_mask=mask, output_hidden_states=True)
                    t_logits = t_outputs.logits
                    t_hidden = t_outputs.hidden_states[-1] # Final hidden state before head
                
                # Get Student Logits (Hook captures hidden state automatically)
                s_logits, _ = student(ids)
                s_hidden = student_hiddens[0]
                
                # Standard Sub-Atomic Logit Loss
                logit_loss, kl, ce = distillation_loss(s_logits, t_logits, labels)
                
                # Hidden-State Matching Loss (MSE)
                # Map student hidden to teacher hidden
                s_hidden_mapped = adapter(s_hidden)
                
                # Only compare non-padded tokens
                valid_mask = mask.unsqueeze(-1).expand_as(t_hidden).bool()
                mse_loss = F.mse_loss(s_hidden_mapped[valid_mask], t_hidden[valid_mask])
                
                # Combine Logit & Hidden-State Loss
                total_loss = (logit_loss + 0.5 * mse_loss) / grad_accum

            scaler.scale(total_loss).backward()
            time.sleep(0.1) 

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
                
                if current_step % 5 == 0:
                    print(f"Step {current_step:5d} | Total: {total_loss.item()*grad_accum:.4f} | KL: {kl.item():.2f} | MSE: {mse_loss.item():.2f}")

                if current_step > start_step and current_step % 50 == 0:
                    save_state(student, adapter, current_step, CHECKPOINT_PATH)
                
                torch.cuda.empty_cache(); gc.collect()

    except KeyboardInterrupt:
        pass
    finally:
        save_state(student, adapter, current_step if 'current_step' in locals() else start_step, CHECKPOINT_PATH)

if __name__ == "__main__":
    run_hacks()
