#!/usr/bin/env python
"""
IMMORTAL BDH HATCHLING - SLEEP-SAFE AUTO-RESUME
- Self-Healing: Automatically restarts training on any crash.
- Triple Redundancy: Tries 3 different backups if one is corrupted.
- Deep Distillation: 12 Layers, 224 Seq Len, SOTA logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from pathlib import Path
import sys
import time
import gc
import os
import shutil

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- CONFIGURATION ---
MODEL_NAME = "Qwen3.5-0.8B"
CHECKPOINT_DIR = Path("checkpoints/ultimate")
PRIMARY_CHECKPOINT = CHECKPOINT_DIR / "hatchling_latest.pt"
BACKUP_CHECKPOINT = CHECKPOINT_DIR / "hatchling_backup.pt"
SAFE_CHECKPOINT = CHECKPOINT_DIR / "hatchling_safe.pt"

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

def load_any_checkpoint(model, device):
    """Try primary, then backup, then safe checkpoint."""
    for path in [PRIMARY_CHECKPOINT, BACKUP_CHECKPOINT, SAFE_CHECKPOINT]:
        if path.exists():
            try:
                print(f"Attempting to load: {path.name}")
                ckpt = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model'])
                return ckpt.get('step', 0)
            except Exception as e:
                print(f"Checkpoint {path.name} is corrupted: {e}")
    return 0

def save_triple_redundancy(model, step):
    """Rotating backup system: safe -> backup -> primary"""
    torch.cuda.empty_cache()
    state = {'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'step': step}
    
    # Rotate existing files
    if BACKUP_CHECKPOINT.exists():
        if SAFE_CHECKPOINT.exists(): os.remove(SAFE_CHECKPOINT)
        os.rename(BACKUP_CHECKPOINT, SAFE_CHECKPOINT)
    if PRIMARY_CHECKPOINT.exists():
        os.rename(PRIMARY_CHECKPOINT, BACKUP_CHECKPOINT)
    
    torch.save(state, PRIMARY_CHECKPOINT)
    print(f"\n[IMMORTAL] Step {step} secured with triple-redundancy.")

def run_training_session():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # 1. Models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    
    config = MultiScaleBDHConfig(
        vocab_size=teacher.config.vocab_size,
        n_embd=256, n_layer=12, n_head=8, ffn_dim=1024,
        max_seq_len=224, decay_rates=[0.95, 0.99, 0.995], hebbian_lr=0.0005
    )
    
    student = MultiScaleBDH(config).to(device)
    start_step = load_any_checkpoint(student, device)
    
    # 2. Data
    with open("data/tinystories.txt", 'r', encoding='utf-8') as f:
        stories = [line.strip() for line in f if line.strip()][:(100000 + start_step*16)]

    def collate(batch):
        enc = tokenizer(batch, padding='max_length', truncation=True, max_length=224, return_tensors='pt')
        return enc['input_ids'], enc['attention_mask']

    dataloader = DataLoader(stories, batch_size=1, shuffle=True, collate_fn=collate)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=0.01)
    num_steps = 20000
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_steps)
    for _ in range(start_step): scheduler.step()
    scaler = torch.amp.GradScaler('cuda')

    # 4. Loop
    grad_accum = 32
    student.train()
    session_start = time.time()

    print(f"Resuming Immortal Training from Step {start_step}...")
    
    for i, (ids, mask) in enumerate(dataloader):
        current_step = (i // grad_accum) + start_step
        
        ids, mask = ids.to(device), mask.to(device)
        labels = ids.clone()
        labels[mask == 0] = -100

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                t_logits = teacher(ids, attention_mask=mask).logits
            s_logits, _ = student(ids)
            loss = distillation_loss(s_logits, t_logits, labels) / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            if current_step % 2 == 0:
                print(f"Step {current_step:5d} | Loss: {loss.item()*grad_accum:.4f} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

            if current_step > 0 and current_step % 100 == 0:
                save_triple_redundancy(student, current_step)

        del t_logits, s_logits, loss

if __name__ == "__main__":
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            run_training_session()
            print("Training reached end of data or completed steps.")
            break
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"\n[CRASH DETECTED] Error: {e}")
            print("Waiting 10 seconds for VRAM to clear and restarting...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(10)
            # Loop continues, run_training_session() is called again
