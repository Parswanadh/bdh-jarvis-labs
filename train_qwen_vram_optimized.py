"""
VRAM-OPTIMIZED SOTA BDH Training
=================================

Both teacher and student on GPU for MAXIMUM speed!
Uses teacher's tokenizer directly to avoid projection layer.

Memory Strategy:
- Teacher: Qwen 3.5 0.8B (752M) - fp16 (~1.5GB)
- Student: BDH 30M params - fp16 (~60MB for model)
- Shared embeddings tied to vocab (~200MB)
- Activations: ~1-2GB per batch
- Total: ~5-6GB (fits in 8GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time
import json

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Training
    temperature = 3.0
    alpha = 0.7
    beta = 0.3
    learning_rate = 1e-4
    warmup_steps = 500
    max_grad_norm = 1.0

    # Memory settings
    batch_size = 8
    max_seq_len = 256
    gradient_accumulation = 2

    # Session
    session_hours = 2.0
    checkpoint_freq = 30  # minutes


# ============================================================================
# TINYSTORIES DATASET
# ============================================================================

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=256, num_samples=500000):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        data_path = Path(data_path)
        if data_path.exists():
            print(f"[LOAD] Loading stories from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                all_stories = [line.strip() for line in f if line.strip() and len(line.strip()) > 50]
            print(f"[OK] Loaded {len(all_stories)} stories")

            import random
            random.seed(42)
            self.stories = random.sample(all_stories, min(num_samples, len(all_stories)))
            print(f"[SAMPLE] Using {len(self.stories)} stories")
        else:
            self.stories = ["Once upon a time " * 50] * 10000

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7, beta=0.3):
    """KL divergence + Cross-entropy."""

    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    B, T, V = shift_student.shape
    shift_student = shift_student.view(-1, V)
    shift_teacher = shift_teacher.view(-1, V)
    shift_labels = shift_labels.view(-1)

    # Ignore padding (0 = pad in our tokenizer)
    mask = shift_labels != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    shift_student = shift_student[mask]
    shift_teacher = shift_teacher[mask]
    shift_labels = shift_labels[mask]

    # KL
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # CE
    ce_loss = F.cross_entropy(shift_student, shift_labels, reduction='mean')

    return alpha * kl_div + beta * ce_loss


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }

    torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_e{epoch}_s{step}.pt")
    torch.save(checkpoint, Path(checkpoint_dir) / "latest.pt")

    best_path = Path(checkpoint_dir) / "best.pt"
    if not best_path.exists() or loss < torch.load(best_path)['loss']:
        torch.save(checkpoint, best_path)
        print(f"[BEST] {loss:.4f}")


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"

    if not checkpoint_path.exists():
        return 0, 0, float('inf')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"[RESUME] E{checkpoint['epoch']} S{checkpoint['step']} L={checkpoint['loss']:.4f}")
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train():
    print("="*80)
    print("VRAM-OPTIMIZED SOTA TRAINING")
    print("Teacher + Student on GPU")
    print("="*80)
    print()

    cfg = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load TEACHER on GPU
    print("[TEACHER] Loading Qwen 3.5 0.8B to GPU...")
    teacher_path = "Qwen3.5-0.8B"

    if not Path(teacher_path).exists():
        print(f"[ERROR] Model not found at {teacher_path}")
        return

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # Load to GPU!
        trust_remote_code=True
    )
    teacher_model.eval()

    teacher_vocab = len(teacher_tokenizer)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"[OK] Teacher on GPU: {teacher_params:,} params, vocab={teacher_vocab:,}")
    print()

    # Create STUDENT with same vocab (no projection needed!)
    print("[STUDENT] Creating compact BDH...")
    print("[INFO] Using teacher's vocab to avoid projection layer")

    bdh_config = MultiScaleBDHConfig(
        vocab_size=teacher_vocab,  # Same as teacher!
        n_embd=384,      # Smaller embedding
        n_layer=6,       # Fewer layers
        n_head=6,
        ffn_dim=1536,
        dropout=0.1,
        max_seq_len=cfg.max_seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    student_model = MultiScaleBDH(bdh_config).to(device)
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"[OK] Student: {student_params:,} params, vocab={teacher_vocab:,}")
    print(f"[VRAM CHECK] Student body: {student_params * 2 / 1e9:.2f} GB (fp16)")
    print(f"[VRAM CHECK] Embedding: {teacher_vocab * bdh_config.n_embd * 2 / 1e9:.2f} GB")
    print()

    # Checkpoint
    checkpoint_dir = Path("checkpoints/qwen35_vram")
    start_epoch, start_step, best_loss = load_checkpoint(checkpoint_dir, student_model)

    # Dataset
    print("[DATASET] Loading TinyStories...")
    dataset = TinyStoriesDataset(
        "data/tinystories.txt",
        teacher_tokenizer,
        max_seq_len=cfg.max_seq_len,
        num_samples=500000
    )
    print(f"[OK] {len(dataset)} stories")
    print()

    # DataLoader
    def collate_fn(batch):
        encodings = teacher_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids']
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Don't predict last token
        return {'input_ids': input_ids, 'labels': labels}

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    print(f"[DATALOADER] {len(dataloader)} batches/epoch")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Training
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()

    session_start = time.time()
    last_checkpoint = session_start
    total_tokens = 0

    student_model.train()
    teacher_model.eval()  # Keep in eval mode
    warmup_step = 0

    epoch = start_epoch

    while True:
        elapsed = time.time() - session_start
        if elapsed >= cfg.session_hours * 3600:
            print(f"\n[SESSION] Complete after {elapsed/60:.0f}m!")
            break

        epoch += 1
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            elapsed = time.time() - session_start
            if elapsed >= cfg.session_hours * 3600:
                break

            step = start_step + epoch * len(dataloader) + batch_idx

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Teacher forward (no grad, stays on GPU)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits

            # Student forward
            student_logits, _ = student_model(input_ids)

            # Loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=cfg.temperature,
                alpha=cfg.alpha,
                beta=cfg.beta
            )

            loss = loss / cfg.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                # Warmup
                if warmup_step < cfg.warmup_steps:
                    lr_scale = (warmup_step + 1) / cfg.warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = cfg.learning_rate * lr_scale
                    warmup_step += 1

                torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.gradient_accumulation
            epoch_batches += 1
            total_tokens += input_ids.numel()

            # Progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / epoch_batches
                lr = optimizer.param_groups[0]['lr']
                elapsed_min = elapsed / 60
                remaining = (cfg.session_hours * 3600 - elapsed) / 60

                print(f"[E{epoch}] [{batch_idx+1:5d}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"T: {total_tokens//1e6}M | {elapsed_min:.0f}m | ~{remaining:.0f}m left")

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated(0) / 1e9
                    mem_max = torch.cuda.max_memory_allocated(0) / 1e9
                    print(f"       VRAM: {mem:.2f}GB (peak: {mem_max:.2f}GB)")

            # Checkpoint
            if time.time() - last_checkpoint >= cfg.checkpoint_freq * 60:
                avg_loss = epoch_loss / max(epoch_batches, 1)
                save_checkpoint(student_model, optimizer, epoch, step, avg_loss, checkpoint_dir)
                last_checkpoint = time.time()

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"\n[EPOCH {epoch}] Loss: {avg_loss:.4f}")

        save_checkpoint(student_model, optimizer, epoch, step, avg_loss, checkpoint_dir)

        if elapsed >= cfg.session_hours * 3600:
            break

    # Final
    total_time = time.time() - session_start
    final_loss = epoch_loss / max(epoch_batches, 1)

    print("\n" + "="*80)
    print("SESSION COMPLETE")
    print("="*80)
    print(f"[Time] {total_time/60:.0f}m")
    print(f"[Tokens] {total_tokens:,}")
    print(f"[Speed] {total_tokens/total_time:.0f} tok/s")
    print(f"[Loss] {final_loss:.4f}")
    print()


if __name__ == "__main__":
    train()
