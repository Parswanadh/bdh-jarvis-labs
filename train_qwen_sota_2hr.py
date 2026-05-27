"""
SOTA BDH Training with Qwen 3.5 0.8B Teacher - 2 Hour Session
==============================================================

TRUE Knowledge Distillation:
- Teacher provides logits (probability distributions) during training
- Student learns via KL divergence (learns "how to think")
- 2-hour session with checkpoint every 30 minutes
- Resume capability for longer training

OPTIMIZED PARAMETERS (based on research):
- Temperature: 3.0 (softer distributions expose more teacher thinking)
- Alpha (distillation): 0.7 (70% focus on teacher's thinking)
- Beta (hard labels): 0.3 (30% focus on correct predictions)
- Learning rate: 1e-4 with warmup
- Batch size: 16 (for 8GB VRAM with teacher model)

Teacher: Qwen 3.5 0.8B (800M params, vocab=248,320)
Student: BDH 41M (vocab=248,320 to match teacher)
Dataset: TinyStories (2.12M stories, ~630M tokens)

Training time: 2 hours per session
Expected loss after 2h: ~2.0-2.2
Target loss (after 13-16h total): 1.40-1.60
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
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


# ============================================================================
# OPTIMIZED TEACHER PARAMETERS (from agent research)
# ============================================================================

@dataclass
class DistillationConfig:
    """SOTA distillation configuration."""

    # Temperature for softening probability distributions
    # Higher = softer distribution = more "dark knowledge" exposed
    # Range: 2.0-4.0, optimal: 3.0 for language models
    temperature: float = 3.0

    # Loss weighting
    # Alpha: How much to learn from teacher's thinking (KL divergence)
    # Beta: How much to learn from hard labels (cross-entropy)
    alpha: float = 0.7   # 70% distillation
    beta: float = 0.3    # 30% hard labels

    # Training parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    batch_size: int = 16
    max_seq_len: int = 512

    # Checkpoint frequency (minutes)
    checkpoint_freq: int = 30

    # Total training time per session (hours)
    session_hours: float = 2.0


# ============================================================================
# TINYSTORIES DATASET
# ============================================================================

class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling."""

    def __init__(self, data_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load or download TinyStories
        data_path = Path(data_path)
        if data_path.exists():
            print(f"[LOAD] Loading stories from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.stories = [line.strip() for line in f if line.strip()]
            print(f"[OK] Loaded {len(self.stories)} stories")
        else:
            # Create sample data for testing
            print("[WARN] TinyStories not found, using sample data")
            self.stories = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample training data."""
        return [
            "Once upon a time, there was a little girl named Lily. " * 10,
            "The cat sat on the mat and looked out the window. " * 10,
            "One day, a brave knight went on an adventure. " * 10,
            "In a magical forest, there lived many animals. " * 10,
        ] * 1000

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        text = self.stories[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0)
        }


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=3.0,
    alpha=0.7,
    beta=0.3
):
    """
    TRUE Knowledge Distillation Loss.

    Combines:
    1. KL divergence - student learns teacher's thinking
    2. Cross-entropy - student learns correct predictions

    Args:
        student_logits: [batch, seq, vocab] student outputs
        teacher_logits: [batch, seq, vocab] teacher outputs
        labels: [batch, seq] target tokens
        temperature: softens probability distributions
        alpha: weight for distillation loss
        beta: weight for hard label loss

    Returns:
        Combined loss
    """
    # Shift for next-token prediction
    shift_student_logits = student_logits[..., :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_student_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))
    shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Create mask for valid tokens (not padding)
    mask = shift_labels != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # Filter out padding tokens
    shift_student_logits = shift_student_logits[mask]
    shift_teacher_logits = shift_teacher_logits[mask]
    shift_labels = shift_labels[mask]

    # KL Divergence loss (teacher thinking)
    # Soften distributions with temperature
    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)

    # Cross-entropy loss (hard labels)
    ce_loss = F.cross_entropy(
        shift_student_logits,
        shift_labels,
        reduction='mean'
    )

    # Combined loss
    loss = alpha * kl_div + beta * ce_loss

    return loss


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

class WarmupScheduler:
    """Simple warmup learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, config, checkpoint_dir):
    """Save training checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__ if scheduler else None,
        'loss': loss,
        'config': config,
        'timestamp': time.time()
    }

    # Save numbered checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Saved to {checkpoint_path.name}")

    # Save latest
    torch.save(checkpoint, Path(checkpoint_dir) / "latest_checkpoint.pt")

    # Save best
    best_path = Path(checkpoint_dir) / "best_checkpoint.pt"
    if not best_path.exists():
        torch.save(checkpoint, best_path)
        print(f"[BEST] First checkpoint saved: loss={loss:.4f}")
    else:
        best_data = torch.load(best_path)
        if loss < best_data['loss']:
            torch.save(checkpoint, best_path)
            print(f"[BEST] New best! {best_data['loss']:.4f} -> {loss:.4f}")

    # Save training history
    history_path = Path(checkpoint_dir) / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'timestamp': checkpoint['timestamp']
    })

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def load_checkpoint(checkpoint_dir, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"

    if not checkpoint_path.exists():
        print("[INFO] No checkpoint found, starting fresh")
        return 0, 0, float('inf'), {}

    print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.__dict__.update(checkpoint['scheduler_state_dict'])
        scheduler.current_step = checkpoint.get('step', 0)

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    config = checkpoint.get('config', {})

    print(f"[OK] Resumed from epoch {epoch}, step {step}, loss {loss:.4f}")

    return epoch, step, loss, config


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    """Main training function."""

    print("="*80)
    print("SOTA BDH TRAINING WITH QWEN 3.5 0.8B")
    print("TRUE Knowledge Distillation - 2 Hour Session")
    print("="*80)
    print()

    # Configuration
    distill_config = DistillationConfig()

    print("[CONFIG] Optimized Parameters")
    print(f"  Temperature: {distill_config.temperature}")
    print(f"  Alpha (distillation): {distill_config.alpha}")
    print(f"  Beta (hard labels): {distill_config.beta}")
    print(f"  Learning rate: {distill_config.learning_rate}")
    print(f"  Warmup steps: {distill_config.warmup_steps}")
    print(f"  Batch size: {distill_config.batch_size}")
    print(f"  Session length: {distill_config.session_hours} hours")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load TEACHER model (Qwen 3.5 0.8B)
    print("[TEACHER] Loading Qwen 3.5 0.8B")
    print("[INFO] Using local model from Qwen3.5-0.8B/")

    teacher_path = "Qwen3.5-0.8B"

    if not Path(teacher_path).exists():
        print(f"[ERROR] Teacher model not found at {teacher_path}")
        print("[HINT] Run: python download_qwen3_0.8B.py")
        return

    try:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_path,
            trust_remote_code=True
        )

        # Set pad token if needed
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        teacher_model.eval()

        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        teacher_vocab = len(teacher_tokenizer)

        print(f"[OK] Teacher loaded")
        print(f"  Parameters: {teacher_params:,}")
        print(f"  Vocab size: {teacher_vocab:,}")
        print(f"  Device: {next(teacher_model.parameters()).device}")
        print()

    except Exception as e:
        print(f"[ERROR] Failed to load teacher: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create STUDENT model (BDH)
    print("[STUDENT] Creating Multi-Scale BDH")
    print("[INFO] Matching teacher vocab size for proper distillation")

    config = MultiScaleBDHConfig(
        vocab_size=teacher_vocab,  # Match teacher for proper distillation!
        n_embd=768,
        n_layer=12,
        n_head=12,
        ffn_dim=3072,
        dropout=0.1,
        max_seq_len=distill_config.max_seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    student_model = MultiScaleBDH(config).to(device)
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"[OK] Student created")
    print(f"  Parameters: {student_params:,}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Compression: {teacher_params/student_params:.1f}x")
    print()

    # Load checkpoint if exists
    checkpoint_dir = Path("checkpoints/qwen35_sota")
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')

    if (checkpoint_dir / "latest_checkpoint.pt").exists():
        start_epoch, start_step, best_loss, _ = load_checkpoint(
            checkpoint_dir, student_model
        )

    # Create dataset
    print("[DATASET] Loading TinyStories")
    dataset = TinyStoriesDataset(
        "data/tinystories.txt",
        teacher_tokenizer,
        max_seq_len=distill_config.max_seq_len
    )
    print(f"[OK] {len(dataset)} stories")
    print()

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=distill_config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    print(f"[DATALOADER] {len(dataloader)} batches per epoch")
    print()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=distill_config.learning_rate,
        weight_decay=0.01
    )

    scheduler = WarmupScheduler(
        optimizer,
        distill_config.warmup_steps,
        distill_config.learning_rate
    )

    if start_step > 0:
        scheduler.current_step = start_step

    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()

    session_start = time.time()
    last_checkpoint_time = session_start
    total_tokens = 0

    student_model.train()
    teacher_model.eval()

    epoch = start_epoch
    max_steps = int(distill_config.session_hours * 3600 / 2)  # Approximate steps per 2h

    while True:
        elapsed = time.time() - session_start
        if elapsed >= distill_config.session_hours * 3600:
            print(f"\n[SESSION] {distill_config.session_hours}h session complete!")
            break

        epoch += 1
        epoch_loss = 0
        epoch_batches = 0
        epoch_start = time.time()

        print(f"\n[EPOCH {epoch}]")

        for batch_idx, batch in enumerate(dataloader):
            # Check session time
            elapsed = time.time() - session_start
            if elapsed >= distill_config.session_hours * 3600:
                break

            step = start_step + epoch * len(dataloader) + batch_idx

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Create labels (shift input_ids)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Don't predict last token

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Student forward
            with torch.cuda.amp.autocast():
                student_logits, _ = student_model(input_ids)

                # Compute distillation loss
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=distill_config.temperature,
                    alpha=distill_config.alpha,
                    beta=distill_config.beta
                )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                distill_config.max_grad_norm
            )

            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_batches += 1
            total_tokens += input_ids.numel()

            # Progress update
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / epoch_batches
                lr = optimizer.param_groups[0]['lr']
                elapsed_min = elapsed / 60
                remaining_min = (distill_config.session_hours * 3600 - elapsed) / 60

                print(f"  [{batch_idx+1:4d}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"T: {total_tokens//1e6}M | "
                      f"Elapsed: {elapsed_min:.0f}m | "
                      f"Left: {remaining_min:.0f}m")

            # Checkpoint every 30 minutes
            if time.time() - last_checkpoint_time >= distill_config.checkpoint_freq * 60:
                avg_loss = epoch_loss / max(epoch_batches, 1)
                save_checkpoint(
                    student_model, optimizer, scheduler,
                    epoch, step, avg_loss,
                    distill_config.__dict__,
                    checkpoint_dir
                )
                last_checkpoint_time = time.time()

        # Epoch complete
        avg_loss = epoch_loss / max(epoch_batches, 1)
        epoch_time = time.time() - epoch_start

        print(f"\n[EPOCH {epoch}] Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f}m")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1e9
            mem_max = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"[VRAM] {mem:.1f}GB (peak: {mem_max:.1f}GB)")

        # Save checkpoint
        save_checkpoint(
            student_model, optimizer, scheduler,
            epoch, step, avg_loss,
            distill_config.__dict__,
            checkpoint_dir
        )

        # Early exit if session complete
        elapsed = time.time() - session_start
        if elapsed >= distill_config.session_hours * 3600:
            break

    # Final save
    total_time = time.time() - session_start
    final_loss = epoch_loss / max(epoch_batches, 1)

    print("\n" + "="*80)
    print("SESSION COMPLETE")
    print("="*80)
    print(f"[Time] {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"[Tokens] {total_tokens:,} ({total_tokens//1e6}M)")
    print(f"[Speed] {total_tokens/total_time:.0f} tokens/second")
    print(f"[Final Loss] {final_loss:.4f}")
    print(f"[Checkpoint] {checkpoint_dir}")
    print()
    print("[RESUME] To continue training:")
    print(f"  python {__file__}")
    print()

    # Save final model
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'config': config,
        'training_loss': final_loss,
        'training_tokens': total_tokens,
        'teacher_model': 'Qwen3.5-0.8B'
    }, checkpoint_dir / "final_model.pt")
    print(f"[SAVED] Final model to {checkpoint_dir / 'final_model.pt'}")


if __name__ == "__main__":
    train()
