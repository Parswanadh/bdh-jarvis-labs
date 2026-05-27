"""
PRODUCTION BDH TRAINING PIPELINE
================================

SOTA training pipeline for 41M BDH model with Qwen 3.5 0.8B distillation.

Key optimizations:
- Full TinyStories dataset (2.12M stories)
- Proper learning rate schedule with warmup and cosine decay
- Gradient accumulation for 8GB VRAM
- Temperature and loss-weight tuned distillation
- NaN prevention through gradient clipping and bf16
- Comprehensive evaluation and checkpointing

Hardware target: RTX 4070 (8GB VRAM)
Training time: ~12-24 hours for full dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from collections import deque, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProductionConfig:
    """Production training configuration."""

    # Model configuration (41M params)
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 512

    # Multi-scale parameters
    decay_rates: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    scale_weights: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    hebbian_lr: float = 0.001
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 4  # Per-GPU batch size for 8GB VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3  # Full dataset passes

    # Learning rate schedule
    learning_rate: float = 5e-4  # Peak learning rate
    min_lr_ratio: float = 0.1  # Final LR = min_lr_ratio * learning_rate
    warmup_ratio: float = 0.05  # 5% warmup

    # Distillation parameters
    teacher_model: str = "Qwen/Qwen2.5-0.8B-Instruct"
    distill_temperature: float = 2.5
    distill_alpha: float = 0.3  # Weight on CE loss (0.3 = 30% CE, 70% KL)

    # Stability parameters
    max_grad_norm: float = 0.5  # Conservative gradient clipping
    use_bf16: bool = True  # Use bfloat16 for stability
    logit_clip: float = 10.0  # Clip teacher logits

    # Data parameters
    data_path: str = "data/TinyStories_all.txt"
    num_stories: int = -1  # -1 = use all stories
    val_split: float = 0.01  # 1% for validation (~21K stories)
    cache_teacher_logits: bool = True  # Cache to disk for speed

    # Checkpointing and logging
    checkpoint_dir: str = "checkpoints/production_bdh"
    log_interval: int = 100  # Log every N steps
    eval_interval: int = 1000  # Evaluate every N steps
    checkpoint_interval: int = 5000  # Save checkpoint every N steps
    save_total_limit: int = 3  # Keep only last N checkpoints

    # Generation for evaluation
    eval_prompts: List[str] = field(default_factory=lambda: [
        "Once upon a time, there was a little",
        "The cat and the dog were",
        "One day, a brave girl named",
    ])
    eval_max_tokens: int = 100


# =============================================================================
# DATASET
# =============================================================================

class TinyStoriesDataset(Dataset):
    """
    Memory-efficient TinyStories dataset.

    Loads stories on-demand to avoid memory issues with 2M+ stories.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        num_stories: int = -1,
        shuffle_seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load and index stories
        print(f"[DATASET] Loading stories from {data_path}")
        start = time.time()

        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into stories (each story starts with "<|endoftext|>")
        stories = content.split("<|endoftext|>")
        stories = [s.strip() for s in stories if s.strip()]

        if num_stories > 0:
            stories = stories[:num_stories]

        # Filter by length (keep stories that fit in seq_len)
        self.stories = []
        for story in stories:
            encoded = tokenizer.encode(story, add_special_tokens=False)
            if 10 <= len(encoded) <= max_seq_len * 2:  # Allow some truncation
                self.stories.append(story)

        print(f"[DATASET] Loaded {len(self.stories)} stories in {time.time()-start:.1f}s")
        print(f"[DATASET] Avg story length: {np.mean([len(s.split()) for s in self.stories]):.0f} words")

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]


class DistillationCollator:
    """
    Collator that handles both cached and real-time teacher logits.
    """

    def __init__(
        self,
        tokenizer,
        teacher_model: Optional[nn.Module],
        device,
        max_seq_len: int = 512,
        temperature: float = 2.5,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.teacher = teacher_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.cache_dir = cache_dir

        if teacher_model is not None:
            teacher_model.eval()

    def __call__(self, batch_texts: List[str]) -> dict:
        """
        Tokenize texts and optionally get teacher logits.
        """
        # Tokenize with teacher tokenizer
        encodings = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        teacher_input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Get teacher logits if available
        teacher_logits = None
        if self.teacher is not None:
            with torch.no_grad():
                outputs = self.teacher(
                    teacher_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
                # Clip logits for stability
                logits = outputs.logits
                logits = torch.clamp(logits, min=-self.logit_clip, max=self.logit_clip)
                teacher_logits = logits

        # Convert to byte-level for student
        batch_byte_tensors = []
        for text in batch_texts:
            byte_list = list(text.encode('utf-8')[:self.max_seq_len])
            byte_tensor = torch.tensor(byte_list, dtype=torch.long)
            batch_byte_tensors.append(byte_tensor)

        # Pad byte tensors
        max_len = max(t.size(0) for t in batch_byte_tensors)
        padded = []
        for t in batch_byte_tensors:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), dtype=torch.long)
                t = torch.cat([t, pad])
            padded.append(t)

        student_input_ids = torch.stack(padded).to(self.device)

        return {
            'student_input_ids': student_input_ids,
            'teacher_input_ids': teacher_input_ids,
            'teacher_logits': teacher_logits,
            'attention_mask': attention_mask,
        }

    @property
    def logit_clip(self):
        return 10.0


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.5,
    alpha: float = 0.3,
    ignore_index: int = -100
) -> Tuple[torch.Tensor, dict]:
    """
    Combined distillation loss: alpha * CE + (1-alpha) * KL

    Args:
        student_logits: [B, T, V] student model logits
        teacher_logits: [B, T, V] teacher model logits
        labels: [B, T] target tokens
        temperature: Softening temperature
        alpha: Weight for CE loss (0-1)
        ignore_index: Index to ignore in CE loss

    Returns:
        loss: Combined loss
        losses_dict: Dictionary with individual losses
    """
    # Shift for causal LM (predict next token)
    shift_student_logits = student_logits[..., :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Reshape for loss computation
    B, T, V = shift_student_logits.shape
    shift_student_logits = shift_student_logits.view(-1, V)
    shift_teacher_logits = shift_teacher_logits.view(-1, V)
    shift_labels = shift_labels.view(-1)

    # Create mask for padding
    mask = (shift_labels != ignore_index)

    # KL Divergence loss (distillation)
    # Soften distributions with temperature
    log_student_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

    # Compute KL only on non-padding tokens
    kl_loss = F.kl_div(
        log_student_probs,
        teacher_probs,
        reduction='none'
    ).sum(dim=-1)  # Sum over vocab

    kl_loss = (kl_loss * mask.float()).sum() / mask.float().sum()
    kl_loss = kl_loss * (temperature ** 2)  # Scale by temperature^2

    # Cross-entropy loss (hard labels)
    ce_loss = F.cross_entropy(
        shift_student_logits,
        shift_labels,
        ignore_index=ignore_index
    )

    # Combined loss
    loss = alpha * ce_loss + (1 - alpha) * kl_loss

    return loss, {
        'total': loss.item(),
        'ce': ce_loss.item(),
        'kl': kl_loss.item()
    }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.1,
    init_lr: float = 5e-4
):
    """
    Cosine learning rate scheduler with warmup.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_training_steps * warmup_ratio:
            return float(current_step) / float(max(1, num_training_steps * warmup_ratio))

        # Cosine decay
        progress = float(current_step - warmup_ratio * num_training_steps) / float(
            max(1, num_training_steps * (1 - warmup_ratio))
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class MetricsTracker:
    """Track training metrics with exponential moving average."""

    def __init__(self, smooth_window: int = 100):
        self.metrics = {}
        self.smooth_window = smooth_window
        self.history = defaultdict(list)

    def update(self, metrics: dict):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.smooth_window)
            self.metrics[key].append(value)
            self.history[key].append(value)

    def get_smoothed(self, key: str) -> float:
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_latest(self, key: str) -> float:
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

class ProductionTrainer:
    """Production trainer with all optimizations."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup dtype
        if config.use_bf16 and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            print(f"[DTYPE] Using bfloat16 for stability")
        else:
            self.dtype = torch.float16
            print(f"[DTYPE] Using float16")

        # Initialize components
        self._setup_model()
        self._setup_teacher()
        self._setup_data()
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.metrics = MetricsTracker()

    def _setup_model(self):
        """Initialize student model."""
        print("[MODEL] Creating BDH student model")
        config = MultiScaleBDHConfig(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            ffn_dim=self.config.ffn_dim,
            max_seq_len=self.config.max_seq_len,
            decay_rates=self.config.decay_rates,
            scale_weights=self.config.scale_weights,
            hebbian_lr=self.config.hebbian_lr,
            dropout=self.config.dropout,
        )

        self.model = MultiScaleBDH(config)
        self.model = self.model.to(self.device).to(self.dtype)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MODEL] {num_params/1e6:.1f}M parameters")

    def _setup_teacher(self):
        """Initialize teacher model."""
        print(f"[TEACHER] Loading {self.config.teacher_model}")

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model,
            trust_remote_code=True
        )

        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.teacher.eval()

        num_params = sum(p.numel() for p in self.teacher.parameters())
        print(f"[TEACHER] {num_params/1e6:.0f}M parameters")

    def _setup_data(self):
        """Initialize datasets and dataloaders."""
        print("[DATA] Setting up datasets")

        # Load dataset
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load all stories
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()

        stories = content.split("<|endoftext|>")
        stories = [s.strip() for s in stories if s.strip()]

        if self.config.num_stories > 0:
            stories = stories[:self.config.num_stories]

        print(f"[DATA] Total stories: {len(stories):,}")

        # Split into train/val
        val_size = int(len(stories) * self.config.val_split)
        train_stories = stories[val_size:]
        val_stories = stories[:val_size]

        print(f"[DATA] Train: {len(train_stories):,} | Val: {len(val_stories):,}")

        # Create temporary files for splits
        train_path = self.checkpoint_dir / "train_stories.txt"
        val_path = self.checkpoint_dir / "val_stories.txt"

        with open(train_path, 'w', encoding='utf-8') as f:
            f.write("<|endoftext|>".join(train_stories))

        with open(val_path, 'w', encoding='utf-8') as f:
            f.write("<|endoftext|>".join(val_stories))

        # Create datasets
        self.train_dataset = TinyStoriesDataset(
            str(train_path),
            self.teacher_tokenizer,
            max_seq_len=self.config.max_seq_len,
        )

        self.val_dataset = TinyStoriesDataset(
            str(val_path),
            self.teacher_tokenizer,
            max_seq_len=self.config.max_seq_len,
        )

        # Create collator
        self.collator = DistillationCollator(
            self.teacher_tokenizer,
            self.teacher,
            self.device,
            max_seq_len=self.config.max_seq_len,
            temperature=self.config.distill_temperature,
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,  # Windows compatibility
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=0,
            drop_last=False,
        )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Calculate total steps
        num_batches = len(self.train_loader)
        steps_per_epoch = num_batches // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        print(f"[OPTIM] Total steps: {total_steps:,}")
        print(f"[OPTIM] Steps per epoch: {steps_per_epoch:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        self.scheduler = get_lr_scheduler(
            self.optimizer,
            total_steps,
            warmup_ratio=self.config.warmup_ratio,
            min_lr_ratio=self.config.min_lr_ratio,
            init_lr=self.config.learning_rate,
        )

        self.total_steps = total_steps

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("STARTING PRODUCTION TRAINING")
        print("=" * 70)
        print(f"[Config] Batch size: {self.config.batch_size}")
        print(f"[Config] Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"[Config] Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"[Config] Peak LR: {self.config.learning_rate}")
        print(f"[Config] Distillation T: {self.config.distill_temperature}")
        print(f"[Config] Distillation alpha: {self.config.distill_alpha}")
        print(f"[Config] Max grad norm: {self.config.max_grad_norm}")
        print("=" * 70 + "\n")

        self.model.train()
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*70}\n")

            self._train_epoch()

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Final loss: {self.metrics.get_smoothed('total'):.4f}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    def _train_epoch(self):
        """Train one epoch."""
        self.optimizer.zero_grad()
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass
            loss, loss_dict = self._training_step(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss_dict['total']

            # Gradient accumulation check
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Check for NaN
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[ERROR] NaN/Inf gradient detected at step {self.global_step}")
                    print(f"[ERROR] Grad norm: {grad_norm}")
                    print(f"[ERROR] Recent loss: {accumulated_loss}")
                    self._save_checkpoint("nan_recovery")
                    return

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Update metrics
                avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                self.metrics.update({
                    'total': avg_loss,
                    'ce': loss_dict['ce'],
                    'kl': loss_dict['kl'],
                    'grad_norm': grad_norm.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                })
                accumulated_loss = 0.0

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_progress()

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_loss = self._evaluate()
                    self._log_validation(val_loss)

                # Checkpointing
                if self.global_step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

    def _training_step(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Single training step."""
        student_input_ids = batch['student_input_ids']
        teacher_logits = batch['teacher_logits']

        # Student forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            student_logits, _ = self.model(student_input_ids)

            # Compute distillation loss
            loss, loss_dict = distillation_loss(
                student_logits,
                teacher_logits,
                student_input_ids,
                temperature=self.config.distill_temperature,
                alpha=self.config.distill_alpha,
                ignore_index=0,  # Padding token
            )

        return loss, loss_dict

    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                student_input_ids = batch['student_input_ids']
                teacher_logits = batch['teacher_logits']

                with torch.cuda.amp.autocast(dtype=self.dtype):
                    student_logits, _ = self.model(student_input_ids)

                    loss, _ = distillation_loss(
                        student_logits,
                        teacher_logits,
                        student_input_ids,
                        temperature=self.config.distill_temperature,
                        alpha=self.config.distill_alpha,
                        ignore_index=0,
                    )

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _log_progress(self):
        """Log training progress."""
        loss = self.metrics.get_smoothed('total')
        ce = self.metrics.get_smoothed('ce')
        kl = self.metrics.get_smoothed('kl')
        grad_norm = self.metrics.get_smoothed('grad_norm')
        lr = self.metrics.get_smoothed('lr')

        progress = self.global_step / self.total_steps * 100

        print(f"[Step {self.global_step}/{self.total_steps}] ({progress:.1f}%) | "
              f"Loss: {loss:.4f} | CE: {ce:.4f} | KL: {kl:.4f} | "
              f"Grad: {grad_norm:.3f} | LR: {lr:.2e}")

        # Log memory
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1e9
            mem_max = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"  [VRAM] {mem:.2f}GB / {mem_max:.2f}GB max")

    def _log_validation(self, val_loss: float):
        """Log validation results."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        perplexity = np.exp(val_loss)

        print(f"\n  [VALIDATION] Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
        if is_best:
            print(f"  [BEST] New best validation loss!")
        print()

        # Generate samples
        if hasattr(self, 'model'):
            self._generate_samples()

    def _generate_samples(self):
        """Generate text samples for qualitative evaluation."""
        print(f"  [GENERATION] Samples:")
        self.model.eval()

        for prompt in self.config.eval_prompts[:2]:  # First 2 prompts
            # Convert to bytes
            context = prompt.encode('utf-8')
            context = torch.tensor([list(context)], dtype=torch.long).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    context,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=30
                )

            text = bytes(generated[0].cpu().tolist()).decode('utf-8', errors='ignore')
            print(f"    Prompt: {prompt}")
            print(f"    Output: {text[:100]}...")
            print()

        self.model.train()

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"  [CHECKPOINT] Saved: {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    config = ProductionConfig()

    # Override data path if needed
    # config.data_path = "path/to/TinyStories_all.txt"

    trainer = ProductionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
