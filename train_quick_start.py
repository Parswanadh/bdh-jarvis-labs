"""
QUICK START TRAINING SCRIPT
===========================

Simplified version of production training for quick iteration.
Uses smaller dataset and more aggressive settings for faster results.

Good for:
- Testing the training pipeline
- Quick experiments with hyperparameters
- Verifying hardware compatibility

For production results, use train_production_bdh.py instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


# =============================================================================
# SIMPLIFIED CONFIG
# =============================================================================

@dataclass
class QuickConfig:
    """Quick start configuration."""

    # Model (41M params)
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 256  # Shorter for speed

    # Training
    batch_size: int = 8  # Higher for shorter seq
    gradient_accumulation: int = 4
    num_epochs: int = 5
    learning_rate: float = 5e-4
    warmup_steps: int = 500

    # Distillation
    teacher_model: str = "Qwen/Qwen2.5-0.8B-Instruct"
    temperature: float = 2.5
    alpha: float = 0.3  # 30% CE, 70% KL

    # Stability
    max_grad_norm: float = 1.0
    use_bf16: bool = True

    # Data
    data_path: str = "data/TinyStories_all.txt"
    num_stories: int = 100000  # 100K for quick run (~3-4 hours)

    # Logging
    log_interval: int = 50
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints/quick_start"


# =============================================================================
# SIMPLE DATASET
# =============================================================================

class SimpleDataset(Dataset):
    """Simple in-memory dataset."""

    def __init__(self, stories: List[str], max_len: int = 256):
        self.stories = stories
        self.max_len = max_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]


# =============================================================================
# MAIN TRAINING CLASS
# =============================================================================

class QuickTrainer:
    """Simplified trainer for quick iteration."""

    def __init__(self, config: QuickConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup dtype
        self.dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16

        # Print info
        print("=" * 60)
        print("QUICK START TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Stories: {config.num_stories:,}")
        print(f"Batch size: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
        print("=" * 60)
        print()

        # Initialize
        self._setup_model()
        self._setup_teacher()
        self._setup_data()
        self._setup_optimizer()

    def _setup_model(self):
        """Create student model."""
        print("[MODEL] Creating BDH (41M params)")
        config = MultiScaleBDHConfig(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            ffn_dim=self.config.ffn_dim,
            max_seq_len=self.config.max_seq_len,
        )

        self.model = MultiScaleBDH(config).to(self.device).to(self.dtype)
        params = sum(p.numel() for p in self.model.parameters())
        print(f"[OK] {params/1e6:.1f}M parameters\n")

    def _setup_teacher(self):
        """Load teacher model."""
        print(f"[TEACHER] Loading {self.config.teacher_model}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model,
            trust_remote_code=True
        )

        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.teacher.eval()
        params = sum(p.numel() for p in self.teacher.parameters())
        print(f"[OK] {params/1e6:.0f}M parameters\n")

    def _setup_data(self):
        """Load and prepare data."""
        print("[DATA] Loading stories...")
        data_path = Path(self.config.data_path)

        if not data_path.exists():
            print(f"[ERROR] Data file not found: {data_path}")
            print("[HINT] Run: python download_tinystories.py")
            raise FileNotFoundError(data_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()

        stories = [s.strip() for s in content.split("") if s.strip()]
        stories = stories[:self.config.num_stories]

        # Filter by length
        valid_stories = []
        for s in stories:
            encoded = self.teacher_tokenizer.encode(s, add_special_tokens=False)
            if 10 <= len(encoded) <= self.config.max_seq_len * 2:
                valid_stories.append(s)

        # Split
        split = int(len(valid_stories) * 0.95)
        self.train_stories = valid_stories[:split]
        self.val_stories = valid_stories[split:]

        print(f"[OK] Train: {len(self.train_stories):,} | Val: {len(self.val_stories):,}\n")

    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        num_steps = (len(self.train_stories) // self.config.batch_size) // self.config.gradient_accumulation * self.config.num_epochs
        self.total_steps = num_steps

        print(f"[OPTIM] Total steps: ~{num_steps:,}\n")

    def train(self):
        """Main training loop."""
        print("[TRAIN] Starting...")
        print()

        self.model.train()
        global_step = 0
        best_loss = float('inf')
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Simple batches
            for i in range(0, len(self.train_stories), self.config.batch_size):
                batch_stories = self.train_stories[i:i + self.config.batch_size]

                # Process batch
                loss_dict = self._train_batch(batch_stories)

                # Accumulate
                if (i // self.config.batch_size + 1) % self.config.gradient_accumulation == 0:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Check for NaN
                    if torch.isnan(grad_norm):
                        print(f"\n[ERROR] NaN gradient at step {global_step}")
                        self._save_checkpoint("nan_error", global_step, loss_dict['loss'])
                        return

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.config.log_interval == 0:
                        elapsed = time.time() - start_time
                        rate = global_step / elapsed
                        eta = (self.total_steps - global_step) / rate

                        print(f"  Step {global_step}/{self.total_steps} | "
                              f"Loss: {loss_dict['loss']:.4f} | "
                              f"CE: {loss_dict['ce']:.4f} | "
                              f"KL: {loss_dict['kl']:.4f} | "
                              f"Grad: {grad_norm:.3f} | "
                              f"ETA: {eta/60:.0f}m")

                    # Validation
                    if global_step % self.config.save_interval == 0:
                        val_loss = self._evaluate()
                        is_best = val_loss < best_loss
                        if is_best:
                            best_loss = val_loss

                        print(f"  [VAL] Loss: {val_loss:.4f} | Perplexity: {np.exp(val_loss):.2f}")
                        if is_best:
                            print(f"  [BEST] New best!")
                        self._save_checkpoint("best" if is_best else "checkpoint", global_step, val_loss)

                        # Sample
                        self._generate_sample()

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print(f"Total time: {(time.time() - start_time)/3600:.1f}h")
        print(f"Best loss: {best_loss:.4f}")
        print("=" * 60)

    def _train_batch(self, stories: List[str]) -> dict:
        """Train on one batch."""
        # Tokenize with teacher
        encodings = self.teacher_tokenizer(
            stories,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt"
        )

        input_ids = encodings['input_ids'].to(self.device)

        # Get teacher logits
        with torch.no_grad():
            outputs = self.teacher(input_ids)
            teacher_logits = torch.clamp(outputs.logits, -10, 10)

        # Convert to bytes for student
        byte_inputs = []
        for s in stories:
            byte_list = list(s.encode('utf-8')[:self.config.max_seq_len])
            byte_inputs.append(byte_list)

        # Pad
        max_len = max(len(b) for b in byte_inputs)
        padded = []
        for b in byte_inputs:
            padded.append(b + [0] * (max_len - len(b)))
        student_input_ids = torch.tensor(padded, dtype=torch.long).to(self.device)

        # Forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            student_logits, _ = self.model(student_input_ids)

            # Loss
            loss, loss_dict = self._distillation_loss(
                student_logits, teacher_logits, student_input_ids
            )

        # Backward
        loss.backward()

        return loss_dict

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute distillation loss."""
        # Shift for causal
        s_logits = student_logits[..., :-1, :].contiguous()
        t_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        B, T, V = s_logits.shape
        s_logits = s_logits.view(-1, V)
        t_logits = t_logits.view(-1, V)
        labels = labels.view(-1)

        mask = labels != 0

        # KL loss
        log_s = F.log_softmax(s_logits / self.config.temperature, dim=-1)
        t = F.softmax(t_logits / self.config.temperature, dim=-1)
        kl = F.kl_div(log_s, t, reduction='none').sum(-1)
        kl = (kl * mask.float()).sum() / mask.float().sum()
        kl = kl * (self.config.temperature ** 2)

        # CE loss
        ce = F.cross_entropy(s_logits, labels, ignore_index=0)

        # Combined
        loss = self.config.alpha * ce + (1 - self.config.alpha) * kl

        return loss, {'loss': loss.item(), 'ce': ce.item(), 'kl': kl.item()}

    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        losses = []

        for i in range(0, len(self.val_stories), self.config.batch_size):
            batch = self.val_stories[i:i + self.config.batch_size]

            # Tokenize
            encodings = self.teacher_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len,
                return_tensors="pt"
            )
            input_ids = encodings['input_ids'].to(self.device)

            with torch.no_grad():
                outputs = self.teacher(input_ids)
                teacher_logits = torch.clamp(outputs.logits, -10, 10)

            # Byte inputs
            byte_inputs = []
            for s in batch:
                byte_list = list(s.encode('utf-8')[:self.config.max_seq_len])
                byte_inputs.append(byte_list)

            max_len = max(len(b) for b in byte_inputs)
            padded = []
            for b in byte_inputs:
                padded.append(b + [0] * (max_len - len(b)))
            student_input_ids = torch.tensor(padded, dtype=torch.long).to(self.device)

            with torch.cuda.amp.autocast(dtype=self.dtype):
                student_logits, _ = self.model(student_input_ids)
                loss, _ = self._distillation_loss(student_logits, teacher_logits, student_input_ids)

            losses.append(loss.item())

        self.model.train()
        return np.mean(losses)

    def _generate_sample(self):
        """Generate a sample."""
        prompt = "Once upon a time, there was a little"
        context = list(prompt.encode('utf-8'))
        context = torch.tensor([context], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(context, max_new_tokens=50, temperature=0.8)

        text = bytes(generated[0].cpu().tolist()).decode('utf-8', errors='ignore')
        print(f"  [GEN] {text[:100]}...")
        self.model.train()

    def _save_checkpoint(self, name: str, step: int, loss: float):
        """Save checkpoint."""
        path = self.checkpoint_dir / f"{name}_{step}.pt"
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        # Also save as latest
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir / "latest.pt")


def main():
    config = QuickConfig()
    trainer = QuickTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
