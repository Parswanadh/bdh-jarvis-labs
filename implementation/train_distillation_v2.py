"""
BDH v2 Knowledge Distillation Training Pipeline
================================================

Day 5 of the BDH 9-day master plan.

Complete knowledge distillation pipeline with:
- KL divergence distillation with temperature scaling
- Vocab projection layer (teacher 151936 -> student 32000) via low-rank projection
- CKA (Centered Kernel Alignment) loss for hidden state distillation
- Gradient accumulation
- Checkpoint save/resume
- Validation loop
- Early stopping
- Cosine LR schedule with linear warmup
- Gradient clipping (max_norm=1.0)
- Synthetic dataset fallback for pipeline testing
- argparse CLI with all hyperparameters
- Logging every 100 steps

Two predefined configs:
- v2_small:  256d, 6L, 4H  (~25M params)
- v2_medium: 512d, 12L, 8H (~150M params)

Usage:
    python -m implementation.train_distillation_v2 --config v2_small
    python -m implementation.train_distillation_v2 --config v2_medium --epochs 10
    python -m implementation.train_distillation_v2 --synthetic --steps 1000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from implementation.bdh_v2_clean import BDHv2Config, BDHv2, VocabProjection


# ============================================================================
# Predefined Configs
# ============================================================================

PRESET_CONFIGS = {
    "v2_small": {
        "n_embd": 256,
        "n_layer": 6,
        "n_head": 4,
        "ffn_dim": 1024,
    },
    "v2_medium": {
        "n_embd": 512,
        "n_layer": 12,
        "n_head": 8,
        "ffn_dim": 2048,
    },
}


# ============================================================================
# Synthetic Dataset
# ============================================================================


class SyntheticDistillDataset(Dataset):
    """
    Synthetic dataset for pipeline testing.
    Generates random token IDs and fake teacher logits.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 128,
        student_vocab: int = 32000,
        teacher_vocab: int = 151936,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.student_vocab = student_vocab
        self.teacher_vocab = teacher_vocab

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(idx)

        input_ids = torch.randint(0, self.student_vocab, (self.seq_len,), generator=g)
        teacher_logits = torch.randn(self.seq_len, self.teacher_vocab, generator=g)

        return {
            "input_ids": input_ids,
            "teacher_logits": teacher_logits,
            "labels": input_ids.clone(),
        }


class RealDataset(Dataset):
    """
    Wrapper for real distillation data.
    Expects a directory with .pt files containing:
    - input_ids: [T]
    - teacher_logits: [T, teacher_vocab]
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if max_samples:
            self.files = self.files[:max_samples]
        self.cache = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx not in self.cache:
            data = torch.load(self.files[idx], weights_only=True)
            self.cache[idx] = data
        return self.cache[idx]


# ============================================================================
# Loss Functions
# ============================================================================


def kl_div_loss_with_temp(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL divergence loss with temperature scaling.

    Args:
        student_logits: [B, T, student_vocab]
        teacher_logits: [B, T, student_vocab] (already projected)
        temperature: softmax temperature

    Returns:
        scalar KL loss
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
        log_target=False,
    )
    return loss * (temperature**2)


def cka_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    Centered Kernel Alignment loss for hidden state distillation.

    Uses linear CKA (more stable than RBF for distillation).

    Args:
        student_hidden: [B, T, d_student]
        teacher_hidden: [B, T, d_teacher]

    Returns:
        scalar CKA loss (1 - CKA similarity)
    """
    student_centered = student_hidden - student_hidden.mean(dim=1, keepdim=True)
    teacher_centered = teacher_hidden - teacher_hidden.mean(dim=1, keepdim=True)

    K_s = torch.bmm(student_centered, student_centered.transpose(1, 2))
    K_t = torch.bmm(teacher_centered, teacher_centered.transpose(1, 2))

    hsic = torch.bmm(K_s, K_t)
    hsic_loss = -hsic.diagonal(dim1=1, dim2=2).sum(dim=1).mean()

    norm_s = torch.bmm(K_s, K_s)
    norm_t = torch.bmm(K_t, K_t)
    trace_s = norm_s.diagonal(dim1=1, dim2=2).sum(dim=1)
    trace_t = norm_t.diagonal(dim1=1, dim2=2).sum(dim=1)
    norm = torch.sqrt(trace_s * trace_t + 1e-8)

    cka_similarity = -hsic_loss / norm
    return 1.0 - cka_similarity.mean()


# ============================================================================
# LR Scheduler
# ============================================================================


class CosineLRWithWarmup:
    """Cosine LR schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

    def state_dict(self) -> Dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: Dict[str, Any]):
        self.current_step = state["current_step"]


# ============================================================================
# Checkpoint Manager
# ============================================================================


class CheckpointManager:
    """Handles save/resume of training state."""

    def __init__(self, save_dir: str, max_keep: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.best_val_loss = float("inf")
        self.checkpoints = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineLRWithWarmup,
        step: int,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        extra: Optional[Dict] = None,
    ):
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "extra": extra or {},
        }

        if is_best:
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_loss

        ckpt_path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, ckpt_path)
        self.checkpoints.append(ckpt_path)

        while len(self.checkpoints) > self.max_keep:
            old = self.checkpoints.pop(0)
            if old.exists():
                old.unlink()

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineLRWithWarmup,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if path is None:
            ckpts = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
            if not ckpts:
                raise FileNotFoundError("No checkpoints found")
            path = str(ckpts[-1])

        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        return {
            "step": checkpoint["step"],
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "extra": checkpoint.get("extra", {}),
        }

    def has_checkpoint(self) -> bool:
        return len(list(self.save_dir.glob("checkpoint_step_*.pt"))) > 0


# ============================================================================
# Early Stopping
# ============================================================================


class EarlyStopping:
    """Stop training when val loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ============================================================================
# Trainer
# ============================================================================


class DistillationTrainer:
    """Complete knowledge distillation trainer for BDH v2."""

    def __init__(
        self,
        model: BDHv2,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineLRWithWarmup,
        device: torch.device,
        config: argparse.Namespace,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.ce_weight = config.ce_weight
        self.kl_weight = config.kl_weight
        self.cka_weight = config.cka_weight
        self.temperature = config.temperature
        self.grad_accum_steps = config.grad_accum_steps
        self.log_interval = config.log_interval
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

        self.ckpt_manager = CheckpointManager(
            config.output_dir, max_keep=config.max_checkpoints
        )

        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        self._setup_logging()

    def _setup_logging(self):
        log_path = Path(self.config.output_dir) / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("BDHv2Distill")

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation loss."""
        input_ids = batch["input_ids"].to(self.device)
        teacher_logits = batch["teacher_logits"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        student_logits, _ = self.model(input_ids)
        projected_teacher = self.model.project_teacher_logits(teacher_logits)

        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            reduction="mean",
        )

        kl_loss = kl_div_loss_with_temp(
            student_logits, projected_teacher, self.temperature
        )

        cka_loss_val = torch.tensor(0.0, device=self.device)
        if self.cka_weight > 0 and return_hidden:
            x = self.model.token_embedding(input_ids)
            for layer in self.model.layers:
                x, _, _ = layer(x)
            student_hidden = self.model.norm_f(x)

            if self.model.vocab_projection is not None:
                teacher_hidden = (
                    projected_teacher @ self.model.vocab_projection.up.weight
                )
                teacher_hidden = teacher_hidden[:, :, : student_hidden.size(-1)]
                cka_loss_val = cka_loss(student_hidden, teacher_hidden)

        total_loss = (
            self.ce_weight * ce_loss
            + self.kl_weight * kl_loss
            + self.cka_weight * cka_loss_val
        )

        loss_dict = {
            "total": total_loss.item(),
            "ce": ce_loss.item(),
            "kl": kl_loss.item(),
            "cka": cka_loss_val.item(),
        }

        return total_loss, loss_dict

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        total_loss, loss_dict = self.compute_loss(batch, return_hidden=True)
        total_loss = total_loss / self.grad_accum_steps
        total_loss.backward()
        return loss_dict

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "ce": 0.0, "kl": 0.0, "cka": 0.0}
        n_batches = 0
        step_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            loss_dict = self.train_step(batch)

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            if self.global_step > 0 and self.global_step % self.log_interval == 0:
                avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
                elapsed = time.time() - step_start_time
                lr = self.scheduler.get_lr()

                log_msg = (
                    f"Step {self.global_step} | "
                    f"Epoch {epoch} | "
                    f"Loss: {avg_losses['total']:.4f} "
                    f"(CE: {avg_losses['ce']:.4f}, "
                    f"KL: {avg_losses['kl']:.4f}, "
                    f"CKA: {avg_losses['cka']:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                self.logger.info(log_msg)
                step_start_time = time.time()

        avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        val_losses = {"total": 0.0, "ce": 0.0, "kl": 0.0, "cka": 0.0}
        n_batches = 0

        for batch in val_loader:
            _, loss_dict = self.compute_loss(batch, return_hidden=False)
            for k, v in loss_dict.items():
                val_losses[k] += v
            n_batches += 1

        avg_losses = {k: v / max(n_batches, 1) for k, v in val_losses.items()}
        return avg_losses

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int = 10,
        start_epoch: int = 0,
        start_step: int = 0,
    ):
        """Full training loop with validation, checkpointing, and early stopping."""
        self.global_step = start_step
        self.logger.info(
            f"Starting training from epoch {start_epoch}, step {start_step}"
        )
        self.logger.info(f"Total epochs: {epochs}")
        self.logger.info(
            f"Loss weights: CE={self.ce_weight}, KL={self.kl_weight}, CKA={self.cka_weight}"
        )
        self.logger.info(f"Temperature: {self.temperature}")
        self.logger.info(f"Gradient accumulation: {self.grad_accum_steps} steps")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            train_losses = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start

            self.logger.info(
                f"Epoch {epoch} | Train Loss: {train_losses['total']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.val_losses.append(val_losses["total"])
                self.logger.info(f"Epoch {epoch} | Val Loss: {val_losses['total']:.4f}")

                should_stop = self.early_stopping.step(val_losses["total"])

                is_best = val_losses["total"] < self.ckpt_manager.best_val_loss
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    epoch=epoch,
                    val_loss=val_losses["total"],
                    is_best=is_best,
                    extra={"train_losses": train_losses},
                )

                if should_stop:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}, "
                        f"best val loss: {self.early_stopping.best_loss:.4f}"
                    )
                    break
            else:
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    epoch=epoch,
                    val_loss=train_losses["total"],
                    is_best=False,
                    extra={"train_losses": train_losses},
                )

        self.logger.info("Training complete.")
        self.logger.info(f"Final best val loss: {self.ckpt_manager.best_val_loss:.4f}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BDH v2 Knowledge Distillation Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="v2_small",
        choices=list(PRESET_CONFIGS.keys()),
        help="Preset model config (v2_small or v2_medium)",
    )
    parser.add_argument(
        "--n_embd", type=int, default=None, help="Override embedding dim"
    )
    parser.add_argument("--n_layer", type=int, default=None, help="Override num layers")
    parser.add_argument("--n_head", type=int, default=None, help="Override num heads")
    parser.add_argument("--ffn_dim", type=int, default=None, help="Override FFN dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--max_seq_len", type=int, default=2048, help="Max sequence length"
    )

    parser.add_argument(
        "--student_vocab",
        type=int,
        default=32000,
        help="Student vocab size (TinyLlama)",
    )
    parser.add_argument(
        "--teacher_vocab", type=int, default=151936, help="Teacher vocab size (Qwen3.5)"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument(
        "--min_lr_ratio", type=float, default=0.1, help="Min LR ratio for cosine decay"
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--ce_weight", type=float, default=0.2, help="Cross-entropy loss weight"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.5, help="KL divergence loss weight"
    )
    parser.add_argument("--cka_weight", type=float, default=0.3, help="CKA loss weight")

    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to real data directory"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic dataset for testing"
    )
    parser.add_argument(
        "--synthetic_samples", type=int, default=10000, help="Num synthetic samples"
    )
    parser.add_argument(
        "--synthetic_seq_len", type=int, default=128, help="Synthetic sequence length"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/bdh_v2_distill",
        help="Output directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--max_checkpoints", type=int, default=3, help="Max rolling checkpoints to keep"
    )

    parser.add_argument(
        "--early_stopping_patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Min delta for early stopping",
    )

    parser.add_argument(
        "--log_interval", type=int, default=100, help="Log every N steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    preset = PRESET_CONFIGS[args.config]
    model_config = BDHv2Config(
        vocab_size=args.student_vocab,
        n_embd=args.n_embd or preset["n_embd"],
        n_layer=args.n_layer or preset["n_layer"],
        n_head=args.n_head or preset["n_head"],
        ffn_dim=args.ffn_dim or preset["ffn_dim"],
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_vocab_projection=True,
        teacher_vocab_size=args.teacher_vocab,
    )

    model = BDHv2(model_config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    if args.synthetic or args.data_dir is None:
        print("Using synthetic dataset")
        full_dataset = SyntheticDistillDataset(
            num_samples=args.synthetic_samples,
            seq_len=args.synthetic_seq_len,
            student_vocab=args.student_vocab,
            teacher_vocab=args.teacher_vocab,
        )
    else:
        print(f"Loading real data from {args.data_dir}")
        full_dataset = RealDataset(args.data_dir)

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineLRWithWarmup(
        optimizer=optimizer,
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    trainer = DistillationTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=args,
    )

    start_epoch = 0
    start_step = 0
    if args.resume or trainer.ckpt_manager.has_checkpoint():
        resume_path = args.resume
        if resume_path is None:
            ckpts = sorted(trainer.ckpt_manager.save_dir.glob("checkpoint_step_*.pt"))
            if ckpts:
                resume_path = str(ckpts[-1])
        if resume_path:
            print(f"Resuming from {resume_path}")
            state = trainer.ckpt_manager.load(model, optimizer, scheduler, resume_path)
            start_epoch = state["epoch"] + 1
            start_step = state["step"]
            print(f"Resumed: epoch={start_epoch}, step={start_step}")

    config_path = Path(args.output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        start_step=start_step,
    )

    print("Done.")


if __name__ == "__main__":
    main()
