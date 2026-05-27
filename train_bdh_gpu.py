"""
BDH-GPU Training Script
=======================

Complete training pipeline for 10M BDH-GPU model.
Trains on any text dataset using byte-level tokenization.

Features:
- Automatic mixed precision (AMP) for faster training
- Gradient clipping for stability
- Learning rate scheduling with warmup
- Checkpoint saving and loading
- Progress tracking with loss calculation
- Works on CPU or GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from bdh_gpu_10m import BDHConfig, BDHGPUTensor, count_parameters


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Data
    dataset_path: str = "input.txt"  # Path to text file
    val_dataset_path: Optional[str] = None  # Optional validation data

    # Model
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512  # Context length

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    warmup_iters: int = 100

    # Optimization
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # System
    device: str = "auto"  # auto, cpu, cuda
    compile: bool = True  # Use torch.compile for speedup
    dtype: str = "bfloat16"  # float32, bfloat16, float16

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    resume_from: Optional[str] = None


class TextDataset(Dataset):
    """
    Simple text dataset that loads text and converts to byte-level tokens.
    No tokenizer needed - uses raw bytes (ASCII/UTF-8).
    """
    def __init__(self, text: str, max_seq_len: int = 512):
        self.text = text
        self.max_seq_len = max_seq_len

        # Convert text to bytes (0-255)
        self.data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def __len__(self):
        # Number of possible sequences
        return max(0, len(self.data) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        # Get chunk of data
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]  # Input
        y = chunk[1:self.max_seq_len + 1]  # Target (shifted by 1)
        return x, y


def load_data(path: str) -> str:
    """Load text data from file"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using dummy data.")
        return "Hello world! " * 1000

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_lr(it: int, config: TrainingConfig) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay.

    Args:
        it: Current iteration
        config: Training configuration

    Returns:
        Learning rate for this iteration
    """
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * coeff


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig
) -> dict:
    """
    Estimate loss on train and validation sets.

    Args:
        model: The model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        config: Training configuration

    Returns:
        Dictionary with train and val losses
    """
    model.eval()
    losses = {}

    for split, loader in [("train", train_loader)] + (
        [("val", val_loader)] if val_loader else []
    ):
        total_loss = 0.0
        total_samples = 0

        for i, (x, y) in enumerate(loader):
            if i >= config.eval_iters:
                break

            x = x.to(config.device)
            y = y.to(config.device)

            # Forward pass
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        losses[split] = total_loss / total_samples

    model.train()
    return losses


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainingConfig,
    scaler: torch.cuda.amp.GradScaler,
    it: int
) -> float:
    """
    Single training step.

    Args:
        model: The model
        x: Input tokens
        y: Target tokens
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        scaler: GradScaler for mixed precision
        it: Current iteration

    Returns:
        Loss value
    """
    # Get learning rate
    lr = get_lr(it, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Forward pass with automatic mixed precision
    with torch.cuda.amp.autocast(dtype=config.dtype):
        logits, _ = model(x, y)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        loss = F.cross_entropy(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    scaler.scale(loss).backward()

    # Gradient clipping
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    it: int,
    loss: float,
    path: str
):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'iteration': it,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> int:
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    it = checkpoint['iteration']
    print(f"Checkpoint loaded from {path}, resuming from iteration {it}")
    return it


def generate_sample(
    model: nn.Module,
    context: str,
    max_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """Generate text sample"""
    model.eval()

    # Convert context to bytes
    context_bytes = context.encode('utf-8')
    idx = torch.tensor([[b for b in context_bytes]], dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    # Convert back to text
    generated_bytes = generated[0].tolist()
    return bytes(generated_bytes).decode('utf-8', errors='ignore')


def main():
    """Main training loop"""

    # ========== CONFIGURATION ==========
    config = TrainingConfig(
        # Model architecture
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        max_seq_len=512,

        # Training hyperparameters
        batch_size=32,  # Adjust based on GPU memory
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=500,
        eval_iters=100,
        warmup_iters=200,

        # Optimization
        weight_decay=0.1,
        grad_clip=1.0,

        # System
        device="auto",
        compile=True,
        dtype="bfloat16",

        # Checkpointing
        checkpoint_dir="checkpoints",
        save_interval=1000,
    )

    # ========== DEVICE SETUP ==========
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {config.device}")

    # Set data type
    if config.dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif config.dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32
    config.dtype = ptdtype
    torch.set_float32_matmul_precision('high')

    # ========== DATA LOADING ==========
    print("Loading data...")

    # For demo, create dummy data if no file exists
    train_text = load_data(config.dataset_path)
    val_text = load_data(config.val_dataset_path) if config.val_dataset_path else None

    print(f"Training data size: {len(train_text):,} characters")

    # Create datasets
    train_dataset = TextDataset(train_text, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_text:
        val_dataset = TextDataset(val_text, config.max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    print(f"Training batches: {len(train_loader)}")

    # ========== MODEL SETUP ==========
    print("\nCreating model...")
    model_config = BDHConfig(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )

    model = BDHGPUTensor(model_config).to(config.device)
    count_parameters(model)

    # Compile model for speedup (PyTorch 2.0+)
    if config.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled!")

    # ========== OPTIMIZER SETUP ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))

    # ========== RESUME FROM CHECKPOINT ==========
    start_iter = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_iter = load_checkpoint(model, optimizer, config.resume_from)
        start_iter += 1

    # ========== TRAINING LOOP ==========
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    model.train()
    train_loader_iter = iter(train_loader)
    t0 = time.time()

    for it in range(start_iter, config.max_iters):
        # Get batch (wrap around if needed)
        try:
            x, y = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            x, y = next(train_loader_iter)

        x = x.to(config.device)
        y = y.to(config.device)

        # Training step
        loss = train_step(
            model, x, y, optimizer, None, config, scaler, it
        )

        # Log progress
        if it % 100 == 0 or it == 0:
            t1 = time.time()
            tokens_per_sec = config.batch_size * config.max_seq_len / (t1 - t0)
            t0 = t1
            lr = get_lr(it, config)
            print(f"iter {it:5d} | loss {loss:.4f} | lr {lr:.2e} | tokens/sec {tokens_per_sec:.0f}")

        # Evaluation
        if it % config.eval_interval == 0 and it > 0:
            losses = estimate_loss(model, train_loader, val_loader, config)
            print(f"\n--- Iteration {it} ---")
            print(f"Train loss: {losses['train']:.4f}")
            if 'val' in losses:
                print(f"Val loss: {losses['val']:.4f}")

            # Generate sample
            if config.device == "cuda":
                sample = generate_sample(
                    model,
                    context="The meaning of life is",
                    max_tokens=200,
                    temperature=0.8,
                    device=config.device
                )
                print(f"\nSample:\n{sample}\n")

            model.train()

        # Save checkpoint
        if it % config.save_interval == 0 and it > 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{it}.pt")
            save_checkpoint(model, optimizer, config, it, loss, checkpoint_path)

    # ========== FINAL SAVE ==========
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, config, config.max_iters, loss, final_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
