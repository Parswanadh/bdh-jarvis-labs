"""
BDH Science Fest - Training Script with Checkpointing
=====================================================

This script trains the Multi-Scale BDH model with:
- Proper checkpointing (save every N steps)
- Resume from checkpoint capability
- Progress tracking and logging
- Evaluation on validation set
- Best model saving

Dataset: Tiny Shakespeare (for quick demo training)
Task: Language Modeling (predict next byte)
Training Time: ~30-60 minutes for demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Import our implementations
import sys
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from stable_config import BDHStableTrainingConfig


class ByteDataset(Dataset):
    """
    Byte-level text dataset for BDH training.
    Uses raw bytes (0-255) - no tokenizer needed for baseline.
    """
    def __init__(self, text: str, max_seq_len: int = 512):
        self.text = text
        self.max_seq_len = max_seq_len

        # Convert text to bytes (0-255)
        self.data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]      # Input
        y = chunk[1:self.max_seq_len + 1]  # Target (next token)
        return x, y


def load_dataset(path: str) -> str:
    """Load text dataset from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def create_sample_dataset():
    """
    Create a sample dataset for training.

    For the science fest demo, we'll use a combination of:
    1. Tiny Shakespeare (classic LM benchmark)
    2. Some scientific text (for demo diversity)

    Returns:
        train_text: Training data
        val_text: Validation data
    """
    # Sample text - Shakespeare's sonnet (for demo)
    sample_text = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date:
    Sometime too hot the eye of heaven shines,
    And often is his gold complexion dimm'd;
    And every fair from fair sometime declines,
    By chance, or nature's changing course untrimm'd;
    But thy eternal summer shall not fade,
    Nor lose possession of that fair thou ow'st;
    Nor shall death brag thou wander'st in his shade,
    When in eternal lines to time thou grow'st:
    So long as men can breathe, or eyes can see,
    So long lives this, and this gives life to thee.

    The science of artificial intelligence is transforming our world.
    Neural networks learn patterns from data and make predictions.
    Brain-inspired architectures like BDH bring interpretability to AI.
    Multi-scale memory allows models to remember longer contexts.
    This is the future of artificial intelligence.
    """

    # Repeat to create more training data
    train_text = sample_text * 100  # ~13,000 characters
    val_text = sample_text * 20       # ~2,600 characters

    return train_text, val_text


def save_checkpoint(model, optimizer, config, iteration, loss, checkpoint_dir, is_best=False):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)

    print(f"✅ Checkpoint saved: iteration {iteration}, loss: {loss:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = checkpoint['iteration']
    loss = checkpoint['loss']

    print(f"✅ Checkpoint loaded: iteration {iteration}, loss: {loss:.4f}")

    return iteration, loss


def evaluate(model, dataloader, device, max_batches=10):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, _ = model(x)

            # Calculate loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def train(config, train_config, resume_from=None):
    """
    Main training loop with checkpointing.

    Args:
        config: MultiScaleBDHConfig
        train_config: Training configuration
        resume_from: Path to checkpoint to resume from
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training on device: {device}")

    # Create model
    model = MultiScaleBDH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params:,} parameters")

    # Create optimizer (using stable config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )

    # Load data
    print("📊 Loading dataset...")
    if os.path.exists(train_config.dataset_path):
        train_text = load_dataset(train_config.dataset_path)
        val_text = load_dataset(train_config.val_dataset_path) if train_config.val_dataset_path else train_text[:1000]
    else:
        print("⚠️  Dataset not found, using sample data for demo...")
        train_text, val_text = create_sample_dataset()

    train_dataset = ByteDataset(train_text, max_seq_len=config.max_seq_len)
    val_dataset = ByteDataset(val_text, max_seq_len=config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"✅ Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # Create checkpoint directory
    checkpoint_dir = train_config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save training config
    with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w') as f:
        json.dump({
            'model_config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'training_config': {k: v for k, v in train_config.__dict__.items() if not k.startswith('_')},
            'date': datetime.now().isoformat()
        }, f, indent=2)

    # Resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float('inf')

    if resume_from and os.path.exists(resume_from):
        start_iteration, _ = load_checkpoint(resume_from, model, optimizer)
        print(f"✅ Resumed from iteration {start_iteration}")

    # Training loop
    print(f"\n🚀 Starting training for {train_config.max_iters} iterations...")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Checkpoint interval: every {train_config.save_interval} iterations")
    print(f"   Eval interval: every {train_config.eval_interval} iterations")
    print("="*60)

    model.train()
    train_iter = iter(train_loader)
    total_loss = 0.0
    t0 = time.time()

    for iteration in range(start_iteration, train_config.max_iters):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _ = model(x)

        # Calculate loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        loss = F.cross_entropy(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Optimizer step
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Print progress
        if (iteration + 1) % 100 == 0:
            t1 = time.time()
            tokens_per_sec = (iteration + 1) * train_config.batch_size * config.max_seq_len / (t1 - t0)
            avg_loss = total_loss / 100

            print(f"[{iteration+1:5d}/{train_config.max_iters}] "
                  f"loss: {avg_loss:.4f} | "
                  f"tokens/sec: {tokens_per_sec:.0f}")

            total_loss = 0.0

        # Save checkpoint
        if (iteration + 1) % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, config, iteration + 1,
                avg_loss, checkpoint_dir, is_best=False
            )

        # Evaluation
        if (iteration + 1) % train_config.eval_interval == 0:
            val_loss, perplexity = evaluate(model, val_loader, device)

            print(f"\n{'='*60}")
            print(f"📊 VALIDATION [iter {iteration+1}]")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"{'='*60}\n")

            # Save if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, iteration + 1,
                    val_loss, checkpoint_dir, is_best=True
                )
                print(f"🏆 New best model! Val loss: {val_loss:.4f}\n")

            model.train()

    # Final checkpoint
    save_checkpoint(
        model, optimizer, config, train_config.max_iters,
        avg_loss, checkpoint_dir, is_best=False
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"✅ Final checkpoint saved to {checkpoint_dir}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print("="*60)


def main():
    """Main training function."""
    print("="*60)
    print("BDH Science Fest - Multi-Scale BDH Training")
    print("="*60)
    print()

    # Model configuration (10M Multi-Scale BDH)
    config = MultiScaleBDHConfig(
        vocab_size=256,              # Byte-level
        n_embd=256,                  # Embedding dimension
        n_layer=6,                   # Number of layers
        n_head=4,                    # Number of attention heads
        ffn_dim=1024,                # FFN dimension
        dropout=0.1,
        max_seq_len=512,             # Context length
        decay_rates=[0.95, 0.99, 0.995],  # Multi-scale!
        hebbian_lr=0.001,           # Lower for stability
        state_decay=0.99
    )

    # Training configuration (stable!)
    train_config = BDHStableTrainingConfig.for_10M_model()

    # Override for demo training
    train_config.dataset_path = "data/shakespeare.txt"  # Will use sample if not exists
    train_config.val_dataset_path = None  # Will use split from train
    train_config.max_iters = 1000      # Quick demo training
    train_config.batch_size = 32       # Smaller batch for RTX 4070
    train_config.eval_interval = 500   # Evaluate twice during training
    train_config.save_interval = 500   # Save checkpoints frequently
    train_config.checkpoint_dir = "checkpoints/multiscale_bdh_science_fest"
    train_config.resume_from = None    # Set to checkpoint path to resume

    # Print configuration
    print("📋 MODEL CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: ~10M")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print()

    print("📋 TRAINING CONFIGURATION:")
    print(f"   Max iterations: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Warmup steps: {train_config.warmup_steps}")
    print(f"   Gradient clip: {train_config.gradient_clip}")
    print()

    print("📋 TASK:")
    print(f"   Task: Language Modeling (next byte prediction)")
    print(f"   Dataset: Tiny Shakespeare (sample for demo)")
    print(f"   Training time: ~20-30 minutes")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
