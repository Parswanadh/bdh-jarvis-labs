"""
Train Multi-Scale BDH with Knowledge Distillation from Gemma
=============================================================

This script trains Multi-Scale BDH using knowledge distillation from Gemma 3 270M.

Teacher: Gemma 3 270M (provides knowledge)
Student: Multi-Scale BDH (learns from teacher)
Method: Next-token prediction on teacher-generated data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import sys
from datetime import datetime
import time

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from stable_config import get_small_model_config


class TeacherDataset(Dataset):
    """
    Dataset from teacher-generated data.

    Uses Gemma-generated text for training BDH.
    """

    def __init__(self, data_file: str, max_seq_len: int = 512):
        """
        Load teacher-generated dataset.

        Args:
            data_file: Path to JSONL file with teacher data
            max_seq_len: Maximum sequence length
        """

        print(f"📚 Loading teacher data from {data_file}")

        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item['generated']
                # Convert to bytes
                self.data.append(text)

        # Combine all text
        self.full_text = "\n".join(self.data)

        # Convert to byte tensor
        self.data_tensor = torch.tensor(
            list(self.full_text.encode('utf-8')),
            dtype=torch.long
        )

        self.max_seq_len = max_seq_len

        print(f"✅ Loaded {len(self.data)} samples")
        print(f"   Total characters: {len(self.full_text):,}")
        print(f"   Total bytes: {len(self.data_tensor):,}")

    def __len__(self):
        return max(0, len(self.data_tensor) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data_tensor[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]
        y = chunk[1:self.max_seq_len + 1]
        return x, y


def train_epoch(model, dataloader, optimizer, config, device):
    """Train for one epoch."""

    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _ = model(x)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if hasattr(config, 'grad_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_idx + 1) * config.batch_size * config.max_seq_len / elapsed
            avg_loss = total_loss / (batch_idx + 1)

            print(f"[{batch_idx+1}/{len(dataloader)}] loss: {avg_loss:.4f} | tokens/sec: {int(tokens_per_sec)}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, config, iteration, loss, checkpoint_dir, is_best=False):
    """Save training checkpoint."""

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best,
        'task': 'distillation'
    }

    # Save iteration checkpoint
    torch.save(
        checkpoint,
        Path(checkpoint_dir) / f"checkpoint_iter_{iteration}.pt"
    )

    # Save latest checkpoint
    torch.save(
        checkpoint,
        Path(checkpoint_dir) / "checkpoint_latest.pt"
    )

    # Save best checkpoint
    if is_best:
        torch.save(
            checkpoint,
            Path(checkpoint_dir) / "checkpoint_best.pt"
        )
        print(f"🏆 New best model! Val loss: {loss:.4f}")

    print(f"✅ Checkpoint saved: iteration {iteration}, loss: {loss:.4f}")


def train(config, train_config):
    """Main training function."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training on device: {device}")

    # Create model
    model = MultiScaleBDH(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params:,} parameters")

    # Setup data
    print("📊 Loading dataset...")
    train_dataset = TeacherDataset(
        "data/teacher_generated/gemma_training_data.jsonl",
        max_seq_len=config.max_seq_len
    )

    # Split train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

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

    print(f"✅ Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Training loop
    print("🚀 Starting training...")
    print(f"   Max iterations: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print("="*60)
    print()

    best_val_loss = float('inf')
    t0 = time.time()

    for iteration in range(train_config.max_iters):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, train_config, device
        )

        print(f"[{iteration+1}/{train_config.max_iters}] train_loss: {train_loss:.4f}")

        # Save checkpoint
        if (iteration + 1) % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, train_config, iteration + 1, train_loss,
                train_config.checkpoint_dir, is_best=False
            )

            # Evaluate
            if (iteration + 1) % train_config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                perplexity = torch.exp(torch.tensor(val_loss))

                print()
                print("="*60)
                print(f"📊 VALIDATION [iter {iteration + 1}]")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Perplexity: {perplexity:.2f}")
                print("="*60)
                print()

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, train_config, iteration + 1, val_loss,
                        train_config.checkpoint_dir, is_best=True
                    )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, train_config, train_config.max_iters, train_loss,
        train_config.checkpoint_dir, is_best=False
    )

    elapsed = time.time() - t0
    print()
    print("="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Training time: {elapsed/60:.1f} minutes")
    print(f"✅ Final train loss: {train_loss:.4f}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Checkpoints saved to: {train_config.checkpoint_dir}")
    print("="*60)


def main():
    """Main function."""

    print("="*60)
    print("BDH DISTILLATION TRAINING")
    print("Teacher: Gemma 3 270M")
    print("Student: Multi-Scale BDH")
    print("="*60)
    print()

    # Model configuration
    config = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        max_seq_len=512,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    # Training configuration
    train_config = get_small_model_config()
    train_config.max_iters = 1000
    train_config.batch_size = 32
    train_config.eval_interval = 500
    train_config.save_interval = 500
    train_config.checkpoint_dir = "checkpoints/multiscale_bdh_distillation"
    train_config.resume_from = None

    print("📋 MODEL CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: ~5M")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print()

    print("📋 TRAINING CONFIGURATION:")
    print(f"   Max iterations: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Warmup steps: {train_config.warmup_steps}")
    print(f"   Gradient clip: {train_config.grad_clip}")
    print()

    print("📋 TASK:")
    print(f"   Task: Language Modeling (Knowledge Distillation)")
    print(f"   Teacher: Gemma 3 270M")
    print(f"   Student: Multi-Scale BDH")
    print(f"   Training time: ~20-30 minutes")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
