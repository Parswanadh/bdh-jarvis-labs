"""
BDH-RD Training Script
======================

Trains BDH with Recurrent Depth (BDH-RD) - a "Mythos-style" loop
built natively into the BDH architecture.

Usage:
    python train_bdh_recurrent.py --config basic
    python train_bdh_recurrent.py --config large --batch_size 32
    python train_bdh_recurrent.py --resume checkpoints/bdh_rd.pt

Author: Expert ML Engineer
Date: 2026-04-23
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import json
import time
import gc
import os
from contextlib import nullcontext

from bdh_recurrent import BDHRecurrent, BDHRecurrentConfig, BDHRDLoss, MultiScaleBDHConfig
from multiscale_bdh import count_parameters


# ============================================================================
# Configuration
# ============================================================================

@torch.no_grad()
def estimate_loss(
    model: BDHRecurrent,
    dataloader: DataLoader,
    eval_iters: int = 100,
    device: torch.device = None
):
    """Estimate loss on validation set."""
    model.eval()
    total_loss = 0
    loop_depths = []

    for i, batch in enumerate(dataloader):
        if i >= eval_iters:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)

        logits, _, metrics = model(x, return_loop_metrics=True)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

        if metrics:
            loop_depths.append(metrics.get('avg_loop_depth', 1))

    model.train()
    avg_loss = total_loss / min(len(dataloader), eval_iters)
    avg_loops = sum(loop_depths) / len(loop_depths) if loop_depths else 1

    return {'loss': avg_loss, 'avg_loops': avg_loops}


# ============================================================================
# Dataset
# ============================================================================

class TinyStoriesDataset(Dataset):
    """Tiny Stories dataset for training."""
    def __init__(self, filepath: str, max_seq_len: int = 192):
        self.max_seq_len = max_seq_len
        self.tokens = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    text = line.strip()
                    # Convert to byte tokens
                    token_ids = list(text.encode('utf-8')[:max_seq_len])
                    if len(token_ids) < max_seq_len:
                        token_ids += [32] * (max_seq_len - len(token_ids))
                    self.tokens.append(token_ids)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Training Functions
# ============================================================================

def save_checkpoint(
    model: BDHRecurrent,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    path: str,
    epoch: int,
    step: int,
    best_loss: float
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': {
            'base_config': {
                'vocab_size': model.bdh.config.vocab_size,
                'n_embd': model.bdh.config.n_embd,
                'n_layer': model.bdh.config.n_layer,
                'n_head': model.bdh.config.n_head,
                'ffn_dim': model.bdh.config.ffn_dim,
                'dropout': model.bdh.config.dropout,
                'max_seq_len': model.bdh.config.max_seq_len,
            },
            'max_loops': model.config.max_loops,
            'use_learnable_decay': model.config.use_learnable_decay,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def train(
    model: BDHRecurrent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: dict,
    checkpoint_path: Path = None
):
    """Main training loop."""
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')

    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_step = checkpoint['step']
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch}, step {start_step}, best_loss {best_loss:.4f}")

    loss_fn = BDHRDLoss(model.config)

    # Training loop
    global_step = start_step
    log_interval = config.get('log_interval', 10)
    eval_interval = config.get('eval_interval', 100)
    save_interval = config.get('save_interval', 500)

    for epoch in range(start_epoch, config.get('epochs', 10)):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            if global_step <= start_step:
                continue

            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, states, metrics = model(x, return_states=True, return_loop_metrics=True)

            # Compute loss
            loss, losses = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                metrics,
                states
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.get('max_grad_norm', None):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['max_grad_norm']
                )

            optimizer.step()
            scheduler.step()

            # Log metrics
            epoch_loss += losses['total']
            epoch_steps += 1

            if global_step % log_interval == 0:
                print(f"Step {global_step}: "
                      f"loss={losses['total']:.4f} "
                      f"(ce={losses['ce']:.4f}, "
                      f"loop={losses['loop_penalty']:.4f}, "
                      f"stability={losses['stability']:.4f}) "
                      f"loops={metrics.get('actual_loops', 1)}")

            # Evaluation
            if global_step % eval_interval == 0:
                val_metrics = estimate_loss(model, val_loader, device=device)
                print(f"[EVAL Step {global_step}] "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"avg_loops={val_metrics['avg_loops']:.2f}")

                if val_metrics['loss'] < best_loss:
                    best_loss = val_metrics['loss']
                    save_checkpoint(
                        model, optimizer, scheduler,
                        str(checkpoint_path.parent / "best_model.pt"),
                        epoch, global_step, best_loss
                    )

            # Save checkpoint
            if global_step % save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    str(checkpoint_path),
                    epoch, global_step, best_loss
                )

        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='basic',
                        choices=['basic', 'large', 'xl'],
                        help='Configuration preset')
    parser.add_argument('--data', type=str, default='data/tinystories.txt',
                        help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=192,
                        help='Sequence length')
    parser.add_argument('--max_loops', type=int, default=4,
                        help='Maximum recurrent loops')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Configuration
    configs = {
        'basic': {
            'n_embd': 256,
            'n_layer': 6,
            'n_head': 4,
            'ffn_dim': 1024,
            'dropout': 0.1,
        },
        'large': {
            'n_embd': 512,
            'n_layer': 8,
            'n_head': 8,
            'ffn_dim': 2048,
            'dropout': 0.1,
        },
        'xl': {
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'ffn_dim': 3072,
            'dropout': 0.1,
        }
    }

    base_config = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=configs[args.config]['n_embd'],
        n_layer=configs[args.config]['n_layer'],
        n_head=configs[args.config]['n_head'],
        ffn_dim=configs[args.config]['ffn_dim'],
        dropout=configs[args.config]['dropout'],
        max_seq_len=args.seq_len,
    )

    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'log_interval': 10,
        'eval_interval': 100,
        'save_interval': 500,
        'max_grad_norm': 1.0,
    }

    # Create BDH-RD model
    rd_config = BDHRecurrentConfig(
        base_config=base_config,
        max_loops=args.max_loops,
        use_learnable_decay=True,
        use_per_token_exit=True,
    )

    model = BDHRecurrent(rd_config).to(device)

    # Count parameters
    count_parameters(model)

    # Data
    train_dataset = TinyStoriesDataset(args.data, max_seq_len=args.seq_len)
    # Split for validation
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps // 10, T_mult=1
    )

    # Resume if checkpoint exists
    checkpoint_path = None
    if args.resume:
        checkpoint_path = Path(args.resume)
    else:
        checkpoint_path = Path("checkpoints/bdh_recurrent.pt")

    # Create checkpoint directory
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Train
    print("\n" + "="*60)
    print("BDH-RD Training Starting")
    print("="*60)
    print(f"Model: BDH with {rd_config.max_loops} max loops")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config,
        checkpoint_path=checkpoint_path
    )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler,
        str(checkpoint_path.parent / "final_model.pt"),
        args.epochs, total_steps, best_loss
    )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
