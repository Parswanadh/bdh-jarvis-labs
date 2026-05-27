"""
BDH Training Configuration Comparison Script
=============================================

This script compares unstable (Transformer-default) vs stable (BDH-optimized)
training configurations on a small dataset.

Use this to validate that the stable configuration actually works better!

Usage:
    python test_stable_config.py --mode unstable
    python test_stable_config.py --mode stable
    python test_stable_config.py --mode both

Author: BDH Training Stabilization Specialist
Date: February 25, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import os
from datetime import datetime
import time

# Import BDH model and configurations
import sys
sys.path.append('.')

try:
    from bdh_gpu_10m import BDHConfig, BDHGPUTensor
except ImportError:
    print("Warning: bdh_gpu_10m.py not found. Using mock for testing.")
    # Mock for testing
    class BDHConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BDHGPUTensor(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.n_embd,
                    nhead=config.n_head,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    batch_first=True
                ) for _ in range(config.n_layer)
            ])
            self.norm_f = nn.LayerNorm(config.n_embd)
            self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.output_projection.weight = self.token_embedding.weight

        def forward(self, idx, state=None, return_state=False):
            x = self.token_embedding(idx)
            for layer in self.layers:
                x = layer(x)
            x = self.norm_f(x)
            logits = self.output_projection(x)
            if return_state:
                return logits, state
            return logits, state

from implementation.stable_config import (
    BDHStableTrainingConfig,
    MimeticInitializer,
    get_learning_rate_schedule,
    get_comparison_table
)


class SimpleTextDataset(Dataset):
    """Simple synthetic text dataset for testing."""

    def __init__(self, seq_len=512, num_samples=1000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Generate synthetic byte-level data
        self.data = torch.randint(0, 256, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx, :self.seq_len]
        y = self.data[idx, 1:self.seq_len + 1]
        return x, y


class TrainingMetrics:
    """Track training metrics for comparison."""

    def __init__(self, mode):
        self.mode = mode
        self.losses = []
        self.lrs = []
        self.grad_norms = []
        self.times = []
        self.start_time = time.time()

    def log(self, loss, lr, grad_norm):
        self.losses.append(loss)
        self.lrs.append(lr)
        self.grad_norms.append(grad_norm)
        self.times.append(time.time() - self.start_time)

    def save(self, path):
        """Save metrics to JSON file."""
        data = {
            'mode': self.mode,
            'losses': self.losses,
            'lrs': self.lrs,
            'grad_norms': self.grad_norms,
            'times': self.times,
            'total_time': self.times[-1] if self.times else 0,
            'final_loss': self.losses[-1] if self.losses else None,
            'min_loss': min(self.losses) if self.losses else None,
            'max_grad_norm': max(self.grad_norms) if self.grad_norms else None,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results saved to {path}")
        print(f"{'='*60}")
        print(f"Total time: {data['total_time']:.2f}s")
        print(f"Final loss: {data['final_loss']:.4f}")
        print(f"Min loss: {data['min_loss']:.4f}")
        print(f"Max grad norm: {data['max_grad_norm']:.4f}")


def get_config(mode: str) -> BDHStableTrainingConfig:
    """
    Get training configuration for specified mode.

    Args:
        mode: 'unstable' or 'stable'

    Returns:
        Training configuration
    """
    if mode == 'unstable':
        # Transformer-default configuration (UNSTABLE for BDH!)
        return BDHStableTrainingConfig(
            # Architecture
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024,
            dropout=0.1,
            max_seq_len=512,

            # UNSTABLE: Transformer defaults
            initializer_range=0.02,   # ← TOO LARGE for BDH!
            warmup_steps=500,          # ← TOO SHORT for BDH!
            learning_rate=6e-4,        # ← TOO HIGH for BDH!
            grad_clip=1.0,

            # Training
            batch_size=16,
            gradient_accumulation_steps=1,
            max_iters=2000,  # Shorter run for testing
            eval_interval=200,
            eval_iters=50,
        )
    elif mode == 'stable':
        # BDH-optimized configuration (STABLE!)
        return BDHStableTrainingConfig(
            # Architecture
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024,
            dropout=0.1,
            max_seq_len=512,

            # STABLE: BDH-optimized
            initializer_range=0.006,  # ← 3x smaller
            warmup_steps=5000,         # ← 10x longer (will be truncated by max_iters)
            learning_rate=3e-4,        # ← 2x lower
            grad_clip=1.0,

            # Training
            batch_size=16,
            gradient_accumulation_steps=1,
            max_iters=2000,  # Shorter run for testing
            eval_interval=200,
            eval_iters=50,

            # Stability options
            use_mimetic_init=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train(model, config, mode):
    """
    Train BDH model with given configuration.

    Args:
        model: BDH model
        config: Training configuration
        mode: 'unstable' or 'stable'

    Returns:
        TrainingMetrics object
    """
    print(f"\n{'='*60}")
    print(f"Training with {mode.upper()} configuration")
    print(f"{'='*60}")
    print(f"Initializer range: {config.initializer_range}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Gradient clip: {config.grad_clip}")
    print(f"Max iters: {config.max_iters}")
    print(f"{'='*60}\n")

    # Create dataset
    dataset = SimpleTextDataset(seq_len=512, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # Get learning rate schedule
    lr_schedule = get_learning_rate_schedule(config)

    # Metrics
    metrics = TrainingMetrics(mode)

    # Training loop
    model.train()
    dataloader_iter = iter(dataloader)
    nan_count = 0
    spike_count = 0

    for it in range(config.max_iters):
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x, y = next(dataloader_iter)

        # Get learning rate
        lr = lr_schedule(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        # Check for NaN
        if torch.isnan(loss):
            nan_count += 1
            print(f"⚠️  NaN loss at iteration {it}!")
            if nan_count >= 3:
                print(f"❌ Too many NaN losses. Stopping training.")
                break
            continue

        # Check for loss spike
        if metrics.losses and loss.item() > 2.0 * metrics.losses[-1]:
            spike_count += 1
            print(f"⚠️  Loss spike at iteration {it}: {loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = 0.0

        # Optimizer step
        optimizer.step()

        # Log metrics
        if it % 10 == 0:
            metrics.log(loss.item(), lr, grad_norm)
            print(f"Iter {it:4d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Grad: {grad_norm:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Summary ({mode})")
    print(f"{'='*60}")
    print(f"NaN losses: {nan_count}")
    print(f"Loss spikes: {spike_count}")
    print(f"Final loss: {metrics.losses[-1]:.4f}" if metrics.losses else "No valid losses")
    print(f"Min loss: {min(metrics.losses):.4f}" if metrics.losses else "N/A")
    print(f"{'='*60}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare BDH training configurations')
    parser.add_argument('--mode', type=str, choices=['unstable', 'stable', 'both'],
                        default='both', help='Configuration mode to test')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output-dir', type=str, default='benchmarking/results',
                        help='Output directory for results')
    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'comparison_{timestamp}')

    modes = ['unstable', 'stable'] if args.mode == 'both' else [args.mode]

    results = {}

    for mode in modes:
        # Get configuration
        config = get_config(mode)

        # Create model
        bdh_config = BDHConfig(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

        model = BDHGPUTensor(bdh_config).to(device)

        # Apply mimetic initialization if configured
        if config.use_mimetic_init:
            initializer = MimeticInitializer(config)
            initializer.apply_to_bdh_model(model)
            print("Applied mimetic initialization")

        # Train
        metrics = train(model, config, mode)

        # Save results
        result_path = os.path.join(output_dir, f'{mode}_metrics.json')
        metrics.save(result_path)
        results[mode] = metrics

        # Clean up
        del model

    # Create comparison summary
    if 'unstable' in results and 'stable' in results:
        comparison_path = os.path.join(output_dir, 'comparison_summary.json')
        comparison = {
            'unstable': {
                'final_loss': results['unstable'].losses[-1] if results['unstable'].losses else None,
                'min_loss': min(results['unstable'].losses) if results['unstable'].losses else None,
                'max_grad_norm': max(results['unstable'].grad_norms) if results['unstable'].grad_norms else None,
            },
            'stable': {
                'final_loss': results['stable'].losses[-1] if results['stable'].losses else None,
                'min_loss': min(results['stable'].losses) if results['stable'].losses else None,
                'max_grad_norm': max(results['stable'].grad_norms) if results['stable'].grad_norms else None,
            },
            'improvement': {
                'final_loss_pct': (
                    (results['unstable'].losses[-1] - results['stable'].losses[-1]) /
                    results['unstable'].losses[-1] * 100
                    if results['unstable'].losses and results['stable'].losses else None
                ),
            }
        }

        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Unstable final loss: {comparison['unstable']['final_loss']:.4f}")
        print(f"Stable final loss:   {comparison['stable']['final_loss']:.4f}")
        if comparison['improvement']['final_loss_pct']:
            print(f"Improvement:          {comparison['improvement']['final_loss_pct']:.1f}%")
        print(f"{'='*60}\n")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
