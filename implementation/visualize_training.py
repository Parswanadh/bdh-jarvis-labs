"""
BDH Training Visualization Script
==================================

Visualize training results comparing unstable vs stable configurations.

Usage:
    python visualize_training.py --results-dir benchmarking/results/comparison_YYYYMMDD_HHMMSS

Author: BDH Training Stabilization Specialist
Date: February 25, 2026
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(results_dir, mode):
    """Load training metrics from JSON file."""
    path = os.path.join(results_dir, f'{mode}_metrics.json')
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison(results_dir, save_path=None):
    """
    Create comparison plots for unstable vs stable training.

    Args:
        results_dir: Directory containing metric JSON files
        save_path: Optional path to save the figure
    """
    # Load metrics
    unstable_metrics = load_metrics(results_dir, 'unstable')
    stable_metrics = load_metrics(results_dir, 'stable')

    if unstable_metrics is None and stable_metrics is None:
        print(f"No metrics found in {results_dir}")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BDH Training Configuration Comparison: Unstable vs Stable',
                 fontsize=16, fontweight='bold')

    # Plot 1: Loss curves
    ax = axes[0, 0]
    if unstable_metrics and unstable_metrics['losses']:
        iters_unstable = np.linspace(0, len(unstable_metrics['losses']) * 10,
                                     len(unstable_metrics['losses']))
        ax.plot(iters_unstable, unstable_metrics['losses'],
               label='Unstable (Transformer defaults)', color='red', alpha=0.7)
    if stable_metrics and stable_metrics['losses']:
        iters_stable = np.linspace(0, len(stable_metrics['losses']) * 10,
                                   len(stable_metrics['losses']))
        ax.plot(iters_stable, stable_metrics['losses'],
               label='Stable (BDH optimized)', color='blue', alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Learning rate schedule
    ax = axes[0, 1]
    if unstable_metrics and unstable_metrics['lrs']:
        iters_unstable = np.linspace(0, len(unstable_metrics['lrs']) * 10,
                                     len(unstable_metrics['lrs']))
        ax.plot(iters_unstable, unstable_metrics['lrs'],
               label='Unstable', color='red', alpha=0.7)
    if stable_metrics and stable_metrics['lrs']:
        iters_stable = np.linspace(0, len(stable_metrics['lrs']) * 10,
                                   len(stable_metrics['lrs']))
        ax.plot(iters_stable, stable_metrics['lrs'],
               label='Stable', color='blue', alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Gradient norms
    ax = axes[1, 0]
    if unstable_metrics and unstable_metrics['grad_norms']:
        iters_unstable = np.linspace(0, len(unstable_metrics['grad_norms']) * 10,
                                     len(unstable_metrics['grad_norms']))
        ax.plot(iters_unstable, unstable_metrics['grad_norms'],
               label='Unstable', color='red', alpha=0.7)
    if stable_metrics and stable_metrics['grad_norms']:
        iters_stable = np.linspace(0, len(stable_metrics['grad_norms']) * 10,
                                   len(stable_metrics['grad_norms']))
        ax.plot(iters_stable, stable_metrics['grad_norms'],
               label='Stable', color='blue', alpha=0.7)

    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
              label='Target (< 1.0)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms (Lower is Better)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Training speed (time per iteration)
    ax = axes[1, 1]
    if unstable_metrics and unstable_metrics['times']:
        times_unstable = np.diff([0] + unstable_metrics['times'])
        ax.plot(np.arange(len(times_unstable)) * 10, times_unstable,
               label='Unstable', color='red', alpha=0.7)
    if stable_metrics and stable_metrics['times']:
        times_stable = np.diff([0] + stable_metrics['times'])
        ax.plot(np.arange(len(times_stable)) * 10, times_stable,
               label='Stable', color='blue', alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time per 10 iters (s)')
    ax.set_title('Training Speed', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING COMPARISON SUMMARY")
    print("="*60)

    if unstable_metrics:
        print("\n🔴 UNSTABLE Configuration (Transformer defaults):")
        print(f"   Initializer range: 0.02")
        print(f"   Warmup steps: 500")
        print(f"   Learning rate: 6e-4")
        print(f"   Final loss: {unstable_metrics.get('final_loss', 'N/A'):.4f}")
        print(f"   Min loss: {unstable_metrics.get('min_loss', 'N/A'):.4f}")
        print(f"   Max grad norm: {unstable_metrics.get('max_grad_norm', 'N/A'):.4f}")

    if stable_metrics:
        print("\n🔵 STABLE Configuration (BDH optimized):")
        print(f"   Initializer range: 0.006")
        print(f"   Warmup steps: 5000")
        print(f"   Learning rate: 3e-4")
        print(f"   Final loss: {stable_metrics.get('final_loss', 'N/A'):.4f}")
        print(f"   Min loss: {stable_metrics.get('min_loss', 'N/A'):.4f}")
        print(f"   Max grad norm: {stable_metrics.get('max_grad_norm', 'N/A'):.4f}")

    # Calculate improvement
    if unstable_metrics and stable_metrics:
        if unstable_metrics.get('final_loss') and stable_metrics.get('final_loss'):
            improvement = (
                (unstable_metrics['final_loss'] - stable_metrics['final_loss']) /
                unstable_metrics['final_loss'] * 100
            )
            print(f"\n📊 IMPROVEMENT: {improvement:.1f}% lower loss with stable config")

    print("="*60 + "\n")


def plot_convergence_comparison(results_dirs, labels, save_path=None):
    """
    Plot convergence comparison across multiple runs.

    Args:
        results_dirs: List of directory paths
        labels: List of labels for each directory
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('BDH Training Convergence Comparison',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dirs)))

    # Plot loss curves
    ax = axes[0]
    for results_dir, label, color in zip(results_dirs, labels, colors):
        metrics = load_metrics(results_dir, 'stable') or load_metrics(results_dir, 'unstable')
        if metrics and metrics['losses']:
            iters = np.linspace(0, len(metrics['losses']) * 10, len(metrics['losses']))
            ax.plot(iters, metrics['losses'], label=label, color=color, alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot gradient norms
    ax = axes[1]
    for results_dir, label, color in zip(results_dirs, labels, colors):
        metrics = load_metrics(results_dir, 'stable') or load_metrics(results_dir, 'unstable')
        if metrics and metrics['grad_norms']:
            iters = np.linspace(0, len(metrics['grad_norms']) * 10, len(metrics['grad_norms']))
            ax.plot(iters, metrics['grad_norms'], label=label, color=color, alpha=0.7)

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Stability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize BDH training results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing training result JSON files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the figure (optional)')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='Multiple result directories to compare')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple runs
        labels = [os.path.basename(d) for d in args.compare]
        plot_convergence_comparison(args.compare, labels, args.output)
    else:
        # Single comparison (unstable vs stable)
        plot_comparison(args.results_dir, args.output)


if __name__ == '__main__':
    main()
