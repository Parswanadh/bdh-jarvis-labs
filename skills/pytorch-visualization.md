# PyTorch Visualization Skill

## Purpose
Create compelling, publication-quality visualizations for BDH benchmark results and science fair presentations.

## When to Use This Skill
- Creating graphs and charts from benchmark data
- Visualizing model behavior (state matrices, attention patterns)
- Designing before/after comparison graphics
- Preparing figures for posters and presentations

## Key Visualization Types

### 1. Memory Retention Curves (CRITICAL)
Shows how state matrix activation decays over sequence length.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_retention_curves(results_dict, save_path='visualization/retention_curves.png'):
    """
    results_dict = {
        'baseline': {100: 0.36, 500: 0.0065, 1000: 0.00004, 2000: 0.0},
        'multiscale': {100: 0.45, 500: 0.15, 1000: 0.05, 2000: 0.015}
    }
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, results in results_dict.items():
        lengths = sorted(results.keys())
        retentions = [results[l] for l in lengths]

        # Plot points
        ax.plot(lengths, retentions, 'o-', label=model_name, linewidth=2, markersize=8)

    # Log scale for y-axis (retention is exponential decay)
    ax.set_yscale('log')

    # Styling
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('State Retention (%)', fontsize=12, fontweight='bold')
    ax.set_title('BDH Memory Retention: Baseline vs Multi-Scale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Highlight key improvements
    ax.axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Baseline limit (~500)')
    ax.axvline(x=2000, color='green', linestyle='--', alpha=0.5, label='Multi-scale limit (~2000)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

**Expected Output:**
- Blue line (baseline) drops to near-zero at 500 tokens
- Orange line (multiscale) maintains signal to 2000 tokens
- Clear visual of 4× improvement

### 2. Training Speed Comparison
Shows tokens/second for byte-level vs BBPE.

```python
def plot_training_speed(speed_results, save_path='visualization/training_comparison.png'):
    """
    speed_results = {
        'Byte-level': 10000,
        'BBPE (8K)': 27500
    }
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = list(speed_results.keys())
    speeds = list(speed_results.values())

    bars = ax.bar(models, speeds, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Styling
    ax.set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
    ax.set_title('BDH Training Throughput: Byte-Level vs BBPE', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speeds) * 1.2)

    # Add speedup annotation
    speedup = speeds[1] / speeds[0]
    ax.annotate(f'{speedup:.1f}× speedup',
                xy=(0.5, speeds[1]), xytext=(0.5, speeds[1] * 1.1),
                ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 3. Token Reduction Scatter Plot
Shows sequence length reduction for various texts.

```python
def plot_token_reduction(reduction_data, save_path='visualization/token_reduction.png'):
    """
    reduction_data = [
        {'text': 'short sentence', 'byte': 19, 'bbpe': 5},
        {'text': 'medium paragraph...', 'byte': 150, 'bbpe': 42},
        ...
    ]
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    byte_lengths = [d['byte'] for d in reduction_data]
    bbpe_lengths = [d['bbpe'] for d in reduction_data]

    # Scatter plot
    ax.scatter(byte_lengths, bbpe_lengths, alpha=0.6, s=50, edgecolors='black')

    # Diagonal line (no reduction)
    max_len = max(max(byte_lengths), max(bbpe_lengths))
    ax.plot([0, max_len], [0, max_len], 'r--', label='No reduction', linewidth=2)

    # Styling
    ax.set_xlabel('Byte-level Token Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('BBPE Token Count', fontsize=12, fontweight='bold')
    ax.set_title('Token Reduction: Byte-Level vs BBPE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add reduction ratio annotation
    avg_reduction = np.mean(byte_lengths) / np.mean(bbpe_lengths)
    ax.text(0.05, 0.95, f'Average Reduction: {avg_reduction:.1f}×',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 4. State Matrix Heatmap
Visualize what the BDH model "remembers".

```python
import seaborn as sns

def plot_state_matrix(state_matrix, title='BDH Synaptic State Matrix', save_path='visualization/state_matrix.png'):
    """
    state_matrix: torch.Tensor of shape [n_embd, n_embd]
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to numpy
    state_np = state_matrix.detach().cpu().numpy()

    # Plot heatmap
    im = ax.imshow(state_np, cmap='viridis', aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Synaptic Strength', rotation=270, labelpad=15, fontsize=11)

    # Styling
    ax.set_xlabel('Presynaptic Neuron Index', fontsize=12)
    ax.set_ylabel('Postsynaptic Neuron Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 5. Training Loss Curves
Compare convergence stability.

```python
def plot_training_curves(loss_histories, save_path='visualization/training_curves.png'):
    """
    loss_histories = {
        'Baseline (unstable)': [4.1, 3.8, 3.5, 3.0, 2.8, 2.5, ...],
        'Stabilized config': [4.0, 3.7, 3.4, 3.1, 2.9, 2.7, ...]
    }
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, losses in loss_histories.items():
        ax.plot(losses, label=model_name, linewidth=2)

    # Styling
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('BDH Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Smooth curves (moving average)
    for model_name, losses in loss_histories.items():
        smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
        ax.plot(smoothed, '--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

## Visualization Best Practices

### Color Schemes
- **Sequential data** (curves): Use distinct colors (blue, orange, green)
- **Categorical data**: Use colorblind-friendly palette
- **Heatmaps**: Use perceptually uniform colormaps (viridis, plasma)

### Figure Quality
```python
# High DPI for publications/printing
plt.savefig('figure.png', dpi=300, bbox_inches='tight')

# Standard DPI for screen
plt.savefig('figure.png', dpi=150, bbox_inches='tight')

# PDF for editing (Illustrator, Inkscape)
plt.savefig('figure.pdf', bbox_inches='tight')
```

### Typography
```python
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})
```

### Layout
```python
# Use tight_layout to prevent label clipping
plt.tight_layout()

# Or specify explicit margins
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
```

## Multi-Panel Figures

For posters, create multi-panel figures:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Retention curves
plot_retention_curves(results, ax=axes[0, 0])

# Panel 2: Training speed
plot_training_speed(speeds, ax=axes[0, 1])

# Panel 3: Token reduction
plot_token_reduction(reductions, ax=axes[1, 0])

# Panel 4: State matrix
plot_state_matrix(state, ax=axes[1, 1])

# Add panel labels
for idx, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.05, string.ascii_uppercase[idx],
            transform=ax.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization/combined_figure.png', dpi=300)
```

## Animation (Optional - for Advanced Demos)

```python
from matplotlib.animation import FuncAnimation

def animate_state_evolution(state_evolution):
    """
    state_evolution: List of state matrices over time
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        im = ax.imshow(state_evolution[frame], cmap='viridis')
        ax.set_title(f'State Matrix at t={frame}')
        return [im]

    anim = FuncAnimation(fig, update, frames=len(state_evolution),
                        interval=200, blit=True)

    anim.save('visualization/state_evolution.gif', writer='pillow', fps=5)
```

## Common Issues & Solutions

**Issue:** Figures look blurry in presentation
**Solution:** Increase DPI to 300, use vector formats (PDF, SVG)

**Issue:** Labels overlap
**Solution:** Use `plt.tight_layout()` or rotate labels: `plt.xticks(rotation=45)`

**Issue:** Colors not distinguishable
**Solution:** Use colorblind-friendly palette or add patterns

**Issue:** Heatmap has poor contrast
**Solution:** Use symmetric colormap (`cmap='RdBu'`) if data has negative values

## Related Files
- `visualization/visualize_results.py` - Main visualization script
- `benchmarking/results/*.json` - Input data for graphs
- `presentation/slides.md` - Where figures are used
- `presentation/poster_content.md` - Poster figures

## Verification
Run: `python visualization/visualize_results.py --test-mode` (generates sample figures)

## Quick Start Template

```python
# visualization/visualize_results.py

import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    # Load benchmark results
    with open('benchmarking/results/comparison_report.json', 'r') as f:
        results = json.load(f)

    # Generate all figures
    plot_retention_curves(results['memory_retention'])
    plot_training_speed(results['training_speed'])
    plot_token_reduction(results['token_reduction'])

    print("All visualizations saved to visualization/")

if __name__ == '__main__':
    main()
```
