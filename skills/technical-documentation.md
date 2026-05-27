# Technical Documentation Skill

## Purpose
Write clear, comprehensive documentation for code changes, architecture decisions, and user guides.

## When to Use This Skill
- Documenting code changes and design decisions
- Writing README files and user guides
- Creating inline code comments
- Preparing research reports and publications

## Documentation Types

### 1. IMPLEMENTATION_CHANGES.md
**Purpose:** Explain what was changed, why, and how it works.

**Template:**
```markdown
# BDH Implementation Changes

## Overview
This document describes the architectural improvements made to the baseline BDH implementation for the science fair project.

## Changes Made

### 1. Multi-Scale State Matrices
**Problem:** Baseline BDH uses single decay rate (0.99), limiting effective memory to ~500 tokens.

**Solution:** Implemented multi-scale state matrices with decay rates [0.95, 0.99, 0.995].

**Implementation:**
- Modified `LinearAttention` class to support multiple state matrices
- Added `MultiScaleBDHConfig` with configurable decay rates
- State update rule: `E_i(t+1) = decay_i * E_i(t) + lr * (Q ⊗ V)`

**Files Changed:**
- `implementation/multiscale_bdh.py` (new)
- `implementation/train_multiscale.py` (new)

**Results:**
- Effective memory extended from ~500 to ~2000 tokens (4× improvement)
- Memory overhead: 3× baseline (~3MB for 256×256×3)
- See `visualization/retention_curves.png` for details

### 2. BBPE Tokenization
**Problem:** Byte-level tokenization (vocab=256) requires 3-5× longer sequences.

**Solution:** Implemented Byte-Level BPE with vocab_size=8192.

**Implementation:**
- Trained BBPE tokenizer on sample dataset
- Modified embedding layer: 256 → 8192 vocabulary
- Maintains byte-level foundation (starts with byte alphabet)

**Files Changed:**
- `implementation/bbpe_bdh.py` (new)
- `tokenizer-model/` (new directory)

**Results:**
- Token reduction: 3-4× fewer tokens
- Training speedup: 2.5-3× faster
- See `visualization/training_comparison.png` for details

### 3. Training Stabilization
**Problem:** BDH is 20-30% harder to train than Transformers.

**Solution:** Implemented proper initialization, warmup, and gradient clipping.

**Implementation:**
- Initializer range: 0.02 → 0.006
- Warmup steps: 500 → 5000
- Gradient clipping: disabled → 1.0
- Learning rate: tuned to 3e-4

**Files Changed:**
- `implementation/stable_config.py` (new)

**Results:**
- Stable convergence (no loss spikes)
- 20-30% faster convergence
- See `visualization/training_curves.png` for details

## Architecture Diagrams

### Multi-Scale State Flow
```
Input Tokens
    ↓
Embedding [B, T, 256]
    ↓
LinearAttention (with Multi-Scale State)
    ├─→ State_Fast: decay=0.95, E_f[256×256]
    ├─→ State_Med:  decay=0.99, E_m[256×256]
    └─→ State_Slow: decay=0.995, E_s[256×256]
    ↓
Combined Output
    ↓
FFN + Gating
    ↓
Next Token Prediction
```

## Performance Summary

| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Memory (500 tokens) | 0.6% retention | 15% retention | **24×** |
| Memory (2000 tokens) | ~0% retention | 1.5% retention | **∞** |
| Training speed | 10K tok/s | 27.5K tok/s | **2.75×** |
| Seq length | 512 tokens | 150 tokens | **3.4× shorter** |

## Usage

### Training Multi-Scale BDH
\`\`\`bash
python implementation/train_multiscale.py \\
    --config implementation/stable_config.py \\
    --dataset data/shakespeare.txt \\
    --output checkpoints/multiscale_bdh.pt
\`\`\`

### Training BBPE-BDH
\`\`\`bash
# First train tokenizer
python implementation/train_tokenizer.py \\
    --input data/shakespeare.txt \\
    --vocab_size 8192 \\
    --output tokenizer-model/

# Then train model
python implementation/train_multiscale.py \\
    --tokenizer tokenizer-model/tokenizer.json \\
    --config implementation/stable_config.py \\
    --dataset data/shakespeare.txt \\
    --output checkpoints/bbpe_bdh.pt
\`\`\`

## References
- BDH Paper: arXiv:2509.26507
- BBPE: ByT5 paper (https://arxiv.org/abs/2105.13626)
- Multi-scale timescales: Kording et al. (2001)
```

### 2. USER_GUIDE.md
**Purpose:** Help users understand and use the improved BDH.

**Template:**
```markdown
# Improved BDH User Guide

## Quick Start

### Installation
\`\`\`bash
# Install dependencies
pip install torch tokenizers matplotlib numpy

# Clone repo
git clone https://github.com/your-repo/bdh-improvements
cd bdh-improvements
\`\`\`

### Training Your First Model

#### 1. Prepare Data
\`\`\`bash
# Create text file
echo "Your training text here..." > data/my_data.txt
\`\`\`

#### 2. Train Tokenizer (Optional - for BBPE)
\`\`\`bash
python implementation/train_tokenizer.py \\
    --input data/my_data.txt \\
    --vocab_size 8192 \\
    --output tokenizer-model/
\`\`\`

#### 3. Train Model
\`\`\`bash
# Multi-scale BDH (byte-level)
python implementation/train_multiscale.py \\
    --config implementation/stable_config.py \\
    --dataset data/my_data.txt \\
    --output checkpoints/my_model.pt

# BBPE-BDH
python implementation/train_multiscale.py \\
    --tokenizer tokenizer-model/tokenizer.json \\
    --config implementation/stable_config.py \\
    --dataset data/my_data.txt \\
    --output checkpoints/my_model_bbpe.pt
\`\`\`

### Generating Text

\`\`\`python
from implementation.multiscale_bdh import MultiScaleBDH
import torch

# Load model
model = MultiScaleBDH.from_pretrained('checkpoints/my_model.pt')
model.eval()

# Generate
context = "The meaning of life is"
output = model.generate(context, max_tokens=100)
print(output)
\`\`\`

## Configuration Options

### Multi-Scale Config
\`\`\`python
from implementation.multiscale_bdh import MultiScaleBDHConfig

config = MultiScaleBDHConfig(
    vocab_size=256,          # Byte-level (or 8192 for BBPE)
    n_embd=256,             # Embedding dimension
    n_layer=6,              # Number of layers
    decay_rates=[0.95, 0.99, 0.995],  # Multi-scale decay
    hebbian_lr=0.01,        # Hebbian learning rate
    state_decay=0.99        # Legacy single decay (unused in multiscale)
)
\`\`\`

### Training Config
\`\`\`python
from implementation.stable_config import StableTrainingConfig

config = StableTrainingConfig(
    batch_size=32,
    learning_rate=3e-4,
    warmup_steps=5000,      # Important for stability!
    gradient_clip=1.0,      # Important for stability!
    initializer_range=0.006 # Important for stability!
)
\`\`\`

## Troubleshooting

### Issue: Training loss becomes NaN
**Solution:**
1. Reduce learning rate to 1e-4
2. Check initializer_range is 0.006 (not 0.02)
3. Enable gradient clipping (1.0)
4. Use bfloat16 instead of float16

### Issue: GPU out of memory
**Solution:**
1. Reduce batch_size (32 → 16 → 8)
2. Reduce n_embd (256 → 128)
3. Reduce n_layer (6 → 4)
4. Use gradient checkpointing

### Issue: Slow training
**Solution:**
1. Use BBPE tokenization (2-3× speedup)
2. Enable torch.compile
3. Use bfloat16 (faster than float32)
4. Increase batch_size (if GPU memory allows)

## Advanced Usage

### Custom Decay Rates
\`\`\`python
# Fast memory only (short-term)
config = MultiScaleBDHConfig(decay_rates=[0.90, 0.95])

# Long-term memory focused
config = MultiScaleBDHConfig(decay_rates=[0.98, 0.99, 0.995, 0.999])

# Biologically-inspired timescales
config = MultiScaleBDHConfig(decay_rates=[
    0.90,   # STP: 100ms scale
    0.95,   # STP: 1s scale
    0.98,   # LTP: 10s scale
    0.99,   # LTP: 100s scale
    0.995,  # Structural: 1000s scale
])
\`\`\`

## Benchmarks

Run the benchmark suite:
\`\`\`bash
python benchmarking/benchmark_suite.py \\
    --baseline checkpoints/baseline_bdh.pt \\
    --improved checkpoints/multiscale_bdh.pt \\
    --output benchmarking/results/
\`\`\`

## Citation

If you use this work, please cite:
\`\`\`bibtex
@misc{bdh_improvements_2026,
  title={Multi-Scale Memory for Brain-Inspired AI},
  author={Your Name},
  year={2026},
  note={Science Fair Project}
}
\`\`\`
```

### 3. Code Comments
**Purpose:** Explain complex code sections inline.

**Guidelines:**
\`\`\`python
class MultiScaleLinearAttention(nn.Module):
    """
    Multi-scale linear attention for BDH.

    Implements Hebbian learning at multiple timescales, inspired by
    fast/medium/slow synaptic plasticity in the brain.

    Args:
        config: MultiScaleBDHConfig with decay_rates list

    Example:
        >>> config = MultiScaleBDHConfig(decay_rates=[0.95, 0.99, 0.995])
        >>> attn = MultiScaleLinearAttention(config)
        >>> output, states = attn(x, return_state=True)
    \"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decay_rates = config.decay_rates

        # Initialize one state matrix per timescale
        # Shape: [num_scales, n_embd, n_embd]
        self.states = [
            torch.zeros(config.n_embd, config.n_embd)
            for _ in range(len(config.decay_rates))
        ]

    def update_state(self, Q, V):
        """
        Update all state matrices using Hebbian learning.

        Hebbian rule: "Neurons that fire together, wire together"
        E_new = decay * E_old + lr * (Q ⊗ V)

        Args:
            Q: Query tensor [batch, seq_len, n_embd]
            V: Value tensor [batch, seq_len, n_embd]
        \"""
        for i, decay in enumerate(self.decay_rates):
            # Outer product captures co-activation patterns
            hebbian_update = torch.einsum('bti,bti->ij', Q, V)

            # Decay old state, add new learning
            self.states[i] = decay * self.states[i] + self.config.hebbian_lr * hebbian_update
\`\`\`

## Documentation Best Practices

### DO's ✅
- Explain WHY, not just WHAT
- Use examples for complex concepts
- Keep comments up-to-date with code changes
- Use consistent formatting
- Include diagrams for architecture

### DON'Ts ❌
- Don't document obvious code (e.g., `i += 1  # increment i`)
- Don't let comments diverge from code
- Don't use jargon without definition
- Don't write walls of text (break into sections)
- Don't assume reader has deep context

## Related Files
- `docs/IMPLEMENTATION_CHANGES.md` - Technical changes
- `docs/USER_GUIDE.md` - User documentation
- `README.md` - Project overview
- All implementation files (inline comments)

## Quick Checklist

### Before Committing Code
- [ ] All new functions have docstrings
- [ ] Complex logic has inline comments
- [ ] README.md is updated
- [ ] Implementation changes are documented
- [ ] Examples are provided

### Before Release
- [ ] User guide is complete
- [ ] Installation instructions work
- [ ] Troubleshooting section covers common issues
- [ ] Code examples are tested
- [ ] Citation information is provided

## Verification
Run: `python -m pydoc implementation.multiscale_bdh` (should show docstrings)
\