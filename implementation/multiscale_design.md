# Multi-Scale BDH Architecture Design

**Author:** Multi-Scale BDH Architect
**Date:** 2025-02-25
**Status:** Design Document - DAY 1

## Executive Summary

This document outlines the design for extending BDH memory retention from ~500 tokens to ~2000+ tokens through multi-scale synaptic state matrices. The design draws inspiration from biological synaptic plasticity at multiple timescales.

## Problem Statement

The baseline BDH implementation uses a single state matrix with decay rate 0.99, which provides:
- **Effective memory:** ~500 tokens
- **Limitation:** Information decays too quickly for long-range dependencies
- **Goal:** Extend to 2000+ tokens without significantly increasing model size

## Solution: Multi-Scale Synaptic States

### Biological Inspiration

Neural systems exhibit plasticity at multiple timescales:
1. **STP (Short-term Plasticity):** Fast facilitation/depression (100ms scale)
2. **LTP (Long-term Potentiation):** Medium-term strengthening (seconds-minutes)
3. **Structural Changes:** Slow consolidation (hours-days)

Our implementation mimics this with three decay rates:
- **Fast (0.95):** ~100 token timescale - captures recent context
- **Medium (0.99):** ~500 token timescale - maintains conversation history
- **Slow (0.995):** ~1000-2000 token timescale - preserves long-term patterns

## Architecture Design

### 1. Configuration Class

```python
@dataclass
class MultiScaleBDHConfig(BDHConfig):
    """Extended configuration for multi-scale BDH"""
    decay_rates: list = field(default_factory=lambda: [0.95, 0.99, 0.995])
    num_scales: int = 3
    scale_weights: list = field(default_factory=lambda: [0.2, 0.3, 0.5])  # Weight for each scale
```

**Design Decisions:**
- Decay rates are configurable for experimentation
- Scale weights allow biasing toward slower timescales
- Maintains backward compatibility with base BDHConfig

### 2. State Matrix Architecture

```python
class MultiScaleLinearAttention(nn.Module):
    """Linear attention with multi-scale state matrices"""

    def __init__(self, config: MultiScaleBDHConfig):
        # Three state matrices for different timescales
        self.E_fast = None   # Fast decay (0.95)
        self.E_med = None    # Medium decay (0.99)
        self.E_slow = None   # Slow decay (0.995)

        self.decay_rates = config.decay_rates
        self.scale_weights = config.scale_weights
```

**Memory Overhead Analysis:**
- Single state: 256 × 256 × 4 bytes = 256 KB
- Three states: 256 × 256 × 4 bytes × 3 = 768 KB
- **Overhead:** ~512 KB additional (acceptable for RTX 4070 8GB)

### 3. State Update Rule

For each scale i with decay rate γ_i:

```
E_i ← γ_i × E_i + η × (Q ⊗ V)

where:
- E_i: State matrix at scale i
- γ_i: Decay rate for scale i (0.95, 0.99, 0.995)
- η: Hebbian learning rate
- Q ⊗ V: Outer product (averaged over batch/sequence)
```

**Key Design Principles:**
1. **Independence:** Each scale updates independently
2. **Hebbian Learning:** Same update rule for all scales
3. **Weighted Combination:** States combined based on scale_weights

### 4. Forward Pass with Multi-Scale States

```python
def forward(self, x, state=None, return_state=False):
    # ... standard linear attention ...

    if return_state:
        # Initialize states if needed
        if state is None:
            E_fast, E_med, E_slow = self._init_states(x.device)
        else:
            E_fast, E_med, E_slow = state

        # Update each scale independently
        hebbian = self._compute_hebbian_update(Q, V)

        E_fast = self.decay_rates[0] * E_fast + self.config.hebbian_lr * hebbian
        E_med  = self.decay_rates[1] * E_med  + self.config.hebbian_lr * hebbian
        E_slow = self.decay_rates[2] * E_slow + self.config.hebbian_lr * hebbian

        # Combine states (weighted average)
        state_combined = (
            self.scale_weights[0] * E_fast +
            self.scale_weights[1] * E_med +
            self.scale_weights[2] * E_slow
        )

        return out, (E_fast, E_med, E_slow)
```

### 5. Layer Architecture

```python
class MultiScaleBDHLayer(nn.Module):
    """BDH layer with multi-scale states"""

    def __init__(self, config: MultiScaleBDHConfig):
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiScaleLinearAttention(config)
        self.ffn = ReLULowRankFFN(config)
```

**Note:** We can reuse the existing ReLULowRankFFN - no changes needed.

### 6. Complete Model

```python
class MultiScaleBDH(nn.Module):
    """Multi-scale BDH model"""

    def __init__(self, config: MultiScaleBDHConfig):
        self.token_embedding = nn.Embedding(...)
        self.layers = nn.ModuleList([
            MultiScaleBDHLayer(config) for _ in range(config.n_layer)
        ])
        self.norm_f = nn.LayerNorm(...)
        self.output_projection = nn.Linear(...)
```

## Expected Performance Improvements

### Memory Retention Analysis

| Decay Rate | Effective Memory | Timescale |
|------------|------------------|-----------|
| 0.95       | ~100 tokens      | Fast      |
| 0.99       | ~500 tokens      | Medium    |
| 0.995      | ~1000-2000 tokens| Slow      |
| **Combined** | **~1500-2000 tokens** | **Multi-scale** |

**Theoretical Basis:**
- After t steps with decay γ, retention ≈ γ^t
- At t=1000: γ=0.99 retains ~0.00001, but γ=0.995 retains ~0.006
- Multi-scale combines all: weighted average preserves information

## Implementation Plan

### DAY 2 Tasks
1. Create `implementation/multiscale_bdh.py`
   - MultiScaleBDHConfig class
   - MultiScaleLinearAttention class
   - MultiScaleBDHLayer class
   - MultiScaleBDH model class

2. Create `implementation/train_multiscale.py`
   - Training loop for multi-scale model
   - State management across batches
   - Checkpoint saving/loading

3. Test with small dataset
   - Verify state persistence
   - Measure memory retention
   - Profile performance

### DAY 3 Tasks
1. Integration with benchmarking suite
2. Fix any bugs found during testing
3. Optimize performance if needed

## Validation Strategy

### Unit Tests
- [ ] State matrices initialized correctly
- [ ] Each scale maintains independent decay
- [ ] State persistence across batch boundaries
- [ ] No cross-contamination between scales

### Integration Tests
- [ ] Training converges without instability
- [ ] Memory retention measured at t=100, 500, 1000, 2000
- [ ] Comparison with baseline BDH

### Performance Metrics
- **Memory overhead:** <1MB additional per model
- **Speed overhead:** <10% slower than baseline
- **Retention improvement:** 3-4x baseline

## Risks and Mitigations

### Risk 1: Training Instability
**Mitigation:** Lower hebbian_lr (0.001 instead of 0.01)

### Risk 2: State Explosion
**Mitigation:** Use float16 for state matrices

### Risk 3: Poor Convergence
**Mitigation:** Tune scale_weights via hyperparameter search

## Future Enhancements

1. **Adaptive Decay Rates:** Learn decay rates during training
2. **More Scales:** Add 4-5 scales for even longer memory
3. **Hierarchical States:** Organize scales hierarchically
4. **Gated Updates:** Learn when to update each scale

## References

1. BDH Paper: "The Dragon Hatchling" (arXiv:2509.26507)
2. Multi-scale skill: `skills/multiscale-bdh-implementation.md`
3. Baseline implementation: `bdh_gpu_10m.py`

## Appendix: Mathematical Derivation

### Single-scale retention:
After t steps: E(t) = γ^t × E(0)

### Multi-scale retention:
E_combined(t) = Σ w_i × γ_i^t × E_i(0)

For t=1000:
- Fast: 0.95^1000 ≈ 0
- Medium: 0.99^1000 ≈ 0.00001
- Slow: 0.995^1000 ≈ 0.006
- Combined: 0.2×0 + 0.3×0.00001 + 0.5×0.006 ≈ 0.003

**Result:** 300x better retention than medium-only!
