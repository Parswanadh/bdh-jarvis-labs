# Multi-Scale BDH Implementation Skill

## Purpose
Guide the implementation of multi-scale synaptic state matrices in BDH architecture to extend memory retention.

## When to Use This Skill
- Implementing multi-scale decay in BDH layers
- Designing state matrix update mechanisms
- Debugging multi-scale state issues
- Optimizing memory usage for multiple states

## Key Capabilities

### 1. Multi-Scale Architecture Design
- Design multi-scale state matrices with different decay rates
- Understand Hebbian learning dynamics at multiple timescales
- Balance memory overhead vs performance improvement

### 2. State Management
- Initialize multiple state matrices correctly
- Handle state persistence across forward passes
- Manage state reset and accumulation

### 3. Implementation Details
```python
class MultiScaleBDHConfig(BDHConfig):
    decay_rates: list = [0.95, 0.99, 0.995]  # Fast, medium, slow timescales
    num_scales: int = 3

class MultiScaleLinearAttention(nn.Module):
    def __init__(self, config):
        # Multiple state matrices
        self.states = [
            torch.zeros(config.n_embd, config.n_embd)
            for _ in range(config.num_scales)
        ]
        self.decay_rates = config.decay_rates

    def update_state(self, Q, V):
        for i, decay in enumerate(self.decay_rates):
            hebbian = torch.outer(Q.flatten(), V.flatten())
            self.states[i] = decay * self.states[i] + 0.01 * hebbian
```

## Testing Checklist
- [ ] Each state matrix maintains proper decay rate
- [ ] Memory overhead is acceptable (3× baseline = ~3MB for 256×256×3)
- [ ] Retention curves show extended memory (measure at t=100, 500, 1000, 2000)
- [ ] Multi-scale states are independent (no cross-contamination)
- [ ] State persistence works across batch boundaries

## Expected Results
- Single decay (0.99): ~500 token effective memory
- Multi-scale [0.95, 0.99, 0.995]: ~1000-2000 token effective memory
- Biologically plausible (fast/medium/slow synaptic plasticity)

## Related Files
- `bdh_gpu_10m.py` - Baseline BDH implementation
- `implementation/multiscale_bdh.py` - Multi-scale implementation (new)
- `implementation/train_multiscale.py` - Training script (new)
- `benchmarking/benchmark_suite.py` - Benchmark tests

## Biological Inspiration
Multi-scale timescales are inspired by:
- **STP (Short-term plasticity):** Fast facilitation/depression (100ms scale)
- **LTP (Long-term potentiation):** Medium-term synaptic strengthening (seconds-minutes)
- **Structural changes:** Slow synaptic consolidation (hours-days)

Our implementation mimics this with three decay rates:
- **0.95:** Fast timescale (~100 tokens)
- **0.99:** Medium timescale (~500 tokens)
- **0.995:** Slow timescale (~1000-2000 tokens)

## Common Issues & Solutions

**Issue:** State matrices grow too large
**Solution:** Use lower precision (float16) or reduce n_embd

**Issue:** Training instability with multi-scale
**Solution:** Use lower hebbian_lr (0.001 instead of 0.01)

**Issue:** States don't persist correctly
**Solution:** Ensure state is returned and passed between calls

## Performance Tips
1. Pre-allocate state matrices during __init__
2. Use in-place operations where possible
3. Batch state updates across all scales
4. Profile memory usage during training

## Verification
Run: `python -c "from implementation.multiscale_bdh import MultiScaleBDH; print('Multi-scale BDH imported successfully')"`
