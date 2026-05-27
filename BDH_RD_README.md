# BDH-RD: BDH with Recurrent Depth
# =================================

**BDH-RD** is a "Mythos-style" recurrent depth architecture built natively into the BDH framework.

## Key Insight

OpenMythos implements "looped inference" where the same transformer layers are applied repeatedly:
```
h[t+1] = A·h[t] + B·e + Transformer(h[t], e)
```

But **BDH already has the core components**:
- State matrix `E[t]` = recurrent state
- Decay rate `λ` = A matrix (just fixed, not learned)
- Hebbian update `η·(Q⊗V)` = the learning rule

So instead of integrating OpenMythos, we **enhance BDH with looping**:

```
for loop in range(max_loops):
    state = decay * state + η * (Q ⊗ V)  # Hebbian update
    output = attention(state)
    if convergence_criterion(state):
        break
```

## Architecture

```
Input → Embedding → Loop { BDH Layer → State Update → Halting Check } → Output

Key additions over vanilla BDH:
1. Loop position embeddings (like Mythos's loop-index embedding)
2. Learnable decay (Mythos A matrix, but BDH-optimized)
3. Adaptive Computation Time (ACT) for token-level early exit
4. Loop regularization to encourage efficient computation
```

## Files Created

| File | Description |
|------|-------------|
| `implementation/bdh_recurrent.py` | BDH-RD model with recurrent depth |
| `train_bdh_recurrent.py` | Training script with checkpointing |
| `runpod_bdh_rd_setup.sh` | RunPod pod setup script |

## Usage

### Basic Training
```bash
python train_bdh_recurrent.py --config basic --batch_size 64 --epochs 10
```

### Resume Training
```bash
python train_bdh_recurrent.py --config basic --resume checkpoints/bdh_recurrent.pt
```

### Generate with Loop Tracking
```python
from bdh_recurrent import BDHRecurrent, BDHRecurrentConfig

model = BDHRecurrent(config).cuda()
output, loop_metrics = model.generate(context, return_loop_info=True)
print(f"Used {loop_metrics['loops_used']} loops per token")
```

## Expected Behavior

| Aspect | Vanilla BDH | BDH-RD |
|--------|-------------|--------|
| Reasoning depth | Fixed by layers | Variable via loops |
| Parameter efficiency | N layers = N params | N layers = infinite effective depth |
| Memory retention | Multi-scale states | Same + recurrent state |
| Training stability | Standard | Loop regularization added |
| Early exit | No | Per-token ACT halting |

## Cost Estimate (L4 GPU - Secure Cloud @ $0.39/hr)

| Training Run | Hours | Cost |
|--------------|-------|------|
| Basic (6 layers, 4 loops) | 12-24 hrs | $4.68-9.36 |
| Large (8 layers, 8 loops) | 24-48 hrs | $9.36-18.72 |
| Full experiment (3 configs) | ~100 hrs | ~$39 |

**$400 budget covers**: ~4-5 full experiments with tuning

## Next Steps

1. Test BDH-RD on RunPod with your dataset
2. Compare loop depths across different input types
3. Tune halting threshold for your use case
4. Evaluate against vanilla BDH on validation set

## Why This Beats Integrating OpenMythos

| Consideration | OpenMythos Integration | Native BDH-RD |
|---------------|------------------------|---------------|
| **Complexity** | High (two frameworks) | Low (one framework) |
| **Attention** | GQA/MLA (complex) | Linear attention (simpler) |
| **Memory** | Single recurrent state | Multi-scale + recurrent |
| **Hebbian learning** | No | Built-in |
| **Biological plausibility** | Low | High |
| **VRAM usage** | ~1.5GB (3B) | ~2GB (similar size) |

**Result**: You get the Mythos-style looping benefits with your existing BDH infrastructure.
