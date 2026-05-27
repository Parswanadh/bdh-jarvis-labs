# BDH Production Training Guide

## Overview

Complete training pipeline for 41M BDH model with Qwen 3.5 0.8B distillation on TinyStories.

## Key Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Data** | Full 2.12M TinyStories | 55K was only 2.5% - undertraining |
| **Tokens** | ~400M-1.2B tokens | Matches Chinchilla optimal for 41M params |
| **Epochs** | 3 | Full dataset coverage |
| **Batch Size** | 4 (micro) x 8 (accumulation) = 32 | Fits 8GB VRAM |
| **Peak LR** | 5e-4 | Distillation benefits from higher LR |
| **Schedule** | 5% warmup + cosine decay to 10% | Standard for stable training |
| **Temperature** | 2.5 | Balances dark knowledge vs hard targets |
| **Alpha (CE weight)** | 0.3 | 70% distillation, 30% CE loss |
| **Grad Clip** | 0.5 | Prevents NaN from exploding gradients |
| **Dtype** | bfloat16 | No scaling needed, more stable than fp16 |

## Training Timeline

```
Total steps: ~200K
Estimated time: 12-24 hours on RTX 4070

Milestones:
- Step 10K: Initial checkpoint, loss ~3.0-3.5
- Step 50K: Should see loss ~2.5-2.8
- Step 100K: Loss ~2.2-2.5
- Step 200K: Target loss ~2.0-2.3
```

## Data Requirements

### TinyStories Dataset

1. Download full dataset:
```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all.txt
```

2. Or use the download script:
```bash
python download_tinystories.py
```

3. Expected file size: ~1.5GB uncompressed

### Why Full Dataset?

- Current 55K stories = ~2.5% of available data
- For 41M params, Chinchilla suggests ~800M tokens optimal
- TinyStories full = ~400M tokens (simple domain)
- More data = better generalization, less overfitting

## Stability Measures

### NaN Prevention

1. **Gradient Clipping**: `max_grad_norm=0.5`
   - Aggressive clipping prevents explosion
   - Monitor grad_norm in logs

2. **bfloat16**: Use instead of float16
   - Wider exponent range prevents overflow
   - No loss scaler needed
   - Supported on RTX 40xx

3. **Logit Clipping**: Clamp teacher logits to [-10, 10]
   - Prevents extreme values in softmax
   - Stabilizes KL divergence

4. **Loss Monitoring**: Track CE and KL separately
   - If KL spikes: reduce temperature or alpha
   - If CE spikes: check for data issues

### Recovery from NaN

If training crashes at step 116K again:

1. Load latest checkpoint before NaN
2. Reduce `learning_rate` by 50%
3. Reduce `max_grad_norm` to 0.3
4. Consider reducing `distill_temperature` to 2.0

## Evaluation

### Metrics to Track

1. **Training Loss**: Should decrease monotonically (with noise)
2. **Validation Loss**: Monitor for overfitting
3. **Perplexity**: `exp(validation_loss)`, target < 10
4. **Gradient Norm**: Should be < 1.0 consistently
5. **Generation Quality**: Qualitative assessment

### Overfitting Detection

```
Healthy: train_loss and val_loss decrease together
Overfitting: val_loss plateaus while train_loss continues
Underfitting: both losses still decreasing at end
```

For 41M params on 2M stories, overfitting is unlikely. More likely to underfit.

### Validation Schedule

- Every 1000 steps (~3-5 minutes)
- Save best checkpoint based on val_loss
- Generate 2-3 samples for qualitative check

## Hyperparameter Tuning

### Learning Rate

| Issue | Symptom | Fix |
|-------|---------|-----|
| Too high | NaN, loss spikes | Reduce to 3e-4 |
| Too low | Slow convergence, plateau | Increase to 7e-4 |
| Just right | Smooth decrease | 5e-4 |

### Temperature

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 1.0 | Hard targets, no dark knowledge | Not recommended |
| 2.0 | Moderate softening | Good baseline |
| 2.5 | Balanced (default) | Recommended |
| 4.0+ | Very soft, emphasize dark knowledge | May need more epochs |

### Alpha (Loss Weight)

| Alpha | CE Weight | KL Weight | Use Case |
|-------|-----------|-----------|----------|
| 0.1 | 10% | 90% | Maximum distillation |
| 0.3 | 30% | 70% | Balanced (default) |
| 0.5 | 50% | 50% | More emphasis on hard targets |

## Quick Start

```bash
# 1. Download data
python download_tinystories.py

# 2. Install dependencies
pip install torch transformers

# 3. Run training
python train_production_bdh.py

# 4. Monitor logs
# Logs print to console with:
# - Step progress
# - Loss components (CE, KL)
# - Gradient norms
# - VRAM usage
# - Validation results
```

## Expected Results

Based on similar distillation work:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Final Loss | < 2.3 | < 2.2 | < 2.0 |
| Perplexity | < 10 | < 9 | < 7.5 |
| Generation | Coherent | Logical | Creative |

## Troubleshooting

### OOM (Out of Memory)

```python
# Reduce batch size
batch_size = 2  # Was 4

# Reduce sequence length
max_seq_len = 256  # Was 512

# Reduce accumulation
gradient_accumulation_steps = 4  # Was 8
```

### Slow Training

```python
# Increase batch size if VRAM allows
batch_size = 8

# Reduce logging frequency
log_interval = 500  # Was 100

# Cache teacher logits (advanced)
cache_teacher_logits = True
```

### Loss Not Decreasing

1. Check data: Are stories loading correctly?
2. Check learning rate: Maybe too low?
3. Check teacher: Is it generating sensible outputs?
4. Check student: Is forward pass correct?

### NaN Loss

1. Reduce `learning_rate`
2. Reduce `max_grad_norm`
3. Reduce `distill_temperature`
4. Check for corrupted data
5. Ensure using bfloat16, not fp16

## Advanced: Progressive Distillation

For even better results, consider progressive training:

```python
# Phase 1: High temp, high distillation weight (steps 0-50K)
temperature = 4.0
alpha = 0.1

# Phase 2: Moderate values (steps 50K-150K)
temperature = 2.5
alpha = 0.3

# Phase 3: Lower temp, more CE (steps 150K-200K)
temperature = 1.5
alpha = 0.5
```

This requires modifying the training loop to update these values mid-training.
