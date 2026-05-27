# BDH v2 Reproducibility Checklist

> Day 9 — Integration tests, reproducibility checklist, and submission preparation.
> Last updated: 2026-05-17

This checklist ensures that any researcher can reproduce all BDH v2 results from scratch.

---

## 1. Environment Setup

### 1.1 Python Version
- [ ] Python 3.10+ (tested on 3.10, 3.11, 3.12)
- [ ] Create a clean virtual environment:
  ```bash
  python -m venv .venv_bdh
  source .venv_bdh/bin/activate   # Linux/macOS
  .venv_bdh\Scripts\activate      # Windows
  ```

### 1.2 PyTorch Version
- [ ] PyTorch 2.1+ with CUDA 12.1 (or CPU-only fallback)
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- [ ] Verify installation:
  ```python
  import torch
  print(torch.__version__)
  print(torch.cuda.is_available())
  ```

### 1.3 Dependencies
- [ ] Install all dependencies:
  ```bash
  pip install -e ".[dev]"
  ```
- [ ] Or manually:
  ```bash
  pip install pytest numpy scipy matplotlib
  ```
- [ ] Record exact versions:
  ```bash
  pip freeze > requirements_frozen.txt
  ```

### 1.4 Known Dependency Issues
| Package | Issue | Workaround |
|---------|-------|------------|
| `scipy` | May fail on Windows without MSVC Build Tools | Install Visual Studio Build Tools or use `conda install scipy` |
| `matplotlib` | Backend errors on headless servers | Set `MPLBACKEND=Agg` |
| `psutil` | Optional for memory tracking | Tests still pass without it |

---

## 2. Random Seed Configuration

### 2.1 Seeds Used in Experiments
| Experiment | Seed(s) |
|------------|---------|
| Unit tests | N/A (deterministic fixtures) |
| Benchmark suite | 42, 123, 456 |
| Ablation study | 42, 123, 456 |
| Training pipeline | 42 (default, override with `--seed`) |

### 2.2 Seed Setting Code
All entry points set seeds via:
```python
import torch, numpy as np, random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2.3 Reproducibility Notes
- [ ] `torch.backends.cudnn.deterministic = True` is set for exact reproducibility
- [ ] `torch.backends.cudnn.benchmark = False` disables auto-tuner
- [ ] DataLoader workers use seeded generators per sample (see `SyntheticDistillDataset`)
- [ ] Multi-GPU training: use `torch.cuda.manual_seed_all()` (already implemented)

---

## 3. Hardware Requirements

### 3.1 Minimum (CPU-only, small model)
| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores (x86_64 or ARM64) |
| RAM | 8 GB |
| Storage | 2 GB (code + checkpoints) |
| GPU | Not required |

### 3.2 Recommended (GPU, medium model)
| Resource | Requirement |
|----------|-------------|
| CPU | 8+ cores |
| RAM | 16 GB |
| Storage | 10 GB (code + checkpoints + data) |
| GPU | NVIDIA GPU with 8 GB+ VRAM (RTX 3060 or better) |
| CUDA | 12.1+ |

### 3.3 Full-scale (v2_medium, distillation)
| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA A100 40GB or equivalent |
| RAM | 32 GB |
| Storage | 50 GB (for teacher data) |
| Multi-GPU | Optional (DDP supported) |

### 3.4 GPU Compatibility Matrix
| GPU | v2_small | v2_medium | Notes |
|-----|----------|-----------|-------|
| RTX 3060 (12GB) | Yes | Yes (batch_size=4) | Reduce batch size if OOM |
| RTX 4070 (12GB) | Yes | Yes | Good for ablation |
| A100 (40GB) | Yes | Yes | Recommended for full runs |
| A100 (80GB) | Yes | Yes | Can use larger batches |
| CPU only | Yes (slow) | Yes (very slow) | Use `--device cpu` |

---

## 4. Dataset Preparation

### 4.1 Synthetic Data (for testing/pipeline validation)
- [ ] No preparation needed — generated on-the-fly by `SyntheticDistillDataset`
- [ ] Run pipeline test:
  ```bash
  python -m implementation.train_distillation_v2 --synthetic --steps 100
  ```

### 4.2 Real Distillation Data
- [ ] Teacher model: Qwen3.5-0.8B (151936 vocab)
- [ ] Student tokenizer: TinyLlama (32000 vocab)
- [ ] Data format: `.pt` files with `input_ids` [T] and `teacher_logits` [T, 151936]
- [ ] Directory structure:
  ```
  data/distillation/
  ├── sample_0001.pt
  ├── sample_0002.pt
  └── ...
  ```
- [ ] Generate data (if needed):
  ```bash
  python generate_teacher_data.py --model Qwen3.5-0.8B --output data/distillation/
  ```

### 4.3 Benchmark Data
- [ ] All benchmarks use synthetic data — no external downloads required
- [ ] Data is generated deterministically from seeds inside each benchmark class

---

## 5. Training Commands

### 5.1 Unit Tests
```bash
python -m pytest implementation/test_bdh_v2.py -v
```

### 5.2 Integration Tests
```bash
python -m pytest tests/test_integration.py -v
```

### 5.3 Validation (Days 2-4 fixes)
```bash
python implementation/validate_fixes.py
```

### 5.4 Training — Small Model (v2_small)
```bash
python -m implementation.train_distillation_v2 \
    --config v2_small \
    --synthetic \
    --epochs 5 \
    --batch_size 8 \
    --lr 3e-4 \
    --output_dir outputs/bdh_v2_small \
    --seed 42
```

### 5.5 Training — Medium Model (v2_medium)
```bash
python -m implementation.train_distillation_v2 \
    --config v2_medium \
    --synthetic \
    --epochs 10 \
    --batch_size 4 \
    --lr 3e-4 \
    --output_dir outputs/bdh_v2_medium \
    --seed 42
```

### 5.6 Training — Resume from Checkpoint
```bash
python -m implementation.train_distillation_v2 \
    --config v2_small \
    --synthetic \
    --epochs 10 \
    --resume outputs/bdh_v2_small/checkpoint_step_500.pt
```

### 5.7 Benchmark Suite (Single Benchmark)
```bash
python benchmarking/benchmark_suite_v2.py \
    --benchmark perplexity \
    --seeds 42 123 456 \
    --output benchmarking/results/perplexity.json
```

### 5.8 Full Ablation Study
```bash
python benchmarking/benchmark_suite_v2.py \
    --ablation \
    --seeds 42 123 456 \
    --output benchmarking/results/ablation.json
```

### 5.9 SOTA Comparison
```bash
python benchmarking/sota_comparison.py \
    --output-dir benchmarking/results/sota_comparison \
    --format all
```

---

## 6. Expected Outputs and Validation Criteria

### 6.1 Unit Tests
- [ ] 33 tests pass (see `implementation/test_bdh_v2.py`)
- [ ] Expected runtime: < 30 seconds on CPU
- [ ] All tests should pass deterministically

### 6.2 Integration Tests
- [ ] 28 tests pass (see `tests/test_integration.py`)
- [ ] Expected runtime: < 60 seconds on CPU
- [ ] Key validations:
  - Training loop completes without NaN/Inf
  - Checkpoint resume produces identical trajectory
  - All 5 fix configs run forward + backward
  - Vocab projection integrates with KL loss
  - Generation produces valid token IDs
  - Gradient clipping constrains norms
  - LR scheduler follows warmup + cosine decay
  - Memory stays bounded for 512-token sequences

### 6.3 Benchmark Results (Expected Ranges)
| Benchmark | Baseline | +all_fixes | Metric |
|-----------|----------|------------|--------|
| Perplexity | 20-50 | 15-40 | Lower is better |
| Needle-in-Haystack | 0.1-0.5 | 0.2-0.7 | Accuracy |
| Sequence Reversal | 0.05-0.3 | 0.1-0.5 | Exact match |
| Copy Task | 0.3-0.8 | 0.5-0.9 | Exact match |
| Associative Recall | 0.1-0.4 | 0.2-0.6 | Accuracy |

### 6.4 Training Loss (v2_small, synthetic, 5 epochs)
- [ ] Final train loss < 5.0
- [ ] Loss should decrease monotonically (with some noise)
- [ ] No NaN/Inf at any step

### 6.5 SOTA Comparison
- [ ] Markdown table generated
- [ ] CSV export generated
- [ ] Radar chart PNG generated
- [ ] Full JSON with positioning analysis generated

---

## 7. How to Reproduce Each Figure in the Paper

### 7.1 Figure 1: BDH Architecture Diagram
- [ ] Source: `docs/figures/architecture_diagram.pdf` (manually created)
- [ ] Tools: TikZ (LaTeX) or draw.io
- [ ] Reproduction: Edit `docs/neurips_paper.tex` `\input{figures/architecture_diagram}`

### 7.2 Figure 2: Multi-Scale Memory Decay
- [ ] Run: Extract state matrix norms during training
  ```python
  # In training loop:
  for layer in model.layers:
      for scale, state in enumerate(states[layer]):
          norm = state.norm().item()
  ```
- [ ] Plot: Decay curves for 3 timescales (λ=0.95, 0.99, 0.995)
- [ ] Source data: `outputs/bdh_v2_medium/training.log`

### 7.3 Figure 3: Ablation Study Results
- [ ] Run full ablation:
  ```bash
  python benchmarking/benchmark_suite_v2.py --ablation --seeds 42 123 456
  ```
- [ ] Output: `benchmarking/results/ablation.json`
- [ ] Plot: Bar chart with error bars (mean ± std across seeds)
- [ ] Script: Use the comparison table printed to console

### 7.4 Figure 4: SOTA Radar Chart
- [ ] Run:
  ```bash
  python benchmarking/sota_comparison.py --format radar
  ```
- [ ] Output: `benchmarking/results/sota_comparison/radar_chart.png`
- [ ] Already publication-ready at 300 DPI

### 7.5 Figure 5: Training Loss Curves
- [ ] Parse `training.log` files from multiple runs
- [ ] Plot train loss and val loss over steps
- [ ] Compare baseline vs +all_fixes

### 7.6 Figure 6: Memory Efficiency Comparison
- [ ] Run benchmarks with `MemoryTracker`
- [ ] Compare peak GPU memory: BDH vs Transformer at equal params
- [ ] Source: `benchmarking/benchmark_suite_v2.py` MemoryTracker class

---

## 8. Known Issues and Workarounds

### 8.1 CUDA OOM on Large Sequences
- **Symptom**: `CUDA out of memory` with seq_len > 512
- **Fix**: Reduce batch size or use gradient accumulation
- **Command**: `--batch_size 2 --grad_accum_steps 8`

### 8.2 Slow CPU Training
- **Symptom**: Training takes hours on CPU
- **Fix**: Use `--synthetic --steps 100` for pipeline validation only
- **Note**: Full training requires GPU

### 8.3 Non-Deterministic Results Across Runs
- **Symptom**: Different seeds produce different benchmark scores
- **Fix**: This is expected — use 3 seeds and report mean ± std
- **Note**: Same seed should produce identical results

### 8.4 Windows Console Encoding Errors
- **Symptom**: `UnicodeEncodeError` when printing Greek letters
- **Fix**: Set `PYTHONIOENCODING=utf-8` or use `chcp 65001`
- **Already handled**: SOTA comparison replaces λ with "lambda"

### 8.5 scipy Import Error
- **Symptom**: `ModuleNotFoundError: No module named 'scipy'`
- **Fix**: `pip install scipy`
- **Note**: Only needed for benchmark significance tests

### 8.6 psutil Import Warning
- **Symptom**: Memory tracking shows 0.0 MB on some systems
- **Fix**: `pip install psutil`
- **Note**: Non-critical — benchmarks still run without it

---

## 9. Verification Script

Run this one-liner to verify the full setup:

```bash
python -m pytest implementation/test_bdh_v2.py tests/test_integration.py -v --tb=short 2>&1 | tail -5
```

Expected output:
```
=== X passed, 0 failed in Ys ===
```

---

## 10. Checklist Sign-Off

| Item | Status | Verified By | Date |
|------|--------|-------------|------|
| Environment setup | Complete | — | 2026-05-17 |
| Seed configuration | Complete | — | 2026-05-17 |
| Hardware requirements | Documented | — | 2026-05-17 |
| Dataset preparation | Documented | — | 2026-05-17 |
| Training commands | Documented | — | 2026-05-17 |
| Expected outputs | Documented | — | 2026-05-17 |
| Figure reproduction | Documented | — | 2026-05-17 |
| Known issues | Documented | — | 2026-05-17 |
