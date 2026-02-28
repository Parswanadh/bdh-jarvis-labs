# BDH Benchmarking Skill

## Purpose
Design and execute comprehensive benchmarks to measure BDH improvements quantitatively.

## When to Use This Skill
- Creating test suites for BDH models
- Measuring model performance (perplexity, speed, memory)
- Comparing baseline vs improved versions
- Collecting data for visualizations and presentations

## Key Metrics

### 1. Memory Retention (PRIMARY METRIC)
**What:** Measure how well state matrix maintains information over time

**How:**
```python
def test_memory_retention(model, sequence_lengths=[100, 500, 1000, 2000]):
    results = {}
    for length in sequence_lengths:
        # Create test sequence
        x = torch.randn(1, length, model.config.n_embd)

        # Get state after processing
        _, state = model(x, return_state=True)

        # Measure retention (L2 norm of state matrix)
        initial_norm = torch.norm(state).item()
        results[length] = initial_norm

    return results
```

**Expected Results:**
- Baseline (0.99): 36% @ t=100, 0.6% @ t=500, ~0% @ t=1000
- Multi-scale [0.95, 0.99, 0.995]: 15% @ t=500, 5% @ t=1000, 1-2% @ t=2000

### 2. Training Throughput
**What:** Measure tokens processed per second during training

**How:**
```python
import time

def benchmark_training_speed(model, dataset, batch_size=32):
    start = time.time()
    tokens_processed = 0

    for batch in dataset:
        loss = model(batch)
        tokens_processed += batch.numel()

    elapsed = time.time() - start
    tokens_per_sec = tokens_processed / elapsed
    return tokens_per_sec
```

**Expected Results:**
- Byte-level: ~10,000 tokens/sec (RTX 4070)
- BBPE: ~25,000-30,000 tokens/sec (2-3× speedup)

### 3. Token Reduction (BBPE)
**What:** Measure reduction in sequence length

**How:**
```python
def test_token_reduction(texts, byte_tokenizer, bbpe_tokenizer):
    results = []
    for text in texts:
        byte_tokens = len(byte_tokenizer.encode(text).ids)
        bbpe_tokens = len(bbpe_tokenizer.encode(text).ids)
        reduction = byte_tokens / bbpe_tokens
        results.append(reduction)

    return {
        'mean_reduction': np.mean(results),
        'median_reduction': np.median(results),
        'std': np.std(results)
    }
```

**Expected Results:**
- Mean reduction: 3-4×
- Median reduction: 2.5-3.5×
- Std: 0.5-1.0 (consistency)

### 4. Convergence Stability
**What:** Measure training loss curve smoothness

**How:**
```python
def analyze_convergence(loss_history):
    # Measure variance (lower = more stable)
    variance = np.var(loss_history[100:])  # Skip first 100 iterations

    # Measure convergence speed
    convergence_point = find_where_loss_plateaus(loss_history)

    return {
        'stability': 1.0 / (1.0 + variance),  # Higher = more stable
        'convergence_iters': convergence_point
    }
```

**Expected Results:**
- Baseline (unstable): High variance, slow convergence
- Stabilized config: Low variance, 20-30% faster convergence

### 5. Perplexity
**What:** Standard language modeling metric

**How:**
```python
def compute_perplexity(model, test_data):
    total_loss = 0
    total_tokens = 0

    for batch in test_data:
        logits = model(batch)
        loss = F.cross_entropy(logits, batch)
        total_loss += loss.item() * batch.numel()
        total_tokens += batch.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity
```

**Expected Results:**
- Baseline BDH: Similar to GPT-2 (paper claims)
- Improved BDH: Similar or slightly better (due to stability)

## Test Datasets

### Quick Iteration (Day 1)
- **Tiny Shakespeare**: 111KB text, fast to test
- Use for: Quick validation, debugging

### Standard Benchmarks (Day 2-3)
- **Wikitext-2**: Standard language modeling benchmark
- Use for: Published results, comparison

### Custom Demo Set (Day 3)
- **Science demo text**: Curated for live demo
- Use for: Presentation, showing improvements

## Benchmark Suite Structure

```python
# benchmarking/benchmark_suite.py

class BDHBenchmarkSuite:
    def __init__(self, models):
        self.models = models  # {'baseline': model1, 'multiscale': model2, ...}

    def run_all_benchmarks(self):
        results = {}

        # 1. Memory Retention
        results['memory_retention'] = self.benchmark_memory_retention()

        # 2. Training Speed
        results['training_speed'] = self.benchmark_training_speed()

        # 3. Token Reduction
        results['token_reduction'] = self.benchmark_token_reduction()

        # 4. Convergence
        results['convergence'] = self.benchmark_convergence()

        # 5. Perplexity
        results['perplexity'] = self.benchmark_perplexity()

        return results

    def save_results(self, results, path='benchmarking/results/benchmark_results.json'):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
```

## Testing Checklist

### Before Benchmarking
- [ ] All models loaded correctly
- [ ] Test datasets prepared and validated
- [ ] GPU memory available (check with `nvidia-smi`)
- [ ] Output directories created
- [ ] Baseline results collected first

### During Benchmarking
- [ ] Each benchmark runs 3+ times (take median)
- [ ] Monitor GPU memory usage
- [ ] Log any errors or anomalies
- [ ] Save intermediate results

### After Benchmarking
- [ ] Results saved to JSON
- [ ] Statistical analysis completed
- [ ] Comparison report generated
- [ ] Visualizations created from results

## Result Format

### JSON Structure
```json
{
  "baseline": {
    "memory_retention": {
      "100": 0.36,
      "500": 0.0065,
      "1000": 0.00004,
      "2000": 0.0
    },
    "training_speed": {
      "tokens_per_sec": 10000,
      "samples_per_sec": 52
    },
    "token_reduction": {
      "mean": 1.0,
      "median": 1.0,
      "std": 0.0
    },
    "convergence": {
      "stability": 0.7,
      "convergence_iters": 3500
    },
    "perplexity": {
      "wikitext2": 25.3
    }
  },
  "multiscale": {
    "memory_retention": {
      "100": 0.45,
      "500": 0.15,
      "1000": 0.05,
      "2000": 0.015
    },
    ...
  }
}
```

### Comparison Report
```markdown
# BDH Benchmark Results Comparison

## Memory Retention Improvement
| Sequence Length | Baseline | Multi-scale | Improvement |
|----------------|----------|-------------|-------------|
| 100 tokens     | 36%      | 45%         | +25%        |
| 500 tokens     | 0.6%     | 15%         | **+2400%** |
| 1000 tokens    | ~0%      | 5%          | **∞**       |
| 2000 tokens    | ~0%      | 1.5%        | **∞**       |

## Training Speedup (BBPE)
| Metric | Byte-level | BBPE | Speedup |
|--------|-----------|------|---------|
| Tokens/sec | 10,000 | 27,500 | **2.75×** |
| Seq length | 512 | 150 | **3.4× shorter** |

## Conclusion
Multi-scale decay extends effective memory from ~500 tokens to ~2000 tokens (**4× improvement**).
BBPE reduces sequence length by 3.4× while maintaining byte-level foundation.
```

## Common Issues & Solutions

**Issue:** Benchmarks inconsistent between runs
**Solution:** Set random seeds (`torch.manual_seed(42)`) and use fixed test data

**Issue:** GPU memory overflow during benchmarking
**Solution:** Reduce batch size or sequence length, or clear cache between runs

**Issue:** Results don't show expected improvement
**Solution:** Verify implementation is correct (check state updates, decay rates)

**Issue:** Benchmarking takes too long
**Solution:** Use smaller test dataset for quick iteration, full dataset for final results

## Performance Tips

1. **Warm-up runs**: First run includes compilation overhead, discard it
2. **Batch sizing**: Use largest batch that fits in GPU memory
3. **GPU synchronization**: Use `torch.cuda.synchronize()` before timing
4. **Multiple runs**: Run 3+ times and take median (avoid outliers)

```python
# Proper timing pattern
import torch

model.train()
for _ in range(3):  # Warm-up
    _ = model(batch)

torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    _ = model(batch)
torch.cuda.synchronize()
elapsed = time.time() - start
```

## Related Files
- `benchmarking/benchmark_suite.py` - Main benchmark suite
- `benchmarking/test_datasets/` - Test data
- `benchmarking/results/` - Results storage
- `visualization/visualize_results.py` - Create graphs from results

## Verification
Run: `python benchmarking/benchmark_suite.py --test-mode` (quick test with small data)
