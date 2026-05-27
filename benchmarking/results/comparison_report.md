# BDH Benchmark Results Comparison

Generated: 2026-03-20 09:11:16

## Memory Retention Improvement

| Sequence Length | baseline | multiscale |
|---------------|---------------|
| 100 tokens | 0.0005 | 0.0005 |
| 500 tokens | 0.0001 | 0.0001 |
| 1000 tokens | 0.0000 | 0.0000 |
| 2000 tokens | 0.0000 | 0.0000 |

## Training Throughput

| Model | Tokens/sec | Samples/sec |
|-------|-----------|-------------|
| baseline | 151492 | 295.9 |
| multiscale | 156311 | 305.3 |

## Perplexity (lower is better)

| Model | Perplexity | Loss |
|-------|------------|------|
| baseline | 290.31 | 5.6709 |
| multiscale | 276.08 | 5.6207 |

## Convergence Stability

| Model | Stability | Convergence Iter | Final Loss |
|-------|-----------|------------------|------------|
| baseline | 0.7644 | 741 | 3.4642 |
| multiscale | 0.7546 | 1000 | 3.5041 |
