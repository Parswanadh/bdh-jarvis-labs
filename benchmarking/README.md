# BDH Benchmarking Framework

**Performance Analysis Infrastructure for BDH Science Fest Sprint**

---

## Overview

This directory contains the complete benchmarking and analysis framework for measuring and comparing BDH model performance.

## Directory Structure

```
benchmarking/
├── measurement_logger.py      # Core logging and metrics collection
├── benchmark_runner.py         # Comprehensive benchmark suite
├── statistical_analysis.py     # Statistical testing and analysis
├── results/
│   ├── comparison_report_template.md  # Template for final report
│   ├── baseline/               # Baseline model results
│   ├── improved/               # Improved model results
│   └── [RUN_ID]/              # Timestamped run directories
└── test_datasets/              # Benchmark datasets
```

## Quick Start

### 1. Run Baseline Benchmarks

```bash
cd D:/projects/BDH
python benchmarking/benchmark_runner.py
```

This runs:
- Memory retention tests (t=100, 500, 1000, 2000)
- Training speed benchmarks
- Convergence stability analysis
- GPU memory profiling

### 2. Collect Improved Model Results

After implementing improvements, run again to collect comparison data.

### 3. Generate Comparison Report

```python
from benchmarking.statistical_analysis import ComparisonReportGenerator

generator = ComparisonReportGenerator()
# Add comparisons...
generator.generate_report()
```

## Metrics Collected

### Primary Metrics

| Metric | Description | Target Time Steps |
|--------|-------------|-------------------|
| Memory Retention | Accuracy of token prediction after t steps | 100, 500, 1000, 2000 |
| Training Speed | Tokens processed per second | All configurations |
| Token Efficiency | Tokens needed to reach target loss | 100, 500, 1000, 2000 |
| Convergence Stability | Loss variance during training | Last 100 steps |

### Secondary Metrics

- GPU memory usage (allocated/reserved)
- Gradient norms
- Learning rate dynamics
- State matrix evolution

## Statistical Analysis

### Tests Performed

1. **Independent t-test** - Parametric test for mean differences
2. **Welch's t-test** - For unequal variances
3. **Mann-Whitney U** - Non-parametric alternative
4. **Wilcoxon signed-rank** - For paired samples
5. **Cliff's Delta** - Non-parametric effect size

### Multiple Comparison Correction

Applied to control false discovery rate:
- Bonferroni (conservative)
- Holm (step-down)
- Benjamini-Hochberg (FDR control)

## Output Files

### Per Run

Each benchmark run creates a timestamped directory:

```
results/[RUN_ID]/
├── config.json                    # Model configuration
├── baseline_metrics.json          # Baseline values
├── training_metrics.jsonl         # Incremental training data
├── training_metrics.csv           # Training metrics export
├── memory_retention.json          # Memory test results
├── comparisons.json               # All comparisons made
├── benchmark_results.json         # Complete benchmark data
├── summary_report.md              # Human-readable summary
└── statistical_comparison.json    # Statistical analysis
```

### Final Reports

- `comparison_report_template.md` - Template for Day 2-3 reporting
- `statistical_comparison_report.md` - Generated statistical analysis
- `key_improvements.md` - Summary for demo (Day 3)

## Coordinate With

- **T6 (benchmark-architect)**: Framework design and setup
- **T8 (visualization-specialist)**: Graphs and charts
- **T9 (demo-choreographer)**: Key metrics for presentation
- **T14 (progress-tracker)**: Status reporting

## Day 1 Tasks (Complete)

- [x] Set up measurement tools (`measurement_logger.py`)
- [x] Create benchmark runner (`benchmark_runner.py`)
- [x] Design statistical analysis framework (`statistical_analysis.py`)
- [x] Prepare results directory structure
- [x] Create comparison report template

## Day 2 Tasks (Pending)

- [ ] Run baseline benchmarks with T6
- [ ] Collect improved model data
- [ ] Perform statistical analysis
- [ ] Generate comparison tables
- [ ] Provide data to T8 for visualization

## Day 3 Tasks (Pending)

- [ ] Create final comparison report
- [ ] Identify key improvements for demo
- [ ] Generate summary statistics
- [ ] Provide compelling narrative

## Usage Examples

### Example 1: Logging Training Metrics

```python
from benchmarking.measurement_logger import PerformanceLogger, TrainingMetrics
from datetime import datetime

logger = PerformanceLogger()

metrics = TrainingMetrics(
    timestamp=datetime.now().isoformat(),
    epoch=1,
    step=1000,
    loss=2.345,
    learning_rate=0.001,
    tokens_per_second=5000,
    gpu_memory_allocated_gb=4.5,
    gpu_memory_reserved_gb=5.0,
    sequence_length=512,
    batch_size=16
)

logger.log_training_step(metrics)
```

### Example 2: Statistical Comparison

```python
from benchmarking.statistical_analysis import ComparisonReportGenerator

generator = ComparisonReportGenerator()

baseline_scores = [3.5, 3.4, 3.6, 3.3, 3.5]
improved_scores = [2.8, 2.9, 2.7, 2.8, 2.9]

generator.add_comparison(
    metric_name="Validation Loss",
    baseline=baseline_scores,
    improved=improved_scores,
    higher_is_better=False
)

report = generator.generate_report()
```

### Example 3: Running Full Benchmark Suite

```python
from bdh_gpu_10m import BDHConfig, BDHGPUTensor
from benchmarking.benchmark_runner import ComprehensiveBenchmarkSuite

config = BDHConfig(vocab_size=256, n_embd=256, n_layer=6)
model = BDHGPUTensor(config)

suite = ComprehensiveBenchmarkSuite(model, device="cuda")
results = suite.run_all_benchmarks()

# Automatically generates:
# - JSON results files
# - Summary report
# - CSV exports
```

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU benchmarks)
- scipy 1.9+ (for statistical tests)
- numpy 1.21+

## Troubleshooting

### Out of Memory Errors

Reduce batch size or sequence length in benchmark configurations.

### Statistical Test Warnings

Small sample sizes (<10) may produce unreliable results. Increase iterations.

### Slow Benchmarks

Reduce `num_iterations` in speed benchmarks for faster results.

---

**Maintained By:** T7 (Performance Analyst)
**Last Updated:** 2025-02-25
