"""
Performance Measurement Logger for BDH Science Fest Sprint
===========================================================

This module provides comprehensive logging and measurement tools for
collecting performance metrics during BDH model training and evaluation.

Key Metrics Collected:
1. Memory retention at different time steps (t=100, 500, 1000, 2000)
2. Training tokens per second
3. Token reduction ratios
4. Convergence stability metrics
5. Loss curves and training dynamics
6. GPU memory usage
7. Parameter efficiency metrics
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
import sys


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    timestamp: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    tokens_per_second: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    sequence_length: int
    batch_size: int
    gradient_norm: Optional[float] = None
    state_matrix_norm: Optional[float] = None


@dataclass
class MemoryRetentionMetrics:
    """Memory capability metrics at specific time steps"""
    timestamp: str
    time_step: int
    retention_accuracy: float
    retrieval_latency_ms: float
    state_decay_rate: float
    hebbian_learning_rate: float
    context_length: int


@dataclass
class ComparisonMetrics:
    """Baseline vs Improved comparison metrics"""
    metric_name: str
    baseline_value: float
    improved_value: float
    improvement_ratio: float
    percentage_improvement: float
    statistical_significance: Optional[str] = None
    confidence_interval: Optional[str] = None


class PerformanceLogger:
    """
    Central logger for all performance metrics.

    Outputs:
    - JSON files for structured data
    - Human-readable reports
    - CSV exports for analysis
    """

    def __init__(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)

        # Initialize metric storage
        self.training_metrics: List[TrainingMetrics] = []
        self.memory_retention: Dict[int, MemoryRetentionMetrics] = {}
        self.comparison_data: Dict[str, List[ComparisonMetrics]] = {}
        self.timing_data: Dict[str, List[float]] = {}

        # Configuration storage
        self.config_data: Dict[str, Any] = {}

        # Baseline data (for comparison)
        self.baseline_metrics: Dict[str, float] = {}

        print(f"[PerformanceLogger] Initialized logging to: {self.run_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Store model and training configuration"""
        self.config_data.update(config)
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[PerformanceLogger] Configuration saved")

    def log_training_step(self, metrics: TrainingMetrics) -> None:
        """Record metrics from a training step"""
        self.training_metrics.append(metrics)

        # Write to incremental file
        training_file = self.run_dir / "training_metrics.jsonl"
        with open(training_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')

    def log_memory_retention(self, metrics: MemoryRetentionMetrics) -> None:
        """Record memory retention at a specific time step"""
        self.memory_retention[metrics.time_step] = metrics

        memory_file = self.run_dir / "memory_retention.json"
        with open(memory_file, 'w') as f:
            data = {str(k): asdict(v) for k, v in self.memory_retention.items()}
            json.dump(data, f, indent=2)

    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set a baseline value for comparison"""
        self.baseline_metrics[metric_name] = value
        baseline_file = self.run_dir / "baseline_metrics.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.baseline_metrics, f, indent=2)
        print(f"[PerformanceLogger] Baseline set: {metric_name} = {value}")

    def log_comparison(self, metric_name: str, improved_value: float,
                       statistical_test: Optional[str] = None) -> ComparisonMetrics:
        """Create comparison against baseline"""
        if metric_name not in self.baseline_metrics:
            raise ValueError(f"No baseline set for {metric_name}")

        baseline_value = self.baseline_metrics[metric_name]
        improvement_ratio = improved_value / baseline_value if baseline_value != 0 else float('inf')
        percentage_improvement = ((improved_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0

        comparison = ComparisonMetrics(
            metric_name=metric_name,
            baseline_value=baseline_value,
            improved_value=improved_value,
            improvement_ratio=improvement_ratio,
            percentage_improvement=percentage_improvement,
            statistical_significance=statistical_test
        )

        if metric_name not in self.comparison_data:
            self.comparison_data[metric_name] = []
        self.comparison_data[metric_name].append(comparison)

        # Save to file
        comparison_file = self.run_dir / "comparisons.json"
        with open(comparison_file, 'w') as f:
            data = {k: [asdict(v) for v in vals] for k, vals in self.comparison_data.items()}
            json.dump(data, f, indent=2)

        print(f"[PerformanceLogger] Comparison: {metric_name}")
        print(f"  Baseline: {baseline_value:.4f} -> Improved: {improved_value:.4f}")
        print(f"  Improvement: {percentage_improvement:+.2f}%")

        return comparison

    @contextmanager
    def measure_time(self, operation_name: str):
        """Context manager for timing operations"""
        start = time.perf_counter()
        start_gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            gpu_used = (torch.cuda.memory_allocated() - start_gpu) / 1024**3 if torch.cuda.is_available() else 0

            if operation_name not in self.timing_data:
                self.timing_data[operation_name] = []
            self.timing_data[operation_name].append(elapsed)

            print(f"[Timing] {operation_name}: {elapsed:.4f}s (GPU: {gpu_used:.2f}GB)")

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of timing data"""
        summary = {}
        for op, times in self.timing_data.items():
            if times:
                summary[op] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "median": np.median(times),
                    "count": len(times)
                }
        return summary

    def export_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        report_lines = [
            "# BDH Performance Analysis Report",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "",
            "## Configuration",
            "```"
        ]

        # Add configuration
        for key, value in self.config_data.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("```")
        report_lines.append("")

        # Add training metrics summary
        if self.training_metrics:
            losses = [m.loss for m in self.training_metrics]
            tps = [m.tokens_per_second for m in self.training_metrics]

            report_lines.extend([
                "## Training Summary",
                f"- Total steps: {len(self.training_metrics)}",
                f"- Final loss: {losses[-1]:.4f}",
                f"- Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%",
                f"- Avg tokens/sec: {np.mean(tps):.0f}",
                f"- Peak tokens/sec: {np.max(tps):.0f}",
                ""
            ])

        # Add memory retention summary
        if self.memory_retention:
            report_lines.extend([
                "## Memory Retention",
                ""
            ])
            for t_step, metrics in self.memory_retention.items():
                report_lines.extend([
                    f"### t={t_step}",
                    f"- Retention accuracy: {metrics.retention_accuracy:.2%}",
                    f"- Retrieval latency: {metrics.retrieval_latency_ms:.2f}ms",
                    f"- State decay rate: {metrics.state_decay_rate:.4f}",
                    ""
                ])

        # Add comparisons summary
        if self.comparison_data:
            report_lines.extend([
                "## Performance Improvements",
                ""
            ])

            for metric_name, comparisons in self.comparison_data.items():
                latest = comparisons[-1]
                report_lines.extend([
                    f"### {metric_name}",
                    f"- Baseline: {latest.baseline_value:.4f}",
                    f"- Improved: {latest.improved_value:.4f}",
                    f"- **Improvement: {latest.percentage_improvement:+.2f}%** ({latest.improvement_ratio:.2f}x)",
                    ""
                ])

        # Add timing summary
        timing_summary = self.get_timing_summary()
        if timing_summary:
            report_lines.extend([
                "## Operation Timing",
                ""
            ])
            for op, stats in timing_summary.items():
                report_lines.extend([
                    f"### {op}",
                    f"- Mean: {stats['mean']:.4f}s",
                    f"- Std: {stats['std']:.4f}s",
                    f"- Min: {stats['min']:.4f}s",
                    f"- Max: {stats['max']:.4f}s",
                    ""
                ])

        report_content = "\n".join(report_lines)

        # Save report
        report_file = self.run_dir / "summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        print(f"[PerformanceLogger] Summary report saved to: {report_file}")

        return report_content

    def export_csv(self) -> None:
        """Export metrics to CSV files for analysis"""
        import csv

        # Training metrics CSV
        if self.training_metrics:
            csv_file = self.run_dir / "training_metrics.csv"
            with open(csv_file, 'w', newline='') as f:
                if self.training_metrics:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.training_metrics[0]).keys())
                    writer.writeheader()
                    for metric in self.training_metrics:
                        writer.writerow(asdict(metric))
            print(f"[PerformanceLogger] Training CSV exported: {csv_file}")

        # Comparisons CSV
        if self.comparison_data:
            csv_file = self.run_dir / "comparisons.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric_name', 'baseline_value', 'improved_value',
                               'improvement_ratio', 'percentage_improvement'])
                for metric_name, comparisons in self.comparison_data.items():
                    for comp in comparisons:
                        writer.writerow([
                            metric_name, comp.baseline_value, comp.improved_value,
                            comp.improvement_ratio, comp.percentage_improvement
                        ])
            print(f"[PerformanceLogger] Comparisons CSV exported: {csv_file}")


class StatisticalAnalyzer:
    """
    Statistical analysis tools for comparing baseline vs improved models.
    """

    @staticmethod
    def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
        """Compute confidence interval for mean"""
        if len(data) < 2:
            return (np.mean(data), np.mean(data))

        import scipy.stats as stats
        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (mean - h, mean + h)

    @staticmethod
    def t_test(baseline: List[float], improved: List[float]) -> Dict[str, float]:
        """Perform independent t-test"""
        import scipy.stats as stats
        t_stat, p_value = stats.ttest_ind(baseline, improved)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline)**2 + np.std(improved)**2) / 2)
        cohens_d = (np.mean(improved) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05
        }

    @staticmethod
    def wilcoxon_test(baseline: List[float], improved: List[float]) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        import scipy.stats as stats
        if len(baseline) != len(improved):
            min_len = min(len(baseline), len(improved))
            baseline = baseline[:min_len]
            improved = improved[:min_len]

        stat, p_value = stats.wilcoxon(baseline, improved)
        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }


def print_system_info():
    """Print system and environment information"""
    import platform
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    print("=" * 60)


if __name__ == "__main__":
    # Test the logger
    print_system_info()

    logger = PerformanceLogger()

    # Log sample config
    logger.log_config({
        "model": "BDH-GPU-10M",
        "n_layers": 6,
        "n_embd": 256,
        "batch_size": 32,
        "learning_rate": 0.001
    })

    # Set baselines
    logger.set_baseline("loss", 3.5)
    logger.set_baseline("tokens_per_second", 1000)
    logger.set_baseline("memory_retention_t100", 0.75)

    # Log some training steps
    for i in range(5):
        metrics = TrainingMetrics(
            timestamp=datetime.now().isoformat(),
            epoch=0,
            step=i,
            loss=3.5 - i * 0.1,
            learning_rate=0.001,
            tokens_per_second=1000 + i * 50,
            gpu_memory_allocated_gb=2.5,
            gpu_memory_reserved_gb=3.0,
            sequence_length=256,
            batch_size=32
        )
        logger.log_training_step(metrics)

    # Log comparison
    logger.log_comparison("loss", 2.5)
    logger.log_comparison("tokens_per_second", 1500)

    # Generate report
    print("\n" + logger.export_summary_report())
    logger.export_csv()

    print("\n[PerformanceLogger] Test complete!")
