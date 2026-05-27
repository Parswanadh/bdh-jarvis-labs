"""
BDH Comprehensive Benchmark Suite
==================================

Measures BDH model performance across 5 key metrics:
1. Memory Retention (state matrix activation over time)
2. Training Throughput (tokens/second)
3. Token Reduction (byte-level vs BBPE)
4. Convergence Stability (loss curve analysis)
5. Perplexity (standard language modeling metric)

Author: Benchmark Architect (T13)
Date: 2025-02-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add parent directory to path to import BDH model
sys.path.insert(0, str(Path(__file__).parent.parent))
from bdh_gpu_10m import BDHConfig, BDHGPUTensor


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite"""
    # Model configs
    baseline_config: BDHConfig = None
    improved_configs: Dict[str, BDHConfig] = None

    # Dataset paths
    tiny_shakespeare_path: str = "benchmarking/test_datasets/tiny_shakespeare.txt"
    wikitext2_path: str = "benchmarking/test_datasets/wikitext-2.txt"

    # Benchmark settings
    sequence_lengths: List[int] = None
    num_runs: int = 3
    batch_size: int = 32
    max_iters: int = 1000  # For convergence benchmark

    # Device
    device: str = "auto"

    # Output
    results_dir: str = "benchmarking/results"

    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [100, 500, 1000, 2000]
        if self.improved_configs is None:
            self.improved_configs = {}


class TextDataset(Dataset):
    """Simple text dataset for benchmarking"""
    def __init__(self, text: str, max_seq_len: int = 512):
        self.text = text
        self.max_seq_len = max_seq_len
        self.data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]
        y = chunk[1:self.max_seq_len + 1]
        return x, y


class MemoryRetentionBenchmark:
    """
    Measures how well the state matrix maintains information over time.

    Key Metric: L2 norm of state matrix at different sequence lengths.
    Higher norm = better memory retention.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self, model: nn.Module, model_name: str) -> Dict[str, float]:
        """
        Run memory retention benchmark.

        Args:
            model: BDH model to benchmark
            model_name: Name of the model variant

        Returns:
            Dictionary mapping sequence_length -> retention_ratio
        """
        print(f"\n{'='*60}")
        print(f"Memory Retention Benchmark: {model_name}")
        print(f"{'='*60}")

        model.eval()
        device = next(model.parameters()).device

        retention_results = {}

        for seq_len in self.config.sequence_lengths:
            # Create random test sequence
            x = torch.randint(0, 256, (1, seq_len), device=device)

            with torch.no_grad():
                # Forward pass and get final state
                _, state = model(x, return_state=True)

                # Measure retention (L2 norm of state matrix)
                if state is not None:
                    state_norm = torch.norm(state).item()
                    # Normalize by expected initial norm
                    retention_ratio = state_norm / (seq_len ** 0.5)
                else:
                    retention_ratio = 0.0

            retention_results[str(seq_len)] = retention_ratio
            print(f"  Sequence length {seq_len:4d}: {retention_ratio:.4f}")

        self.results[model_name] = retention_results
        return retention_results


class TrainingThroughputBenchmark:
    """
    Measures training speed in tokens per second.

    Key Metric: Average tokens processed per second during training.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self, model: nn.Module, model_name: str, dataset: Dataset) -> Dict[str, float]:
        """
        Run training throughput benchmark.

        Args:
            model: BDH model to benchmark
            model_name: Name of the model variant
            dataset: Training dataset

        Returns:
            Dictionary with throughput metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Throughput Benchmark: {model_name}")
        print(f"{'='*60}")

        model.train()
        device = next(model.parameters()).device

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Setup optimizer (minimal for benchmarking)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Warm-up runs
        print("  Warming up...")
        for _ in range(3):
            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Benchmark runs
        tokens_per_sec_runs = []
        samples_per_sec_runs = []

        for run in range(self.config.num_runs):
            loader_iter = iter(loader)
            total_tokens = 0
            total_samples = 0

            # Synchronize GPU
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()

            # Process 10 batches
            for _ in range(10):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    x, y = next(loader_iter)

                x, y = x.to(device), y.to(device)

                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_tokens += x.numel()
                total_samples += x.size(0)

            # Synchronize GPU
            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            samples_per_sec = total_samples / elapsed

            tokens_per_sec_runs.append(tokens_per_sec)
            samples_per_sec_runs.append(samples_per_sec)

            print(f"  Run {run+1}: {tokens_per_sec:.0f} tokens/sec, {samples_per_sec:.1f} samples/sec")

        # Take median
        median_tokens_per_sec = np.median(tokens_per_sec_runs)
        median_samples_per_sec = np.median(samples_per_sec_runs)

        result = {
            "tokens_per_sec": float(median_tokens_per_sec),
            "samples_per_sec": float(median_samples_per_sec),
            "std_tokens_per_sec": float(np.std(tokens_per_sec_runs)),
        }

        self.results[model_name] = result
        print(f"  Median: {median_tokens_per_sec:.0f} tokens/sec")

        return result


class TokenReductionBenchmark:
    """
    Measures token reduction achieved by BBPE vs byte-level encoding.

    Key Metric: Ratio of byte tokens to BBPE tokens (higher = better).
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self, texts: List[str], model_name: str = "byte_level") -> Dict[str, float]:
        """
        Run token reduction benchmark.

        Args:
            texts: List of text samples to analyze
            model_name: Name of the tokenizer variant

        Returns:
            Dictionary with reduction statistics
        """
        print(f"\n{'='*60}")
        print(f"Token Reduction Benchmark: {model_name}")
        print(f"{'='*60}")

        byte_token_counts = []

        for text in texts[:100]:  # Sample 100 texts for speed
            byte_tokens = len(text.encode('utf-8'))
            byte_token_counts.append(byte_tokens)

        result = {
            "mean_tokens": float(np.mean(byte_token_counts)),
            "median_tokens": float(np.median(byte_token_counts)),
            "std_tokens": float(np.std(byte_token_counts)),
        }

        self.results[model_name] = result
        print(f"  Mean tokens: {result['mean_tokens']:.1f}")
        print(f"  Median tokens: {result['median_tokens']:.1f}")

        return result

    def compute_reduction_ratio(self, byte_results: Dict, bbpe_results: Dict) -> Dict[str, float]:
        """Compute reduction ratio between byte-level and BBPE."""
        return {
            "mean_reduction": byte_results["mean_tokens"] / bbpe_results["mean_tokens"],
            "median_reduction": byte_results["median_tokens"] / bbpe_results["median_tokens"],
        }


class ConvergenceStabilityBenchmark:
    """
    Measures training convergence stability.

    Key Metrics:
    - Stability: Inverse of loss variance (higher = more stable)
    - Convergence iterations: When loss plateaus
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self, model: nn.Module, model_name: str, dataset: Dataset) -> Dict[str, float]:
        """
        Run convergence stability benchmark.

        Args:
            model: BDH model to benchmark
            model_name: Name of the model variant
            dataset: Training dataset

        Returns:
            Dictionary with convergence metrics
        """
        print(f"\n{'='*60}")
        print(f"Convergence Stability Benchmark: {model_name}")
        print(f"{'='*60}")

        model.train()
        device = next(model.parameters()).device

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Track loss history
        loss_history = []

        print("  Training for convergence analysis...")
        loader_iter = iter(loader)

        for it in range(self.config.max_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())

            if it % 100 == 0:
                print(f"    Iter {it:4d}: loss {loss.item():.4f}")

        # Compute metrics (skip first 100 for stability)
        loss_array = np.array(loss_history[100:])
        variance = np.var(loss_array)
        stability = 1.0 / (1.0 + variance)

        # Find convergence point (when loss plateaus)
        convergence_point = self._find_convergence_point(loss_history)

        result = {
            "stability": float(stability),
            "loss_variance": float(variance),
            "final_loss": float(loss_history[-1]),
            "convergence_iter": int(convergence_point),
        }

        self.results[model_name] = result
        print(f"  Stability: {stability:.4f}")
        print(f"  Final loss: {loss_history[-1]:.4f}")
        print(f"  Convergence at iteration: {convergence_point}")

        return result

    def _find_convergence_point(self, loss_history: List[float], window: int = 100) -> int:
        """Find when loss starts to plateau."""
        if len(loss_history) < window * 2:
            return len(loss_history)

        for i in range(window, len(loss_history) - window):
            recent = np.mean(loss_history[i:i+window])
            prev = np.mean(loss_history[i-window:i])

            # If improvement is less than 1%, consider converged
            if (prev - recent) / prev < 0.01:
                return i

        return len(loss_history)


class PerplexityBenchmark:
    """
    Standard language modeling perplexity metric.

    Key Metric: Perplexity = exp(cross_entropy_loss)
    Lower perplexity = better model performance.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def run(self, model: nn.Module, model_name: str, dataset: Dataset) -> Dict[str, float]:
        """
        Run perplexity benchmark.

        Args:
            model: BDH model to benchmark
            model_name: Name of the model variant
            dataset: Test dataset

        Returns:
            Dictionary with perplexity metrics
        """
        print(f"\n{'='*60}")
        print(f"Perplexity Benchmark: {model_name}")
        print(f"{'='*60}")

        model.eval()
        device = next(model.parameters()).device

        # Create dataloader
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')

                total_loss += loss.item()
                total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        result = {
            "perplexity": float(perplexity),
            "cross_entropy_loss": float(avg_loss),
            "num_tokens": int(total_tokens),
        }

        self.results[model_name] = result
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Cross-entropy loss: {avg_loss:.4f}")

        return result


class BDHBenchmarkSuite:
    """
    Main benchmark suite orchestrator.

    Runs all 5 benchmarks and generates comprehensive reports.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models = {}
        self.all_results = {}

        # Initialize benchmark runners
        self.memory_benchmark = MemoryRetentionBenchmark(config)
        self.throughput_benchmark = TrainingThroughputBenchmark(config)
        self.token_benchmark = TokenReductionBenchmark(config)
        self.convergence_benchmark = ConvergenceStabilityBenchmark(config)
        self.perplexity_benchmark = PerplexityBenchmark(config)

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        print(f"Using device: {self.device}")

    def add_model(self, name: str, model: nn.Module):
        """Add a model to benchmark."""
        self.models[name] = model.to(self.device)
        print(f"Added model: {name}")

    def load_test_data(self) -> Tuple[str, str]:
        """Load test datasets."""
        print("\nLoading test datasets...")

        # Try Tiny Shakespeare first
        tiny_shakespeare = self._load_dataset(
            self.config.tiny_shakespeare_path,
            default_text="To be, or not to be, that is the question: " * 1000
        )

        # Try WikiText-2
        wikitext2 = self._load_dataset(
            self.config.wikitext2_path,
            default_text="The quick brown fox jumps over the lazy dog. " * 5000
        )

        return tiny_shakespeare, wikitext2

    def _load_dataset(self, path: str, default_text: str) -> str:
        """Load dataset from file or use default."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"  Warning: {path} not found, using default text")
            return default_text

    def run_all_benchmarks(self, fast_mode: bool = False) -> Dict[str, Any]:
        """
        Run complete benchmark suite on all models.

        Args:
            fast_mode: If True, run reduced benchmarks for quick testing

        Returns:
            Complete benchmark results dictionary
        """
        print("\n" + "="*60)
        print("BDH BENCHMARK SUITE")
        print("="*60)

        # Load test data
        tiny_shakespeare, wikitext2 = self.load_test_data()

        # Create dataset
        dataset = TextDataset(tiny_shakespeare, max_seq_len=512)

        # Prepare text samples for token reduction
        text_samples = [tiny_shakespeare[i:i+100] for i in range(0, min(len(tiny_shakespeare), 10000), 100)]

        # Run benchmarks for each model
        for model_name, model in self.models.items():
            print(f"\n{'#'*60}")
            print(f"# BENCHMARKING MODEL: {model_name.upper()}")
            print(f"{'#'*60}")

            self.all_results[model_name] = {}

            # 1. Memory Retention
            memory_results = self.memory_benchmark.run(model, model_name)
            self.all_results[model_name]["memory_retention"] = memory_results

            # 2. Training Throughput (skip in fast mode)
            if not fast_mode:
                throughput_results = self.throughput_benchmark.run(model, model_name, dataset)
                self.all_results[model_name]["training_throughput"] = throughput_results

            # 3. Token Reduction (byte-level baseline)
            token_results = self.token_benchmark.run(text_samples, model_name)
            self.all_results[model_name]["token_reduction"] = token_results

            # 4. Convergence Stability (skip in fast mode)
            if not fast_mode:
                # Reset model for convergence benchmark
                model_copy = self._reset_model(model)
                convergence_results = self.convergence_benchmark.run(model_copy, model_name, dataset)
                self.all_results[model_name]["convergence"] = convergence_results

            # 5. Perplexity
            perplexity_results = self.perplexity_benchmark.run(model, model_name, dataset)
            self.all_results[model_name]["perplexity"] = perplexity_results

        return self.all_results

    def _reset_model(self, model: nn.Module) -> nn.Module:
        """Create a fresh copy of model for convergence testing."""
        # Reinitialize model
        config = model.config
        new_model = BDHGPUTensor(config).to(self.device)
        return new_model

    def save_results(self, path: Optional[str] = None) -> str:
        """
        Save benchmark results to JSON file.

        Args:
            path: Output path (default: config.results_dir/benchmark_results.json)

        Returns:
            Path to saved results file
        """
        if path is None:
            os.makedirs(self.config.results_dir, exist_ok=True)
            path = os.path.join(self.config.results_dir, "benchmark_results.json")

        with open(path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"\nResults saved to: {path}")
        return path

    def generate_comparison_report(self) -> str:
        """
        Generate markdown comparison report.

        Returns:
            Path to generated report
        """
        if len(self.models) < 2:
            print("Need at least 2 models for comparison")
            return ""

        report_path = os.path.join(self.config.results_dir, "comparison_report.md")

        with open(report_path, 'w') as f:
            f.write("# BDH Benchmark Results Comparison\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Memory Retention Comparison
            f.write("## Memory Retention Improvement\n\n")
            f.write("| Sequence Length |")
            for name in self.models.keys():
                f.write(f" {name} |")
            f.write("\n|")
            for _ in self.models.keys():
                f.write("---------------|")
            f.write("\n")

            for seq_len in self.config.sequence_lengths:
                f.write(f"| {seq_len} tokens |")
                for name in self.models.keys():
                    val = self.all_results[name]["memory_retention"].get(str(seq_len), 0)
                    f.write(f" {val:.4f} |")
                f.write("\n")

            # Training Throughput Comparison
            f.write("\n## Training Throughput\n\n")
            f.write("| Model | Tokens/sec | Samples/sec |\n")
            f.write("|-------|-----------|-------------|\n")
            for name in self.models.keys():
                if "training_throughput" in self.all_results[name]:
                    tps = self.all_results[name]["training_throughput"]["tokens_per_sec"]
                    sps = self.all_results[name]["training_throughput"]["samples_per_sec"]
                    f.write(f"| {name} | {tps:.0f} | {sps:.1f} |\n")

            # Perplexity Comparison
            f.write("\n## Perplexity (lower is better)\n\n")
            f.write("| Model | Perplexity | Loss |\n")
            f.write("|-------|------------|------|\n")
            for name in self.models.keys():
                ppl = self.all_results[name]["perplexity"]["perplexity"]
                loss = self.all_results[name]["perplexity"]["cross_entropy_loss"]
                f.write(f"| {name} | {ppl:.2f} | {loss:.4f} |\n")

            # Convergence Comparison
            f.write("\n## Convergence Stability\n\n")
            f.write("| Model | Stability | Convergence Iter | Final Loss |\n")
            f.write("|-------|-----------|------------------|------------|\n")
            for name in self.models.keys():
                if "convergence" in self.all_results[name]:
                    stab = self.all_results[name]["convergence"]["stability"]
                    conv = self.all_results[name]["convergence"]["convergence_iter"]
                    loss = self.all_results[name]["convergence"]["final_loss"]
                    f.write(f"| {name} | {stab:.4f} | {conv} | {loss:.4f} |\n")

        print(f"Comparison report saved to: {report_path}")
        return report_path


def create_baseline_config() -> BDHConfig:
    """Create baseline BDH configuration (single decay rate 0.99)."""
    return BDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        state_decay=0.99,
        hebbian_lr=0.01,
        max_seq_len=2048,
    )


def create_multiscale_config() -> BDHConfig:
    """Create multi-scale BDH configuration (future: multiple decay rates)."""
    # For now, use different decay rate
    return BDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        state_decay=0.995,  # Slower decay for better retention
        hebbian_lr=0.01,
        max_seq_len=2048,
    )


def main():
    """Main entry point for benchmark suite."""
    print("="*60)
    print("BDH BENCHMARK SUITE")
    print("="*60)

    # Create configuration
    config = BenchmarkConfig(
        baseline_config=create_baseline_config(),
        improved_configs={"multiscale": create_multiscale_config()},
        num_runs=3,
        batch_size=32,
        device="auto",
    )

    # Create benchmark suite
    suite = BDHBenchmarkSuite(config)

    # Add models
    baseline_model = BDHGPUTensor(config.baseline_config)
    suite.add_model("baseline", baseline_model)

    # Add improved variant
    multiscale_model = BDHGPUTensor(config.improved_configs["multiscale"])
    suite.add_model("multiscale", multiscale_model)

    # Run benchmarks
    results = suite.run_all_benchmarks(fast_mode=False)

    # Save results
    suite.save_results()

    # Generate comparison report
    suite.generate_comparison_report()

    print("\n" + "="*60)
    print("BENCHMARK SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
