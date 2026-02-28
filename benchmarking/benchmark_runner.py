"""
Comprehensive Benchmark Runner for BDH Science Fest Sprint
===========================================================

This module runs comprehensive benchmarks on BDH models to collect
performance metrics for before/after comparisons.

Key Benchmarks:
1. Memory retention at t=100, 500, 1000, 2000
2. Training speed (tokens/second)
3. Token efficiency (reduction ratio)
4. Convergence stability
5. Long-context performance
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from bdh_gpu_10m import BDHConfig, BDHGPUTensor
from measurement_logger import (
    PerformanceLogger, TrainingMetrics, MemoryRetentionMetrics,
    StatisticalAnalyzer
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark"""
    benchmark_name: str
    model_name: str
    timestamp: str
    metrics: Dict[str, float]
    raw_data: Optional[Dict] = None


class MemoryRetentionBenchmark:
    """
    Benchmark memory retention capabilities at different time steps.

    Tests:
    - t=100: Short-term memory
    - t=500: Medium-term memory
    - t=1000: Long-term memory
    - t=2000: Extended memory
    """

    def __init__(self, model: BDHGPUTensor, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.logger = PerformanceLogger()

    def generate_test_sequence(self, length: int) -> torch.Tensor:
        """Generate a test sequence with known patterns"""
        # Create a sequence with repeating patterns for memory testing
        pattern = torch.randint(0, 256, (32,))
        sequence = pattern.repeat(length // 32 + 1)[:length]
        return sequence.unsqueeze(0).to(self.device)

    def test_retention_at_step(self, time_step: int) -> MemoryRetentionMetrics:
        """Test memory retention at a specific time step"""
        print(f"\n[MemoryBenchmark] Testing retention at t={time_step}")

        # Generate test sequence
        test_sequence = self.generate_test_sequence(time_step + 100)
        context = test_sequence[:, :time_step]
        target = test_sequence[:, time_step:time_step + 1]

        # Measure retrieval latency
        start_time = time.perf_counter()

        with torch.no_grad():
            # Forward pass with state
            state = None
            for i in range(0, time_step, 256):
                chunk = context[:, i:i+256]
                _, state = self.model(chunk, state=state, return_state=True)

            # Try to predict next token
            logits, _ = self.model(context[:, -256:], state=state, return_state=False)
            predictions = logits[:, -1, :].argmax(dim=-1)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate accuracy (did we predict correctly?)
        accuracy = (predictions == target).float().item()

        # Get config values
        state_decay = self.model.config.state_decay
        hebbian_lr = self.model.config.hebbian_lr

        metrics = MemoryRetentionMetrics(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            time_step=time_step,
            retention_accuracy=accuracy,
            retrieval_latency_ms=latency_ms,
            state_decay_rate=state_decay,
            hebbian_learning_rate=hebbian_lr,
            context_length=time_step
        )

        self.logger.log_memory_retention(metrics)

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Latency: {latency_ms:.2f}ms")

        return metrics

    def run_all_retention_tests(self) -> Dict[int, MemoryRetentionMetrics]:
        """Run all memory retention tests"""
        print("\n" + "="*60)
        print("Running Memory Retention Benchmarks")
        print("="*60)

        time_steps = [100, 500, 1000, 2000]
        results = {}

        for t in time_steps:
            try:
                result = self.test_retention_at_step(t)
                results[t] = result
            except Exception as e:
                print(f"  [ERROR] Failed at t={t}: {e}")

        return results


class TrainingSpeedBenchmark:
    """
    Benchmark training speed and efficiency.
    """

    def __init__(self, model: BDHGPUTensor, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.logger = PerformanceLogger()

    def benchmark_forward_pass(self, batch_size: int, seq_len: int,
                               num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark forward pass speed"""
        print(f"\n[SpeedBenchmark] Forward pass: bs={batch_size}, seq={seq_len}")

        # Create dummy input
        x = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(x)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(x)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        # Calculate metrics
        tokens_per_second = (batch_size * seq_len * num_iterations) / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000

        print(f"  Tokens/sec: {tokens_per_second:,.0f}")
        print(f"  Avg time: {avg_time_ms:.2f}ms")

        return {
            "tokens_per_second": tokens_per_second,
            "avg_time_ms": avg_time_ms,
            "total_time_s": elapsed
        }

    def benchmark_training_step(self, batch_size: int, seq_len: int,
                                num_iterations: int = 50) -> Dict[str, float]:
        """Benchmark full training step (forward + backward)"""
        print(f"\n[SpeedBenchmark] Training step: bs={batch_size}, seq={seq_len}")

        # Create dummy input and target
        x = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)
        target = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(5):
            logits, _ = self.model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, 256), target.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        losses = []
        for _ in range(num_iterations):
            optimizer.zero_grad()

            logits, _ = self.model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, 256), target.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        # Calculate metrics
        tokens_per_second = (batch_size * seq_len * num_iterations) / elapsed
        avg_loss = np.mean(losses)
        loss_std = np.std(losses)

        print(f"  Tokens/sec: {tokens_per_second:,.0f}")
        print(f"  Avg loss: {avg_loss:.4f} ± {loss_std:.4f}")

        return {
            "tokens_per_second": tokens_per_second,
            "avg_loss": avg_loss,
            "loss_std": loss_std,
            "total_time_s": elapsed
        }

    def benchmark_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Benchmark GPU memory usage"""
        print(f"\n[SpeedBenchmark] Memory usage: bs={batch_size}, seq={seq_len}")

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        x = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(x)

        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3

        print(f"  Allocated: {memory_allocated:.2f}GB")
        print(f"  Reserved: {memory_reserved:.2f}GB")

        return {
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved
        }

    def run_all_speed_benchmarks(self) -> Dict[str, Dict]:
        """Run all speed benchmarks"""
        print("\n" + "="*60)
        print("Running Training Speed Benchmarks")
        print("="*60)

        results = {}

        configurations = [
            (8, 128),
            (16, 256),
            (32, 512),
            (8, 1024),
        ]

        for bs, seq in configurations:
            key = f"bs{bs}_seq{seq}"

            # Forward pass
            forward_results = self.benchmark_forward_pass(bs, seq)
            results[f"{key}_forward"] = forward_results

            # Training step
            training_results = self.benchmark_training_step(bs, seq, num_iterations=30)
            results[f"{key}_training"] = training_results

            # Memory
            memory_results = self.benchmark_memory_usage(bs, seq)
            results[f"{key}_memory"] = memory_results

        return results


class ConvergenceBenchmark:
    """
    Benchmark convergence stability and training dynamics.
    """

    def __init__(self, model: BDHGPUTensor, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.logger = PerformanceLogger()

    def benchmark_convergence(self, num_steps: int = 500,
                              batch_size: int = 16,
                              seq_len: int = 256) -> Dict[str, any]:
        """Run short training run to measure convergence"""
        print(f"\n[ConvergenceBenchmark] Running {num_steps} steps")

        # Create synthetic training data
        data = torch.randint(0, 256, (num_steps * batch_size, seq_len)).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        losses = []
        learning_rates = []
        gradient_norms = []
        tokens_per_sec_list = []

        self.model.train()

        for step in range(num_steps):
            start = time.perf_counter()

            # Get batch
            idx = step * batch_size
            x = data[idx:idx + batch_size, :seq_len]
            target = x[:, 1:].clone()
            x_in = x[:, :-1]

            optimizer.zero_grad()

            # Forward pass
            logits, _ = self.model(x_in)
            loss = nn.CrossEntropyLoss()(logits.view(-1, 256), target.view(-1))

            # Backward pass
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            gradient_norms.append(grad_norm.item())

            optimizer.step()
            scheduler.step()

            elapsed = time.perf_counter() - start
            tokens_per_sec = (batch_size * (seq_len - 1)) / elapsed

            losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])
            tokens_per_sec_list.append(tokens_per_sec)

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{num_steps}: loss={loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}, tps={tokens_per_sec:.0f}")

        # Analyze convergence
        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-10:])
        loss_reduction = (initial_loss - final_loss) / initial_loss

        loss_variance = np.var(losses[-100:])
        avg_gradient_norm = np.mean(gradient_norms[-100:])
        avg_tokens_per_sec = np.mean(tokens_per_sec_list[-100:])

        print(f"\n  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss reduction: {loss_reduction:.2%}")
        print(f"  Loss variance (last 100): {loss_variance:.6f}")
        print(f"  Avg gradient norm: {avg_gradient_norm:.4f}")
        print(f"  Avg tokens/sec: {avg_tokens_per_sec:.0f}")

        return {
            "num_steps": num_steps,
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_reduction": float(loss_reduction),
            "loss_variance": float(loss_variance),
            "avg_gradient_norm": float(avg_gradient_norm),
            "avg_tokens_per_second": float(avg_tokens_per_sec),
            "losses": losses,
            "learning_rates": learning_rates,
            "gradient_norms": gradient_norms
        }


class ComprehensiveBenchmarkSuite:
    """
    Main benchmark suite that runs all tests.
    """

    def __init__(self, model: BDHGPUTensor, device: str = "cuda"):
        self.model = model
        self.device = device
        self.logger = PerformanceLogger()
        self.results = {}

    def run_all_benchmarks(self) -> Dict[str, any]:
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("BDH COMPREHENSIVE BENCHMARK SUITE")
        print("="*60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.get_num_params()/1e6:.2f}M")
        print("="*60)

        all_results = {}

        # 1. Memory Retention Benchmarks
        print("\n[1/4] Memory Retention Benchmarks")
        memory_bench = MemoryRetentionBenchmark(self.model, self.device)
        all_results["memory_retention"] = memory_bench.run_all_retention_tests()

        # 2. Training Speed Benchmarks
        print("\n[2/4] Training Speed Benchmarks")
        speed_bench = TrainingSpeedBenchmark(self.model, self.device)
        all_results["training_speed"] = speed_bench.run_all_speed_benchmarks()

        # 3. Convergence Benchmarks
        print("\n[3/4] Convergence Benchmarks")
        conv_bench = ConvergenceBenchmark(self.model, self.device)
        all_results["convergence"] = conv_bench.benchmark_convergence(num_steps=300)

        # 4. Save results
        print("\n[4/4] Saving Results")
        self._save_results(all_results)

        # Generate summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: Dict):
        """Save benchmark results to JSON"""
        results_dir = self.logger.run_dir
        results_file = results_dir / "benchmark_results.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_types(results)

        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def _print_summary(self, results: Dict):
        """Print summary of key metrics"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        # Memory retention
        if "memory_retention" in results:
            print("\nMemory Retention:")
            for t, metrics in results["memory_retention"].items():
                print(f"  t={t}: {metrics.retention_accuracy:.2%} accuracy")

        # Training speed
        if "training_speed" in results:
            print("\nTraining Speed (tokens/sec):")
            for key, value in results["training_speed"].items():
                if "tokens_per_second" in value:
                    print(f"  {key}: {value['tokens_per_second']:,.0f}")

        # Convergence
        if "convergence" in results:
            conv = results["convergence"]
            print(f"\nConvergence:")
            print(f"  Loss reduction: {conv['loss_reduction']:.2%}")
            print(f"  Final loss: {conv['final_loss']:.4f}")
            print(f"  Stability (variance): {conv['loss_variance']:.6f}")

        print("="*60)


def main():
    """Main function to run benchmarks"""
    from measurement_logger import print_system_info
    print_system_info()

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create model
    print("\nCreating BDH-GPU 10M model...")
    config = BDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
    )

    model = BDHGPUTensor(config)

    # Run benchmarks
    suite = ComprehensiveBenchmarkSuite(model, device)
    results = suite.run_all_benchmarks()

    # Export reports
    suite.logger.export_summary_report()
    suite.logger.export_csv()

    print("\n[INFO] Benchmark suite complete!")
    print(f"[INFO] Results directory: {suite.logger.run_dir}")


if __name__ == "__main__":
    main()
