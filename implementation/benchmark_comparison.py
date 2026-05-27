"""
Baseline vs Multi-Scale BDH Comparison Script
==============================================

This script compares the baseline BDH implementation with the multi-scale
implementation to demonstrate the memory retention improvement.

Usage:
    python benchmark_comparison.py

Metrics:
- Memory retention at different distances (100, 500, 1000, 2000 tokens)
- State matrix norms over time
- Memory overhead
- Inference speed comparison

Author: Multi-Scale BDH Architect (T1)
Date: 2025-02-25
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Import baseline and multi-scale implementations
import sys
sys.path.append('.')

try:
    from bdh_gpu_10m import BDHConfig, BDHGPUTensor
    from implementation.multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
    print("✓ Both implementations imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure both bdh_gpu_10m.py and implementation/multiscale_bdh.py are available")
    sys.exit(1)


class RetentionBenchmark:
    """
    Benchmark for measuring memory retention in BDH models.

    Tests how well the model retains information at different distances.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

    def create_baseline_model(self) -> BDHGPUTensor:
        """Create baseline BDH model."""
        config = BDHConfig(
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024,
            dropout=0.0,  # No dropout for testing
            state_decay=0.99,  # Single decay rate
        )
        model = BDHGPUTensor(config).to(self.device)
        model.eval()
        return model

    def create_multiscale_model(self) -> MultiScaleBDH:
        """Create multi-scale BDH model."""
        config = MultiScaleBDHConfig(
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024,
            dropout=0.0,  # No dropout for testing
            decay_rates=[0.95, 0.99, 0.995],
            scale_weights=[0.2, 0.3, 0.5],
            hebbian_lr=0.001,
        )
        model = MultiScaleBDH(config).to(self.device)
        model.eval()
        return model

    def measure_retention_baseline(
        self,
        model: BDHGPUTensor,
        context: str,
        distances: List[int] = [100, 500, 1000, 2000]
    ) -> Dict[int, float]:
        """
        Measure baseline BDH retention at different distances.

        Args:
            model: Baseline BDH model
            context: Context string to remember
            distances: List of distances to test

        Returns:
            Dictionary mapping distance to retention score
        """
        retention = {}

        # Convert context to bytes
        context_bytes = context.encode('utf-8')
        max_dist = max(distances)

        # Create long sequence
        full_sequence = list(context_bytes) + [32] * max_dist  # 32 = space
        full_sequence = torch.tensor([full_sequence], dtype=torch.long).to(self.device)

        # Get state after context
        context_len = len(context_bytes)
        context_tokens = full_sequence[:, :context_len]
        _, state = model(context_tokens, return_state=True)

        # Measure initial state norm
        initial_norm = state.norm().item()

        # Process remaining tokens and measure state decay
        for dist in distances:
            tokens_to_process = full_sequence[:, context_len:context_len + dist]

            with torch.no_grad():
                _, state_after = model(tokens_to_process, state=state.clone(), return_state=True)

            final_norm = state_after.norm().item()
            retention[dist] = final_norm / initial_norm if initial_norm > 0 else 0.0

        return retention

    def measure_retention_multiscale(
        self,
        model: MultiScaleBDH,
        context: str,
        distances: List[int] = [100, 500, 1000, 2000]
    ) -> Dict[int, Dict[str, float]]:
        """
        Measure multi-scale BDH retention at different distances.

        Args:
            model: Multi-scale BDH model
            context: Context string to remember
            distances: List of distances to test

        Returns:
            Dictionary mapping distance to dict of retention scores per scale
        """
        retention = {}

        # Convert context to bytes
        context_bytes = context.encode('utf-8')
        max_dist = max(distances)

        # Create long sequence
        full_sequence = list(context_bytes) + [32] * max_dist  # 32 = space
        full_sequence = torch.tensor([full_sequence], dtype=torch.long).to(self.device)

        # Get states after context
        context_len = len(context_bytes)
        context_tokens = full_sequence[:, :context_len]
        _, states = model(context_tokens, return_states=True)

        # Measure initial state norms for each scale
        scale_names = ["fast", "medium", "slow", "combined"]
        initial_norms = []
        for i in range(3):
            initial_norms.append(states[0][i].norm().item())
        # Combined norm
        initial_norms.append(
            sum(0.2 * states[0][0].norm().item(),
                0.3 * states[0][1].norm().item(),
                0.5 * states[0][2].norm().item())
        )

        # Process remaining tokens and measure state decay
        for dist in distances:
            tokens_to_process = full_sequence[:, context_len:context_len + dist]

            with torch.no_grad():
                _, states_after = model(tokens_to_process, states=states, return_states=True)

            dist_retention = {}
            for i, name in enumerate(scale_names):
                if i < 3:
                    final_norm = states_after[0][i].norm().item()
                else:
                    # Combined
                    final_norm = sum(
                        0.2 * states_after[0][0].norm().item(),
                        0.3 * states_after[0][1].norm().item(),
                        0.5 * states_after[0][2].norm().item()
                    )

                dist_retention[name] = final_norm / initial_norms[i] if initial_norms[i] > 0 else 0.0

            retention[dist] = dist_retention

        return retention

    def measure_inference_speed(
        self,
        model,
        seq_len: int = 512,
        batch_size: int = 4,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Measure inference speed.

        Args:
            model: The model to benchmark
            seq_len: Sequence length
            batch_size: Batch size
            num_iterations: Number of iterations

        Returns:
            Dictionary with timing metrics
        """
        model.eval()

        # Create random input
        x = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Measure
        torch.cuda.synchronize() if self.device == "cuda" else None
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)

        torch.cuda.synchronize() if self.device == "cuda" else None
        end = time.time()

        total_time = end - start
        avg_time = total_time / num_iterations
        tokens_per_sec = (batch_size * seq_len * num_iterations) / total_time

        return {
            "total_time": total_time,
            "avg_time_ms": avg_time * 1000,
            "tokens_per_sec": tokens_per_sec,
        }

    def measure_memory_overhead(self) -> Dict[str, float]:
        """
        Measure memory overhead of multi-scale vs baseline.

        Returns:
            Dictionary with memory metrics
        """
        # State matrix size: n_embd × n_embd × 4 bytes (float32)
        n_embd = 256

        baseline_state_size = n_embd * n_embd * 4  # bytes
        multiscale_state_size = baseline_state_size * 3  # Three scales

        return {
            "baseline_state_bytes": baseline_state_size,
            "multiscale_state_bytes": multiscale_state_size,
            "overhead_bytes": multiscale_state_size - baseline_state_size,
            "overhead_kb": (multiscale_state_size - baseline_state_size) / 1024,
        }


def print_comparison_table(
    baseline_retention: Dict[int, float],
    multiscale_retention: Dict[int, Dict[str, float]]
):
    """Print comparison table of retention scores."""
    print("\n" + "=" * 80)
    print("MEMORY RETENTION COMPARISON")
    print("=" * 80)
    print(f"{'Distance':<12} {'Baseline':<12} {'Multi-Scale':<14} {'Improvement':<12}")
    print("-" * 80)

    for dist in sorted(baseline_retention.keys()):
        base = baseline_retention[dist]
        multi = multiscale_retention[dist]["combined"]
        improvement = (multi / base - 1) * 100 if base > 0 else 0

        print(f"{dist:<12} {base:<12.6f} {multi:<14.6f} {improvement:+.1f}%")

    print("-" * 80)

    # Print per-scale breakdown
    print("\nMulti-Scale Breakdown:")
    print(f"{'Distance':<12} {'Fast':<10} {'Medium':<10} {'Slow':<10} {'Combined':<10}")
    print("-" * 60)

    for dist in sorted(multiscale_retention.keys()):
        r = multiscale_retention[dist]
        print(f"{dist:<12} {r['fast']:<10.6f} {r['medium']:<10.6f} {r['slow']:<10.6f} {r['combined']:<10.6f}")


def run_full_benchmark():
    """Run complete benchmark comparison."""
    print("\n" + "=" * 80)
    print("BASELINE VS MULTI-SCALE BDH BENCHMARK")
    print("=" * 80)

    benchmark = RetentionBenchmark()

    # Test context
    context = "The secret code is 12345. Remember this for the test."

    # Measure retention
    print("\n[1/4] Measuring baseline retention...")
    baseline_model = benchmark.create_baseline_model()
    baseline_retention = benchmark.measure_retention_baseline(baseline_model, context)

    print("\n[2/4] Measuring multi-scale retention...")
    multiscale_model = benchmark.create_multiscale_model()
    multiscale_retention = benchmark.measure_retention_multiscale(multiscale_model, context)

    print("\n[3/4] Measuring inference speed...")
    print("\nBaseline speed:")
    baseline_speed = benchmark.measure_inference_speed(baseline_model)
    print(f"  Tokens/sec: {baseline_speed['tokens_per_sec']:.0f}")

    print("\nMulti-scale speed:")
    multiscale_speed = benchmark.measure_inference_speed(multiscale_model)
    print(f"  Tokens/sec: {multiscale_speed['tokens_per_sec']:.0f}")

    speed_overhead = (baseline_speed['tokens_per_sec'] / multiscale_speed['tokens_per_sec'] - 1) * 100
    print(f"\nSpeed overhead: {speed_overhead:+.1f}%")

    print("\n[4/4] Measuring memory overhead...")
    memory = benchmark.measure_memory_overhead()
    print(f"  Baseline state: {memory['baseline_state_bytes'] / 1024:.1f} KB")
    print(f"  Multi-scale state: {memory['multiscale_state_bytes'] / 1024:.1f} KB")
    print(f"  Overhead: {memory['overhead_kb']:.1f} KB")

    # Print comparison
    print_comparison_table(baseline_retention, multiscale_retention)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate average improvement
    improvements = []
    for dist in baseline_retention.keys():
        if baseline_retention[dist] > 0:
            improvement = (multiscale_retention[dist]["combined"] / baseline_retention[dist] - 1) * 100
            improvements.append(improvement)

    avg_improvement = np.mean(improvements) if improvements else 0

    print(f"Average retention improvement: {avg_improvement:.1f}%")
    print(f"Memory overhead: {memory['overhead_kb']:.1f} KB")
    print(f"Speed overhead: {speed_overhead:.1f}%")

    print("\n✓ Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_benchmark()
