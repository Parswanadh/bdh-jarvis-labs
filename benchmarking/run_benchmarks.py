"""
Quick Test Runner for BDH Benchmark Suite
==========================================

This script runs a quick test of the benchmark suite to verify
everything is working correctly before running the full benchmarks.

Usage:
    python benchmarking/run_benchmarks.py --test-mode
    python benchmarking/run_benchmarks.py --full-mode
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_suite import (
    BDHBenchmarkSuite,
    BenchmarkConfig,
    create_baseline_config,
    create_multiscale_config,
    BDHConfig,
    BDHGPUTensor,
)


def create_test_models(config: BenchmarkConfig):
    """Create test models for benchmarking."""
    models = {}

    # Baseline model
    baseline_cfg = BDHConfig(
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
    models["baseline"] = BDHGPUTensor(baseline_cfg)

    # Multi-scale model (slower decay)
    multiscale_cfg = BDHConfig(
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
    models["multiscale"] = BDHGPUTensor(multiscale_cfg)

    return models


def run_test_mode():
    """Run quick test mode with reduced benchmarks."""
    print("="*60)
    print("BDH BENCHMARK SUITE - TEST MODE")
    print("="*60)
    print("Running quick verification tests...")
    print()

    config = BenchmarkConfig(
        baseline_config=create_baseline_config(),
        improved_configs={"multiscale": create_multiscale_config()},
        sequence_lengths=[100, 500, 1000],  # Reduced for testing
        num_runs=2,  # Reduced for testing
        batch_size=16,  # Smaller batch for testing
        device="auto",
    )

    suite = BDHBenchmarkSuite(config)

    # Add models
    models = create_test_models(config)
    for name, model in models.items():
        suite.add_model(name, model)

    # Run benchmarks in fast mode
    results = suite.run_all_benchmarks(fast_mode=True)

    # Save results
    suite.save_results(
        os.path.join(config.results_dir, "test_results.json")
    )

    # Generate comparison report
    suite.generate_comparison_report()

    print("\n" + "="*60)
    print("TEST MODE COMPLETE")
    print("="*60)
    print("\nTo run full benchmarks:")
    print("  python benchmarking/run_benchmarks.py --full-mode")


def run_full_mode():
    """Run full benchmark suite."""
    print("="*60)
    print("BDH BENCHMARK SUITE - FULL MODE")
    print("="*60)
    print("Running complete benchmark suite...")
    print("This will take approximately 30-60 minutes.")
    print()

    config = BenchmarkConfig(
        baseline_config=create_baseline_config(),
        improved_configs={"multiscale": create_multiscale_config()},
        sequence_lengths=[100, 500, 1000, 2000],
        num_runs=3,
        batch_size=32,
        max_iters=1000,
        device="auto",
    )

    suite = BDHBenchmarkSuite(config)

    # Add models
    models = create_test_models(config)
    for name, model in models.items():
        suite.add_model(name, model)

    # Run benchmarks
    results = suite.run_all_benchmarks(fast_mode=False)

    # Save results
    suite.save_results()

    # Generate comparison report
    suite.generate_comparison_report()

    print("\n" + "="*60)
    print("FULL BENCHMARK SUITE COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {config.results_dir}")
    print("\nNext steps:")
    print("  1. Review results in benchmarking/results/")
    print("  2. Check comparison_report.md for summary")
    print("  3. Use results for visualization and presentation")


def main():
    parser = argparse.ArgumentParser(description="BDH Benchmark Suite Runner")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run quick test mode (reduced benchmarks)"
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Run full benchmark suite"
    )

    args = parser.parse_args()

    if args.test_mode:
        run_test_mode()
    elif args.full_mode:
        run_full_mode()
    else:
        print("Please specify --test-mode or --full-mode")
        print("\nQuick test (recommended first):")
        print("  python benchmarking/run_benchmarks.py --test-mode")
        print("\nFull benchmarks:")
        print("  python benchmarking/run_benchmarks.py --full-mode")


if __name__ == "__main__":
    main()
