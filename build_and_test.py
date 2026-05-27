"""
BDH Science Fest - Full Build and Test Script

This script orchestrates the complete build, test, and benchmark process
for all BDH improvements (multi-scale, BBPE, stable training).

Run this after activating the bdh-fest conda environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step_num, text):
    print(f"{Colors.OKCYAN}[STEP {step_num}] {Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def run_command(cmd, description, critical=True):
    """Run a command and return success status"""
    print_step(cmd, f"Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print_success(f"{description} - SUCCESS")
            if result.stdout:
                print(result.stdout[:500])  # Print first 500 chars
            return True
        else:
            print_error(f"{description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            if critical:
                sys.exit(1)
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{description} - TIMEOUT (5 minutes)")
        if critical:
            sys.exit(1)
        return False
    except Exception as e:
        print_error(f"{description} - ERROR: {e}")
        if critical:
            sys.exit(1)
        return False

def main():
    """Main build and test orchestration"""

    print_header("BDH SCIENCE FEST - FULL BUILD AND TEST")

    # Change to project directory
    project_dir = Path("D:/projects/BDH")
    os.chdir(project_dir)
    print_success(f"Working directory: {os.getcwd()}")

    results = {
        "environment_check": False,
        "tokenizer_train": False,
        "multiscale_test": False,
        "stable_config_test": False,
        "benchmark_run": False,
        "visualization_generate": False
    }

    # ========================================================================
    # STEP 1: Environment Check
    # ========================================================================
    print_step(1, "Checking Python Environment")
    try:
        import torch
        print_success(f"PyTorch {torch.__version__} installed")
        print_success(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_success(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        results["environment_check"] = True
    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        print_warning("Please run setup_conda_env.bat first")
        sys.exit(1)

    # ========================================================================
    # STEP 2: Train BBPE Tokenizer
    # ========================================================================
    print_step(2, "Training BBPE Tokenizer")

    # Check if tokenizer already trained
    tokenizer_path = project_dir / "tokenizer-model" / "tokenizer.json"
    if tokenizer_path.exists():
        print_success("Tokenizer already trained - skipping")
        results["tokenizer_train"] = True
    else:
        # Create sample training data
        sample_data = project_dir / "paper_text.txt"
        if not sample_data.exists():
            print_warning("Sample data not found, creating from BDH paper...")
            # Use existing text from limitations document
            source_file = project_dir / "BDH_COMPLETE_LIMITATIONS_AND_SOLUTIONS.md"
            if source_file.exists():
                import shutil
                shutil.copy(source_file, sample_data)
                print_success(f"Created training data from {source_file.name}")

        # Train tokenizer
        success = run_command(
            f'python implementation/train_tokenizer_v2.py --input {sample_data} --output tokenizer-model',
            "Train BBPE Tokenizer",
            critical=False
        )
        results["tokenizer_train"] = success

    # ========================================================================
    # STEP 3: Test Multi-Scale BDH Implementation
    # ========================================================================
    print_step(3, "Testing Multi-Scale BDH Implementation")

    try:
        sys.path.insert(0, str(project_dir / "implementation"))
        from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

        # Create config
        config = MultiScaleBDHConfig(
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            decay_rates=[0.95, 0.99, 0.995]
        )

        # Create model
        model = MultiScaleBDH(config)
        print_success("Multi-Scale BDH model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Decay rates: {config.decay_rates}")

        # Test forward pass
        import torch
        batch_size, seq_len = 2, 128
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, state = model(x, return_state=True)
        print_success(f"Forward pass successful - Output shape: {logits.shape}")
        print(f"  State matrices: {len(state)} (one per scale)")

        results["multiscale_test"] = True

    except Exception as e:
        print_error(f"Multi-scale test failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # STEP 4: Test Stable Training Configuration
    # ========================================================================
    print_step(4, "Testing Stable Training Configuration")

    try:
        from stable_config import BDHStableTrainingConfig

        # Create stable config for 10M model
        config = BDHStableTrainingConfig.for_10M_model()
        print_success("Stable training config created")
        print(f"  Initializer range: {config.initializer_range}")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Gradient clip: {config.gradient_clip}")

        results["stable_config_test"] = True

    except Exception as e:
        print_error(f"Stable config test failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # STEP 5: Run Quick Benchmark (Test Mode)
    # ========================================================================
    print_step(5, "Running Benchmarks (Test Mode)")

    try:
        # Run test mode benchmarks (fast, synthetic data)
        success = run_command(
            'python benchmarking/run_benchmarks.py --test-mode',
            "Run Test Benchmarks",
            critical=False
        )
        results["benchmark_run"] = success

    except Exception as e:
        print_error(f"Benchmark failed: {e}")

    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print_step(6, "Generating Visualizations")

    try:
        success = run_command(
            'python visualization/visualize_results.py --test-mode',
            "Generate Visualizations",
            critical=False
        )
        results["visualization_generate"] = success

    except Exception as e:
        print_error(f"Visualization failed: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("BUILD AND TEST SUMMARY")

    print("\nResults:")
    print("-" * 60)
    for step, success in results.items():
        status = f"{Colors.OKGREEN}✅ PASS{Colors.ENDC}" if success else f"{Colors.FAIL}❌ FAIL{Colors.ENDC}"
        print(f"  {step}: {status}")
    print("-" * 60)

    total_steps = len(results)
    passed_steps = sum(results.values())
    print(f"\nTotal: {passed_steps}/{total_steps} steps passed")

    if passed_steps == total_steps:
        print_success("\n🎉 ALL TESTS PASSED! BDH is ready for the science fest!")
        return 0
    else:
        print_warning("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
