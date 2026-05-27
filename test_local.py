"""
Local Test Script - Verify Everything Works
=============================================

This script tests all components locally before deploying to Jarvis Labs.
"""

import subprocess
import sys
from pathlib import Path

def test_component(name, command):
    """Test a component by running a command."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )

        if result.returncode == 0:
            print(f"[PASS] {name} - PASSED")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}")
            return True
        else:
            print(f"[FAIL] {name} - FAILED")
            print(f"   Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} - TIMEOUT")
        return False
    except Exception as e:
        print(f"[ERROR] {name} - ERROR: {e}")
        return False


def main():
    """Run all tests."""

    print("="*60)
    print("LOCAL TEST SUITE - JARVIS LABS PREPARATION")
    print("="*60)

    results = []

    # Test 1: Python availability
    results.append(test_component(
        "Python 3.10",
        ["python", "--version"]
    ))

    # Test 2: Required packages
    results.append(test_component(
        "PyTorch",
        ["python", "-c", "import torch; print(f'PyTorch {torch.__version__}')"]
    ))

    # Test 3: BDH implementation
    results.append(test_component(
        "BDH Implementation",
        ["python", "-c", "import sys; sys.path.insert(0, 'implementation'); from multiscale_bdh import MultiScaleBDH; print('BDH OK')"]
    ))

    # Test 4: Ollama (if available)
    try:
        ollama_running = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5
        )
        if ollama_running.returncode == 0:
            print(f"\n{'='*60}")
            print("Testing: Ollama")
            print('='*60)
            print("[OK] Ollama is running")
            if "gemma3:270m" in ollama_running.stdout:
                print("[OK] Gemma 3 270M is available")
                results.append(True)
            else:
                print("[WARN] Gemma 3 270M not found (run: ollama pull gemma3:270m)")
                results.append(True)
        else:
            print(f"\n{'='*60}")
            print("Testing: Ollama")
            print('='*60)
            print("[WARN] Ollama not running (will run on Jarvis Labs)")
            results.append(True)
    except:
        print(f"\n{'='*60}")
        print("Testing: Ollama")
        print('='*60)
        print("[WARN] Ollama not found (will install on Jarvis Labs)")
        results.append(True)

    # Test 5: GPU availability
    print(f"\n{'='*60}")
    print("Testing: GPU/CUDA")
    print('='*60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[WARN] CUDA not available (using CPU)")
        results.append(True)
    except Exception as e:
        print(f"[WARN] Could not check GPU: {e}")
        results.append(True)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("\n[PASS] ALL TESTS PASSED!")
        print("\nReady for Jarvis Labs deployment!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Jarvis Labs BDH distillation pipeline'")
        print("3. git push")
        print("4. Clone in Jarvis Labs and run: ./jarvis_setup.sh")
    else:
        print("\n[WARN] Some tests failed - fix before deploying")

    print("="*60)


if __name__ == "__main__":
    main()
