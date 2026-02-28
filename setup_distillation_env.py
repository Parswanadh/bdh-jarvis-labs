"""
BDH + Gemma 3 270M Distillation Setup
======================================

This script sets up the environment for knowledge distillation where:
- Teacher: Gemma 3 270M (via Hugging Face)
- Student: Multi-Scale BDH

Install required dependencies.
"""

import subprocess
import sys

def install_dependencies():
    """Install required packages for distillation."""

    print("="*60)
    print("📦 Installing Distillation Dependencies")
    print("="*60)
    print()

    packages = [
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
    ]

    for package in packages:
        print(f"📥 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed")
        print()

    print("="*60)
    print("✅ All dependencies installed!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Get Hugging Face token with Gemma access")
    print("2. Run: python generate_teacher_data.py")
    print("3. Run: python train_distillation.py")
    print()

if __name__ == "__main__":
    install_dependencies()
