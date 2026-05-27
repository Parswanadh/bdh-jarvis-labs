#!/bin/bash
echo "=========================================="
echo "🚀 JARVIS LABS L4 TRAINING LAUNCHER"
echo "=========================================="

# Ensure we are in the right directory
cd "$(dirname "$0")"

# Install missing dependencies for 4-bit quantization and speed
echo "[1/2] Installing required packages for L4 GPU..."
pip install -q transformers torch accelerate bitsandbytes sentencepiece scipy

echo "[2/2] Launching Massive Parallel Training Script..."
# We run the python script. It will auto-stop after 4.8 hours to safely save before the 5 hour limit.
python jarvis_l4_super_bdh.py

echo "=========================================="
echo "✅ TRAINING COMPLETE OR TIMELIMIT REACHED."
echo "=========================================="
