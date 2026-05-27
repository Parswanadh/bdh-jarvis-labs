#!/bin/bash
# RunPod Setup Script for BDH-RD Training
# ========================================
# This script sets up a RunPod L4 GPU pod for training BDH-RD
#
# Usage:
#   1. Create a pod on RunPod with L4 GPU
#   2. Run this script in the pod terminal
#   3. Wait for setup to complete (~5-10 minutes)
#
# Cost: $0.39/hr on Secure Cloud L4 (24GB VRAM)

set -e

echo "========================================"
echo "BDH-RD RunPod Setup Script"
echo "========================================"
echo "GPU: NVIDIA L4 (24GB VRAM)"
echo "Cost: \$0.39/hr (Secure Cloud)"
echo "========================================"

# Update system
echo "[1/10] Updating system..."
apt-get update -y && apt-get upgrade -y

# Install dependencies
echo "[2/10] Installing system dependencies..."
apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    htop \
    nvtop

# Create workspace
echo "[3/10] Creating workspace..."
WORKSPACE="/workspace"
mkdir -p "$WORKSPACE/checkpoints"
mkdir -p "$WORKSPACE/data"

# Install Python dependencies
echo "[4/10] Installing Python dependencies..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch 2.1.0 with CUDA 12.1..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related
pip install transformers==4.38.0
pip install datasets==2.16.0
pip install bitsandbytes==0.41.0
pip install accelerate==0.26.0

# Install additional tools
pip install wandb tensorboard
pip install triton==2.1.0  # For Flash Attention (optional speedup)

# Clone the repository (if using git)
echo "[5/10] Setting up BDH-RD repository..."
cd "$WORKSPACE"

# Copy the project files (these should be uploaded via RunPod volume)
# Or clone from git if available
if [ -d ".git" ]; then
    git pull origin main 2>/dev/null || true
fi

# Verify GPU
echo "[6/10] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check CUDA
echo "[7/10] Checking CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'PyTorch version: {torch.__version__}')"

# Test PyTorch with GPU
echo "[8/10] Testing PyTorch GPU..."
python3 -c "
import torch
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f'GPU test passed! Result shape: {z.shape}')
"

# Setup project structure
echo "[9/10] Setting up project structure..."

# Create implementation symlink
if [ -d "/workspace/BDH/implementation" ]; then
    ln -sf /workspace/BDH/implementation /workspace/implementation 2>/dev/null || true
fi

# Create data directory
echo "[10/10] Creating data directory..."
mkdir -p /workspace/data

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "Ready to train BDH-RD!"
echo ""
echo "Example training commands:"
echo ""
echo "  # Basic training"
echo "  cd /workspace"
echo "  python train_bdh_recurrent.py --config basic --batch_size 64 --epochs 10"
echo ""
echo "  # Large model training"
echo "  python train_bdh_recurrent.py --config large --batch_size 32 --epochs 10"
echo ""
echo "  # Resume from checkpoint"
echo "  python train_bdh_recurrent.py --config basic --resume checkpoints/bdh_recurrent.pt"
echo ""
echo "Monitoring:"
echo "  - GPU usage: nvtop"
echo "  - Train: python -m tensorboard --logdir runs"
echo "  - Check disk: df -h"
echo "  - Check memory: free -h"
echo ""
echo "Cost: \$0.39/hr"
echo "========================================"
