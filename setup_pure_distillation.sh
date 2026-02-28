#!/bin/bash
# ============================================================================
# SETUP FOR PURE KNOWLEDGE DISTILLATION
# ============================================================================
# Installs dependencies for true knowledge distillation with Gemma 3

set -e

echo "============================================================"
echo "PURE DISTILLATION SETUP"
echo "============================================================"
echo ""

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Please install Miniconda first."
    exit 1
fi

echo "[OK] Conda found"

# Create/update conda environment
echo ""
echo "============================================================"
echo "STEP 1: Setting up conda environment"
echo "============================================================"
echo ""

if conda env list | grep -q "bdh-pure"; then
    echo "[INFO] Environment 'bdh-pure' exists, updating..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bdh-pure
else
    echo "[INFO] Creating new environment 'bdh-pure'..."
    conda create -n bdh-pure python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bdh-pure
fi

echo "[OK] Environment activated: bdh-pure"
echo ""

# Install PyTorch with CUDA
echo "============================================================"
echo "STEP 2: Installing PyTorch with CUDA"
echo "============================================================"
echo ""

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[OK] PyTorch installed"
echo ""

# Install HuggingFace libraries
echo "============================================================"
echo "STEP 3: Installing HuggingFace Transformers"
echo "============================================================"
echo ""

pip install transformers datasets accelerate

echo "[OK] HuggingFace libraries installed"
echo ""

# Install other dependencies
echo "============================================================"
echo "STEP 4: Installing other dependencies"
echo "============================================================"
echo ""

pip install numpy matplotlib

echo "[OK] All dependencies installed"
echo ""

# Verify installation
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"
echo ""

python -c "
import torch
import transformers
print(f'[OK] PyTorch: {torch.__version__}')
print(f'[OK] Transformers: {transformers.__version__}')
print(f'[OK] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[OK] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[OK] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Environment: bdh-pure"
echo ""
echo "Next steps:"
echo "  conda activate bdh-pure"
echo "  python train_pure_distillation.py"
echo ""
echo "This will:"
echo "  1. Download Gemma 3 1B model (teacher)"
echo "  2. Download TinyStories dataset"
echo "  3. Train BDH with PURE knowledge distillation"
echo ""
echo "Training time: ~45-60 minutes on A100"
echo "============================================================"
