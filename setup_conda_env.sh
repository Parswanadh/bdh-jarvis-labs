#!/bin/bash
# BDH Science Fest - PyTorch Installation Script (Linux/Mac)
# This script creates a conda environment and installs PyTorch for BDH development

echo "========================================"
echo "BDH Science Fest - Environment Setup"
echo "========================================"
echo

echo "Step 1: Creating conda environment 'bdh-fest'..."
conda create -n bdh-fest python=3.10 -y
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi
echo
echo "✅ Environment 'bdh-fest' created successfully!"
echo

echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bdh-fest
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate environment"
    exit 1
fi
echo
echo "✅ Environment activated!"
echo

echo "Step 3: Installing PyTorch with CUDA 12.4 support..."
echo "(This may take a few minutes...)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi
echo
echo "✅ PyTorch installed successfully!"
echo

echo "Step 4: Installing additional dependencies..."
pip install tokenizers matplotlib seaborn plotly numpy jupyter
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo
echo "✅ All dependencies installed!"
echo

echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if [ $? -ne 0 ]; then
    echo "WARNING: Could not verify PyTorch installation"
fi
echo

echo "========================================"
echo "✅ SETUP COMPLETE!"
echo "========================================"
echo
echo "To activate the environment, run:"
echo "  conda activate bdh-fest"
echo
echo "PyTorch is ready for BDH development!"
echo
