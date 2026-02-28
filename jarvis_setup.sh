#!/bin/bash
# ============================================================================
# Jarvis Labs A100 80GB - Complete Setup Script
# ============================================================================
# This script sets up everything for BDH distillation training

set -e  # Exit on error

echo "============================================================"
echo "JARVIS LABS A100 - BDH DISTILLATION SETUP"
echo "============================================================"
echo ""

# System info
echo "🖥️  System Information:"
nvidia-smi
echo ""
free -h
echo ""
python --version
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y git curl
echo ""

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "📥 Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
else
    echo "✅ Conda already installed"
fi

# Create conda environment
echo ""
echo "🐍 Creating conda environment..."
conda create -n bdh-jarvis python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bdh-jarvis

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install --no-cache-dir \
    numpy \
    tqdm \
    datasets \
    transformers

# Install Ollama
echo ""
echo "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server in background
echo ""
echo "▶️  Starting Ollama server..."
ollama serve &
sleep 10

# Pull Gemma 3 270M model
echo ""
echo "📥 Pulling Gemma 3 270M model..."
ollama pull gemma3:270m

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Generate training data: python generate_teacher_data.py"
echo "2. Train BDH model: python train_a100.py"
echo ""
