# BDH-GPU 10M: Setup and Installation Guide

## Prerequisites

Before running the BDH-GPU implementation, you need to install PyTorch.

---

## Step 1: Install Python Dependencies

### Option A: Using pip (Recommended)

```bash
# For CUDA 12.x (NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU-only (no GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Additional dependencies
pip install numpy matplotlib
```

### Option B: Using conda

```bash
# For CUDA 12.x
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# For CPU-only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

---

## Step 2: Verify Installation

Run the test script to verify everything is working:

```bash
cd D:/projects/BDH
python test_bdh_gpu.py
```

Expected output:
```
============================================================
BDH-GPU 10M - Quick Test
============================================================

1. Checking PyTorch installation...
   PyTorch version: 2.x.x
   CUDA available: True/False
   GPU: [Your GPU model if available]

2. Importing BDH-GPU model...
   ✓ Model imported successfully

3. Creating 10M BDH-GPU model...
BDH-GPU Model initialized
Total parameters: 10.23M
   ✓ Model created with 10.23M parameters

... (more tests)

✓ All tests passed!
```

---

## Step 3: Quick Test Run

Test the model directly:

```bash
python bdh_gpu_10m.py
```

This should:
- Create the model
- Test forward pass
- Generate sample text

---

## Step 4: Prepare Training Data

Create a text file for training:

```bash
# Option 1: Use any text file
echo "Your training text here..." > input.txt

# Option 2: Download a dataset (example: Shakespeare)
curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > input.txt

# Option 3: Use your own data
# Just place your text file as input.txt in the BDH directory
```

---

## Step 5: Train the Model

```bash
python train_bdh_gpu.py
```

### Training Output

You'll see:
```
Creating BDH-GPU 10M model...
BDH-GPU Model initialized
Total parameters: 10.23M
Compiling model with torch.compile...
Model compiled!

============================================================
STARTING TRAINING
============================================================

iter     0 | loss 4.1234 | lr 3.00e-06 | tokens/sec 15000
iter   100 | loss 3.4567 | lr 1.50e-04 | tokens/sec 15234
iter   200 | loss 2.9876 | lr 3.00e-04 | tokens/sec 15123
...
```

---

## Hardware Requirements

### Minimum (CPU Only)
- CPU: Any modern multi-core processor
- RAM: 8GB (16GB recommended)
- Storage: 1GB free space

### Recommended (GPU)
- GPU: NVIDIA GTX 1650 or better
- VRAM: 2GB+ (4GB+ recommended)
- RAM: 16GB
- Storage: 1GB free space

### Ideal (Fast Training)
- GPU: NVIDIA RTX 4060 or better
- VRAM: 8GB+
- RAM: 32GB
- Storage: SSD (faster data loading)

---

## Common Installation Issues

### Issue: "No module named 'torch'"

**Solution:** Install PyTorch first
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: CUDA not available

**Symptoms:** `CUDA available: False`

**Solutions:**
1. Check if you have an NVIDIA GPU
2. Install NVIDIA drivers
3. Install CUDA toolkit (12.x recommended)
4. Reinstall PyTorch with CUDA support

### Issue: Out of Memory during training

**Solutions:**
1. Reduce batch size in `train_bdh_gpu.py`:
   ```python
   batch_size = 16  # Was 32
   ```

2. Reduce sequence length:
   ```python
   max_seq_len = 256  # Was 512
   ```

3. Use smaller model:
   ```python
   n_embd = 128  # Was 256
   n_layer = 4   # Was 6
   ```

### Issue: Slow training on CPU

**Solutions:**
1. Use GPU if available
2. Reduce dataset size
3. Use smaller model
4. Enable torch.compile:
   ```python
   compile = True
   ```

---

## Configuration for Different Setups

### For CPU Training (Slow but works)

```python
config = TrainingConfig(
    batch_size=8,           # Small batch
    max_seq_len=256,        # Shorter sequences
    compile=True,           # Enable compilation
    device="cpu",
    max_iters=1000,         # Fewer iterations for testing
)
```

### For GPU with 2GB VRAM

```python
config = TrainingConfig(
    batch_size=16,
    max_seq_len=512,
    compile=True,
    device="cuda",
    dtype="float16",        # Use FP16 to save memory
)
```

### For GPU with 8GB+ VRAM (Recommended)

```python
config = TrainingConfig(
    batch_size=64,
    max_seq_len=512,
    compile=True,
    device="cuda",
    dtype="bfloat16",       # Best performance
)
```

---

## Quick Reference

### Files Overview

| File | Purpose |
|------|---------|
| `bdh_gpu_10m.py` | Core model implementation (10M params) |
| `train_bdh_gpu.py` | Training script with all features |
| `test_bdh_gpu.py` | Test script to verify installation |
| `README_IMPLEMENTATION.md` | Detailed documentation |
| `SETUP_GUIDE.md` | This file |

### Common Commands

```bash
# Test installation
python test_bdh_gpu.py

# Quick test of model
python bdh_gpu_10m.py

# Train on default data
python train_bdh_gpu.py

# Train with custom data
python train_bdh_gpu.py  # Edit train_bdh_gpu.py to set dataset_path
```

---

## Next Steps After Installation

1. ✅ Verify installation with `test_bdh_gpu.py`
2. ✅ Test model with `bdh_gpu_10m.py`
3. ✅ Train on sample data
4. ✅ Experiment with different configurations
5. ✅ Train on your own data
6. ✅ Analyze the model's internal state
7. ✅ Generate text with your trained model

---

## Getting Help

If you encounter issues:

1. Check error messages carefully
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Try reducing batch size or model size
5. Refer to the main README: `README_IMPLEMENTATION.md`

---

## References

- Original paper: `2509.26507v1.pdf`
- Official repo: https://github.com/pathwaycom/bdh
- PyTorch installation: https://pytorch.org/get-started/locally/

---

**You're all set! Run `python test_bdh_gpu.py` to begin. 🚀**
