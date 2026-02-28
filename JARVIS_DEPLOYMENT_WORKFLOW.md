# Jarvis Labs Deployment - Complete Workflow

This guide shows the complete workflow from training on Jarvis Labs to downloading the model locally.

---

## Overview

```
Jarvis Labs A100 (Train) → GitHub (Storage) → Your Local Machine (Use)
```

---

## Step 1: Deploy to Jarvis Labs & Train

SSH into Jarvis Labs and run:

```bash
# Clone repository
git clone https://github.com/Parswanadh/bdh-jarvis-labs
cd bdh-jarvis-labs

# Make scripts executable
chmod +x run_jarvis_training.sh

# Run complete pipeline (automated)
./run_jarvis_training.sh
```

**What happens:**
- ✅ Installs Miniconda, PyTorch, Ollama (5 min)
- ✅ Generates training data with Gemma 3 270M (20-30 min)
- ✅ Trains Multi-Scale BDH for 5 epochs (30-40 min)
- ✅ Saves checkpoints to `checkpoints/bdh_a100/`

**Total time:** ~75 minutes

---

## Step 2: Upload to GitHub (From Jarvis Labs)

After training completes on Jarvis Labs:

```bash
# Upload trained model to GitHub
chmod +x upload_to_github.sh
./upload_to_github.sh
```

**What happens:**
- ✅ Commits all checkpoints to git
- ✅ Pushes to GitHub repository
- ✅ Model available from anywhere

**Files uploaded:**
```
checkpoints/bdh_a100/
├── checkpoint_iter_1.pt
├── checkpoint_iter_2.pt
├── checkpoint_iter_3.pt
├── checkpoint_iter_4.pt
├── checkpoint_iter_5.pt
├── checkpoint_best.pt   ← Use this one!
└── checkpoint_latest.pt
```

---

## Step 3: Download Locally

On your local machine:

```bash
# Clone repository (or pull if already cloned)
git clone https://github.com/Parswanadh/bdh-jarvis-labs
cd bdh-jarvis-labs

# Pull latest changes (including checkpoints)
git pull origin master

# Run download script (optional, verifies download)
chmod +x download_from_github.sh
./download_from_github.sh
```

**What happens:**
- ✅ Downloads all checkpoints from GitHub
- ✅ Verifies file integrity
- ✅ Shows checkpoint info

---

## Step 4: Test Model Locally

```bash
# Test the downloaded model
python test_downloaded_model.py
```

**What happens:**
- ✅ Loads checkpoint_best.pt
- ✅ Displays model info (loss, epoch, parameters)
- ✅ Generates sample text

**Or use in your code:**

```python
from implementation.multiscale_bdh import MultiScaleBDH
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/bdh_a100/checkpoint_best.pt')

# Recreate model
model = MultiScaleBDH(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
text = model.generate("The future of AI is", max_tokens=100)
print(text)
```

---

## Quick Reference

### Jarvis Labs Commands

```bash
# Train
./run_jarvis_training.sh

# Upload after training
./upload_to_github.sh
```

### Local Machine Commands

```bash
# Download model
git pull origin master
./download_from_github.sh

# Test model
python test_downloaded_model.py
```

---

## File Locations

**On Jarvis Labs (after training):**
```
~/bdh-jarvis-labs/checkpoints/bdh_a100/checkpoint_best.pt
```

**On GitHub (after upload):**
```
https://github.com/Parswanadh/bdh-jarvis-labs/tree/master/checkpoints/bdh_a100
```

**On your local machine (after download):**
```
bdh-jarvis-labs/checkpoints/bdh_a100/checkpoint_best.pt
```

---

## Troubleshooting

### Checkpoints not uploading to GitHub

**Problem:** Large files (>50MB) may show warnings but still upload

**Solution:** GitHub allows files up to 100MB. You'll see warnings but upload succeeds.

### Download is slow

**Problem:** Large checkpoint files (~56MB each × 7 files = ~400MB total)

**Solution:** Use `git pull` instead of `git clone` if you already have the repo.

### Model not loading locally

**Problem:** Missing dependencies or wrong path

**Solution:**
```bash
# Install dependencies
pip install torch numpy

# Verify checkpoint exists
ls checkpoints/bdh_a100/

# Test with verbose output
python test_downloaded_model.py checkpoints/bdh_a100/checkpoint_best.pt
```

---

## Success Criteria

✅ **Training completes on Jarvis Labs** (loss < 2.5)
✅ **Checkpoints uploaded to GitHub** (all 7 files)
✅ **Model downloaded locally** (~400MB)
✅ **Test generation works** (generates coherent text)
✅ **Ready for science fair demo!**

---

**Repository:** https://github.com/Parswanadh/bdh-jarvis-labs
**Training Time:** ~75 minutes on Jarvis Labs A100
**Cost:** ~₹100-125
**Model Size:** ~5M parameters
