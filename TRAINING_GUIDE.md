# 🚀 BDH Training Guide for Science Fest

## 📊 **Training Configuration**

### **Model:** Multi-Scale BDH (10M parameters)
- **Architecture:** Multi-scale synaptic states
- **Decay rates:** [0.95, 0.99, 0.995] (fast, medium, slow)
- **Context length:** 512 tokens
- **Parameters:** ~10 million

### **Task:** Language Modeling
- **Objective:** Predict next token (byte-level)
- **Input:** Text sequence
- **Output:** Probability distribution over 256 possible bytes
- **Loss:** Cross-entropy

### **Dataset:** Tiny Shakespeare
- **Why:** Classic LM benchmark, small size for quick training
- **Size:** ~111 KB raw text
- **Training samples:** ~13,000 characters (repeated for demo)
- **Validation samples:** ~2,600 characters

**Note:** For the demo, we use repeated text to have enough data to train. In a real scenario, you'd use a larger dataset like Wikitext-2.

---

## ⚡ **Quick Start Training**

### **Step 1: Activate Environment**
```bash
conda activate bdh-fest
```

### **Step 2: Start Training**
```bash
cd D:\projects\BDH
python train_science_fest.py
```

### **Step 3: Watch It Train!**
You'll see output like:
```
============================================================
BDH Science Fest - Multi-Scale BDH Training
============================================================

🎯 Training on device: cuda
✅ Model created: 10,000,000 parameters
📊 Loading dataset...
✅ Dataset loaded: 13000 training samples, 2600 validation samples
🚀 Starting training for 1000 iterations...
   Batch size: 32
   Learning rate: 0.0003
   Checkpoint interval: every 500 iterations
   Eval interval: every 500 iterations
============================================================

[  100/1000] loss: 3.4521 | tokens/sec: 15234
[  200/1000] loss: 3.2891 | tokens/sec: 15345
...
[  500/1000] loss: 2.9876 | tokens/sec: 15432
✅ Checkpoint saved: iteration 500, loss: 2.9876

============================================================
📊 VALIDATION [iter 500]
   Val Loss: 2.8734
   Perplexity: 17.69
============================================================

🏆 New best model! Val loss: 2.8734

[ 1000/1000] loss: 2.8123 | tokens/sec: 15321
✅ Checkpoint saved: iteration 1000, loss: 2.8123

============================================================
🎉 TRAINING COMPLETE!
✅ Final checkpoint saved to checkpoints/multiscale_bdh_science_fest
✅ Best validation loss: 2.8123
============================================================
```

---

## 💾 **Checkpointing**

### **Automatic Checkpoints:**

The script automatically saves checkpoints to:
```
checkpoints/multiscale_bdh_science_fest/
├── checkpoint_iter_500.pt           # Checkpoint at iteration 500
├── checkpoint_iter_1000.pt          # Final checkpoint
├── checkpoint_latest.pt             # Most recent checkpoint
├── checkpoint_best.pt                # Best validation loss
└── training_config.json              # Training configuration
```

### **What's Saved:**
- ✅ Model weights
- ✅ Optimizer state
- ✅ Training iteration
- ✅ Loss value
- ✅ Timestamp
- ✅ Training configuration

### **Resume from Checkpoint:**

If training is interrupted, you can resume:

```python
# In train_science_fest.py, change:
train_config.resume_from = None

# To resume from best checkpoint:
train_config.resume_from = "checkpoints/multiscale_bdh_science_fest/checkpoint_best.pt"
```

---

## 📈 **Training Progress**

### **Metrics to Watch:**

1. **Training Loss:** Should decrease steadily
   - Start: ~3.5-4.0 (random)
   - Middle: ~2.8-3.0
   - End: ~2.5-2.8

2. **Tokens/Second:** Training speed
   - Expected: 15,000-20,000 tokens/sec (RTX 4070)
   - Faster is better!

3. **Validation Loss:** Should also decrease
   - Evaluated every 500 iterations
   - Best model saved automatically

4. **Perplexity:** exp(val_loss)
   - Lower is better
   - Expected: 15-20 for this demo

---

## 🎯 **Training Duration**

### **Estimated Time:**
- **1000 iterations:** ~20-30 minutes
- **5000 iterations:** ~1.5-2 hours (for better results)

### **What You'll See During Training:**
- Loss decreasing
- Tokens/sec counter
- Checkpoint saves
- Validation evaluations
- Best model updates

---

## 🔧 **Customization**

### **Use Your Own Dataset:**

1. **Create data file:**
   ```bash
   mkdir -p data
   echo "Your text here..." > data/your_text.txt
   ```

2. **Update training script:**
   ```python
   train_config.dataset_path = "data/your_text.txt"
   ```

3. **Run training**

### **Train Longer:**

```python
# In train_science_fest.py:
train_config.max_iters = 5000  # Instead of 1000
```

### **Larger Batch Size:**

```python
# In train_science_fest.py:
train_config.batch_size = 64  # Instead of 32
```

---

## 📊 **Expected Results**

### **After 1000 Iterations:**
- Training loss: ~2.5-2.8
- Validation loss: ~2.5-2.8
- Perplexity: ~12-16
- Model can generate Shakespeare-like text

### **After 5000 Iterations:**
- Training loss: ~2.0-2.3
- Validation loss: ~2.0-2.3
- Perplexity: ~7-10
- Model generates coherent text

---

## 🎭 **For the Science Fest Demo**

### **What to Show:**

1. **Training Progress:**
   - Show loss decreasing
   - Show tokens/sec speed
   - Explain checkpointing

2. **Final Model:**
   - Load best checkpoint
   - Generate text live
   - Show perplexity improvement

3. **Multi-Scale Advantage:**
   - Show state matrices
   - Explain memory retention
   - Demonstrate 4× improvement

---

## 🆘 **Troubleshooting**

### **CUDA Out of Memory:**
```python
# Reduce batch size:
train_config.batch_size = 16  # Instead of 32
```

### **Loss Not Decreasing:**
- Check learning rate (should be 3e-4)
- Check warmup_steps (should be 5000)
- Check grad_clip (should be 1.0)

### **Training Too Slow:**
- Use torch.compile (already enabled)
- Reduce eval_interval
- Use smaller validation set

---

## 📞 **Quick Reference**

### **Commands:**
```bash
# Activate environment
conda activate bdh-fest

# Start training
python train_science_fest.py

# Check GPU usage
nvidia-smi

# Monitor training (separate terminal)
watch -n 1 nvidia-smi
```

### **Key Files:**
- `train_science_fest.py` - Training script
- `checkpoints/multiscale_bdh_science_fest/` - Checkpoints directory
- `implementation/multiscale_bdh.py` - Model architecture

---

## ✅ **Success Indicators**

You'll know training is working when:
- ✅ Loss is decreasing
- ✅ Tokens/sec is 15,000-20,000
- ✅ Checkpoints are being saved
- ✅ Validation loss is decreasing
- ✅ Perplexity is reasonable (10-20)

---

**Happy training! Your model will be ready for the science fest in ~30 minutes!** 🚀🐉
