# Jarvis Labs Quick Start - Pure Distillation

## You're on Jarvis Labs - Let's Train!

### Option 1: Pure Knowledge Distillation (RECOMMENDED ✅)

**TRUE distillation where teacher guides student during training**

```bash
# 1. Pull latest code
cd ~/bdh-jarvis-labs
git pull origin master

# 2. Setup environment
chmod +x setup_pure_distillation.sh
./setup_pure_distillation.sh

# 3. Activate environment
conda activate bdh-pure

# 4. Run training
python train_pure_distillation.py
```

**What happens:**
- ✅ Downloads Gemma 3 1B model (teacher)
- ✅ Downloads TinyStories dataset
- ✅ Trains BDH with pure distillation (KL divergence)
- ✅ Student learns from teacher's probability distributions

**Time:** ~60 minutes
**Result:** True knowledge distillation (student learned "how to think")

---

### Option 2: Simple TinyStories Training

**Direct training on dataset (simpler, faster)**

```bash
# 1. Pull latest code
cd ~/bdh-jarvis-labs
git pull origin master

# 2. Activate existing environment
conda activate base  # or bdh-fest

# 3. Train on TinyStories
python train_a100.py
```

**What happens:**
- ✅ Trains BDH on TinyStories dataset
- ✅ Standard language modeling
- ✅ Simpler approach

**Time:** ~30-40 minutes
**Result:** Good language model (but not true distillation)

---

## Which Should You Choose?

### Choose Pure Distillation (Option 1) if:
- ✅ Want TRUE knowledge distillation
- ✅ Want to claim "student learned from teacher"
- ✅ Want state-of-the-art technique
- ✅ Have 60 minutes
- ✅ Want best science fair project

### Choose Simple Training (Option 2) if:
- ✅ Want simple, working solution
- ✅ Have limited time (40 min)
- ✅ OK with standard language modeling
- ✅ Don't need to claim distillation

---

## Recommendation: **Pure Distillation (Option 1)**

For your science fair, pure distillation is **much better** because:

1. **Honest claim:** "We used knowledge distillation" (TRUE)
2. **Technical depth:** Shows understanding of KL divergence, logits, probability distributions
3. **Impressive:** 200x compression while preserving knowledge
4. **State-of-the-art:** Using cutting-edge technique

---

## After Training Completes

### Upload to GitHub

```bash
# Upload trained model
chmod +x upload_to_github.sh
./upload_to_github.sh
```

### Download Locally

```bash
# On your local machine
git clone https://github.com/Parswanadh/bdh-jarvis-labs
cd bdh-jarvis-labs
git pull origin master
```

Your trained model will be in:
```
checkpoints/bdh_pure_distillation/checkpoint_best.pt
```

---

## Quick Reference

**Pure Distillation Commands:**
```bash
git pull
chmod +x setup_pure_distillation.sh
./setup_pure_distillation.sh
conda activate bdh-pure
python train_pure_distillation.py
```

**Simple Training Commands:**
```bash
git pull
conda activate base
python train_a100.py
```

**Upload After Training:**
```bash
./upload_to_github.sh
```

---

## Need Help?

Check these files:
- `PURE_DISTILLATION_GUIDE.md` - Detailed technical guide
- `JARVIS_DEPLOYMENT_WORKFLOW.md` - Complete workflow
- `USAGE_GUIDE.md` - General usage guide

---

**Ready to train! Run Option 1 for pure distillation. 🚀**
