# ⚡ QUICK SETUP INSTRUCTIONS

## 🚀 **Option 1: Double-Click Setup (Easiest)**

1. **Open File Explorer**
2. **Navigate to:** `D:\projects\BDH`
3. **Double-click:** `setup_pytorch.bat`
4. **Wait 5-10 minutes** for installation
5. **Done!** ✅

---

## 🔧 **Option 2: Manual Setup (If Script Fails)**

### **Step 1: Open Anaconda Prompt**
- Press Windows key
- Type "Anaconda Prompt"
- Press Enter

### **Step 2: Create Environment**
```bash
conda create -n bdh-fest python=3.10 -y
```

### **Step 3: Install PyTorch**
```bash
conda run -n bdh-fest pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Wait 5-10 minutes** ⏳

### **Step 4: Install Dependencies**
```bash
conda run -n bdh-fest pip install tokenizers matplotlib seaborn plotly numpy jupyter
```

### **Step 5: Verify Installation**
```bash
conda run -n bdh-fest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
```

---

## ✅ **After Installation**

### **Activate Environment:**
```bash
conda activate bdh-fest
```

### **Run Build & Test:**
```bash
python build_and_test.py
```

---

## 🚨 **Troubleshooting**

### **"conda not recognized"**
**Solution:** Use "Anaconda Prompt" from Start Menu (not regular CMD)

### **"CUDA not available"**
**Solution:** Update NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx

### **"Network timeout"**
**Solution:** Check internet connection, run setup again

### **"Out of memory"**
**Solution:** Close other applications, try again

---

## 📞 **What This Does**

✅ Creates conda environment `bdh-fest`
✅ Installs PyTorch with CUDA 12.4 support
✅ Installs tokenizers library
✅ Installs matplotlib, seaborn, plotly
✅ Installs numpy, jupyter
✅ Verifies installation

---

## ⏱️ **Time Estimate**

- **Download:** 2-5 minutes (depends on internet)
- **Installation:** 3-5 minutes
- **Total:** 5-10 minutes

---

## 🎯 **Success Indicator**

When you see:
```
PyTorch: 2.x.x
CUDA: True
```

**You're ready to run the build!** 🎉

---

## 🚀 **Next Steps**

After successful installation:

1. **Activate environment:**
   ```bash
   conda activate bdh-fest
   ```

2. **Run full build:**
   ```bash
   python build_and_test.py
   ```

3. **Check results** in `visualization/` folder

4. **Review benchmarks** in `benchmarking/results/`

---

**Good luck! The setup should take 5-10 minutes.** 🚀
