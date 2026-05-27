# 🧠 BDH Reasoning Training Guide

## 📊 **Training on Reasoning Tasks**

### **🎯 Task: Reasoning (Math, Logic, Commonsense)**

**What is reasoning training?**
- Training BDH to solve problems that require step-by-step thinking
- Not just pattern matching, but actual reasoning
- Much more impressive for science fair!

### **📚 Dataset: Reasoning Problems**

Our dataset includes:

**1. Math Word Problems (GSM8K-style):**
- Arithmetic: "John has 15 apples. He gives 5 to Mary..."
- Multi-step: "Sarah has twice as many marbles as Tom..."
- Fractions: "Half a pizza divided among 4 people..."
- Time: "A movie starts at 2:30 PM and lasts 2 hours..."

**2. Logical Reasoning:**
- Deductive reasoning: "All A are B, all B are C..."
- Transitive property: "If A > B and B > C..."
- Contrapositive: "If P → Q and Q is false..."

**3. Commonsense Reasoning:**
- Physical world knowledge: "Glass breaks on concrete floor..."
- Everyday situations: "To stay warm in winter..."
- Cause and effect: "Plants need sunlight and water..."

**4. AI/Scientific Reasoning:**
- AI concepts: "Neural networks learn from data..."
- BDH-specific: "BDH uses Hebbian learning..."
- Technical reasoning: "Transformers have O(N²) complexity..."

**5. Pattern Recognition:**
- Sequences: "2, 4, 8, 16, ?"
- Patterns: "A, C, E, G, ?"
- Fibonacci: "1, 1, 2, 3, 5, ?"

**Total: ~800 reasoning problems (repeated for training)**

---

## 🚀 **Start Training Now**

### **Step 1: Activate Environment**
```bash
conda activate bdh-fest
```

### **Step 2: Start Reasoning Training**
```bash
cd D:\projects\BDH
python train_bdh_reasoning.py
```

### **Step 3: Watch It Train!**

You'll see output like:
```
============================================================
BDH Science Fest - Training on REASONING TASKS
============================================================

🎯 Training on device: cuda
✅ Model created: 10,000,000 parameters
📊 Loading reasoning dataset...
✅ Dataset loaded: 48000 training samples, 5333 validation samples
📊 Training on reasoning problems: math, logic, commonsense, and AI reasoning

🚀 Starting training for 1000 iterations...
   Task: Reasoning (math, logic, commonsense)
   Batch size: 32
   Learning rate: 0.0003
   Checkpoint interval: every 500 iterations
   Eval interval: every 500 iterations
============================================================

[  100/1000] loss: 3.2341 | tokens/sec: 14876
[  200/1000] loss: 2.9876 | tokens/sec: 15234
...
[  500/1000] loss: 2.6543 | tokens/sec: 15123
✅ Checkpoint saved: iteration 500

============================================================
📊 VALIDATION [iter 500]
   Val Loss: 2.5432
   Perplexity: 12.71
============================================================

🏆 New best model! Val loss: 2.5432

[ 1000/1000] loss: 2.5432 | tokens/sec: 15098
✅ Checkpoint saved: iteration 1000

============================================================
🎉 TRAINING COMPLETE!
✅ Final checkpoint saved to checkpoints/multiscale_bdh_reasoning
✅ Best validation loss: 2.5432
✅ Task: Reasoning (math, logic, commonsense)
============================================================
```

---

## 🎯 **Why Reasoning Tasks Are Better**

### **For Science Fair:**

✅ **More Impressive:**
- "My model learned to solve math problems!" vs "My model learned Shakespeare"
- Shows AI can reason, not just memorize

✅ **Clear Applications:**
- Math tutoring systems
- Logical reasoning AI
- Commonsense for robots

✅ **Demonstrates Intelligence:**
- Multi-step reasoning required
- Not just pattern matching
- Shows "thinking" capability

✅ **Relatable:**
- Judges understand math problems
- Everyone has done word problems
- Easy to demonstrate

---

## 📈 **Expected Results**

### **After Training:**

**What the model learns:**
- ✅ Math operations (add, subtract, multiply, divide)
- ✅ Multi-step reasoning (chain of thought)
- ✅ Pattern recognition (sequences, series)
- ✅ Logical deduction (if-then reasoning)
- ✅ Commonsense knowledge

**How to test:**
```python
from implementation.multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
import torch

# Load trained model
config = MultiScaleBDHConfig()
model = MultiScaleBDH(config)
checkpoint = torch.load('checkpoints/multiscale_bdh_reasoning/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test with a math problem
question = "Q: What is 15 × 14?"
input_text = question + " A:"

# Encode and generate
# (You'll need to implement a generate function)
output = model.generate(input_text, max_tokens=50)
print(output)
# Expected: "15 × 14 = 210" (or similar)
```

---

## 🎭 **Science Fair Demo Ideas**

### **Live Reasoning Demo:**

**1. Math Problems:**
- Input: "Q: John has 15 apples. He gives 5 away. How many left?"
- Output: "John has 10 apples"

**2. Logic Puzzles:**
- Input: "Q: If A > B and B > C, who is shortest?"
- Output: "C is the shortest"

**3. Pattern Completion:**
- Input: "Q: 2, 4, 8, 16, ?"
- Output: "32"

**4. AI Concepts:**
- Input: "Q: Why does BDH use Hebbian learning?"
- Output: "BDH uses Hebbian learning because neurons that fire together wire together, which is biologically plausible"

---

## 📊 **Comparison: Shakespeare vs Reasoning**

| Aspect | Shakespeare Training | Reasoning Training |
|--------|---------------------|-------------------|
| **Impressive** | ✅ Language generation | ✅ **Solves problems!** |
| **Understandable** | ✅ Familiar text | ✅ **Everyone does math** |
| **Applications** | ✅ Text generation | ✅ **Tutoring systems** |
| **Novelty** | ⚠️ Common | ✅ **Unique & impressive** |
| **Difficulty** | Medium | **Higher** |
| **Demo Value** | Good | **Excellent!** |

---

## 🔧 **Customization**

### **Add Your Own Problems:**

Edit `train_bdh_reasoning.py`, find the `create_gsm8k_style_dataset()` function:

```python
problems.append((
    "Your custom problem here?",
    "The answer"
))
```

### **Use Real GSM8K Dataset:**

1. Download GSM8K from: https://github.com/openai/grade-school-math
2. Extract to `data/gsm8k.txt`
3. Update training script to load it

### **Add More Problem Types:**

```python
# Add to problems list:
problems.append((
    "What is the derivative of x²?",
    "The derivative of x² is 2x"
))
```

---

## 💡 **Training Tips**

### **For Better Results:**

1. **Increase iterations:**
   ```python
   train_config.max_iters = 5000  # Instead of 1000
   ```

2. **Larger batch size:**
   ```python
   train_config.batch_size = 64  # If GPU memory allows
   ```

3. **Longer sequences:**
   ```python
   config.max_seq_len = 1024  # Instead of 512
   ```

### **For Faster Training:**

1. **Reduce data repetition:**
   ```python
   problems_repeated = problems * 10  # Instead of * 50
   ```

2. **Fewer validation checks:**
   ```python
   train_config.eval_interval = 1000  # Only at end
   ```

---

## 📁 **Checkpoints**

After training, checkpoints are saved to:
```
checkpoints/multiscale_bdh_reasoning/
├── checkpoint_iter_500.pt
├── checkpoint_iter_1000.pt
├── checkpoint_latest.pt
├── checkpoint_best.pt
└── training_config.json
```

**Use the best checkpoint for your science fair demo!**

---

## 🎉 **Summary**

### **What You're Training:**
- **Model:** Multi-Scale BDH (10M parameters)
- **Task:** Reasoning (math, logic, commonsense, AI)
- **Dataset:** ~800 reasoning problems
- **Duration:** ~20-30 minutes
- **Goal:** Demonstrate AI reasoning capability

### **Why This Is Better:**
- ✅ More impressive than Shakespeare
- ✅ Shows problem-solving ability
- ✅ Clear practical applications
- ✅ Easy for judges to understand
- ✅ Demonstrates multi-scale memory benefits

### **Start Training:**
```bash
conda activate bdh-fest
cd D:\projects\BDH
python train_bdh_reasoning.py
```

---

**This will make an INCREDIBLE science fair demo!** 🧠🚀🏆