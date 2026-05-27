# BDH-GPU 10M: Complete Implementation Guide

## 🐉 What is BDH-GPU?

**BDH (Baby Dragon Hatchling)** is a biologically-inspired language model architecture that bridges:
- **Transformers** (current state-of-the-art AI)
- **Brain models** (neural networks in the human brain)

### Key Innovations

| Feature | Transformer | BDH-GPU | Why It Matters |
|---------|-------------|---------|----------------|
| **Attention** | Softmax O(N²) | Linear O(N) | Faster, scales to long sequences |
| **State** | KV cache | Synaptic matrix E | Interpretable, Hebbian learning |
| **Activations** | Dense, mixed | Sparse (~5%), positive | Biologically plausible |
| **Structure** | Fixed | Emergent modularity | Self-organizing circuits |
| **Memory** | Context window | Working memory (~100-500 tokens) | Similar to human span |

---

## 📁 Files Included

```
BDH/
├── bdh_gpu_10m.py          # Complete 10M BDH-GPU model
├── train_bdh_gpu.py         # Training script with all features
├── README_IMPLEMENTATION.md # This file
└── 2509.26507v1.pdf        # Original research paper
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Minimal dependencies
pip install torch numpy
```

### 2. Test the Model

```bash
python bdh_gpu_10m.py
```

This will:
- Create a 10M parameter BDH-GPU model
- Show parameter count
- Test forward pass
- Generate sample text

### 3. Train on Your Data

Create a file `input.txt` with your training data:
```bash
echo "The quick brown fox jumps over the lazy dog. " > input.txt
```

Then train:
```bash
python train_bdh_gpu.py
```

---

## 🏗️ Architecture Deep Dive

### High-Level Structure

```
INPUT TEXT (bytes)
    │
    ▼
┌─────────────────┐
│  TOKEN EMBED    │  [256 x 256] - Learn character representations
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   POSITION ENC  │  RoPE - Rotary Position Embeddings
└─────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│            BDH LAYER (×6)                            │
│  ┌────────────────┐         ┌────────────────┐      │
│  │ LINEAR ATTN    │         │  ReLU-FFN      │      │
│  │ (Excitatory)   │    +    │ (Inhibitory)   │      │
│  │                │         │                │      │
│  │ Q = x @ Wq     │         │ h = ReLU(x@W1) │      │
│  │ K = x @ Wk     │         │ out = h @ W2   │      │
│  │ V = x @ Wv     │         │                │      │
│  │                │         │ Sparse: ~5%    │      │
│  │ STATE UPDATE:  │         │ active neurons │      │
│  │ E ← Hebbian    │         │                │      │
│  └────────────────┘         └────────────────┘      │
│           │                        │                │
│           └────────────┬───────────┘                │
│                        ▼                            │
│              MULTIPLICATIVE GATING                  │
│              (x * sigmoid(attn + ffn))             │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  OUTPUT PROJ    │  [256 x 256] - Predict next byte
└─────────────────┘
    │
    ▼
LOGITS → Cross-Entropy Loss
```

### The Magic: Synaptic State Matrix

```
State Matrix E [256 x 256] - The Working Memory

Initial State (t=0):              After "The cat" (t=2):
┌─────────────────────┐          ┌─────────────────────┐
│ 0  0  0  0  0  0   │          │ •  •  █  █  •  •   │
│ 0  0  0  0  0  0   │    →     │ •  •  █  █  •  •   │
│ 0  0  0  0  0  0   │          │ •  █  •  █  •  •   │
│ ...                │          │ ...                │
└─────────────────────┘          └─────────────────────┘
      Empty                         Learned associations!

"Neurons that fire together, wire together"
- Hebbian learning strengthens synapses
- State decays over time (working memory)
- Sparse: only ~5% of connections active
- Interpretable: read what model is "thinking"
```

---

## 🔧 Configuration Options

### Model Architecture

```python
@dataclass
class BDHConfig:
    vocab_size: int = 256          # Byte-level (no tokenizer!)
    n_embd: int = 256              # Embedding dimension
    n_layer: int = 6               # Number of layers
    n_head: int = 4                # Attention heads
    ffn_dim: int = 1024            # FFN internal dimension
    dropout: float = 0.1           # Dropout rate
    state_decay: float = 0.99      # Memory decay per token
    hebbian_lr: float = 0.01       # Synaptic learning rate
```

### Parameter Scaling

| Model Size | n_embd | n_layer | ffn_dim | Total Params |
|------------|--------|---------|---------|--------------|
| 1M         | 128    | 4       | 512     | ~1M          |
| **10M**    | **256**| **6**   | **1024**| **~10M**     |
| 100M       | 512    | 8       | 2048    | ~100M        |
| 1B         | 1024   | 12      | 4096    | ~1B          |

---

## 🎯 Training Guide

### Basic Training

```bash
python train_bdh_gpu.py
```

### Custom Configuration

Edit `train_bdh_gpu.py`:

```python
config = TrainingConfig(
    # Data
    dataset_path="your_data.txt",
    val_dataset_path="your_val_data.txt",

    # Model size
    n_embd=256,
    n_layer=6,
    ffn_dim=1024,

    # Training
    batch_size=32,
    learning_rate=3e-4,
    max_iters=5000,
    max_seq_len=512,

    # Hardware
    device="auto",  # cuda or cpu
    compile=True,   # Use torch.compile for speed
)
```

### Training Tips

1. **Start small**: Test with `max_iters=100` first
2. **Monitor loss**: Should decrease steadily
3. **GPU memory**: Reduce `batch_size` if OOM
4. **Speed**: Enable `compile=True` for 2-3x speedup
5. **Mixed precision**: Uses bfloat16 by default

### Expected Training Times

| Hardware | Tokens/sec | Time for 10K iterations |
|----------|------------|-------------------------|
| RTX 4090 | ~50,000    | ~2 hours                |
| RTX 4060 | ~15,000    | ~6 hours                |
| GTX 1650 | ~5,000     | ~18 hours               |
| CPU      | ~500       | ~7 days                 |

---

## 📊 Monitoring Training

### Console Output

```
Creating BDH-GPU 10M model...
Total parameters: 10,234,567 (10.23M)

iter     0 | loss 4.1234 | lr 3.00e-06 | tokens/sec 15000
iter   100 | loss 3.4567 | lr 1.50e-04 | tokens/sec 15234
iter   200 | loss 2.9876 | lr 3.00e-04 | tokens/sec 15123
...

--- Iteration 500 ---
Train loss: 2.4567
Val loss: 2.5432

Sample:
The meaning of life is to find happiness and purpose in...
```

### Checkpoints

Saved automatically to `checkpoints/`:
```
checkpoints/
├── checkpoint_1000.pt
├── checkpoint_2000.pt
├── checkpoint_3000.pt
└── final_model.pt
```

Load checkpoint:
```python
model.load_state_dict(torch.load('checkpoints/final_model.pt'))
```

---

## 🎨 Text Generation

### Interactive Generation

```python
from bdh_gpu_10m import BDHGPUTensor, BDHConfig
import torch

# Load model
config = BDHConfig(n_embd=256, n_layer=6)
model = BDHGPUTensor(config)
model.load_state_dict(torch.load('checkpoints/final_model.pt')['model_state_dict'])
model.eval()
model.to('cuda')

# Generate
context = "The future of AI is"
context_bytes = context.encode('utf-8')
idx = torch.tensor([[b for b in context_bytes]], dtype=torch.long).to('cuda')

with torch.no_grad():
    generated = model.generate(idx, max_new_tokens=200, temperature=0.8)

text = bytes(generated[0].tolist()).decode('utf-8', errors='ignore')
print(text)
```

### Temperature Settings

| Temperature | Behavior |
|-------------|----------|
| 0.1 - 0.3   | Very conservative, repetitive |
| 0.5 - 0.8   | Balanced (recommended) |
| 0.9 - 1.2   | Creative, diverse |
| 1.5+        | Very random, may be incoherent |

---

## 🔬 Analysis & Visualization

### View Synaptic State

```python
# Get state matrix during generation
logits, state = model(idx, return_state=True)
state_matrix = state.reshape(256, 256).cpu().numpy()

import matplotlib.pyplot as plt
plt.imshow(state_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Synaptic State Matrix')
plt.show()
```

### Sparsity Analysis

```python
# Check activation sparsity
activation = model.layers[0].ffn.W1(x)
sparsity = (activation == 0).float().mean()
print(f"Sparsity: {sparsity*100:.1f}%")
# Expected: ~95% (only 5% active)
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce these in config:
batch_size = 16          # Was 32
max_seq_len = 256        # Was 512
n_embd = 128            # Was 256 (smaller model)
```

### Slow Training

```python
# Enable these:
compile = True          # torch.compile for 2-3x speedup
dtype = "bfloat16"      # Mixed precision
```

### NaN Loss

```python
# Reduce learning rate:
learning_rate = 1e-4    # Was 3e-4

# Enable gradient clipping:
grad_clip = 1.0
```

### Poor Generation Quality

- Train for more iterations (increase `max_iters`)
- Use more training data
- Try different temperature (0.5 - 1.0)
- Check that loss is decreasing

---

## 📚 Key Differences from Transformers

### 1. Linear Attention

```python
# Standard Transformer (O(N²))
attn = softmax(Q @ K.T / sqrt(d)) @ V

# BDH-GPU Linear (O(N))
attn = Q @ (K.T @ V)
```

**Benefit**: Faster, scales to longer sequences

### 2. Multiplicative Gating

```python
# Transformer (additive)
x = x + attn_out + ffn_out

# BDH-GPU (multiplicative)
x = x * sigmoid(attn_out + ffn_out)
```

**Benefit**: Biologically plausible, gate modulation

### 3. State Matrix

```python
# Transformer: KV cache
past_keys, past_values = []

# BDH-GPU: Synaptic state
state = decay * state + lr * (Q ⊙ V)  # Hebbian!
```

**Benefit**: Interpretable, working memory

### 4. Positive Activations

```python
# Transformer: Various activations
x = GELU(x) or x = SwiGLU(x)

# BDH-GPU: ReLU only
x = ReLU(x)  # Always >= 0
```

**Benefit**: Sparse, interpretable, biological

---

## 🎓 Learning Resources

1. **Paper**: `2509.26507v1.pdf` (in this directory)
2. **Code Study**: Read `bdh_gpu_10m.py` line by line
3. **Experiments**: Modify parameters and observe effects
4. **Visualization**: Plot state matrices, attention patterns

---

## 🚀 Next Steps

1. **Train** on your own data
2. **Experiment** with different model sizes
3. **Visualize** the synaptic state matrix
4. **Compare** with standard Transformers
5. **Fine-tune** on specific tasks

---

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{bdh2025,
  title={The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain},
  author={Kosowski, Adrian and Uznański, Przemysław and Chorowski, Jan and Bartoszkiewicz, Michał and Stamirowska, Zuzanna},
  journal={arXiv preprint arXiv:2509.26507},
  year={2025}
}
```

---

## 📄 License

This implementation is provided for educational and research purposes.
Please refer to the original paper and Pathway's repository for licensing details.

---

**Happy Training! 🐉**

For questions or issues, refer to:
- Original paper: `2509.26507v1.pdf`
- Pathway repository: https://github.com/pathwaycom/bdh
- Technical blog: https://pathway.com/research/bdh
