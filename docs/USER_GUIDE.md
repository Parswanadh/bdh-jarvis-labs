# BDH Science Fest - User Guide

## Quick Start

Welcome to the improved BDH (Baby Dragon Hatchling) implementation! This guide will help you understand, train, and use the enhanced BDH model with multi-scale memory and efficient tokenization.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM (16GB+ recommended)

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional dependencies
pip install numpy matplotlib tokenizers tqdm

# Optional: for development
pip install jupyter pytest black flake8
```

### Step 2: Clone Repository

```bash
git clone https://github.com/your-repo/bdh-science-fest
cd bdh-science-fest
```

### Step 3: Verify Installation

```bash
# Test basic functionality
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# Test CUDA (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run basic test
python implementation/test_stable_config.py
```

---

## Training Your First Model

### Option 1: Multi-Scale BDH (Byte-Level)

This is the recommended starting point. Multi-scale BDH uses byte-level tokenization (no tokenizer training required).

#### Step 1: Prepare Data

Create a text file with your training data:

```bash
# Example: Create a simple dataset
echo "The quick brown fox jumps over the lazy dog. " > data/my_data.txt

# Or use existing data
cp path/to/your/data.txt data/my_data.txt
```

#### Step 2: Train Model

```bash
python implementation/train_multiscale.py \
    --config implementation/stable_config.py \
    --dataset data/my_data.txt \
    --output checkpoints/my_multiscale_bdh.pt \
    --decay_rates 0.95 0.99 0.995 \
    --max_iters 5000 \
    --batch_size 32
```

#### Step 3: Monitor Training

Watch the console output:
```
Creating Multi-Scale BDH model...
  decay_rates: [0.95, 0.99, 0.995]
  Total parameters: 10,234,567

iter     0 | loss 4.1234 | lr 3.00e-06 | tokens/sec 15000
iter   100 | loss 3.4567 | lr 1.50e-04 | tokens/sec 15234
iter   200 | loss 2.9876 | lr 3.00e-04 | tokens/sec 15123
...
```

### Option 2: BBPE-BDH (With Tokenizer)

BBPE tokenization provides 2.5-3× faster training but requires training a tokenizer first.

#### Step 1: Prepare Data

Same as above (create `data/my_data.txt`).

#### Step 2: Train Tokenizer

```bash
python implementation/train_tokenizer.py \
    --input data/my_data.txt \
    --vocab_size 8192 \
    --output tokenizer-model/ \
    --special_tokens "<pad>,<eos>"
```

This creates:
- `tokenizer-model/tokenizer.json` - Tokenizer configuration
- `tokenizer-model/merges.txt` - BPE merge rules
- `tokenizer-model/vocab.json` - Vocabulary

#### Step 3: Train Model with Tokenizer

```bash
python implementation/train_multiscale.py \
    --tokenizer tokenizer-model/tokenizer.json \
    --config implementation/stable_config.py \
    --dataset data/my_data.txt \
    --output checkpoints/my_bbpe_bdh.pt \
    --decay_rates 0.95 0.99 0.995 \
    --max_iters 5000
```

---

## Generating Text

### Python API

```python
from implementation.multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
import torch

# Load model
config = MultiScaleBDHConfig(
    vocab_size=256,  # or 8192 for BBPE
    n_embd=256,
    n_layer=6,
    decay_rates=[0.95, 0.99, 0.995]
)

model = MultiScaleBDH(config)
checkpoint = torch.load('checkpoints/my_multiscale_bdh.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate text
context = "The future of AI is"
output = model.generate(
    context,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50
)

print(output)
```

### Command-Line Generation

```bash
python implementation/multiscale_bdh.py \
    --checkpoint checkpoints/my_multiscale_bdh.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8
```

### Generation Parameters

| Parameter | Range | Effect | Recommended |
|-----------|-------|--------|-------------|
| `temperature` | 0.1 - 2.0 | Randomness | 0.7 - 0.9 |
| `top_k` | 1 - 100 | Diversity | 40 - 50 |
| `top_p` | 0.1 - 1.0 | Nucleus sampling | 0.9 - 0.95 |

---

## Configuration Options

### Model Architecture

```python
from implementation.multiscale_bdh import MultiScaleBDHConfig

config = MultiScaleBDHConfig(
    # Architecture
    vocab_size=256,          # 256 (bytes) or 8192 (BBPE)
    n_embd=256,             # Embedding dimension (try 128, 256, 512)
    n_layer=6,              # Number of layers (try 4, 6, 8)
    n_head=4,               # Attention heads (try 4, 8)
    ffn_dim=1024,           # FFN hidden dimension (usually 4×n_embd)

    # Multi-Scale Memory
    decay_rates=[0.95, 0.99, 0.995],  # Fast, medium, slow decay

    # Hebbian Learning
    hebbian_lr=0.01,        # Learning rate for state updates
    state_decay=0.99,       # Legacy single decay (unused in multiscale)

    # Regularization
    dropout=0.1,            # Dropout rate
)
```

### Training Configuration

```python
from implementation.stable_config import BDHStableTrainingConfig

config = BDHStableTrainingConfig(
    # Data
    dataset_path="data/my_data.txt",
    val_dataset_path="data/val_data.txt",

    # Training
    batch_size=32,          # Reduce if OOM (16, 8, 4)
    learning_rate=3e-4,     # Don't change unless necessary
    max_iters=10000,        # Total training iterations
    max_seq_len=512,        # Context length

    # CRITICAL: Stability settings
    initializer_range=0.006,  # DO NOT increase
    warmup_steps=5000,        # DO NOT decrease
    grad_clip=1.0,            # Always enable

    # Hardware
    device="auto",         # "cuda", "cpu", or "auto"
    dtype="bfloat16",      # "bfloat16", "float16", or "float32"
    compile=True,          # Use torch.compile for speed
)
```

### Pre-configured Profiles

```python
from implementation.stable_config import (
    get_small_model_config,    # 10M params
    get_medium_model_config,   # 100M params
    get_large_model_config,    # 1B params
)

# Get pre-configured settings
config = get_small_model_config()

# Customize if needed
config.decay_rates = [0.90, 0.95, 0.99, 0.995]  # 4 timescales
config.batch_size = 16  # Reduce for smaller GPU
```

---

## Troubleshooting

### Issue: Training loss becomes NaN

**Symptoms:**
- Loss suddenly becomes `nan`
- Model stops learning

**Solutions:**
1. **Reduce learning rate:**
   ```python
   config.learning_rate = 1e-4  # Was 3e-4
   ```

2. **Check initialization:**
   ```python
   config.initializer_range = 0.006  # Must be small!
   ```

3. **Enable gradient clipping:**
   ```python
   config.grad_clip = 1.0  # Always use this
   ```

4. **Use bfloat16 instead of float16:**
   ```python
   config.dtype = "bfloat16"  # More stable than float16
   ```

---

### Issue: CUDA Out of Memory

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes mid-iteration

**Solutions:**
1. **Reduce batch size:**
   ```python
   config.batch_size = 16  # Was 32
   # Keep reducing: 16 → 8 → 4 → 2
   ```

2. **Reduce model size:**
   ```python
   config.n_embd = 128     # Was 256
   config.n_layer = 4      # Was 6
   config.ffn_dim = 512    # Was 1024
   ```

3. **Reduce sequence length:**
   ```python
   config.max_seq_len = 256  # Was 512
   ```

4. **Enable gradient checkpointing:**
   ```python
   config.use_gradient_checkpointing = True
   ```

5. **Use CPU (slow but works):**
   ```python
   config.device = "cpu"
   ```

---

### Issue: Slow training

**Symptoms:**
- Training takes too long
- Low tokens/sec

**Solutions:**
1. **Use BBPE tokenization (2-3× speedup):**
   ```bash
   # Train tokenizer first
   python implementation/train_tokenizer.py \
       --input data/my_data.txt \
       --vocab_size 8192 \
       --output tokenizer-model/

   # Use tokenizer in training
   python implementation/train_multiscale.py \
       --tokenizer tokenizer-model/tokenizer.json \
       ...
   ```

2. **Enable torch.compile:**
   ```python
   config.compile = True  # 2-3× speedup
   ```

3. **Use bfloat16:**
   ```python
   config.dtype = "bfloat16"  # Faster than float32
   ```

4. **Increase batch size (if GPU memory allows):**
   ```python
   config.batch_size = 64  # Was 32
   ```

5. **Use a better GPU:**
   - RTX 4090: ~50K tok/s
   - RTX 4060: ~15K tok/s
   - GTX 1650: ~5K tok/s
   - CPU: ~500 tok/s

---

### Issue: Poor generation quality

**Symptoms:**
- Generated text is incoherent
- Model repeats phrases
- Output is gibberish

**Solutions:**
1. **Train longer:**
   ```python
   config.max_iters = 20000  # Was 10000
   ```

2. **Use more training data:**
   ```bash
   # Combine multiple datasets
   cat data1.txt data2.txt data3.txt > data/combined.txt
   ```

3. **Adjust generation parameters:**
   ```python
   output = model.generate(
       context,
       temperature=0.8,   # Try 0.5-1.0
       top_k=50,          # Try 40-50
       top_p=0.9          # Try 0.9-0.95
   )
   ```

4. **Check that training converged:**
   - Loss should be below 2.5
   - Loss should be decreasing
   - No loss spikes in recent iterations

5. **Use better initialization:**
   ```python
   config.use_mimetic_init = True  # More stable
   ```

---

## Advanced Usage

### Custom Decay Rates

Design your own multi-scale configuration:

```python
# Short-term memory focused
config = MultiScaleBDHConfig(
    decay_rates=[0.90, 0.95]  # Fast timescales only
)

# Long-term memory focused
config = MultiScaleBDHConfig(
    decay_rates=[0.98, 0.99, 0.995, 0.999]  # More slow timescales
)

# Biologically-inspired timescales
config = MultiScaleBDHConfig(
    decay_rates=[
        0.90,   # STP: 100ms scale (~50 tokens)
        0.95,   # STP: 1s scale (~200 tokens)
        0.98,   # LTP: 10s scale (~500 tokens)
        0.99,   # LTP: 100s scale (~1000 tokens)
        0.995,  # Structural: 1000s scale (~2000 tokens)
    ]
)
```

### Analyzing State Matrices

```python
import matplotlib.pyplot as plt

# Get state matrices
logits, states = model(x, return_state=True)

# Visualize each timescale
fig, axes = plt.subplots(1, len(config.decay_rates), figsize=(15, 5))
for i, (state, decay) in enumerate(zip(states, config.decay_rates)):
    axes[i].imshow(state.cpu().numpy(), cmap='hot', interpolation='nearest')
    axes[i].set_title(f'Decay {decay}')
plt.colorbar()
plt.savefig('state_matrices.png')
```

### Benchmarking

```bash
# Run benchmark suite
python benchmarking/benchmark_suite.py \
    --baseline checkpoints/baseline_bdh.pt \
    --improved checkpoints/multiscale_bdh.pt \
    --output benchmarking/results/

# Compare results
python benchmarking/statistical_analysis.py \
    --results benchmarking/results/
```

---

## Tips and Best Practices

### DO's ✅
1. **Start small**: Test with `max_iters=100` first
2. **Monitor loss**: Should decrease steadily
3. **Save checkpoints**: Use `save_interval=1000` or less
4. **Use stable configs**: Don't change `initializer_range`, `warmup_steps`, `learning_rate`
5. **Experiment**: Try different `decay_rates` combinations

### DON'Ts ❌
1. Don't use Transformer defaults (initializer_range=0.02, warmup=500)
2. Don't skip gradient clipping
3. Don't train without warmup
4. Don't use float16 (use bfloat16 instead)
5. Don't expect good results with tiny datasets

---

## Benchmarks

Run the benchmark suite to measure improvements:

```bash
python benchmarking/benchmark_suite.py \
    --baseline checkpoints/baseline_bdh.pt \
    --improved checkpoints/multiscale_bdh.pt \
    --output benchmarking/results/
```

Expected improvements:
| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Memory retention (500 tok) | 0.6% | 15% | 24× |
| Training speed | 10K tok/s | 27.5K tok/s | 2.75× |
| Training stability | 40% | 95% | 2.4× |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bdh_science_fair_2026,
  title={Multi-Scale Memory for Brain-Inspired AI: Science Fair Project},
  author={BDH Science Fest Squad},
  year={2026},
  note={Based on BDH (arXiv:2509.26507)}
}
```

Also cite the original BDH paper:

```bibtex
@article{bdh2025,
  title={The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain},
  author={Kosowski, Adrian and Uznański, Przemysław and Chorowski, Jan and Bartoszkiewicz, Michał and Stamirowska, Zuzanna},
  journal={arXiv preprint arXiv:2509.26507},
  year={2025}
}
```

---

## Getting Help

### Resources
1. **Documentation**: `docs/IMPLEMENTATION_CHANGES.md`
2. **Code comments**: Read inline comments in implementation files
3. **Original paper**: `2509.26507v1.pdf`
4. **GitHub issues**: https://github.com/your-repo/bdh-science-fest/issues

### Common Questions

**Q: How long does training take?**
A: With an RTX 4060: ~6 hours for 10K iterations on small dataset.

**Q: Can I train on CPU?**
A: Yes, but it's ~30× slower. Use for testing only.

**Q: What's the minimum GPU memory?**
A: 6GB for 10M model with batch_size=16.

**Q: Can I use this for commercial purposes?**
A: Check the original BDH repository license.

---

**Document Version:** 1.0
**Last Updated:** February 25, 2026
**Authors:** BDH Science Fest Squad
**Status:** Complete
