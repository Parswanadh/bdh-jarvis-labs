# BBPE-BDH Training Integration Guide

## Overview

This guide explains how to integrate BBPE (Byte-Level BPE) tokenization with the BDH training pipeline to achieve 2-4× token reduction and 2-3× training speedup.

## Performance Results

### Benchmark Results (Latest)

| Metric | Value | Notes |
|--------|-------|-------|
| **Token Reduction** | **3.67×** | Achieved on paper_text.txt |
| **Processing Speed** | **170,897 tokens/sec** | Encoding performance |
| **Vocabulary Size** | **5,895** | Trained from 256 bytes |
| **Encoding Time** | **0.011s** | For 100 samples |

### Comparison Table

| Approach | Tokens | Bytes | Reduction |
|----------|--------|-------|-----------|
| Byte-level (baseline) | 6,663 | 6,663 | 1× |
| BBPE (8K vocab) | 1,818 | 6,663 | **3.67×** |

## Quick Start

### 1. Load and Use Tokenizer

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer-model/tokenizer.json")

# Encode text
text = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer.encode(text)

# Get token IDs
tokens = encoding.ids
print(f"Tokens: {tokens}")  # [143, 829, 1512, ...]
print(f"Count: {len(tokens)}")  # 18 tokens (vs 44 bytes)
```

### 2. Benchmark Your Data

```python
from implementation.bbpe_training_integration import benchmark_tokenization

results = benchmark_tokenization(
    "your_data.txt",
    "tokenizer-model/tokenizer.json",
    num_samples=1000
)

print(f"Token reduction: {results['reduction_ratio']:.2f}x")
print(f"Processing speed: {results['tokens_per_second']:,.0f} tokens/sec")
```

### 3. Create Training Data Splits

```python
from implementation.bbpe_training_integration import create_training_data_splits

train_path, val_path, test_path = create_training_data_splits(
    input_file="your_corpus.txt",
    output_dir="data_splits",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### 4. Integrate with Training (When PyTorch Available)

```python
from implementation.bbpe_training_integration import BBPEDataModule

# Create data module
data_module = BBPEDataModule(
    tokenizer_path="tokenizer-model/tokenizer.json",
    train_path="data_splits/train.txt",
    val_path="data_splits/val.txt",
    batch_size=32,
    max_seq_len=2048
)

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Use in training loop
for batch in train_loader:
    # batch is tensor of token IDs
    logits, loss = model(batch)
    # ... training code
```

## Integration with BDH Models

### Method 1: Use BBPEBDHModel (Recommended)

```python
from implementation.bbpe_bdh import create_bbpe_bdh_model

# Create model with BBPE tokenizer
model = create_bbpe_bdh_model(
    tokenizer_path="tokenizer-model/tokenizer.json",
    n_embd=256,
    n_layer=6,
    n_head=4,
    ffn_dim=1024
)

# Encode text
tokens = model.encode_text("Your text here")

# Generate text
generated = model.generate_text(
    prompt="The BDH architecture",
    max_new_tokens=100,
    temperature=0.8
)
```

### Method 2: Modify Existing BDH Model

If you have an existing BDH model, update the config:

```python
from bdh_gpu_10m import BDHConfig, BDHGPUTensor
from implementation.bbpe_bdh import BBPEBDHConfig

# Use BBPE config (automatically loads vocab size)
config = BBPEBDHConfig.from_tokenizer(
    tokenizer_path="tokenizer-model/tokenizer.json",
    n_embd=256,
    n_layer=6,
    n_head=4
)

# Create model
model = BDHGPUTensor(config)

# Note: Token embedding will have 5895 vocab size instead of 256
# Output projection will also be 5895
```

## Training Pipeline Modifications

### Key Changes Needed

1. **Data Loading:**
   - Use `BBPETextDataset` or `BBPEDataModule`
   - Tokenizes text on-the-fly using BBPE
   - Handles padding and batching automatically

2. **Model Configuration:**
   - Update `vocab_size` from 256 to 5895
   - Token embedding: `nn.Embedding(5895, n_embd)`
   - Output projection: `nn.Linear(n_embd, 5895)`

3. **Special Tokens:**
   - `[PAD]` (ID: 0) - Padding
   - `[UNK]` (ID: 1) - Unknown
   - `[CLS]` (ID: 2) - Classification
   - `[SEP]` (ID: 3) - Separator
   - `` (ID: 4) - Mask

### Example Training Loop

```python
import torch
from implementation.bbpe_training_integration import BBPEDataModule
from implementation.bbpe_bdh import create_bbpe_bdh_model

# Setup
device = "cuda"
tokenizer_path = "tokenizer-model/tokenizer.json"

# Create model
model = create_bbpe_bdh_model(
    tokenizer_path=tokenizer_path,
    n_embd=256,
    n_layer=6,
    n_head=4
).to(device)

# Create data module
data_module = BBPEDataModule(
    tokenizer_path=tokenizer_path,
    train_path="data_splits/train.txt",
    val_path="data_splits/val.txt",
    batch_size=32,
    max_seq_len=1024  # Can use longer sequences due to BBPE!
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader = data_module.train_dataloader()

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        # Forward pass
        logits, _ = model(batch[:, :-1])
        targets = batch[:, 1:]

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

## Expected Training Speedup

With BBPE tokenization:

| Metric | Byte-level (256) | BBPE (5895) | Improvement |
|--------|------------------|-------------|-------------|
| Sequence Length | 2048 tokens | ~700 tokens | **3× shorter** |
| Tokens/Second | 10,000 | ~25,000 | **2.5× faster** |
| Memory per Batch | High | Lower | **3× less** |
| Training Speed | 1× | 2-3× | **2-3× faster** |

### Why Faster?

1. **Shorter Sequences:** BBPE reduces sequence length by 3-4×
2. **Less Padding:** More efficient batching
3. **Faster Computation:** Fewer tokens to process
4. **Better GPU Utilization:** Longer effective sequences in same time

## Testing

### Run Tokenizer Test

```bash
python implementation/test_tokenizer_only.py
```

Expected output: `✅ SUCCESS: Token reduction target (2-4x) achieved!`

### Run Integration Benchmark

```bash
python implementation/bbpe_training_integration.py
```

Expected output:
```
Token reduction: 3.67x
Processing speed: 170,897 tokens/sec
```

## Troubleshooting

### Issue: PyTorch Not Installed

**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: Tokenizer Not Found

**Solution:**
```bash
python implementation/train_tokenizer_v2.py --input paper_text.txt --vocab_size 8192
```

### Issue: High [UNK] Rate

**Solution:**
- Increase vocab_size (e.g., 16384)
- Train on more diverse data
- Increase training data size

### Issue: Out of Memory

**Solution:**
- Reduce batch_size
- Reduce max_seq_len
- Use gradient accumulation

## Files Reference

| File | Purpose |
|------|---------|
| `tokenizer-model/tokenizer.json` | Trained BBPE tokenizer |
| `implementation/train_tokenizer_v2.py` | Tokenizer training script |
| `implementation/bbpe_bdh.py` | BBPE-BDH model integration |
| `implementation/bbpe_training_integration.py` | Training pipeline integration |
| `implementation/test_tokenizer_only.py` | Tokenizer validation |
| `implementation/bbpe_results.md` | Detailed results |

## Next Steps

1. **Install PyTorch** (if not installed)
2. **Test Integration** with your BDH model
3. **Benchmark Training** compare byte-level vs BBPE
4. **Optimize Hyperparameters** for BBPE sequences
5. **Measure Training Speedup** expected: 2-3× faster

## Support

For issues or questions:
- Check `implementation/bbpe_results.md` for detailed results
- See `skills/bbpe-tokenization.md` for BBPE theory
- Contact: T2 (tokenization-engineer)

---

**Last Updated:** 2025-02-25
**Status:** Ready for Integration
**Token Reduction Achieved:** 3.67× ✅
