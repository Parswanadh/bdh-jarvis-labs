# BBPE Tokenizer Quick Start Guide

## Overview

The BBPE (Byte-Level BPE) tokenizer has been successfully trained and is ready for use with BDH. This guide shows you how to use it.

## Quick Start

### 1. Load and Use the Tokenizer

```python
from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("tokenizer-model/tokenizer.json")

# Encode text
text = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer.encode(text)

# Get token IDs
token_ids = encoding.ids
print(f"Token IDs: {token_ids}")

# Get number of tokens
num_tokens = len(token_ids)
print(f"Number of tokens: {num_tokens}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded text: {decoded}")
```

### 2. Calculate Token Reduction

```python
# Compare byte-level vs BBPE tokens
text = "Your text here"

# Byte-level count (baseline)
byte_count = len(text.encode('utf-8'))

# BBPE token count
encoding = tokenizer.encode(text)
bbpe_count = len(encoding.ids)

# Calculate reduction
reduction = byte_count / bbpe_count
print(f"Byte tokens: {byte_count}")
print(f"BBPE tokens: {bbpe_count}")
print(f"Reduction: {reduction:.2f}x")
```

### 3. Batch Processing

```python
# Process multiple texts
texts = [
    "The quick brown fox",
    "Hello, world!",
    "BDH architecture"
]

for text in texts:
    encoding = tokenizer.encode(text)
    print(f"{text[:30]} -> {len(encoding.ids)} tokens")
```

## Tokenizer Specifications

- **Type**: Byte-Level BPE (BBPE)
- **Vocabulary Size**: 5895 tokens
- **Target Vocab Size**: 8192
- **Special Tokens**:
  - [PAD] (ID: 0) - Padding
  - [UNK] (ID: 1) - Unknown
  - [CLS] (ID: 2) - Classification
  - [SEP] (ID: 3) - Separator
  - "" (ID: 4) - Mask

## Performance Results

Average token reduction: **2.95×** (meets 2-4× target)

| Text Sample | Byte Tokens | BBPE Tokens | Reduction |
|-------------|-------------|-------------|-----------|
| "The quick brown fox jumps..." | 44 | 18 | 2.44× |
| "Hello, world!" | 13 | 6 | 2.17× |
| "This is a test of BBPE..." | 44 | 13 | 3.38× |
| "BDH: The Dragon Hatchling..." | 39 | 9 | 4.33× |

## Integration with BDH

See `implementation/bbpe_bdh.py` for the BBPE-BDH integration module:

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

# Get statistics
stats = model.get_token_reduction_stats(["sample1", "sample2"])
print(f"Avg reduction: {stats['avg_reduction_ratio']:.2f}x")
```

## Testing

Run the tokenizer test:

```bash
python implementation/test_tokenizer_only.py
```

Expected output: `✅ SUCCESS: Token reduction target (2-4x) achieved!`

## Next Steps

1. **Install PyTorch** (if not already installed)
2. **Integrate with BDH training pipeline**
3. **Benchmark training speedup** (expected: 2-3× faster)
4. **Test on larger datasets**

## Files

- `tokenizer-model/tokenizer.json` - Trained tokenizer model
- `implementation/train_tokenizer_v2.py` - Training script
- `implementation/bbpe_bdh.py` - BDH integration
- `implementation/test_tokenizer_only.py` - Test script
- `implementation/bbpe_results.md` - Full results

---

For more details, see `implementation/bbpe_results.md`
