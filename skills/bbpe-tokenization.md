# BBPE Tokenization Skill

## Purpose
Implement Byte-Level BPE tokenization for BDH to improve efficiency while maintaining byte-level foundation.

## When to Use This Skill
- Training tokenizers on datasets
- Integrating BBPE with BDH architecture
- Comparing byte-level vs BBPE performance
- Optimizing sequence length for training speed

## Key Capabilities

### 1. BBPE Training
```python
from tokenizers import Tokenizer, models, trainers

# Train BBPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
trainer = trainers.BpeTrainer(
    vocab_size=8192,  # Sweet spot: 2-4× token reduction
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
    initial_alphabet=list(range(256))  # Start with bytes
)
tokenizer.train(["your_data.txt"], trainer=trainer)
```

### 2. Integration with BDH
```python
class BBPEBDHConfig(BDHConfig):
    vocab_size: int = 8192  # Instead of 256

# Update embedding layer
self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

# Update output projection
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

### 3. Performance Measurement
```python
# Token reduction metric
original_tokens = len(byte_level_tokens)
bbpe_tokens = len(bbpe_tokens)
reduction_ratio = original_tokens / bbpe_tokens  # Target: 2-4×
```

## Testing Checklist
- [ ] Tokenizer trains successfully on dataset
- [ ] Encode/decode roundtrip works correctly
- [ ] Token reduction achieved: 2-4× fewer tokens
- [ ] All special characters/emojis handled correctly
- [ ] Training speedup measurable: 2-3× faster
- [ ] Perplexity comparable or better than baseline

## Vocabulary Size Selection

| Size | Token Reduction | Training Speed | Memory | Recommendation |
|------|----------------|----------------|--------|----------------|
| 256 (byte-level) | 1× (baseline) | 1× | Minimal | Baseline |
| 4096 | 2-3× | 2× | Low | Quick iteration |
| **8192** | **3-4×** | **2-3×** | **Medium** | **Sweet spot** |
| 16384 | 4-5× | 3× | High | Diminishing returns |
| 32768 | 5-6× | 3× | High | Near standard BPE |

**Recommended:** 8192 for optimal balance

## Expected Results

### Token Reduction
- Byte-level: "The quick brown fox" = 19 tokens
- BBPE: "The quick brown fox" = 5-6 tokens
- **Reduction: 3-4×**

### Training Speed
- Byte-level: 10,000 tokens/sec
- BBPE: 25,000-30,000 tokens/sec
- **Speedup: 2-3×**

### Biological Plausibility
- Still byte-based (starts with byte alphabet)
- Common patterns merged (like word formation in brain)
- Maintains interpretability

## Related Files
- `implementation/bbpe_bdh.py` - BBPE integration (new)
- `implementation/train_tokenizer.py` - Tokenizer training script (new)
- `tokenizer-model/` - Trained tokenizer directory
- `benchmarking/benchmark_suite.py` - Token reduction metrics

## Common Issues & Solutions

**Issue:** Tokenizer training is slow
**Solution:** Use smaller sample dataset (1-10MB) for training

**Issue:** Unknown token ([UNK]) rate too high
**Solution:** Increase vocab_size or ensure training data covers character set

**Issue:** Embedding layer too large
**Solution:** Reduce vocab_size or use embedding tying (lm_head = token_embedding.T)

**Issue:** Special tokens interfere with text
**Solution:** Use rare byte values as special tokens (< 32)

## Data Preparation

### For Training Tokenizer
```bash
# Create sample dataset
head -c 1000000 your_data.txt > tokenizer_train.txt

# Train tokenizer
python implementation/train_tokenizer.py \
    --input tokenizer_train.txt \
    --vocab_size 8192 \
    --output tokenizer-model
```

### For Training BDH
```python
# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer-model/tokenizer.json")

# Encode text
tokens = tokenizer.encode("Your text here").ids
```

## Performance Tips
1. Use tokenizer.pre_tokenize() for batch processing
2. Cache encoded tokens for repeated use
3. Use tokenizer.enable_padding() for batch training
4. Profile tokenizer.encode() during data loading

## Verification
Run: `python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('tokenizer-model/tokenizer.json'); print('BBPE tokenizer loaded:', t.encode('Hello').ids)"`

## Comparison: BBPE vs Alternatives

| Method | Byte-based | Token Reduction | Training Speed | Biological Plausibility |
|--------|-----------|----------------|----------------|-------------------------|
| Byte-level (256) | ✅ Yes | 1× (baseline) | 1× | ✅ Very High |
| **BBPE (8K)** | ✅ **Yes** | **3-4×** | **2-3×** | ✅ **High** |
| Standard BPE (50K) | ❌ No | 5-6× | 3× | ⚠️ Medium |
| WordPiece (30K) | ❌ No | 4-5× | 2-3× | ❌ Low |

**BBPE is optimal for BDH:** maintains byte-level foundation while improving efficiency.
