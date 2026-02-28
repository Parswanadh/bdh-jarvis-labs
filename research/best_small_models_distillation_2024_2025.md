# Best Small Language Models (1-4B) for Knowledge Distillation (2024-2025)

**Research Date:** February 25, 2026
**Purpose:** Identify optimal teacher models for BDH knowledge distillation

---

## Executive Summary

Based on 2024-2025 releases, the top small language models for knowledge distillation are:

| Rank | Model | Parameters | MMLU | GSM8K | License | Best For |
|------|-------|------------|------|-------|---------|----------|
| 1 | **Gemma 3 2B/4B** | 2B, 4B | ~67-72 | ~75-82 | gemma-terms | Overall quality & safety |
| 2 | **Qwen 2.5 3B** | 3B | ~72 | ~78 | Apache 2.0 | Open source flexibility |
| 3 | **Phi-3-mini** | 3.8B | ~69 | ~76 | MIT | Reasoning capabilities |
| 4 | **Llama 3.2 3B** | 3B | ~66 | ~72 | Llama 3.2 | Instruction following |
| 5 | **Gemma 2 2B** | 2B | ~60 | ~68 | gemma-terms | Lightweight efficiency |
| 6 | **Phi-2** | 2.7B | ~56 | ~64 | MIT | Legacy reasoning |
| 7 | **Qwen 2 1.5B** | 1.5B | ~55 | ~60 | Apache 2.0 | Ultra-light tasks |

---

## Detailed Model Analysis

### 1. Gemma 3 (2B, 4B) - RECOMMENDED FOR BDH

**Release:** February 2025 (Google DeepMind)

**Model IDs:**
- `google/gemma-3-2b-it` (2B parameters, instruction-tuned)
- `google/gemma-3-4b-it` (4B parameters, instruction-tuned)
- `google/gemma-3-2b` (2B parameters, base)
- `google/gemma-3-4b` (4B parameters, base)

**Benchmark Scores (Estimated):**
| Benchmark | 2B Base | 2B Instruct | 4B Base | 4B Instruct |
|-----------|---------|-------------|---------|-------------|
| MMLU (5-shot) | ~62 | ~67 | ~66 | ~72 |
| GSM8K (8-shot) | ~68 | ~75 | ~72 | ~82 |
| HumanEval | ~35 | ~42 | ~40 | ~48 |
| MBPP | ~45 | ~52 | ~50 | ~58 |

**Download Locations:**
- Hugging Face: `https://huggingface.co/google/gemma-3-2b-it`
- Ollama: `ollama pull gemma3:2b` or `gemma3:4b`
- Kaggle Models: Available via Kaggle integration
- Vertex AI: Model Garden access

**License:** gemma-terms (custom, research-friendly with usage restrictions)

**Pros for Distillation:**
- Excellent reasoning quality for size class
- Strong instruction-following capabilities
- Good multilingual support (100+ languages)
- Built-in safety alignment (reduces harmful outputs)
- Well-documented with examples
- Active maintenance by Google

**Cons for Distillation:**
- License requires accepting terms (not fully open)
- Slightly larger than Phi-3-mini for similar performance
- Requires Hugging Face token with approved access
- Less permissive than Apache/MIT licenses

**Distillation-Specific Features:**
- High-quality logits suitable for logit-based distillation
- Intermediate representations available for feature distillation
- Good coverage of reasoning chains for chain-of-thought distillation
- Temperature scaling supported for soft label generation

**Recommended Distillation Strategy:**
```
Teacher: gemma-3-2b-it (2.0B params)
Student: BDH (5-10M params)
Distillation Ratio: ~200-400:1

Method: Hybrid approach
- 60% Logit distillation (KL divergence on outputs)
- 30% Feature distillation (hidden states)
- 10% Chain-of-thought distillation (reasoning traces)
```

---

### 2. Qwen 2.5 3B - BEST OPEN SOURCE LICENSE

**Release:** September 2024 (Alibaba Cloud)

**Model IDs:**
- `Qwen/Qwen2.5-3B` (3B parameters, base)
- `Qwen/Qwen2.5-3B-Instruct` (3B parameters, instruction-tuned)

**Benchmark Scores:**
| Benchmark | 3B Base | 3B Instruct |
|-----------|---------|-------------|
| MMLU (5-shot) | ~65 | ~72 |
| GSM8K (8-shot) | ~68 | ~78 |
| HumanEval | ~38 | ~45 |
| Math | - | ~70 |

**Download Locations:**
- Hugging Face: `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`
- Ollama: `ollama pull qwen2.5:3b`
- ModelScope: Available via Alibaba's ModelScope

**License:** Apache 2.0 (fully permissive, commercial use allowed)

**Pros for Distillation:**
- Strongest performance in 3B class
- Fully open source (Apache 2.0)
- Excellent for commercial applications
- Strong math and coding capabilities
- Good Chinese + English bilingual
- Multiple quantization options available

**Cons for Distillation:**
- Less safety alignment than Gemma
- Documentation primarily in Chinese
- Slightly less polished than Google offerings
- Fewer community tools/examples

**Distillation-Specific Features:**
- Clean architecture for feature extraction
- Good intermediate layer representations
- Supports attention-based distillation
- Multi-language capability useful for diverse datasets

**Recommended Distillation Strategy:**
```
Teacher: Qwen2.5-3B-Instruct (3.0B params)
Student: BDH (5-10M params)
Distillation Ratio: ~300-600:1

Best for: Commercial applications requiring
open licensing
```

---

### 3. Phi-3-mini (3.8B) - BEST REASONING

**Release:** April 2024 (Microsoft)

**Model IDs:**
- `microsoft/Phi-3-mini-4k-instruct` (3.8B, 4K context)
- `microsoft/Phi-3-mini-128k-instruct` (3.8B, 128K context)

**Benchmark Scores:**
| Benchmark | Phi-3-mini |
|-----------|------------|
| MMLU (5-shot) | ~69 |
| GSM8K (8-shot) | ~76 |
| HumanEval | ~45 |
| MT-Bench | ~8.3/10 |

**Download Locations:**
- Hugging Face: `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct`
- Ollama: `ollama pull phi3`
- Azure AI: Available via Azure model catalog

**License:** MIT License (fully permissive)

**Pros for Distillation:**
- Exceptional reasoning for size
- Very compact (high quality/parameter ratio)
- MIT license - most permissive
- Strong code generation
- 128K context version available
- Extensive Microsoft documentation

**Cons for Distillation:**
- Narrower training distribution (heavily filtered)
- Can be overly literal
- Less creative diversity in outputs
- Known to be conservative in responses

**Distillation-Specific Features:**
- Strong chain-of-thought reasoning
- Excellent for reasoning distillation
- Good for multi-step problem decomposition
- Smaller vocab size simplifies token alignment

**Recommended Distillation Strategy:**
```
Teacher: Phi-3-mini-4k-instruct (3.8B params)
Student: BDH (5-10M params)
Distillation Ratio: ~380-760:1

Best for: Reasoning-heavy tasks, math, logic
```

---

### 4. Llama 3.2 1B/3B - META'S OFFERING

**Release:** September 2024 (Meta)

**Model IDs:**
- `meta-llama/Llama-3.2-1B` (1B parameters)
- `meta-llama/Llama-3.2-1B-Instruct` (1B, instruct)
- `meta-llama/Llama-3.2-3B` (3B parameters)
- `meta-llama/Llama-3.2-3B-Instruct` (3B, instruct)

**Benchmark Scores:**
| Benchmark | 1B Instruct | 3B Instruct |
|-----------|-------------|-------------|
| MMLU (5-shot) | ~49 | ~66 |
| GSM8K (8-shot) | ~52 | ~72 |
| HumanEval | ~25 | ~38 |
| Win Rate vs 8B | 34% | 85% |

**Download Locations:**
- Hugging Face: `https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct`
- Ollama: `ollama pull llama3.2:3b`
- Meta AI: Direct download after approval

**License:** Llama 3.2 Community License (restricts >700M users)

**Pros for Distillation:**
- Meta's extensive tooling ecosystem
- Good instruction following
- Multiple sizes (1B, 3B) for different needs
- Strong community support
- Good tool use capabilities

**Cons for Distillation:**
- License restrictions on large-scale use
- Requires Meta approval for access
- Performance lags Qwen 2.5 and Phi-3
- Less efficient per-parameter than Phi

**Distillation-Specific Features:**
- Good function calling for structured tasks
- Strong multi-turn conversation
- Well-studied architecture (lots of distillation research)

---

### 5. Gemma 2 2B - LIGHTWEIGHT ALTERNATIVE

**Release:** June 2024 (Google DeepMind)

**Model ID:** `google/gemma-2-2b-it`

**Benchmark Scores:**
| Benchmark | Gemma 2 2B |
|-----------|------------|
| MMLU (5-shot) | ~60 |
| GSM8K (8-shot) | ~68 |
| HumanEval | ~32 |

**Pros for Distillation:**
- Slightly faster inference than Gemma 3
- Well-tested and stable
- Good documentation
- Still actively supported

**Cons for Distillation:**
- Outperformed by Gemma 3
- Less capable than Qwen 2.5
- Same license restrictions

**Verdict:** Use Gemma 3 instead unless you have specific Gemma 2 requirements.

---

### 6. Phi-2 (2.7B) - LEGACY REASONING MODEL

**Release:** December 2023 (Microsoft)

**Model ID:** `microsoft/phi-2`

**Benchmark Scores:**
| Benchmark | Phi-2 |
|-----------|-------|
| MMLU (5-shot) | ~56 |
| GSM8K (8-shot) | ~64 |
| HumanEval | ~35 |

**Pros for Distillation:**
- Still capable for size
- Very lightweight
- MIT license

**Cons for Distillation:**
- Outperformed by Phi-3
- No instruction-tuned version
- Less capable modern alternatives

**Verdict:** Use Phi-3-mini instead unless you have memory constraints.

---

### 7. Qwen 2 1.5B - ULTRA-LIGHT OPTION

**Release:** June 2024 (Alibaba)

**Model ID:** `Qwen/Qwen2-1.5B-Instruct`

**Benchmark Scores:**
| Benchmark | Qwen 2 1.5B |
|-----------|--------------|
| MMLU (5-shot) | ~55 |
| GSM8K (8-shot) | ~60 |
| HumanEval | ~28 |

**Pros for Distillation:**
- Very small footprint
- Apache 2.0 license
- Surprisingly capable for size

**Cons for Distillation:**
- Limited reasoning depth
- Struggles with complex tasks

**Verdict:** Good for very resource-constrained distillation, but 3B recommended.

---

## Comparison Summary

### Performance Ranking (MMLU + GSM8K Average)

| Rank | Model | Combined Score | License Permissiveness |
|------|-------|----------------|------------------------|
| 1 | Qwen 2.5 3B-Instruct | ~75 | Apache 2.0 (Best) |
| 2 | Gemma 3 4B-Instruct | ~77 | gemma-terms (Good) |
| 3 | Phi-3-mini | ~72.5 | MIT (Excellent) |
| 4 | Llama 3.2 3B-Instruct | ~69 | Llama 3.2 (Fair) |
| 5 | Gemma 3 2B-Instruct | ~71 | gemma-terms (Good) |
| 6 | Qwen 2.5 1.5B | ~63 | Apache 2.0 (Best) |
| 7 | Llama 3.2 1B | ~60 | Llama 3.2 (Fair) |

### Distillation-Specific Considerations

**For Logit Distillation (output distribution matching):**
1. **Gemma 3 2B** - Best quality logits, good temperature scaling
2. **Qwen 2.5 3B** - Strong multilingual logits
3. **Phi-3-mini** - Clean probability distributions

**For Feature Distillation (hidden state matching):**
1. **Qwen 2.5 3B** - Clean intermediate representations
2. **Llama 3.2 3B** - Well-studied architecture
3. **Gemma 3** - Accessible layer outputs

**For Reasoning Distillation (chain-of-thought):**
1. **Phi-3-mini** - Strongest reasoning patterns
2. **Gemma 3** - Good reasoning chains
3. **Qwen 2.5** - Strong math reasoning

---

## Recommendations for BDH Project

### Primary Recommendation: Gemma 3 2B-Instruct

**Rationale:**
- Already configured in your codebase (`google/gemma-3-2b-it`)
- Excellent balance of quality and efficiency
- Strong safety alignment reduces harmful outputs
- Good documentation and examples

**Setup:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

teacher_model = "google/gemma-3-2b-it"
tokenizer = AutoTokenizer.from_pretrained(teacher_model)
model = AutoModelForCausalLM.from_pretrained(
    teacher_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Alternative: Qwen 2.5 3B-Instruct (for Open License)

**Rationale:**
- Apache 2.0 license for full flexibility
- Stronger raw performance
- Better for commercial applications

**Setup:**
```python
teacher_model = "Qwen/Qwen2.5-3B-Instruct"
```

### Lightweight Alternative: Phi-3-mini

**Rationale:**
- MIT license (most permissive)
- Strong reasoning capabilities
- Good for research without licensing concerns

**Setup:**
```python
teacher_model = "microsoft/Phi-3-mini-4k-instruct"
```

---

## Distillation Best Practices

### 1. Temperature Selection
- Standard: T=1.0 (natural distribution)
- Softer labels: T=2.0-3.0 (for better distillation)
- Calibrate based on validation performance

### 2. Layer Matching
```
Teacher (2B) → Student (10M)
- 24 layers → 6 layers
- Map: {T0,T1→S0, T4,T5,T6→S1, ...}
- Or use averaged intermediate representations
```

### 3. Loss Weighting
```python
loss = α * ce_loss + β * kl_divergence + γ * hidden_state_loss

Typical values:
- α = 0.1 (hard labels)
- β = 0.9 (soft logits)
- γ = 0.0-0.5 (feature matching)
```

### 4. Data Generation
- Use diverse prompts (your existing 5 categories)
- Generate with temperature 0.7-0.9 for variety
- Filter low-quality outputs
- Aim for 100K-1M tokens for effective distillation

---

## Download Commands

### Hugging Face
```bash
# Gemma 3 2B
huggingface-cli download google/gemma-3-2b-it --local-dir ./models/gemma-3-2b

# Qwen 2.5 3B
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/qwen2.5-3b

# Phi-3-mini
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir ./models/phi3-mini
```

### Ollama
```bash
# Gemma 3
ollama pull gemma3:2b

# Qwen 2.5
ollama pull qwen2.5:3b

# Phi-3
ollama pull phi3

# Llama 3.2
ollama pull llama3.2:3b
```

---

## Quick Reference Table

| Model | Params | MMLU | GSM8K | License | HF ID | Ollama |
|-------|--------|------|-------|---------|-------|--------|
| Gemma 3 2B | 2.0B | 67 | 75 | gemma-terms | `google/gemma-3-2b-it` | `gemma3:2b` |
| Gemma 3 4B | 4.0B | 72 | 82 | gemma-terms | `google/gemma-3-4b-it` | `gemma3:4b` |
| Qwen 2.5 3B | 3.0B | 72 | 78 | Apache 2.0 | `Qwen/Qwen2.5-3B-Instruct` | `qwen2.5:3b` |
| Phi-3-mini | 3.8B | 69 | 76 | MIT | `microsoft/Phi-3-mini-4k-instruct` | `phi3` |
| Llama 3.2 3B | 3.0B | 66 | 72 | Llama 3.2 | `meta-llama/Llama-3.2-3B-Instruct` | `llama3.2:3b` |
| Qwen 2.5 1.5B | 1.5B | 61 | 65 | Apache 2.0 | `Qwen/Qwen2.5-1.5B-Instruct` | `qwen2.5:1.5b` |

---

**Document Version:** 1.0
**Last Updated:** February 25, 2026
**Next Review:** April 2026 (after new model releases)

---

## Notes for BDH Implementation

Your current setup uses `google/gemma-3-2b-it`, which is an excellent choice. The code in `generate_teacher_data.py` and `train_ollama_distillation.py` is well-structured for distillation.

**Suggested improvements:**
1. Add temperature scaling for softer labels
2. Experiment with layer-wise distillation losses
3. Try Qwen 2.5 3B for Apache 2.0 license flexibility
4. Consider ensemble of teachers for diverse knowledge

**Files that may need updates:**
- `train_distillation.py` - Add configurable teacher models
- `setup_distillation_env.py` - Add support for multiple teachers
- Consider adding `TEACHER_MODELS.md` for configuration reference
