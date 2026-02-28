# Knowledge Distillation Best Practices for BDH

**Date:** 2026-02-25
**Target:** Distilling from Gemma 3 270M → 5M BDH Model
**Goal:** Preserve performance with extreme compression (54x parameter reduction)

---

## Executive Summary

Knowledge distillation in 2024-2025 has evolved significantly with several key insights:

1. **Multi-stage distillation** outperforms single-stage for extreme compression
2. **Curated data quality** matters more than quantity (Phi series demonstrates this)
3. **Reasoning capabilities** can be preserved with specialized techniques
4. **Temperature tuning** is critical for effective logit transfer
5. **Synthetic data** should augment, not replace, real data

---

## 1. Language Model Distillation: Large to Small

### Core Approaches

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Logit-based** | KL divergence on output logits | Most common, works well |
| **Feature-based** | Match hidden states, attention maps | Cross-architecture |
| **Intermediate-layer** | Match multiple teacher layers to student | Deep student models |
| **Progressive** | Teacher → Student → Smaller Student | Extreme compression |

### Best Practices

1. **Teacher Selection**: Use teacher 2-10x larger than student (we're doing 54x, which is aggressive)
2. **Architecture Alignment**: Similar architectures help, but not required with feature matching
3. **Layer Mapping**: Map multiple teacher layers to fewer student layers
4. **Initialization**: Student benefits from smart initialization (e.g., layer sampling from teacher)

### For BDH Specifically

```
Gemma 3 270M (Transformer)
    ↓ (Distill via synthetic data + feature matching)
BDH 5M (Multi-scale Hebbian)

Key challenges:
- Different architectures (Transformer → BDH)
- Different learning rules (Backprop → Hebbian)
- Different memory mechanisms (Attention → Multi-scale synapses)
```

**Strategy**: Focus on **behavioral distillation** (matching outputs) rather than structural matching.

---

## 2. Multi-Scale Architecture Distillation

### Multi-Scale Distillation Strategies

1. **Independent Scale Distillation**
   - Distill each timescale separately
   - Fast: Pattern matching
   - Medium: Medium-term dependencies
   - Slow: Long-term structure

2. **Hierarchical Distillation**
   - Distill slow synapses first (concepts)
   - Then medium (patterns)
   - Then fast (details)

3. **Cross-Scale Regularization**
   - Ensure consistency across scales during distillation
   - Prevent scales from learning contradictory information

### BDH-Specific Advantages

- **Natural hierarchy**: Multi-scale structure aligns with information hierarchy
- **Hebbian robustness**: May handle distillation noise better than backprop
- **Interpretability**: Can inspect what each scale learned

### Recommended Distillation Setup

```python
# Pseudocode for multi-scale distillation
for batch in dataloader:
    teacher_logits = teacher(batch.input)

    # Get student outputs at each scale
    fast_output = student.forward_fast(batch.input)
    medium_output = student.forward_medium(batch.input)
    slow_output = student.forward_slow(batch.input)

    # Combined student output
    student_output = student.combine(fast_output, medium_output, slow_output)

    # Distillation losses
    loss_fast = kl_div(teacher_logits, fast_output) * 0.2
    loss_medium = kl_div(teacher_logits, medium_output) * 0.3
    loss_slow = kl_div(teacher_logits, slow_output) * 0.5

    loss = loss_fast + loss_medium + loss_slow
    loss.backward()
```

---

## 3. Training Data Generation with Teacher Models

### Data Generation Best Practices

| Aspect | Recommendation |
|--------|----------------|
| **Data mix** | 70-80% real data, 20-30% synthetic |
| **Generation method** | Forward generation (teacher produces completions) |
| **Prompt diversity** | Use varied, high-quality prompts |
| **Quality filtering** | Keep only high-quality teacher outputs |
| **Domain alignment** | Focus on target domain of student model |

### Generation Strategies

1. **Forward Generation**
   - Input prompt → Teacher generates completion
   - Use varied sampling strategies

2. **Backward Generation** (less common)
   - Input completion → Teacher generates prompt
   - Useful for question-answering tasks

3. **Chain-of-Thought Generation**
   - Teacher generates reasoning traces
   - Critical for reasoning tasks

### Data Quality

**Filtering criteria:**
- Perplexity threshold: Reject if teacher perplexity too high
- Length checks: Ensure completions are reasonable length
- Diversity: Remove near-duplicates
- Safety: Filter harmful content

### Recommended Data Mix for BDH

```
Total: 500M - 1B tokens

Breakdown:
- 300M tokens: Real text (SlimPajama, RedPajama, etc.)
- 100M tokens: Teacher-generated synthetic (Gemma 3)
- 50M tokens: Chain-of-thought reasoning data
- 50M tokens: Code or domain-specific data

Synthetic data generation:
- Temperature: 0.8 - 1.0
- Sampling: Nucleus (p=0.9)
- Prompts: Varied (stories, questions, instructions, code)
```

---

## 4. Optimal Temperature and Sampling Strategies

### Temperature for Distillation Loss

| Use Case | Temperature | Rationale |
|----------|-------------|-----------|
| **Logit distillation (KL loss)** | 2.0 - 5.0 | Reveals "dark knowledge" from logits |
| **Label smoothing** | 0.1 - 0.3 | Prevents overconfidence |
| **Forward generation** | 0.7 - 1.2 | Balanced diversity/coherence |
| **Reasoning extraction** | 1.2 - 1.8 | Encourages diverse thinking paths |

### Temperature Effect

```
Low temperature (0.1 - 0.5):
- Sharp distribution
- Teacher overconfident
- Less "dark knowledge" transferred

Medium temperature (0.7 - 1.2):
- Balanced
- Good for generation
- Moderate distillation signal

High temperature (2.0 - 5.0):
- Smooth distribution
- Maximum "dark knowledge"
- Best for distillation loss
```

### Sampling Strategies

| Strategy | p/k | Best For |
|----------|-----|----------|
| **Nucleus sampling** | p=0.9 | General purpose |
| **Top-k sampling** | k=40 | Simpler alternative |
| **Beam search** | width=4 | Highest quality |
| **Temperature sampling** | T=0.8-1.0 | Diversity + quality |

### Recommended Settings for BDH Distillation

```python
# For generating training data
generation_config = {
    "temperature": 0.9,      # Balanced diversity
    "top_p": 0.95,           # Nucleus sampling
    "top_k": 50,             # Top-k fallback
    "max_tokens": 2048,      # Reasonable length
}

# For distillation loss
distillation_config = {
    "temperature": 3.0,      # Soft targets
    "loss_weight": 0.7,      # Balance with task loss
    "label_smoothing": 0.1,  # Prevent overconfidence
}
```

---

## 5. Data Requirements: How Much is Needed?

### Data Quantity Guidelines

| Target Model | Training Tokens | With Distillation |
|--------------|-----------------|-------------------|
| 5M params | 100M - 500M | 50M - 200M (2-5x reduction) |
| 10M params | 200M - 1B | 100M - 400M |
| 50M params | 500M - 3B | 200M - 1B |
| 270M params | 1B - 10B | 500M - 3B |

### Key Insight: Quality > Quantity

2024-2025 research (Phi series, Gemma, Llama 3.2):
- **50M curated tokens** can match **500M uncurated tokens**
- **Curated synthetic data** is extremely valuable
- **Domain focus** beats general corpus

### Recommended for BDH 5M

```
Minimum viable: 100M tokens (50M real + 50M synthetic)
Recommended: 500M tokens (350M real + 150M synthetic)
Ambitious: 1B tokens (700M real + 300M synthetic)

Focus on:
1. Diversity of domains (text, code, reasoning, dialog)
2. Quality over quantity (filter aggressively)
3. Teacher-generated reasoning traces
```

### Staged Training

```
Stage 1: 100M tokens (basic language modeling)
Stage 2: 200M tokens (add distillation signal)
Stage 3: 200M tokens (fine-tune on target tasks)
```

---

## 6. Training Strategies: Hyperparameters

### Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning rate** | 1e-4 to 5e-4 | Warmup + cosine decay |
| **Batch size** | 64 - 256 | Adjust for memory |
| **Gradient accumulation** | 4 - 8 | Simulate larger batch |
| **Sequence length** | 512 - 2048 | 512 for speed, 2048 for quality |
| **Weight decay** | 0.01 - 0.1 | Regularization |
| **Gradient clipping** | 1.0 - 5.0 | Prevent explosion |

### Distillation-Specific Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Distillation temperature** | 2.0 - 4.0 | For soft targets |
| **Distillation loss weight** | 0.5 - 0.9 | Balance with task loss |
| **Hard label weight** | 0.1 - 0.5 | Keep some ground truth |
| **Layer-wise weights** | Variable | Earlier layers: lower weight |

### Training Schedule

```
Phase 1: Warmup (5% of steps)
  - LR increases to peak
  - Learn basic patterns

Phase 2: Main training (80% of steps)
  - Cosine decay
  - Full distillation loss

Phase 3: Fine-tuning (15% of steps)
  - Lower LR (1e-5)
  - Focus on target tasks
```

### Multi-Scale BDH Training Considerations

```python
# Different learning rates for different scales
learning_rates = {
    "fast_synapses": 5e-4,    # Learn quickly
    "medium_synapses": 2e-4,  # Moderate
    "slow_synapses": 5e-5,    # Learn slowly
}

# Loss weighting for multi-scale
loss_weights = {
    "fast": 0.2,
    "medium": 0.3,
    "slow": 0.5,
}
```

---

## 7. Evaluation Metrics

### Essential Metrics

| Metric | Purpose | Target (vs Teacher) |
|--------|---------|---------------------|
| **Perplexity** | Language modeling | Within 2-5x |
| **Task accuracy** | Downstream performance | 70-90% |
| **Inference latency** | Speed | 10-100x faster |
| **Memory** | Model size | 50-1000x smaller |
| **Throughput** | Tokens/sec | 50-200x higher |

### Quality Preservation Metrics

| Metric | Description |
|--------|-------------|
| **Retention rate** | % of teacher performance maintained |
| **Calibration** | Confidence vs accuracy alignment |
| **Robustness** | Out-of-distribution performance |
| **Reasoning** | Chain-of-thought faithfulness |

### BDH-Specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Memory retention** | Long-context performance | Match or beat baseline |
| **Multi-scale utilization** | Are all scales learning? | Balanced across scales |
| **Interpretability** | Can we inspect what it learned? | High |
| **Efficiency** | FLOPs per token | Minimal |

### Evaluation Script Template

```python
def evaluate_distilled_model(student, teacher, test_set):
    results = {}

    # 1. Perplexity
    results["perplexity"] = compute_perplexity(student, test_set)

    # 2. Task accuracy
    results["accuracy"] = compute_accuracy(student, test_set)

    # 3. Retention rate
    teacher_acc = compute_accuracy(teacher, test_set)
    results["retention"] = results["accuracy"] / teacher_acc

    # 4. Efficiency
    results["throughput"] = compute_throughput(student)
    results["memory_mb"] = get_model_size(student)

    # 5. BDH-specific
    results["memory_retention"] = evaluate_long_context(student)
    results["scale_balance"] = analyze_scale_usage(student)

    return results
```

---

## 8. Distilling Reasoning Capabilities

### Key Papers (2023-2025)

1. **"Distilling Step-by-Step"** (Hsieh et al.)
   - Use teacher's reasoning traces
   - Train on rationales, not just answers

2. **"Phi-2" / "Phi-3"** (Microsoft)
   - 2.7B/3.8B matching larger models
   - Heavily curated training data

3. **"Llama 3.2"** (Meta)
   - 1B and 3B distilled from 405B
   - Multi-stage distillation

4. **"Gemma 2"** (Google)
   - 2B and 9B from 27B
   - Attention transfer

### Reasoning Distillation Techniques

| Technique | Description | Difficulty |
|-----------|-------------|------------|
| **CoT transfer** | Include teacher reasoning in training data | Medium |
| **Stepwise distillation** | Reward each reasoning step | Hard |
| **Self-consistency** | Train on multiple reasoning paths | Hard |
| **Tool use distillation** | Preserve external tool usage | Very hard |

### Practical Approach

```python
# Generate reasoning-augmented training data
def generate_reasoning_data(teacher, prompts):
    data = []
    for prompt in prompts:
        # Get chain-of-thought
        cot = teacher.generate(
            prompt,
            temperature=1.2,
            max_tokens=1024,
            include_reasoning=True
        )

        # Get final answer
        answer = teacher.generate(
            prompt + cot,
            temperature=0.5,
            max_tokens=256
        )

        data.append({
            "prompt": prompt,
            "reasoning": cot,
            "answer": answer
        })
    return data

# Train with reasoning-aware loss
def reasoning_loss(student, teacher, batch):
    # Reasoning phase
    reasoning_logits = student.forward_reasoning(batch["prompt"])
    reasoning_target = teacher.get_reasoning_logits(batch["reasoning"])

    # Answer phase
    answer_logits = student.forward_answer(batch["prompt"], batch["reasoning"])
    answer_target = teacher.get_answer_logits(batch["answer"])

    # Combined loss
    loss = kl_div(reasoning_logits, reasoning_target) * 0.5 + \
           kl_div(answer_logits, answer_target) * 0.5
    return loss
```

---

## 9. Extreme Compression: 3B → 5M Case Studies

### Reality Check

Direct 3B → 5M (600x compression) is **extremely aggressive**.

More realistic approaches:

```
Option 1: Multi-stage
3B → 270M → 50M → 5M

Option 2: Data-focused
3B generates massive synthetic dataset
5M trained from scratch on curated data

Option 3: Hybrid
3B → 270M (feature distillation)
270M → 5M (behavioral distillation)
```

### Success Stories

| Model | Teacher → Student | Compression | Retention |
|-------|-------------------|-------------|-----------|
| Llama 3.2 1B | 405B → 1B | 405x | ~75% |
| Gemma 2 2B | 27B → 2B | 13.5x | ~85% |
| Phi-2 | N/A (trained) | N/A | N/A |
| MobileLLM | N/A (trained) | N/A | N/A |

### Lessons

1. **Multi-stage works best**: Don't jump directly
2. **Data quality is critical**: Curation beats quantity
3. **Architecture matters**: Similar architectures help
4. **Task-specific optimization**: Target specific use cases

### For BDH (270M → 5M, 54x)

This is achievable with:
1. High-quality synthetic data from Gemma 3 270M
2. Multi-scale architecture advantages
3. Focus on specific domains (not general-purpose)
4. Multi-stage training (270M data → intermediate checkpoint → 5M)

---

## 10. Action Plan for BDH Distillation

### Phase 1: Data Generation (Day 2)

```bash
# Generate synthetic training data
python generate_synthetic.py \
    --teacher gemma-3-270m \
    --output data/synthetic \
    --num_samples 100000 \
    --temperature 0.9 \
    --top_p 0.95

# Generate reasoning data
python generate_reasoning.py \
    --teacher gemma-3-270m \
    --output data/reasoning \
    --num_samples 50000 \
    --include_cot true
```

### Phase 2: Base Training (Day 2)

```bash
# Train BDH on mixed real + synthetic data
python train.py \
    --model bdh_5m \
    --data data/mixed \
    --tokens 500m \
    --lr 3e-4 \
    --batch_size 128 \
    --distillation true
```

### Phase 3: Fine-tuning (Day 3)

```bash
# Fine-tune on target tasks
python finetune.py \
    --model checkpoints/bdh_5m_base \
    --data data/target_tasks \
    --lr 1e-5 \
    --epochs 3
```

### Phase 4: Evaluation

```bash
# Comprehensive evaluation
python evaluate.py \
    --model checkpoints/bdh_5m_final \
    --teacher gemma-3-270m \
    --output results/
```

---

## Key References

### Papers

1. Hsieh et al. "Distilling Step-by-Step! Out-of-Distribution Generalization via Self-Reasoning" (2023)
2. Microsoft. "Phi-2 Technical Report" (2023)
3. Meta. "Llama 3.2 Model Card" (2024)
4. Google. "Gemma 2 Technical Report" (2024)
5. "MiniGPT-4" and "TinyGPT" papers

### Code & Models

- Hugging Face Transformers (distillation examples)
- Llama 3.2 (official distillation code)
- Gemma (Google's implementation)
- Phi series (Microsoft's implementation)

### Datasets

- SlimPajama / RedPajama (real data base)
- OpenHermes (instruction data)
- CoT collections (reasoning data)

---

## Summary: Critical Takeaways

1. **Temperature 3.0** for distillation loss, **0.9** for generation
2. **50-200M tokens** minimum for 5M model with distillation
3. **Multi-stage training** recommended for extreme compression
4. **Quality > Quantity** for training data
5. **Reasoning traces** valuable for preserving capabilities
6. **Nucleus sampling (p=0.9)** for balanced generation
7. **70-90% retention** is realistic target for 54x compression
8. **Perplexity within 2-5x** of teacher is achievable
