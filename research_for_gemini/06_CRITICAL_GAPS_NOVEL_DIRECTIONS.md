# Critical Research Gaps + Novel Directions: What the World Hasn't Seen

> **Purpose**: This document synthesizes all research findings to identify the biggest gaps in current AI architectures and proposes novel directions that could lead to breakthrough results. Designed for Gemini deep research to generate actionable research proposals.

---

## PART 1: THE BIG PICTURE — WHERE WE ARE AND WHERE WE'RE NOT

### 1.1 The Current Landscape (2026)

**What works at scale:**
- Hybrid architectures (Transformer + SSM/linear attention) at 10B-100B+ parameters
- MoE for sparse activation (DeepSeek-V3: 671B total, 37B active)
- RL post-training for reasoning (DeepSeek-R1, o1, o3)
- Progressive context extension (Qwen: 4K → 256K → 1M)
- Knowledge distillation for model compression

**What doesn't work at scale:**
- Pure sub-quadratic models beyond ~3B parameters
- Pure RNNs for long-range recall
- Fixed decay rates (data-independent)
- Byte-level tokenization without subword fallback

### 1.2 The Fundamental Tensions

| Tension | Side A | Side B |
|---------|--------|--------|
| **Memory vs. Recall** | O(1) memory (SSM, linear) | Perfect recall (full attention) |
| **Efficiency vs. Quality** | Fast, cheap training | State-of-the-art performance |
| **Biological vs. Engineering** | Brain-plausible mechanisms | GPU-optimized mechanisms |
| **Interpretability vs. Performance** | Readable internal states | Black-box optimization |
| **Local vs. Global** | Local learning rules (Hebbian) | Global optimization (backprop) |

---

## PART 2: CRITICAL RESEARCH GAPS

### Gap 1: The Hebbian-Scaling Gap

**Problem:** Despite 50+ years of Hebbian theory, no Hebbian-based architecture has been trained at scale for language modeling. BDH is the closest attempt but at only 70M params.

**Why it matters:** Hebbian learning is the brain's learning rule. If it can scale, it would enable:
- Continual learning without catastrophic forgetting
- Energy-efficient training (local updates, no backprop through time)
- Interpretable memory (read the state matrix to see what the model learned)

**What's missing:**
- Proper outer product Hebbian update (BDH uses element-wise)
- Data-dependent gating for Hebbian learning rate
- Hardware-optimized kernels for Hebbian state updates
- Scaling laws for Hebbian architectures

**Novel direction:** **Hebbian-Gated DeltaNet** — combine Hebbian learning with data-dependent gating:
```
S_t = exp(g_t) * S_{t-1} + η(input) * (Q ⊗ V - S_{t-1} @ K @ K^T)
```
Where η(input) is an input-dependent Hebbian learning rate, and the delta rule enables both writing and erasing.

### Gap 2: The Multi-Scale Expressiveness Gap

**Problem:** Multi-scale memory (BDH's 3 decay rates, RetNet's multi-scale retention) is empirically effective but theoretically unexplained. What computational problems can multi-scale solve that single-scale cannot?

**Why it matters:** Language operates at multiple timescales. A theory of multi-scale expressiveness would guide architecture design.

**What's missing:**
- Formal analysis of multi-scale state tracking capacity
- Optimal number of scales (BDH uses 3, RetNet uses per-head)
- Optimal decay rate selection (BDH uses hand-tuned [0.95, 0.99, 0.995])
- Connection to wavelet transforms and multi-resolution analysis

**Novel direction:** **Adaptive Multi-Scale Hebbian Network** — let the model learn its own decay rates:
```
γ_i(input) = sigmoid(W_γ_i(input))  # input-dependent decay rates
E_i(t+1) = γ_i(input) * E_i(t) + η * (Q ⊗ V)
```
This combines BDH's multi-scale with Mamba's selectivity.

### Gap 3: The Cross-Architecture Distillation Gap

**Problem:** Distilling Transformer knowledge into SSM/linear attention models is still an art, not a science. MOHAWK works for Mamba, but what about BDH?

**Why it matters:** If we can efficiently distill frontier models (GPT-4, R1) into efficient architectures, we get GPT-4 quality at 1/100th the inference cost.

**What's missing:**
- General theory of cross-architecture knowledge transfer
- Optimal initialization for BDH from Transformer weights
- How to distill reasoning capabilities into linear attention models
- Whether BDH's multiplicative gating helps or hurts distillation

**Novel direction:** **Architecture-Agnostic Distillation via CKA** — use Centered Kernel Alignment to match hidden states regardless of architecture:
- Match teacher and student representations at the kernel level
- No need for dimension matching or learned projections
- Works for any architecture pair (Transformer → BDH, Mamba → RWKV, etc.)

### Gap 4: The Reasoning-in-Compact-Models Gap

**Problem:** Small models (≤3B) cannot learn long chain-of-thought reasoning from distillation. They need shorter, simpler reasoning chains matching their intrinsic capacity.

**Why it matters:** If we can teach reasoning to compact models (70M-500M), we get reasoning-capable AI that runs on phones.

**What's missing:**
- Theory of "reasoning capacity" as a function of model size
- Optimal CoT granularity for each model scale
- Whether BDH's architecture (multiplicative gating, Hebbian memory) helps or hurts reasoning
- How to distill self-verification and backtracking into compact models

**Novel direction:** **Hebbian Reasoning Memory** — use BDH's state matrix as an explicit reasoning workspace:
- The state matrix accumulates "reasoning steps" over time
- Multi-scale memory enables short-term (current step) and long-term (overall plan) reasoning
- Multiplicative gating enables "attention to reasoning" — the model can focus on specific reasoning steps

### Gap 5: The Hardware Co-Design Gap

**Problem:** No hardware-optimized kernel exists for Hebbian state updates, multi-scale decay, or BDH's multiplicative gating.

**Why it matters:** An architecture is only as good as its kernel. FlashAttention made Transformers practical. BDH needs its own FlashAttention.

**What's missing:**
- CUDA kernel for parallel Hebbian state updates
- Triton/Pallas kernel for multi-scale decay
- Sparse activation kernel that skips inactive neurons
- Kernel for multiplicative gating with gradient checkpointing

**Novel direction:** **FlashHebbian** — a CUDA kernel that:
- Computes K^T @ V in parallel (the Hebbian update)
- Applies multi-scale decay in a single fused operation
- Exploits sparsity by skipping zero activations
- Uses tensor cores for the outer product

### Gap 6: The Interpretability Quantification Gap

**Problem:** BDH claims interpretability (readable state matrix), but this is qualitative, not measured against a standard.

**Why it matters:** If BDH's state matrix is truly interpretable, it could replace black-box models in safety-critical applications.

**What's missing:**
- Quantitative metrics for interpretability (e.g., "synapse X encodes concept Y with confidence Z")
- Comparison to Sparse Autoencoders (SAEs) for interpretability
- Whether BDH's monosemantic synapses are more interpretable than transformer SAE features
- Tools for visualizing and analyzing the state matrix

**Novel direction:** **Synapse-Level Mechanistic Interpretability** — treat each entry E[i,j] as a measurable feature:
- Correlate synapse values with human-interpretable concepts
- Compare BDH synapses to SAE features in Transformers
- Build a "synapse dictionary" that maps state entries to concepts

### Gap 7: The Continual Learning Gap

**Problem:** No current architecture supports continual learning (learning new knowledge without forgetting old knowledge) at scale.

**Why it matters:** The brain learns continually. AI models are trained once and frozen. Continual learning would enable AI that improves over time.

**What's missing:**
- How BDH's Hebbian state matrix could enable continual learning
- Whether multi-scale decay naturally supports forgetting old vs. retaining important knowledge
- How to prevent catastrophic forgetting in linear attention models
- Benchmarks for continual learning in language models

**Novel direction:** **Hebbian Continual Learning** — use BDH's state matrix as a continual learning mechanism:
- The state matrix naturally accumulates knowledge over time
- Multi-scale decay provides natural forgetting of irrelevant information
- Hebbian learning is inherently local and continual (no need for backprop)
- The slow scale (γ=0.995) retains important knowledge long-term

---

## PART 3: NOVEL ARCHITECTURE PROPOSALS

### Proposal 1: BDH-2 — The Gated Hebbian Architecture

**Core idea:** Fix BDH's biggest gaps while preserving its unique advantages.

**Changes from BDH:**
1. **Outer product Hebbian update:** `E = γ·E + η·(Q ⊗ V)` instead of element-wise
2. **Data-dependent gating:** `γ(input) = sigmoid(W_γ(input))` instead of fixed decay
3. **Local convolution:** Add causal Conv1d (kernel_size=4) before linear attention
4. **Hybrid attention:** 3:1 ratio (3 BDH layers + 1 full attention layer), following Qwen3.5
5. **Multi-token prediction:** Add MTP head for t+1, t+2 prediction
6. **Proper RoPE:** Enable RoPE on the full-attention layers

**Expected improvements:**
- 2-3× better quality at same parameter count
- Stable training (data-dependent gating prevents vanishing gradients)
- Long-context capability (hybrid attention for recall)
- Richer training signals (MTP)

### Proposal 2: HeLa-BDH — Hebbian Learning + Associative Memory

**Core idea:** Combine BDH with HeLa-Mem's associative memory mechanism.

**Architecture:**
- BDH as the base model
- HeLa-Mem's Hebbian memory module as an external memory
- Cross-attention between BDH state and HeLa-Mem memory
- Enables retrieval-augmented reasoning

**Expected improvements:**
- Near-perfect associative recall (HeLa-Mem's strength)
- BDH's efficiency for local processing
- Interpretable memory (both BDH state and HeLa-Mem memory are readable)

### Proposal 3: Multi-Scale DeltaNet + Hebbian

**Core idea:** Unify DeltaNet's delta rule with BDH's multi-scale Hebbian learning.

**State update:**
```
For each scale i:
  S_i(t) = γ_i * S_i(t-1) + η_i * (v_t @ k_t^T - S_i(t-1) @ k_t @ k_t^T)

Combined: S = Σ w_i * S_i
```

**Why this is novel:**
- Delta rule enables writing AND erasing (BDH only accumulates)
- Multi-scale provides different timescales for different types of memory
- The combination could solve the recall-efficiency wall

### Proposal 4: Sparse MoE-BDH

**Core idea:** Exploit BDH's ~5% sparse activations as a free MoE.

**Mechanism:**
- Identify which neurons are active for each token
- Route computation only to active neurons
- Each "expert" is a subset of neurons
- No routing network needed — sparsity determines routing

**Expected improvements:**
- 20× computation reduction (if 5% sparsity is real)
- No routing overhead (unlike traditional MoE)
- Dynamic computation per token

### Proposal 5: Complex-Valued Hebbian Network

**Core idea:** Inspired by Mamba-3's complex-valued SSM, use complex numbers in BDH's state matrix.

**State update:**
```
E(t+1) = γ * E(t) * e^{iθ(input)} + η * (Q ⊗ V)
```

**Why this is novel:**
- Complex rotation enables state tracking (Mamba-3's key insight)
- Hebbian learning provides associative memory
- The combination could solve both recall and state tracking

---

## PART 4: RESEARCH AGENDA — WHAT TO DO NEXT

### Phase 1: Fix BDH's Critical Bugs (Week 1-2)
1. Implement outer product Hebbian update (Q ⊗ V instead of Q ⊙ V)
2. Enable RoPE in multi-scale BDH
3. Fix vocab mismatch in distillation (add projection layer)
4. Benchmark against baseline

### Phase 2: Add Data-Dependent Gating (Week 3-4)
1. Implement input-dependent decay rates: γ(input) = sigmoid(W_γ(input))
2. Implement input-dependent Hebbian learning rate: η(input)
3. Compare training stability vs. fixed rates
4. Benchmark quality improvement

### Phase 3: Add Local Convolution + Hybrid Attention (Week 5-6)
1. Add causal Conv1d (kernel_size=4) before linear attention
2. Add 1 full-attention layer per 4 BDH layers (3:1 ratio)
3. Benchmark against pure BDH and pure Transformer

### Phase 4: Cross-Architecture Distillation (Week 7-8)
1. Implement CKA-based hidden state matching
2. Distill from Qwen3.5-0.8B into BDH
3. Compare with logit-only distillation
4. Measure reasoning capability transfer

### Phase 5: Multi-Token Prediction (Week 9-10)
1. Add MTP heads for t+1, t+2 prediction
2. Train with λ * L_MTP auxiliary loss
3. Measure improvement in effective memory
4. Repurpose MTP for speculative decoding

### Phase 6: Scale to 500M-1B Parameters (Week 11-16)
1. Design scaling strategy (layer count, width, heads)
2. Train on high-quality data (books, papers, code)
3. Benchmark against comparable Transformer and Mamba models
4. Publish results

---

## PART 5: PAPERS TO READ FOR NOVEL DIRECTIONS

### Hebbian + Memory
1. **HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents** — arXiv:2604.16839
2. **Hebbian Memory-Augmented Recurrent Networks: Engram Neurons** — arXiv:2507.21474
3. **Hebbian Learning through the Lens of Sparse Autoencoders** — OpenReview 2026
4. **Hebbian learning the local structure of language** — arXiv:2503.02057
5. **Fast Weight Programmers** — Schmidhuber, 1992

### Complex-Valued Networks
6. **Mamba-3: Improved Sequence Modeling using State Space Principles** — arXiv:2603.15569
7. **Complex-Valued Neural Networks: A Survey** — arXiv (search for latest)

### Continual Learning
8. **Continual Learning in Language Models** — search arXiv 2025-2026
9. **Hebbian Continual Learning** — search arXiv

### Sparse Computation
10. **Sparse Autoencoders for LLM Interpretability** — various 2024-2026
11. **Dynamic Sparse Training** — search arXiv

### Hardware Co-Design
12. **FlashAttention-2** — arXiv:2307.08621
13. **FlashLinearAttention** — GitHub: fla-org/flash-linear-attention
14. **Triton: GPU Programming** — OpenAI

### Unified Formulations
15. **The FLA Library: Unified Gated Linear Attention** — GitHub: fla-org/flash-linear-attention
16. **Transformers are SSMs** — arXiv:2405.21060
