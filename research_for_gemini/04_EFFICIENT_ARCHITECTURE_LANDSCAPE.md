# Efficient Architecture Landscape: SSMs, Linear Attention, Hybrids (2024-2026)

> **Purpose**: Complete technical comparison of all sub-quadratic architectures. Designed for Gemini deep research to identify where BDH fits and what it can learn from each architecture.

---

## PART 1: COMPLETE ARCHITECTURE COMPARISON

### 1.1 Core Mechanism Comparison

| Architecture | Core Mechanism | State Type | Train Parallel? | Inference | State Size |
|---|---|---|---|---|---|
| **Transformer** | Softmax Attention | KV Cache (O(N)) | Yes | O(N) memory | N × d |
| **Linear Transformer** | Kernel Attention (no softmax) | Matrix (d × d) | Yes | O(1) | d² |
| **RetNet** | Multi-Scale Retention | Matrix (d × d) | Yes (3 modes) | O(1) | d² |
| **GLA** | Gated Linear Attention | Matrix (d × d) | Yes (chunkwise) | O(1) | d² |
| **DeltaNet** | Delta Rule (Householder) | Matrix (d × d) | Yes (chunkwise) | O(1) | d² |
| **Based** | Taylor Linear + SWA | Matrix + KV (window) | Yes | O(window) | d² + W×d |
| **GSA** | Two-pass GLA + softmax | Matrix (slots) | Yes | O(1) | slots×d |
| **Mamba-1** | Selective SSM (scan) | Vector (N) | Yes (parallel scan) | O(1) | N |
| **Mamba-2** | SSD (matrix form) | Matrix (d × d) | Yes | O(1) | d² |
| **Mamba-3** | Complex-valued SSM + MIMO | Matrix (d × d) | Yes | O(1) | d²/2 |
| **RWKV-5/6** | Token-shift + WKV | Vector + diagonal | Yes | O(1) | d |
| **RWKV-7** | Generalized Delta Rule | Matrix (d × d) | Yes | O(1) | d² |
| **RWKV-8** | DeepEmbed + ROSA | Sparse + automaton | Partial | O(1) | Variable |
| **Griffin** | RG-LRU + Local Attn | Vector + KV (window) | Yes | O(window) | d + W×d |
| **Hawk** | RG-LRU (pure RNN) | Vector (d) | Yes | O(1) | d |
| **Jamba** | Mamba + Attention + MoE | Mixed | Yes | Mixed | Mixed |
| **Samba** | Mamba + SWA | Vector + KV (window) | Yes | O(window) | N + W×d |
| **Zamba** | Mamba + Shared Attention | Vector + KV (global) | Yes | O(N) | N + d² |
| **BDH** | Hebbian State Matrix | Matrix (d × d) | Yes | O(1) | d² |

### 1.2 Expressiveness & Theoretical Limits

| Architecture | State Tracking | Regular Languages | TC⁰ Limit | Associative Recall |
|---|---|---|---|---|
| Transformer | Limited | No | Bounded by TC⁰ | Excellent |
| Mamba-1 | Yes | Yes | Exceeds TC⁰ | Moderate |
| Mamba-2 | Partial | Partial | Exceeds TC⁰ | Moderate |
| Mamba-3 | **Yes (complex-valued)** | **Yes** | **Exceeds TC⁰** | **Good** |
| RWKV-6 | Partial | Partial | Exceeds TC⁰ | Moderate |
| **RWKV-7** | **Yes (2 layers for S5)** | **Yes (4 layers)** | **Exceeds TC⁰** | **Good** |
| GLA | Partial | Partial | Exceeds TC⁰ | Moderate |
| **DeltaNet** | **Yes** | **Yes** | **Exceeds TC⁰** | **Strong** |
| **BDH** | **Yes (Hebbian)** | **Unknown** | **Likely exceeds** | **Moderate-Strong** |

### 1.3 Training Speed (Relative to FlashAttention-2)

| Architecture | Training Speed | Notes |
|---|---|---|
| FlashAttention-2 | 1.0x | Baseline |
| **FlashLinearAttention (GLA)** | **1.2-1.5x** | Faster than FA2 even at 1K |
| **Based** | **1.45-1.55x** | 45-55% faster prefill than FA2 |
| Mamba-1 | 0.7-0.9x | Custom CUDA kernel needed |
| Mamba-2 | 1.0-1.2x | Improved over Mamba-1 |
| RWKV-7 | 0.8-1.0x | Stable, spike-free |
| RetNet | 1.3-1.5x | Chunkwise parallel |

### 1.4 Inference Throughput (Relative)

| Architecture | Decode Speed | Memory (per token) |
|---|---|---|
| Transformer | 1.0x | O(N) — grows with context |
| **Based** | **24x** | O(window) — tiny constant |
| **GLA** | **8-12x** | O(1) — fixed matrix |
| **Mamba-2** | **5-8x** | O(1) — fixed vector |
| **RWKV-7** | **5-10x** | O(1) — fixed matrix |
| **Griffin** | **3-5x** | O(window) |
| **BDH** | **5-10x** | O(1) — fixed matrix |

---

## PART 2: WHAT WORKS AND WHAT DOESN'T

### 2.1 What Works (Proven at Scale)

1. **Hybrid architectures dominate:** Every major player (Google, AI21, Microsoft) converged on hybrid designs. Pure sub-quadratic models cannot match Transformer quality at scale.

2. **Data-dependent gating is essential:** Models without gating (naive linear attention) fail. GLA, Mamba, RWKV all use input-dependent gates.

3. **Multi-scale memory is powerful:** RetNet's multi-scale retention, BDH's multi-scale decay, and Mamba-3's MIMO formulation all confirm that multiple memory timescales are critical.

4. **Delta rule updates outperform additive updates:** DeltaNet and RWKV-7 both use delta-rule-like updates, which enable associative recall that additive linear attention cannot achieve.

5. **Sliding window attention is the cheapest quality boost:** Adding even tiny SWA windows (64-128 tokens) to linear models recovers 90%+ of full attention quality (Based, Samba, Griffin).

6. **Complex-valued states enable state tracking:** Mamba-3's key innovation shows that complex-valued SSMs (equivalent to data-dependent rotary embeddings) solve state tracking without increasing state size.

7. **Chunkwise parallel training is the key enabler:** FlashLinearAttention, parallel DeltaNet, and chunkwise RetNet all prove that you can have parallel training AND recurrent inference.

### 2.2 What Doesn't Work

1. **Pure linear attention (no gating):** Consistently underperforms on recall-intensive tasks. The kernel approximation loses too much information.

2. **Pure RNNs at scale:** Hawk beats Mamba at small scale but Google's own Griffin (hybrid) is what shipped as RecurrentGemma. Pure RNNs struggle with long-range recall.

3. **Pure SSMs without attention:** Mamba-1/2 struggle with in-context retrieval. Jamba's ablation showed Mamba-1 + Attention works better than Mamba-2 + Attention in hybrids.

4. **Convolutional views (H3, Hyena):** Proven to struggle on recall tasks theoretically and empirically (Arora et al. ICLR 2024).

5. **Fixed decay rates:** Data-independent decay (RetNet's original gamma) is weaker than data-dependent decay (Mamba, GLA, RWKV-7).

6. **Scaling pure sub-quadratic beyond ~3B:** No pure sub-quadratic model has been demonstrated competitive at 14B+ scale. Only hybrids (Jamba 94B active, Griffin 14B) work at that scale.

---

## PART 3: ARCHITECTURE DEEP DIVES

### 3.1 Mamba Family

**Mamba-1 (arXiv:2312.00752):**
- Selective SSM: input-dependent parameters (A, B, C, Δ)
- Hardware-aware parallel scan for training
- Recurrent inference with O(1) state
- **Key innovation:** Selectivity — the model decides what to remember/forget based on input

**Mamba-2 (arXiv:2405.21060):**
- Shows Transformers are SSMs (structured state space duality)
- Matrix-form SSM: state is [d × d] instead of vector [N]
- Enables chunkwise parallel training
- **Key insight:** SSMs and linear attention are mathematically equivalent under certain conditions

**Mamba-3 (arXiv:2603.15569) — LATEST:**
- Complex-valued SSM — equivalent to data-dependent rotary embeddings
- MIMO (Multi-Input Multi-Output) formulation
- Solves state tracking without increasing state size
- **Key innovation:** Complex numbers enable rotation in state space, solving the state tracking problem that plagued Mamba-1/2

### 3.2 RWKV Family

**RWKV-6 (COLM 2024):**
- Token-shift + WKV (Weighted Key-Value) mechanism
- Matrix-valued states
- Had training spike issues (fixed in RWKV-7)

**RWKV-7 "Goose" (arXiv:2503.14456):**
- Generalized Delta Rule
- 2 layers needed for S5 state tracking, 4 layers for regular languages
- Stable, spike-free training
- **Key innovation:** Delta rule update enables associative recall

**RWKV-8:**
- DeepEmbed + ROSA (Recurrent Online State Automaton)
- Sparse activations
- Partial parallel training

### 3.3 Gated Linear Attention (GLA)

**GLA (arXiv:2312.06635):**
- Gated linear attention with input-dependent gates
- Chunkwise parallel training via FlashLinearAttention
- 1.2-1.5x faster than FlashAttention-2
- **Key innovation:** Gating makes linear attention competitive with softmax attention

**ReGLA (arXiv:2502.01578):**
- Refined GLA with improved gating mechanisms
- Better stability at scale

### 3.4 DeltaNet

**DeltaNet (arXiv:2406.06484):**
- Delta rule (Widrow-Hoff) update: `S_t = S_{t-1} + β_t * (v_t @ k_t^T - S_{t-1} @ k_t @ k_t^T)`
- Parallel chunkwise training
- Strong associative recall
- **Key innovation:** Delta rule enables writing and erasing from state, not just accumulating

### 3.5 RetNet

**RetNet (arXiv:2307.08621):**
- Multi-scale retention with fixed decay rates
- Three modes: parallel (training), recurrent (inference), chunkwise (hybrid)
- **Key innovation:** Unified training/inference formulation

**YOCO / Gated RetNet (arXiv:2405.16712):**
- Added gating to RetNet
- Improved performance over original

### 3.6 Griffin / Hawk (Google DeepMind)

**Griffin (arXiv:2402.19427):**
- RG-LRU (Real-Gated Linear Recurrent Unit) + Local Attention
- Hybrid: 2/3 gated linear recurrence, 1/3 local attention
- Shipped as RecurrentGemma (open model)
- **Key insight:** Pure RNNs need local attention for quality

**Hawk:**
- Pure RG-LRU (no attention)
- Beats Mamba at small scale but loses to Griffin at scale

### 3.7 Based

**Based (arXiv:2402.18668):**
- Taylor series linear attention + Sliding Window Attention
- 1.45-1.55x faster prefill than FA2
- 24x faster decode
- **Key insight:** Taylor approximation of softmax kernel + small SWA window is extremely efficient

### 3.8 Jamba (AI21)

**Jamba (arXiv:2403.19887):**
- Hybrid Transformer-Mamba with MoE
- 1:7 ratio (attention:Mamba)
- Jamba-1.5-Large: 94B active / 398B total
- **Key insight:** At scale, Mamba + Attention + MoE is the winning formula

### 3.9 Samba

**Samba (arXiv:2406.07522):**
- Simple hybrid: Mamba + Sliding Window Attention
- 1:1 ratio (SWA:Mamba)
- ICLR 2025
- **Key insight:** SWA is the cheapest way to boost linear model quality

---

## PART 4: BDH'S POSITION IN THE LANDSCAPE

### 4.1 BDH vs. Closest Competitors

| Dimension | BDH | Closest Competitors |
|---|---|---|
| **Hebbian Learning** | Core mechanism (K^T V outer product with decay) | RWKV-7 (generalized delta rule), DeltaNet (Widrow-Hoff) |
| **Multi-Scale Memory** | 3 decay rates (0.95, 0.99, 0.995) per head | RetNet (fixed multi-scale gamma), Mamba-3 (MIMO) |
| **Positive Orthant (ReLU)** | Enforced on Q, K, V | Linear Transformer (ELU+1), Based (Taylor) |
| **Multiplicative Gating** | x * sigmoid(attn + ffn) | GLA (output gate), RWKV (receptance gate) |
| **Byte-Level Tokenization** | No tokenizer needed | Most use subword tokenizers |
| **Biological Plausibility** | Explicit design goal | HeLa-Mem, ENN (but not for LM) |

### 4.2 BDH's Advantages

1. **BDH = Hebbian + Multi-Scale + Sparse + Multiplicative:** This combination is unique. RWKV-7 has the delta rule but not multi-scale. RetNet has multi-scale but not Hebbian updates. DeltaNet has the delta rule but not biological sparsity.

2. **BDH's multi-scale is more principled than RetNet's:** RetNet uses fixed gamma values per head. BDH uses explicit fast/medium/slow decay rates that map to working memory, paragraph context, and story theme — a cognitively grounded design.

3. **BDH's ReLU sparsity is biologically grounded:** Most linear attention models don't enforce sparsity. The ~5% active neuron claim aligns with biological observations and could enable future sparse computation optimizations.

4. **BDH is the only architecture explicitly designed for interpretability:** The state matrix E is meant to be read and understood. This is a differentiator for scientific/educational applications.

### 4.3 BDH's Gaps vs. State-of-the-Art

1. **No data-dependent gating:** BDH uses fixed decay rates. Mamba, GLA, RWKV-7 all use input-dependent gates. **This is the single biggest architectural gap.**

2. **No convolution:** Mamba's causal conv1d (kernel size 4) is critical for local pattern extraction. BDH lacks this.

3. **No hybrid attention:** Every successful architecture at scale adds some form of attention (SWA, local, or global). BDH is pure linear attention.

4. **Not scaled beyond 70M:** The landscape shows that pure sub-quadratic models struggle past 3B. BDH hasn't been tested at scales where architectural limitations become apparent.

5. **No chunkwise parallel training algorithm:** BDH trains with standard parallel linear attention but lacks the optimized chunkwise kernel that GLA, DeltaNet, and RetNet use.

---

## PART 5: CRITICAL GAPS IN THE FIELD

### Gap 1: The Recall-Efficiency Wall
All pure recurrent/linear models struggle with associative recall. The Based paper formalized this as a state-size vs. recall tradeoff. **No architecture has solved this without adding attention back.**

### Gap 2: No Sub-Quadratic Model at Frontier Scale
Jamba-1.5-Large (94B active) is the largest hybrid. No pure Mamba, RWKV, or GLA model exists at 70B+ parameters. The scaling laws for these architectures are unknown.

### Gap 3: Training Stability at Scale
RWKV-6 had training spike issues (fixed in RWKV-7). Mamba requires careful initialization. GLA needs normalization. No architecture is "plug and play" at billion-parameter scale.

### Gap 4: Theoretical Understanding of Expressiveness
We know Transformers are TC⁰-bounded. We know RWKV-7 and DeltaNet exceed TC⁰. But the exact computational class of Mamba-3, GLA, and BDH remains unknown.

### Gap 5: Hebbian Learning in Deep Networks
Despite 50+ years of Hebbian theory, no Hebbian-based architecture has been trained at scale for language modeling. BDH is the closest attempt but at only 70M params.

### Gap 6: Hardware Co-Design
FlashAttention won because it was designed for GPU hardware. FlashLinearAttention is the first serious attempt for linear attention. But no hardware-optimized kernel exists for Hebbian state updates, multi-scale decay, or delta-rule parallelization beyond what the FLA library provides.

### Gap 7: Long-Context Without Attention
SSMs and RNNs compress everything into fixed state. This fundamentally limits "needle in haystack" retrieval. The only proven solutions add attention (SWA, sparse, or global).

---

## PART 6: FIRST PRINCIPLES OF EFFICIENT SEQUENCE MODELING

### Principle 1: The Memory-Recall Tradeoff is Fundamental
You cannot have O(1) memory and perfect recall simultaneously. The tradeoff curve is:
- **Full attention:** Perfect recall, O(N) memory
- **Linear attention:** Imperfect recall, O(1) memory
- **Hybrid:** Tunable recall, O(window) memory

### Principle 2: Data-Dependence is Non-Negotiable
Any efficient sequence model MUST have input-dependent state transitions. Fixed transitions (LSTM, vanilla RNN) cannot perform content-based reasoning. This is why Mamba's selectivity, GLA's gating, and RWKV's receptance are essential.

### Principle 3: Multi-Scale Memory is Required for Natural Language
Language operates at multiple timescales simultaneously: phonemes (fast), words (medium), sentences (slow), discourse (very slow). Single-scale models cannot capture this hierarchy efficiently.

### Principle 4: Parallel Training + Recurrent Inference is the Gold Standard
The ideal architecture trains in parallel (like Transformer) but infers recurrently (like RNN). This requires a mathematical duality between the two forms. RetNet, GLA, Mamba-2, and DeltaNet all achieve this.

### Principle 5: The State Update Rule Determines Expressiveness
- Additive update (linear attention): Cannot do associative recall
- Delta rule (DeltaNet, RWKV-7): Can do associative recall
- Selective SSM (Mamba): Can do content-based reasoning
- Complex-valued (Mamba-3): Can do state tracking

### Principle 6: Hybridization is Inevitable at Scale
Pure architectures hit a quality ceiling. The winning formula is:
- **Recurrent backbone** (SSM, linear attention, RNN) for efficiency
- **Local attention** (sliding window) for recall
- **Global attention** (sparse, rare) for long-range retrieval

### Principle 7: Hardware Efficiency Determines Practical Success
An architecture is only as good as its kernel. FlashAttention made Transformers practical. FlashLinearAttention is doing the same for linear attention. Custom CUDA/Pallas kernels are required for competitive performance.

---

## PART 7: OPEN RESEARCH QUESTIONS

1. **Can Hebbian Learning Scale to Billion-Parameter Models?** BDH demonstrates the concept at 70M. But can a Hebbian state matrix (d × d) scale to d=4096 or d=8192? The memory cost is d² per layer, which becomes prohibitive. **Potential solution:** Low-rank Hebbian updates, sparse state matrices, or structured state (like Mamba's diagonal + low-rank).

2. **What is the Optimal Hybrid Ratio?** Jamba uses 1:7 (attention:Mamba). Griffin uses 1:2 (local attention:RG-LRU). Samba uses 1:1 (SWA:Mamba). Qwen3.5 uses 1:3 (attention:GDN). **No systematic study exists** on the optimal ratio as a function of model scale, task type, and context length.

3. **Can We Combine Hebbian Learning with Data-Dependent Gating?** BDH has Hebbian learning but fixed decay. Mamba has data-dependent gating but no Hebbian updates. **The combination** (Hebbian delta rule with input-dependent learning rate and decay) could be the missing link. RWKV-7 partially does this.

4. **What is the Expressiveness Class of Multi-Scale Hebbian Networks?** BDH's multi-scale decay creates a hierarchy of memory timescales. What computational problems can this solve that single-scale models cannot? Can it solve S5 state tracking? Can it recognize all regular languages?

5. **Can We Eliminate the KV Cache Entirely?** The ultimate goal: O(1) memory AND perfect recall. This requires a state representation that is both compact and lossless. **Potential approaches:** Learned compression (autoencoder state), neurosymbolic representations (ROSA), or infinite-state mechanisms.

6. **How Do These Architectures Behave Under Test-Time Compute Scaling?** With o1/o3-style reasoning, models generate long chains of thought. Which architecture is most efficient for 100K+ token reasoning traces? Mamba-3's MIMO formulation is explicitly designed for this.

7. **Can Sparse Activations Enable MoE-Free Scaling?** BDH claims ~5% sparse activations. If true, this is equivalent to a 20× MoE without routing overhead. **Can this be exploited** for dynamic computation, where inactive neurons are skipped entirely?

8. **What is the Role of Complex-Valued Representations?** Mamba-3 shows complex-valued states enable state tracking. RetNet uses complex decay (e^{-iθ}). **Are complex numbers fundamental** to efficient sequence modeling, or just one useful parameterization?

9. **Can We Unify All Linear Attention Variants?** The FLA library shows that GLA, RetNet, Mamba-2, DeltaNet, RWKV-6, and HGRN2 are all special cases of gated linear attention. **Is there a single unified formulation** that captures all of them? What is the most general form?

10. **What Architectures Emerge from Scaling Laws?** We have scaling laws for Transformers (Kaplan et al., Chinchilla). **Do the same laws apply** to Mamba, RWKV, GLA, and BDH? Or do they have fundamentally different scaling behavior?

---

## PART 8: PAPERS TO READ (Recommended Reading Order)

### Start with Surveys
1. **Efficient Attention Mechanisms Survey** — https://attention-survey.github.io
2. **Efficient Attention Mechanisms for LLMs: A Survey** — arXiv:2507.19595
3. **Sub-Quadratic Architectures Survey** — ICLR 2026

### Foundation
4. **Mamba: Linear-Time Sequence Modeling** — arXiv:2312.00752
5. **Retentive Network: A Successor to Transformer** — arXiv:2307.08621

### Evolution
6. **Mamba-2: Transformers are SSMs** — arXiv:2405.21060
7. **Mamba-3: Improved Sequence Modeling** — arXiv:2603.15569

### Gating Breakthrough
8. **Gated Linear Attention (GLA)** — arXiv:2312.06635
9. **DeltaNet: Parallelizing Linear Transformers with Delta Rule** — arXiv:2406.06484

### Hybrids
10. **Griffin: Mixing Gated Linear Recurrences with Local Attention** — arXiv:2402.19427
11. **Jamba: Hybrid Transformer-Mamba MoE** — arXiv:2403.19887
12. **Samba: Simple Hybrid SSM + Sliding Window Attention** — arXiv:2406.07522

### RWKV Trajectory
13. **RWKV-6: Eagle and Finch** — arXiv:2409.07146
14. **RWKV-7 "Goose" with Expressive Dynamic State Evolution** — arXiv:2503.14456

### Recall Analysis
15. **Based: Linear + Sliding Window Attention** — arXiv:2402.18668

### Hebbian Connections
16. **HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents** — arXiv:2604.16839
17. **BDH paper (The Dragon Hatchling)** — arXiv:2509.26507

### Long Context
18. **HSA-UltraLong: 16M context via Hierarchical Sparse Attention** — arXiv:2511.23319
19. **HiCI: Hierarchical Construction-Integration for Long-Context** — arXiv:2603.20843
