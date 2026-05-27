# Subquadratic (SubQ) — 12M Context Window: Architecture Analysis & BDH Integration Strategy

> **Purpose**: Deep analysis of Subquadratic's SubQ model and SSA architecture. Explains how it works, what's verified vs. claimed, and how BDH can leverage or learn from it. Designed for Gemini deep research.

---

## PART 1: WHAT IS SUBQUADRATIC?

### Company Overview

| Detail | Value |
|--------|-------|
| **Company** | Subquadratic (formerly "Aldea") |
| **Location** | Miami, Florida |
| **Founded** | Emerged from stealth May 5, 2026 |
| **Team** | ~13 people, 11 PhD researchers |
| **Team Background** | Meta, Google, Oxford, Cambridge, ByteDance, Adobe, BYU |
| **CEO** | Justin Dangel |
| **CTO** | Alexander (Alex) Whedon (former Head of Generative AI at Meta) |
| **Seed Funding** | $29M at ~$500M valuation |
| **Investors** | Justin Mateen (Tinder co-founder), Javier Villamizar (ex-SoftBank Vision Fund), early investors in Anthropic, OpenAI, Stripe, Brex |

### The Model: SubQ

| Detail | Value |
|--------|-------|
| **Model name** | SubQ (SubQ 1M-Preview for production) |
| **Architecture** | SSA — Subquadratic Sparse Attention |
| **Production context** | 1M tokens (API) |
| **Research context** | 12M tokens (gated to partners) |
| **Target** | 50M tokens by Q4 2026, 100M tokens mentioned |
| **Weights** | Closed (not open-sourced) |
| **Base model** | Built on open-source base models (likely DeepSeek or Kimi family) — CTO confirmed they don't train from scratch |
| **Products** | SubQ API, SubQ Code (CLI coding agent), SubQ Search (deep research tool) |

---

## PART 2: SSA — SUBQUADRATIC SPARSE ATTENTION

### 2.1 The Core Problem

Standard transformer attention: every token compares to every other token → O(n²) complexity.
- 1,000 tokens → 1,000,000 comparisons
- 12,000,000 tokens → 144,000,000,000,000 comparisons (144 trillion)

This quadratic scaling makes million-token contexts economically impossible on dense transformers.

### 2.2 The SSA Solution

**Core idea:** For each query token, the model selects a small subset of positions to attend to based on content, then computes exact attention only over those selected positions.

```
For each query token q_i:
  1. Compute similarity scores: sim(q_i, k_j) for all j
  2. Select top-k most relevant positions (content-dependent)
  3. Compute exact softmax attention only over selected positions
  4. Output = weighted sum of selected values
```

**Key insight:** The selection mechanism itself is NOT quadratic. This is what differentiates SSA from prior sparse attention approaches.

### 2.3 How SSA Avoids the "Indexer Trap"

Previous sparse attention approaches (like DeepSeek Sparse Attention / DSA) had a critical flaw: the indexer that selects which keys to attend to must score each query against each key — which is itself quadratic.

**SSA's claimed innovation:** The selection depends on content in a way that doesn't require computing all pairwise scores. The mechanism uses:

1. **Content-dependent routing:** The model decides where to look based on meaning, not fixed positional patterns
2. **Hierarchical clustering:** Clusters similar tokens and computes attention at the cluster level first, then zooms into relevant tokens within clusters
3. **Local + global attention patterns:** Certain tokens always get local attention (nearby context), while global tokens can attend across the entire sequence
4. **Linear selection:** The selection step scales linearly, not quadratically

**Mathematical formulation (reconstructed from public descriptions):**
```
Complexity: O(n · k) instead of O(n²)
where:
  n = sequence length
  k = number of tokens selected per query (k << n)
```

At 12M tokens, if k is kept small (e.g., k ≈ n/1000), compute drops by ~1000×.

### 2.4 What Makes SSA Different from Prior Approaches

| Approach | Selection Method | Selection Cost | Content-Dependent? |
|----------|-----------------|----------------|-------------------|
| **Fixed sparse** (Longformer, BigBird) | Pre-defined patterns | O(n) | No |
| **Sliding window** | Local window only | O(n·w) | No |
| **State space** (Mamba, RWKV) | Compressed state | O(n) | Partially |
| **Linear attention** (GLA, DeltaNet) | Kernel approximation | O(n) | Partially |
| **DeepSeek NSA/DSA** | Indexer scores all pairs | O(n²) for selection | Yes |
| **SSA (SubQ)** | Content-dependent routing | **O(n) claimed** | **Yes** |

**The claimed breakthrough:** SSA achieves content-dependent selection (like DSA) without the quadratic selection cost (unlike DSA).

---

## PART 3: CLAIMED PERFORMANCE

### 3.1 Benchmarks (Self-Reported, Not Independently Verified)

| Benchmark | SubQ | Claude Opus 4.6 | GPT-5.4/5.5 | Gemini 3.1 Pro |
|-----------|------|----------------|-------------|----------------|
| **RULER 128K** | 95-97% | 94.8% | — | — |
| **MRCR v2** | 83.0 | 78 / 32.2 | 74.0 / 39 | 23 |
| **SWE-Bench Verified** | 81.8-82.4% | 80.8-81.4% | — | 80.6% |
| **Needle-in-Haystack 12M** | 92.1% | N/A (doesn't operate at this length) | N/A | N/A |

### 3.2 Speed and Cost

| Context Length | Speedup vs. FlashAttention-2 | FLOP Reduction |
|---------------|------------------------------|----------------|
| 128K tokens | 7.2× | 8× |
| 256K tokens | 13.2× | — |
| 512K tokens | 23.0× | — |
| 1M tokens | 52.2× | 62.5× |
| 12M tokens | — | ~1,000× |

**Cost comparison (RULER 128K):**
- SubQ: ~$8
- Claude Opus 4.6: ~$2,600
- **SubQ is ~325× cheaper** for the same task

**Production pricing:** ~1/5 the cost of frontier models (Claude Opus, GPT-5.5)

### 3.3 Infrastructure

- Trained stably at 1M+ tokens
- Linear memory scaling across training pipeline
- Distributed sequence parallelism to shard sequences across devices
- Runs on neocloud GPUs (hyperscalers "too expensive")

---

## PART 4: TRAINING PIPELINE

### 4.1 Three-Stage Training

1. **Pre-training:** Base model trained on massive, diverse datasets with long-context representations. Built on open-source base models (likely DeepSeek or Kimi family).

2. **Supervised fine-tuning:** Structured tasks including reasoning and code generation.

3. **Reinforcement learning:** Specifically targeting long-context retrieval failures. Teaches the model to aggressively use distant context rather than defaulting to nearby information — a subtle failure mode that degrades performance in existing systems.

### 4.2 Key Training Insight

The RL stage is critical: most long-context models fail not because they can't process long sequences, but because they **default to attending to nearby tokens** even when relevant information is far away. SubQ's RL specifically penalizes this behavior and rewards retrieval from arbitrary positions.

---

## PART 5: SKEPTICISM AND CAVEATS

### 5.1 What's Verified vs. What's Claimed

| Aspect | Status |
|--------|--------|
| Architecture exists | **Verified** — API is live, products ship |
| Team credentials | **Verified** — PhDs from Meta, Google, Oxford, etc. |
| 12M context window | **Claimed** — gated to partners, no independent verification |
| 1,000× compute reduction | **Claimed** — no technical paper yet |
| Benchmark scores | **Self-reported** — single runs due to cost, company admits SWE-Bench margin is "harness as much as model" |
| Technical paper | **Not yet published** — "forthcoming" |
| Open weights | **No** — closed weights, API only |
| Base model identity | **Not disclosed** — CTO confirmed built on open-source base |

### 5.2 Historical Precedent for Skepticism

**Magic.dev (August 2024):**
- Announced 100M-token context model
- Claimed 1,000× efficiency advantage
- Raised ~$500M
- As of early 2026: no public evidence of real-world usage

**The pattern:** Bold context window claims → big funding → silence. Subquadratic is aware of this history and explicitly addresses it in their technical blog.

### 5.3 Technical Concerns Raised by Researchers

1. **No published paper:** The architectural details are high-level. No one outside the company can reproduce the results.
2. **Single-run benchmarks:** Each benchmark was run once due to inference costs.
3. **Model is smaller than frontier labs:** SubQ is "much smaller" than Opus/GPT-5, which could explain lower cost but also limits general reasoning.
4. **Built on open-source base:** The core innovation is the attention mechanism, not the base model. This is practical but means the results depend heavily on the quality of the base model.
5. **Past sub-quadratic architectures** (Mamba, RWKV) underperformed standard attention at frontier scale or ended up as hybrids.

### 5.4 What Experts Think

> "I think the architecture is probably real. The SSA mechanism makes conceptual sense. The team has the background to build something like this. What I genuinely don't know is whether it holds up at scale, whether the reasoning ability matches Opus or GPT-5 on tasks that go beyond long-context retrieval, and whether the benchmark numbers survive independent testing."
> — AI researcher analysis (MayhemCode, May 2026)

---

## PART 6: HOW SSA COMPARES TO BDH

### 6.1 Fundamental Similarities

| Dimension | BDH | SSA (SubQ) |
|-----------|-----|------------|
| **Goal** | O(N) complexity | O(n·k) complexity (k << n) |
| **Approach** | Linear attention via Q(K^TV) | Sparse attention via content-dependent selection |
| **Memory** | Fixed state matrix | Linear KV cache |
| **Position awareness** | RoPE (should be enabled) | Implicit via content-dependent selection |
| **Biological inspiration** | Hebbian learning | No (purely engineering) |
| **Interpretability** | State matrix is readable | Attention patterns are readable |

### 6.2 Fundamental Differences

| Dimension | BDH | SSA (SubQ) |
|-----------|-----|------------|
| **Mechanism** | Accumulates state (Hebbian) | Selects tokens (sparse attention) |
| **Recall** | Degrades with distance (decay) | Preserved at arbitrary distance (selection) |
| **Training** | Harder than Transformer | Built on existing base models |
| **Scale demonstrated** | ~70M parameters | Unknown (smaller than frontier) |
| **Open source** | Yes | No |
| **Proven at scale** | No | Partially (API live, but benchmarks self-reported) |

### 6.3 What BDH Can Learn from SSA

1. **Content-dependent selection is the key:** BDH's linear attention treats all tokens proportionally. SSA's content-dependent selection is what enables reliable long-context retrieval. BDH needs a mechanism to selectively attend, not just accumulate.

2. **RL for long-context training:** SubQ's RL stage specifically targets the "nearby token bias" failure mode. BDH could benefit from similar RL training to force the model to use its full memory range.

3. **Hierarchical processing:** SSA clusters tokens and processes at cluster level first. BDH's multi-scale memory (fast/medium/slow) is conceptually similar but operates at the state level, not the token level.

4. **Built on existing base models:** SubQ didn't train from scratch — they took an open-source model and replaced the attention mechanism. BDH could follow the same approach: take a small open-source model and replace its attention with BDH's linear attention + Hebbian state.

---

## PART 7: HOW BDH CAN LEVERAGE SSA

### 7.1 Strategy 1: Hybrid BDH-SSA Architecture

**Idea:** Combine BDH's Hebbian state with SSA's content-dependent selection.

```
For each token:
  1. Use SSA to select top-k relevant positions
  2. Compute BDH linear attention only over selected positions
  3. Update Hebbian state with selected positions
  4. Output = multiplicative gating of attention + FFN
```

**Why this works:**
- SSA handles long-range recall (selects the right tokens)
- BDH handles local processing (Hebbian state for working memory)
- Combined complexity: O(n·k) for selection + O(k) for linear attention = O(n·k)
- The Hebbian state provides persistent memory beyond the selected positions

**Expected improvement:**
- Reliable needle-in-haystack retrieval at 1M+ tokens
- BDH's interpretability preserved (state matrix still readable)
- Training easier (SSA selection provides clear gradients)

### 7.2 Strategy 2: BDH as the Selection Mechanism

**Idea:** Use BDH's Hebbian state to determine which tokens to attend to.

```
For each token:
  1. Query the Hebbian state: "what do I need to remember?"
  2. Use state similarity to select top-k positions
  3. Compute exact attention over selected positions
  4. Update Hebbian state
```

**Why this is novel:**
- BDH's state matrix encodes what the model has learned about token relationships
- Instead of computing similarity scores from scratch (like SSA), use the pre-computed state
- The state is already O(1) to query — no quadratic selection cost

**This is the killer idea:** BDH's Hebbian state IS the selection index. No need to compute all pairwise scores — the state already encodes the relationships.

### 7.3 Strategy 3: Progressive Context Training (Qwen + SubQ combined)

**Idea:** Combine Qwen's progressive context extension with SubQ's RL for long-context retrieval.

```
Stage 1: Train BDH at 512 tokens (standard)
Stage 2: Expand to 2048 tokens (multi-scale helps)
Stage 3: Expand to 8192 tokens (add SSA-like selection)
Stage 4: Expand to 32768 tokens (RL for long-context retrieval)
Stage 5: Expand to 131072+ tokens (full SSA + BDH hybrid)
```

**Why this works:**
- Qwen proved progressive context extension works
- SubQ proved RL fixes the "nearby token bias"
- BDH's multi-scale memory naturally supports multiple timescales

### 7.4 Strategy 4: Use SubQ as a Teacher for Distillation

**Idea:** If SubQ API becomes accessible, use it as a teacher to distill long-context capabilities into BDH.

```
Teacher: SubQ (12M context, SSA architecture)
Student: BDH (fixed memory, linear attention)
Distillation:
  - Logits distillation on long-context tasks
  - Hidden state distillation for reasoning patterns
  - Needle-in-haystack retrieval as a training objective
```

**Why this works:**
- SubQ has proven long-context capabilities (if claims hold)
- BDH can learn the "reasoning patterns" from SubQ's outputs
- Knowledge distillation is 3-5B tokens vs. trillions for pretraining

### 7.5 Strategy 5: BDH for the "Last Mile" of SSA

**Idea:** Use SSA for long-range retrieval, BDH for local reasoning.

```
SSA handles: Finding relevant information across 12M tokens
BDH handles: Reasoning over the retrieved information
```

**Why this works:**
- SSA is great at retrieval but may not be great at reasoning (smaller model)
- BDH's Hebbian state is great at working memory and reasoning
- Combined: SSA retrieves, BDH reasons

---

## PART 8: ACTIONABLE NEXT STEPS FOR BDH

### Immediate (Week 1-2)
1. **Study SSA's selection mechanism:** When the technical paper is released, analyze the exact algorithm for content-dependent selection.
2. **Implement a toy SSA:** Build a simplified version of content-dependent selection in PyTorch to understand the mechanics.
3. **Benchmark BDH on needle-in-haystack:** Test BDH's current long-context retrieval capability to establish a baseline.

### Short-term (Week 3-6)
4. **Add content-dependent gating to BDH:** Implement input-dependent decay rates (inspired by SSA's selection).
5. **Implement hierarchical processing:** Cluster tokens and process at multiple levels (inspired by SSA's clustering).
6. **Add RL for long-context retrieval:** Train BDH with a reward signal for retrieving information from arbitrary positions.

### Medium-term (Week 7-12)
7. **Build BDH-SSA hybrid:** Combine BDH's Hebbian state with SSA's content-dependent selection.
8. **Distill from SubQ (if API accessible):** Use SubQ as a teacher for long-context capabilities.
9. **Scale to 500M parameters:** Test if the hybrid architecture scales.

### Long-term (Month 4-6)
10. **Target 1M+ context:** Progressive training to million-token contexts.
11. **Publish technical paper:** Open-source the architecture and benchmarks.
12. **Compete with SubQ on benchmarks:** Independent verification of capabilities.

---

## PART 9: PAPERS AND RESOURCES TO FOLLOW

### Subquadratic Resources
1. **SubQ Launch Announcement** — https://subq.ai/introducing-subq
2. **SSA Technical Blog** — https://subq.ai/how-ssa-makes-long-context-practical
3. **SubQ Documentation** — https://subq.ai/docs
4. **Technical Report** — Forthcoming (watch for release)

### Related Sparse Attention Papers
5. **DeepSeek Native Sparse Attention (NSA)** — ACL 2025 Best Paper
6. **DeepSeek Sparse Attention (DSA)** — shipping in DeepSeek V3.2-exp
7. **Longformer** — arXiv:2004.05150
8. **BigBird** — arXiv:2007.14062
9. **Routing Transformer** — arXiv:2003.05997
10. **Reformer** — arXiv:2001.04451

### Context Extension Papers
11. **Qwen2.5-1M Technical Report** — https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf
12. **Dual Chunk Attention (DCA)** — arXiv:2402.17463
13. **YaRN** — Peng et al., 2023
14. **LongRoPE2** — ICML 2025

### Linear Attention / SSM
15. **Mamba-3** — arXiv:2603.15569
16. **RWKV-7** — arXiv:2503.14456
17. **Gated DeltaNet** — ICLR 2025
18. **FlashLinearAttention** — GitHub: fla-org/flash-linear-attention

---

## PART 10: CRITICAL ASSESSMENT

### What's Likely True
- The SSA architecture is real (API is live, products ship)
- Content-dependent selection is conceptually sound
- The team has the background to build something like this
- Long-context capability at 1M tokens is plausible
- Cost reduction at long contexts is real (linear vs. quadratic scaling)

### What Needs Verification
- 12M context window quality (gated, no independent testing)
- 1,000× compute reduction (no technical paper)
- Benchmark scores (self-reported, single runs)
- Reasoning capability beyond retrieval (SWE-Bench margin is "harness as much as model")
- Whether it holds up at scale (model is smaller than frontier labs)

### What BDH Should Do
- **Don't try to replicate SSA** — it's closed-source and the technical details aren't public
- **Do learn from SSA's principles** — content-dependent selection, RL for long-context, hierarchical processing
- **Do combine BDH's unique strengths** — Hebbian state as selection index, multi-scale memory, interpretability
- **Do prepare for the technical paper** — when it's released, analyze it immediately and extract applicable ideas

### The Bottom Line

Subquadratic's SSA is the most credible attempt at pure sub-quadratic attention at frontier scale. If the claims hold, it changes the economics of long-context AI overnight. For BDH, SSA represents both a **competitor** (same goal: efficient long-context) and an **inspiration** (content-dependent selection is what BDH lacks).

The highest-leverage move for BDH: **use the Hebbian state as the selection index**. This is something SSA can't do — it computes selection from scratch every time. BDH's state already encodes token relationships. If BDH can query its state to select relevant tokens, it gets SSA's benefits without the selection cost.
