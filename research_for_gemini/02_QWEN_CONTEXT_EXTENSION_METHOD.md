# The Context Extension Problem: How Qwen Achieved 36K Training → 1M Release

> **Purpose**: This document explains the complete methodology behind Qwen's context window extension breakthrough. It is designed for Gemini deep research to identify transferable techniques for BDH-style architectures.

---

## PART 1: THE FUNDAMENTAL PROBLEM

### Why Context Extension is Hard

RoPE (Rotary Position Embedding) encodes position as rotation angle:
```
θ_i = base^(-2i/d)
```

At large positions, high-frequency dimensions **alias** (rotate multiple times), making positions indistinguishable. The model never saw these large relative distances during training.

**The Core Challenge:** How do you train a model on sequences of length L_train and have it work on sequences of length L_infer >> L_train?

---

## PART 2: QWEN'S COMPLETE METHODOLOGY

### 2.1 Qwen2.5-1M: The 36K → 1M Pipeline

This is the most documented case of extreme context extension.

#### Progressive Pretraining (5 stages)

| Stage | Context | RoPE Base Frequency | Data Mix |
|-------|---------|--------------------|----------|
| 1 | 4,096 | 10,000 | From Qwen2.5 intermediate checkpoint |
| 2 | 32,768 | 1,000,000 | ABF applied |
| 3 | 65,536 | 1,000,000 | 75% at max length, 25% shorter |
| 4 | 131,072 | 5,000,000 | 75% at max length, 25% shorter |
| 5 | **262,144** | **10,000,000** | 75% at max length, 25% shorter |

**Key Insight:** RoPE base frequency scales UP with context length. Higher base → slower rotation → can represent larger positions without aliasing.

**Data Composition Rule:** At each expansion stage, use 75% sequences at current max length, 25% shorter. This prevents catastrophic forgetting of short-context capabilities.

#### RULER Benchmark Progression (Qwen2.5-14B-1M)

| Training Length | RULER Avg | 4K | 8K | 16K | 32K | 64K | 128K |
|----------------|-----------|----|----|-----|-----|-----|------|
| 32K | 82.3 | - | - | - | - | - | - |
| 65K | 86.8 | - | - | - | - | 56.0 | - |
| 131K | 92.5 | 96.5 | 95.9 | 93.0 | 92.6 | 93.0 | 83.8 |
| **262K** | **92.7** | 95.6 | 93.8 | 93.1 | 94.1 | 91.9 | **87.6** |

#### Post-Training
- **SFT Stage 1:** Short instructions only (up to 32K) — preserves short-context performance
- **SFT Stage 2:** Mixed short (32K) + long (256K) instructions
- **RL:** Trained on short text (up to 8K) — alignment generalizes to long context

#### Length Extrapolation to 1M (Training-Free)

Model trained to 256K, extended to 1M at inference via two techniques:

**1. Dual Chunk Attention (DCA):**
- Divides sequence into chunks (chunk size < pretraining length)
- Uses three attention patterns:
  - **Intra-Chunk:** Tokens within same chunk — preserves original relative positions
  - **Inter-Chunk:** Tokens in different, non-adjacent chunks — uses repeated sequences as relative positions
  - **Successive-Chunk:** Tokens in adjacent chunks — preserves short-range relative positions
- **Result:** The relative position matrix is remapped so the model never sees distances larger than training

**2. YaRN Attention Scaling:**
- Scales `q dot k` by `mscale`
- `r = 0.1 * ln(s) + 1`, where `s` = ratio of inference length to training length
- Prevents attention distraction on long sequences

**Remarkable Result:** Even Qwen2.5-7B-Instruct (trained on only 32K) achieves **>80% accuracy on 1M passkey retrieval** with DCA alone.

---

## PART 3: QWEN3 ARCHITECTURE

### 3.1 Architecture Summary

- Standard decoder-only transformer with GQA, SwiGLU, RMSNorm, pre-normalization
- **Removed QKV-bias** from Qwen2, **added QK-Norm** for stable training
- **RoPE with ABF:** base frequency raised from 10,000 to **1,000,000**
- Context lengths: 32K for small models (0.6B, 1.7B), **128K** for larger models (4B-32B, MoE)

### 3.2 Three-Stage Pretraining

| Stage | Tokens | Context Length | Purpose |
|-------|--------|---------------|---------|
| S1: General | ~30T | 4,096 | General knowledge, 119 languages |
| S2: Reasoning | ~5T | 4,096 | STEM, coding, knowledge-intensive |
| S3: Long Context | hundreds of billions | **32,768** | Long-context extension |

**S3 data composition:** 75% text between 16K-32K tokens, 25% text between 4K-16K tokens.

### 3.3 Inference-Time Extension (4× beyond training)

Qwen3 uses **YaRN + DCA together** at inference:
- 32K trained → 128K at inference (4× extrapolation)

---

## PART 4: QWEN3.5 — THE HYBRID ATTENTION REVOLUTION

### 4.1 Architecture Shift

Qwen3.5 abandons pure transformer attention for a **hybrid architecture**:

**3:1 ratio** — every 4 blocks: 3 Gated DeltaNet (linear attention) + 1 Gated Attention (full softmax)

### 4.2 Gated DeltaNet (GDN) Mechanism

```
State update: S_t = exp(g_t) * S_{t-1} + β_t * (v_t @ k_t^T)
Output:       o_t = S_t^T @ q_t
```

Components per layer:
1. **4 input projections:** QKV, Z (gate), A (decay scalar), B (learning rate)
2. **Causal Conv1D** (kernel_size=4): local receptive field before recurrence
3. **Delta rule recurrence:** matrix-valued state [Dk × Dv] per head — acts as learnable associative memory
4. **Per-head RMSNorm** on delta output
5. **Gate multiply:** SiLU(z) × normed_delta_out
6. **Output projection**

### 4.3 Why This Matters for Context

| Feature | Standard Attention | Gated DeltaNet |
|---------|-------------------|----------------|
| Complexity | O(n²) quadratic | Near O(n) linear |
| Memory | Scales with seq length | Near-constant (fixed state) |
| KV Cache | All layers | Only 25% of layers |
| 1M tokens | Prohibitively expensive | Feasible |

### 4.4 Qwen3.5 Model Specs

| Model | Params | Layers | Context | Type |
|-------|--------|--------|---------|------|
| 397B-A17B | 397B (17B active) | 60 | 262K (1M hosted) | MoE |
| 122B-A10B | 122B (10B active) | 48 | 262K | MoE |
| 35B-A3B | 35B (3B active) | 40 | 262K | MoE |
| 27B | 27B | 64 | 262K | Dense |
| 9B | 9B | 32 | 262K (1M ext) | Dense |
| 4B | 4B | 32 | 262K (1M ext) | Dense |
| 2B | 2B | 24 | 262K | Dense |
| 0.8B | 0.8B | 24 | 262K | Dense |

### 4.5 Qwen3.5-0.8B Specifics

- 24 layers: 6 × (3 × GDN → 1 × Gated Attention)
- Hidden dim: 1,024, FFN: 3,584
- GDN: 16 V-heads, 16 QK-heads, head dim 128
- Gated Attention: 8 Q, 2 KV, head dim 256, RoPE dim 64
- ~1.6GB VRAM at BF16, ~0.5GB at 4-bit
- Runs video understanding on phones

---

## PART 5: POSITION ENCODING TECHNIQUES COMPARISON

### 5.1 ABF (Adjusted/Adaptive Base Frequency)

**Training-time technique** (not inference-only like YaRN)

- Increases RoPE base frequency from 10,000 to 1,000,000 (or up to 10,000,000 for Qwen2.5-1M)
- Formula: `θ_i = base^(-2i/d)` — larger base → smaller θ → slower rotation
- **Proactive design** (buy right-sized shoes) vs NTK's **reactive fix** (resize shoes at the door)

### 5.2 YaRN (Yet another RoPE extensioN)

- **Frequency-aware** RoPE scaling: different scaling for different frequency dimensions
- Low-frequency dimensions (large-scale structure) get less scaling
- High-frequency dimensions (fine detail) get more scaling
- Includes **attention scaling** (mscale) to prevent attention distraction on long sequences
- Qwen uses YaRN **together with DCA** — YaRN alone is insufficient
- **Warning:** Static YaRN can degrade short-sequence performance

### 5.3 NTK-Aware Scaling

- Dynamically adjusts RoPE base at inference based on sequence length
- `new_base = 10000 * (k^(d/(d-2)))` where k = target_length / train_length
- Qwen moved away from this in favor of ABF + YaRN + DCA combination

### 5.4 Position Interpolation (PI)

- Linearly downscales position indices
- Qwen explicitly **avoids** this — "We avoid linearly downscaling the position indices"

### 5.5 Dual Chunk Attention (DCA) — The Core Extrapolation Mechanism

**Problem:** RoPE-based LLMs degrade on sequences longer than training because they encounter **unseen large relative positional distances** between Q and K in attention computation.

**DCA Solution:** Divides the sequence into chunks and uses three attention patterns:

1. **Intra-Chunk Attention:** Tokens within the same chunk. Preserves original relative positions (distance is short).

2. **Inter-Chunk Attention:** Tokens in different, non-adjacent chunks. Uses **repeated sequences** as relative positions — the maximum distance never exceeds pretraining length.

3. **Successive-Chunk Attention:** Tokens in adjacent chunks. Preserves short-range relative positions within a local window; falls back to inter-chunk pattern for longer distances.

**Mathematical Effect:** The relative position matrix is **remapped** so that the gray areas (large unseen distances) are replaced with values the model has seen during training. This is **training-free** — no weight updates needed.

---

## PART 6: THE EXTRAPOLATION HIERARCHY

```
No training needed:
  DCA alone: 32K → 1M (32×) — basic retrieval works

Minimal training:
  ABF + short context training: 4K → 128K (32×) with YaRN+DCA

Full progressive training:
  4K → 256K progressive + DCA+YaRN: 256K → 1M (4×) — full quality

Architectural solution:
  Gated DeltaNet hybrid: 262K native, no extrapolation needed
```

---

## PART 7: FIRST PRINCIPLES OF CONTEXT EXTENSION

### Three Axes of Solution

**A. Training-Time (build the capacity)**
1. **ABF:** Increase RoPE base frequency so rotation is slower, pushing aliasing to much larger positions
2. **Progressive curriculum:** Gradually increase context length (4K → 32K → 65K → 131K → 262K), each stage building on the previous
3. **Data composition:** Always mix shorter sequences (25%) to prevent catastrophic forgetting

**B. Inference-Time (extrapolate beyond training)**
1. **DCA:** Remap relative positions so the model never sees distances larger than training
2. **YaRN scaling:** Adjust attention logits to prevent distraction on long sequences
3. **Key insight:** DCA alone can extend 32K → 1M (32×!) with no training, but quality improves dramatically with longer training

**C. Architectural (eliminate the bottleneck)**
1. **Gated DeltaNet:** Replace O(n²) attention with O(n) recurrence — no position encoding needed for linear layers
2. **Hybrid 3:1:** Keep full attention at 25% of layers for precision, use linear attention for 75% for efficiency
3. **Result:** 262K native context even on 0.8B parameter models

---

## PART 8: IMPLICATIONS FOR BDH-STYLE ARCHITECTURES

### What Qwen3.5 Proves

1. **Linear attention is production-ready** at scale (397B parameters)
2. **3:1 hybrid ratio** is the sweet spot — pure linear attention loses precision, pure attention is too expensive
3. **Matrix-valued state** (delta rule) outperforms vector-valued state (Mamba) for associative memory
4. **Local convolution** (kernel_size=4) before recurrence provides short-range precision that pure recurrence lacks

### Critical Implications for BDH

1. **Qwen's Gated DeltaNet uses `S_t = exp(g_t) * S_{t-1} + β_t * (v_t @ k_t^T)`** — this is a **linear attention** formulation with gating
2. **The delta rule is mathematically related to linear attention's kernel feature map approach**
3. **Qwen's success validates that linear/recurrent mechanisms can replace softmax attention for most layers**
4. **The 25% full-attention layers are critical** — BDH should consider whether a hybrid approach is needed or if BDH's architecture already captures what full attention provides
5. **Qwen's GDN has no RoPE on linear layers** — position encoding is only needed on the 25% full-attention layers. If BDH is purely linear, it may need its own position handling strategy

### What BDH Could Learn from Qwen

1. **Add data-dependent gating:** BDH uses fixed decay rates. GDN uses `exp(g_t)` — input-dependent decay. This is the single biggest architectural upgrade BDH could make.

2. **Add local convolution:** GDN has causal Conv1d (kernel_size=4) before recurrence. This provides short-range precision. BDH lacks this entirely.

3. **Consider hybrid attention:** Even Qwen3.5, with its powerful GDN, keeps 25% full-attention layers. BDH is pure linear attention — adding even a small percentage of full attention could dramatically improve quality.

4. **Progressive context training:** BDH could adopt Qwen's progressive curriculum — start at 512 tokens, expand to 1024, 2048, 4096, etc.

5. **Use outer product for Hebbian update:** GDN's state is [Dk × Dv] matrix. BDH should use outer product Q ⊗ V, not element-wise Q ⊙ V.

---

## PART 9: PAPERS TO READ

### Qwen Context Extension
1. **Qwen2.5-1M Technical Report** — https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf
2. **Qwen3 Technical Report** — arXiv:2505.09388
3. **Qwen3.5-Omni Technical Report** — arXiv:2604.15804
4. **Dual Chunk Attention (DCA)** — arXiv:2402.17463
5. **YaRN** — Peng et al., 2023
6. **ABF (Adaptive Base Frequency)** — arXiv:2309.16039 (Xiong et al.)
7. **QwenLong-L1.5** — arXiv:2512.12967
8. **QwenLong-L1** — arXiv (progressive context RL)
9. **Qwen3-Coder-Next** — arXiv:2603.00729
10. **Gated Delta Networks (ICLR 2025)** — https://github.com/nvlabs/gateddeltanet

### Position Encoding
11. **RoPE: Rotary Position Embedding** — Su et al., 2021
12. **LongRoPE2: Near-Lossless Context Window Scaling** — ICML 2025, PMLR 267:54203-54218
13. **AlphaRoPE: Frequency-Adaptive RoPE Scaling** — ICLR 2026
14. **SWAN: NoPE + Sliding Window Attention** — EMNLP 2025
15. **Rope to Nope and Back Again: Hybrid Attention Strategy** — NeurIPS 2025

### Long Context
16. **HSA-UltraLong: 16M context via Hierarchical Sparse Attention** — arXiv:2511.23319
17. **HiCI: Hierarchical Construction-Integration for Long-Context** — arXiv:2603.20843
18. **DySCO: Dynamic Attention-Scaling Decoding** — arXiv:2602.22175
19. **LongCat ZigZag Attention (LoZA)** — arXiv:2512.23966
