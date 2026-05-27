# Agent 30: Edge Multimodal Fusion Scout

## Research Report: Underexplored Architectural and Systems Questions for Edge Multimodal Fusion

**Date:** April 2026  
**Context:** 8GB VRAM RTX 4070, low-cost compute, software-hardware co-design focus  
**Team Interest:** Small models, grounding, multimodal reasoning, audio-text-image fusion  
**Mission:** Identify research gaps that yield scientific insight rather than pure benchmark chasing

---

## 1. Current Literature

### Fusion Taxonomy (Well-Established, Poorly Studied at Edge Scale)
The multimodal fusion literature classifies approaches into three stages:

| Fusion Stage | Mechanism | Strength | Weakness on Edge |
|---|---|---|---|
| Early fusion | Concatenate raw/patch-level visual + text tokens at input | Rich cross-modal reasoning, best spatial understanding | 4,096+ visual tokens per image, high memory cost [web:18] |
| Intermediate fusion | Cross-attention blocks where LLM attends to visual tokens periodically | Balanced efficiency and reasoning quality | Complex architecture, harder to quantize uniformly |
| Late fusion | Process vision and language independently, ensemble near output | 50-70% lower memory, faster inference | Weaker spatial reasoning, misses subtle relationships [web:18] |

Current work treats this as an architectural choice fixed at design time. There is almost no systematic literature on **fusion depth as a runtime-tunable variable** under VRAM pressure.

### Token and Patch Budget Optimization
Recent advances focus on reducing visual token overhead:
- **Token-Efficient VLM (TEVA):** Detects key regions and applies dynamic patch sampling to capture fine-grained details without increasing token count [web:22]
- **SparseVLM:** Training-free token sparsification that rates visual token significance using text-token relevance scores, achieving adaptive layer-wise sparsity [web:27]
- **Delta-LLaVA:** Base-then-specialize alignment for token-efficient vision-language understanding [web:13]
- **Text-as-image compression:** Rendering long text as an image reduces decoder tokens by ~50% while maintaining accuracy on retrieval and summarization tasks [web:8]

These methods focus on *vision-to-text* efficiency. The dual question—how text-token budget and visual-token budget trade off under a **fixed total context budget** on edge hardware—remains unexplored.

### Edge Deployment of VLMs
- **Qwen2-VL-2B-Instruct** achieves sub-second responsiveness on edge but lags in reasoning-intensive domains (spatial relations, social navigation) [web:7]
- **LLaMA-3.2** edge deployment preserves near-cloud accuracy with reduced latency [web:7]
- **Gemma 4** supports text+image on all sizes, with audio additionally on E2B and E4B edge models [web:6]
- **Phi-4-multimodal** (5.6B parameters) processes text, images, and speech/audio simultaneously for on-device performance [web:30]
- Benchmarks on RTX 3070 8GB show Qwen3.5-9B Q4_K_M runs effectively, proving 8GB VRAM can host 7-9B parameter models with quantization [web:36]

### Hardware-Software Co-Design
Hardware-software co-design has emerged as a holistic approach targeting 2x energy efficiency over existing GPU solutions by simultaneously optimizing model structure, compiler stack, and runtime controls [web:21][web:26].

### Audio-Text-Image Fusion
Lightweight tri-modal fusion is severely underexplored. Most multimodal research is vision-language only. Phi-4-multimodal (5.6B) is one of the few production examples, but its fusion mechanisms are not fully ablated in open literature [web:30].

---

## 2. Key Underexplored Questions

### Q1: Fusion Depth as a Scientific Variable
Can we treat early/intermediate/late fusion not as a fixed architecture but as a **runtime-controllable depth knob**? For example, can a model dynamically shift from early fusion (for complex spatial reasoning tasks) to late fusion (for simple classification tasks) based on available VRAM?

### Q2: Token Budget Tradeoffs Under Hard Memory Caps
Given a fixed 8GB VRAM ceiling and a fixed total context length (e.g., 4096 tokens), what is the optimal allocation between text tokens and visual patches? Is it better to have fewer, higher-resolution visual patches or more, lower-resolution ones when the task requires fine-grained grounding?

### Q3: Grounded Reasoning Quality Under Token Pressure
Current grounded VLMs (e.g., G2VLM for spatial 3D, R-VLM for GUI grounding) require dense visual tokens [web:16][web:19]. How does grounding accuracy (IoU, spatial relation correctness) degrade as visual tokens are compressed? Can late fusion ever match early fusion for grounding tasks if visual tokens are aggressively pruned?

### Q4: Lightweight Audio-Text-Image Fusion Architectures
There is no open academic benchmark for sub-7B audio-text-image fusion on edge devices. What is the minimal fusion architecture needed for tasks like "describe this scene using the sound and the image"? Can audio be fused at a later stage than vision because it carries less spatial information?

### Q5: Fusion Choice Interacts with Quantization Strategy
Does early fusion benefit more from INT4 quantization than late fusion, because cross-modal attention matrices are redundant? Or does late fusion allow *heterogeneous quantization* (vision encoder at INT8, language model at INT4)?

### Q6: Modality Dropout for Robustness
If one modality fails on edge (e.g., microphone unavailable, camera blocked), how should the fusion architecture degrade? Late fusion degrades gracefully (drop one model's output); early fusion degrades catastrophically. Can intermediate fusion be designed with built-in modality dropout gates?

---

## 3. Feasibility on 8GB VRAM

### Hardware Reality Check
- **RTX 4070 Laptop:** 8GB GDDR6, ~7.2GB usable under OS overhead [web:43]
- **RTX 3070 8GB benchmarks:** Qwen3.5-9B Q4_K_M runs successfully at 4K-32K context with ~7.2GB VRAM usage, 2.01 tokens/sec generation [web:36][web:43]
- **Qwen2.5-VL-7B:** Officially requires 16GB FP16, but fits 8GB with 4-bit quantization and min/max pixel tuning [web:37][web:41]
- **Qwen2-VL-2B:** Sub-second responsiveness confirmed on edge hardware [web:7]

### What Fits Comfortably
| Model | Size | Quantization | VRAM (est.) | Feasibility |
|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | 2B | Q4_K_M | ~2.5GB | Excellent |
| PaliGemma-3B | 3B | Q4 | ~3GB | Excellent |
| LLaMA-3.2-3B-Vision | 3B | Q4_K_M | ~3.5GB | Excellent |
| Qwen2.5-VL-7B | 7B | Q4_K_M | ~6-7GB | Tight but viable |
| Phi-4-multimodal | 5.6B | Q4_K_M | ~5-6GB | Viable |
| Gemma 4 (E2B/E4B) | 2B/4B | Built-in edge | ~2-4GB | Excellent [web:6] |

### Visual Token Budget Math
Qwen2.5-VL default visual token range: 4-16,384 tokens per image [web:41].  
At 8GB VRAM, running Qwen2.5-VL-7B Q4 with a 4K context budget:
- Text tokens: 512-1024
- Visual tokens: must stay under ~1024 (at 4 tokens per patch, that's 256 patches = ~448x448 effective resolution)
- This creates a **hard tradeoff**: high-res grounding vs. long-form reasoning.

---

## 4. Five Candidate Paper Ideas

### Paper 1: "Fusion Depth as a Runtime Variable: A Systematic Study of Early, Intermediate, and Late Fusion Under 8GB VRAM"
**Type:** Systems + Architecture  
**Idea:** Build a single VLM backbone where fusion depth can be shifted at inference time by selectively routing visual tokens. Ablate across VQA, grounding, and spatial reasoning tasks. Measure accuracy, latency, and VRAM simultaneously. Show a Pareto frontier.  
**Novelty:** No prior work treats fusion depth as a controllable runtime parameter.

### Paper 2: "Patch-Budgeted Grounding: How Visual Token Allocation Affects Spatial Reasoning on Edge VLMs"
**Type:** Empirical Analysis  
**Idea:** Fix total context length at 4096. Sweep visual patch count from 64 to 2048. Measure grounding accuracy (IoU) on ScreenSpot, spatial relation benchmarks, and custom robotics tasks. Characterize the "grounding cliff"—the patch threshold below which spatial reasoning collapses.  
**Novelty:** Token budgets are usually optimized for throughput; this optimizes them for *grounding quality* under memory pressure.

### Paper 3: "TriFusion Nano: Minimal Audio-Text-Image Fusion at Sub-7B Scale for Edge Devices"
**Type:** Architecture  
**Idea:** Design a lightweight tri-modal model where audio is late-fused (compressed to a single query vector), vision is intermediate-fused (cross-attention at layers 4-8), and text is the backbone. Target 3-5B parameters. Evaluate on audio-visual scene description and instruction following.  
**Novelty:** First open, ablated, sub-7B audio-text-image architecture designed specifically for edge.

### Paper 4: "Heterogeneous Quantization for Multimodal Fusion: Different Bit Widths for Different Modalities"
**Type:** Systems / Co-Design  
**Idea:** Investigate whether modality-specific quantization (vision encoder at INT8 to preserve spatial detail, text backbone at INT4 for speed, audio at INT4/FP16 depending on task) outperforms uniform quantization. Measure on RTX 4070 8GB using custom inference engine (e.g., llama.cpp fork or vLLM with custom quant configs).  
**Novelty:** Quantization is treated as uniform in all VLM deployment work. Modalities have different redundancy profiles.

### Paper 5: "Tiered Evaluation of Edge Multimodal Models: A Framework Beyond Accuracy"
**Type:** Evaluation / Benchmark  
**Idea:** Propose a two-tier evaluation framework (Tier-1: scientific insight, Tier-2: deployment utility). Tier-1 measures: fusion-depth sensitivity, modality-alignment gap, grounding consistency under token pressure. Tier-2 measures: latency (mean + tail), energy per inference (Joules), and graceful degradation under modality failure. Release dataset and code.  
**Novelty:** Edge evaluation is fragmented. A unified, tiered framework that balances scientific rigor and deployment relevance does not exist.

---

## 5. Evaluation Plan Suggestions

### Tier-1 Metrics (Scientific Insight)
| Metric | What It Measures | How to Compute |
|---|---|---|
| Fusion-depth sensitivity | How much accuracy changes when fusion is shifted earlier/later | Ablation study: fix model size, vary fusion layer |
| Modality-alignment gap | Distance between vision and text embeddings in shared space | Modality gap analysis (cosine distance between normalized embeddings) [web:14] |
| Grounding consistency | Does grounding accuracy correlate with patch count? | IoU vs. patch count curves on spatial reasoning benchmarks |
| Token efficiency ratio | Performance retained per token spent | (Baseline accuracy - Compressed accuracy) / tokens saved |
| Cross-modal retrieval accuracy | Can the model retrieve correct text given image/audio? | Standard recall@K on multimodal retrieval sets [web:14] |

### Tier-2 Metrics (Deployment Utility)
| Metric | What It Measures | How to Compute |
|---|---|---|
| End-to-end latency | Real-world responsiveness | Wall-clock time, prefill + decode, averaged over 100 runs |
| Tail latency (P95/P99) | Worst-case performance | 95th/99th percentile latency [web:23] |
| Energy per inference | Battery impact | Joules per inference via NVIDIA-smi or power meter [web:28] |
| VRAM footprint | Whether it fits 8GB | Peak allocation during inference |
| Graceful degradation | Robustness to modality failure | Accuracy when one modality is zeroed out vs. full input |
| Throughput | Batch processing on edge | Inferences per second at batch=1 (typical edge scenario) [web:28] |

### Benchmarks to Use
- **Vision-Language:** VQAv2, OKVQA, TextVQA (reasoning); ScreenSpot, RefCOCO (grounding)
- **Spatial Reasoning:** Custom robotics scene + spatial relation dataset; or use G2VLM-style 3D grounding tasks [web:16]
- **Audio-Text-Image:** No standard benchmark exists—must construct from AudioCaps + Flickr30k + audio segments, or use VGGSound + captions
- **Edge-specific:** MLPerf Tiny is too vision-only. Propose custom edge-multimodal benchmark suite.

### Evaluation Protocol
1. **Controlled ablations:** Fix total parameter count, vary only fusion depth/token budget
2. **Pareto analysis:** For every model configuration, plot (accuracy, latency, VRAM) as a 3D Pareto surface
3. **Statistical rigor:** Report mean ± std across multiple seeds and hardware thermal states [web:23]
4. **Real-device testing:** All numbers must come from actual RTX 4070 Laptop or equivalent, not cloud T4/A100 simulations

---

## 6. Tier-1 vs Tier-2 Framing

### Tier-1: Research-Grade Insight
**Goal:** Produce knowledge that generalizes beyond a single model or deployment.  
**Characteristics:**
- Ablation studies where only one variable changes at a time (fusion depth, patch count, quantization scheme)
- Measures that reveal *why* a model behaves a certain way (e.g., attention-head visualization showing which layers rely on vision)
- Findings that inform future architecture design (e.g., "audio should be late-fused because its attention patterns are sparse")
- Reproducible on any 8GB GPU, not just your specific laptop

**Examples from this report:**
- The "grounding cliff" patch threshold (Paper 2)
- Fusion-depth sensitivity curves (Paper 1)
- Heterogeneous quantization theory (Paper 4)

### Tier-2: Deployment-Grade Utility
**Goal:** Produce artifacts that can be directly used by practitioners deploying VLMs on edge hardware.  
**Characteristics:**
- End-to-end benchmarks with real latency and energy numbers
- Docker images / Ollama Modelfiles / vLLM configs that run out-of-the-box
- Decision trees ("if your task is X and your VRAM is Y, use configuration Z")
- Graceful degradation tests under real sensor failure conditions

**Examples from this report:**
- RTX 4070 8GB feasibility matrix (Section 3)
- Tier-2 evaluation metrics: tail latency, energy per inference (Section 5)
- Deployment configs for heterogeneous quantization (Paper 4)

### Why This Framing Matters
Most edge-AI research is either (a) pure systems engineering with no scientific insight, or (b) pure architecture research with no deployment validation. The Tier-1/Tier-2 split ensures your team produces *both*:
- Tier-1 gives you publishable, generalizable knowledge
- Tier-2 gives you hackathon-ready, industry-credible artifacts
- Together, they differentiate you from benchmark chasers

---

## 7. Final Recommendation

### Immediate Actions (Next 4 Weeks)
1. **Establish the baseline:** Pick 3 small VLMs that fit 8GB VRAM: **Qwen2-VL-2B**, **LLaMA-3.2-3B-Vision**, and **PaliGemma-3B**. Benchmark them on VQAv2 + a simple grounding task (ScreenSpot). Record VRAM, latency (mean + P95), and accuracy. This is your Tier-2 reality anchor.

2. **Run the first scientific experiment:** Fix Qwen2-VL-2B. Vary `min_pixels` and `max_pixels` to create 5 different visual token budgets (64, 256, 512, 1024, 2048 patches). Measure accuracy on a vision+language task and a grounding task. Plot accuracy vs. patch count. This gives you a Tier-1 grounding cliff curve.

3. **Survey audio-text-image landscape:** List every open-weight model under 7B that supports audio+text+image. Likely candidates: Phi-4-multimodal, Gemma 4 E2B/E4B (if audio weights released), Qwen2-Audio variants. Document what fusion mechanisms they use. This identifies the gap for Paper 3.

### Medium-Term Strategy (Next 3 Months)
- **Paper 1 or 2** is the lowest-risk first publication: both use existing models, no training from scratch, pure ablation + analysis.
- **Paper 3** is the highest-impact but requires model implementation effort. Pursue only if team has bandwidth to train/finetune a 3-5B model.
- **Paper 4** is systems-heavy but leverages your hardware-software co-design interest. Good as a second paper if you build a custom inference pipeline.
- **Paper 5** should be developed in parallel as a living framework that every other paper feeds into.

### Resource Allocation
| Resource | Priority | Rationale |
|---|---|---|
| RTX 4070 Laptop (8GB) | Critical | All Tier-2 numbers must be real |
| RunPod / Colab credits | Medium | For training/finetuning (8GB insufficient for training) |
| Secondary Dell Latitude | Low-Medium | Distributed inference experiments, not training |
| Budget ceiling | ~15K INR for hardware | Stick to quantization and inference optimization, not new GPUs |

### The Core Thesis to Pursue
> **On 8GB VRAM, fusion is not an architectural choice—it is a resource allocation problem.** The scientific insight lies in treating visual patches, text tokens, and audio frames as competing resources under a hard memory cap, then deriving the optimal allocation policy for each task type (grounding, reasoning, retrieval).

This framing connects your team's interests (small models, grounding, co-design) with a genuine research gap (no one systematically studies fusion as a resource allocation problem). It avoids giant cloud models, avoids pure benchmark chasing, and produces both Tier-1 insight and Tier-2 artifacts.

---

*Report compiled by Agent 30 for the AlienX/BDH edge multimodal team.*
