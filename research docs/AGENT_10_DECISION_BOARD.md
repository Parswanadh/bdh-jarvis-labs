# AGENT_10_DECISION_BOARD.md
## Synthesis Director — Research Direction Decision Board
**Prepared for:** Student Team | Amrita Vishwa Vidyapeetham, Bangalore  
**Hardware:** RTX 4070 8GB VRAM  
**Timeline:** 8 weeks (1–2 months to technical results)  
**Date:** April 25, 2026

---

## SECTION 1: TOP 10 CANDIDATE RESEARCH DIRECTIONS

| # | Direction | Core Idea | Fit to Constraints |
|---|-----------|-----------|-------------------|
| 1 | **CoT Distillation for Small Models** | Transfer chain-of-thought reasoning from large frontier models (DeepSeek-R1, GPT-4o) to 1–3B student models via supervised fine-tuning with mixed-length rationale data | HIGH — QLoRA on 8GB, annotatable, clear benchmarks |
| 2 | **Speculative Decoding Draft Model Design** | Train or tune a custom draft model (≤500M params) to maximize token acceptance rate with a 7B target model, targeting 2–3× speedup on local hardware | HIGH — fits 8GB, measurable wall-clock speedup metric |
| 3 | **Hybrid SSM-Attention (Mamba-style) Fine-tuning** | Fine-tune or ablate Mamba-3/Zamba-style hybrid architectures on domain-specific tasks; study quality vs. latency tradeoffs at ≤7B scale | MEDIUM — architecture novelty is high, but training SSM hybrids is more complex |
| 4 | **Multimodal Grounding via TWIST/LoRA** | Fine-tune a small vision-language model (e.g., LLaVA-1.6 7B, Qwen2-VL 2B) for visual grounding tasks using forget-free LoRA methods | HIGH — Qwen2-VL 2B fits 8GB, annotatable bbox/grounding data exists |
| 5 | **Memory-Augmented Inference (NMRet-style)** | Attach an external neural memory/retrieval module to a frozen small LLM, enabling multi-hop QA without expanding context window | MEDIUM — no training needed for base model, but system engineering heavy |
| 6 | **HW-SW Co-Design: LLM Accelerator Verilog Sketch** | Prototype a compute-in-memory or systolic-array-like accelerator for linear attention / SSM layers in Verilog; benchmark energy efficiency against software baseline | MEDIUM — high novelty, fits academic background, but needs EDA toolchain |
| 7 | **Mix Distillation (Complexity-Adaptive CoT)** | Train student models using mixed-difficulty reasoning chains (short + long CoT) per "Small Models Struggle to Learn from Strong Reasoners" findings | HIGH — directly actionable on 8GB, improves on basic distillation |
| 8 | **Speech + Text Multimodal LLM (Voice Grounding)** | Fine-tune a small speech-LLM (Whisper encoder + 1–3B decoder) to jointly handle ASR and instruction following; test cross-lingual generalization | LOW–MEDIUM — interesting but requires audio datasets, less differentiating |
| 9 | **Retrieval-Augmented Small Model Reasoning** | Attach BM25/DPR retriever to a 1–3B model and train the reader head with GRPO or DPO to reason over retrieved passages | MEDIUM — well-trodden path in 2024; novelty is lower unless domain is niche |
| 10 | **Process Reward Model (PRM) for Small Models** | Train a small verifier model that scores intermediate reasoning steps, used as training signal for the policy model (self-play loop) | MEDIUM-HIGH — emerging technique but requires step-level annotations |

---

## SECTION 2: ELIMINATION LOGIC

### Eliminated / Deprioritized Directions

**Direction 3 — Hybrid SSM-Attention Fine-tuning**  
ELIMINATED. Mamba-3 and Zamba require custom CUDA kernels and modified training loops not natively supported in HuggingFace transformers. Debugging time on a single 8GB GPU within 8 weeks is prohibitive. Reserve for a follow-up hardware semester.

**Direction 5 — Memory-Augmented Inference (NMRet-style)**  
DEPRIORITIZED. Heavy system engineering (vector store integration, retrieval loop, multi-step reasoning coordination) produces infrastructure rather than ML research contribution. No clear paper-worthy novelty within the timeline.

**Direction 8 — Speech + Text Multimodal LLM**  
ELIMINATED. Audio dataset curation, codec tokenization, and ASR alignment add 2+ weeks of preprocessing overhead. Benchmarks are less standardized. The team's annotation support is better spent on text/vision tasks.

**Direction 9 — RAG Reasoning**  
ELIMINATED. Saturated research area in 2024–2025. Unless tied to a highly specific domain (e.g., Malayalam/Kannada legal text), it will not clear novelty bar for publication or competition entry. Standard baselines are already strong.

**Direction 6 — HW-SW Co-Design Verilog Sketch**  
CONDITIONAL KEEP (solo side project). The Michigan co-design paper on compute-in-memory SSMs is novel (April 2026). However, functional RTL simulation + software benchmark comparison is a full semester project. If team has Verilog experience, recommended as a parallel 2-person subtrack, not the primary research thrust.

**Direction 10 — PRM for Small Models**  
DEPRIORITIZED. Step-level annotations are expensive and time-consuming to produce. Without existing annotated math competition datasets (MATH-500 process labels), this is blocked by data. Viable if annotation team can produce 3,000+ step-labeled examples.

---

## SECTION 3: TOP 3 FINAL CHOICES

---

### CHOICE A: Mix Distillation + CoT Complexity-Adaptive Reasoning in SLMs
**Research Directions Merged:** #1 + #7  
**Core Hypothesis:** Small language models (1–3B) fail to learn from long frontier-model CoT traces because the reasoning complexity exceeds their capacity. A curriculum that blends short, medium, and long CoT chains — sourced from both large (R1/GPT-4o) and small (Phi-3.5, Qwen-2.5-1.5B) teacher models — will outperform naive distillation on reasoning benchmarks (GSM8K, ARC-Challenge, HellaSwag).

**Why it wins:**
- Directly actionable with QLoRA on 8GB (1.5B–3B student)
- Annotation team can label difficulty tiers on existing datasets
- Clear improvement over D-CoT / SCoTD baselines (Feb 2026)
- Benchmarking is standardized (lm-evaluation-harness)
- Strong submission target: EMNLP 2026 / ACL 2026 workshop

**VERDICT: YES ✅**

---

### CHOICE B: Small VLM Visual Grounding via Forget-Free LoRA (TWIST-style)
**Research Directions Merged:** #4  
**Core Hypothesis:** Fine-tuning Qwen2-VL-2B or LLaVA-1.6-Mistral-7B with a dual-expert LoRA (one frozen image-understanding adapter, one learnable grounding adapter) achieves competitive grounded captioning and zero-shot localization without catastrophic forgetting, on a single 8GB GPU.

**Why it wins:**
- Qwen2-VL-2B fits within 8GB with QLoRA
- TWIST (ICCV 2025) provides direct architectural blueprint
- Annotation team can create grounding bounding-box labels from COCO/RefCOCO
- Multimodal grounding is an underexplored area at ≤3B scale
- Results are visually demonstrable (bounding boxes, captions)

**VERDICT: YES ✅**

---

### CHOICE C: Custom Speculative Decoding Draft Model for Domain-Specific Speedup
**Research Directions Merged:** #2  
**Core Hypothesis:** A domain-specialized draft model (trained on technical/academic corpora, ≤300M params) achieves higher token acceptance rate than a generic draft model when paired with a 7B target, yielding measurable inference speedup (>2.5×) on 8GB hardware in domain-specific settings.

**Why it wins:**
- No training of the target model needed (frozen 7B in 4-bit)
- Draft model training fits easily on 8GB (300M params)
- Wall-clock speedup is an objective, reproducible metric
- Directly applicable to AlienX / local LLM deployment goals
- Connects to hardware-software co-design interests

**VERDICT: YES (conditional) ✅ — strongest if team has inference engineering interest**

---

## SECTION 4: PRIMARY PAPER

**"Skip-Thinking: Chunk-wise Chain-of-Thought Distillation Enables Small Language Models to Learn Complex Reasoning"**  
- Source: EMNLP 2025 Main Conference (arXiv 2505.18642)
- Authors: Multiple (ACL Anthology 2025.emnlp-main.610)
- URL: https://arxiv.org/abs/2505.18642
- Why Primary: Directly proposes chunk-wise CoT distillation where long reasoning chains are segmented and matched to student capacity — maps exactly to Choice A. Provides training code, evaluation on GSM8K/MATH, and ablation studies. Reproducible on ≤3B models.

**Companion paper:** "Small Models Struggle to Learn from Strong Reasoners" (arXiv 2502.12143, Feb 2025) — introduces Mix Distillation, providing the curriculum mechanism.

---

## SECTION 5: FALLBACK PAPER

**"TWIST & SCOUT: Grounding Multimodal LLM-Experts by Forget-Free Tuning"**  
- Source: ICCV 2025
- URL: https://openaccess.thecvf.com/content/ICCV2025/papers/Bhowmik_TWIST__SCOUT_Grounding_Multimodal_LLM-Experts_by_Forget-Free_Tuning
- Why Fallback: If CoT distillation stalls due to data quality or benchmark saturation, pivot to Choice B. TWIST provides a complete training recipe (twin-expert LoRA, stepwise tuning), evaluated on grounded captioning, RefCOCO, and zero-shot localization. All components are reproducible with Qwen2-VL-2B + HuggingFace PEFT.

---

## SECTION 6: 8-WEEK EXECUTION SNAPSHOT

### PRIMARY TRACK — Choice A: Mix Distillation CoT SLM

| Week | Milestone | Output |
|------|-----------|--------|
| 1 | Reproduce SCoTD / Skip-Thinking baseline on Phi-3.5-mini (3.8B, 4-bit); run GSM8K eval | Baseline accuracy numbers |
| 2 | Generate CoT traces from DeepSeek-R1-Distill-1.5B and GPT-4o-mini via API (500 examples each difficulty tier) | Tiered annotation dataset v1 |
| 3 | Annotation team: label 1000 examples across 3 difficulty levels (short / medium / long CoT); quality review | Gold-labeled mix distillation dataset |
| 4 | Train Mix Distillation student (Qwen-2.5-1.5B, QLoRA r=16, α=32) on tiered data; overnight run | Student model checkpoint |
| 5 | Evaluate on GSM8K, ARC-Challenge, HellaSwag; compare vs. naive distillation and base model | Delta table (primary result) |
| 6 | Ablation: vary mix ratio (100% long / 50-50 / 100% short CoT); overnight runs | Ablation curves |
| 7 | Write-up: method, results, related work; create demo notebook | Draft paper / report |
| 8 | Polish, submit to workshop / upload to HuggingFace; push code to GitHub | Public artifact |

### FALLBACK TRACK — Choice B: VLM Visual Grounding (TWIST-style)

| Week | Milestone | Output |
|------|-----------|--------|
| 1 | Set up Qwen2-VL-2B + HuggingFace PEFT; run RefCOCO zero-shot baseline | Baseline grounding accuracy |
| 2 | Implement dual-expert LoRA: freeze image-understanding adapter, create grounding adapter | Modified model architecture |
| 3 | Annotation team: create 500 bounding-box labels from COCO val2017 images | Grounding SFT dataset |
| 4 | Stage 1 training: grounding-only data, 2 epochs, AdamW 2e-4; overnight run | Stage 1 checkpoint |
| 5 | Stage 2 training: joint instruction tuning with CB-style mixed data | Stage 2 checkpoint |
| 6 | Evaluate: grounded captioning (CIDEr/METEOR), RefCOCO accuracy, catastrophic forgetting probe | Full evaluation table |
| 7 | Ablation: single LoRA vs. twin LoRA vs. full fine-tune (memory consumption vs. performance) | Architecture comparison |
| 8 | Demo video + report + GitHub release | Public artifact |

---

## SECTION 7: FINAL VERDICT SUMMARY

| # | Direction | Verdict | Reason |
|---|-----------|---------|--------|
| A | Mix Distillation CoT (SLM Reasoning) | **YES ✅ — PRIMARY** | Fits 8GB, annotatable, clear novelty over D-CoT baseline, publishable |
| B | VLM Visual Grounding (TWIST-style) | **YES ✅ — FALLBACK** | Complete recipe exists, Qwen2-VL-2B fits hardware, visual demos strong |
| C | Speculative Decoding Draft Model | **YES ✅ — OPTIONAL PARALLEL** | Good for AlienX/inference engineering; low compute, high engineering value |
| 3 | SSM-Hybrid Fine-tuning | NO ❌ | Custom CUDA kernels, too complex for 8-week window |
| 5 | Memory-Augmented Inference | NO ❌ | Infrastructure-heavy, low ML novelty within timeline |
| 8 | Speech LLM | NO ❌ | Data overhead, saturated benchmarks, less differentiating |
| 9 | RAG Reasoning | NO ❌ | Saturated area, no novelty unless rare domain |
| 6 | HW-SW Co-Design (Verilog) | CONDITIONAL ⚠️ | Viable as 2-person side track in parallel; not primary |
| 10 | PRM for Small Models | CONDITIONAL ⚠️ | Viable only if annotation team can produce step-level labels |

---

## FINAL RECOMMENDATION (One Sentence)

**Start with Choice A (Mix Distillation CoT) as primary track using Skip-Thinking + Mix Distillation methodology; pivot to Choice B (TWIST VLM Grounding) if data quality issues arise by Week 3; run Choice C as a 1-person parallel inference engineering track throughout.**

---
*Generated by Agent 10: Synthesis Director — April 25, 2026*
