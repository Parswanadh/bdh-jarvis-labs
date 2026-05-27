# AGENT 06: Benchmark & Evaluation Framework Design
## Tier-1 Publishable Opportunities for Small-Model Multimodal Reasoning

**Date:** April 2026  
**Focus Areas:** Small-model multimodal reasoning, grounded faithfulness, hybrid architecture reasoning, edge-constrained reasoning, memory-grounding behavior, cross-domain grounding  
**Design Principle:** Cheaper than model invention, yet Tier-1 venue relevant (NeurIPS Datasets/Benchmarks, ICML, ACL, ICLR, CVPR)

---

## 1. Benchmark Gaps

### 1.1 Small-Model Multimodal Reasoning
Current multimodal reasoning benchmarks (EMMA, MathVista, MMMU) target large models and emphasize text-dominant reasoning or shallow visual cues [web:2]. There is **no dedicated benchmark** for small models (<3B parameters) that tests **organic cross-modal reasoning** where neither modality can be solved independently. Existing small-model studies evaluate 72 SLMs across 17 reasoning benchmarks but focus on unimodal settings [web:22].

### 1.2 Grounded Faithfulness at Small Scale
Faithfulness evaluation pipelines like FaithEvi [web:6] and MMHal-Bench [web:23] target large multimodal models. Small models exhibit different hallucination patterns (more frequent object substitution rather than invention) that are not captured by existing coarse-grained metrics. No benchmark isolates **step-level perceptual faithfulness** for sub-3B parameter models.

### 1.3 Hybrid Architecture Reasoning
Hybrid Transformer-SSM architectures achieve 2.1x inference throughput with 91% in-context learning retention [web:7], yet **no unified reasoning benchmark** exists that tests where attention layers are strictly necessary versus where SSM layers suffice. Current benchmarks conflate architecture types rather than isolating reasoning patterns that expose hybrid strengths/weaknesses [web:12].

### 1.4 Edge-Constrained Multimodal Reasoning
TinyML benchmarking focuses on unimodal latency and energy [web:8]. Edge-deployed multimodal models like MiniCPM-V [web:39] lack standardized evaluation under **real constraints**: thermal throttling, battery-bound inference, dynamic memory pressure, and sensor noise. No benchmark integrates energy-latency-accuracy tradeoffs for multimodal edge tasks [web:36].

### 1.5 Memory-Grounding Behavior
Episodic, semantic, and procedural memory evaluation for LLMs exists [web:14][web:31], but **multimodal memory-grounding** (visual episodic recall, cross-modal semantic retrieval) is virtually unbenchmarked. LoCoMo evaluates long-context dialogues but is text-only [web:14]. SORT tasks assess episodic memory but not in multimodal contexts [web:31].

### 1.6 Cross-Domain Grounding
Cross-modal retrieval benchmarks (WikiDO, MultiHaystack) test retrieval accuracy [web:4][web:44], but **cross-domain grounding**—maintaining consistent object/attribute bindings when visual and textual domains shift (e.g., medical → natural → industrial)—is not systematically evaluated for small models. Domain-specific VLM benchmarks exist [web:24] but lack cross-domain transfer protocols.

---

## 2. Evaluation Gaps

| Gap Area | Current State | Missing Element |
|----------|---------------|-----------------|
| **Step-wise faithfulness** | Chain-level scores only [web:6] | Per-step object extraction + grounding for small models |
| **Energy-reasoning tradeoff** | Energy benchmarks exist for TinyML [web:8] | Multimodal "useful tokens per joule" metric |
| **Retrieval vs. Reasoning** | Conflated in VQA benchmarks [web:4] | Disentangled evaluation: retriever accuracy vs. reasoning accuracy |
| **Dense context reasoning** | Needle-in-Haystack dominates [web:35] | Global pattern recognition across dense multimodal contexts |
| **Adversarial robustness** | Adversarial perturbations tested on large models | Structured adversarial testing for quantized/pruned small multimodal models [web:22] |
| **Memory architecture scaling** | RAG vs. ICL compared [web:14] | Systematic ablation of memory architectures on small multimodal models |

---

## 3. What Current Papers Fail to Measure

1. **Fine-grained failure modes of small multimodal models**: Current benchmarks report aggregate accuracy. They do not report *how* small models fail differently from large models (e.g., texture bias over shape, linguistic shortcut exploitation) [web:22].

2. **Real-world edge deployment degradation**: Papers report benchmark scores on clean data. They do not measure performance under sensor noise, low-light conditions, thermal throttling, or concurrent app memory pressure [web:8][web:36].

3. **Faithfulness-reasoning tension**: No benchmark measures whether forcing higher faithfulness (grounding every claim to visual evidence) degrades reasoning quality in small models—a critical tradeoff for parameter-constrained systems [web:6].

4. **Cross-modal interference**: When text and visual inputs conflict (adversarially or naturally), small models show predictable collapse toward textual bias. No benchmark quantifies this interference pattern [web:27].

5. **Hybrid layer utilization**: Hybrid architecture papers report aggregate throughput and perplexity [web:7]. They do not report *which layers* are activated during reasoning tasks versus retrieval tasks, obscuring whether the hybrid design is actually utilized.

6. **Long-term multimodal memory**: Even long-context benchmarks like MMReD [web:35] test dense reasoning but not *memory consolidation*—the ability to integrate information across temporally distant multimodal episodes.

---

## 4. Five Candidate Benchmark Papers

### Candidate 1: EdgeMM-Reason — Edge-Constrained Small Multimodal Reasoning Benchmark
**Core Idea:** A benchmark that tests multimodal reasoning under explicit compute, memory, and energy budgets on edge devices. Tasks require cross-modal reasoning (diagram + text, video + audio) with inference constrained to <2GB RAM, <5W power draw, and <500ms latency.

**Key Metrics:**
- Accuracy under thermal throttling
- "Useful tokens per joule" [web:36]
- Degradation curve as battery level drops
- Recovery from memory pressure eviction

**Dataset Sources:** Adapt MathVista [web:5], ScienceQA [web:10], Ego4D video clips; downsample to edge-appropriate resolution.

**Novelty:** First benchmark to make edge constraints a *first-class evaluation dimension* rather than a post-hoc deployment consideration.

---

### Candidate 2: FaithBench-S — Step-Level Faithfulness for Small Multimodal Models
**Core Idea:** Adapt FaithEvi [web:6] pipeline to small models (<3B) with modifications for their failure patterns. Extract claimed objects per reasoning step, verify existence via visual grounding, and aggregate into chain-level and step-level faithfulness scores.

**Key Innovations:**
- Automated object extractor fine-tuned on small model outputs (since small models generate different claim structures)
- Local (per-step) and global (chain-level) faithfulness disaggregation
- Faithfulness-reasoning frontier curve (Pareto tradeoff)
- Hallucination taxonomy specific to small models: substitution > invention > omission

**Dataset:** 5K image-question pairs across 12 object categories, building on MMHal-Bench structure [web:23] but scaled for automated evaluation.

---

### Candidate 3: HybridReason-Bench — Diagnostic Reasoning for Hybrid Architectures
**Core Idea:** A reasoning benchmark that explicitly diagnoses *when* hybrid Transformer-SSM models succeed/fail by task type. Tasks are categorized by required context length, need for precise retrieval, and sequential structure.

**Task Categories:**
- **Local retrieval**: Short-span exact match (should favor SSM layers)
- **Global integration**: Long-range dependency resolution (should favor attention layers)
- **Structured reasoning**: Math, code, causal chains (should require both)
- **Noisy contexts**: Irrelevant distractors mixed with signal (tests layer routing)

**Key Metric:** Layer activation map per task type—using probe-based analysis to measure which hybrid layers are actually engaged [web:7][web:12].

---

### Candidate 4: CrossGround-Bench — Cross-Domain Visual Grounding for Small Models
**Core Idea:** Systematic evaluation of how well small multimodal models maintain grounding accuracy when visual/textual domains shift. Models trained on natural images are tested on medical, industrial, satellite, and synthetic domains without fine-tuning.

**Structure:**
- 4 source domains → 4 target domains = 16 transfer pairs
- Grounding tasks: referring expression comprehension, phrase grounding, embodied pointing [web:18]
- Metric: IoU retention rate (target domain IoU / source domain IoU)

**Novelty:** Quantifies *domain grounding decay*—how quickly spatial-textual alignment degrades under distribution shift. Critical for edge deployment where training and runtime domains often differ.

---

### Candidate 5: MemModal-Bench — Multimodal Memory-Grounded Behavior Evaluation
**Core Idea:** Evaluate memory-augmented small multimodal models across episodic, semantic, and procedural memory types in multimodal contexts [web:14][web:31].

**Task Designs:**
- **Episodic**: Recall the order of visual events from a video sequence; answer questions about spatial-temporal context [web:31]
- **Semantic**: Retrieve object properties across modalities ("What color was the car in the image from three episodes ago?")
- **Procedural**: Follow multimodal instruction sequences where earlier steps modify later visual contexts

**Key Metric:** Memory-token efficiency (accuracy per memory retrieval token) and memory-type transfer (does episodic training improve semantic performance?)

---

## 5. Annotation Plans

### Automated + Human Hybrid Pipeline

| Stage | Method | Cost Estimate | Quality Control |
|-------|--------|---------------|-----------------|
| **Data Collection** | Curate from existing datasets (ScienceQA, Ego4D, VCR, NLVR2, RefCOCO) | Low (licensing only) | Deduplication against training corpora |
| **Question Generation** | GPT-4o / Gemini 2.5 Pro for initial generation; template-based for scale | ~$0.10/sample | Human spot-check 10%; inter-annotator agreement >0.85 |
| **Visual Grounding Verification** | Automated IoU checking against annotated bounding boxes (where available) | Very low | Manual verification for ambiguous cases |
| **Faithfulness Scoring** | FaithEvi-style pipeline: extract claims → object detection → preference polling [web:6] | ~$0.05/sample | LLM-as-judge calibrated against human ratings on 200 samples |
| **Edge Labels** | Automated profiling on reference hardware (Jetson Orin Nano, Raspberry Pi 5, phone SoC) | Hardware cost only | Cross-device validation to ensure generalizable constraints |
| **Cross-Domain Labels** | Domain metadata from dataset sources; manual verification of domain boundaries | Medium | Expert annotation for medical/industrial domain classification |

### Annotation Efficiency Tactics
- **Synthetic adversarial generation**: Use image editing to create faithful/unfaithful pairs automatically (following HallusionBench [web:27])
- **LLM judge calibration**: Fine-tune a small judge model (e.g., LLaMA-3.1-8B) on 500 human-annotated examples, then deploy at scale
- **Crowdsourcing selective verification**: Only send borderline cases to human annotators; clear cases resolved automatically

---

## 6. Baseline Model List

### Small Multimodal Models (<3B parameters)
| Model | Parameters | Modality | Architecture | Notes |
|-------|-----------|----------|--------------|-------|
| **MiniCPM-V 2.6** | ~2.8B | Vision-Language | Edge-optimized | Baseline for edge tasks [web:39] |
| **Phi-4-multimodal** | ~2.4B | Vision-Audio-Language | Transformer | Strong small-model baseline |
| **MoLMO** | ~1B | Vision-Language | Open-weights | Good for grounding tasks [web:18] |
| **LLaVA-Phi-3** | ~3B | Vision-Language | Transformer | Standard lightweight baseline |
| **Gemma-3-4B-IT** | ~4B | Vision-Language | Transformer | Slightly above target but common comparison point |
| **Qwen2.5-VL 3B** | ~3B | Vision-Language | Transformer | Strong Asian-market edge model |

### Hybrid Architecture Models
| Model | Type | Use Case |
|-------|------|----------|
| **Jamba** | Transformer-Mamba Hybrid | Long-context reasoning baseline |
| **Zamba** | Hybrid SSM-Attention | Efficiency-focused comparison |
| **Gated Linear Attention variants** | Linear attention hybrids | Throughput comparison |

### Memory-Augmented Variants
- Standard + RAG (retrieval-augmented)
- Standard + In-Context Learning (episodic)
- Standard + Prompt-optimized (procedural)
- Standard + Agentic memory (semantic) [web:14]

---

## 7. Fastest-to-Publish Direction

### Recommendation: **FaithBench-S** (Step-Level Faithfulness for Small Multimodal Models)

**Rationale:**
1. **Low data cost**: Can reuse existing image-question datasets (MMHal-Bench [web:23], POPE, CHAIR) with minimal new annotation
2. **Automated evaluation**: FaithEvi-style pipeline [web:6] is highly automatable—extract claims, detect objects, compute IoU/grounding scores without human-in-the-loop at scale
3. **Clear gap**: Existing faithfulness work targets large models; small model failure patterns are different but unstudied
4. **High relevance**: Hallucination in small models is a critical blocker for edge deployment, directly addressing industry needs
5. **Fast baseline comparison**: 6+ small models are readily evaluable via API or local inference; no training required

**Publication Timeline Estimate:**
- Month 1: Data curation + automated pipeline development
- Month 2: Model evaluation + human calibration study (200 samples)
- Month 3: Analysis + paper writing
- Month 4: Submission to **ACL/EMNLP** (benchmark track) or **NeurIPS Datasets & Benchmarks**

**Venue Fit:**
- **Primary target**: ACL/EMNLP Benchmark Track, NeurIPS Datasets & Benchmarks
- **Secondary**: ICLR (if emphasizing the faithfulness-reasoning Pareto frontier as a learning problem)

---

## Summary Table: Benchmark Candidates at a Glance

| Benchmark | Core Gap | Novel Metric | Data Cost | Compute Cost | Time to Submit | Target Venue |
|-----------|----------|--------------|-----------|--------------|----------------|--------------|
| **FaithBench-S** | Small model faithfulness | Step-level faithfulness score + faithfulness-reasoning frontier | Low | Low | 4 months | ACL/EMNLP, NeurIPS DB |
| **EdgeMM-Reason** | Edge-constrained multimodal reasoning | Useful tokens per joule | Medium | Medium (hardware needed) | 6 months | MobiSys, MLSys, NeurIPS DB |
| **HybridReason-Bench** | Hybrid architecture diagnostic | Layer activation map by task type | Low | Low | 5 months | ICLR, ICML |
| **CrossGround-Bench** | Cross-domain grounding decay | IoU retention rate | Medium | Low | 5 months | CVPR, ICCV |
| **MemModal-Bench** | Multimodal memory types | Memory-token efficiency | Medium | Low | 6 months | NeurIPS, ACL |

---

## Strategic Notes

- Benchmark papers are **citation magnets** when they become standard evaluation references. Target a gap that the research community is already discussing but lacks tools to measure.
- The "small model" angle is timely: 2025-2026 has seen explosive interest in SLMs, but evaluation infrastructure lags behind model releases [web:26].
- Hybrid architectures are a major trend (Jamba, Zamba, Griffin) but evaluation is fragmented. A unified benchmark would capture significant attention.
- Partnering with a hardware group (for EdgeMM-Reason) or a vision group (for CrossGround-Bench) can strengthen submission credibility without requiring deep domain expertise.
- FaithBench-S requires the least new infrastructure and can leverage off-the-shelf vision models for automated scoring—ideal for a student-led project with limited compute.
