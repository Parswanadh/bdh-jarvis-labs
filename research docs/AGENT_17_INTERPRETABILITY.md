# AGENT_17_INTERPRETABILITY.md
# Mechanistic Interpretability & Probing: Underexplored Opportunities
### Agent 17 — Mechanistic Interpretability and Probing Specialist
**Date:** April 25, 2026 | **Context:** Student team, small/hybrid/memory-augmented/multimodal models

---

## 1. Survey of Recent Probing & Interpretability Methods

### 1.1 Linear Probing (Workhorse Method)
**What it is:** Train lightweight classifiers (logistic regression, difference-of-means vector)
on frozen hidden states to detect whether a property is linearly separable.

**Recent findings (2024–2026):**
- Layer-wise linear probes on LLMs show correctness signals emerge in *middle layers*
  and saturate before the final layers — suggesting early crystallization of "confidence"
  (Arxiv 2509.10625, 2026).
- Sycophancy behavior is linearly encoded in specific attention heads
  (EACL 2026 — "Sycophancy Hides Linearly in Attention Heads").
- Rhetorical question representations are probed with PCA-reduced embeddings + 3
  probe variants: diffMean, hinge-loss, logistic (Arxiv 2604.14128, 2026).

**Feasibility for students:** ⭐⭐⭐⭐⭐  
Requires only scikit-learn + model activations. Runs on CPU for small models.

---

### 1.2 Sparse Autoencoders (SAEs)
**What it is:** Train overcomplete autoencoders with L1 sparsity on residual stream
activations to decompose polysemantic neurons into monosemantic *feature directions*.

**Recent findings:**
- Sparse Feature Circuits paper (Anthropic, 2024/2025) shows SAEs can discover
  causally implicated subnetworks of interpretable features.
- SAE features at mid-layers are most informative for detecting RAG hallucinations
  (RAGLens, Arxiv 2512.08892, 2025).
- SAE-NOs (Neural Operators) extend to functional/multi-resolution representations
  (Arxiv 2509.03738, 2026).
- NeurIPS 2025: SAE feature explanations can be falsified + revised via auto-interpretability.

**Feasibility for students:** ⭐⭐⭐  
Requires GPU for training; EleutherAI's SAE library helps. Works well on GPT-2 scale.

---

### 1.3 Activation Patching / Causal Tracing
**What it is:** Intervene on specific activations during a forward pass to measure
causal contribution. Identifies which token positions / layers / heads "matter" for output.

**Key tools:** TransformerLens (Neel Nanda), baukit, pyvene.
**Feasibility for students:** ⭐⭐⭐⭐  
Works out-of-box on GPT-2, Pythia, TinyLlama. Low GPU overhead.

---

### 1.4 Attention Attribution & Gradient-Based Methods
**What it is:** Rollout, GradCAM, integrated gradients on attention weights to attribute
output tokens to input tokens/patches.

**Multimodal extension:**
- Cross-modal attribution: which image patches drive text outputs.
- Intra-modal attribution (MSEA framework, Arxiv 2509.22415, 2025): multi-scale
  aggregation is needed because single-patch attribution is noisy/fragmented.

**Feasibility for students:** ⭐⭐⭐⭐  
captum (PyTorch), BertViz, and LLaVA-specific toolkits.

---

### 1.5 Hidden-State Geometry (Representation Analysis)
**What it is:** Analyze the structure of embedding space — cluster separation,
rank, curvature, isotropy — using PCA, CKA (Centered Kernel Alignment),
and the decomposition h = mean_context + mean_token + residual.

**Recent finding:**
- NeurIPS 2024 (OpenReview 1M0qIxVKf6): Disentangling hidden states into
  context-mean + token-mean + interaction term reveals interpretable transformer geometry.
- Hugging Face paper (2025): Sequential layers show near-perfect linear relationship
  (Procrustes similarity ≈ 0.99) — layers transform, not recreate.

**Feasibility for students:** ⭐⭐⭐⭐  
NumPy/SciPy + sklearn. No GPU needed for analysis after activation extraction.

---

### 1.6 Contrast-Consistent Search (CCS / Probing Truth)
**What it is (Burns et al.):** Find linear directions in hidden states that are consistent
across statement/negation pairs — captures implicit truth representation.

**Feasibility for students:** ⭐⭐⭐  
A few hundred labeled pairs + logistic regression. Works on small models.

---

## 2. Which Methods Transfer to Your Setup

| Setup | Best Methods | Notes |
|---|---|---|
| **Small models** (GPT-2, Pythia, TinyLlama) | Linear probing, SAE, activation patching, CKA | Full mechanistic toolkit applies; fast iteration |
| **Hybrid models** (Mamba+Attn, RetNet) | Layer contribution via ablation, hidden-state CKA | Need custom hooks; attention vs. SSM state comparison |
| **Grounded reasoning (RAG)** | SAE on mid-layers, faithfulness probing, causal tracing on retrieved tokens | RAGLens (2025) is a direct template |
| **Memory-augmented** | Gate activation tracking, hidden-state drift analysis | Track how memory gate values change token influence |
| **Multimodal (LLaVA, CLIP-based)** | Intra-modal attribution (MSEA), cross-modal probing, patch importance | Need VQA probe datasets; use LLaVA-7B/BLIP-2 |

---

## 3. Five Mechanistic Paper Directions

### Direction 1: "Where Does Grounding Crystallize? Layer-Wise Faithfulness in RAG"
**Core question:** At which transformer layer does a RAG-augmented model "commit"
to retrieved evidence vs. parametric memory?

**Method:**
- Extract residual stream activations at each layer while varying retrieval quality
  (correct doc, wrong doc, no doc).
- Train linear probes to predict faithfulness label at each layer.
- Perform activation patching: swap residual from "correct doc" run into
  "no doc" run at layer L — find minimal L that restores faithful output.

**Novelty:** Most RAG probing is output-level. This is the first systematic
layer-resolved causal analysis for RAG faithfulness. Extends RAGLens (2025) causally.

**Datasets:** TriviaQA, NQ-Open, HotpotQA with controlled retrieval corruption.

---

### Direction 2: "SSM vs. Attention: A Comparative Geometry of Hybrid Model Hidden States"
**Core question:** Do attention sublayers and SSM sublayers (Mamba-style) encode
qualitatively different information, or is hidden-state geometry equivalent?

**Method:**
- Use CKA to compare layer representations across Mamba, pure-attention, and hybrid
  (e.g., Jamba, Zamba) models matched for parameter count.
- Probe for syntactic, semantic, and positional features at each sublayer type.
- Measure isotropy and rank of representations for each sublayer.

**Novelty:** No systematic mechanistic comparison between hybrid SSM/attention
sublayers exists. Explains *why* hybrids sometimes outperform pure architectures.

---

### Direction 3: "Memory as Bias: How External Memory Gates Modulate Attention Geometry"
**Core question:** When a memory-augmented model (e.g., LM2, Titans-style) reads
from external memory, does it create a distinct directional shift in hidden-state geometry,
or does it blend seamlessly?

**Method:**
- Extract hidden states with and without memory read events.
- Measure directional shift (cosine distance) and rank change in residual stream.
- Train probes to classify "memory-influenced" vs "non-memory-influenced" tokens.
- Causal patch memory gate activations to measure counterfactual influence.

**Novelty:** Interpretability of memory gate dynamics is almost completely unstudied.

---

### Direction 4: "Patch vs. Pixel: What Do Vision Tokens Actually Encode After Cross-Modal Fusion?"
**Core question:** After multimodal fusion in LLaVA/BLIP-2 models, what semantic
content remains in individual vision patch tokens vs. what is lost or redistributed?

**Method:**
- Extract patch-token hidden states at each decoder layer.
- Use linear probes for object category, spatial position, color, texture.
- Compare: pre-fusion visual encoder outputs vs. post-fusion transformer states.
- Apply MSEA multi-scale attribution to check which probed properties causally
  influence final generated tokens.

**Novelty:** Existing work studies cross-modal attention patterns, not what
the vision tokens themselves encode after full multimodal fusion.

---

### Direction 5: "Do Small Models Lie to Themselves? CCS-Based Truth Probing Across Scale"
**Core question:** Does internal truth representation (via CCS) exist even in
models too small to reliably express calibrated uncertainty in outputs?

**Method:**
- Apply Burns et al. CCS on Pythia 70M → 1.4B → 6.9B scale family.
- Probe for truth direction using negation-consistent linear probes.
- Correlate quality of internal truth direction with output calibration (ECE).
- Test if truth direction can be externally leveraged (steering) even when
  model outputs are poorly calibrated.

**Novelty:** CCS has been applied mainly to large models. Small-model regime
is unknown; has direct implications for TinyML and on-device AI safety.

---

## 4. Negative-Result Framing Options

| Negative Outcome | Framing |
|---|---|
| Linear probes fail to predict faithfulness | "Grounding is non-linearly encoded: challenges the Linear Representation Hypothesis for retrieval" |
| SSM and attention layers show identical CKA geometry | "Hybrid advantage is behavioral, not representational: architecture diversity doesn't diversify geometry" |
| Memory gates don't shift hidden-state direction measurably | "Memory-augmentation is transparent: external memory acts as soft prompt rather than geometric shift" |
| Patch tokens lose object semantics post-fusion | "Vision tokens are consumed by fusion: modality-specific semantics don't survive cross-modal integration" |
| CCS truth direction absent in small models | "Truth representation requires scale: CCS requires minimum parameter budget for linear structure to emerge" |
| Activation patching shows diffuse, non-localizable circuits | "RAG faithfulness is distributed: contradicts modular circuit hypothesis for grounded generation" |

> All of these are **publishable negative results** at interpretability venues
> (ICLR Workshop, BlackboxNLP, EMNLP findings). Negative results with clean
> methodology are increasingly valued in the mech interp community (MI Workshop ICML 2026).

---

## 5. Minimal Experiment Design

### Setup Requirements (Student-Feasible)
- **Models:** Pythia-160M/410M, TinyLlama-1.1B, GPT-2, LLaVA-7B (optional, cloud)
- **Hardware:** 8GB VRAM (RTX 4070) for activation extraction; CPU for probe training
- **Libraries:** TransformerLens, baukit/pyvene, scikit-learn, matplotlib, captum
- **Time budget:** 1 experiment / 2 weeks for a 3-person team

### Minimal Pipeline (Direction 1 as Template)

```
Step 1: Dataset Preparation (Day 1–2)
  - Sample 500 QA pairs from TriviaQA
  - Create 3 conditions: correct doc, wrong doc, no doc
  - Tokenize + label with faithfulness binary label

Step 2: Activation Extraction (Day 3–4)
  - Forward pass all 1500 examples through model
  - Save residual stream at every layer (or every 2nd layer)
  - Store as HDF5 for fast retrieval (~2–5 GB for TinyLlama)

Step 3: Layer-Wise Probe Training (Day 5–7)
  - For each layer: fit logistic regression on train split
  - Report AUROC per layer → plot probe accuracy curve
  - Identify "crystallization layer" (max accuracy)

Step 4: Causal Patching (Day 8–10)
  - Use TransformerLens hooks to swap activations between conditions
  - Sweep across layers; measure change in generation faithfulness
  - Report minimal-depth causal layer

Step 5: Write-Up (Day 11–14)
  - 4-page EMNLP short paper or workshop submission
```

### Compute Budget
| Task | Hardware | Time |
|---|---|---|
| Activation extraction (TinyLlama, 1500 samples) | RTX 4070 8GB | ~1–2 hours |
| Probe training (all layers) | CPU | ~20 minutes |
| Activation patching | RTX 4070 | ~2–4 hours |
| SAE training (GPT-2 scale) | RTX 4070 | ~4–8 hours |

---

## 6. Top Recommendation

### ★ Priority Direction: "Where Does Grounding Crystallize? (Direction 1)"

**Why this is the best entry point for your team:**

1. **Directly relevant to your existing work** — If AlienX / GPT-OSS Vision involves
   retrieval or grounding, this is interpretability of your own system.

2. **Incremental over existing work** — RAGLens (2025) uses SAEs for hallucination
   detection but lacks causal patching. Adding activation patching turns it into
   a mechanistic story, not just a correlational probe.

3. **Student-feasible** — TinyLlama + TriviaQA + TransformerLens + sklearn.
   Entire pipeline runs on your RTX 4070 laptop in < 1 day of compute.

4. **Clear story structure** — "Early layers: no grounding. Middle layers: grounding
   emerges. Final layers: committed." This is a compelling narrative for a paper.

5. **Multiple publishable outcomes** — Positive result → short paper. Negative
   result (distributed grounding) → negative-result paper. Both are publishable.

6. **Extends to multimodal** — Can run exact same analysis on LLaVA with visual
   context replacing text retrieval. Same framework, new modality = second paper.

### Recommended Stack
```
Model:       TinyLlama-1.1B (or Pythia-410M for faster iteration)
Framework:   TransformerLens (hooks), pyvene (patching), sklearn (probes)
Dataset:     TriviaQA-open + NQ-Open (HuggingFace datasets)
Analysis:    matplotlib layer-curve plots, CKA heatmaps
Submission:  EMNLP 2026 Findings / BlackboxNLP 2026 / ACL Workshop
```

---
*Generated by Agent 17 — Mechanistic Interpretability and Probing Specialist*
*AlienX Research Stack | April 2026*
