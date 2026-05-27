# AGENT 18: Compression & Efficiency Science Scout

## Research Brief: When Efficiency Becomes Science

> **Core Thesis:** Efficiency constraints in modern LLMs are no longer mere engineering targets. They reveal architectural tradeoffs, expose reasoning fragilities, and define new operating regimes where quantization, pruning, and token efficiency become the *subject* of scientific inquiry rather than just a post-hoc fix.

---

## 1. Literature Map

### 1.1 Pruning in Hybrid Models (SSM + Transformer)

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| Structured SSM pruning | **On Pruning State-Space LLMs** (2025) [web:43] | SSM states tolerate up to 50% unstructured pruning; SSM *heads* are fragile. Output projection more sensitive than input projection. |
| Hybrid compression | **Mamba-Shedder** (2025) [web:35] | Pruning Mamba blocks + Transformer blocks causes severe degradation; granular strategies (Mamba block → MLP → MHA → SSM channel) perform best. |
| Stability-aware pruning | **EMNLP 2025: Efficient Unstructured Pruning of Mamba** [web:5] | Recurrent dynamics of SSMs require stability-aware pruning; global strategies outperform layer-wise approaches. |
| Group-aware hybrid pruning | **NeurIPS 2025 Poster** [web:10] | Structural integrity of SSM blocks must be preserved; combined with FFN/embedding/layer pruning + distillation yields unified recipe. |
| SparseSSM | **Tuo et al.** [web:39] | Prune 50% of SSM weights *without* fine-tuning and observe *no* zero-shot accuracy loss. |

**Open observation:** Hybrid models (Mamba-Transformer, MoE-SSM) have *emergent* pruning sensitivities that differ from pure Transformers. The interaction between sparse expert routing and state-space recurrence creates coupled fragility modes that remain uncharacterized.

### 1.2 Quantization Behavior in Multimodal / SSM Models

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| SSM quantization recipe | **PTQ for Selective SSMs** (2025) [web:11] | 8-bit weight-activation quantized Mamba 2.8B achieves 1.72× lower latency on Orin Nano 8G with only 0.9% accuracy drop. |
| Edge SSM quantization | **Quantizing Small-Scale SSMs for Edge AI** (2025) [web:6] | 4-bit outside SSM layer; state matrices (A, B, C, D) need careful analysis. State `x` and `A` can go lower than 4-bit with task-dependent degradation. |
| Fine-grained formats | **Comprehensive Study of Low-Bit Formats** (2025) [web:13] | INT4 consistently outperforms FP4 with Hadamard rotation; symmetric clipping resolves gradient bias in QAT. |
| Multimodal token pruning | **Grounding-Aware Token Pruning (GAP)** (2025) [web:23] | Standard pruning destroys visual grounding (REC drops drastically); simple position-ID adjustment recovers 90% of original performance with zero training overhead. |

**Open observation:** SSM quantization is *not* a solved problem. The state vector `x` and discretization matrix `A` behave differently from attention weights — their quantization sensitivity is task-dependent and understudied in multimodal settings.

### 1.3 Memory-Reasoning Pareto Frontiers

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| Optimal CoT length | **When More is Less** (2025) [web:36] | Excessive chain-of-thought length impairs reasoning; optimal length scales with task difficulty but decreases with model capability. |
| Thinking-Optimal Scaling | **TOPS** (2025) [web:40] | Different domains have different optimal scaled-length distributions; shortest correct System-2 response is optimal. |
| Budget-aware agents | **Anytime Verified Agents (AVA)** (2025) [web:21] | Adaptive compute allocation (search, tool use, verification) achieves 20–40% cost reduction at equivalent reliability. |
| Self-Budgeter | **SelfBudgeter** (2025) [web:25] | Self-adaptive token allocation: pre-estimate reasoning cost, allocate budget dynamically; achieves 74.47% token savings. |
| KV cache quantization | **KVTuner** (2025) [web:20] | Sensitivity-aware mixed-precision KV cache quantization enables sub-4-bit (3.59-bit) with minimal degradation. |

**Open observation:** The memory-reasoning frontier is rarely studied as a *Pareto curve* with controlled variables. Most work optimizes one axis at a time (e.g., token count *or* KV cache size). Joint optimization under hard memory budgets is underexplored.

### 1.4 Low-Bit Inference Effects on Reasoning

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| Systematic degradation | **Quantization Meets Reasoning** (2025) [web:16] | AWQ/GPTQ introduce up to 32.39% accuracy degradation on Llama-3; average 11.31% drop on math reasoning. |
| Local vulnerability | **Exploring and Mitigating Low-Bit Degradation** (2026) [web:17] | PTQ disproportionately elevates *method* and *execution* errors (not conceptual); first vulnerable step cascades to final answer. 332 curated examples + 3–5 min single-GPU compute recover 4-bit reasoning to full-precision baseline. |
| Format comparison | **GRAINED LOW-BIT QUANTIZATION FORMATS** [web:8] | INT formats provide dual advantage of superior accuracy and greater hardware efficiency; fine-grained INT outperforms FP at low bits. |

**Open observation:** Reasoning degradation is *not uniform*. Low-bit inference disproportionately harms arithmetic execution and procedural step tracking, not high-level planning. This creates an opportunity to design *selective* precision models that protect reasoning-critical paths.

### 1.5 Efficient Grounding

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| Direct attention supervision | **Direct Visual Grounding by Directing Attention** (2026) [web:32] | Standard next-token prediction provides insufficient signal for visual grounding; direct attention supervision improves geometric tasks, pointing, and REC. |
| GAP method | **Grounding-Aware Token Pruning** (2025) [web:23] | Pruning weakens grounding; position-ID adjustment (GAP) recovers REC to 51.42% (90% of unpruned). |
| Lightweight REC | **YORO** [web:31] | End-to-end visual grounding with detection tokens; single forward pass without expensive region proposal networks. |

**Open observation:** Grounding is computationally expensive because it requires fine-grained spatial-textual alignment. Current pruning/quantization methods optimize for generic perplexity, not grounding accuracy. The *geometry* of attention in grounding tasks is fundamentally different from next-token prediction.

### 1.6 Budget-Aware Evaluation Frameworks

| Dimension | Key Work | Finding |
|-----------|----------|---------|
| Cost-quality Pareto | **SPECS** (2026) [web:12] | Speculative test-time scaling achieves Pareto-frontier latency/accuracy trade-off. |
| Adaptive allocation | **AVA** (2025) [web:21] | Uncertainty estimation + value-of-information + selective verification cascades with early exits. |
| Self-Budgeter | **SelfBudgeter** (2025) [web:25] | Dual-phase training: pre-estimate cost, then allocate tokens rationally by problem complexity. |
| Pareto inference | **BentoML** [web:7] | No universal best config; Pareto-optimal configurations reveal how teams must balance TTFT, throughput, quality, and cost. |

---

## 2. Underexplored Questions

### 2.1 Pruning in Hybrid Models — What Remains Unknown?

1. **Coupled sparse routing + SSM dynamics:** In MoE-SSM hybrids (e.g., Jamba, Nemotron 3 Super) [web:2], how does expert routing interact with state-space recurrence when both are pruned? The literature treats them separately.
2. **Cross-layer state dependency:** SSM layers have recurrent memory. Pruning layer `L` affects not just output quality at `L` but *state initialization* for layer `L+1`. No work has characterized this propagation.
3. **Head vs. state sensitivity tradeoff:** The finding that SSM heads are fragile [web:43] but states are robust suggests a *structured* pruning problem: which parameters control "what to remember" vs. "how to update." This is unmapped.

### 2.2 Quantization in SSM / Multimodal — What Remains Unknown?

1. **State matrix `A` quantization limits:** The discretization matrix `A` in Mamba controls exponential decay / state evolution. How low can `A` go before recurrent stability breaks? The task-dependence noted in [web:6] is anecdotal, not systematic.
2. **Cross-modal quantization coupling:** In MLLMs, vision encoder outputs are quantized alongside language model weights. Does vision quantization create *spatial* errors that propagate differently from language token errors during grounding? No study exists.
3. **Reasoning-specific quantization budgets:** Low-bit quantization disproportionately hurts arithmetic execution [web:17]. Can we assign *different bit-widths* to arithmetic vs. planning modules? This requires identifying which parameters serve which reasoning function.

### 2.3 Memory-Reasoning Pareto — What Remains Unknown?

1. **Joint KV-cache × CoT-length optimization:** Current work optimizes KV cache compression [web:20] and CoT length [web:36] independently. The *joint* frontier — where KV cache limits force shorter CoT, and shorter CoT changes attention patterns — is unstudied.
2. **Hardware-aware reasoning scaling:** TOPS [web:40] optimizes length distribution, but assumes infinite memory. What is the optimal CoT length *given* a fixed KV-cache budget on a specific device?
3. **State-space memory models:** For SSMs, memory is not KV cache but hidden state `h`. How does `h` compression (pruning/quantization) interact with reasoning length? This is unique to SSMs and entirely open.

### 2.4 Efficient Grounding — What Remains Unknown?

1. **Token-pruning vs. spatial resolution:** Pruning visual tokens reduces compute but may destroy fine-grained spatial discrimination. The *resolution-compute* frontier for grounding is unmapped.
2. **Quantization-aware grounding metrics:** Standard grounding metrics (IoU, accuracy) are binary. Where along the precision axis does grounding degrade? Is there a "graceful degradation" region? No quantization-aware grounding benchmark exists.
3. **Dynamic grounding budgets:** Can the model decide *how many visual tokens* to retain based on query complexity? A "visual SelfBudgeter" does not exist.

---

## 3. Five Candidate Paper Ideas

### Paper 1: "Coupled Pruning in MoE-SSM Hybrids: When Sparse Routing Meets Recurrent Memory"

**Idea:** Systematically study the interaction between expert pruning and SSM state pruning in hybrid architectures (Jamba, Nemotron 3 Super). Map the 2D pruning space (expert sparsity × state sparsity) and characterize coupled failure modes.

**Novelty:** First work to treat MoE routing and SSM dynamics as a *coupled* optimization problem. Shows that pruning experts and SSM states independently (current practice) is suboptimal and can create hidden instability.

**Key experiment:** Prune Jamba at varying (expert-sparsity, state-sparsity) ratios; measure perplexity, reasoning accuracy, and state-divergence metrics. Identify "safe operating regions."

### Paper 2: "State-Aware Quantization: Why the Mamba `A` Matrix is Not Like Attention Weights"

**Idea:** The discretization matrix `A` in selective SSMs controls exponential decay rates. Quantizing `A` to low precision changes the *eigenstructure* of state evolution. Derive stability bounds for quantized `A` and show that task-dependent reasoning degradation correlates with spectral perturbation.

**Novelty:** First principled analysis linking quantization bit-width to *dynamical system stability* in SSMs. Moves beyond empirical "try and see" to predictive theory.

**Key experiment:** Quantize `A` at varying precisions (8-bit → 2-bit); measure (a) zero-shot accuracy, (b) state divergence over sequence length, (c) eigenvalue perturbation. Show that reasoning tasks requiring long-horizon state tracking degrade before short-horizon tasks.

### Paper 3: "Selective Precision Reasoning: Protecting Arithmetic Paths in Low-Bit LLMs"

**Idea:** Building on the finding that low-bit quantization disproportionately harms *execution* and *method* errors [web:17], design a selective-precision architecture where arithmetic operations (calculator module, number tokens, step separators) are protected at higher precision while general text generation runs at INT2/INT3.

**Novelty:** Challenges the "uniform quantization" paradigm. Treats reasoning as a heterogeneous process requiring heterogeneous precision. Shows that 2-bit general + 8-bit arithmetic can match 4-bit uniform at lower memory cost.

**Key experiment:** Implement mixed-precision forward pass with arithmetic-operation detection; evaluate on GSM8K, MATH, AIME. Compare memory vs. accuracy Pareto curves against uniform AWQ/GPTQ.

### Paper 4: "The Joint KV-Cache / CoT-Length Pareto Frontier: Memory-Constrained Optimal Reasoning"

**Idea:** Current optimal-CoT-length work [web:36] assumes infinite KV cache. Combine KV-cache quantization [web:20] with CoT-length optimization [web:40] to derive the *true* memory-reasoning Pareto frontier under device constraints.

**Novelty:** First work to jointly optimize memory (KV cache bits × length) and reasoning quality (CoT length × accuracy). Proposes a hardware-aware scaling law: optimal CoT length `L*` is a function of both task difficulty `D` and KV-cache budget `M`.

**Key experiment:** Sweep (KV-bit-width, CoT-length) on Qwen/LLaMA models; measure reasoning accuracy. Fit a parametric Pareto surface. Show that aggressive KV-cache quantization shifts the optimal CoT length *downward* — shorter reasoning is better when memory is tight.

### Paper 5: "Visual SelfBudgeter: Adaptive Token Allocation for Efficient Multimodal Grounding"

**Idea:** Extend SelfBudgeter [web:25] to multimodal grounding. The model pre-estimates visual complexity from the query+image and allocates a visual token budget. High-spatial-resolution queries get more tokens; simple queries get aggressive pruning.

**Novelty:** First adaptive visual token allocation method. Grounding performance is preserved because token count is matched to *task spatial complexity*, not image size.

**Key experiment:** Train a lightweight budget-estimator on REC datasets. Evaluate on RefCOCO/RefCOCO+ with varying token budgets. Show that adaptive allocation matches full-token performance at 40–60% of compute.

---

## 4. What is Feasible on 8GB VRAM

Given your RTX 4070 laptop (8GB VRAM), here is the realistic experimental landscape:

| Model Scale | Configuration | VRAM | Feasible? |
|-------------|---------------|------|-----------|
| **Mamba-2.8B** | FP16 inference | ~5.6 GB | ✅ Full precision baseline |
| **Mamba-2.8B** | 8-bit PTQ (weight-activation) [web:11] | ~2.8 GB | ✅ With significant headroom |
| **Mamba-2.8B** | 4-bit QLoRA fine-tuning [web:22] | ~3–4 GB | ✅ Full fine-tuning on 8GB |
| **LLaMA-3 8B** | QLoRA 4-bit (training) [web:22] | ~6–7 GB | ✅ Possible but tight |
| **LLaMA-3 8B** | AWQ/GPTQ 4-bit inference | ~4 GB | ✅ Easy |
| **LLaMA-3 8B** | 2-bit inference (selective precision) | ~2 GB | ✅ (Paper 3 idea) |
| **Hybrid (Jamba/MoE)** | 4-bit + sparse expert routing | ~4–6 GB | ✅ Small-scale experiments |
| **Multimodal (LLaVA-1.5 7B)** | 4-bit inference + visual token pruning | ~5–6 GB | ✅ (Paper 5 idea) |
| **Vision encoder + SSM LM** | 4-bit both modalities | ~4–5 GB | ✅ End-to-end feasible |

### Practical Constraints & Opportunities:

- **Full fine-tuning** is impossible on 8GB. QLoRA is your primary tool [web:22].
- **You can study** models up to ~8B parameters in quantized regimes. This is sufficient for all 5 paper ideas above.
- **Batch size = 1** for most experiments. Gradient accumulation is required.
- **Smol-Mamba-1.9B** and **Phi-Mamba-1.5B** [web:43] are ideal — small enough for fast iteration, large enough to show meaningful scaling behaviors.
- **QLoRA + 4-bit** allows fine-tuning Mamba-2.8B or LLaMA-3 8B. This enables the "332 examples + 3–5 min" recovery experiments from [web:17].
- **Edge deployment studies** on your 8GB card directly address the deployment-relevance that reviewers demand. "We validate on consumer hardware" is a strong framing.
- **RunPod / cloud GPUs** can supplement for larger sweeps, but the core experiments should be reproducible on your laptop.

---

## 5. Reviewer-Proof Framing

### 5.1 Why Efficiency is Science, Not Engineering

To avoid the "this is just engineering" reviewer objection, frame each paper as follows:

| Reframe Engineering → Science | |
|--------------------------------|---|
| "We pruned a model to run faster" | → "Pruning reveals a *coupled failure mode* between sparse routing and recurrent dynamics that was previously invisible." |
| "We quantized to save memory" | → "Quantization perturbation of the `A` matrix changes the *spectral structure* of state evolution, linking numerical precision to dynamical stability." |
| "We shortened reasoning chains" | → "We derive a *memory-aware scaling law* for optimal CoT length, showing hardware constraints fundamentally alter reasoning strategy." |
| "We compressed visual tokens" | → "Token pruning exposes that grounding accuracy depends on *spatial-textual attention geometry*, not just perplexity." |

### 5.2 Reviewer-Resistant Claims

1. **"We provide the first systematic map of..."** — reviewers love maps. Map the pruning space, quantization space, or Pareto space.
2. **"We uncover a counterintuitive phenomenon..."** — e.g., shorter CoT can be better under memory constraints; SSM states tolerate 50% pruning but heads do not.
3. **"We derive a theoretical bound / scaling law..."** — spectral perturbation bounds for quantized `A`; memory-reasoning scaling law.
4. **"Our method is validated on consumer hardware"** — directly addresses deployment relevance and shows the author understands real-world constraints.
5. **"We open-source code and benchmarks"** — reviewers increasingly expect this. Your GitHub portfolio history supports this framing.

### 5.3 Venue Targeting

| Paper | Target Venues |
|-------|--------------|
| Paper 1 (Coupled Pruning) | NeurIPS, ICML, ICLR |
| Paper 2 (State-Aware Quantization) | ICLR, NeurIPS (theory + systems) |
| Paper 3 (Selective Precision) | EMNLP, ACL, ICML |
| Paper 4 (Joint Pareto) | NeurIPS, ICML (systems track) |
| Paper 5 (Visual SelfBudgeter) | CVPR, ECCV, NeurIPS |

---

## 6. Top 2 Directions

### 🥇 Direction A: "Selective Precision Reasoning" (Paper 3)

**Why this is the strongest candidate:**

1. **Directly addresses a known, unexplained phenomenon:** The finding that quantization disproportionately harms execution/method errors [web:17] is recent (2025–2026) but the *solution* (heterogeneous precision) is unstudied. You would be first.
2. **Feasible on 8GB VRAM:** Requires only QLoRA fine-tuning on 4–8B models with mixed-precision hooks. The "332 examples + 3–5 min" precedent [web:17] shows minimal compute is needed for strong results.
3. **Strong reviewer story:** "We show that reasoning is not a monolithic process — it has arithmetic-critical paths that need protection." This is a scientific claim about model internals, not just an engineering trick.
4. **Generalizes across methods:** The framework is "quantizer- and architecture-agnostic" [web:17], meaning you can test on Qwen, LLaMA, and potentially Mamba.
5. **Clear evaluation:** GSM8K, MATH, AIME provide standard benchmarks. Memory savings are measurable. The Pareto improvement is provable.

**Immediate next step:**
- Instrument a 4-bit quantized LLaMA-3 8B to track which layers/tokens handle arithmetic operations.
- Identify arithmetic-critical weight matrices (feed-forward layers near number tokens, step-separator tokens).
- Protect these at 8-bit while running the rest at INT2/INT3.
- Measure accuracy and memory. If it works, you have a paper.

---

### 🥈 Direction B: "The Joint KV-Cache / CoT-Length Pareto Frontier" (Paper 4)

**Why this is the second-strongest candidate:**

1. **Connects two active research threads:** KV-cache quantization [web:20] and optimal CoT length [web:36] are both hot topics. Nobody has combined them.
2. **Derives a new scaling law:** The claim that `L* = f(D, M)` — optimal reasoning length depends on both task difficulty and memory budget — is a novel theoretical contribution.
3. **Hardware-relevant:** Validating on your 8GB laptop makes this a real-world systems paper, not just a theory exercise.
4. **SSM angle:** This direction naturally extends to SSMs where memory is hidden state, not KV cache. You can pioneer "state-aware reasoning scaling" — a completely new concept.

**Immediate next step:**
- Fix a model (Qwen-7B or LLaMA-3 8B).
- Sweep KV-cache bit-width (2, 3, 4, 8, 16) and CoT length (controlled via prompt / generation limits).
- Measure reasoning accuracy on math benchmarks.
- Fit a parametric surface `Accuracy = f(KV_bits, CoT_length, task_difficulty)`.
- Show that the optimal CoT length shifts downward as KV bits decrease.
- For SSMs: sweep hidden-state compression instead of KV cache.

---

## 7. Suggested Reading Priority

| Priority | Paper | Why |
|----------|-------|-----|
| 🔴 Critical | [web:17] "Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning" | Foundation for Paper 3 |
| 🔴 Critical | [web:36] "When More is Less: Understanding Chain-of-Thought Length in LLMs" | Foundation for Paper 4 |
| 🔴 Critical | [web:43] "On Pruning State-Space LLMs" | Foundation for understanding SSM fragility |
| 🟡 Important | [web:11] "A Post-Training Quantization Recipe for Selective State Space Models" | SSM quantization baseline |
| 🟡 Important | [web:25] "SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning" | Budget-aware framework |
| 🟡 Important | [web:23] "Grounding-Aware Token Pruning" | Multimodal efficiency |
| 🟢 Context | [web:2] "Nemotron 3 Super: Hybrid Mamba-MoE" | State-of-the-art hybrid architecture |
| 🟢 Context | [web:13] "Comprehensive Study of Fine-Grained Low-Bit Formats" | Format theory |

---

*Generated by Agent 18: Compression & Efficiency Science Scout*
*Date: April 25, 2026*
