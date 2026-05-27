# AGENT_19_SYNTHETIC_DATA_CURRICULUM.md
# Agent 19: Synthetic Data and Curriculum Designer
# Mission: Research whether carefully designed synthetic data, curricula, or distillation
# schemes can unlock publishable small-model reasoning results under low compute.
# Generated: April 25, 2026

---

## 1. CURRENT STATE OF THE FIELD

### Overview
The field has matured rapidly since 2023. The core question — can synthetic data replace
human-annotated corpora for training small reasoning models under low compute — now has a
nuanced answer: **yes, if and only if data quality, diversity, and grounding mechanisms
are carefully controlled.**

Three major paradigm shifts define 2024–2026:

**A. Trace Distillation as SFT Bootstrap**
Distilling reasoning traces from larger teacher models (GPT-4o, DeepSeek-R1, Llama-3.1-70B)
into smaller student models (1B–8B) has become the dominant low-compute strategy.
DeepSeek-R1's cold-start approach using ~800k verified synthetic traces demonstrated that
high-quality SFT data dramatically outperforms pure RL for small models in terms of
compute efficiency. SYNTHETIC-1 (Prime Intellect, Feb 2025) scaled this to 2M verified
reasoning chains, and SYNTHETIC-2 (June 2025) added difficulty-annotated RL tasks with
pass@k filtering across multiple smaller models.

**B. Self-Play / RLVR Without External Data**
"Absolute Zero" (Zhao et al., May 2025, arXiv:2505.03335) introduced a paradigm
(Absolute Zero Reasoner, AZR) where a single model proposes tasks to maximize its own
learning progress, solves them, and uses a code executor as an external verifier.
AZR achieves SOTA on coding and mathematical reasoning benchmarks without any external
human-curated data. This is publishable at scale even for sub-7B models.

**C. Curriculum Learning Revival**
Curriculum learning for small code LMs (ACL SRW 2024, arXiv:2407.10194) showed that a
well-designed difficulty schedule significantly improves code execution accuracy for
1M-parameter GPT models. Effect is strong for execution tasks, weaker for completion —
indicating task-type specificity matters.

### Key Numbers at a Glance
| Method               | Model Size  | Data Cost      | Compute       | Benchmark Gain |
|----------------------|-------------|----------------|---------------|----------------|
| Trace Distillation   | 1B–8B       | ~Zero labeling | 1–4 A100 days | +10–30% vs baseline |
| AZR Self-Play        | 7B–32B      | Zero external  | Moderate RL   | SOTA on math/code |
| Curriculum Learning  | 1M–7B       | Reuse existing | Low           | +5–15% code exec |
| Agent Distillation   | 0.5B–3B     | Teacher-gen    | Very low      | Matches next-tier |

---

## 2. WHAT ALREADY FAILED IN LITERATURE

### 2.1 Model Collapse from Pure Synthetic Iteration
The most well-documented failure mode: training models iteratively on their own synthetic
outputs without injecting real data causes progressive distribution collapse. Shumailov
et al. (Nature, 2024) formally proved this. Key result: even 1-in-1000 synthetic samples
under a "replace" paradigm (discard old data each iteration) triggers collapse. Larger
models amplify the effect below certain interpolation thresholds — scaling does NOT save
you. The failure begins as silent tail-erasure (rare patterns vanish), then manifests as
stylistic homogenization and brittleness.

**Papers confirming this:**
- "Model Collapse" — Nature (2024): formal proof of collapse under recursive synthetic training
- "A Tale of Tails" — (ICLR/ACL proceedings): tail distribution collapse quantified
- "Strong Model Collapse" — OpenReview (2025): larger models collapse harder in replace paradigm
- "How Bad Is Training on Synthetic Data?" — statistical analysis, ICLR 2025:
  mixing real+synthetic below a threshold threshold avoids collapse; pure synthetic does not.

### 2.2 Self-Play Without Learnable Information Growth
Self-play loops that don't increase learnable information across iterations plateau and
collapse. From arXiv:2603.02218 (Feb 2026): after multiple RL self-play iterations,
learnable information fluctuates rather than grows, Solver capability degrades, and
Proposer generates repetitive patterns. Multi-reward RL alone without a verifiable
external signal fails to sustain evolution.

### 2.3 Naive Trace Distillation: Style Over Substance
OpenReview (2025): "Style over Substance: Distilled Language Models Reason Via Surface
Patterns." Models distilled on synthetic reasoning traces match performance of real-trace
distilled models — but astonishingly, performance even increases when synthetic traces
are altered to lead to WRONG answers. This reveals that small models learning from
distilled traces are largely picking up stylistic formatting patterns, NOT actual
reasoning ability transfer. This is a major negative result: if you distill without
verification, you may be training style mimicry, not reasoning.

### 2.4 Curriculum Learning Is Task-Specific
The ACL SRW 2024 paper (arXiv:2407.10194) explicitly showed curriculum learning
significantly helps code EXECUTION but NOT code completion. Generalization of curriculum
benefits across task types is not guaranteed. Ill-designed curricula (wrong difficulty
metric) can hurt more than random ordering.

### 2.5 Synthetic Diversity Without Human Anchoring
arXiv:2511.01490: Low source diversity in synthetic data increases adversarial robustness
risk for small models. High diversity with large models increases harm risk. Neither end
is safe without human-authored anchoring data. Models also develop self-preference bias —
preferring synthetic-style outputs — which can only be eliminated via human fine-tuning data.

---

## 3. WHAT REMAINS PROMISING

### 3.1 Verified Self-Play with External Grounding Signal (HIGH PROMISE)
AZR (Absolute Zero Reasoner) is the most important result: a model can propose tasks,
solve them, and use a CODE EXECUTOR as an objective external verifier. This sidesteps
model collapse because the verifier is not the model itself. Critically applicable to
math, code, and any domain with programmatically checkable correctness. Scales across
model sizes. Publishable even at 1B–3B scale with appropriate benchmarks.

**Why it works:** The verifier provides learnable information that is external and
objective — bypassing the recursive contamination problem entirely.

### 3.2 Student-Adaptive Trace Generation (HIGH PROMISE)
"Reverse Speculative Decoding" (OpenReview, Feb 2026, arXiv via ICLR): teacher proposes
tokens, student ACCEPTS/REJECTS based on its own probability distribution. This generates
student-friendly reasoning traces. Applied to Qwen3-0.6B: standard distillation
DEGRADED performance by 20.5%; RSD-generated traces IMPROVED it by 4.9%. This is
directly publishable and computationally very cheap.

### 3.3 Difficulty-Annotated RL Data + Pass@k Filtering (HIGH PROMISE)
SYNTHETIC-2 (Prime Intellect, June 2025): annotate RL tasks with pass@k rates from
multiple smaller models. Tasks too easy (high pass@k) or too hard (zero pass@k) are
filtered out. This creates a "Goldilocks zone" of curriculum difficulty — consistent with
classical curriculum learning theory but now applied at scale to synthetic RL data.
Computationally cheap: uses existing small models as annotators.

### 3.4 Agent Distillation: Tool-Using Behavior Transfer (PROMISING)
HuggingFace/arXiv:2505.17612: distilling not just reasoning but full tool-use behavior
from LLM agents into 0.5B–3B models. Models with retrieval+code tools at 0.5B match
CoT-distilled models of the next size tier. Low cost, high leverage for grounding tasks.

### 3.5 CoT-Self-Instruct: Answer-Consistency Filtering (PROMISING)
arXiv:2507.23751: pipeline where LLM plans and reasons to generate new instructions from
seeds, then filters by answer consistency (verifiable tasks) or Reward-Informed Pruning
(non-verifiable). Outperforms curated datasets like s1k on challenging benchmarks.
Fully automatable; the seed corpus is the only human artifact required.

### 3.6 Annotation-Free Multimodal KG Construction (EMERGING)
VaLiK (arXiv:2503.12972): zero-shot, annotation-free multimodal knowledge graph
construction from images + text. Eliminates manual annotation entirely for MMKG.
Knowledge distillation then compresses the KG representation. Directly applicable to
building grounded synthetic multimodal training data at near-zero cost.

---

## 4. DATASETS THAT CAN BE AUTO-GENERATED CHEAPLY

### 4.1 Code Execution Traces
**Source:** Python/JS scripts with known outputs
**Method:** Generate programs procedurally, execute, record I/O pairs as (question, trace, answer)
**Cost:** CPU-only, essentially free
**Use:** Math reasoning, algorithmic reasoning, curriculum difficulty scoring via test pass rate
**Tool:** AZR framework (open source: github.com/andrewzh112/absolute-zero-reasoner)

### 4.2 Math Chain-of-Thought via Symbolic Solvers
**Source:** SymPy, WolframAlpha API (free tier), Lean4/Isabelle for proof generation
**Method:** Generate problem templates, solve symbolically, format as CoT traces
**Cost:** Near-zero; fully local with SymPy
**Use:** Distillation cold-start data, RL verifiable rewards
**Dataset Examples:** SYNTHETIC-1 (2M traces), SYNTHETIC-2 (difficulty-annotated)

### 4.3 Grounding Annotations via Vision-Language Models
**Source:** Existing image datasets (COCO, CC3M subsets already open)
**Method:** Run open VLMs (LLaVA-1.6, InternVL2-2B) to generate bounding box
             descriptions, scene graphs, and QA pairs automatically
**Cost:** Single GPU (e.g., RTX 4070 laptop), ~hours per 10k images
**Use:** Synthetic grounding training data for small multimodal models

### 4.4 Instruction Pairs via Self-Instruct + Consistency Filter
**Source:** Seed set of 175–1000 human-written instructions
**Method:** CoT-Self-Instruct pipeline (arXiv:2507.23751): LLM generates new tasks from
             seeds, answer-consistency filtering removes hallucinated outputs
**Cost:** API calls to teacher LLM (can use local Llama-3.1-8B or Qwen-2.5-7B)
**Use:** Instruction tuning, reasoning SFT for small models

### 4.5 Procedurally Generated Spatial/Logical Tasks
**Source:** Rule-based generators (grid worlds, constraint puzzles, spatial QA)
**Method:** Python generators with deterministic solvers; infinite diversity possible
**Cost:** CPU-only, zero LLM inference cost
**Use:** Grounding evaluation, compositional reasoning, curriculum progression tasks

### 4.6 Synthetic Clinical/Domain NER via 70B Teacher
**Source:** Publicly available clinical templates or domain text skeletons
**Method:** Llama-3.1-70B generates QA+supporting evidence; smaller model fine-tuned
**Cost:** Low RunPod cost for teacher inference; student trains locally
**Use:** Domain-specific reasoning distillation (replicates Nature paper, publishable)

---

## 5. FIVE CANDIDATE PAPERS

### Paper 1: Absolute Zero Reasoner (AZR)
**Citation:** Andrew Zhao et al., "Absolute Zero: Reinforced Self-play Reasoning with Zero
              Data," arXiv:2505.03335, May 2025
**Why:** Defines the new frontier of zero-external-data self-improving reasoning models.
         Code executor as verifier eliminates model collapse risk. Achieves SOTA on coding
         and math without any human data. NeurIPS 2025 poster.
**Relevance:** Direct template for low-compute self-play curriculum system.

### Paper 2: Reverse Speculative Decoding (RSD) for Student-Friendly Traces
**Citation:** Jaehoon Kim et al., "In Their Own Words: Reasoning Traces Tailored for Small
              Models," OpenReview ICLR 2026
**Why:** Proves that teacher-to-student distillation must adapt to student capacity.
         Standard traces HURT 0.6B models; RSD-adapted traces help. Directly actionable.
**Relevance:** Combines trace distillation + curriculum difficulty matching.

### Paper 3: Style over Substance in Trace Distillation
**Citation:** Anonymous, "Style over Substance: Distilled Language Models Reason Via
              Surface-Level Patterns," OpenReview, 2025
**Why:** Critical negative result. Shows distilled reasoning is largely stylistic.
         Sets the right null hypothesis for any distillation experiment.
**Relevance:** Motivates verified-only distillation; warns against unverified synthetic traces.

### Paper 4: Self-Play Only Evolves with Learnable Information Growth
**Citation:** Anonymous, "Self-Play Only Evolves When Self-Synthetic Pipeline Ensures
              Learnable Information," arXiv:2603.02218, Feb 2026
**Why:** Formally characterizes why self-play stagnates. Proposes triadic Proposer/Solver/
         Verifier framework. Directly addresses the failure mode of naive self-play.
**Relevance:** Design blueprint for sustainable self-evolving curriculum.

### Paper 5: Agent Distillation into Sub-3B Models
**Citation:** Anonymous, "Distilling LLM Agent into Small Models with Retrieval and Code
              Tools," arXiv:2505.17612, May 2025
**Why:** Shows 0.5B model with tools matches 1.5B CoT-distilled without tools.
         Tool use is a force-multiplier; distillation transfers tool-using behavior.
**Relevance:** Directly applicable to grounding-augmented small model research.

---

## 6. BEST OVERLAP WITH GROUNDING / MEMORY / HYBRIDS

### 6.1 Grounding × Synthetic Data
**Strongest overlap:** Verified self-play using code executors (AZR) is a form of
programmatic grounding — the executor is the "world" the model must correctly reason
about. Extend this to spatial grounding: procedurally generated map/grid tasks with
verifiable answers are a direct analog. VaLiK's annotation-free MMKG also auto-generates
grounded visual-semantic relationships.

**Research gap (publishable):** Apply AZR-style self-play to spatial/visual grounding
tasks where a simulator (PyBullet, MiniGrid) serves as the external verifier instead
of a code executor. No human annotation required; infinite task diversity.

### 6.2 Memory × Synthetic Curriculum
**Strongest overlap:** Curriculum learning requires tracking what the model knows (i.e.,
a form of model memory). Difficulty-annotated RL data (SYNTHETIC-2 pass@k approach) is
effectively a curriculum that uses the model's current capability as a memory state.

**Research gap (publishable):** Parametric memory modules (e.g., external key-value
stores) updated via synthetic curriculum steps, where retrieval accuracy gates which
curriculum stage is active. Cheap to implement on Qwen-2.5-1.5B + FAISS.

### 6.3 Hybrid: Retrieval-Augmented Synthetic Distillation
Agent Distillation (arXiv:2505.17612) already shows tool-using behavior transfer.
The retrieval component in that framework is essentially a memory subsystem. Combining
RSD-style student-adaptive trace generation with retrieval-augmented teacher traces
is an underexplored hybrid:
- Teacher generates traces with explicit retrieval steps
- Student learns when to retrieve vs. reason internally
- Synthetic corpus is auto-generated from Wikipedia + verifiable QA templates

**Cost:** 1x 8GB GPU for student training; teacher can run via API or RunPod.

### 6.4 Multimodal Grounding × Self-Play
Self-play through computational runtimes for chart reasoning (ACL Findings 2025,
arXiv:2025) showed VLMs can iteratively improve multimodal reasoning via self-play.
Extend to:
- Table/chart QA with programmatic verifiers (pandas-based answer checking)
- Diagram spatial reasoning with procedurally generated SVG scenes
- Both are auto-generatable at near-zero cost

---

## 7. FINAL GO / NO-GO SUGGESTIONS

### ✅ GO: AZR-Style Verified Self-Play at 1B–3B Scale
**Verdict: STRONG GO**
Absolute Zero paradigm is proven, open-source, and scales down. A publishable result
requires: (a) novel verifier domain (not just code/math — e.g., symbolic logic, spatial
planning), (b) sub-3B model size (underexplored in original paper), (c) ablation showing
learnable-information growth metric across iterations. Compute: feasible on RTX 4070 +
RunPod A100 spot instances. Timeline: 4–8 weeks to publishable negative/positive result.

### ✅ GO: RSD (Reverse Speculative Decoding) for Sub-1B Models
**Verdict: GO**
The ICLR 2026 paper only tests Qwen3-0.6B on one setting. Extension to Qwen2.5-0.5B,
Phi-3.5-mini, and SmolLM2-135M across multiple reasoning domains is immediately
publishable as an empirical analysis paper. Very low compute: no training beyond SFT on
pre-generated traces. Completely feasible on laptop GPU.

### ✅ GO: Difficulty-Filtered Curriculum for Multimodal Grounding
**Verdict: GO**
Pass@k difficulty annotation applied to multimodal grounding tasks (not just text
reasoning) is unexplored. Procedure: generate synthetic chart/diagram QA, annotate
difficulty with PaliGemma-3B pass@k, train VLM with curriculum order. Low-cost,
novel domain, clear evaluation protocol.

### ⚠️ CONDITIONAL GO: CoT-Self-Instruct for Domain-Specific SLMs
**Verdict: CONDITIONAL**
Strong results exist for general reasoning. For publishability, must target a domain with
clear evaluation benchmarks (medical, legal, code security) AND use a fully local teacher
(no GPT-4 dependency). If using Llama-3.1-8B as teacher and evaluating on domain-specific
held-out test sets, publishable. Risk: domain shift from synthetic seeds to real eval data.

### ⛔ NO-GO: Iterative Self-Training Without External Verifier
**Verdict: HARD NO-GO**
Pure iterative self-training (model trains on its own outputs repeatedly) is formally
proven to cause distribution collapse. The literature is saturated with negative results.
Any paper in this space without a novel mitigation mechanism will be rejected. Do not
pursue unless the contribution IS the mitigation mechanism itself.

### ⛔ NO-GO: Naive Trace Distillation Without Verification
**Verdict: NO-GO**
"Style over Substance" (2025) kills this direction unless the paper's contribution is
explicitly about what IS being transferred (i.e., the negative result IS the paper).
Training on unverified synthetic reasoning traces risks training stylistic mimicry, not
reasoning. Only proceed with answer-verified or execution-verified traces.

### ⚠️ CONDITIONAL GO: Synthetic Multimodal Annotation Pipelines
**Verdict: CONDITIONAL**
VaLiK-style annotation-free MMKG construction is promising but requires strong evaluation
methodology to be publishable (must compare to human-annotated baselines on held-out tasks).
If grounding accuracy on VQA/GQA benchmarks improves measurably, publishable.
Risk: VLM-generated annotations inherit VLM hallucinations — must include hallucination
rate measurement as part of paper contribution.

---

## SUMMARY TABLE

| Direction                          | Publishability | Compute Cost | Risk Level | Verdict        |
|------------------------------------|---------------|--------------|------------|----------------|
| AZR Self-Play (novel domain)       | High          | Medium       | Low        | ✅ STRONG GO   |
| RSD Sub-1B Distillation            | High          | Very Low     | Low        | ✅ GO          |
| Difficulty-Filtered MM Curriculum  | Medium-High   | Low          | Medium     | ✅ GO          |
| CoT-Self-Instruct (domain-spec)    | Medium        | Low          | Medium     | ⚠️ CONDITIONAL |
| Iterative Self-Training (no verif) | Low           | Low          | High       | ⛔ NO-GO       |
| Naive Trace Distillation           | Low           | Low          | High       | ⛔ NO-GO       |
| Annotation-Free MMKG               | Medium        | Medium       | Medium     | ⚠️ CONDITIONAL |

---

## KEY REFERENCES

1. Zhao et al. (2025). "Absolute Zero: Reinforced Self-play Reasoning with Zero Data."
   arXiv:2505.03335. https://arxiv.org/abs/2505.03335

2. Kim et al. (2026). "In Their Own Words: Reasoning Traces Tailored for Small Models."
   OpenReview ICLR 2026. https://openreview.net/forum?id=Ntr3O8mpEX

3. Anonymous (2025). "Style over Substance: Distilled Language Models Reason Via Surface
   Patterns." OpenReview. https://openreview.net/forum?id=5wAfbEs34A

4. Anonymous (2026). "Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable
   Information." arXiv:2603.02218. https://arxiv.org/html/2603.02218v1

5. Anonymous (2025). "Distilling LLM Agent into Small Models with Retrieval and Code Tools."
   arXiv:2505.17612. https://huggingface.co/papers/2505.17612

6. Shumailov et al. (2024). "Model Collapse." Nature. (Model collapse formal proof.)

7. ACL SRW (2024). "Curriculum Learning for Small Code Language Models."
   arXiv:2407.10194. https://arxiv.org/abs/2407.10194

8. Prime Intellect (2025). "SYNTHETIC-2: Verified Reasoning Traces."
   https://www.primeintellect.ai/blog/synthetic-2

9. VaLiK (2025). "Aligning Vision to Language: Annotation-Free Multimodal Reasoning."
   arXiv:2503.12972. https://arxiv.org/html/2503.12972v2

10. Nature Medicine (2025). "Synthetic Data Distillation Enables Extraction of Clinical Knowledge."
    https://www.nature.com/articles/s41746-025-01681-4

---
*Generated by Agent 19 — Synthetic Data and Curriculum Designer*
*Part of the AlienX Small Model Research Pipeline*
