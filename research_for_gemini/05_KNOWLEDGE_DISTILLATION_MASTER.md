# Knowledge Distillation Master Guide: Cross-Architecture, Reasoning, and Dark Knowledge

> **Purpose**: Complete technical guide to knowledge distillation for language models, with focus on cross-architecture distillation (Transformer → BDH-style). Designed for Gemini deep research.

---

## PART 1: FOUNDATIONAL CONCEPTS

### 1.1 What is Knowledge Distillation?

Knowledge distillation (KD) transfers knowledge from a large, capable **teacher** model to a smaller, efficient **student** model. The student learns not just from raw data, but from the teacher's "thoughts" — its probability distributions over outputs.

### 1.2 The Original Framework (Hinton et al., 2015)

```
L_KD = α * CE(student, hard_labels) + (1-α) * KL(softmax(teacher/T), softmax(student/T))
```

Where:
- `T` = temperature (controls softness of distribution)
- `KL` = Kullback-Leibler divergence
- `α` = balance between hard labels and soft labels

### 1.3 Dark Knowledge

**Dark knowledge** is the information in the teacher's soft probability distribution over *incorrect* classes/tokens. It captures:

1. **Inter-class relationships:** Similar classes get similar probabilities (e.g., "cat" and "dog" both get moderate probability when the answer is "cat").
2. **Semantic similarity structure:** The teacher's internal geometry of the output space.
3. **Uncertainty calibration:** Which alternatives the teacher considers plausible vs. impossible.
4. **Compositional structure:** How the teacher decomposes complex predictions into constituent parts.
5. **Inductive bias:** The teacher's learned priors about which outputs are "close" to each other.

**Why it matters for LLMs:** The teacher's distribution over the vocabulary at each token position encodes syntactic constraints, semantic relationships, and world knowledge that hard labels (single token) completely discard.

**Limitations:** KL divergence gradients scale with teacher logit magnitude, so dark knowledge in low-probability channels gets weak gradient signal. Ranking losses and symmetric divergences (JS, TVD) help.

---

## PART 2: DISTILLATION LOSS FUNCTIONS

### 2.1 Complete Comparison

| Loss | Formula | Property | When to Use |
|---|---|---|---|
| **Forward KL** | KL[p_teacher || q_student] | Mode-covering, asymmetric | Classification, when student capacity is high |
| **Reverse KL** | KL[q_student || p_teacher] | Mode-seeking, asymmetric | Generative LLMs, prevents learning long-tail noise |
| **JS Divergence** | 0.5*KL[p||m] + 0.5*KL[q||m] where m=(p+q)/2 | Symmetric, bounded | Balances mode-averaging vs. collapsing |
| **TVD** | 0.5 * Σ|p(x) - q(x)| | Symmetric, bounded, metric | When distributions have little overlap |
| **Sinkhorn** | Regularized OT distance | Geometry-aware, handles non-overlap | Feature distillation, intermediate layers |
| **Wasserstein** | inf E[d(x,y)] | Captures manifold geometry | Cross-category comparison, feature distillation |
| **Kendall's tau** | Rank correlation | Scale-invariant, ordinal | Auxiliary to KL, helps low-prob channels |
| **alpha-divergence** | D_α(p||q) | Sparse support (α=1.5) | Pre-training, SFT, distillation (general purpose) |
| **DKL/IKL** | wMSE + soft CE | Decoupled | Breaking KL asymmetry, class-wise global info |

### 2.2 Key Insights

**Forward KL (standard KD):** Mode-covering. Student tries to cover all modes of the teacher's distribution. Good when student has high capacity. Can cause student to learn long-tail noise.

**Reverse KL (MiniLLM):** Mode-seeking. Student focuses on major modes, avoids void regions. Better for truthfulness/reliability. Preferred for generative LLMs.

**Symmetric losses (JS, TVD):** Balance the mode-averaging of forward KL and mode-collapse of reverse KL. They consistently outperform asymmetric losses in text generation.

**Kendall's tau ranking loss:** Gradients are invariant to channel scale. Helps low-probability channels that KL ignores. Works as a plug-and-play auxiliary loss.

**Alpha-divergence (α=1.5):** Sparse support. Performs well across pre-training, SFT, and distillation. General-purpose choice.

---

## PART 3: LOGITS DISTILLATION TECHNIQUES

### 3.1 Temperature Scaling

- **Temperature T:** Divides logits before softmax. T in [2,8] optimal for most tasks.
- **Higher T:** Exposes more dark knowledge but adds noise.
- **tau² scaling:** Compensates for gradient magnitude reduction from temperature scaling.

### 3.2 Top-K Distillation

- Only distill on top-K teacher logits to reduce compute.
- Sparse distributions work well with polynomial approximation + quantization (DistillKit).
- **Key insight:** For vocab=248K, computing KL on all tokens is wasteful. Top-512 captures 99.8% of the signal.

### 3.3 Sub-Atomic Distillation (BDH-specific)

- Used in JARVIS production model.
- Select top-512 teacher logits via `torch.topk`.
- Gather corresponding student logits via `torch.gather`.
- Compute KL only on these 512 dimensions.
- **Result:** Drastic VRAM reduction, purer training signal.

---

## PART 4: HIDDEN STATE / INTERMEDIATE LAYER DISTILLATION

### 4.1 Techniques

**Cosine Loss (DistilBERT):**
- Requires matching dimensions between student and teacher.
- Limits compression to ~2×.
- Simple, stable.

**CKA (Centered Kernel Alignment) — ICLR 2025:**
- Matches hidden states of **different** dimensionalities.
- Enables >2× compression.
- Works from random initialization (no teacher weight transfer needed).
- **Breakthrough:** Can distill a 7B teacher into a 1B student via hidden state matching.

**Linear Projection (TinyBERT):**
- Learnable projector maps student to teacher dimension.
- Still SOTA for same-architecture distillation.

**MSE on Hidden States:**
- Simple, stable.
- Used in iterative layer-wise distillation.

### 4.2 HLD at Scale (arXiv:2605.11513)

- First large-scale HLD study on decoder-only pretraining.
- Shows systematic perplexity gain but inconsistent downstream improvement.
- Hyperparameter sensitivity is a concern.
- **Key finding:** HLD helps perplexity but doesn't always translate to better task performance.

---

## PART 5: CROSS-ARCHITECTURE DISTILLATION (Transformer → SSM/Linear Attention)

### 5.1 MOHAWK 3-Phase Approach (Most Successful)

**Phase 1: Matrix Orientation**
- Align SSM mixing matrix with teacher attention matrix.
- Mathematical bridge between attention and SSM representations.

**Phase 2: Hidden-State Alignment**
- Match per-layer hidden representations.
- Use CKA or learned projections for dimension matching.

**Phase 3: Weight Transfer + End-to-End KD**
- Transfer remaining weights (embeddings, norms, MLP, LM head).
- Fine-tune with logit KD.

**Result:** Phi-Mamba from Phi-1.5 in only 3B tokens (vs. trillions for pretraining from scratch).

### 5.2 Attention to Mamba 2-Stage Approach (Pure SSM, No Attention Blocks)

**Stage 1:** Distill Softmax Attention → Linear Attention via Hedgehog (kernel trick).

**Stage 2:** Use Linear Attention weights as Mamba initialization, fine-tune.

### 5.3 CAB (Cross-architecture via Attention Bridge)

- MLP-based bridge maps (Q,K) from Transformer to (B,C) in Mamba.
- Flexible cross-layer alignment strategies.
- Avoids explicit attention matrix computation.

### 5.4 Key Architectural Insights

1. **Most knowledge in Transformers is in the FFN/channel mixer, not attention.**
2. **Mamba can perform both sequence and channel mixing simultaneously.**
3. **Hybrid models (retaining some attention layers) perform best.**
4. **Phi-Mamba retains 4 attention layers out of Phi-1.5's full set.**
5. **Only 3B-5B tokens needed vs. trillions for pretraining from scratch.**

### 5.5 General Cross-Architecture Principles

- **Progressive granularity:** Match fine-to-coarse (mixing matrices → hidden states → outputs).
- **Principled initialization:** Don't random-init; bridge architectures mathematically.
- **Weight transfer:** Transfer shared components (embeddings, norms, LM head).
- **Dimension matching:** Use CKA or learned projections for hidden state alignment.

---

## PART 6: REASONING DISTILLATION

### 6.1 Critical Findings

**Small Model Learnability Gap (arXiv:2502.12143):**
- Models ≤3B do NOT benefit from long CoT.
- They perform better with shorter, simpler reasoning chains matching their intrinsic capacity.

**Non-monotonic granularity (arXiv:2502.18001):**
- Stronger small models benefit from finer-grained reasoning.
- Weaker models need simpler CoT supervision.

**Stronger teacher ≠ better student:**
- Diversity and complexity in CoT supervision can outweigh accuracy alone.

### 6.2 Effective Recipes

**Phi-4-Mini-Reasoning (3.8B beats 7B/8B R1-distilled):**
1. Stage 1: Large-scale mid-training on diverse distilled long-CoT data (iterative)
2. Stage 2: SFT on high-quality compact long-CoT data
3. Stage 3: Rollout Verifiable Reward (reuses wrong LLM samples)
4. Stage 4: RL with verifiable reward

**SIKeD (Self-guided Iterative KD):**
- Teach multiple strategies (CoT, L2M, PoT).
- Student uses on-policy outputs to choose best strategy.
- Iteratively mix LLM data with self-generated data.
- Adaptive: add LLM data only for student errors.

**Mix Distillation:**
- Combine long + short CoT examples.
- Or reasoning from both larger and smaller models.

**Coarse-to-fine (TinyThinker):**
- 3-stage reasoning (recall → analyze → summarize) + self-reflection with DPO.

**Post-thinking (answer before rationale):**
- More robust to rationale errors.
- Naturally focuses on hard samples.

### 6.3 Data-Centric Best Practices (DC-CoT benchmark)

- **Structured logic tasks (math, code):** Reverse Thinking + Teacher Correctness filtering
- **Open-ended linguistic tasks:** Answer Augmentation + LLM-as-a-Judge
- **Agentic/visual tasks:** LLM-as-a-Judge filtering required

---

## PART 7: ON-POLICY vs OFF-POLICY DISTILLATION

### 7.1 Off-Policy (SeqKD, SFT)
- Train on pre-generated teacher outputs.
- Cheap but distribution mismatch.
- Student sees only teacher's distribution, not its own.

### 7.2 On-Policy (MiniLLM, GKD)
- Student generates its own sequences.
- Teacher provides token-level supervision.
- Addresses exposure bias but requires live teacher server.

### 7.3 Lightning OPD (Offline On-Policy Distillation)
- Offline on-policy with precomputed teacher log-probs.
- Requires **teacher consistency** (same teacher for SFT + OPD).
- 4× efficiency gain.

### 7.4 Theory
- Online advantage comes from model misspecification, not just on-policy sampling.
- When student can represent teacher, offline SFT is sufficient.

---

## PART 8: CURRICULUM DISTILLATION

### 8.1 Progressive Distillation (ICLR 2025 Oral)
- Use intermediate teacher checkpoints.
- The "helpful noise" during the teacher's phase transition provides an implicit curriculum.
- Only 2 checkpoints needed: one during phase transition + final.

### 8.2 Curriculum Extraction (arXiv:2503.17494)
- Extract curriculum from fully-trained teacher via random projection of hidden representations.
- No checkpoint storage needed.

### 8.3 POCL (arXiv:2506.05695)
- "Progressive overload" — rank samples by student confidence (easy to hard).
- Incrementally introduce with rising temperature.

### 8.4 Pro-KD
- Student trains alongside teacher, following teacher's training footsteps.
- Adaptive decreasing temperature.

---

## PART 9: MULTI-TEACHER DISTILLATION

### 9.1 Current State
- Underexplored for LLMs.
- How to optimally combine signals from multiple teachers of different architectures/capabilities?

### 9.2 Pre-training Distillation Design Space (GLM-4 study)
- Larger students benefit more from distillation.
- Online logits useful but less impactful than offline data.

### 9.3 Safety vs. Utility in Distillation
- Soft-label distillation compromises safety up to 50pp more than SFT.
- No effective mitigation yet.

---

## PART 10: FIRST PRINCIPLES OF KNOWLEDGE TRANSFER

### Principle 1: Information Bottleneck
The student's limited capacity is a bottleneck. Distillation compresses the teacher's knowledge through this bottleneck more efficiently than raw data because the teacher's outputs are already a compressed, structured representation.

### Principle 2: Distribution Matching
KD works by matching the student's output distribution to the teacher's. The choice of divergence determines *how* the mismatch is penalized (mode-covering vs. mode-seeking).

### Principle 3: Inductive Bias Transfer
The teacher's internal representations encode learned inductive biases. Feature/hidden-state distillation transfers these biases, not just outputs.

### Principle 4: Curriculum as Optimization Aid
Easier teachers/examples first reduces the optimization landscape's complexity, helping the student find better minima.

### Principle 5: On-Policy Correction
The student's distribution at inference differs from training data distribution. On-policy distillation corrects this mismatch by training on the student's own generated sequences.

### Principle 6: Dark Knowledge as Relational Learning
Soft labels teach relationships between outputs, not just individual outputs. This is the core mechanism behind KD's generalization improvement.

---

## PART 11: OPEN RESEARCH QUESTIONS FOR BDH-STYLE ARCHITECTURES

1. **SSM/Linear Attention distillation:** What is the optimal recipe for distilling Transformer knowledge into BDH's sequence mixing mechanism? Should we use MOHAWK's 3-phase approach or the 2-stage Hedgehog→Mamba bridge?

2. **Hidden state compression:** Can CKA-based hidden state matching enable BDH to achieve higher compression ratios than cosine/projection-based methods while maintaining reasoning capabilities?

3. **Reasoning in compact models:** Given the Small Model Learnability Gap, what is the optimal CoT granularity for BDH's parameter budget? Should we use Mix Distillation or SIKeD's iterative strategy selection?

4. **Loss function selection:** For BDH's specific architecture, which divergence (reverse KL, JS, TVD, Sinkhorn, alpha=1.5) provides the best utility-safety tradeoff?

5. **Progressive distillation for BDH:** Can we extract an implicit curriculum from a fully-trained teacher via random projection, avoiding checkpoint storage while accelerating BDH training?

6. **Multi-teacher ensembles:** If BDH benefits from diverse knowledge sources, how to optimally combine logits/hidden states from multiple teachers (e.g., reasoning-specialized + general-purpose)?

7. **Online vs offline for BDH:** Given BDH's efficiency focus, is Lightning OPD's offline on-policy approach with teacher consistency the right choice, or does BDH benefit enough from live teacher interaction to justify the cost?

8. **Safety-preserving distillation:** How to distill into BDH without the 50pp safety degradation observed with soft-label methods? Is hard-label SFT with teacher-generated data the safer path?

9. **Cross-architecture weight transfer:** If BDH shares any components with Transformers (e.g., embedding layer, LM head), what is the optimal weight transfer strategy?

10. **Iterative self-improvement:** Can BDH use self-data distillation (like the pruning recovery method) for iterative self-improvement without external teachers?

---

## PART 12: PAPERS TO READ

### Foundational
1. **Hinton et al. (2015) - Distilling Knowledge** — arXiv:1503.02531
2. **Sanh et al. (2019) - DistilBERT** — arXiv:1910.01108
3. **Kim & Rush (2016) - SeqKD** — arXiv:1606.07947

### MiniLLM & Reverse KL
4. **MiniLLM (Gu et al., 2024) ICLR** — arXiv:2306.08543
5. **f-DISTILL (Wen et al., 2023) ACL** — arXiv:2307.15190
6. **SinKD (2024)** — arXiv:2402.17110
7. **GKD (Agarwal et al., 2024)** — arXiv:2402.16138

### Hidden State Distillation
8. **FitNets (Romero et al., 2015)** — arXiv:1412.6550
9. **CKA Hidden State Matching (ICLR 2025)** — ICLR 2025 Proceedings
10. **HLD for LLM Pre-Training (2026)** — arXiv:2605.11513
11. **Iterative Layer-wise Distillation (2025)** — arXiv:2511.05085

### Cross-Architecture (SSM)
12. **MOHAWK (Bick et al., 2024) NeurIPS** — arXiv:2408.10189
13. **Attention to Mamba (Moudgil et al., 2026)** — arXiv:2604.14191
14. **CAB (Cross-architecture via Attention Bridge)** — OpenReview
15. **State Space Duality (Dao & Gu, 2024)** — arXiv:2405.21060

### Reasoning Distillation
16. **Small Model Learnability Gap (2025)** — arXiv:2502.12143
17. **Key Factors for CoT Distillation (2025)** — arXiv:2502.18001
18. **SIKeD (Adarsh et al., ACL 2025)** — arXiv:2410.18574
19. **Phi-4-Mini-Reasoning (2025)** — arXiv:2504.21233
20. **TinyThinker (2024)** — arXiv:2412.08024
21. **PRADA (2025)** — ACL 2025
22. **Stepwise Attention CoT (EMNLP 2025)** — ACL 2025
23. **CRV + CogPO (2025)** — arXiv:2504.09802
24. **DC-CoT Benchmark (2025)** — arXiv:2505.18759

### Self-Distillation & Iterative
25. **Self-Data Distillation (Thangarasa et al., MLSys 2025)** — arXiv:2410.09982
26. **OPSD (On-Policy Self-Distillation, ICLR 2026)** — OpenReview
27. **DynSDPB (2024)** — arXiv:2411.16991

### Curriculum & Progressive
28. **Progressive Distillation Implicit Curriculum (Panigrahi et al., ICLR 2025 Oral)** — ICLR 2025
29. **Curriculum Extraction (2025)** — arXiv:2503.17494
30. **POCL (2025)** — arXiv:2506.05695
31. **Pro-KD (2021)** — arXiv:2110.08532

### Online vs Offline, Multi-Teacher
32. **Online vs Offline IL Theory (2025)** — OpenReview
33. **Pre-training Distillation Design Space (2024)** — arXiv:2410.16215
34. **Lightning OPD (2026)** — arXiv:2604.13010
35. **Safety vs Utility in Distillation (2025)** — OpenReview
36. **Llama 3.1 405B → 8B/70B (2024)** — arXiv:2410.18588

### Loss Functions Beyond KL
37. **Beyond I-Con (2025)** — arXiv:2509.04734
38. **WKD Wasserstein (2024)** — arXiv:2412.08139
39. **Kendall's tau Ranking Loss (2025)** — arXiv:2409.17823
40. **DKL/IKL (2023)** — arXiv:2305.13948
41. **f-divergence losses for LMs (2025)** — arXiv:2501.18537
