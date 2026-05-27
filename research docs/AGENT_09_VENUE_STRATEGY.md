# AGENT 09: Venue & Strategy Analysis

**Date:** April 25, 2026  
**Analyst:** Agent 9 — Venue & Strategy Analyst  
**Target Timeline:** 8 weeks (completion ~ June 20, 2026)  
**Team Profile:** Single B.Tech student (4th sem), 8GB RTX 4070 laptop, limited paid compute, fast prototyping capability

---

## Executive Summary

Given the 8-week window and single-developer constraint, **no NeurIPS/ICLR main-track submission is realistic before late 2026**. The two viable paths are: (1) an **EMNLP 2026 ARR submission** (~30 days to draft) targeting the efficiency or evaluation track as the Tier-1 swing, and (2) a **NeurIPS/ICML 2026 workshop paper** as the safer fallback. The hardware-software co-design angle (Bit-by-Bit GPU) is too long-cycle for 8 weeks but provides excellent positioning narrative. The highest-probability accepted contribution is a **benchmark or evaluation paper** on low-resource reasoning architectures — light on training compute, heavy on design rigor and analysis.

---

## 1. Venue-by-Venue Expectations Matrix

| Venue | Next Deadline | Acceptance Rate | Fit for This Team | Evidence Required | Realistic in 8 Weeks? |
|---|---|---|---|---|---|
| **NeurIPS 2026 Main Track** | May 4 (abstract) — **9 days away** | ~20-25% | Low | Novel algorithm + strong theory + large-scale experiments | ❌ No — impossible in 9 days |
| **NeurIPS 2026 Evaluations & Datasets** | May 4 (abstract) — **9 days away** | ~25-30% | Medium | Rigorous evaluation protocol or dataset + community need | ❌ No — deadline too soon |
| **NeurIPS 2026 Position Papers** | May 4 (abstract) — **9 days away** | ~20-25% | Medium-High | Bold, well-argued vision + technical grounding | ❌ No — deadline too soon |
| **NeurIPS 2026 Competition Track** | May 15 — **20 days away** | ~30-40% | Medium | Competition design + infrastructure + clear evaluation | ⚠️ Possible if repurposing existing project quickly |
| **NeurIPS 2026 Workshops** | Varies (usually post-main) | ~35-50% | **High** | Preliminary results + clear direction + community relevance | ✅ Yes — best fallback target |
| **ICLR 2027 Main** | Sep 2025 — **already passed** | ~25-30% | Low | Novel deep learning theory + strong empirical validation | ❌ No — deadline passed |
| **ICML 2026 Main** | Jan 2026 — **already passed** | ~22-27% | Low | Novel algorithm + large-scale validation + theory | ❌ No — deadline passed |
| **ACL 2026 Main** | Jan 2026 — **already passed** | ~20-25% | Medium | Linguistically motivated NLP contribution | ❌ No — deadline passed |
| **EMNLP 2026 Main (via ARR)** | **May 25, 2026** — ~30 days | ~22-28% | **High** | Empirical NLP methods + solid experiments | ✅ Tight but viable for ARR submission |
| **EMNLP 2026 Workshops** | Post-ARR, ~August | ~35-50% | **High** | Focused NLP topic + preliminary results | ✅ Yes — comfortable target |
| **CVPR/ICCV/ECCV Workshops** | CVPR 2026 passed; ECCV Mar 2026 passed | ~30-45% | Low-Medium | Vision contribution + visual results | ❌ No — vision angle not developed enough |

**Key Insight:** All major 2026 conference main-track deadlines (NeurIPS, ICML, ACL, ICLR, CVPR, ECCV) are either **already passed or occurring within 9-20 days**. The only main-track deadline still realistically reachable is **EMNLP 2026 via ARR (May 25)**, and the most forgiving targets are **workshops at NeurIPS, EMNLP, or ICML 2026** [web:11][web:18][web:39].

---

## 2. What Evidence Each Venue Demands

### Tier-1 Venues (NeurIPS, ICLR, ICML Main Track)

These venues demand **substantial, novel contributions that advance the state of the art** [web:39]. Reviewers expect:

- **Novel algorithmic contribution**: A new method, architecture, or training paradigm with clear differentiation from existing work
- **Theoretical grounding**: At minimum, convergence analysis, complexity bounds, or principled justification; preferably proofs or theoretical insights
- **Large-scale empirical validation**: Results on standard benchmarks with statistical significance, ablation studies, and comparisons against strong baselines
- **Reproducibility**: Code release, clear hyperparameters, deterministic seeds
- **Community relevance**: The problem must matter to a significant subset of the ML community

**Why this team cannot satisfy in 8 weeks:** Single developer, 8GB VRAM, no compute cluster access. Training a new architecture from scratch with proper baselines would require 2-3 months minimum even with clever optimization [cite:8].

### NeurIPS Evaluations & Datasets Track (Renamed 2026)

This track now explicitly treats **"evaluation as a scientific object of study"** with broadened scope [web:26]. Demands:

- **Rigorous evaluation design**: Clear metrics, controlled conditions, statistical testing
- **Community need**: The benchmark must fill a genuine gap
- **Pilot results**: Evidence that the evaluation protocol works on at least 3-5 representative models
- **Analysis depth**: Beyond "we tested X models" — what does the evaluation reveal about model behavior?

**Why this team could satisfy (if not for deadline):** Low compute requirement — mostly inference and metric design. The Auto-GIT pipeline could generate a "research-to-code" evaluation benchmark. However, the **May 4 deadline makes this impossible** [web:20].

### EMNLP 2026 (via ARR)

EMNLP uses ACL Rolling Review with a **May 25 submission deadline** for the review cycle that feeds into EMNLP commitment [web:18]. Demands:

- **Empirical methods focus**: Must involve NLP tasks, language data, or linguistic analysis
- **Original, unpublished research**: No prior publication of core contribution
- **Solid experimental section**: Even short papers need controlled experiments
- **Positioning in NLP literature**: Clear related work in NLP/CL venues

**ARR advantage:** Papers can be revised and resubmitted to future cycles if rejected, and commitment to EMNLP happens in August after reviews are received [web:18]. This provides a **built-in revision path**.

### NeurIPS/ICML/EMNLP Workshops

Workshop papers are **shorter (4-6 pages)**, evaluated with understanding that work may be preliminary [web:39]. Demands:

- **Focused scope**: Must clearly fit the workshop's CFP theme
- **Clear direction**: Even preliminary work needs a sharp thesis
- **Honest about limitations**: Reviewers expect transparency about what is and isn't done
- **Community fit**: Workshop audience should find the topic relevant

**Acceptance rates: 30-50%**, much more forgiving than main track [web:39].

---

## 3. What This Team Can Realistically Satisfy in 8 Weeks

### Strengths Inventory
- **Fast prototyping**: Uses AI assistants (Claude, Copilot, Kimi, Grok) as force multipliers [cite:5]
- **Multi-language capability**: Python (ML), C/C++ (embedded), Verilog (hardware), JavaScript (tools)
- **Hardware-software integration mindset**: Can design co-optimized systems
- **Local LLM expertise**: Ollama, quantization, INT4/INT8 conversions [cite:6]
- **Agent orchestration experience**: Auto-GIT's 20 press agents + 6 debating agents [cite:5]
- **Networking access**: Connections at Sarvam AI, smallest.ai via GitHub Constellation [cite:3]

### Constraints Inventory
- **Compute**: 8GB RTX 4070 laptop only; no sustained cloud GPU budget
- **Time**: Academic semester commitments limit sustained deep work
- **Team size**: Single developer (no research group for distributed experiments)
- **Baselines**: Cannot train or fine-tune large models from scratch
- **Theory**: No formal theoretical training (proofs, convergence analysis)

### 8-Week Deliverable Reality Check

| Deliverable | Feasibility | Notes |
|---|---|---|
| New architecture trained from scratch | ❌ Impossible | Need 10x+ more compute and 3-6 months |
| Systematic benchmark of 5-8 existing models on constrained reasoning | ✅ High | Inference-only, fits 8GB VRAM with quantized models |
| Agentic pipeline for research-to-code evaluation | ✅ High | Builds on Auto-GIT directly |
| FPGA/GPU simulation with Verilator (Bit-by-Bit) | ⚠️ Medium | 8 weeks enough for proof-of-concept simulation, not tapeout |
| Hybrid SSM-Transformer small-scale experiments (<100M params) | ⚠️ Medium | Possible with careful batch sizing, limited to toy datasets |
| Position paper on hardware-software co-design for edge AI | ✅ High | No experiments needed; strong vision + existing simulation evidence |
| MoE distillation experiments (4 experts → 1 orchestrator) | ⚠️ Medium | Can use existing open-source models; distillation is inference-heavy |

**Critical insight:** In 8 weeks, this team can produce **strong evaluation/analysis papers** or **position papers**, but not **training-heavy algorithm papers**. The compute constraint forces an evaluation-centric strategy, which happens to align perfectly with the NeurIPS ED track philosophy — but the deadline mismatch forces EMNLP or workshops as the practical target [web:26][cite:8].

---

## 4. Top 5 Direction-to-Venue Mappings

### Mapping 1: "Constrained Reasoning Benchmark for Edge-Deployable Architectures"
- **Direction:** Systematic evaluation of SSMs, hybrid SSM-Transformers, and small MoEs on reasoning tasks under 8GB VRAM limits
- **Best Venue Fit:** EMNLP 2026 (ARR) — Efficiency Track or Main
- **Probability of Strong Submission:** **70%**
- **Reviewer Expectations:** Clear task definition, controlled comparison protocol, statistical testing, relevance to efficiency community
- **Tier-1 vs Tier-2 Separator:** Tier-1 includes *new insights* about why certain architectures fail/prevail under constraints; Tier-2 is just a "we tested 8 models" table without analysis depth
- **8-Week Viability:** ✅ High — mostly inference, no training from scratch
- **Evidence Needed:** 3-5 reasoning tasks, 6-10 model variants (quantized), throughput/latency/accuracy tradeoff curves, ablation on sequence length

### Mapping 2: "Auto-GIT: A Multi-Agent Evaluation Framework for Research-to-Code Generation"
- **Direction:** Benchmark measuring how well agentic systems convert ML papers to runnable repositories
- **Best Venue Fit:** EMNLP 2026 (ARR) — Generation Track; or NeurIPS 2026 Workshop on AI for Science/Agents
- **Probability of Strong Submission:** **65%**
- **Reviewer Expectations:** Clear evaluation protocol, human or automatic judgment of code correctness, comparison with baselines (e.g., single-agent approaches)
- **Tier-1 vs Tier-2 Separator:** Tier-1 includes *error taxonomy* (what types of papers/code fail and why); Tier-2 is just accuracy scores without analysis
- **8-Week Viability:** ✅ High — builds directly on existing Auto-GIT system [cite:5]
- **Evidence Needed:** 20-50 paper-code pairs, execution success rate, comparison with GPT-4/Copilot on same task, failure mode analysis

### Mapping 3: "Bit-by-Bit: A Case Study in Hardware-Software Co-Design for Sparse Transformer Acceleration"
- **Direction:** Position paper / systems paper arguing for specialized sparse attention hardware, with simulation evidence
- **Best Venue Fit:** NeurIPS 2026 Workshop on Hardware-Efficient ML; or ICLR Workshop
- **Probability of Strong Submission:** **60%**
- **Reviewer Expectations:** Concrete hardware simulation numbers (Verilator), clear speedup claims, honesty about limitations, connection to real ML workloads
- **Tier-1 vs Tier-2 Separator:** Tier-1 has *end-to-end speedup numbers* on a real model running on the simulated hardware; Tier-2 is just RTL code without workload mapping
- **8-Week Viability:** ⚠️ Medium — Verilog simulation possible, but full accelerator integration is ambitious
- **Evidence Needed:** Verilator simulation of sparse attention unit, cycle-accurate performance vs. CPU baseline, area/power estimates, mapping of at least one small Transformer to the architecture

### Mapping 4: "Synaptic Graph Networks: Graph-Structured Memory for Low-Resource Reasoning"
- **Direction:** Neural-symbolic hybrid using graph memory with SSM backbone for constrained reasoning
- **Best Venue Fit:** ICLR 2027 (if targeting Sep 2026 deadline) or NeurIPS 2026 Workshop on Structured Models/Neuro-Symbolic AI
- **Probability of Strong Submission:** **50%**
- **Reviewer Expectations:** Novel architecture description, proof-of-concept experiments, comparison with pure Transformer and pure SSM baselines
- **Tier-1 vs Tier-2 Separator:** Tier-1 demonstrates *generalization gains* on compositional reasoning tasks; Tier-2 is a toy demonstration on a single dataset
- **8-Week Viability:** ⚠️ Medium — architecture design is fast, but proper baselines need careful experimentation
- **Evidence Needed:** 2-3 compositional reasoning tasks, comparison with Mamba-2, Transformer, and GNN baselines, ablation on graph connectivity

### Mapping 5: "Small-Scale MoE Distillation: Debating Experts Under Compute Constraints"
- **Direction:** Distilling 4 small expert models into a Mixture-of-Experts with internal debate, targeting <1B parameters
- **Best Venue Fit:** NeurIPS 2026 Workshop on Efficient ML or ICLR Workshop; EMNLP 2026 (ARR)
- **Probability of Strong Submission:** **45%**
- **Reviewer Expectations:** Clear distillation protocol, comparison with single-model baseline of same parameter count, evidence that "debate" improves over naive MoE routing
- **Tier-1 vs Tier-2 Separator:** Tier-1 shows *emergent capability* from debate (e.g., math reasoning improves); Tier-2 is just parameter-matched comparison without qualitative analysis
- **8-Week Viability:** ⚠️ Medium — distillation is inference-heavy but feasible; challenge is getting meaningful results with tiny models
- **Evidence Needed:** 2-3 reasoning tasks, comparison with dense baseline of same size, ablation on number of experts, qualitative examples of debate improving answers

---

## 5. Final Two-Track Recommendation

### Track A: Tier-1 Swing — EMNLP 2026 ARR (Efficiency/Evaluation Track)

**Selected Direction:** "Constrained Reasoning Benchmark for Edge-Deployable Architectures" (Mapping 1)

**Why this is the Tier-1 swing:**
- **Deadline reachable:** May 25 ARR submission gives ~30 days for a first draft — tight but ARR allows revision cycles [web:18]
- **Novelty angle:** No existing benchmark specifically tests *reasoning* under 8GB VRAM constraints across SSM/hybrid/MoE architectures [cite:8]
- **Compute match:** Pure inference evaluation — all models can be run quantized/INT4 on the RTX 4070
- **Positioning:** Fits EMNLP's empirical methods focus and the growing efficiency sub-community
- **Hook:** "What happens to reasoning quality when you replace attention with SSM blocks? We measure it."

**Reviewer Expectations at EMNLP:**
- Clear task selection: which reasoning tasks matter for NLP? (e.g., multi-hop QA, logical inference, mathematical reasoning)
- Controlled protocol: same tokenizer, same context length, same evaluation script
- Model coverage: at least 6-8 model variants spanning pure Transformer, pure SSM, hybrid, and small MoE
- Analysis depth: not just accuracy, but *scaling behavior* with sequence length, *memory usage*, *throughput*
- Reproducibility: all models open-source, evaluation code public

**What separates Tier-1 from Tier-2 framing:**
| Tier-1 Framing | Tier-2 Framing |
|---|---|
| "We reveal a systematic degradation pattern in compositional reasoning when attention is replaced by SSM blocks, and propose a hybrid configuration that recovers 90% accuracy at 5× throughput" | "We compare 8 models on 3 tasks and report their scores" |
| Includes throughput/latency/accuracy Pareto frontier | Includes accuracy table only |
| Identifies failure modes by reasoning type (e.g., transitive inference fails on pure SSM) | No analysis of why models differ |
| Ablation on quantization impact (FP16 vs INT8 vs INT4) | Single precision setting |
| Open-source benchmark package with easy model addition | Static results without release |

**8-Week Plan for Track A:**
| Week | Task |
|---|---|
| Week 1-2 | Task selection (3 reasoning benchmarks), model selection (8 variants), evaluation framework code |
| Week 3-4 | Run all evaluations on RTX 4070, collect throughput + accuracy data |
| Week 5 | Analysis: Pareto curves, failure mode taxonomy, quantization impact |
| Week 6 | Draft paper (EMNLP format), related work section, write up findings |
| Week 7 | Internal review, figure refinement, reproducibility package |
| Week 8 | ARR submission (May 25) or polish for next ARR cycle |

**Probability estimate:** 60% for ARR acceptance (good reviews), 40% for EMNLP commitment, 70% for eventual acceptance at some ACL venue after revision cycle.

---

### Track B: Safer Fallback — NeurIPS 2026 Workshop on Efficient ML / Hardware-Aware ML

**Selected Direction:** "Bit-by-Bit: Simulation Evidence for Sparse Attention Hardware Acceleration" (Mapping 3) + "Auto-GIT: A Multi-Agent Evaluation Framework" (Mapping 2) as dual submission

**Why this is the safer fallback:**
- **No hard deadline pressure:** Workshop deadlines are typically after the main conference (July-September)
- **Preliminary work accepted:** Workshop reviewers expect work-in-progress, not finished contributions [web:39]
- **Two-pronged approach:** Submit two workshop papers to increase acceptance probability
- **Honest about limitations:** Can present Bit-by-Bit as "simulation-stage co-design exploration" rather than fabricated silicon results

**Workshop Selection Strategy:**

| Workshop | CFP Theme Fit | Submission Window | Acceptance Rate | Best Match |
|---|---|---|---|---|
| NeurIPS Efficient ML (EML) | Hardware-aware training/inference, quantization, sparse ops | July-Aug 2026 | ~40% | Bit-by-Bit GPU simulation |
| NeurIPS AI for Science | AI systems that accelerate research, code generation | July-Aug 2026 | ~35% | Auto-GIT evaluation |
| NeurIPS Agents | Multi-agent systems, agent evaluation | July-Aug 2026 | ~40% | Auto-GIT debating agents |
| EMNLP Efficiency | Low-resource NLP, efficient architectures | Post-ARR | ~45% | Constrained reasoning benchmark |

**Reviewer Expectations at Workshops:**
- Clear problem statement within workshop scope
- Honest assessment of what is implemented vs. planned
- Preliminary results with trend evidence (not final numbers)
- Community relevance to workshop attendees
- 4-6 pages, concise, focused

**What separates Tier-1 from Tier-2 workshop submissions:**
| Tier-1 Workshop Paper | Tier-2 Workshop Paper |
|---|---|
| Has preliminary results + clear roadmap to full paper | Is just an idea without any evidence |
| Connects to current workshop discussion topics | Generic submission that could go anywhere |
| Includes comparison with at least one baseline | No baselines, only the proposed approach |
| Honest limitations section that builds trust | Hides limitations or overclaims |
| Clear next steps: "We will extend X for NeurIPS/ICLR 2027" | No forward path |

**8-Week Plan for Track B:**
| Week | Task |
|---|---|
| Week 1-2 | Bit-by-Bit: Complete Verilator simulation of sparse attention unit; Auto-GIT: Design evaluation protocol |
| Week 3-4 | Bit-by-Bit: Run cycle-accurate benchmarks vs CPU; Auto-GIT: Collect 20 paper-code evaluation pairs |
| Week 5 | Both: Draft workshop papers (4-6 pages each), identify target workshops |
| Week 6 | Both: Internal review, prepare reproducibility artifacts |
| Week 7-8 | Submit to workshops as deadlines open; use any extra time for Track A revision |

**Probability estimate:** 65% for at least one workshop acceptance, 80% if submitting to 2-3 relevant workshops.

---

## 6. Strategic Recommendations

### Immediate Actions (Next 7 Days)
1. **Commit to Track A direction** — Select the 3 reasoning tasks and 6-8 model variants for the EMNLP benchmark. Do not oscillate.
2. **Draft the EMNLP paper skeleton** — Write the abstract, introduction, and related work first. This clarifies the contribution immediately.
3. **Set up evaluation harness** — A single Python script that can load any HuggingFace model, run a task, and log accuracy + latency + memory.

### Parallel Track B Preparation
4. **Identify 3 target workshops** — Monitor NeurIPS 2026 workshop announcements; join mailing lists.
5. **Advance Bit-by-Bit simulation** — Even 2-3 hours/week on Verilator simulation produces enough for a workshop paper.

### Risk Mitigation
6. **ARR fallback plan** — If May 25 ARR submission is too rushed, submit to the next ARR cycle (typically monthly). The paper can be committed to a future venue (e.g., NAACL, EACL 2027) [web:18].
7. **Compute backup** — Register for GitHub Student Pack cloud credits (if not already done), identify free Colab/Kaggle GPU slots for overflow experiments.
8. **Collaboration option** — Reach out to the Sarvam AI contact (Vinayakan) for advice on framing, not for co-authorship. Industry perspective strengthens positioning.

### The "NeurIPS 2027" Long Game
9. The most realistic path to NeurIPS/ICLR main track is **not 2026 but 2027** (deadlines: NeurIPS ~May 2027, ICLR ~Sep 2027). Use the 8-week sprint to produce:
   - A published workshop paper (Track B)
   - A strong ARR-reviewed paper (Track A)
   - Preliminary simulation results (Bit-by-Bit)

   Then spend 3-4 months expanding the best result into a full NeurIPS/ICLR 2027 submission with proper compute (possibly via internship or cloud credits).

---

## 7. Summary Decision Matrix

| Criterion | Track A: EMNLP ARR | Track B: Workshops |
|---|---|---|
| **Venue Prestige** | Main conference (high) | Workshop (medium) |
| **Acceptance Probability** | 40-60% | 65-80% |
| **Deadline Pressure** | High (May 25) | Low (July-September) |
| **Compute Required** | Low (inference only) | Low (simulation + inference) |
| **Novelty Barrier** | Medium (needs analysis depth) | Low (preliminary accepted) |
| **Revision Path** | ARR cycles enable iteration | Workshop feedback informs future |
| **Long-term Value** | Strong publication for portfolio | Networking + visibility |
| **Recommended Effort Split** | 60% | 40% |

**Final Recommendation:** Execute both tracks in parallel with a 60/40 effort split. The EMNLP ARR submission is the Tier-1 swing that builds the strongest publication asset. The workshop papers are the safety net that guarantees output, visibility, and community feedback. Both leverage the team's actual strengths (fast prototyping, evaluation design, hardware-software thinking) while honestly respecting the 8GB VRAM constraint.

---

*Agent 9 Assessment Complete.*
