# AGENT_11_AGENTIC_REASONING.md
# Agent 11: Agentic Reasoning and Tool-Use Scout
**Mission:** Research underexplored opportunities in small-model agentic reasoning under hard resource constraints
**Date:** April 25, 2026
**Hardware Target:** RTX 4070 Laptop, 8GB VRAM
**Timeline:** 1–2 months to technical results

---

## 1. Executive Summary

The agentic AI landscape in 2026 is bifurcated: frontier-model labs (Anthropic, OpenAI, Google DeepMind) own the
large-model, cloud-compute end of the spectrum, while a rapidly growing but underserved niche asks a sharper question:
**can small models (3B–8B params, ≤8GB VRAM) be made reliably agentic?** The answer is "yes, partially" — but the
*mechanisms* by which small models succeed or fail in agentic loops are still poorly understood, and this is where
publishable, differentiated research lives.

Key findings from this scan:

1. **Tool use is the biggest lever for small models.** A 4B model with tools outperforms a 32B model without tools on
   GAIA benchmarks (arXiv 2601.11327). This finding is recent (Jan 2026) and its mechanistic cause is unexplored.

2. **Planning vs. Reacting under resource pressure is unsolved.** Plan-and-Execute is efficient for deterministic
   tasks; ReAct adapts better to noisy observations — but nobody has done a rigorous ablation on *small* models
   specifically. For 7B-class models, "thinking too much" (chain-of-thought) can *hurt* tool-call accuracy.

3. **Memory in agent loops is still benchmarked poorly.** Existing memory benchmarks test recall, not the write-
   manage-read lifecycle under context budget constraints relevant to 8GB VRAM inference.

4. **Browser-grounded reasoning for small models is wide open.** Frontier agents dominate WebArena/GAIA web tasks;
   small-model browser agents are nearly unstudied.

5. **Evaluation is broken.** Most top benchmarks (SWE-bench, GAIA Level 3) are already saturated by frontier models
   or contaminated. Small-model agentic evaluation — especially under token and context budget constraints — lacks
   dedicated tooling.

**Bottom line for Parshu's team:** The sweet spot is the intersection of **(a) constrained context management,
(b) tool-selection robustness, and (c) failure mode characterization** for 3B–8B models. These areas have few
papers, are feasible on RTX 4070, and are exactly what workshop/track reviewers at ACL/EMNLP/NeurIPS Workshops
need to see in 2026.

---

## 2. Literature Map

### 2.1 Saturated / Crowded Zones (avoid or reframe)

| Topic | Saturation Signal | Notes |
|---|---|---|
| ReAct / Chain-of-Thought prompting | Huge body of work; diminishing returns | ReAct 2022, ToT 2023, etc. |
| RAG + Agent integration | Massive industry adoption, many surveys | Not differentiated unless paired with constraint |
| LLM function calling benchmarks (BFCL) | Leaderboard-saturated, frontier-dominated | BFCL v4 dominated by Claude/GPT-4o |
| SWE-bench coding agents | Contamination confirmed; benchmark degraded | OpenAI stopped reporting |
| Multimodal VQA (visual QA only) | Extremely crowded; QwenVL/InternVL dominate | OK only if you add agentic loop |
| Memory: episodic recall benchmarks | Static recall well-studied | Only novel if you study write/manage/prune under budget |

### 2.2 Active / Growing (entry is still possible)

| Topic | Signal | Key Papers |
|---|---|---|
| Small model + tool use (agentic) | Very recent breakthrough (Jan 2026) | arXiv 2601.11327 |
| Context engineering for agents | First-class research framing emerging | Context-Bench (Letta 2025), ACE framework |
| Memory under context budget pressure | 2025–2026 surge; evaluation gaps remain | arXiv 2603.07670, 2604.21480 |
| Agent failure mode taxonomy | Almost no work below 13B | JetBrains blog; IBM 2025 |
| Multimodal small agents (GUI+vision) | MiniCPM-V; Qwen2.5-VL; Magma; underexplored | Nature 2025 (MiniCPM-V) |
| Evaluation under compute constraints | No dedicated benchmark | Gap confirmed in arXiv 2604.19299 |

### 2.3 Underexplored (publishable white space)

| Topic | Why it's open |
|---|---|
| Failure mode taxonomy for <8B agents | No systematic study; mostly qualitative |
| Planning overhead vs. accuracy tradeoff for 3–7B models | Only studied for large models |
| Token budget–aware tool selection | Tool confusion at >10 tools known; no fix for small models |
| Multi-session agent memory with prune policies | Benchmarks measure recall, not prune accuracy |
| Browser-grounded reasoning for SLMs | WebArena/GAIA dominated by frontier models; SLM baseline missing |
| Hybrid routing: SLM handles routine + LLM escalation | Described as direction; no rigorous benchmarking |

---

## 3. Top 6 Underexplored Gaps

### Gap 1 — Failure Modes of Sub-8B Agents Are Taxonomically Unstudied
Existing failure mode research targets large models (≥13B) or is qualitative blog/IBM-style. For small models
running tool-augmented ReAct or Plan-and-Execute loops, the *specific* failure types (hallucinated parameters,
wrong tool selection, context truncation mid-reasoning, compounding error recovery) are undocumented at scale.
**Why publishable:** Failure taxonomy papers consistently get strong workshop reception; fills an obvious empirical gap.

### Gap 2 — Token-Budget–Aware Tool Selection for Small Models
Anthropic's agent skills framework demonstrates 70–90% token reduction with progressive context disclosure.
However, this was developed and tested for frontier models. For 3–8B models, where effective context is even smaller,
no paper has studied how to dynamically prune tool descriptions, compress observation windows, or route tool calls
under strict token budgets. **Why publishable:** Directly actionable; connects to context engineering literature.

### Gap 3 — Planning Strategy Selection as a Function of Model Size
ReAct vs Plan-and-Execute is studied comparatively for large models. New evidence shows that for 4B models, "thinking
too much" (full CoT in agentic loops) *degrades* tool coordination (arXiv 2601.11327). This is a model-size-dependent
effect with no mechanistic explanation and no prescriptive rule. **Why publishable:** Clean hypothesis; empirically
testable in 1–2 months; directly useful to practitioners.

### Gap 4 — External Memory Prune Policies Under VRAM Constraints
Memory-augmented agent loops (RAG memory, episodic stores) are well-studied for recall performance. But under 8GB
VRAM, the memory bank itself competes with the KV-cache, and no paper studies optimal prune/eviction policies for
external agent memory specifically in this hardware regime. **Why publishable:** Novel constraint; connects to the
growing memory taxonomy literature (arXiv 2603.07670).

### Gap 5 — Browser-Grounded Reasoning Benchmarks for SLMs
WebArena, GAIA web tasks, and Mind2Web are almost entirely frontier-model benchmarks. No paper has established a
baseline for 3–8B models on browser-grounded agentic tasks with realistic internet latency and constrained context
windows. **Why publishable:** Benchmarking/analysis paper; lower experimental burden; strong community need.

### Gap 6 — Hybrid SLM/LLM Routing with Hard Compute Budgets
The modular agentic architecture (SLM for routine tasks, LLM escalation) is described in NVIDIA's SLM survey and
mentioned in practitioner blogs, but no paper has formalized the routing decision, built a training protocol for the
router, or benchmarked under real latency constraints. **Why publishable:** First rigorous treatment of a widely
assumed but unstudied architecture.

---

## 4. Benchmark and Dataset Suggestions

### 4.1 Existing Benchmarks Viable for Small Models

| Benchmark | Focus | SLM Feasible? | Notes |
|---|---|---|---|
| GAIA Level 1 | General assistant + tool use | Yes (Level 1 only) | Level 2/3 saturated by frontier |
| AgentBench (subset) | Multi-environment reasoning | Partially | Use OS + web shopping subsets |
| τ-bench / τ²-bench | Tool use + policy adherence | Yes | Service domain; realistic tool calls |
| BFCL v3 (Simple subset) | Function calling accuracy | Yes | Use non-parallel call subsets |
| Context-Bench (Letta) | Context management in agents | Yes | Directly relevant to Gap 2/4 |
| WebArena (local subset) | Browser + GUI navigation | Partially | Run with local mock server |
| ToolBench (subset) | Tool selection from large catalogs | Yes | Test with limited tool sets (5–20) |

### 4.2 Datasets You Can Build / Adapt (Novel Contribution)

1. **SLM-FailBench** — Collect and annotate agent trajectories from Qwen2.5-7B / Phi-4 / Gemma-3-4B on existing
   benchmarks, label failure modes (hallucinated params, wrong tool, context truncation, planning loop, recovery
   failure). ~500 annotated trajectories = novel dataset contribution.

2. **BudgetTool-Eval** — Variants of τ-bench tasks with token budgets imposed (1K, 2K, 4K token limits per turn).
   Measures graceful degradation under budget. Feasible to construct in 2 weeks.

3. **MemPrune-Agent** — Multi-session agent tasks where memory must be pruned between sessions due to VRAM limits.
   Evaluate downstream task accuracy as a function of prune policy.

---

## 5. Reviewer Risk Assessment

| Risk Category | Level | Mitigation |
|---|---|---|
| "This is just prompt engineering" | HIGH for ReAct/prompting papers | Frame as empirical study with ablations, not a new prompting trick |
| "Results only on weak models" | MEDIUM | Use multiple model families (Qwen, Phi, Gemma); show scaling behavior |
| "Benchmark contamination" | HIGH for GAIA/SWE | Avoid GAIA Level 2+; create held-out splits; use τ-bench which has domain variants |
| "No comparison to frontier baseline" | MEDIUM | Explicitly frame as resource-constrained setting; compare to best ≤8B baselines |
| "Dataset too small" | HIGH for taxonomy papers | 500+ annotated trajectories minimum; release dataset publicly |
| "No theoretical grounding" | LOW for workshop papers | Add one analytical section showing token-efficiency bounds |
| "Overlap with concurrent work" | MEDIUM for memory papers | Surge in memory papers 2025–2026; must cite arXiv 2603.07670 and differentiate |
| "Incremental over Tool Orchestra / NVIDIA SLM survey" | MEDIUM | Differentiate on the hard-constraint framing specifically |

---

## 6. Top 2 Realistic Paper Ideas

### Paper R1 — "When Thinking Hurts: Planning Strategy Selection for Sub-8B Agentic Models"

**Core claim:** For models ≤7B, heavy chain-of-thought planning in agentic loops degrades tool-call accuracy, and
a lightweight strategy-selector that routes tasks to ReAct vs. lightweight Plan-and-Execute based on task complexity
and available context budget recovers most of the performance gap.

**Method:**
1. Run Qwen2.5-7B, Phi-4-mini, Gemma-3-4B on τ-bench and AgentBench subsets under three conditions:
   (a) ReAct baseline, (b) Plan-and-Execute with full CoT, (c) Adaptive strategy selector (binary classifier on
   task descriptor).
2. Measure tool-call accuracy, context usage, error recovery rate.
3. Train a 5M-param strategy router using synthetic task complexity labels.

**Feasibility on 8GB VRAM:** All inference runs at INT4/INT8 via llama.cpp or vLLM. Training the router is tiny.
Total compute: ~40–60 hours of GPU time spread over 1 month.

**Venue:** ACL 2026 System Track, EMNLP 2026 Findings, or *Efficient NLP Workshop* at ACL/EMNLP.
**Reviewer concern:** Show it generalizes across ≥3 model families.

---

### Paper R2 — "Taxonomy and Frequency Analysis of Agentic Failure Modes in Sub-8B LLMs"

**Core claim:** Failure modes of small model agents are systematic and model-size-dependent; we provide the first
annotated corpus of 600+ agent failure trajectories, a taxonomy of 7 failure types, and a lightweight failure
predictor that can abort loops before they cascade.

**Method:**
1. Run 3 small models (Qwen2.5-7B-Instruct, Phi-4-mini, Gemma-3-4B-IT) on GAIA Level 1, τ-bench, and ToolBench
   subset with tool-augmented ReAct.
2. Manually annotate failure trajectories (2 annotators + LLM assist). Define taxonomy: [wrong-tool-selection,
   hallucinated-params, context-truncation-cascade, planning-loop, recovery-overwrite, format-drift, early-termination].
3. Train a trajectory-level failure predictor (lightweight LSTM/linear over action embeddings).
4. Release dataset publicly.

**Feasibility:** 600 trajectories × 3 models = 1800 runs. At 2–5 min/run locally, ~1–2 weeks of GPU time.
**Venue:** EMNLP 2026 Findings, NeurIPS 2026 Efficient ML Workshop, or ACL ARR cycle.
**Novel contribution:** First annotated failure corpus for SLMs; dataset + predictor.

---

## 7. Top 1 Ambitious Paper

### Paper A1 — "VRAM-Aware Agentic Architectures: Memory, Context, and Tool Routing for On-Device LLM Agents"

**Core claim:** We present a unified architecture for on-device agentic reasoning that co-optimizes (a) external
memory prune policy, (b) tool description compression, and (c) planning strategy selection, all under a hard 8GB
VRAM budget, and demonstrate that it achieves 85%+ of frontier model performance on τ-bench while running entirely
locally.

**Architecture components:**
- **MemPrune Controller:** VRAM-aware memory eviction using retrieval importance + recency scoring. Prunes episodic
  store dynamically to keep KV-cache + memory under 6GB.
- **ToolCompressor:** Per-query tool description compression using a small summarizer; selects top-K tools from
  catalog before injecting into context.
- **Adaptive Strategy Selector:** Binary router (Plan-and-Execute vs. ReAct) conditioned on task complexity and
  remaining context budget.

**Why ambitious:**
- Three-component system; each component is independently publishable.
- Requires integration work across llama.cpp / vLLM / LangGraph.
- Timeline risk: 2 months is tight for a full system paper; more realistic for a workshop system demo + arXiv preprint.

**Venue target:** NeurIPS 2026 (Efficient LLMs for Agentic Systems Workshop), or ICLR 2027 (if extended to full
system paper). Alternatively, ACL 2026 System Demonstrations track.

**Reviewer appeal:** End-to-end system with hardware-grounded constraints is highly differentiated. Lack of
theoretical proofs is acceptable at workshop level if empirical gains are solid.

---

## 8. Final Pursue/Skip Ranking

| Rank | Topic | Decision | Rationale |
|---|---|---|---|
| 1 | **Failure mode taxonomy for SLMs (R2)** | **PURSUE FIRST** | Lowest bar to entry; dataset paper; high impact; feasible in 4–5 weeks |
| 2 | **Planning strategy selection for SLMs (R1)** | **PURSUE SECOND** | Clean hypothesis; builds on R2 trajectory data; runs on same models |
| 3 | **VRAM-aware unified agent architecture (A1)** | **PURSUE IF TIME ALLOWS** | Most impactful; use as capstone combining R1+R2 components |
| 4 | **Browser-grounded SLM baseline study** | **SKIP FOR NOW** | WebArena local setup overhead is high; return in semester 2 |
| 5 | **Hybrid SLM/LLM routing** | **SKIP FOR NOW** | Requires API access to LLM for comparison; cost/access risk |
| 6 | **Memory prune benchmarks** | **CONSIDER AS EXTENSION** | Relevant but needs custom dataset construction; bolt onto A1 |
| 7 | **Multimodal grounded agents** | **SKIP** | MiniCPM-V / QwenVL dominate; differentiation very hard in 2 months |
| 8 | **RAG + agent integration** | **SKIP** | Saturated; no differentiated angle without constraint framing |

---

## Appendix: Feasible Models on RTX 4070 (8GB VRAM)

| Model | Params | VRAM (INT4) | Tool Use? | Notes |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | ~4.5 GB | Yes | Best SLM for tool use as of 2025 |
| Phi-4-mini | 3.8B | ~2.5 GB | Yes | Strong reasoning per param |
| Gemma-3-4B-IT | 4B | ~3 GB | Yes | Good baseline; well-documented |
| Qwen3-4B-Instruct | 4B | ~3 GB | Yes | Jan 2026; tool-augmented agentic |
| Mistral-7B-Instruct-v0.3 | 7B | ~4.5 GB | Yes | Established baseline |
| Llama-3.2-3B-Instruct | 3B | ~2 GB | Partial | Use for low-memory ablations |
| MiniCPM-V-2.6 | 8B | ~6 GB | Multimodal | Only if multimodal track added |

**Inference stack:** llama.cpp (GGUF Q4_K_M) or vLLM with INT4 AWQ. For agentic loops: LangGraph or custom Python
loop (lighter overhead). Avoid CrewAI for research — too much scaffolding noise in measurements.

---

*Generated by Agent 11 | Parshu / AlienX Research | April 2026*
