# AGENT 29: Context Engineering and Long-Horizon Agent Design Scout Report

**Date:** April 2026  
**Scope:** Context engineering, memory compaction, long-horizon task decomposition, and prompt-state management for constrained agents and small models.  
**Constraint:** Engineering tips excluded; only defensible research angles reported.

---

## 1. Literature Map

### Core Pillars

**Context Management as a First-Class Primitive**  
Recent work has elevated context maintenance from a passive implementation detail to an active, learnable agent capability. Liu et al. propose CAT, a paradigm that treats context management as a callable tool integrated into agent decision-making. Their trajectory-level supervision framework (CAT-GENERATOR) trains SWE-Compressor, which reaches a 57.6% solved rate on SWE-Bench-Verified and significantly outperforms append-only ReAct agents and static compression baselines under bounded context budgets [page:1].

**Optimizing Compression Guidelines**  
Kang et al. (ACON) frame context compression as an optimization in natural-language space. Given paired trajectories where full context succeeds but compressed context fails, a capable LLM analyzes failure causes and iteratively refines compression guidelines [page:2]. ACON reduces peak token usage by 26–54% while preserving task performance, and distilled compressors retain over 95% accuracy when deployed on smaller language models [page:2].

**Lightweight Memory for Small Models**  
LightMem modularizes memory into short-term (STM), mid-term (MTM), and long-term (LTM) tiers, using small language models (SLMs) for online retrieval and offline consolidation [page:3]. It achieves approximately 2.5 F1 improvement over A-MEM on LoCoMo with median retrieval latency of 83 ms, showing that SLMs can manage memory without repeated large-model calls [page:3].

**OS-Inspired Hierarchies**  
MemGPT introduced an operating-system-inspired memory tiering system with interrupts and context switching, enabling document analysis and multi-session chat beyond native context windows [web:24]. This established the foundational vocabulary for memory hierarchies in agents.

**Context Engineering Practice**  
Anthropic's applied team codifies context engineering as "the art and science of curating what will go into the limited context window," distinguishing it from prompt engineering. They identify compaction, structured note-taking, and sub-agent architectures as the three primary techniques for long-horizon tasks, and document the phenomenon of "context rot" — degradation of recall accuracy as token count grows [page:4].

### Key Gaps

- Most compression research targets large models (32B+); the sub-7B agent regime remains under-measured.
- No standard benchmark isolates context engineering from raw reasoning ability.
- Causal links between context structure (ordering, sectioning, noise) and long-horizon drift are largely anecdotal.
- Sub-agent architectures are described in engineering blogs but rarely evaluated with controlled isolation studies [page:4].
- Browser-based and research-agent settings lack rigorous, small-model context management baselines [web:27].

---

## 2. What Is Engineering Only vs. Scientific Contribution

| Engineering Only | Defensible Scientific Contribution |
|---|---|
| Hand-tuned summarization prompts when context fills up | Formalized compression policy with measured information-loss curves and recovery bounds |
| Ad-hoc sub-agent decomposition for a specific codebase | Isolation protocol with quantified "attention contamination" metrics and generalization across task domains |
| Prompt templates telling the model to "remember key facts" | Hierarchical memory architecture with empirically validated tiering thresholds and latency-accuracy Pareto fronts |
| Using a vector DB to retrieve past turns without measuring noise | Online memory budget allocation as a constrained optimization with measurable regret or learned policies |
| Simply logging trajectories and picking "important" ones | Trajectory-level supervision framework with causal ablation showing context management actions improve final task success |
| One-off compaction heuristics (e.g., drop old tool outputs) | Systematic study of compaction aggressiveness vs. downstream task degradation with statistical significance |

The dividing line is **measurement and generalization**. If a technique is justified only by end-task accuracy on a single benchmark without isolating the context variable, it remains engineering. A scientific contribution must expose context structure as an independent variable, measure its causal effect on agent coherence, and provide transferable mechanisms or metrics.

---

## 3. Five Candidate Paper Ideas

### Idea A: Context Budget Allocation as Markov Decision Processes
**Thesis:** Frame long-horizon context maintenance as a token-budget allocation problem where the agent decides at each milestone what to preserve, summarize, or discard.  
**Novelty:** Formalize the agent's context window as a resource with diminishing marginal returns. Use lightweight RL or SLM-based classifiers to learn a preservation policy, and show that this policy transfers across SWE and web-browsing tasks.  
**Evidence needed:** Token budgets vs. task success curves; ablation of policy vs. static heuristics; generalization across 3B and 7B backbones.  
**Feasibility:** High. Builds directly on CAT [page:1] and ACON [page:2] but targets policy learning rather than fixed guidelines.

### Idea B: Measuring Context Rot in Multi-Turn Agents
**Thesis:** Systematically quantify "context rot" — the degradation of information retrieval accuracy as a function of turn count, context ordering, and signal-to-noise ratio in agent trajectories.  
**Novelty:** Move needle-in-a-haystack benchmarks into dynamic agent settings. Establish decay curves that link context pollution to sub-goal completion failure.  
**Evidence needed:** Controlled interventions on context ordering; measured drift in answer accuracy at turn 10, 50, 100; comparison across model families.  
**Feasibility:** Medium. Requires building a synthetic long-horizon diagnostic suite or adapting OdysseyBench [web:8].

### Idea C: Hierarchical Memory Distillation for Sub-7B Agents
**Thesis:** Determine whether a sub-1B compressor can maintain coherent long-horizon state for a 3–7B agent by enforcing an information bottleneck across STM, MTM, and LTM tiers.  
**Novelty:** Extends LightMem [page:3] and ACON distillation [page:2] into a unified hierarchy with measured recall-to-token tradeoffs specifically for small models.  
**Evidence needed:** Latency and memory overhead budgets; tier-ablation studies; comparison against inline full-context baselines on BrowserGym [web:27] and AppWorld [page:2].  
**Feasibility:** High. Well-scoped for a student project with existing toolchains.

### Idea D: Instruction Isolation and Clean Handoffs in Sub-Agent Decomposition
**Thesis:** Prove that isolating sub-agent contexts prevents "semantic leakage" and improves parent-agent planning fidelity in long-horizon research and coding tasks.  
**Novelty:** Quantify the cost of inlining sub-agent traces vs. returning condensed summaries. Introduce an isolation score and show it correlates with reduced goal drift.  
**Evidence needed:** Parallel experiments with isolated vs. inlined contexts; human/LLM-judged handoff quality; task completion on multi-agent research benchmarks.  
**Feasibility:** Medium. Requires a multi-agent harness but can reuse existing benchmarks like SWE-Bench or BrowserGym.

### Idea E: DiCE — A Diagnostic Benchmark for Context Engineering
**Thesis:** Create a benchmark where task reasoning difficulty is held constant while context engineering demands vary (e.g., distractor injection, required cross-referencing, history dependency).  
**Novelty:** Isolates context management as an independent variable, enabling apples-to-apples comparison of compaction, retrieval, and hierarchical memory methods.  
**Evidence needed:** Pilot studies showing existing agents fail primarily due to context loss rather than reasoning; validated difficulty tiers; reproducible environment harness.  
**Feasibility:** Medium-High. High impact if adopted, but requires significant environment engineering.

---

## 4. Best Benchmarks or Task Suites

| Benchmark | Domain | Why It Fits | Key Metric |
|---|---|---|---|
| **SWE-Bench-Verified** | Software engineering | Gold-standard for long-horizon code agents; CAT/SWE-Compressor validated here [page:1] | Solved rate under token budget |
| **AppWorld / OfficeBench** | Productivity apps | ACON's primary evaluation; multi-step tool use with observable state [page:2] | Peak token reduction + task success |
| **OdysseyBench** | Long-horizon workflows | Explicitly designed for multi-step reasoning across diverse applications [web:8] | Sub-goal completion accuracy |
| **BrowserGym / WebArena / WorkArena** | Web navigation | Realistic browser-based settings where agents must manage DOM state and history [web:27] | Success rate, steps to completion |
| **GAIA** | General assistant | Tests reasoning, web browsing, and tool use at three difficulty levels [web:26] | Accuracy + cost |
| **AgentLAB** | Robustness | Evaluates susceptibility to long-horizon attacks and multi-turn failures [web:22] | Attack success rate / robustness score |

**Usage recommendation:** For rapid iteration on context compression methods, use **AppWorld** or **BrowserGym** [web:27]. For final validation and credibility, report results on **SWE-Bench-Verified** or **OdysseyBench** [web:8]. If the work targets research agents specifically, **GAIA** [web:26] provides a strong general-assistant signal. If evaluating robustness to context manipulation, **AgentLAB** [web:22] offers a long-horizon attack surface.

---

## 5. Evidence Needed for Publishability

To convert any candidate idea into a defensible paper, the following evidence is required:

- **Causal isolation.** Show that modifying context structure directly causes performance change while holding reasoning components constant. Ablations must isolate the context variable.
- **Cross-scale validation.** Demonstrate the method on at least two model sizes (e.g., 3B and 7B) or provide explicit scaling curves that reveal where the approach breaks down.
- **Token-efficiency frontier.** Report Pareto curves of task success vs. peak token usage or latency. A 5% accuracy gain at 3× token cost is not a clean contribution.
- **Failure mode taxonomy.** Catalog how context mismanagement manifests — semantic drift, goal misalignment, tool repetition, hallucinated state — with concrete trajectory examples.
- **Generalization across domains.** Test on at least two distinct long-horizon domains (e.g., coding and web browsing) to claim the method is a general context primitive rather than a domain hack.
- **Reproducible artifacts.** Open-source the harness, compression policies, and evaluation splits. Context engineering is noisy; reproducibility is non-negotiable.
- **Statistical rigor.** Report confidence intervals or repeated trials across task instances. Single-point accuracy claims on small held-out sets are insufficient for archival publication.

---

## 6. Best Recommendation

**Pursue Idea C: Hierarchical Memory Distillation for Sub-7B Agents**, validated on BrowserGym and AppWorld, with cross-scale ablation on 3B and 7B models.

**Rationale:**

1. **Underserved niche.** The literature focuses on large-model compression (32B+) [page:1][page:2] or generic SLM conversation memory [page:3], but no work rigorously measures what hierarchical compression enables for *agentic* small models on long-horizon browser and productivity tasks.
2. **Measurable impact.** You can plot explicit recall-token Pareto fronts and latency budgets, giving reviewers concrete quantitative claims.
3. **Feasible scope.** It does not require building a new benchmark from scratch (reuse BrowserGym/WebArena [web:27] and AppWorld [page:2]) nor training a backbone from scratch (distill existing compressors or train lightweight policy heads).
4. **Direct leverage of existing strengths.** Given your background in model quantization, SLMs, and embedded systems, you can frame this as "efficient agent infrastructure" rather than pure NLP, increasing differentiation.
5. **Engineering-to-science bridge.** By measuring tier-ablation effects and isolation gains, you convert the engineering practice of "summarize old context" into a quantitative contribution with generalizable thresholds.

**Immediate next steps:**
- Baseline a 3B/7B agent (e.g., Qwen2.5 or Llama-3.2) on BrowserGym with naive append-only context and measure degradation at 20, 50, and 100+ turns.
- Implement a two-tier compressor (short-term inline + distilled long-term summary) using an SLM head or lightweight transformer.
- Run tier-ablation studies: disable MTM, disable LTM, vary compression ratio, and plot task success vs. peak tokens.
- Draft the paper around the empirically derived thresholds where each memory tier becomes necessary for coherent long-horizon execution.
