# Agent 20: Human Evaluation and Research Methodology
## Lightweight Protocols for Student Frontier Research

**Version:** 1.0 | **Date:** April 2026  
**Scope:** Rigorous, low-cost human evaluation for LLM agents, RAG systems, memory modules, and tool-use pipelines.

---

## 1. Survey of Human Evaluation Practices

Human evaluation remains the gold standard for dimensions that automated metrics fail to capture, including nuanced reasoning, subjective usefulness, and agentic safety [web:1][web:3]. The field currently relies on three dominant paradigms:

- **Direct Scoring:** Annotators rate a single output on a Likert scale across multiple dimensions (e.g., groundedness, coherence). This is fast but suffers from scale-region bias and low discriminative power between similar systems [web:21].
- **Pairwise Preference:** Annotators choose which of two outputs is better along a specific dimension. This produces more stable rankings and is statistically more efficient than direct scoring for comparative claims [web:21][web:29].
- **Rubric-Based Evaluation (RBE):** Annotators use fine-grained rubrics (typically 0–4 points) with concrete behavioral anchors. RBE improves inter-annotator reliability (IAA) and aligns better with expert judgment than holistic scoring [web:19].

A fourth emerging approach—**LLM-as-a-Judge**—is frequently used as a proxy, but Tier-1 venues increasingly require human validation of these automated judges in the specific domain studied [web:15][web:14]. For student research, the recommended strategy is a hybrid: use rubric-based direct scoring for absolute quality assessment, and pairwise preference for head-to-head comparisons, both backed by small-scale human calibration sets.

---

## 2. Weak and Misleading Practices in Current Papers

A systematic reproduction effort of human evaluation experiments in NLP found critical flaws in *every* experiment reviewed, ranging from coding errors to questionable statistical practices [web:22].

| Flaw | Why It Misleading | Fix |
|------|-------------------|-----|
| **Ad-hoc participant exclusion** | Researchers exclude annotators post-hoc based on "expected" answers without a pre-registered protocol. This artificially inflates agreement and hides variance [web:22]. | Define exclusion criteria *before* annotation begins (e.g., failed attention checks, time thresholds). Report the number excluded and the rationale. |
| **No statistical testing** | Only 33% of ACL/INLG papers with human evaluation report statistical analysis [web:26]. Without it, differences between systems may be due to chance. | Always report confidence intervals and use appropriate tests (Wilcoxon signed-rank for paired, Mann-Whitney U for independent). |
| **Cherry-picked examples** | Papers showcase a few vivid outputs without systematic sampling. This is anecdotal, not evaluative. | Sample randomly from the test set, stratifying by difficulty or domain if needed. Report the sampling frame. |
| **Vague or absent annotation guidelines** | Weak guidelines lead to noise that looks like signal. NAACL 2024 research identified eight recurring vulnerabilities in annotation guidelines [web:9]. | Use decision trees for edge cases, include positive/negative examples, and pilot-test guidelines before full deployment. |
| **Confusing evaluation with demonstration** | Showing that a system *can* produce good output is not evidence that it *usually* does. | Report mean/median scores with variance, not just best-case outputs. |
| **Ignoring annotator demographics** | For agentic systems, cultural and technical background heavily influences judgments of "usefulness" [web:5]. | Report annotator pool size, domain expertise, and compensation. |

---

## 3. Annotation Templates for This Project Space

These templates are optimized for agentic systems (tool use, memory, MCP orchestration) and can be deployed in Google Forms, LimeSurvey, or custom HTML.

### Template A: Groundedness & Faithfulness (RAG / Tool-Use)
**Task:** Given the system context (retrieved documents or tool outputs) and the agent response, evaluate the response.

| Dimension | Score | Anchor Description |
|-----------|-------|--------------------|
| **Groundedness** | 2 | Every factual claim is directly traceable to the provided context or a verified tool output. |
| | 1 | Most claims are supported, but 1–2 minor claims are unsupported or over-generalized. |
| | 0 | Contains a hallucination, contradiction, or fabricated tool result. |
| **Attribution** | 1 | The response explicitly cites/attributes its sources where appropriate. |
| | 0 | Sources are used but not cited, making verification impossible. |

**Decision Rule:** A response scoring 0 on Groundedness is automatically failed, regardless of other scores.

### Template B: Reasoning Correctness & Coherence (Agentic Planning)
**Task:** Given a user request and the agent’s step-by-step reasoning trace (CoT), evaluate the trace.

| Dimension | Score | Anchor Description |
|-----------|-------|--------------------|
| **Reasoning Correctness** | 2 | No logical gaps; each step validly follows from the previous and from the context. |
| | 1 | Minor logical gap or one unsupported assumption that does not derail the final answer. |
| | 0 | Critical flaw in deduction, circular reasoning, or non-sequitur leading to wrong action. |
| **Coherence** | 2 | The trace reads as a unified, understandable plan. Steps are ordered logically. |
| | 1 | Mostly coherent but one abrupt transition or redundant step. |
| | 0 | Incomprehensible ordering or contradictory steps. |

### Template C: Usefulness & Task Completion (End-to-End Agent)
**Task:** Rate whether the agent output actually solves the user’s problem.

| Dimension | Score | Anchor Description |
|-----------|-------|--------------------|
| **Usefulness** | 2 | Fully solves the user’s intent; no further manual cleanup or re-prompting needed. |
| | 1 | Partial solution requiring minor user intervention or missing one secondary constraint. |
| | 0 | Fails the primary task or produces actively harmful/deceptive output. |
| **Instruction Adherence** | 1 | Follows all explicit constraints (format, length, tone, tool restrictions). |
| | 0 | Violates at least one explicit constraint. |

### Template D: Pairwise Preference (Comparative Head-to-Head)
**Task:** You are shown System A and System B outputs for the same prompt. Select which is better, or mark a tie.

- **Question 1 (Overall Quality):** Which response is more helpful and accurate?
- **Question 2 (Groundedness):** Which response is more faithful to the source material?
- **Question 3 (Safety):** Which response is safer and more aligned with instructions?

**Tie Policy:** Allow ties, but instruct annotators to avoid them unless outputs are truly indistinguishable. Record tie frequency—it indicates model convergence.

---

## 4. Inter-Annotator Agreement Recommendations

IAA is not a formality; it is a validity check. If annotators do not agree, the construct being measured is either ill-defined or the guidelines are insufficient [web:10][web:14].

### Metric Selection

| Annotator Count | Data Type | Recommended Metric | Notes |
|-----------------|-----------|--------------------|-------|
| Exactly 2 | Nominal (categories) | **Cohen’s Kappa (κ)** | Corrects for chance agreement using annotator-specific marginals. |
| ≥ 3 | Nominal | **Fleiss’ Kappa** | Generalizes Cohen’s to multiple annotators. Sensitive to class imbalance [web:10]. |
| ≥ 2 | Ordinal / Missing data | **Krippendorff’s Alpha (α)** | Handles incomplete designs and different distance metrics (e.g., ordinal penalties for 1-vs-3 disagreements) [web:10]. |
| Pairwise ranks | Binary preference | **Kendall’s Tau (τ)** or **Bradley-Terry model** | For aggregating pairwise comparison matrices into a global ranking [web:21]. |

### Thresholds and Interpretation

- **κ / α ≥ 0.80:** Excellent. Proceed with analysis. [web:23]
- **0.67 – 0.80:** Acceptable for exploratory work, but the rubric needs refinement before publication.
- **< 0.67:** Unacceptable. Stop. Revise guidelines, add training examples, or simplify the construct [web:6].

### Practical Protocol for Student Teams

1. **Pilot Round:** 3 annotators × 20 examples (10% of target set).
2. **Compute α.** If α < 0.67, hold a reconciliation meeting, identify分歧 items, and rewrite anchors.
3. **Full Round:** Only after pilot α ≥ 0.67.
4. **Report:** Always report the IAA metric, the exact number of annotators, and whether adjudication was used. Never report only "high agreement" without a metric [web:15].

---

## 5. Sample Study Designs for Grounding, Memory, and Agentic Systems

### Study A: Grounding Evaluation for RAG / Tool-Augmented Agents
**Goal:** Measure whether your agent hallucinates or drifts from retrieved/tool context.

- **Design:** Randomly sample 100 queries from your test set. For each query, collect the top-k retrieved chunks or tool outputs and the agent’s final response.
- **Annotations:** Use Template A. Each response annotated by 3 annotators.
- **Analysis:** Report mean Groundedness score, percentage of responses scoring 0 (hallucination rate), and Fleiss’ κ.
- **Cost:** ~300 annotations. At 2 minutes each, ~10 annotator-hours.
- **Variant:** Add an adversarial condition where 20% of contexts contain contradictory information. Measure if the agent detects or propagates the contradiction.

### Study B: Memory Evaluation for Long-Context / Stateful Agents
**Goal:** Assess whether the agent correctly recalls, updates, and does not confabulate memory entries over multi-turn sessions.

- **Design:** Create 25 synthetic multi-turn dialogues (5–10 turns) where facts are introduced, updated, or contradicted in specific turns. Extract the agent’s memory state or final answers dependent on earlier turns.
- **Annotations:** Use Template B (Reasoning Correctness) plus a custom **Memory Fidelity** check: "Did the agent correctly recall the most recent version of the fact?"
- **Analysis:** Report per-turn accuracy curves (does accuracy decay with turn count?), confusion matrix for "update" vs. "ignore" vs. "hallucinate", and Cohen’s κ if two expert annotators score.
- **Cost:** ~75 annotations (if evaluating final state only). ~375 if evaluating every turn.

### Study C: End-to-End Agentic System (MCP / Multi-Tool)
**Goal:** Compare your AlienX-style agentic pipeline against a baseline (e.g., ReAct or direct LLM) on real-world task completion.

- **Design:** Define 50 task scenarios (e.g., "Clone repo X, fix bug Y, open PR Z"). Run both systems. Shuffle outputs and present pairwise.
- **Annotations:** Use Template D (Pairwise). 3 annotators per pair.
- **Analysis:** Aggregate pairwise results into a win/loss/tie matrix. Use the Bradley-Terry model to derive a single latent quality score per system. Report 95% CIs via bootstrapping [web:21].
- **Cost:** 150 pairwise judgments. Faster than direct scoring because humans are better at relative comparison.

---

## 6. Fastest Robust Protocol

For teams with severe time/budget constraints (e.g., hackathon deadlines or course project milestones), use the **Minimum Viable Human Evaluation (MVHE)** protocol.

| Step | Action | N |
|------|--------|---|
| **1. Sample** | Randomly select 50 examples from your test set. | 50 |
| **2. Annotate** | 2 trained annotators score independently using a 3-point rubric on **one primary dimension** (e.g., Faithfulness). | 100 judgments |
| **3. Agree** | Compute Cohen’s κ. If κ ≥ 0.70, average the two scores. If not, adjudicate by discussion and report the adjudicated score separately. | — |
| **4. Test** | Run a Wilcoxon signed-rank test (if comparing two systems on the same items) or Mann-Whitney U (if independent). | — |
| **5. Report** | Report mean ± std, median, κ, p-value, effect size (Cliff’s delta or Cohen’s d), and a histogram of score distributions. | — |

**Rationale:** 50 items is the smallest sample that typically yields stable mean estimates for Likert-scale NLP data; 2 annotators keep costs minimal while still permitting IAA calculation; and restricting to one primary dimension prevents guideline dilution [web:26][web:30].

---

## 7. Common Reviewer Criticisms and Fixes

Based on ARR reviewer guidelines and reproduction studies [web:15][web:22], here are the most frequent objections to human evaluation sections in student/submitted papers, and how to preempt them.

### Criticism 1: "No evidence that human evaluation is reliable"
**Fix:** Compute and report κ or α. State the threshold you targeted and whether you met it.

### Criticism 2: "Sample size is unjustified"
**Fix:** Justify 50+ items by citing the NLP evaluation literature [web:26] or by performing a power analysis (even approximate) based on an expected effect size. Report power (1-β) if possible.

### Criticism 3: "Using LLM-as-a-judge without human validation"
**Fix:** If you use GPT-4/Claude as a judge, validate a subset (e.g., 50 items) against human judgment. Report the correlation (Pearson/Spearman) or agreement (κ) between LLM and human scores. If it is low, do not rely on the LLM judge [web:15].

### Criticism 4: "Lack of statistical significance / overclaiming"
**Fix:** Never say "System A outperforms System B" without a p-value or confidence interval. Use the exact permutation test if parametric assumptions are violated.

### Criticism 5: "Results are not reproducible"
**Fix:** Publish anonymized annotation sheets, rubrics, and participant instructions as supplementary material. Include the prompt templates used to elicit model outputs. Tag the exact model version (e.g., `gpt-4-0125-preview`) [web:15].

### Criticism 6: "Annotator pool is too small or homogeneous"
**Fix:** If you use 2–3 student annotators, be transparent. Acknowledge the limitation and frame the study as an "expert evaluation" or "in-house pilot." If generalizability to end-users is claimed, you must recruit a larger, diverse pool.

### Criticism 7: "The rubric conflates multiple constructs"
**Fix:** Separate dimensions cleanly. If your rubric mixes "fluency" and "correctness," split them. Use one construct per score to keep IAA interpretable.

---

## Quick-Reference Cheat Sheet

| Question | Answer |
|----------|--------|
| How many annotators? | Minimum 2 for pilot, 3 for full study if budget allows. |
| How many items? | Minimum 50 for stable means; 100+ for subgroup analysis. |
| Direct or pairwise? | Use **pairwise** for comparing systems; **direct** for absolute quality monitoring. |
| Which IAA metric? | Cohen’s κ (2 annotators), Fleiss’ κ (3+), Krippendorff’s α (ordinal/missing). |
| What p-value threshold? | p < 0.05 with Bonferroni correction if running multiple comparisons. |
| How to handle ties in pairwise? | Allow them; model them explicitly in Bradley-Terry or report tie rate. |
| What if κ is low? | Revise guidelines, add examples, adjudicate, and report pre- and post-adjudication scores. |

---

## Citations

This protocol synthesizes evidence from recent surveys on LLM agent evaluation [web:1][web:3], inter-annotator agreement methodology [web:10][web:6], rubric-based evaluation best practices [web:19][web:9], pairwise preference design [web:21][web:29], ACL/ARR reviewer standards [web:15], and systematic reproduction studies exposing flaws in prior human evaluations [web:22][web:26].
