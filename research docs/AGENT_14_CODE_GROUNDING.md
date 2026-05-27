# AGENT_14_CODE_GROUNDING.md
# Code as Multimodal Structure: Grounded & Visual-Structural Reasoning

**Agent:** 14 — Code as Multimodal Structure Researcher  
**Date:** April 25, 2026  
**Mission:** Investigate whether code reasoning can be reframed as a grounded or visual-structural reasoning problem — using AST visualization, semantic region localization, graph/image views, and multimodal understanding — with critical skepticism about shallow analogies.

---

## 1. Literature Review

### 1.1 Foundational: Code as Graph / AST Structure

**code2vec (Alon et al., 2019)**  
Represented code as bags of AST path-contexts, mapped to embeddings. Showed that structural paths in ASTs capture semantic meaning (e.g., predicting method names). Still a baseline for structural code representations. Weakness: bag-of-paths loses ordering and hierarchical context.

**code2seq (Alon et al., 2019)**  
Extended code2vec to sequence generation via attention over AST paths. Encoder-decoder with path-level attention. Demonstrated that AST structure + sequential decoding outperforms flat-token approaches on docstring/method-name generation.

**GraphCode2Vec (Ma et al., 2022)**  
First self-supervised GNN-based code embedding using both lexical features and program dependence graphs (PDG). Task-agnostic pretraining; outperformed both generic and task-specific baselines. Establishes the GNN-on-program-graph paradigm as viable for code classification and clone detection.

**GNN-based Code Classification (Sci. Direct, 2023 systematic review)**  
Systematic investigation of GNN architectures for code classification. Key finding: GNNs capture message-passing over CFG/PDG/AST structures and generally improve accuracy, but the gains are not consistent — graph construction choices dominate performance, not the GNN architecture itself.

### 1.2 AST-Guided Generation

**TreeDiff (arXiv 2025-08)**  
First work to integrate AST-aware masking into large language diffusion models for code generation. Uses span-level AST-guided masking (vs. random token masking) during training to encourage structural recovery. Trained on 150K code reasoning samples. Segments prompts into reasoning and solution spans with semantically aligned corruption strategies. Directly validates that structural priors (AST) are useful for diffusion-based code LLMs.

### 1.3 Screenshot-to-Code / Visual Grounding in Code

**VisRefiner (arXiv 2026-02)**  
Framework that learns from *visual differences* between rendered UI outputs and target screenshots. Converts screenshot-to-code into a difference-driven learning paradigm — the model sees what its code produces visually and learns from the delta. Uses VisDiffUI dataset (derived from Vision2UI and WebSight). Key contribution: grounding code generation in pixel-level visual feedback. Advances multimodal LLMs toward human-like reasoning in structured code synthesis.

**ScreenCoder (OpenReview 2025-12)**  
Modular multi-agent framework for converting UI screenshots/design sketches to front-end HTML/CSS. State-of-the-art on screenshot-to-code benchmarks. Validates that modular visual reasoning over UI structure outperforms monolithic end-to-end approaches.

**GeoTikzBridge (arXiv 2026-03)**  
Framework for multimodal code generation for geometric diagrams using TikZ. Enhances local geometric perception and visual reasoning for generating structured diagram code. Demonstrates that geometric visual grounding can be bound to code structure (TikZ commands) — a defensible experiment in visual-structural code generation.

### 1.4 Fault Localization with Grounding

**SemLoc (arXiv 2026-03, Jerry Yang et al.)**  
Most directly relevant recent paper to "code grounding." SemLoc converts free-form LLM semantic reasoning into a *closed intermediate representation* that binds each inferred property to a specific, typed program anchor — enabling runtime checking and attribution to program structure. Builds a "semantic violation spectrum" (constraint-by-test violation matrix) analogous to coverage-based fault localization. Evaluated on SemFault-250 (250 Python programs with single semantic faults from real repos). Results: Top-1 accuracy 42.8%, Top-3 68%, inspection narrowed to 7.6% of executable lines. Counterfactual verification adds 12% further accuracy gain. This is a strong, principled grounding approach — not a shallow analogy.

**LLM-Generated Failure Explanations (arXiv 2026-04)**  
Systematic study of context effects on LLM debugging explanations across 93 configurations on real open-source bugs. Uses gpt-5-mini, DeepSeek-V3.2, and Grok-4.1-fast. Evaluates faithfulness and actionability along 6 criteria (problem ID, causal-chain clarity, actionability, brevity, code reference grounding). Key finding: explicit code references (line numbers, function names) in explanations significantly improve grounding quality — "artifact grounding" is measurable and causal.

### 1.5 Graph Reasoning Benchmarks Applied to Code

**GraphEval36K (NAACL 2025)**  
First comprehensive benchmark for evaluating LLM graph-solving capabilities through coding problems: 40 graph coding problems, 36,900 test cases. Benchmarked 10 LLMs. Private models outperform open-source, but gap is narrowing. Key finding: graph coding is a distinct skill separable from general coding — validates the graph-structure framing.

**Program Synthesis for Visual Programming (ACL 2025)**  
Benchmark based on XLogoOnline — a visual programming environment where programs operate on visual grids. Tests LLMs/multimodal models on synthesizing code that achieves visual goals in a grid world. Directly grounds code synthesis in visual execution semantics.

### 1.6 Multimodal Structural Reasoning (Adjacent)

**SpatialRGPT (NeurIPS 2024)**  
Adds spatial grounding and depth information to VLMs via 3D scene graph data curation. While not code-specific, establishes the "region-aware spatial reasoning" paradigm that can be transplanted to code layout (code regions → semantic regions in spatial layout).

**MVoT — Multimodal Visualization of Thought (ICLR 2025)**  
Enables LLMs to generate image visualizations as part of reasoning chains. Token discrepancy loss improves visual coherence. Relevant because it shows that generating intermediate structural representations (images) improves reasoning on spatial tasks — hypothesis: the same could apply to generating AST/CFG visualizations during code reasoning chains.

---

## 2. Saturated vs. Underexplored Areas

### Saturated (Well-Worked)
| Area | Status | Key Signal |
|---|---|---|
| AST-path embeddings (code2vec family) | Saturated | 2018-era, widely replicated |
| GNN on CFG/PDG for code classification | Saturated | Systematic review 2023 shows diminishing returns |
| Screenshot-to-code (UI reproduction) | Maturing fast | ScreenCoder, VisRefiner both 2025-2026, high competition |
| LLM code generation benchmarks | Oversaturated | HumanEval, MBPP, SWE-bench all exist |
| Code clone detection via embeddings | Saturated | GraphCode2Vec era closed this |

### Underexplored (Genuine Gaps)
| Area | Status | Why Interesting |
|---|---|---|
| Semantic region localization in code (à la SemLoc) | Nascent (2026) | Grounding LLM inferences to typed program anchors is new |
| AST-guided diffusion for code (TreeDiff) | Very new (Aug 2025) | Only one paper; many variants unexplored |
| Visual execution feedback for code debugging | Nascent | VisRefiner does it for UI; no equivalent for general code |
| Code-as-image for small model reasoning | Unexplored | No paper treats rendered code as an image for VLMs |
| Mechanistic explanation of code LLM failures | Emerging | LLM failure explanation paper (Apr 2026) is early work |
| Grounded program synthesis in visual grid worlds | Nascent | ACL 2025 XLogoOnline is a single paper |
| Counterfactual constraint verification for debugging | Very new | SemLoc introduces it; zero follow-up yet |
| Multi-hop graph reasoning over program dependence | Underexplored | GNN-RAG shows multi-hop gains for KG; untried for PDGs |

---

## 3. Five Paper Ideas

### Paper Idea 1: SemanticAnchorBench — Grounding Code LLM Reasoning to Typed Program Regions

**Thesis:** LLMs reason about code in free-form prose; we can measure and improve how well their natural language reasoning is *anchored* to specific, typed program structures (line ranges, AST node types, variable scopes, control flow regions).

**Method:**
- Define a taxonomy of "program anchors": {variable declaration, loop body, branch condition, function return, exception handler, data dependency edge}
- For each LLM output (explanation, fix, bug diagnosis), automatically extract cited program structures using static analysis
- Score anchor precision/recall against ground-truth fault sites
- Fine-tune small models (1-3B) using anchor-supervised SFT + DPO

**What's new:** Extends SemLoc beyond fault localization to general code reasoning. Introduces the first anchor-grounding metric for LLM code outputs.

**Datasets:** SemFault-250 (existing), Defects4J, SWE-bench lite, synthetic anchor-labeled programs (see §4)

**Expected novelty:** High — no existing work measures anchor precision systematically for general code reasoning.

---

### Paper Idea 2: ASTDiff-Small — Distilling AST-Guided Diffusion Code Reasoning to Sub-1B Models

**Thesis:** TreeDiff (2025) shows AST-guided masking improves code reasoning in large diffusion LLMs. We ask: can this structural prior be distilled into small (125M–1B parameter) models that run locally?

**Method:**
- Train teacher TreeDiff-7B on 150K samples (TreeDiff setup)
- Distill to student 350M model using AST-span-aligned KL loss (structural knowledge distillation)
- Evaluate on HumanEval, MBPP, CodeContests, and proposed ASTReason-5K benchmark
- Compare to flat-token distillation (standard KD) and random-masking student

**What's new:** First structural-prior-aware distillation for code; demonstrates whether AST bias survives compression.

**Skepticism check:** The analogy to image distillation is defensible because AST spans are explicit, discrete, and checkable — not a vague "code looks like an image" claim.

**Expected novelty:** Medium-high — structural KD for code is unstudied.

---

### Paper Idea 3: CodeRegionVLM — Visual Semantic Region Localization in Code Snapshots

**Thesis:** When code is *rendered* as a formatted text image (syntax-highlighted, line-numbered, IDE screenshot), can a VLM perform "semantic region localization" — drawing bounding boxes around buggy regions, unused variables, off-by-one loops — analogous to object detection in natural images?

**Method:**
- Dataset: Render 10K Python/JavaScript programs as syntax-highlighted images; annotate bounding boxes for semantic fault regions (using SemFault-250 + Defects4J ground truth)
- Fine-tune a region-aware VLM (e.g., Qwen2.5-VL or R-VLM adapted from GUI grounding) on code-as-image with region labels
- Evaluate region IoU accuracy vs. text-only LLM fault localization

**Skepticism test:** This IS a "code is like an image" analogy — justified only if VLM spatial attention gives qualitatively different error patterns than text attention. We explicitly test this via attention map analysis.

**Why defensible:** GUI grounding works (R-VLM, WinSpot); code IDE screenshots are visually similar to GUIs; line/column coordinates are ground-truthable.

**Expected novelty:** High — no paper has framed code fault localization as visual region detection.

---

### Paper Idea 4: ExecVisDiff — Execution Trace Visualization for Code Debugging with Multimodal Grounding

**Thesis:** Inspired by VisRefiner's visual feedback loop for UI code, we apply *execution trace visualization* as a grounding signal for code debugging. The model sees a rendered diff between expected and actual program state (variable tables, call stack renders, output diffs) and must localize the causal fault.

**Method:**
- For each buggy program, render: (a) expected output, (b) actual output, (c) intermediate variable states as color-coded tables
- Frame this as a visual grounding problem: given rendered execution states, identify the first divergence point
- Fine-tune a multimodal model on (rendered_trace, fault_location) pairs
- Compare to text-only trace analysis (standard LLM debugging) and SemLoc

**Key experiment:** Does visual rendering of execution state provide information not captured by text-only trace? (Ablation: strip images → text-only LLM).

**Datasets:** Synthetic trace-visualization dataset generated from Defects4J + Python buggy programs (see §4).

**Expected novelty:** High — visual execution feedback for debugging has no prior work.

---

### Paper Idea 5: PDG-RAG — Multi-Hop Program Dependence Graph Retrieval for Augmented Code Reasoning

**Thesis:** GNN-RAG shows 8.9–15.5% gains on multi-hop knowledge graph QA by using GNNs to retrieve relevant subgraphs. The same principle applies to program dependence graphs (PDGs): for complex bug queries, retrieve relevant subgraphs (data dependencies, control flow paths) via GNN, verbalize them, and feed to LLM.

**Method:**
- Build PDG for each program using static analysis (joern, tree-sitter)
- Train a GNN (GraphSAGE or GAT) to score PDG subgraph relevance given a query (bug report / test failure description)
- Extract verbalized PDG paths as context for LLM fault localization
- Evaluate on SWE-bench, SemFault-250, Defects4J Top-1/Top-3 fault localization

**What's new:** Applies GNN-RAG paradigm to code; first graph-retrieval-augmented debugger.

**Expected novelty:** Medium — conceptually clear transfer from KG-RAG, but program dependence graphs have different properties than knowledge graphs (cyclic, typed edges, dynamic data dependencies).

---

## 4. Datasets and Synthetic Data Generation Plan

### Existing Datasets
| Dataset | Size | Task | Notes |
|---|---|---|---|
| SemFault-250 | 250 programs | Fault localization | Python, single semantic faults |
| Defects4J | 835 real bugs | Fault localization | Java, ground-truth patches |
| SWE-bench (lite) | 300 issues | End-to-end repair | GitHub issues → patches |
| GraphEval36K | 36,900 test cases | Graph coding | LLM graph algorithm coding |
| XLogoOnline (ACL 2025) | ~500 tasks | Visual program synthesis | Grid-world visual grounding |
| VisDiffUI | ~10K pairs | Screenshot-to-code | UI visual difference |
| TreeDiff training set | 150K samples | Code reasoning | AST-structured code reasoning |

### Synthetic Data Generation Plan

**Step 1: AST-Anchor Labeled Programs**
- Use tree-sitter to parse 50K Python programs from The Stack (HuggingFace)
- Extract all AST node types and their character/line spans
- Inject synthetic faults (mutation operators: variable swap, off-by-one, type error) at known AST locations
- Label: (program, fault_AST_node_type, fault_line_range, natural_language_description)
- Output: ~200K (program, anchor) pairs for SemanticAnchorBench

**Step 2: Rendered Code Images for CodeRegionVLM**
- Render each program using pygments (syntax highlighting) + pillow to 1024x768 images
- Add line numbers, IDE-style gutters (dark theme)
- Annotate bounding boxes using pixel coordinates from line/character spans
- Output: 50K (image, bbox_annotations) pairs

**Step 3: Execution Trace Visualization Dataset**
- Run buggy programs from Defects4J/SemFault with Python `sys.settrace`
- At each function call/return, capture variable state snapshots
- Render snapshots as color-coded HTML tables → screenshot via selenium headless
- Render diffs between expected and actual state using visual diff
- Output: 20K (trace_image_sequence, fault_line) triplets

**Step 4: PDG Corpus**
- Use joern to build PDGs for all programs in SemFault + Defects4J
- Export as NetworkX graphs with typed edges (data dep, control dep, call)
- Store as JSON graph + verbalized path descriptions
- Output: 5K PDGs with traversal-verbalization pairs

**Step 5: Data Augmentation**
- Cross-language: translate Python programs to JavaScript using LLM + AST matching
- Difficulty stratification: classify programs by cyclomatic complexity, nesting depth
- Noise injection: add confounding correct code blocks to test localization precision

---

## 5. Baselines and Evaluation

### Baselines

| Baseline | Method | Why Include |
|---|---|---|
| GPT-4o (text-only) | Prompt with full code text | Upper bound for text-only LLMs |
| SemLoc | Semantic violation spectrum | Best existing grounding method |
| CodeBERT + fine-tune | Pretrained token-level model | Standard NLP-for-code baseline |
| GNNs on AST/CFG | GCN/GAT on program graph | Structural deep learning baseline |
| Coverage-based FL (GZoltar) | Ochiai/Tarantula spectra | Classical fault localization |
| Random | Random line ranking | Sanity check |

### Metrics by Task

**Fault Localization:**
- Top-1, Top-3, Top-5 accuracy (% of bugs where correct line is in top-k)
- Mean First Rank (MFR), Mean Average Rank (MAR)
- Inspection ratio (% of lines inspected before finding fault)

**Code Region Grounding (Visual):**
- Region IoU (bounding box overlap with ground-truth fault region)
- Precision/Recall at IoU threshold 0.5
- Anchor precision (% of cited anchors that are correct program structures)

**Code Generation Quality:**
- Pass@1, Pass@10 (HumanEval/MBPP standard)
- Structural fidelity: AST edit distance from reference
- Visual fidelity for screenshot-to-code: pixel-level SSIM, layout IoU

**Small Model Evaluation:**
- Parameters: 125M, 350M, 1B, 3B
- Benchmark: HumanEval, MBPP, ASTReason-5K, CodeContests
- Inference latency on CPU (laptop-class hardware: 8GB RAM)

---

## 6. Tier-1 Viability Assessment

### Arguments FOR Tier-1 Viability

1. **SemLoc (2026) proves the core thesis is non-trivial and works.** Grounding LLM reasoning to typed program anchors gives 42.8% Top-1 fault localization — substantially better than all baselines, including LLM-only approaches. This is real signal, not shallow analogy.

2. **VisRefiner (2026) proves visual feedback loops for code are experimentally defensible.** Rendering code output as images and learning from visual differences is not a metaphor — it's a working system with measurable SSIM gains.

3. **The graph/structural view has not been exhausted.** While GNN-on-AST for classification is mature, PDG-RAG for multi-hop debugging, and AST-guided diffusion for small models are genuinely open.

4. **Convergent evidence from adjacent fields is strong.** GUI grounding (R-VLM, WinSpot, ScreenSpot-Pro at 61.6% via visual scaling) shows that "spatial region localization in structured UI" works at scale — code IDE views are a natural next domain.

5. **Small-model angle is underexplored and practically important.** If AST structural priors survive compression (as image inductive biases survive to small CNNs), this enables local deployment — directly relevant to TinyML/on-device reasoning.

### Arguments AGAINST Tier-1 Viability

1. **"Code is like an image" is still mostly metaphor for reasoning tasks.** Outside of UI-reproduction (screenshot-to-code), treating code as a rendered image for *logic* tasks (bug finding, correctness) has no published working system. The visual signal may be redundant with text.

2. **LLMs already read line numbers in text.** The main advantage of visual region localization is spatial attention — but code is 1D sequential. Spatial 2D processing may not add information over text with line/column markers.

3. **Data is hard.** Execution trace rendering + annotation is expensive; fault bounding box annotation requires expert knowledge. Scale is limited compared to natural language datasets.

4. **Benchmark saturation risk.** HumanEval, MBPP are overfit; new benchmarks (ASTReason-5K, SemFault-250) are small-scale and may not generalize.

5. **GNN code models have not demonstrated gains on LLM-scale tasks.** All GNN code successes are at classification/clone-detection scale — not at open-ended generation or complex debugging where LLMs dominate.

### Verdict: **CONDITIONALLY Tier-1**

The field is Tier-1 viable **iff** the research focuses on:
- Grounding as a *measurement and constraint* mechanism (SemLoc direction), NOT as a vague "visual similarity" claim
- Structural priors that are *checkable* (AST spans, PDG paths, typed anchors) rather than aesthetic (rendered images for logic)
- Small-model applications where structural priors provide compute-efficiency gains that justify the complexity

It is **NOT Tier-1** if pursued as: "we render code as an image and pass it to a VLM, therefore it's multimodal reasoning." That requires a defensible ablation showing VLM spatial attention captures genuinely different information than text-based attention.

---

## 7. Best Recommendation

### Primary Recommendation: Paper Idea 1 + Paper Idea 4 (Combined Arc)

**The strongest research program** is to build on SemLoc's semantic grounding framework and extend it toward execution-trace-visual feedback — creating a two-paper arc:

**Paper A (SemanticAnchorBench):** Define and evaluate anchor grounding as a general metric for code LLM reasoning. This is immediately publishable, defensible, and fills a real gap. Estimated scope: 3-4 months, 1-2 researchers.

**Paper B (ExecVisDiff):** Use visual execution trace rendering as a grounding signal for debugging — the first system that lets a multimodal model "see" program state divergence and localize faults. This requires the synthetic dataset pipeline from §4 Step 3. Estimated scope: 5-6 months, 2-3 researchers.

### Why This Arc?
1. Paper A provides the evaluation framework that Paper B is evaluated against — clean cumulative contribution.
2. SemLoc (March 2026) is the most directly relevant prior work and has zero published follow-up — the window is open.
3. Execution trace visualization is genuinely novel; the "visual feedback" idea is proven in VisRefiner for UI but untested for logic/debugging.
4. Both papers have clear, skepticism-resistant ablations: strip the visual/grounding component and measure degradation.

### Secondary Recommendation: Paper Idea 5 (PDG-RAG)

If the primary arc is too resource-intensive, **PDG-RAG** is the highest risk-adjusted idea: it's a clean transfer of a working paradigm (GNN-RAG, which has demonstrated 8.9–15.5% multi-hop gains on KGQA) to program dependence graphs. The main risk is that PDGs have fundamentally different topology than knowledge graphs (denser, cyclic, semantically richer edges), which could either be a problem or a paper contribution in itself.

### What to Avoid
- Avoid building a system purely on code-as-rendered-image without a text-only ablation that proves the visual channel adds signal.
- Avoid re-implementing GNN-on-AST for classification — that space is thoroughly mined.
- Avoid benchmarking only on HumanEval/MBPP — use SemFault-250, Defects4J, and SWE-bench for credibility.

---

## References (Key Papers)

1. Alon et al. (2019). *code2vec: Learning Distributed Representations of Code.* POPL 2019.
2. Alon et al. (2019). *code2seq: Generating Sequences from Structured Representations of Code.* ICLR 2019.
3. Ma et al. (2022). *GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence.* MSR 2022.
4. Anonymous (2025). *TreeDiff: AST-Guided Code Generation with Diffusion LLMs.* arXiv:2508.01473.
5. Anonymous (2026). *VisRefiner: Learning from Visual Differences for Screenshot-to-Code Generation.* arXiv:2602.05998.
6. Yang et al. (2026). *SemLoc: Structured Grounding of Free-Form LLM Reasoning into Typed Program Anchors.* arXiv:2603.29109.
7. Anonymous (2026). *Evaluating Faithful, Actionable LLM-Generated Failure Explanations.* arXiv:2604.18309.
8. Luo et al. (2025). *Visual Test-time Scaling for GUI Agent Grounding.* ICCV 2025.
9. Anonymous (2025). *Program Synthesis Benchmark for Visual Programming in XLogoOnline.* ACL 2025.
10. Anonymous (2025). *GraphEval36K: Benchmarking Coding and Reasoning Capabilities of LLMs via Graph Problems.* NAACL 2025.
11. Jin et al. (2024). *GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning.* arXiv:2405.20139.
12. Wang et al. (2024). *SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models.* NeurIPS 2024.
13. Anonymous (2025). *MVoT: Multimodal Visualization-of-Thought.* ICLR 2025.
14. Anonymous (2025). *GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Diagrams.* arXiv:2603.22687.
15. Anonymous (2025). *ScreenCoder: Advancing Visual-to-Code Generation via Modular Multimodal Agents.* OpenReview 2025.
