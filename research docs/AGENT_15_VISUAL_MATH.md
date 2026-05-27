# Agent 15: Visual-Symbolic Math Reasoning Scout

## Mission Brief
This report investigates whether mathematical reasoning can be improved by treating equations, derivations, and symbolic layouts as **grounded visual structures** rather than purely textual sequences. The premise: mathematical notation is inherently spatial and hierarchical—operators have positional meaning, fractions are nested regions, and diagrams carry relational semantics that pure tokenization loses.

---

## 1. Current Literature

### 1.1 Multimodal Math Benchmarks

The evaluation landscape for visual math reasoning has matured rapidly. **MathVista** [web:7] is the foundational consolidated benchmark with 6,141 examples spanning 28 datasets plus 3 newly created ones (IQTest, FunctionQA, PaperQA). It tests logical reasoning on puzzle figures, algebraic reasoning over plots, and scientific reasoning with paper figures. GPT-4V achieves only 49.9% accuracy on MathVista, revealing significant room for improvement [web:7].

**MATH-Vision (MATH-V)** [web:3] advances this further with 3,040 contest-level problems across 16 topics and five difficulty levels, sourced from real math competitions. It addresses limitations in benchmark diversity that MathVista and others exhibited [web:4].

**MathVerse** [web:5] introduced a critical diagnostic angle: it evaluates whether MLLMs truly *see* diagrams or just exploit textual patterns. Using a Chain-of-Thought (CoT) evaluation strategy, MathVerse adaptively extracts reasoning steps and scores each with detailed error analysis, revealing intermediate reasoning quality rather than binary correctness [web:5].

**VisioMath** [web:33] (ICLR 2026) presents the first benchmark with image-based answer options—1,800 K-12 problems where candidate answers are diagrams with subtle visual similarities. State-of-the-art LMMs show consistent accuracy decline as inter-image similarity increases [web:33].

### 1.2 Visual-Symbolic Alignment and Grounding

Recent work establishes that alignment bridges perception and reasoning by connecting visual entities with symbolic or linguistic forms [web:21]. In mathematical contexts, this means linking geometric primitives, chart axes, table layouts, and equation regions with textual predicates or executable intermediates like geometry description languages, proof sketches, and program traces [web:21].

**GeoGen** aligns diagrams with executable programs under symbolic supervision [web:25]. **MathCoder-VL** uses code-based cross-modal supervision to reinforce visual-text alignment and program-level faithfulness [web:25]. **AlphaGeometry** integrates theorem libraries with neural search for geometric deductions [web:25].

**VisionMath** [web:36] proposes a three-stage progressive multimodal reasoning alignment strategy specifically for mathematical OCR, figure understanding, and problem solving, trained on rendered images with multilingual mathematical problems.

### 1.3 Equation Parsing and Operator Localization

Equation parsing—the task of mapping visual equation layouts to grounded symbolic representations—has roots in NLP but extends naturally to vision. Early work on equation parsing focused on identifying mathematical relations within sentences and grounding all variables correctly [web:6].

For visual grounding, **Emerging Localization Properties in Vision-Language Transformers** [web:16] shows that pretrained VL models allow zero-shot open-vocabulary object localization. The proposed **Grounding Everything Module (GEM)** generalizes value-value attention to self-self attention paths, enabling token-level localization without fine-tuning [web:19]. This has direct applicability to localizing operators, variables, and relation symbols in rendered equations.

### 1.4 Small-Model Math Reasoning

**Qwen2.5-Math** [web:34] demonstrated that even small models (1.5B-72B) can achieve strong math reasoning through specialized training. The 7B model scores 91.6 on GSM8K and 55.4 on MATH, outperforming much larger generalist models [web:34]. The 1.5B variant achieves MATH scores around 80 when using Python interpreter tool integration [web:42].

A 1.5B reasoning model (based on Qwen2.5-Math-1.5B) has been shown to beat DeepSeek R1 0120 on math assessments through intensive iteration and specialized training [web:38]. This demonstrates that small models *can* achieve strong reasoning with the right data and training regime.

**MathGenie** [web:23] introduced a back-translation pipeline: augment solutions first, then back-translate to questions. This produces diverse synthetic problems with verified code-integrated solutions. Models from 7B to 70B trained on this data achieve superior performance across five benchmarks [web:23].

### 1.5 Math Faithfulness and Step Correctness

Process-level verification is gaining traction. **Leanabell-Prover-V2** [web:22] integrates a Lean 4 verifier directly into the reasoning loop. Whenever the model generates a Lean snippet, it is immediately executed and verified. Feedback (successful proof-state advancement, type errors, tactic failures) is translated into reward signals for reinforcement learning [web:22].

Formal verification of generated proofs has been explored where modest-size (7B) models emit natural language reasoning *and* formal proof scripts simultaneously, using a theorem prover to verify each step [web:10]. Only fully validated reasoning chains receive high rewards, guiding models toward machine-checkable proofs.

**MathVerse**'s CoT evaluation strategy [web:5] represents a lighter-weight approach: using GPT-4(V) to extract and score each reasoning step with error analysis, revealing where models hallucinate or skip logic.

### 1.6 Diagram-Based and Chart-Based Math Reasoning

**ChartMuseum** [web:30] provides a focused investigation into chart understanding, showing that LVLMs' visual reasoning lags behind textual reasoning. A synthetic dataset solvable only through visual reasoning demonstrates that model performance degrades significantly with increasing visual complexity, while human performance remains robust [web:30].

**ChartBench** [web:31] offers a comprehensive evaluation benchmark for chart understanding, requiring precise numerical extraction from visual representations.

**GeoQA** [web:35] is a foundational geometric QA dataset with 5,010 problems and annotated programs illustrating solution processes. It is 25x larger than prior datasets and enables explicit, explainable numerical reasoning [web:35]. **GeoPQA** [web:39] proposes a two-stage reinforcement learning framework that first enhances visual perception on geometric problems, then fosters reasoning.

### 1.7 Synthetic Dataset Creation from LaTeX

**MathGenie** [web:23] established a viable pipeline for synthetic math data: iterative solution augmentation → question back-translation → verification-based solution filtering. This demonstrates that LaTeX-structured math content can be systematically expanded.

Synthetic data generation using LLMs has been surveyed extensively [web:15], showing that prompt-based generation, retrieval-augmented pipelines, and iterative self-refinement are viable for creating artificial but task-relevant examples—especially in low-resource or specialized domains.

Research on synthetic data for mathematical QA [web:9] shows that Gretel.ai-generated data can improve small language models (LLama-2-7B/13B, Mistral-7B) by approximately 2x accuracy on linear algebra benchmarks.

---

## 2. Real Gaps in the Literature

### Gap 1: No Unified Visual-Symbolic Grounding Framework
Current approaches treat equation parsing, diagram understanding, and symbolic reasoning as separate pipelines. There is no end-to-end system that takes a rendered equation image and grounds every token to its spatial bounding box while maintaining symbolic semantics. GEM [web:19] does zero-shot localization but not symbolic grounding. VisionMath [web:36] does OCR but not structural grounding.

### Gap 2: Missing Operator-Level Localization Benchmark
There is no benchmark that requires models to localize specific operators (+, −, ∫, ∑, etc.) within rendered equations and link them to their semantic roles. Existing benchmarks evaluate end-to-end correctness but not whether the model *knows where* the critical operations occur.

### Gap 3: Small Models Lack Visual-Symbolic Priors
Small math models like Qwen2.5-Math-1.5B [web:34] are text-dominant. When they do process visuals, they rely on generic vision encoders (CLIP-style) that were not pretrained on dense mathematical notation. The visual encoder has no prior for spatial syntax—fraction bars, subscript alignment, matrix delimiters.

### Gap 4: Faithfulness Evaluation Is Underdeveloped
MathVerse [web:5] evaluates CoT quality but requires GPT-4V as a judge, which is expensive and opaque. Lean verification [web:22] is rigorous but limited to formalizable mathematics. There is no scalable, automatic method to check whether a model's reasoning steps are *faithful* to the visual evidence in the problem.

### Gap 5: Synthetic Data Lacks Visual Grounding
Existing synthetic pipelines (MathGenie [web:23]) generate text and code but do not generate paired visual renderings with pixel-level annotations for operators, variables, and regions. LaTeX→image rendering is easy, but pixel-accurate bounding box annotation for symbols is labor-intensive and not automated at scale.

### Gap 6: Chart and Equation Domains Are Siloed
Chart reasoning [web:30] and equation reasoning [web:6] are studied separately. Real mathematical reasoning (e.g., reading a paper figure with an embedded equation, then computing from a table) requires unified processing of mixed visual-math content. No benchmark or model handles this seamlessly.

---

## 3. Opportunities Feasible in 8 Weeks

### Opportunity A: Rendered Equation Operator Localization Dataset (Weeks 1-4)
Create a synthetic dataset of 10K+ rendered equations from LaTeX with automatic operator localization. Use LaTeX compilation + PDF parsing (pdf2image + pdfplumber) to extract bounding boxes for every token. This yields pixel-accurate ground truth for operators, variables, and structural elements.

**Feasibility:** High. LaTeX token bounding boxes can be extracted via `synctex` or by parsing rendered PDFs. Symbol classes (+, −, ×, ÷, ∑, ∫, etc.) are enumerable.

### Opportunity B: Visual-Symbolic Alignment Probe (Weeks 3-6)
Fine-tune a small ViT (e.g., 86M parameters) on the operator localization dataset to learn visual-symbolic priors. Probe whether this improves downstream math reasoning on MathVista equation problems compared to a generic CLIP encoder.

**Feasibility:** High. Single-GPU training (your RTX 4070 8GB) is viable for small ViTs. Use LoRA or full fine-tuning on the localization task, then evaluate zero-shot or linear-probe on math QA.

### Opportunity C: Faithfulness Scoring via Step-Level Visual Verification (Weeks 5-8)
Implement an automatic faithfulness checker: given a model's CoT and the problem image, extract each claimed numerical value or operation, verify it appears in the image, and score step-by-step consistency. Use OCR + rule-based matching as a lightweight proxy for formal verification.

**Feasibility:** Medium. Does not require theorem provers. Can leverage existing OCR (DeepSeek-OCR [web:12], Qwen2.5-VL) and implement a rule-based consistency scorer.

### Opportunity D: Mini Benchmark of 200 Mixed Visual-Math Problems (Weeks 6-8)
Curate 200 problems that require both chart reading *and* equation interpretation (e.g., read values from a plot, plug into a formula). This fills Gap 6 and provides a focused testbed.

**Feasibility:** High. Source from existing datasets or manually create from textbook problems.

---

## 4. Datasets and Benchmarks

| Dataset | Size | Domain | Key Feature | Citation |
|---------|------|--------|-------------|----------|
| MathVista | 6,141 | Mixed (puzzles, plots, paper QA) | First consolidated visual math benchmark | [web:7] |
| MATH-V | 3,040 | Contest math (16 topics, 5 levels) | Real competition problems with visuals | [web:3] |
| MathVerse | ~2,600 | Visual math (6 domains) | CoT step-level evaluation | [web:5] |
| VisioMath | 1,800 | K-12 with image answers | Image-based answer options | [web:33] |
| GeoQA | 5,010 | Geometry | Annotated solution programs | [web:35] |
| ChartBench | Large | Charts and plots | Comprehensive chart evaluation | [web:31] |
| ChartMuseum | Synthetic | Chart visual reasoning | Visual-complexity-controlled synthetic | [web:30] |
| MathGenie data | Scalable | General math | Back-translated synthetic QA | [web:23] |
| VisionMath-OCR | Curated | Multilingual formula OCR | Progressive alignment training | [web:36] |
| PubTabNet | Large | Table recognition | HTML-structure + cell content | [web:40] |

**Recommended for 8-week project:** Combine MathVista equation subset + GeoQA diagram subset + synthetic operator-localization data for a focused study.

---

## 5. Auto-Annotation Opportunities

### 5.1 LaTeX-to-Bounding-Box Pipeline
LaTeX source contains structural information that maps directly to spatial layout. By compiling LaTeX to PDF and using `synctex` or font-level bounding box extraction, every token's pixel coordinates can be recovered automatically. This enables:
- **Symbol-class auto-labeling:** The LaTeX token class (e.g., `\sum`, `\frac`, `x`, `3`) is known at compile time.
- **Region auto-labeling:** Fractions, integrals, and matrices occupy computable bounding regions.
- **Relation auto-labeling:** Spatial adjacency (e.g., `dx` after `∫`) can be computed from relative positions.

### 5.2 Back-Translation with Visual Rendering
Extend MathGenie [web:23] by rendering each synthetic problem to an image and extracting operator bounding boxes. This yields triplets: `(image, LaTeX, bounding_boxes)` automatically at scale.

### 5.3 Synthetic Chart-Equation Pairs
Generate random function plots (matplotlib) with their equations overlaid. Auto-annotate which pixels correspond to the curve, axes, and equation text. This creates perfect ground truth for visual-symbolic alignment training.

### 5.4 Weak Supervision from MathJax Rendering
MathJax (web rendering) produces DOM elements with known structure. Screenshots + DOM inspection yield automatic bounding boxes for web-rendered math. Scraping Wikipedia, Math StackExchange, and arXiv abstracts yields millions of weakly supervised examples.

---

## 6. Realistic Paper Idea (8-Week Achievable)

### Title: *VizMath-1.5B: Visual Operator Grounding Improves Small-Model Math Reasoning*

**Core Claim:** A 1.5B-parameter multimodal model pretrained on operator-localization tasks achieves superior math reasoning on visual problems compared to text-only or generic-vision baselines, with gains concentrated on problems requiring spatial equation understanding.

**Method:**
1. Generate 50K synthetic equation images from LaTeX with auto-extracted operator bounding boxes (Section 5.1).
2. Pretrain a small ViT+LLM architecture (Qwen2.5-Math-1.5B base + tiny ViT) with two tasks: (a) operator localization (predict bounding boxes for query symbols), (b) masked equation completion (predict occluded tokens given image context).
3. Fine-tune on MathVista and GeoQA subsets.
4. Evaluate against the text-only Qwen2.5-Math-1.5B baseline and a CLIP-vision baseline.

**Expected Contribution:**
- First demonstration that explicit operator localization pretraining improves small-model visual math reasoning.
- Release of 50K synthetic equation images with bounding box annotations.
- Lightweight evaluation protocol for spatial math understanding.

**Target Venue:** ACL/EMNLP Short Paper, or NeurIPS Workshop on Mathematical Reasoning.

---

## 7. Ambitious Paper Idea

### Title: *Visual Symbolic Alignment: A Perception-Alignment-Reasoning Paradigm for Mathematical Multimodality*

**Core Claim:** Mathematical reasoning in MLLMs should be structured as three explicit stages: **Perception** (detect and localize all symbolic elements in the image), **Alignment** (construct a structured symbolic graph linking visual entities to semantic roles), and **Reasoning** (traverse the graph to derive the answer). This paradigm, implemented as a modular system with a differentiable alignment layer, achieves state-of-the-art on MATH-V, MathVerse, and VisioMath while providing interpretable, step-verifiable reasoning traces.

**Method:**
1. **Perception Module:** A detection transformer trained on the LaTeX-to-bounding-box pipeline (Section 5.1) outputs a set of symbolic entities with class labels and spatial coordinates.
2. **Alignment Module:** A graph neural network constructs a symbolic layout graph from detected entities, encoding spatial relations (above, below, inside, adjacent-to) as edge types. This graph is grounded in the visual evidence.
3. **Reasoning Module:** A small LLM operates over the symbolic graph (not raw pixels), performing step-by-step reasoning. Each step is traceable to specific visual entities.
4. **Verification Module:** A lightweight checker (not a full theorem prover) validates that each reasoning step uses only aligned entities and that intermediate computations are consistent with detected values.

**Expected Contribution:**
- A unified perception-alignment-reasoning framework specifically for math, inspired by [web:21] but implemented as a working system.
- Demonstration that small models (1.5B-7B) with explicit visual-symbolic alignment outperform larger black-box MLLMs on visual math benchmarks.
- Automatic annotation pipeline scaling to millions of examples from arXiv LaTeX sources.
- Open-source toolkit for visual math grounding.

**Target Venue:** NeurIPS / ICLR / CVPR (depending on emphasis: reasoning/vision/learning).

---

## 8. Final Recommendation

**Execute Opportunity A + B (realistic paper) immediately, with Opportunity C (faithfulness scoring) as an extension if time permits.**

**Rationale:**
- The realistic paper has a clear 8-week path: LaTeX rendering pipeline (2 weeks), small ViT training (3 weeks), evaluation (2 weeks), writeup (1 week).
- The operator localization dataset fills a concrete gap and is auto-annotatable, minimizing manual effort.
- Your background (embedded systems, Python, limited GPU) matches this scope. The RTX 4070 8GB is sufficient for 1.5B models and small ViTs.
- The ambitious paper idea (Section 7) should be framed as the *follow-up* paper, using the dataset and code from the 8-week sprint.

**Immediate Next Steps:**
1. Set up LaTeX auto-rendering pipeline with bounding box extraction (use `python-poppler` or `pdf2image` + `pymupdf`).
2. Curate symbol vocabulary (200 most common math operators/variables).
3. Start with 1K examples to validate pipeline, then scale to 50K.
4. Base model: Qwen2.5-Math-1.5B-Instruct or the open 1.5B weights.
5. Begin training scripts using LoRA on the vision encoder + full fine-tuning on the projector.

**Risk Mitigation:**
- If bounding box extraction proves noisy, switch to weak supervision: render equation *twice* (once clean, once with red boxes) and use difference masking.
- If GPU memory is insufficient for multimodal training, freeze the LLM and train only the vision encoder + adapter.
- If evaluation benchmarks are too hard, create a 500-example internal validation set of textbook problems with known difficulty.

---

*Report compiled by Agent 15 | Visual-Symbolic Math Reasoning Scout*
*Date: April 2026*
*Sources: 40+ papers and benchmarks surveyed*
