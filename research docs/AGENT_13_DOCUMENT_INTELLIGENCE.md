# AGENT 13: Document Intelligence and OCR-Grounded Reasoning Scout

**Date:** April 2026  
**Scope:** Scientific opportunities in document reasoning where layout, OCR, tables, charts, and diagrams matter. Directions selected for research depth, not just enterprise product hacks.

---

## 1. Key Literature

### End-to-End Document VLMs
- **Qwen2.5-VL** (3B / 7B / 72B) is the current standard for document understanding at multiple scales. The 3B model now outperforms last year’s 7B class, and the 7B variant competes with GPT-4o-mini on OCR and document parsing [web:31][web:34]. It supports bounding-box grounding, structured extraction from invoices/forms/tables, and chart/diagram analysis.
- **ColPali** (PaliGemma-3B) bypasses OCR entirely by producing multi-vector visual embeddings of document pages for retrieval, outperforming traditional OCR+tesseract pipelines while being simpler and end-to-end trainable [web:51][web:55].
- **OmniDocBench** establishes a rigorous parsing benchmark across nine document types (academic papers, textbooks, slides, handwritten notes, newspapers) with 19 layout categories and 14 attribute labels [web:22][web:26].

### Chart and Table Reasoning
- **ChartPaLI-5B** showed that transferring LLM reasoning capabilities into a 5B VLM via chart-to-table pre-training and synthesized reasoning traces achieves state-of-the-art on ChartQA without upstream OCR [web:12][web:15].
- **ChartQA-X** demonstrated that fine-tuning with explanation generation improves chart QA accuracy by up to 18.96 percentage points, suggesting reasoning traces are critical [web:38].
- **ChartQAPro** (2025) reveals that current LVLMs drop sharply on harder chart questions—Claude Sonnet 3.5 falls from 90.5% to 55.81%—proving chart reasoning is still brittle [web:42].
- **FaithSCAN** proposes lightweight single-pass hallucination detection for VQA by modeling token-level, visual, and cross-modal uncertainty [web:17].

### Diagram and Multi-Graph Reasoning
- Multi-graph reasoning benchmarks (flowcharts, knowledge graphs, mind maps, route maps) show mind maps are the most discriminative graph type for exposing model weaknesses [web:25][web:29].
- VLMs often take shortcuts on scientific charts, relying on text labels rather than visual elements for deeper reasoning [web:6][web:21].

### Faithfulness and Grounded QA
- **OfficeQA Pro** (2026) is a new enterprise benchmark for multi-document grounded reasoning spanning scanned documents to modern PDFs [web:4].
- Faithfulness hallucination in document-grounded QA remains unsolved; current approaches rely on external knowledge verification or LLM-as-a-Judge paradigms [web:16][web:20].

---

## 2. Gaps That Remain Open

| Gap | Why It Matters | Current Status |
|-----|---------------|--------------|
| **Chart visual reasoning without shortcuts** | Models read text labels instead of measuring bar heights, line slopes, or pie angles [web:6] | Open; ChartQAPro exposes severe drops [web:42] |
| **Faithful multi-page reasoning** | Cross-page evidence selection + numerical reasoning across tables is weak [web:4][web:36] | OfficeQA Pro and SlideVQA exist but solutions are nascent |
| **Hallucination detection in document VLMs** | No reliable single-pass method for detecting when models invent table cells or misread values [web:17] | FaithSCAN works for VQA but not tailored to document tables/charts |
| **Diagram → symbolic representation** | Converting flowcharts, circuit diagrams, or mind maps into executable or verifiable symbolic form is underexplored [web:25] | Mostly manual or pipeline-based |
| **Low-resource table understanding** | Small models (under 7B) struggle with complex table structures, merged cells, and multi-row headers | Open; Qwen2.5-VL-3B is promising but not extensively benchmarked on tables |
| **Screenshot/UI + document hybrid reasoning** | Screen parsing (OmniParser) and document parsing are separate communities [web:57][web:53] | No unified model handles both well |
| **OCR error robustness** | Document QA systems fail silently when OCR misreads a single digit; no explicit OCR-uncertainty modeling | Completely open |

---

## 3. Opportunities Fitting 8GB VRAM

With INT4 weight quantization (W4A16), a 7B–8B parameter VLM fits comfortably into 8GB VRAM [web:27][web:23]. The following directions are specifically viable for your RTX 4070 laptop:

### A. Small Model as Reasoning Guide
- Use Qwen2.5-VL-3B (or fine-tuned variant) as a fast "reasoning scout" that generates attention maps or candidate answers, then validates with a larger cloud model only when confidence is low [web:37].
- This mirrors the "Small VLM is a Precise Guidance" approach from CVPR 2025, where small model attention maps prune visual tokens for large VLMs.

### B. Edge Document VLM with OCR Uncertainty Gates
- Build a pipeline where Qwen2.5-VL-3B or Moondream2 first parses the document, then an auxiliary lightweight classifier flags low-confidence OCR regions.
- The model refrains from answering questions about uncertain regions unless the user explicitly confirms.
- This is novel because no existing system explicitly surfaces OCR confidence to the reasoning layer.

### C. Quantized Chart-to-Table Translator
- Fine-tune a 3B VLM (e.g., PaliGemma-3B or Qwen2.5-VL-3B) on chart-to-table conversion with INT4 quantization.
- The output is a structured table that can be fed to a tiny LLM (Phi-2 / 1B-class) for numerical reasoning.
- Separates visual parsing from symbolic reasoning, making each component tractable on edge hardware.

### D. Screenshot-to-Structured-Data for UI Automation
- Combine OmniParser-style screen parsing with document extraction.
- Target: extracting structured data from desktop application windows (forms, tables in Excel, SAP screens) using a 3B–7B VLM running locally.
- This bridges the UI understanding and document intelligence communities.

---

## 4. Datasets and Baselines

| Dataset | Task | Size | Notes |
|---------|------|------|-------|
| **OmniDocBench** | Document parsing evaluation | 9 document types, 19 layout categories | End-to-end extraction benchmark [web:22][web:26] |
| **DocVQA** | Single-page document QA | 12K+ images, 50K questions | Classic baseline for form/invoice QA [web:44] |
| **SlideVQA** | Multi-slide reasoning | 2.6K decks, 52K slides, 14.5K questions | Requires cross-page evidence selection [web:36][web:40] |
| **ChartQA / PlotQA / FigureQA** | Chart reasoning | ChartQA: 28K human-written + 20K synthetic | ChartQAPro is the harder 2025 variant [web:42] |
| **ChartQA-X** | Chart reasoning with explanations | Augmented ChartQA with gold explanations | Use for explanation-aware fine-tuning [web:38] |
| **OfficeQA Pro** | Multi-document grounded reasoning | Enterprise heterogeneous corpus | New 2026 benchmark [web:4] |
| **Multi-Graph VLM Benchmark** | Multi-graph joint reasoning | 4 graph types, homogeneous + heterogeneous | For diagram reasoning [web:25] |
| **ScreenSpot** | UI understanding / screen parsing | 600+ screenshots across platforms | For UI + document hybrid work [web:57] |

### Recommended Baselines
- **Vision Encoder + OCR:** Tesseract / PaddleOCR + lightweight LLM (baseline)
- **End-to-end VLM:** Qwen2.5-VL-3B or 7B (INT4 quantized)
- **Retrieval-first:** ColPali (PaliGemma-3B) for page-level document retrieval
- **Small VLM:** Moondream2 (1.8B, <2GB RAM) for extreme edge cases [web:7]

---

## 5. Human Evaluation Options

| Method | What It Measures | Effort |
|--------|-----------------|--------|
| **Point-to-answer grounding** | Does the model highlight the correct region for its answer? | Medium: requires bounding box annotation or click-data |
| **Explanation quality rating** | Are chart reasoning explanations logically coherent and grounded? | Medium: Likert scale on ChartQA-X style outputs [web:38] |
| **Hallucination detection task** | Human annotators flag answers that invent values not in the document | Low: binary flag per answer |
| **Cross-page reasoning trace** | For SlideVQA, does the model select the correct evidence slides? | High: requires multi-page annotation |
| **OCR error sensitivity** | Intentionally corrupt single characters/digits; measure answer change | Low: automated injection + human verification of severity |
| **Side-by-side preference** | Compare model outputs against human or GPT-4o reference | Medium: standard pairwise preference test |

---

## 6. One Realistic Paper

### Title: "OCR-Gated Edge Document VLM: Refusing to Hallucinate by Surfacing Uncertainty"

**Problem:** Document VLMs on edge devices (3B–7B, INT4 quantized) silently hallucinate answers when OCR misreads table cells, especially for numerical values. Existing hallucination detectors (FaithSCAN, etc.) work for general VQA but not for structured document content.

**Method:**
1. Use Qwen2.5-VL-3B (INT4) as the base document parser.
2. Add a lightweight per-token confidence head trained to predict OCR reliability for text regions.
3. During QA, if a question targets a low-confidence OCR region, the model outputs "OCR uncertain — please verify [region]" instead of fabricating an answer.
4. Evaluate on DocVQA + OmniDocBench with injected OCR noise.

**Why it fits 8GB VRAM:** Qwen2.5-VL-3B in INT4 is ~1.7GB; the confidence head adds negligible parameters.

**Expected contribution:** First edge-deployable document VLM with explicit OCR-uncertainty-aware abstention.

---

## 7. One Novel High-Risk Paper

### Title: "Chart-to-Program: Synthesizing Executable Visual Programs from Charts for Verifiable Reasoning"

**Problem:** Current VLMs answer chart questions by pattern matching or label reading, which is unverifiable and prone to shortcuts [web:6][web:42]. There is no mechanism to "show work" for chart reasoning the way chain-of-thought shows work for math.

**Method:**
1. Train a small VLM (3B–5B) to parse chart images into an intermediate visual program representation (e.g., "read_bar(blue) → 45", "read_bar(red) → 30", "subtract(45, 30) → 15").
2. The program is executed by a tiny deterministic interpreter (not learned).
3. For answer generation, the model either (a) executes the program or (b) falls back to direct generation if parsing fails, with the program trace available for verification.
4. Evaluate on ChartQAPro and a new synthetic dataset where programs can be mechanically verified against chart ground truth.

**Risk factors:**
- May fail on chart types not in training distribution.
- Program representation design is non-trivial.
- If the parser is too brittle, fallback dominates and value is lost.

**Upside:** If it works, it establishes a verifiable reasoning paradigm for visual documents—something no current VLM offers.

---

## 8. Verdict Table

| Direction | Scientific Interest | 8GB Feasibility | Dataset Availability | Novelty | Risk | Verdict |
|-----------|--------------------|-------------------|---------------------|---------|------|---------|
| OCR-gated edge document VLM | High | ✅ 3B INT4 fits easily | ✅ DocVQA, OmniDocBench | Medium (first edge abstention) | Low | **Strong do** |
| Chart-to-program synthesis | Very High | ✅ 3B parser + tiny interpreter | ⚠️ Requires synthetic program data | Very High | High | **High-risk bet** |
| Small VLM as reasoning guide | Medium | ✅ Already proven [web:37] | ✅ Existing benchmarks | Low | Low | Incremental |
| Multi-graph reasoning VLM | High | ⚠️ 7B may need careful tuning | ✅ New benchmark exists [web:25] | Medium | Medium | Good but crowded |
| Screenshot + document hybrid | Medium | ✅ 3B–7B range | ✅ ScreenSpot available | Medium | Medium | Bridge gap, less foundational |
| Faithfulness detector for tables | High | ✅ FaithSCAN-style is small | ✅ Can annotate from DocVQA | Medium | Low | **Strong do** |
| Quantized chart-to-table translator | Medium | ✅ 3B + 1B LLM | ✅ ChartQA/PlotQA | Medium | Low | Useful engineering, less science |

---

## Closing Notes

The most scientifically interesting directions are:
1. **OCR-gated abstention** — immediately useful, edge-viable, and fills a real gap in faithfulness.
2. **Chart-to-program synthesis** — high risk, but if solved, changes how we think about verifiable visual reasoning.

Both fit your 8GB RTX 4070 constraint with INT4 quantization and 3B–7B parameter models.
