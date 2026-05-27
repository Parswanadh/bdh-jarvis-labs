# AGENT 28: Multimodal Search and Retrieval Scout

## Executive Summary

Multimodal retrieval—searching across text, images, video, and structured documents using multimodal queries—is a rapidly maturing field with significant underexplored opportunities. While benchmarks like **M-BEIR** [web:16], **MM-BRIGHT** [web:21], and **MMDocRAG** [web:22] have standardized evaluation, critical gaps remain in **screenshot-conditioned search**, **low-resource pipelines**, **visual evidence discovery**, and **retrieval usefulness evaluation**. This report identifies 5 candidate research ideas, ranks them by feasibility and impact, and outlines realistic and high-risk paper paths.

---

## 1. State of the Field

### 1.1 Core Benchmarks

| Benchmark | Scale | Focus | Key Metric |
|-----------|-------|-------|------------|
| **M-BEIR** [web:16] | 1.5M queries, 5.6M candidates | Universal instructed retrieval across 8 tasks | Recall@K, nDCG |
| **MM-BRIGHT** [web:21] | 2,803 queries, 29 domains | Reasoning-intensive retrieval (diagrams, charts, screenshots) | nDCG@10 |
| **MMDocRAG** [web:22] | 4,055 QA pairs | Multi-page, cross-modal evidence chains | Evidence selection accuracy |
| **MultiHaystack** [web:29] | 46K candidates, 747 questions | Large-scale cross-modal retrieval + reasoning | Recall@5, task accuracy |
| **VIRA / UniSE** [web:32] | 13M screenshots | Screenshot retrieval and search | SR Recall@K |

### 1.2 Key Technical Trends

- **Universal retrieval models**: Systems like **UniIR** [web:17] and **MM-Embed** [web:20] aim to handle any combination of text/image queries and candidates through instruction tuning and multi-task training.
- **Generative retrieval**: Amazon's **GENIUS** [web:3] uses generative models to produce candidate identifiers directly, then reranks with embeddings—improving efficiency while closing accuracy gaps.
- **Late-interaction multimodal encoders**: Models like **ColQwen2**, **ColPali**, and **GME** [page:2] encode documents at the token/patch level, enabling fine-grained matching without full cross-attention at query time.
- **Reasoning-enhanced retrieval**: **MARVEL** [web:12] achieved a nDCG@10 of **37.9** on MM-BRIGHT (+10.3 over prior SOTA) by combining query expansion, reasoning, and step-by-step reranking.

### 1.3 Persistent Challenges

1. **Modality bias**: VLMs often retrieve visually salient images instead of semantically correct evidence, especially for temporal or factual queries [web:29].
2. **OCR confounding**: A recent study shows that improving OCR alone can recover **+8.9 to +31.1 Top-5 points** for BM25 on visually rich benchmarks, suggesting many "multimodal gaps" are actually transcription gaps [page:2].
3. **Cross-modal reasoning**: Even the best multimodal model (Nomic-Vision) achieves only **27.6 nDCG@10** on MM-BRIGHT's multimodal-to-text task—underperforming the best text-only model (DiVeR: 32.2) [web:24].
4. **Evaluation fragmentation**: No single benchmark measures whether retrieval actually helps downstream reasoning; most evaluate retrieval in isolation.

---

## 2. Missing Benchmarks and Tasks

### 2.1 Critical Gaps

1. **Screenshot-as-Query Search**: Real users search with screenshots (error messages, UI states, diagrams). Existing work (**UniSE** [web:40]) covers 7 categories but lacks a standardized evaluation suite with difficulty grading.
2. **Low-Resource Multimodal Retrieval**: Most benchmarks assume high-quality English text + images. There is no benchmark for multimodal retrieval in low-resource languages with limited OCR quality or training data [web:13][web:25].
3. **Visual Evidence Discovery**: No benchmark explicitly tests "find the figure/table that supports this claim" in an open corpus. MMDocRAG [web:22] is document-level; we need corpus-level visual evidence retrieval.
4. **Retrieval Usefulness Evaluation**: Current metrics (Recall, nDCG) measure correctness but not whether retrieved content actually improves downstream reasoning or generation quality [web:38].
5. **Decomposed Retrieval Benchmarks**: As shown in [page:2], OCR, preprocessing, and retrieval are confounded. The field lacks benchmarks that isolate these components.
6. **Diagram/Image-to-Document Retrieval**: Searching for documents (PDFs, papers, manuals) given a diagram query remains almost entirely unexplored.

### 2.2 Why These Gaps Matter

Real-world tasks begin with search. A user takes a screenshot of an error, a researcher sketches a diagram, or an engineer photographs a component—then needs relevant documents, not just answers. Current systems force these into text queries through manual description, losing critical visual information. Benchmarks that close this loop would unlock both academic and commercial value.

---

## 3. Five Candidate Research Ideas

### Idea 1: Screenshot-Conditioned Document Retrieval Benchmark (SCRBENCH)
**Concept**: A benchmark where queries are screenshots (UI states, error messages, charts, diagrams) and candidates are PDFs, web pages, or technical manuals. Models must retrieve the document that explains or matches the screenshot.
**Novelty**: Extends UniSE/VIRA [web:32] from page-level screenshot retrieval to full document retrieval. Tests whether systems can match visual state → textual explanation.
**Feasibility**: HIGH. Can curate from existing datasets (VisR-Bench, MMDocRAG, arXiv figures) with synthetic query generation using VLMs.

### Idea 2: Low-Resource Multimodal Retrieval Pipeline (LR-MR)
**Concept**: A retrieval system built entirely from small, open components (SigLIP-Base, Phi-3, quantized encoders) that runs on CPU/edge devices, evaluated on a new low-resource benchmark.
**Novelty**: Most work focuses on scaling up; this focuses on scaling *down* while maintaining usability. Relevant to the user's interest in TinyML and embedded AI.
**Feasibility**: HIGH. Can use existing M-BEIR subsets, apply quantization (INT4/INT8), and compare against full-scale baselines.

### Idea 3: Visual Evidence Discovery Benchmark (VEDB)
**Concept**: A corpus-level benchmark where the task is: given a textual claim or hypothesis, retrieve the image/figure/table from a large corpus that best supports or refutes it.
**Novelty**: MMDocRAG [web:22] evaluates evidence selection within a document; VEDB requires searching across thousands of documents. Tests *discovery*, not just *selection*.
**Feasibility**: MEDIUM. Requires building a claim-figure alignment dataset, potentially from scientific literature or Wikipedia.

### Idea 4: Decomposed Multimodal Retrieval Evaluation Framework (DEMREF)
**Concept**: A benchmark and methodology that explicitly isolates OCR quality, preprocessing, encoding, and retrieval components—following the methodology of [page:2].
**Novelty**: Provides a diagnostic toolkit for the field. Instead of single-number leaderboards, provides breakdowns: "your model gains +5 points from better OCR, +2 from encoding, +1 from retrieval."
**Feasibility**: HIGH. Mostly an engineering and annotation effort. Can extend VisR-Bench with controlled OCR ablations.

### Idea 5: Multimodal Retrieval Usefulness Score (MRUS)
**Concept**: An end-to-end metric that measures whether retrieved multimodal content actually improves downstream task performance (QA, reasoning, code generation) relative to text-only retrieval.
**Novelty**: Addresses the "retrieval is good but does it help?" question. Inspired by findings that improving retrieval recall from 80%→95% may only improve answers by 5-10% [web:38].
**Feasibility**: MEDIUM. Requires running full RAG pipelines with and without multimodal retrieval, but can leverage existing VQA and reasoning benchmarks.

---

## 4. What Can Be Done Cheaply

Given typical student resource constraints (single GPU, limited compute, no labeling budget):

1. **Re-annotate existing data**: Take M-BEIR or VisR-Bench queries and reformulate them as screenshot-style queries using synthetic generation (prompt a VLM to "describe this as a user screenshot query"). Cost: API credits only.
2. **Subset evaluation**: Instead of full M-BEIR (5.6M candidates), create a hard-negative subset (e.g., 10K candidates) where current models fail. Enables fast iteration.
3. **OCR ablation replication**: Replicate the controlled OCR study from [page:2] using open-source tools (EasyOCR, PaddleOCR, Tesseract) on a small benchmark subset. Cost: CPU time only.
4. **Quantization experiments**: Evaluate INT4/INT8 multimodal encoders (CLIP, SigLIP) on M-BEIR subsets to establish low-resource baselines. Cost: Single GPU, existing tools.
5. **Synthetic benchmark creation**: Use VLMs to generate "claims" for arXiv figures, then evaluate whether retrieval systems can find the source figure. Cost: API credits + existing arXiv figure datasets.

---

## 5. Best Datasets

| Dataset | Best For | Size | Access |
|---------|----------|------|--------|
| **M-BEIR** [web:16] | Universal multimodal retrieval training/eval | 1.5M queries, 5.6M candidates | HuggingFace |
| **MM-BRIGHT** [web:21] | Reasoning-intensive retrieval (technical diagrams, screenshots) | 2,803 queries, 29 domains | GitHub |
| **MMDocRAG** [web:22] | Document-level multimodal evidence | 4,055 QA pairs | HuggingFace |
| **MultiHaystack** [web:29] | Large-scale cross-modal retrieval + reasoning | 46K candidates, 747 Qs | arXiv |
| **VisR-Bench** [page:2] | Visually rich document retrieval (OCR ablations) | Multi-page documents | Paper/Project |
| **VIRA** [web:32] | Screenshot retrieval (products, papers, news, charts) | 13M screenshots | Project page |
| **UniIR/M-BEIR subset** | Quick prototyping and hard-negative mining | Custom subset | HuggingFace |

---

## 6. Realistic Paper Path

### Target Venue
EMNLP, ACL, SIGIR, or NeurIPS Datasets & Benchmarks track.

### Path: Low-Resource Multimodal Retrieval Benchmark + Baseline System

**Timeline**: 8-12 weeks

1. **Week 1-2**: Select 2-3 low-resource languages (e.g., Hindi, Swahili, Tamil) and identify existing multimodal datasets or build a small corpus using Wikimedia Commons + Wikipedia.
2. **Week 3-4**: Create query-image-document triplets. Use synthetic generation (prompt a VLM) to generate search queries from image captions.
3. **Week 5-6**: Implement baselines: (a) CLIP/SigLIP zero-shot, (b) quantized small encoder (e.g., SigLIP-Base INT8), (c) BM25 over OCR text.
4. **Week 7-8**: Evaluate and analyze failure modes. Focus on OCR quality impact and modality bias.
5. **Week 9-12**: Write paper emphasizing the benchmark contribution and low-resource insights.

**Why this works**: Benchmark papers are highly cited and well-received. The low-resource angle is timely [web:13][web:25]. Requires minimal compute (single GPU for inference, CPU for OCR).

---

## 7. High-Risk Paper Path

### Target Venue
NeurIPS, ICML, or ICLR.

### Path: Unified Multimodal Search Architecture with Generative Retrieval

**Concept**: Build a lightweight generative retrieval model (inspired by GENIUS [web:3]) that can take any combination of text, image, or screenshot as input and generate document identifiers directly, followed by embedding-based reranking.

**Why high-risk**:
- Requires training a new model (not just evaluating existing ones).
- Generative retrieval for multimodal is underexplored; failure modes are unknown.
- Competing against well-funded industry labs (Amazon, Google).

**Why high-reward**:
- If successful, represents a paradigm shift from embedding-based to generative multimodal retrieval.
- Could enable orders-of-magnitude faster retrieval at scale.
- Directly relevant to real-world search products.

**Mitigation**: Start with a small-scale proof-of-concept on M-BEIR subset. Show that a small generative model (e.g., Phi-3 or Qwen-2.5-Instruct fine-tuned) can achieve non-trivial Recall@10 before scaling.

---

## 8. Final Ranking

| Rank | Idea | Feasibility | Impact | Recommended For |
|------|------|-------------|--------|-----------------|
| **1** | **Idea 2: Low-Resource Multimodal Retrieval** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Student project, quick paper |
| **2** | **Idea 4: Decomposed Evaluation Framework** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High citation potential, methodology paper |
| **3** | **Idea 1: Screenshot Document Retrieval Benchmark** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Strong real-world relevance |
| **4** | **Idea 3: Visual Evidence Discovery** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Longer-term, more ambitious |
| **5** | **Idea 5: Retrieval Usefulness Score** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Requires full RAG pipeline setup |
| **6** | **High-Risk: Generative Multimodal Retrieval** | ⭐⭐ | ⭐⭐⭐⭐⭐ | If you have 3-6 months and GPU access |

### Recommended Immediate Action

Start with **Idea 4 (DEMREF)** or **Idea 2 (LR-MR)**. Both can be executed cheaply, produce concrete artifacts (benchmark splits, evaluation code), and lead to well-cited papers. The decomposed evaluation framework is particularly timely given the OCR confounding findings from March 2026 [page:2]—the community is actively looking for better evaluation practices.

---

## References

- [web:16] M-BEIR Benchmark (HuggingFace)
- [web:17] UniIR: Training and Benchmarking Universal Multimodal Information Retrieval
- [web:21] MM-BRIGHT: Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval
- [web:22] MMDocRAG: Benchmarking Retrieval-Augmented Multimodal Generation
- [web:24] MM-BRIGHT arXiv (Jan 2026)
- [web:29] MultiHaystack: Benchmarking Multimodal Retrieval and Reasoning
- [web:32] Unifying Search With Visualized Information Retrieval (VIRA/UniSE)
- [page:2] Retrieval or Representation? Reassessing Benchmark Gaps (Mar 2026)
- [web:3] GENIUS: Amazon Science Generative Universal Multimodal Search
- [web:12] MARVEL: Multimodal Retrieval on MM-BRIGHT
- [web:13] Large Multimodal Models for Low-Resource Languages Survey
- [web:38] RAG Evaluation Metrics Best Practices
- [web:40] UniSE: Universal Screenshot Embeddings (ACL 2025)

---

*Report generated by Agent 28: Multimodal Search and Retrieval Scout*
*Date: April 25, 2026*
