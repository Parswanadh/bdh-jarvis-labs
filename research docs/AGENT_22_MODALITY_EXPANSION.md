# Agent 22: Modality Expansion Scout
## Mission Analysis — Underexplored Modalities Beyond Text+Image

**Date:** April 2026
**Objective:** Identify non-standard modalities that create real novelty vs. distraction for multimodal AI systems.

---

## 1. Modality-by-Modality Analysis

### Audio & Speech
- **Current state:** Well-covered by GPT-4o, WhisBERT, and various audio-text fusion systems using wav2vec/HuBERT encoders [web:6][web:10].
- **Feasibility:** High — audio encoders are mature and pre-trained models exist.
- **Novelty:** Medium. Audio is already a "standard" third modality for frontier labs. Adding it catches up, not differentiates.
- **Key risk:** Temporal alignment complexity and high compute cost for real-time processing [web:10].
- **Verdict:** Feasible but crowded. Better as a secondary addition, not a primary differentiator.

### Document Layout / Visual Documents
- **Current state:** DocLLM proved that layout-aware models (using bounding boxes alone, no image encoder) outperform SOTA on 14/16 document benchmarks [web:21][web:23][web:27]. Docopilot/Doc-750K advances document-level cross-page understanding [web:20].
- **Feasibility:** Very high — avoids expensive vision encoders entirely; only requires text + bounding box coordinates.
- **Novelty:** Medium-High for lightweight approaches. DocLLM's disentangled attention mechanism is a genuine architectural innovation.
- **Key opportunity:** Most models still treat documents as images. A text+layout-only approach is radically more efficient.
- **Verdict:** Strong candidate — underexplored by open-source MLLMs.

### Code Structure / Code Graphs
- **Current state:** CGBridge (2025) connects Code Property Graphs to LLMs via a bridge module, outperforming on code summarization and translation [web:42]. Rep-CodeGen uses LLM agents to generate graph representations automatically [web:38].
- **Feasibility:** Medium — requires parsing source code into AST/CPG representations; toolchain-dependent.
- **Novelty:** High. Code structure (graphs, control flow, data dependencies) is fundamentally different from text tokens. Very few MLLMs natively understand it.
- **Key opportunity:** Software engineering tasks (code review, security analysis, cross-project issue tracking) are multimodal by nature (text + code + images) [web:17].
- **Verdict:** High novelty but requires significant infrastructure. Best for specialized applications.

### Diagram / Chart Data
- **Current state:** ChartLlama and ChartGemma focus on chart understanding and data extraction [web:25][web:35][web:39]. WebCompass benchmarks multimodal web coding evaluation [web:16].
- **Feasibility:** High — chart/diagram data can be rendered as images or extracted as structured tables.
- **Novelty:** Medium. Chart understanding is a solved-enough niche; benchmarks exist (ChartQA).
- **Key risk:** Easy to treat as "just another image" without adding structural understanding. The real gap is reasoning about the underlying data, not recognizing the visual pattern.
- **Verdict:** Niche utility. Valuable but not a broad differentiator.

### UI / Screenshot Understanding
- **Current state:** ScreenAI (Google) achieves SOTA on UI and infographics by treating screens as text+image problems with a unified schema [web:29][web:30][web:32]. Enables autonomous GUI control.
- **Feasibility:** High for understanding; medium for action/control (requires interaction loops).
- **Novelty:** Medium-High for autonomous agent applications. GUI understanding is underserved by open models.
- **Key limitation:** Requires large-scale synthetic data generation and hierarchical AX tree reconstruction [web:32].
- **Verdict:** Excellent for agentic applications, but narrow domain.

### Sensor / Simulated Modalities (IMU, Depth, Thermal)
- **Current state:** ImageBind binds 6 modalities (text, image, audio, depth, thermal, IMU) into shared space [web:5]. Apple's research shows LLMs can do late fusion of audio+IMU for activity recognition with zero-shot F1 above chance [web:11]. SensorGAN handles missing wearable sensor channels [web:7].
- **Feasibility:** Medium — hardware-dependent for real sensors; simulated sensors are easier.
- **Novelty:** High for IoT/robotics. Almost no consumer MLLMs natively process IMU or thermal data.
- **Key insight:** Late fusion with LLMs works without task-specific training, enabling rapid prototyping [web:11].
- **Verdict:** Strongest novelty for embedded/robotics use cases. AlienX hardware focus aligns well here.

---

## 2. Novelty vs. Feasibility Grid

| Modality | Feasibility | Novelty | Risk Level | Recommended Priority |
|----------|-------------|---------|------------|---------------------|
| **Document Layout** | Very High | Medium-High | Low | **P1** |
| **Sensor/IMU/Thermal** | Medium | Very High | Medium | **P1** |
| **Code Structure (Graphs)** | Medium | Very High | Medium | **P2** |
| **UI/Screenshots** | High | Medium-High | Medium | **P2** |
| **Audio/Speech** | High | Medium | Low | P3 |
| **Diagram/Chart** | High | Medium | Low | P3 |

---

## 3. Overlap with Existing Strengths

Given profile: Amrita EAC student, expertise in:
- Embedded systems (Arduino, ESP32, ARM microcontrollers)
- LLM optimization & quantization (INT4/INT8/FP16)
- Hardware-software co-design, Verilog learning
- IoT robotics, motor control, sensor interfacing
- Git/GitHub, Python, C/C++, MCP servers

### Strongest Alignments

| Modality | Why It Fits |
|----------|-------------|
| **Sensor/IMU** | Direct hardware experience. ESP32 + IMU sensors are standard robotics stack. Can build data collection rigs physically. |
| **Document Layout** | Bounding-box approach matches lightweight quantization mindset — no heavy vision encoder needed. |
| **Code Structure** | Heavy GitHub user; understands AST concepts. CGBridge-style graph encoders are natural extension. |
| **UI/Screenshots** | Builds MCP servers and CLI tools; understands interface abstractions. |

### Weakest Alignments
- **Audio/Speech:** No signal processing background (though DSP is in coursework). Mature competition from OpenAI/others.
- **Diagram/Chart:** More of a data visualization niche; doesn't leverage embedded systems strength.

---

## 4. Best Single Extra Modality to Add

### **Document Layout (Bounding-Box Aware Documents)**

**Rationale:**
1. **Highest feasibility-to-novelty ratio** — DocLLM proved this works without any image encoder [web:23].
2. **Matches hardware constraints** — You have an 8GB RTX 4070 laptop and cloud credit limits. Avoiding vision encoders = smaller models = quantizable.
3. **Natural data abundance** — PDFs, forms, invoices, research papers are everywhere.
4. **Architectural innovation path** — Disentangled attention matrices for spatial+text alignment is a genuine paper-worthy contribution.
5. **Complements text+image** — Document layout is often *lost* when documents are flattened to images. Recovering it is novel.

**Implementation sketch:**
- Input: text tokens + bounding box coordinates (x1, y1, x2, y2) per token
- Decompose attention into text-to-text, spatial-to-spatial, and cross-modal matrices
- Pre-train with infill objectives on 750K+ document pages (Doc-750K scale) [web:20]
- Fine-tune for form extraction, table understanding, multi-page reasoning

---

## 5. Worst Distractions to Avoid

1. **Raw Audio as Primary Modality**
   - GPT-4o, Qwen2.5-Audio, and others already dominate. Without speech data pipelines or ASR infrastructure, this is a resource sink with no differentiation.

2. **"General" Video Understanding**
   - Video = image sequence + audio. True temporal reasoning is compute-heavy. Most "video MLLMs" just sample frames and lose motion information. Avoid unless you have a specific temporal reasoning angle.

3. **Omnidirectional / All-Modalities-at-Once**
   - ImageBind-style 6-modality binding is impressive but requires massive datasets. Open-source attempts often underperform because data alignment is harder than architecture.

4. **3D Point Clouds / Depth Maps**
   - Novel but requires specialized datasets (ScanNet, etc.). Doesn't leverage your current hardware/software strengths.

5. **Touch / Haptic Feedback**
   - Research curiosity with no practical data pipeline. Nearly zero training datasets exist.

---

## 6. Top 3 Paper Opportunities

### Paper 1: "QuantizedDoc: Layout-Aware Multimodal Reasoning at the Edge"
- **Idea:** Extend DocLLM's bounding-box approach with INT4 quantization-aware training, proving document understanding can run on 8GB consumer GPUs.
- **Why novel:** No existing work combines layout-aware MLLMs with aggressive quantization.
- **Dataset:** Doc-750K or自建 synthetic document corpus.
- **Target venue:** EMNLP, ACL, or ICCV workshop.
- **Difficulty:** Medium (builds on DocLLM; quantization expertise is already a strength).

### Paper 2: "SensorBridge: Fusing IMU + Text for Embedded Activity Reasoning"
- **Idea:** LLM-based late fusion of IMU time-series (from ESP32/Arduino) + text commands for zero-shot robotics control, following Apple's sensor-fusion LLM approach [web:11].
- **Why novel:** Brings wearable sensor fusion to open-source edge LLMs (Ollama-scale). Demonstrates that 7B models can do zero-shot activity classification from sensor streams.
- **Dataset:** Self-collected IMU data + Ego4D subset.
- **Target venue:** IoT/robotics conference (IROS, ICRA workshop, or SensSys).
- **Difficulty:** Medium-High (requires hardware data collection + LLM fine-tuning).

### Paper 3: "CodeGraphBridge: Lightweight Graph Injection for Small Language Models"
- **Idea:** Adapt CGBridge's graph encoder [web:42] for SLMs (1B-3B parameters). Show that structure-informed prompting improves code completion on resource-constrained devices.
- **Why novel:** CGBridge used frozen LLMs; no one has tested if graph structure helps *small* models. Critical for your SLM-on-laptop vision.
- **Dataset:** CodeXGLUE + custom code graph corpus.
- **Target venue:** ICML/NeurIPS workshop, or ACL (software engineering track).
- **Difficulty:** High (requires building graph extraction pipeline + training GNN bridge module).

---

## Quick Reference: Modality Decision Matrix

| If your goal is... | Add this modality | Avoid |
|-------------------|-------------------|-------|
| Fast publication with novel angle | Document Layout | Audio, Video |
| Leverage embedded systems expertise | Sensor/IMU | 3D point clouds |
| Build an AI agent / GUI automation | UI/Screenshots | Haptic, Touch |
| Differentiate on code understanding | Code Structure (Graphs) | Generic text-image |
| Maximum novelty at any cost | Sensor/IMU + Thermal | "All modalities" approaches |

---

*Compiled by Agent 22 — April 2026*
