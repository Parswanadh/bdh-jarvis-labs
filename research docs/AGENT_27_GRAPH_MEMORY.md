# AGENT_27: Graph Memory and Relational Reasoning Scout
## Final Report: `AGENT_27_GRAPH_MEMORY.md`

---

## 1. Literature Review

Graph-structured memory has evolved from static knowledge bases to dynamic, agent-centric reasoning substrates. Key contributions include:

| Paper / System | Core Contribution | Relevance |
|---|---|---|
| G-Memory (Zhang et al., 2025) | Trainable hierarchical graph memory with RL-based weight optimization for meta-cognition [web:1] | Agent memory + RL synergy |
| AriGraph | Memory graph integrating semantic and episodic memories with episodic update loops [web:3] | Episodic-graph construction |
| KG Structure as Prompt (arXiv 2407.18752) | Using KG topology (common neighbors, metapaths) as prompt structure for SLMs (<1B params) [web:9] | Small-model reasoning with graph prompt |
| MR-MKG (ACL 2024) | Multimodal KG reasoning with relation graph attention; trains only ~2.25% of LLM parameters [web:31] | Multimodal grounded memory |
| GEAR (ACL Findings 2025) | Graph-enhanced agent for multi-hop RAG; beats HippoRAG by up to 9.8% R@15 on MuSiQue [web:22] | Graph retrieval benchmark |
| RAG vs. GraphRAG (arXiv 2502.11371) | Systematic evaluation showing complementarity: RAG wins single-hop factual; GraphRAG wins multi-hop reasoning [web:16] | Controlled comparative study |
| Structured Reasoning / SCR (arXiv 2601.07180) | Decoupled Generate-Verify-Revise scratchpad; reduces output token length up to 50% [web:24] | Structured scratchpads |
| 3Q Scratchpad (2026) | Lightweight prompting format (WHAT_I_KNOW, WHAT_I_NEED, WHAT_I_AM_ASSUMING) for auditable reasoning [web:6] | Minimal structured scratchpad |

**Emerging themes (2025–2026):**
- **Hybrid architectures** dominate production: vector + graph + episodic buffer is the consensus pattern [web:7].
- **Temporal awareness**: Graphiti and Zep build time-indexed edges for episodic memory over evolving knowledge [web:13].
- **Trainable memory**: Graph memory is no longer static retrieval; it is being integrated into RL training loops (G-Memory) [web:1].

---

## 2. Is Graph Memory Overhyped?

**Yes, but only partially.**

### Where hype exceeds evidence
- **Universal replacement claims**: GraphRAG does not consistently beat dense RAG across all tasks. Community-based global search underperforms basic RAG on detail-oriented QA [web:16].
- **Entity-light queries**: For single-hop factual retrieval, vector similarity is sufficient. Graph construction overhead is wasted if queries rarely involve multi-hop relationships [web:16].
- **Cost blindness**: Graph extraction, embedding, and traversal add latency. Many benchmark wins come from systems with >10× the retrieval budget of simple RAG baselines [web:20].

### Where it is grounded
- **Multi-hop reasoning**: GraphRAG sustains accuracy with 10+ entities per query while vector RAG degrades to 0% [web:20].
- **Faithfulness**: Neo4j GraphRAG significantly outperforms FAISS in faithfulness (0.54 vs 0.18) because relational context is preserved [web:23].
- **Cross-document reasoning**: Graph retrieval succeeds 4× more often on aggregation queries requiring cross-document relationships [web:18].

**Verdict**: Graph memory is overhyped as a *default* architecture, but it is legitimately powerful for relational, multi-hop, temporal, and grounded tasks. The strongest papers are those that define the boundary conditions rather than claiming universal superiority.

---

## 3. Five Candidate Research Directions

### Direction A: Graph Scratchpad for Small-Model Reasoning
- **Idea**: Equip a 1–3B parameter SLM with a structured external graph scratchpad (nodes = entities/assumptions, edges = dependencies). At each reasoning step, the model reads the current subgraph state, appends a new node/edge, and outputs the next reasoning action.
- **Why publishable**: SCR (Structured Reasoning) shows structured traces reduce token length by 50% for LLMs [web:24]; no work extends this to *small* models with an explicit graph state machine. A negative result here is also valuable (does the overhead exceed SLM capacity?).
- **Feasibility**: High. Can prototype with NetworkX in-memory graphs and a 1.5B–3B model (Qwen2.5, Phi-4-mini) on 8GB VRAM via 4-bit quantization.
- **Benchmarks**: GSM8K, BBH subsets, MuSiQue (2-hop only).

### Direction B: Temporal Episodic Graph for Long-Horizon Agents
- **Idea**: Build an agent memory graph where nodes are events/observations and edges are temporal relations (before, caused, contradicts). The agent retrieves the relevant *temporal path* rather than the most similar vector chunk.
- **Why publishable**: AriGraph showed episodic-semantic integration [web:3]; temporal indexing is still understudied beyond commercial systems like Graphiti. A clean open benchmark with ablations (temporal vs. non-temporal retrieval) would fill a gap.
- **Feasibility**: Medium. Requires environment with long episodes (WebShop, ALFWorld, or a custom text adventure). Temporal extraction can use lightweight spaCy/regex pipelines.
- **Benchmarks**: ALFWorld, WebShop, or a custom 50-step cooking/task domain.

### Direction C: Knowledge Graph as Negative-Result / Ablation Study
- **Idea**: Run a rigorous controlled ablation: take a standard RAG pipeline and incrementally add graph extraction (entity linking, relation extraction, community detection). Measure at which point the complexity begins to hurt more than help.
- **Why publishable**: RAG vs. GraphRAG systematic evaluation [web:16] exists but only at corpus level. A *per-query-type* decision model ("when to build the graph vs. when vectors are enough") is a practical contribution and provides a reproducible negative-result baseline.
- **Feasibility**: Very high. Can run entirely on CPU + 8GB GPU for the LLM generator phase. Use HotpotQA, 2WikiMultihopQA, and a synthetic single-hop set.
- **Benchmarks**: HotpotQA, 2WikiMultihopQA, MuSiQue; metrics = EM, F1, latency, graph-build cost.

### Direction D: Multimodal Graph Memory for Grounded Tasks
- **Idea**: Construct a multimodal knowledge graph from visual+text observations (robot navigation, scene understanding). A small VLM or multimodal projector aligns image regions to KG nodes.
- **Why publishable**: MR-MKG used full LLMs [web:31]; doing this with <3B-parameter vision models and a graph memory would be novel, especially for embodied AI or IoT robotics (relevant to your hardware background).
- **Feasibility**: Medium. Requires a small VLM (LLaVA-Phi-3, 3.8B, quantized) and a simple embodied environment (AI2-THOR lite or a custom grid world with image observations).
- **Benchmarks**: OK-VQA subset, AI2-THOR object navigation, or custom robotic sorting task.

### Direction E: Graph Retrieval + SLM Code-Generation (Agentic Tool Use)
- **Idea**: Store API/tool schemas as a knowledge graph. The agent retrieves the relevant tool subgraph and generates calls. Compare graph retrieval against vector retrieval of raw documentation for small models.
- **Why publishable**: Structured tool graphs are common in industry (OpenAPI, MCP servers) but under-evaluated in academia. A benchmark measuring SLM tool-use accuracy with graph vs. dense retrieval would be immediately useful.
- **Feasibility**: High. Can use existing APIBench or ToolBench subsets; graph is just the schema DAG. 8GB VRAM easily runs a 3B model for code generation.
- **Benchmarks**: ToolBench (selected subset), APIBench; metrics = pass@1, hallucination rate, correct API chaining.

---

## 4. Benchmarks and Evaluation

| Benchmark | Tests | Best For |
|---|---|---|
| **HotpotQA** | 2-hop multi-hop reasoning; bridge and comparison questions | Graph retrieval ablations, recall@K [web:22] |
| **MuSiQue** | 2–4 hop compositional reasoning; adversarial distractors | Stress-testing graph memory depth [web:22] |
| **2WikiMultihopQA** | Multi-hop with Wikipedia and structured tables | Hybrid graph+table reasoning |
| **GSM8K / BBH** | Math and symbolic reasoning | Graph scratchpad reasoning evaluation [web:27] |
| **ALFWorld / WebShop** | Embodied long-horizon decision making | Temporal episodic graph evaluation |
| **ToolBench / APIBench** | Tool use and API selection | Graph-structured tool schema retrieval |

**Evaluation protocol recommendations:**
1. Use **R@K** and **EM/F1** for QA tasks [web:22].
2. Report **graph-build latency** and **per-query traversal cost** alongside accuracy; otherwise speedups/losses are invisible.
3. Include **faithfulness** metrics (entity coverage, relation hallucination) as graph memory claims to reduce hallucination [web:23].
4. For temporal graphs, define a **temporal consistency score**: fraction of retrieved paths where edge timestamps are monotonic with the query’s implied timeline.

---

## 5. Feasibility and Implementation Notes

### Hardware (8GB RTX 4070)
- **Models**: Qwen2.5-3B-Instruct, Phi-4-mini (3.8B), LLaVA-Phi-3 (3.8B vision) all fit comfortably in 4-bit (≈2–3GB VRAM).
- **Graph engine**: NetworkX (pure Python, in-memory) is sufficient for up to ~100K nodes on CPU. For larger graphs, use Neo4j Community Edition or FalkorDB on a cheap VPS or even locally via Docker [web:37][web:40].
- **Training budget**: Fine-tuning adapters (LoRA, 8–16 rank) on a 3B model takes ~2–4 hours per task on 8GB VRAM if sequences are short (<1K tokens).

### Libraries
- Graph construction: `networkx`, `graspologic`, `neo4j-python-driver`
- KG extraction from text: `spaCy` + `rebel` (relation extraction), or lightweight LLM-based pipelines
- Retrieval evaluation: `beir`, `ir-datasets`
- Agent scaffolding: `smolagents` [web:15] or custom ReAct loop

### Timeline estimate (1–2 months)
- **Week 1–2**: Literature deep dive, environment setup, baseline reproduction (dense RAG on HotpotQA).
- **Week 3–4**: Graph extraction pipeline + retrieval loop implementation.
- **Week 5–6**: Experiment runs, ablations, and metric collection.
- **Week 7–8**: Analysis, writing, and a clean open-source repo.

### Cost constraints
- All proposed directions can be executed without cloud GPU renting if you accept longer training times and use quantized inference.
- Optional: RunPod spot instances ($0.15–0.30/hr) only for final sweep experiments or larger model baselines.

---

## 6. Reviewer Attack Surface

### Likely objections
1. **"You just built a pipeline, not a model."**
   - *Defense*: Frame as a **system contribution** with rigorous ablation. Many accepted papers at ACL/NeurIPS (e.g., GEAR, G-Memory) are architectural pipeline papers with strong evaluation.
   - *Mitigation*: Include a small learned component (e.g., edge-weight RL, or a tiny adapter that scores graph-path relevance) to claim a modeling contribution.

2. **"The gains come from more retrieval budget, not graph structure."**
   - *Defense*: Fix total retrieval token budget across all baselines. Show that graph wins *under the same context length* by structure, not volume.
   - *Mitigation*: Plot accuracy vs. retrieval cost curves; if graph only wins at higher cost, that is an honest and publishable finding.

3. **"Why not just use long-context LLMs?"**
   - *Defense*: Long-context does not solve cross-session continuity, reliable fact updates, or cost at scale [web:7]. Your benchmark should include a "full-context" baseline to show memory beats naive context stuffing.

4. **"Graph extraction quality dominates results; your pipeline is too brittle."**
   - *Defense*: Report extraction F1 explicitly. Run sensitivity analysis: replace your extractor with an oracle (gold entities/relations) to bound upper-bound performance vs. extraction-noise floor.

5. **"Negative results are hard to publish."**
   - *Defense*: Target venues that explicitly welcome them (e.g., *NeurIPS Track on Datasets and Benchmarks*, *ACL Rolling Review* with reproducibility emphasis, *FAccT*, or workshop tracks). Frame negative result as a **decision boundary model**: "We identify the task conditions under which graph memory is harmful."

---

## 7. Best Recommendation

**Primary recommendation: Direction C — Knowledge Graph as Controlled Ablation / Decision Boundary Study.**

### Why
- **Highest feasibility**: Requires no new model architecture; just rigorous evaluation. You can complete it in 4–6 weeks.
- **Highest reviewer defensibility**: You directly address the "when is graph memory worth it?" question that the field is currently debating [web:16][web:7].
- **Negative-result friendliness**: If graphs lose on simple tasks, that is a valuable reproducible result. If they win on multi-hop tasks, you have a clean, bounded claim.
- **Leverages your hardware**: All experiments fit on 8GB VRAM with a 3B model and CPU-based graph traversal.
- **Extensible**: Once the ablation framework exists, Directions A and E become natural follow-ups (swap the retrieval substrate).

### Concrete next step
1. Reproduce the **RAG vs. GraphRAG** unified protocol from [web:16] on **HotpotQA** and **MuSiQue**.
2. Add a **per-query classifier** (simple logistic regression on query features: entity count, relation verbs, question type) that predicts whether graph retrieval or dense retrieval will perform better.
3. Report **accuracy, latency, and cost** for three modes: dense-only, graph-only, and oracle-selector.
4. Publish the code + dataset split as a reproducibility package.

If the selector works, the paper is a **practical system contribution**. If it fails, the paper is a **rigorous negative result with actionable guidelines**.

---

*Report compiled by Agent 27, April 2026.*
