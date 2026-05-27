# BDH NeurIPS 2027 — 9-Day Master Plan

> **Goal:** Transform BDH from a buggy research prototype into a NeurIPS-ready architecture with fixed core algorithm, validated training pipeline, publication-quality results, and complete paper draft.
> **Target:** NeurIPS 2027 submission (May 2027 deadline)
> **Current Date:** May 17, 2026
> **Parallel Tracks:** Research | Engineering | Frontend | Documentation

---

## Phase Overview

| Day | Phase | Theme | Risk |
|-----|-------|-------|------|
| 1 | **Bug Triage & Baseline** | Fix 4 critical bugs, establish clean baseline | HIGH |
| 2 | **Hebbian Outer Product** | Replace element-wise with true outer product state | HIGH |
| 3 | **RoPE + Positional Awareness** | Enable RoPE in multi-scale, validate positional encoding | MEDIUM |
| 4 | **Gating Architecture** | Fix multiplicative gating, add data-dependent decay | HIGH |
| 5 | **Training Pipeline v2** | Rebuild distillation pipeline with vocab projection | MEDIUM |
| 6 | **Benchmarking & Viz** | Run full benchmark suite, generate publication figures | MEDIUM |
| 7 | **SOTA Comparison** | Head-to-head vs Mamba/RWKV/Transformer at equal params | MEDIUM |
| 8 | **Paper Draft + Deployment** | Write NeurIPS paper, deploy interactive demo | MEDIUM |
| 9 | **Integration + Polish** | End-to-end validation, paper review, submission prep | LOW |

---

## Day 1 — Bug Triage & Baseline

| Track | Tasks |
|-------|-------|
| **Research** | Audit all 4 bugs with line-level evidence; document exact failure modes; create bug-to-fix mapping table; review 8 research docs for cross-references to each bug |
| **Engineering** | Create `bdh_v2_clean.py` from scratch (not patching old code); implement correct element-wise Hebbian as baseline (preserve current behavior for comparison); add unit test scaffolding; verify tokenizer loads (TinyLlama 32K vocab); confirm CUDA forward pass works |
| **Frontend** | Run all 18 Playwright tests to confirm website integrity; add "Day 1 Baseline" banner to research dashboard page; verify Three.js animations load correctly; test responsive breakpoints |
| **Documentation** | Create `BUG_TRIAGE_REPORT.md` with severity/impact for each bug; update `research_for_gemini/00_MASTER_INDEX` with bug references; create `DAY1_STATUS.md` with before/after comparison table |

**Dependencies:** None (Day 1 is the root)
**Success Metrics:** 4 bugs documented with line evidence; clean baseline model runs forward pass; 18/18 Playwright tests pass; bug triage report complete
**Risk Level:** HIGH — if baseline doesn't run, all downstream days blocked

---

## Day 2 — Hebbian Outer Product (Critical Bug #1)

| Track | Tasks |
|-------|-------|
| **Research** | Derive memory cost analysis: O(d) → O(d²) per head; calculate VRAM impact for d=512, 8 heads; design hybrid state: keep vector for fast path, add matrix for quality path; review DeltaNet delta rule for comparison |
| **Engineering** | Implement `hebbian_outer_product()` using `torch.einsum('bhd,bhe->bhde', Q, V)`; add memory-efficient chunked outer product for long sequences; create A/B test: element-wise vs outer product on same data; add gradient flow validation (check no NaN/inf); implement state matrix visualization hook |
| **Frontend** | Build interactive Hebbian state matrix heatmap visualization (using Plotly or D3.js); add toggle to switch between element-wise and outer product views; create animated comparison: vector vs matrix state evolution |
| **Documentation** | Write "Hebbian State: Theory vs Implementation" section for paper; document memory trade-off analysis; create figure: outer product state matrix heatmap; update architecture diagram with corrected state update |

**Dependencies:** Day 1 baseline must pass; clean model architecture ready
**Success Metrics:** Outer product forward pass runs without OOM; A/B test shows measurable quality difference (perplexity delta); state matrix visualization renders; memory cost documented
**Risk Level:** HIGH — O(d²) memory may exceed VRAM on RTX 4070; mitigation: use gradient checkpointing, reduce batch size, or use mixed precision

---

## Day 3 — RoPE + Positional Awareness (Critical Bug #2)

| Track | Tasks |
|-------|-------|
| **Research** | Analyze why RoPE was disabled (likely gradient instability); study Qwen's RoPE implementation for reference; design RoPE + linear attention interaction (RoPE rotates Q/K before K^T@V); test relative position encoding as fallback |
| **Engineering** | Port RoPE from base BDH to multi-scale BDH; implement `apply_rope(Q, K, position_ids, base=10000)`; add positional ablation test: no-RoPE vs RoPE on sequence reversal task; verify gradient flow through RoPE; add position-aware benchmark (needle-in-haystack at 512 tokens) |
| **Frontend** | Build positional encoding visualization: show rotation angles per dimension; create interactive demo: input "ABC" vs "CBA", show different outputs with/without RoPE; add position heatmap to architecture diagram |
| **Documentation** | Write "Positional Encoding in Linear Attention" section; document the bug (line 304 `pass # No positional encoding`); create ablation results table; update architecture spec with RoPE parameters |

**Dependencies:** Day 2 outer product state must work (RoPE operates on Q/K before state update)
**Success Metrics:** RoPE-enabled model distinguishes "ABC" from "CBA"; sequence reversal accuracy >80%; needle-in-haystack retrieval works at 512 tokens; no gradient instability
**Risk Level:** MEDIUM — RoPE may cause training instability; mitigation: start with small base frequency, gradually increase

---

## Day 4 — Gating Architecture (Critical Bug #3 + Architectural Gap)

| Track | Tasks |
|-------|-------|
| **Research** | Analyze multiplicative gating gradient flow: ∂out/∂x = gate + x·gate'·(∂attn+∂ffn); identify saturation points (sigmoid → 0); design hybrid residual: `x + α·gate·(attn+ffn)` where α is learnable; study input-dependent gating from Mamba/GLA/Qwen3.5 GDN |
| **Engineering** | Implement data-dependent decay: `γ(input) = sigmoid(W_γ(x))`; implement learnable gating coefficient α; add gradient norm monitoring per layer; create gating visualization: plot gate distribution across layers/tokens; implement fallback additive residual if gate < threshold |
| **Frontend** | Build gating distribution dashboard: histogram of gate values per layer; create live gate visualization during text generation (show which tokens pass through); add "gradient health" indicator (green/yellow/red) |
| **Documentation** | Write "Multiplicative Gating: Analysis and Fix" section; document gradient saturation problem; create gating distribution figures; update training guide with new gating parameters |

**Dependencies:** Day 3 RoPE must be stable (gating operates on attention+FFN outputs)
**Success Metrics:** Gate values don't saturate at 0 (>5% of gates in [0.1, 0.9] range); gradient norms stable across layers; training loss decreases consistently; no vanishing gradient warnings
**Risk Level:** HIGH — gating fix may require rethinking the entire residual connection; mitigation: keep additive residual as fallback path

---

## Day 5 — Training Pipeline v2 (Critical Bug #4 + Infrastructure)

| Track | Tasks |
|-------|-------|
| **Research** | Design vocab projection strategy: evaluate 4 options (learned projection, shared tokenizer, hidden-state CKA distillation, sub-atomic top-K); select optimal approach; design distillation loss: `L = λ·KL + (1-λ)·CE` with temperature scheduling |
| **Engineering** | Implement vocab projection layer: `W_proj: [teacher_vocab] → [student_vocab]`; rebuild `train_distillation_v2.py` with: (1) proper KL divergence, (2) temperature scaling, (3) gradient accumulation, (4) checkpoint resume, (5) validation loop; wire up existing logits data from `D:\logits_production_extended`; add wandb/logging integration; implement early stopping |
| **Frontend** | Build training dashboard: live loss curves, GPU utilization, checkpoint progress; create training timeline visualization; add model comparison widget (baseline vs v2) |
| **Documentation** | Write "Knowledge Distillation Pipeline" section; document vocab mismatch fix; create training configuration reference; update `JARVIS_README.md` with v2 pipeline |

**Dependencies:** Days 2-4 bug fixes must be integrated; logits data available from previous runs
**Success Metrics:** KL divergence actually computed (not falling back to CE only); training loss decreases over 1000 steps; checkpoint saves and resumes correctly; validation perplexity improves over baseline
**Risk Level:** MEDIUM — vocab projection may be lossy; mitigation: use CKA-based hidden state distillation as parallel approach

---

## Day 6 — Benchmarking & Data Visualization

| Track | Tasks |
|-------|-------|
| **Research** | Design benchmark suite: (1) language modeling perplexity on WikiText-2, (2) needle-in-haystack at [256, 512, 1024, 2048], (3) sequence reversal, (4) copy task, (5) associative recall; define evaluation metrics; plan ablation studies (each bug fix independently) |
| **Engineering** | Implement `benchmark_suite.py` with all 5 benchmarks; run full ablation: baseline → +outer-product → +RoPE → +gating → +distillation; generate all benchmark data; create comparison tables; implement statistical significance testing |
| **Frontend** | Build publication-quality visualization suite: (1) perplexity comparison bar chart, (2) memory retention curve (4 scales), (3) ablation waterfall chart, (4) gating distribution heatmap, (5) state matrix evolution animation, (6) training loss curves; export all figures as SVG + PNG at 300 DPI; integrate into website research section |
| **Documentation** | Write "Experimental Results" section for paper; create all figure captions; document benchmark methodology; create supplementary materials table |

**Dependencies:** Day 5 training pipeline must produce trained checkpoints; all bug fixes integrated
**Success Metrics:** All 5 benchmarks complete; ablation shows each fix contributes positively; 6+ publication-quality figures generated; results reproducible (3 runs, std < 5%)
**Risk Level:** MEDIUM — some benchmarks may not show improvement; mitigation: focus on benchmarks where BDH has theoretical advantage (long-context, interpretability)

---

## Day 7 — SOTA Comparison

| Track | Tasks |
|-------|-------|
| **Research** | Select comparison models: Transformer (GPT-2 scale), Mamba-1/2, RWKV-6, GLA, DeltaNet; ensure fair comparison (same params, same data, same training steps); analyze BDH's unique advantages: O(N) + interpretable + bio-plausible; position BDH in the architecture landscape |
| **Engineering** | Run head-to-head benchmarks at equal parameter counts (70M, 100M); measure: perplexity, training speed (tokens/sec), inference memory, context window effective length; create comparison table; implement statistical tests for significance |
| **Frontend** | Build interactive architecture comparison tool: radar chart showing BDH vs 6 alternatives across 8 dimensions; create "BDH Position in Landscape" interactive visualization; add SOTA benchmark results table to website |
| **Documentation** | Write "Related Work" and "Comparison to SOTA" sections; create architecture comparison table; write positioning statement for paper introduction; document experimental setup for reproducibility |

**Dependencies:** Day 6 benchmark results must be complete; comparison model implementations available
**Success Metrics:** Fair comparison at 70M and 100M params; BDH competitive on at least 3 of 5 metrics; comparison data reproducible; positioning statement clear
**Risk Level:** MEDIUM — BDH may underperform on some metrics; mitigation: emphasize unique strengths (interpretability, memory efficiency, bio-plausibility) rather than raw perplexity

---

## Day 8 — NeurIPS Paper Draft + Deployment

| Track | Tasks |
|-------|-------|
| **Research** | Write complete NeurIPS paper draft: Abstract, Intro, Related Work, Method (BDH v2), Experiments, Results, Analysis, Conclusion, Limitations; ensure all claims backed by Day 6-7 data; write supplementary material; prepare rebuttal anticipations |
| **Engineering** | Package BDH v2 as installable Python package (`pip install bdh`); create `requirements.txt` and `setup.py`; write quick-start tutorial notebook; deploy model inference API (FastAPI); create Docker container for reproducibility |
| **Frontend** | Deploy interactive BDH demo: live text generation with state matrix visualization; add paper abstract page to website; create "Try BDH" interactive playground; ensure all 18 Playwright tests still pass; add paper download link |
| **Documentation** | Complete paper draft (8 pages + references); create supplementary materials (ablation tables, additional figures, training logs); write README for code repository; create reproducibility checklist |

**Dependencies:** Days 6-7 results complete; all bug fixes integrated and benchmarked
**Success Metrics:** Complete paper draft (8 pages); interactive demo deployed; all 18 Playwright tests pass; code package installable; reproducibility checklist complete
**Risk Level:** MEDIUM — paper may not meet NeurIPS standards; mitigation: focus on novelty (first fixed Hebbian linear attention with multi-scale memory + data-dependent gating)

---

## Day 9 — Integration, Review & Submission Prep

| Track | Tasks |
|-------|-------|
| **Research** | Full paper review: check all claims against data; verify all citations; ensure no overclaiming; write limitations section honestly; prepare 3-page supplementary; create presentation slides for NeurIPS |
| **Engineering** | End-to-end integration test: fresh env → pip install → run benchmarks → reproduce all results; fix any remaining issues; create CI/CD pipeline; tag v2.0 release; archive all training checkpoints |
| **Frontend** | Final website polish: check all links, animations, responsive design; run full Playwright test suite (18 tests); add "NeurIPS 2027 Submission" badge; create paper teaser video; performance audit (Lighthouse score >90) |
| **Documentation** | Final paper proofread; create submission checklist; prepare anonymized version for double-blind review; write cover letter; create code repository README with paper link; archive all research documents |

**Dependencies:** All previous days complete
**Success Metrics:** Paper draft final; all results reproducible from scratch; 18/18 Playwright tests pass; website Lighthouse >90; submission checklist complete; code repository clean
**Risk Level:** LOW — integration day, most work is verification and polish

---

## Dependency Graph

```
Day 1 (Baseline)
    ↓
Day 2 (Outer Product) ──────────────────────────────────┐
    ↓                                                    │
Day 3 (RoPE) ───────────────────────────────────────────┤
    ↓                                                    │
Day 4 (Gating) ─────────────────────────────────────────┤
    ↓                                                    │
Day 5 (Training Pipeline) ← all bug fixes integrated ───┘
    ↓
Day 6 (Benchmarking) ← trained checkpoints from Day 5
    ↓
Day 7 (SOTA Comparison) ← benchmark results from Day 6
    ↓
Day 8 (Paper + Deploy) ← results from Days 6-7
    ↓
Day 9 (Integration + Polish) ← everything
```

**Parallel tracks within each day:** Research, Engineering, Frontend, Documentation run concurrently where possible.

---

## Risk Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Outer product OOM on RTX 4070 | HIGH | BLOCKING | Use gradient checkpointing; reduce batch to 4; mixed precision; fallback to chunked outer product |
| RoPE causes training instability | MEDIUM | HIGH | Start with small base frequency; gradual warmup; additive residual fallback |
| Gating fix requires full redesign | MEDIUM | HIGH | Keep additive residual as parallel path; hybrid `x + α·gate·residual` |
| Vocab projection loses information | MEDIUM | MEDIUM | Run CKA distillation in parallel; compare both approaches |
| BDH underperforms SOTA benchmarks | HIGH | MEDIUM | Emphasize unique strengths (interpretability, memory, bio-plausibility); don't claim SOTA perplexity |
| Training pipeline doesn't converge | MEDIUM | HIGH | Use proven configs from JARVIS; start with small dataset; verify on toy problem first |
| Paper not novel enough for NeurIPS | LOW | HIGH | Focus on "first fixed Hebbian linear attention" + "multi-scale + data-dependent gating" combination |
| Website breaks during updates | LOW | LOW | Git version control; Playwright tests catch regressions; rollback to last known good |

---

## Success Metrics (End of Day 9)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **4 Critical Bugs Fixed** | 4/4 | Unit tests pass for each fix |
| **Training Pipeline** | KL distillation working | KL loss > 0, not falling back to CE only |
| **Benchmark Suite** | 5/5 benchmarks complete | `benchmark_suite.py` runs end-to-end |
| **Ablation Study** | Each fix shows positive contribution | Perplexity delta > 0 for each |
| **SOTA Comparison** | Competitive on 3+ metrics | Fair comparison at equal params |
| **Publication Figures** | 6+ figures at 300 DPI | SVG + PNG exports |
| **NeurIPS Paper Draft** | 8 pages complete | LaTeX compiled, all references valid |
| **Interactive Demo** | Deployed and working | Live text gen + state viz |
| **Website Tests** | 18/18 Playwright tests pass | `npm test` |
| **Code Package** | `pip install bdh` works | Fresh env install succeeds |
| **Reproducibility** | Results from scratch | Clean env → same results ±5% |
| **Lighthouse Score** | >90 | Performance audit |

---

## Resource Requirements

| Resource | Specification | Usage |
|----------|--------------|-------|
| **GPU** | RTX 4070 (8GB) or A100 (40GB) | Training Days 5-7 |
| **RAM** | 16GB minimum | All days |
| **Storage** | 50GB free | Checkpoints, logits data, benchmarks |
| **Research Docs** | 8 docs in `research_for_gemini/` | Days 1, 4, 7, 8 |
| **Existing Logits** | `D:\logits_production_extended/` (~3GB) | Day 5 training |
| **Website** | `BDH_website/` with 18 tests | Days 1, 6, 8, 9 |
| **Teacher Models** | Qwen3.5-0.8B, Qwen3.5-4B | Day 5 distillation |

---

## Deliverables Checklist

- [ ] `bdh_v2_clean.py` — fixed architecture
- [ ] `hebbian_outer_product()` — true outer product state
- [ ] RoPE enabled in multi-scale BDH
- [ ] Data-dependent gating implementation
- [ ] `train_distillation_v2.py` — working KL distillation
- [ ] `benchmark_suite.py` — 5 benchmarks + ablation
- [ ] 6+ publication-quality figures (SVG + PNG)
- [ ] NeurIPS paper draft (LaTeX, 8 pages)
- [ ] Supplementary material (3 pages)
- [ ] Interactive demo (live text generation)
- [ ] `pip install bdh` package
- [ ] Updated website with all visualizations
- [ ] 18/18 Playwright tests passing
- [ ] Reproducibility checklist
- [ ] Code repository (clean, documented)
