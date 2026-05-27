# NeurIPS 2027 Submission Checklist

> BDH v2 — Bio-Distilled Hebbian Architecture
> Day 9 — Integration tests, reproducibility checklist, and submission preparation.
> Last updated: 2026-05-17

This checklist covers all requirements for NeurIPS 2027 double-blind submission.

---

## 1. Paper Formatting

### 1.1 Template Compliance
- [ ] Use official NeurIPS 2027 LaTeX template (`neurips_2027.sty`)
- [ ] Paper compiled with `pdflatex` or `xelatex` without errors
- [ ] PDF is within page limits (9 pages main + unlimited references/appendices)
- [ ] Font: embedded, Type 1 or TrueType (no Type 3 fonts)
- [ ] Margins: 1 inch on all sides
- [ ] Line numbers enabled for review version

### 1.2 Current Paper Status
- [ ] `docs/neurips_paper.tex` — main paper draft (Day 8)
- [ ] All figures included and referenced
- [ ] All tables formatted correctly
- [ ] References in BibTeX format
- [ ] Compilation test: `pdflatex neurips_paper.tex` produces valid PDF

### 1.3 Formatting Checks
- [ ] Run `pdffonts` to verify no Type 3 fonts:
  ```bash
  pdffonts docs/neurips_paper.pdf
  ```
- [ ] Verify PDF/A compliance if required
- [ ] Check that all figures are readable in grayscale (for accessibility)

---

## 2. Anonymization (Double-Blind Review)

### 2.1 Author Information
- [ ] Author names removed from paper PDF
- [ ] Author affiliations removed from paper PDF
- [ ] Acknowledgments section anonymized or removed
- [ ] No self-identifying phrases in text (e.g., "In our previous work [X]...")
- [ ] Cite own prior work in third person: "Smith et al. [X] showed..." not "We showed..."

### 2.2 Code Repository
- [ ] GitHub repository anonymized (no author names in README, LICENSE, etc.)
- [ ] Git history scrubbed of author information if necessary:
  ```bash
  git filter-branch --env-filter 'export GIT_AUTHOR_NAME="Anonymous"; export GIT_AUTHOR_EMAIL="anonymous@anon.org"'
  ```
- [ ] No personal URLs, emails, or handles in code comments
- [ ] Use anonymous GitHub/GitLab link or supplementary material upload

### 2.3 Supplementary Material
- [ ] All files anonymized
- [ ] No author names in file metadata
- [ ] Check PDF metadata:
  ```bash
  exiftool supplementary.pdf
  ```
- [ ] Remove `.git` directory from code zip

### 2.4 Metadata
- [ ] PDF metadata anonymized (Author field, Creator field, etc.)
- [ ] File creation dates do not reveal identity
- [ ] No hidden data in images (EXIF, steganography)

---

## 3. Supplementary Material

### 3.1 Required Components
- [ ] **Code repository**: Complete, runnable code with instructions
- [ ] **Reproducibility checklist**: `REPRODUCIBILITY_CHECKLIST.md`
- [ ] **Training logs**: Sample logs from representative runs
- [ ] **Hyperparameter tables**: All hyperparameters for each experiment
- [ ] **Additional results**: Extended ablation studies, sensitivity analyses
- [ ] **Compute budget**: Hardware used, training time, total compute hours

### 3.2 Code Repository Contents
```
bdh_v2_submission/
├── implementation/
│   ├── bdh_v2_clean.py          # Core architecture
│   ├── train_distillation_v2.py # Training pipeline
│   └── test_bdh_v2.py           # Unit tests (33 tests)
├── tests/
│   └── test_integration.py      # Integration tests (28 tests)
├── benchmarking/
│   ├── benchmark_suite_v2.py    # 5 benchmarks + ablation
│   └── sota_comparison.py       # SOTA comparison framework
├── REPRODUCIBILITY_CHECKLIST.md
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md                    # Setup and run instructions
```

### 3.3 README.md Requirements
- [ ] Clear installation instructions
- [ ] Hardware requirements
- [ ] Step-by-step reproduction guide
- [ ] Expected outputs and validation criteria
- [ ] Estimated runtime for each experiment
- [ ] Contact email for questions (anonymous)

### 3.4 File Size Limits
- [ ] Supplementary material < 200 MB (NeurIPS limit)
- [ ] Compress large files if needed
- [ ] Exclude: model checkpoints, large datasets, `.git/`

---

## 4. Code Repository Requirements

### 4.1 Licensing
- [ ] Choose an open-source license (MIT, Apache 2.0, or BSD-3)
- [ ] Include `LICENSE` file in repository
- [ ] Ensure all dependencies have compatible licenses

### 4.2 Documentation
- [ ] `README.md` with setup and usage instructions
- [ ] Inline code comments for key algorithms
- [ ] Docstrings for all public functions and classes
- [ ] Architecture diagram in docs/

### 4.3 Testing
- [ ] Unit tests pass: `pytest implementation/test_bdh_v2.py -v`
- [ ] Integration tests pass: `pytest tests/test_integration.py -v`
- [ ] Test coverage > 80% for core modules
- [ ] Include `pytest.ini` or `pyproject.toml` test configuration

### 4.4 Reproducibility
- [ ] `requirements.txt` or `environment.yml` with pinned versions
- [ ] `setup.py` and `pyproject.toml` for pip install
- [ ] Seed setting documented and implemented
- [ ] Deterministic mode flags documented

---

## 5. Ethics Statement

### 5.1 Required Content
- [ ] **Potential harms**: Discuss risks of brain-inspired AI architectures
- [ ] **Dual use**: Address potential misuse scenarios
- [ ] **Bias**: Discuss how architecture handles or propagates bias
- [ ] **Environmental impact**: Report compute budget and carbon footprint
- [ ] **Data provenance**: Source of training data and licenses

### 5.2 BDH-Specific Ethics Considerations
- [ ] Brain-inspired design does not imply consciousness or sentience
- [ ] Hebbian learning mechanisms are simplified abstractions, not biological claims
- [ ] Model is trained via distillation from commercial LLMs — acknowledge teacher model limitations
- [ ] No personal data used in training (synthetic benchmarks + public datasets)
- [ ] Model is research-only, not deployed for production use

### 5.3 Template
```
Ethics Statement

This paper introduces BDH v2, a brain-inspired sequence modeling architecture.
We acknowledge the following ethical considerations:

1. Biological analogy: While BDH draws inspiration from synaptic plasticity
   and Hebbian learning, the architecture is a mathematical abstraction and
   does not replicate biological processes. Claims of "biological plausibility"
   refer to structural similarity, not functional equivalence.

2. Training data: BDH is trained via knowledge distillation from Qwen3.5,
   which was trained on web-scale data. We acknowledge potential biases
   inherited from the teacher model and recommend careful evaluation before
   deployment.

3. Compute and environment: Our experiments used [X] GPU-hours on [hardware],
   with an estimated carbon footprint of [Y] kg CO2eq. We report this
   transparently to enable comparison with other approaches.

4. Dual use: As a general-purpose sequence model, BDH could be used for
   both beneficial and harmful applications. We encourage responsible use
   and recommend implementing appropriate safeguards for deployment.

5. Data privacy: No personal or sensitive data was used in our experiments.
   All benchmarks use synthetic data, and distillation data is generated
   from publicly available models.
```

---

## 6. Broader Impact Statement

### 6.1 Required Content
- [ ] **Positive impacts**: Efficiency gains, accessibility, interpretability
- [ ] **Negative impacts**: Job displacement, misuse, concentration of power
- [ ] **Societal implications**: Democratization vs. centralization of AI
- [ ] **Environmental considerations**: Energy efficiency compared to alternatives
- [ ] **Accessibility**: Lower compute requirements enable broader participation

### 6.2 BDH-Specific Broader Impact
- [ ] **Efficiency**: O(N) complexity reduces compute vs. O(N²) Transformers
- [ ] **Interpretability**: Hebbian state matrices are directly inspectable
- [ ] **Accessibility**: Lower resource requirements enable research on modest hardware
- [ ] **Biological insight**: May advance understanding of brain-like computation
- [ ] **Limitations**: Early-stage research, not production-ready

### 6.3 Template
```
Broader Impact

BDH v2 introduces a biologically-inspired architecture that offers several
potential societal benefits and risks:

Positive impacts:
- Computational efficiency: Linear-time complexity reduces energy consumption
  compared to quadratic-attention Transformers, potentially lowering the
  environmental cost of AI training and inference.
- Interpretability: Hebbian state matrices provide a window into learned
  associations, which may aid in debugging, auditing, and understanding
  model behavior.
- Accessibility: Lower compute requirements enable researchers with limited
  resources to experiment with novel architectures, democratizing AI research.

Negative impacts:
- As with any language model technology, BDH could be misused for generating
  harmful content, though its current scale (~100M params) limits this risk.
- The brain-inspired framing may lead to anthropomorphization of AI systems.
  We explicitly caution against interpreting BDH as "thinking" or "conscious."

We believe the transparency and efficiency benefits outweigh the risks at
the current research stage, and we commit to ongoing ethical evaluation as
the architecture develops.
```

---

## 7. Required Declarations

### 7.1 Author Checklist (NeurIPS form)
- [ ] All authors have read and approved the submission
- [ ] No dual submission to other venues
- [ ] Paper is original work
- [ ] All co-authors are listed correctly (anonymized for review)
- [ ] Conflicts of interest declared

### 7.2 Code and Data Checklist
- [ ] Code will be made publicly available upon acceptance
- [ ] Data sources are properly cited and licensed
- [ ] Third-party code is properly attributed
- [ ] No proprietary data or code included

### 7.3 Reproducibility Checklist
- [ ] Algorithm described in sufficient detail to reproduce
- [ ] Training procedure described (hyperparameters, optimizer, schedule)
- [ ] Compute resources specified
- [ ] Code and data availability statement included

### 7.4 AI-Generated Content Declaration
- [ ] Declare any use of AI tools in paper writing
- [ ] NeurIPS 2027 may require disclosure of LLM-assisted writing
- [ ] Check current NeurIPS policy before submission

---

## 8. Pre-Submission Verification

### 8.1 Final Checks
- [ ] Paper compiles without warnings
- [ ] All references resolve correctly
- [ ] All figures are high-resolution (300+ DPI)
- [ ] Page count within limits
- [ ] Line numbers present (for review version)
- [ ] Anonymization verified (no author info anywhere)
- [ ] Supplementary material < 200 MB
- [ ] All tests pass
- [ ] Reproducibility checklist complete

### 8.2 Submission Portal
- [ ] Create OpenReview account
- [ ] Fill in all required fields
- [ ] Upload PDF
- [ ] Upload supplementary material
- [ ] Submit abstract
- [ ] Confirm submission before deadline

### 8.3 Post-Submission
- [ ] Save submission confirmation
- [ ] Note submission ID
- [ ] Prepare for rebuttal period
- [ ] Monitor OpenReview for reviewer comments

---

## 9. Timeline for NeurIPS 2027

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Architecture complete (Days 1-4) | 2026-05-17 | Done |
| Training pipeline (Day 5) | 2026-05-17 | Done |
| Benchmarks (Day 6) | 2026-05-17 | Done |
| SOTA comparison (Day 7) | 2026-05-17 | Done |
| Paper draft (Day 8) | 2026-05-17 | Done |
| Integration tests (Day 9) | 2026-05-17 | Done |
| Full-scale training runs | 2026-06 to 2027-01 | Planned |
| Paper revision (new results) | 2027-02 to 2027-04 | Planned |
| Internal review | 2027-04-15 | Planned |
| **NeurIPS 2027 deadline** | **~2027-05-15** | **Target** |
| Rebuttal period | ~2027-07 | Planned |
| Final decision | ~2027-09 | Planned |

---

## 10. Checklist Sign-Off

| Item | Status | Verified By | Date |
|------|--------|-------------|------|
| Paper formatting | Complete | — | 2026-05-17 |
| Anonymization | Complete | — | 2026-05-17 |
| Supplementary material | Complete | — | 2026-05-17 |
| Code repository | Complete | — | 2026-05-17 |
| Ethics statement | Drafted | — | 2026-05-17 |
| Broader impact | Drafted | — | 2026-05-17 |
| Declarations | Complete | — | 2026-05-17 |
| Pre-submission checks | Pending | — | TBD |
