# BDH Science Expo Judge Master Playbook

Date: 2026-04-09
Audience: 3-person BDH team preparing for science expo judging tomorrow
Goal: Equip your team to answer high-pressure judge questions with technical depth, honesty, and confidence.

---

## 1. How to Use This Document in 24 Hours

1. Read Sections 2-4 together as a team (shared understanding).
2. Split Sections 6-9 by role (one person leads each domain).
3. Rehearse with Section 10 question bank (2 full mock rounds).
4. Run one live demo using Section 11 and validate fallback plan.
5. Memorize Section 12 "safe claims only" list.

If you only have 45 minutes, read these first:
- Section 2 (30-second + 2-minute pitch)
- Section 5 (codebase map)
- Section 7 (judge scoring logic)
- Section 10 (top 20 judge questions)
- Section 12 (what NOT to claim)

---

## 2. Your Opening Pitch (Use This Verbatim if Needed)

### 30-second version
"We are building and evaluating BDH, a brain-inspired language model architecture that uses linear attention and synaptic-style memory mechanisms. Our project explores whether multi-scale memory traces, practical training pipelines, and distillation workflows can improve long-context behavior while keeping the system interpretable and deployable. We built training, benchmarking, distributed logits generation, and a live web comparison app to test this end to end."

### 2-minute version
"Our project studies BDH, a brain-inspired architecture positioned as an alternative design space to standard Transformers. The repository includes core model code, multi-scale memory variants, multiple training paths, distributed teacher-logits generation, and a live web app comparing BDH against DistilGPT2.

Technically, the core model uses linear attention-style computation and multi-scale decay settings in the implementation. Operationally, we built scripts for quick runs, long resumes, distillation, distributed data generation, benchmark reporting, and demo reliability. For judging readiness, we focused on reproducibility, limitations, and transparent claims.

We are not claiming this beats frontier LLMs. We are showing a research engineering contribution: architecture experimentation, practical training/deployment workflows, and critical evaluation with measurable outputs and clear limitations."

---

## 3. Ground Rules: Judge-Proof Honesty

Use this framework in every answer:
- What we built
- What we measured
- What the current evidence supports
- What is still hypothesis/future work

Never blur these two classes:

### A. Verified in repo artifacts (safe to claim)
1. Webapp benchmark artifact reports BDH vs DistilGPT2 with:
   - BDH perplexity: 85.3688
   - DistilGPT2 perplexity: 33.2828
   - BDH logic_score: 20
   - DistilGPT2 logic_score: 0
   - Source: benchmarking/results/webapp_bdh_vs_distilgpt2.json
2. benchmark_results.json records baseline vs multiscale runs with small perplexity delta and throughput values.
3. webapp/server.py contains CUDA-runtime detection and automatic CPU fallback + retry.
4. webapp/server.py clips sampling logits to tokenizer vocab range to avoid out-of-range tokens.
5. implementation/multiscale_bdh.py has RoPE code present but currently disabled in forward pass with TODO comments.

### B. Draft/marketing-style claims (do not state as proven without rerunning)
Many docs in README/presentation mention values like "13x", "24x", "300x", "95% stability", etc. Treat these as provisional unless reproduced live with scripts and logs.

Judge-safe phrasing:
"Some internal planning docs include stronger target metrics; in this presentation we only stand behind numbers that are reproducible from current benchmark artifacts."

---

## 4. What BDH Is (Technical Core)

BDH in this repository is a research-oriented language-model stack with:
- Linear-attention style computation for sequence efficiency experimentation
- Multi-scale synaptic-state concept in code
- Multiple training/distillation workflows
- Demo and deployment infrastructure

### Core design points found in code/docs
1. Multi-scale configuration in implementation/multiscale_bdh.py:
   - decay_rates default: [0.95, 0.99, 0.995]
   - scale_weights default: [0.2, 0.3, 0.5]
   - hebbian_lr default: 0.001
2. Multiplicative gating is used at layer level in MultiScaleBDHLayer.
3. RoPE components exist, but application is commented out in forward pass.
4. webapp server includes production-minded behaviors:
   - model/tokenizer compatibility checks
   - fallback tokenization path for small vocab
   - runtime warnings surface
   - CPU failover for CUDA runtime failures

---

## 5. Entire Codebase Map: Which Folder Does What

Top-level purpose map for the current workspace:

1. implementation/
- Purpose: Core multi-scale BDH implementation, tokenizer work, stability configs, training helpers.
- Key files:
  - implementation/multiscale_bdh.py
  - implementation/bbpe_tokenizer.py
  - implementation/stable_config.py
  - implementation/train_multiscale.py

2. webapp/
- Purpose: Live backend + frontend demo comparing BDH and DistilGPT2.
- Key files:
  - webapp/server.py
  - webapp/static/index.html
  - webapp/static/app.js
  - webapp/static/styles.css
  - webapp/run_professor_demo.ps1

3. benchmarking/
- Purpose: Benchmark framework and result reporting.
- Key files:
  - benchmarking/benchmark_runner.py
  - benchmarking/statistical_analysis.py
  - benchmarking/results/benchmark_results.json
  - benchmarking/results/comparison_report.md
  - benchmarking/results/webapp_bdh_vs_distilgpt2.json

4. testing/
- Purpose: Integration and verification strategy notes.
- Key files:
  - testing/integration_test.py
  - testing/final_verification.md

5. demo/
- Purpose: Stage-ready demo script, rehearsal notes, fallback plans.
- Key files:
  - demo/demo_script.md
  - demo/backup_plan.md

6. presentation/
- Purpose: Poster/slides/talking-point preparation assets.
- Key files:
  - presentation/slides.md
  - presentation/poster_content.md
  - presentation/talking_points.md
  - presentation/judge_qa_prep.md

7. research/ and research_reports/
- Purpose: Strategy, distillation best practices, roadmap and analysis notes.

8. docs/
- Purpose: User and implementation docs.

9. data/
- Purpose: Training/demo data assets (includes tinystories.txt).

10. checkpoints/
- Purpose: Trained model checkpoints across multiple experiments and profiles.

11. visualization/
- Purpose: Figures for poster/demo (retention curves, state matrix, training plots).

12. Root-level train_*.py and run_*.{bat,ps1,py}
- Purpose: Many experiment entrypoints (quick, resume, distillation, hardware-specific runs).

Practical explanation to judges:
"The repo is organized by lifecycle: implementation, training scripts, evaluation, demo surface, and communication assets."

---

## 6. End-to-End Workflow (What Happens in Practice)

### A. Training workflow
1. Environment setup scripts (setup_*.bat/.ps1/.sh) and guides.
2. Pick a training mode (quick, safe, resume, distillation, hardware-specific).
3. Run train_*.py script with checkpointing enabled.
4. Monitor and resume using monitor/check/resume scripts.
5. Evaluate with benchmark scripts and/or webapp quick benchmark endpoint.

### B. Distillation + distributed logits workflow
1. Teacher logits/data generation scripts:
   - generate_logits.py, generate_logits_node*.py, distribute_logits_core.py
2. Local or multi-node execution using run_logits_node*.ps1 and related scripts.
3. Distillation training via train_distillation*.py and train_pure_distillation*.py variants.
4. Save checkpoints for inference and webapp loading.

### C. Demo/deployment workflow
1. Start web server (webapp/server.py or run_professor_demo.ps1 profile).
2. Load BDH checkpoint and baseline model.
3. Send prompt to /api/generate.
4. Server returns side-by-side outputs + runtime metrics.
5. If CUDA runtime fails, server switches to CPU and retries automatically.

---

## 7. How Judges Actually Score You (Official ISEF Signal)

From Regeneron ISEF Grand Award criteria, scoring is weighted:
- Research Question: 10
- Design and Methodology: 15
- Execution/Data Analysis: 20
- Creativity and Potential Impact: 20
- Presentation: 35
  - Poster: 10
  - Interview: 25

Key implication: Interview quality is heavily weighted. Your answer quality, ownership, limitation awareness, and clarity can decide outcomes.

### What judges are really testing psychologically
1. Authenticity: Did you really do this work?
2. Independence: Can each teammate explain core decisions?
3. Rigor: Do you separate evidence from assumptions?
4. Integrity: Are you honest about limitations and failure modes?
5. Impact thinking: Do you understand where this matters and where it does not?

---

## 8. Team-of-3 Strategy (Who Answers What)

Assign fixed primary roles:

1. Teammate A: Problem + Architecture Lead
- Owns narrative, motivation, model design, literature context.

2. Teammate B: Training + Systems Lead
- Owns scripts, data flow, checkpoints, reproducibility, hardware constraints.

3. Teammate C: Evaluation + Ethics + Demo Lead
- Owns benchmarks, claim discipline, demo reliability, integrity/ethics answers.

### Handoff protocol
- The first person gives a concise answer in <=20 seconds.
- Then says: "[Name] can add implementation detail" if needed.
- Avoid interrupting each other.
- Every answer ends with one concrete artifact reference (script, JSON, screenshot, run log).

### Team rule
If one person does not know, another person can add. Never bluff.

---

## 9. Judge Psychology in One Page

Judges usually ask questions for one of these reasons:

1. Verify authenticity
- "How did you pick these decay rates?"
- They test if you did real design work.

2. Stress test rigor
- "What controls/baselines did you run?"
- They test scientific method quality.

3. Probe limitations
- "Where does this fail?"
- They test maturity and honesty.

4. Evaluate relevance
- "Why should anyone care?"
- They test impact and practical understanding.

5. Validate team contribution
- "Who implemented what?"
- They test whether all members contributed and understand the work.

Best response pattern:
- Context -> Exact action -> Evidence -> Limitation -> Next step

---

## 10. 40 High-Probability Judge Questions + Answer Blueprint

Use the format:
- Short answer (1-2 lines)
- Evidence anchor (file/artifact)
- Limitation line

### A. Problem and motivation
1. What exact problem are you solving?
2. Why BDH instead of a standard Transformer experiment?
3. What is novel in your current repo version?
4. What did you personally contribute vs reuse?
5. Why is this relevant for science, not just coding?

### B. Architecture and algorithm
6. Explain the multi-scale memory idea in simple terms.
7. Why those decay rates?
8. How does linear attention differ from softmax attention?
9. Is your positional encoding active right now?
10. What does multiplicative gating do here?
11. How do state matrices update during inference/training?
12. Is the model interpretable in a practical way?
13. What tradeoffs did this architecture introduce?
14. Where can this architecture underperform?
15. What would you change first if you had one more month?

### C. Training and reproducibility
16. How do I reproduce one of your runs?
17. Which script is your most stable training path?
18. How do you resume interrupted runs?
19. What hardware did you target and why?
20. What are your most common training failures?
21. How do you detect overfitting/instability?
22. How do you keep experiment tracking consistent?
23. Which claims are from measured runs vs target goals?

### D. Distillation and data pipeline
24. Why use distillation in this project?
25. How do teacher logits get generated?
26. Why distributed logits across nodes?
27. How do you ensure shard consistency and resumability?
28. What is your data quality control approach?
29. What teacher-model bias risks exist?
30. How do you validate distilled student behavior?

### E. Evaluation and benchmark honesty
31. Which benchmark numbers are verified right now?
32. Did BDH beat DistilGPT2 on perplexity in your artifact?
33. Why does logic_score differ from perplexity results?
34. How do you avoid cherry-picking metrics?
35. What benchmark is still missing for publication-level confidence?

### F. Demo reliability and deployment
36. What happens if CUDA fails during live demo?
37. How do you avoid tokenizer/model vocab mismatch errors?
38. What if internet/model download fails at venue?
39. How do you keep demo responsive under constraints?
40. What is your fallback if the main demo breaks?

### Example answer pattern for a hard question
Question: "Did your model beat DistilGPT2?"
- Short answer: "On the current artifact, BDH did not beat DistilGPT2 on perplexity, but scored higher on our logic_score metric."
- Evidence: benchmarking/results/webapp_bdh_vs_distilgpt2.json
- Limitation: "This is a small sample benchmark and we treat it as directional, not final proof."
- Next step: "We are expanding evaluations and standardizing metric suites."

---

## 11. Live Demo Runbook (Tomorrow Morning)

### Primary run
1. Activate environment.
2. Start server:
   - python webapp/server.py --host 0.0.0.0 --port 8000 --checkpoint <chosen_checkpoint> --data data/tinystories.txt
3. Open localhost:8000.
4. Run 2 prepared prompts and one judge prompt.

### Reliability checks before judges arrive
1. Hit /api/health.
2. Generate once with short prompt.
3. Confirm both BDH and Distil outputs render.
4. Keep a screenshot backup of last successful run.

### Fallback script (if GPU fails)
1. Let auto CPU fallback happen (server already supports this).
2. If needed, restart with a smaller/stable profile via webapp/run_professor_demo.ps1.
3. Continue with pre-captured benchmark artifacts and explain transparently.

---

## 12. Safe Claims Only (Memorize)

You can safely say:
1. "We built an end-to-end research stack: model implementation, training scripts, distillation/logits pipelines, benchmarking, and web demo."
2. "Our official benchmark artifacts currently show mixed outcomes: stronger logic_score for BDH in one artifact, but weaker perplexity than DistilGPT2."
3. "We engineered robustness features, including CUDA-to-CPU failover and tokenizer-vocab safeguards."
4. "We are actively separating verified measurements from aspirational targets in docs."

Do not claim without fresh proof:
1. "13x/24x/300x improvements" as settled results.
2. "State-of-the-art performance".
3. "Scales to 1B+ demonstrated" unless you show run artifacts.
4. "Always better than baselines".

---

## 13. If a Judge Pushes Hard: Best Responses

Use these high-integrity responses:

1. "Great point. We have not fully validated that yet. Here is what we have measured so far..."
2. "That result appears in planning docs, but we are only claiming what is reproducible from current artifact files."
3. "This is an active limitation in our implementation; here is the exact file where it appears and our planned fix."
4. "We designed this as a research prototype, not as a production claim of superiority over frontier models."

These responses increase trust.

---

## 14. One-Night Rehearsal Plan

1. Round 1 (45 min): Technical grilling
- One teammate plays aggressive judge.
- Use Section 10 only.

2. Round 2 (30 min): Non-technical judge
- Explain architecture in simple language.
- Ban jargon unless defined in one sentence.

3. Round 3 (20 min): Failure simulation
- Assume demo GPU failure and internet failure.
- Practice fallback narrative.

4. Final 10 min: Claims lock
- Team agrees on exact safe claims and banned claims.

---

## 15. External Judge-Research Sources Used

Primary sources:
1. ISEF Grand Award criteria (official scoring + interview emphasis):
   - https://www.societyforscience.org/isef/grand-award/criteria/
2. ISEF judge page (judge qualifications, process, interview windows):
   - https://www.societyforscience.org/isef/grand-award/
3. ISEF international rules and AI guidance links:
   - https://www.societyforscience.org/isef/international-rules/
4. Society for Science scientific integrity policy:
   - https://www.societyforscience.org/scientific-integrity/
5. ISEF FAQ (judge profile and process context):
   - https://www.societyforscience.org/isef/faq/

Secondary context source:
6. Reinventing Science Fairs (Issues in Science and Technology):
   - https://issues.org/reinventing-science-fairs/

---

## 16. Final Judge-Winning Mindset

Your team does not need to claim perfection.
Your team needs to demonstrate:
1. Real ownership of work.
2. Real understanding of tradeoffs.
3. Real integrity in reporting evidence.

That combination is exactly what strong judges reward.
