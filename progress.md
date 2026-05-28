# BDH Open-Access Research Operation Progress

## 🎯 Goal
Establish a 100% Open-Access, validated resource library (MIT/Stanford OCW, YouTube, Open-Source Books) for the 8 technical domains of BDH.

## 🛠️ Swarm Architecture
- **Coordinator (Main)**: Orchestrates micro-tasks and sequence.
- **Searchers (S1-S8)**: Domain-specific open-access resource discovery (YouTube, OCW, Open-books).
- **Validators (V1-V8)**: Link accessibility, paywall check, and relevance verification.
- **Scribes**: Persistence of validated resources to `references/` directory.
- **Synthesizer**: Final library compilation into `FINAL_OPEN_ACCESS_LIBRARY.md`.

## 📜 Operational Rules
1. **No Context Dumping**: Each agent handles one micro-task.
2. **Cross-Validation**: Every resource found by S-agent must be approved by V-agent.
3. **Open-Access Only**: Paywalled papers/articles are strictly rejected.
4. **Priority**: MIT/Stanford OCW > YouTube > Open-Source Books.
5. **Persistence**: All findings must be written to disk in `references/` immediately.

## 📈 Execution Trace
| Step | Task | Agent | Status | Validator | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Init `progress.md` & `references/` | Main | ✅ | N/A | Completed |
| 2 | S1: Linear Attention/SSM Search | S1 | ⏳ | V1 | Pending |
| 3 | S2: Hebbian Learning Search | S2 | ⏳ | V2 | Pending |
| 4 | S3: Multi-scale Memory Search | S3 | ⏳ | V3 | Pending |
| 5 | S4: Sparsity/MoE Search | S4 | ⏳ | V4 | Pending |
| 6 | S5: Gating Mechanism Search | S5 | ⏳ | V5 | Pending |
| 7 | S6: Recurrent Depth/ACT Search | S6 | ⏳ | V6 | Pending |
| 8 | S7: Distillation/Vocab Search | S7 | ⏳ | V7 | Pending |
| 9 | S8: BBPE Theory Search | S8 | ⏳ | V8 | Pending |
| 10 | Cross-Validation of all links | V1-V8 | ⏳ | Coordinator | Pending |
| 11 | Final Library Synthesis | Synth | ⏳ | Main | Pending |
