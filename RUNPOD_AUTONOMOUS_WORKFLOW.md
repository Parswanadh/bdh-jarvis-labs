# RunPod Autonomous Workflow (Web + Books Distillation)

## Goal

Run safe autonomous improvement cycles for BDH using curated world data:
- `web` (fresh world updates)
- `books` (high-quality long-form knowledge)
- `papers` (technical rigor)

The model improves through distillation, meta-cognitive confidence scoring, and rollback safety checks.

---

## 1) Prepare Distillation Data

Create JSONL files:
- `data/distill/world_train.jsonl`
- `data/distill/world_val.jsonl`

Each line:

```json
{"prompt":"...", "teacher_response":"...", "source":"books", "evidence":["..."], "quality":0.95}
```

`source` should be one of:
- `web`
- `books`
- `papers`
- `synthetic`

`quality` is your curation confidence in `[0.0, 1.0]`.

---

## 2) Launch on RunPod

Inside your pod:

```bash
cd /workspace/BDH
chmod +x runpod_autonomous_train.sh
```

Set environment variables:

```bash
export TRAIN_JSONL=/workspace/BDH/data/distill/world_train.jsonl
export VAL_JSONL=/workspace/BDH/data/distill/world_val.jsonl
export BASE_CHECKPOINT=/workspace/BDH/checkpoints/safe/laptop_safe_11L.pt
export OUT_CHECKPOINT=/workspace/BDH/checkpoints/autonomous/cycle_1.pt
export BATCH_SIZE=16
export MAX_SEQ_LEN=256
export EPOCHS=1
export LR=1e-4
```

Run:

```bash
./runpod_autonomous_train.sh
```

---

## 3) Safety and Acceptance Logic

In `implementation/autonomous_self_improvement.py`:
- Computes confidence/uncertainty from output entropy.
- Weights samples by source reliability and quality.
- Trains mostly where confidence is high enough.
- Evaluates before/after cycle.
- Rolls back automatically if validation regresses above threshold.

This gives controlled self-improvement instead of blind self-editing.

---

## 4) Suggested Multi-Cycle Schedule

Use short cycles first:
1. Cycle 1-3: `EPOCHS=1`, strict safety.
2. Cycle 4-10: increase data diversity (`web + books + papers`).
3. Later: tune thresholds only after stable improvements.

Save each cycle checkpoint separately:
- `cycle_1.pt`, `cycle_2.pt`, ...

Never overwrite your best-known-safe checkpoint.

---

## 5) What "Thinking with Proof" Means in Practice

Your intuition is right, but "proof" should be interpreted operationally:
- The model should produce **reasoning traces + evidence references**.
- Meta-cognition should expose **confidence and uncertainty**.
- Training should reward **grounded answers** (evidence-backed), not just fluent text.

That is a practical path toward verifiable behavior without claiming formal theorem-prover guarantees.
