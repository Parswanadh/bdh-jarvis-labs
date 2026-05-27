# BDH Plan

## Goal
Build a high-confidence understanding of the current BDH repository, then choose the best quality-maximizing path for a strict 6-7 hour NVIDIA L4 session.

## What is happening in this codebase
1. **Core model line**: `bdh_gpu_10m.py` (baseline) and `implementation/multiscale_bdh.py` (current improved path).
2. **Training script sprawl**: many `train_*.py` files exist; not all are equally current or reliable.
3. **Jarvis/L4 path exists**: `jarvis_l4_super_bdh.py` is an explicitly time-bounded distillation pipeline with checkpoint+resume.
4. **Docs and code drift**: some docs point to scripts that are now weaker or broken for immediate use.

## Primary execution plan (for L4 6-7h)
1. **Primary route**: use `jarvis_l4_super_bdh.py` with a short calibration phase, then a full run.
2. **Calibration phase (short pilot)**:
   - Validate VRAM headroom, queue behavior, and early loss trend.
   - Tune `batch-size`, `teacher-replicas`, `top-k` before committing full budget.
3. **Full run phase**:
   - Run for remaining session budget with periodic checkpointing.
   - Keep resume safety enabled.
4. **Post-run extraction**:
   - Keep `latest.pt` and best checkpoint for downstream demo/eval.

## Fallback plan
If the primary queue/replica setup is unstable on the target environment, switch to:
- `train_qwen_sota_2hr_efficient.py` (repeatable, simpler loop, compile-clean).

## Avoid list (current state)
- `train_a100.py` (syntax error)
- Scripts that rely on naive vocab projection for "true distillation" quality claims
- Jumping between many `train_*.py` variants mid-session (setup churn wastes the 6-7h window)

## Decision gates during run
1. If VRAM pressure or instability appears in pilot: reduce replicas and/or batch size.
2. If throughput is healthy but learning weak: increase teacher signal density (`top-k`) if memory allows.
3. If the run is unstable after tuning: switch to fallback immediately instead of burning more time.

