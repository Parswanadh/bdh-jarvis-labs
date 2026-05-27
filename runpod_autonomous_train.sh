#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "BDH RUNPOD AUTONOMOUS DISTILLATION"
echo "=========================================="

cd "$(dirname "$0")"

if [[ -z "${TRAIN_JSONL:-}" || -z "${VAL_JSONL:-}" || -z "${OUT_CHECKPOINT:-}" ]]; then
  echo "Required env vars:"
  echo "  TRAIN_JSONL=/workspace/BDH/data/distill/world_train.jsonl"
  echo "  VAL_JSONL=/workspace/BDH/data/distill/world_val.jsonl"
  echo "  OUT_CHECKPOINT=/workspace/BDH/checkpoints/autonomous/cycle_1.pt"
  echo ""
  echo "Optional env vars:"
  echo "  BASE_CHECKPOINT=/workspace/BDH/checkpoints/safe/laptop_safe_11L.pt"
  echo "  BATCH_SIZE=16"
  echo "  MAX_SEQ_LEN=256"
  echo "  EPOCHS=1"
  echo "  LR=1e-4"
  exit 1
fi

echo "[1/2] Installing dependencies..."
pip install -q torch transformers accelerate bitsandbytes sentencepiece scipy

echo "[2/2] Starting autonomous cycle..."
python run_autonomous_cycle.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --checkpoint "${BASE_CHECKPOINT:-}" \
  --out-checkpoint "${OUT_CHECKPOINT}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-seq-len "${MAX_SEQ_LEN:-256}" \
  --epochs "${EPOCHS:-1}" \
  --lr "${LR:-1e-4}"

echo "=========================================="
echo "AUTONOMOUS CYCLE COMPLETED"
echo "=========================================="
