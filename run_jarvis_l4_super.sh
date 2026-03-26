#!/usr/bin/env bash
set -euo pipefail

echo "=== JARVIS L4 SUPER BDH ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

python jarvis_l4_super_bdh.py \
  --teacher-model "Qwen/Qwen2.5-0.5B-Instruct" \
  --data "data/tinystories.txt" \
  --seq-len 192 \
  --batch-size 16 \
  --top-k 256 \
  --teacher-replicas 6 \
  --hours 5 \
  --max-items 500000 \
  --save-every 200 \
  --queue-size 64 \
  --learning-rate 1e-4 \
  --output "checkpoints/jarvis_l4_super/latest.pt" \
  --resume

