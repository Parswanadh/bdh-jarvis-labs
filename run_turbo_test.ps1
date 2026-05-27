#!/usr/bin/env pwsh
"""
Test turbo logits generation with larger batch size (12) + async prefetch.
Run 10 minutes to measure speedup.
"""

$env:DISABLE_ADDMM_CUDA_LT = "1"
$pythonExe = "D:\.conda\envs\bdh-fest\python.exe"

Write-Host "=== TURBO TEST: Batch=12, Top-K=32, Chunks=2048 ===" -ForegroundColor Cyan
Write-Host "Expected speedup: 2.4x (bs 12 vs bs 5) + prefetch overhead reduction"
Write-Host ""

& $pythonExe generate_logits_new_data.py `
    --data data\chat_reasoning_new_v2_45k.jsonl `
    --teacher-model D:\projects\BDH\Qwen3.5-4B `
    --output-root data\teacher_logits_turbo_test `
    --seq-len 128 `
    --top-k 32 `
    --batch-size 12 `
    --num-workers 2 `
    --chunk-rows 2048 `
    --run-hours 0.17 `
    --gpu-dtype fp16 `
    --progress-every 2 `
    --heartbeat-every-sec 5 `
    --allow-reuse
