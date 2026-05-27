# Logits Generation Optimization & Monitoring Guide

## Current Status

**Run 1 (Baseline):** Completed 10-05-2026
- Dataset: 43,076 rows
- Throughput: **6.09 rows/sec**
- Duration: ~2 hours
- Output: 1.71 GB (43 chunks)
- GPU util: **15-19%** ⚠️ SEVERELY UNDERUTILIZED

**Run 2 (In Progress):** Started 17:35 UTC
- Target: 4 more hours → D:\logits_production
- Expected completion: ~21:30 UTC

## Identified Bottlenecks

### 1. GPU Utilization Low (15-19%)
**Problem:** Batch size 5 is too small for Qwen3.5-4B (only 5 sequences processed in parallel)

**Solution:** Increase batch size → 12-16
- BS5: 6 rows/sec, 15% GPU util
- BS12: ~14 rows/sec, 50%+ GPU util (estimated)
- BS16: ~18 rows/sec, 70%+ GPU util (estimated)

### 2. Flash Attention Not Installed
**Problem:** Warning in logs: "The fast path is not available because one of the required library is not installed"

**Solution:** Install flash-linear-attention
```bash
pip install flash-attn
```
**Expected speedup:** 3-5x on transformer attention

### 3. Synchronous Batch Processing
**Problem:** GPU waits for batch preprocessing, CPU idle during inference

**Solution:** Async prefetch (✅ ADDED to generate_logits_new_data.py)
- Background thread loads & transfers next batch while GPU processes current
- Reduces GPU idle time ~20%

### 4. Top-K=128 Too Large
**Problem:** Top-k=128 indices require more memory + slower computation

**Solution:** Reduce to top-k=32 (sufficient for distillation)
- BS5, K128: 6 rows/sec
- BS5, K32: ~12 rows/sec (2x faster)

### 5. Chunk Size 1024 Causes I/O Overhead
**Problem:** Small chunks = frequent disk writes (blocks GPU)

**Solution:** Increase to chunk_rows=2048 or 4096
- Fewer writes, larger sequential I/O
- Less context switching

## Optimization Strategy (Staged)

### Stage 1: Immediate (Test Today)
```bash
# Run turbo test (bs=12, k=32, chunks=2048, 10 min)
powershell -File .\run_turbo_test.ps1

# Expected result: 12-15 rows/sec (2-2.5x baseline)
```

### Stage 2: Install Flash Attention (if available)
```bash
pip install flash-attn
# Requires: CUDA 11.8+, torch 2.0+, GPU compute capability 7.5+
# RTX 4070: ✅ Compatible (compute capability 8.6)
```

### Stage 3: Full Optimized Run
```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_logits_live.ps1 `
  -Data data\chat_reasoning_new_v2_45k.jsonl `
  -OutputRoot D:\logits_production_v2 `
  -BatchSize 16 `
  -TopK 32 `
  -ChunkRows 4096 `
  -RunHours 4
```

**Expected result:** 15-20 rows/sec (2.5-3.3x baseline)
- **Total time for 43k rows:** ~35 min (vs 2 hours baseline)
- **Extended to 4 hours:** ~240k rows (5.6x more data)

## Live Monitoring

### Option 1: Built-in Monitor
```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\monitor_logits_live.ps1
```
Shows:
- GPU util%, mem, temp, power
- Process CPU/RAM
- Chunks written
- Heartbeat: rows/sec, ETA

### Option 2: Watch Log File
```bash
# Find latest log
ls $env:TEMP\logits-live-*.log | sort -Descending | select -First 1 | % { Get-Content -Tail 50 -Wait $_ }
```

### Option 3: Check Heartbeat (Manual)
```bash
# While running
cat D:\logits_production\run_*/heartbeat.json | ConvertFrom-Json
```

### Key Metrics to Watch

| Metric | Good | Poor | Action |
|--------|------|------|--------|
| **GPU util%** | >50% | <20% | Increase batch size |
| **GPU power** | >50W | <10W | Underutilized; boost BS |
| **rows/sec** | >12 | <7 | Check for thermal throttle |
| **GPU temp** | <70°C | >80°C | Reduce batch size, improve cooling |

## Expected Performance Timeline

### Baseline (BS=5, K=128)
- Throughput: 6 rows/sec
- 43k rows: 2 hours
- 4-hour extended: 86k rows

### Target (BS=16, K=32, Flash Attention)
- Throughput: 18-20 rows/sec
- 43k rows: 35 minutes
- 4-hour extended: 250k+ rows

**Improvement: ~4-5x faster = ~5-6x more training data**

## Troubleshooting

### OOM Error on Batch Size Increase
- BS too large for 8GB VRAM
- Solution: `BS = 14, 12, 10` (binary search)
- Also reduce seq_len from 128 → 96

### GPU Stays at Low Utilization After Increase
- Bottleneck shifted to preprocessing/I/O
- Solution: Increase num_workers (2 → 4) or prefetch_factor

### Flash Attention Install Fails
- Old CUDA/torch version
- Solution: Check `torch.cuda.get_device_capability()` reports (8, 6) for RTX4070
- Install: `pip install --no-build-isolation flash-attn`

### Throughput Plateaus After Optimization
- Likely hitting tokenizer, I/O, or preprocessing ceiling
- Next: profile with `py-spy record -o profile.svg -r 100 -- python generate_logits_new_data.py ...`

## Files & Commands

**Main scripts:**
- `generate_logits_new_data.py` - Core generation (✅ async prefetch added)
- `run_logits_live.ps1` - One-command runner
- `monitor_logits_live.ps1` - Live monitor
- `analyze_logits_run.py` - Post-run validation

**Configs for Run 2 Continuation:**
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_logits_live.ps1 `
  -Data data\chat_reasoning_new_v2_45k.jsonl `
  -TeacherModel D:\projects\BDH\Qwen3.5-4B `
  -OutputRoot D:\logits_production `
  -BatchSize 12 `
  -NumWorkers 2 `
  -RunHours 4 `
  -AllowReuse
```

**Test Turbo Config (validates speedup):**
```powershell
powershell -File .\run_turbo_test.ps1
```

## Next Phase: Student Training

Once logits validated (Run 1 + Run 2 combined):

1. Load all chunks from `D:\logits_production\run_*/logits_chunk_*.npz`
2. Implement distillation loss: `KL(student || teacher) + supervised_cross_entropy * assistant_mask`
3. Train ~100M parameter BDH student model
4. Measure perplexity → compare vs baseline

