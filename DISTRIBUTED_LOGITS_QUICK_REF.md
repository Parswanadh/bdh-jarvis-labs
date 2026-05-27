# Distributed Teacher Logits Generation - Quick Reference

## Quick Start

### 1. Setup (one-time)
\\\powershell
# Set GitHub PAT
$env:GITHUB_TOKEN_BDH = "ghp_your_token"
[System.Environment]::SetEnvironmentVariable("GITHUB_TOKEN_BDH", "ghp_...", "User")
\\\

### 2. Run on three laptops
**Laptop 1:** .\run_logits_node1.ps1 -GitPush
**Laptop 2:** .\run_logits_node2.ps1 -GitPush
**Laptop 3:** .\run_logits_node3.ps1 -GitPush

### 3. Test (optional)
\\\powershell
python test_logits_distributed.py
\\\

## Shard Assignment (Deterministic)

- **Node 1:** Stories 0 - 33,333 (33,334 stories)
- **Node 2:** Stories 33,334 - 66,666 (33,333 stories)
- **Node 3:** Stories 66,667 - 99,999 (33,333 stories)

All assignments are computed the same way on each node (no coordinator needed).

## Output Structure

\\\
data/teacher_logits_shards/
├── metadata.json              # Global registry
├── node1/
│   ├── progress.json          # Resume state
│   ├── shard_node1_000.npz    # 500MB chunk
│   ├── shard_node1_000.sha256 # Integrity checksum
│   └── ...
├── node2/ ... 
└── node3/ ...
\\\

## Progress Tracking

Each node tracks: stories_processed, chunks_written, current_batch_idx, status

**Resume:** Automatically continues from last complete chunk

**Idempotent:** Running same node twice is safe (detects completion)

## Git Push

- Requires: \GITHUB_TOKEN_BDH\ environment variable
- Token never stored in code or logs
- Commit message: "[logits] nodeN shards - ISO timestamp"
- Use --git-push flag to enable

## Failure Recovery

**Case 1: Crash during generation**
- Progress saved atomically to progress.json
- Resume: Run with --resume flag

**Case 2: Git push failed**
- Generation still complete locally
- Retry: Run with --git-push flag again

**Case 3: Disk full during chunk**
- Incomplete chunk left (no SHA256)
- Delete incomplete chunk, resume

## Acceptance Tests

Run before deployment:
\\\powershell
python test_logits_distributed.py
\\\

Checks:
- Deterministic shard ranges
- Metadata schema
- Chunk integrity (SHA256)
- Progress tracking
- Git integration
- Resumability
- Error recovery

## Monitoring

\\\powershell
# Check Node 1 progress
Get-Content data/teacher_logits_shards/node1/progress.json | ConvertFrom-Json

# List chunks
ls -Recurse data/teacher_logits_shards/ -Filter "*.npz"

# Verify checksums
ls -Recurse data/teacher_logits_shards/ -Filter "*.npz" |
  ForEach-Object {
    if (Test-Path ".sha256") {
      Write-Host "✓ "
    } else {
      Write-Host "✗ MISSING: "
    }
  }
\\\

## Script Contracts

### generate_logits_node1.py
- Args: --stories N, --resume, --git-push
- Env: GITHUB_TOKEN_BDH (required if --git-push)
- Exit: 0 (success), 2 (gen failed), 3 (push failed)
- Output: data/teacher_logits_shards/node1/

### generate_logits_node2.py
- Same as node1, but output to node2/

### generate_logits_node3.py
- Same as node1, but output to node3/

## Chunk Format

Each .npz contains:
- input_ids: shape (batch, 512)
- logits: shape (batch, 512, vocab_size)
- batch_count, seq_len, vocab_size

Load:
\\\python
import numpy as np
data = np.load("shard_node1_000.npz")
input_ids = data["input_ids"]
logits = data["logits"]
\\\

## Files Created

- distribute_logits_core.py (core framework)
- generate_logits_node{1,2,3}.py (node scripts)
- run_logits_node{1,2,3}.ps1 (PowerShell wrappers)
- test_logits_distributed.py (acceptance tests)

## Files Required

- data/tinystories.txt (100K+ stories)
- tokenizers/bbpe_tokenizer.json (student tokenizer)
- (Models auto-downloaded: Qwen 2.5 0.8B, ~3GB per node)
