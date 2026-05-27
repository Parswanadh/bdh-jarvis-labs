# Professor Meeting Brief - BDH Project

## 1) What this project is doing

You are building and evaluating a **brain-inspired language model family (BDH / MultiScaleBDH)** with:

- Linear-attention style computation in BDH blocks.
- Multi-timescale Hebbian memory traces (`decay_rates=[0.95, 0.99, 0.995]`).
- Multiple training branches (safe, ultimate, 1hour BBPE, distillation variants).
- A real side-by-side comparison webapp against DistilGPT2.

Core implementation and training entry points:

- `implementation/multiscale_bdh.py`
- `train_production_bdh.py`
- `benchmark_vs_baselines.py`
- `webapp/server.py`
- `webapp/static/app.js`

## 2) Where your comparison tool is

Your live comparison system is the web app:

- Backend: `webapp/server.py`
- Frontend: `webapp/static/index.html`
- Live metrics + model spec rendering: `webapp/static/app.js`

API endpoints used by the UI:

- `GET /api/config` (model cards/specs)
- `POST /api/generate` (side-by-side generation)
- `POST /api/benchmark/quick` (quick perplexity/logic benchmark)

## 3) One-command demo launcher (ready for professor)

Use this script:

- `webapp/run_professor_demo.ps1`

Examples:

```powershell
# Latest checkpoint profile
pwsh -File webapp/run_professor_demo.ps1 -Profile latest -EagerLoad

# Stable BBPE profile (stronger logic score in quick benchmark)
pwsh -File webapp/run_professor_demo.ps1 -Profile stable -EagerLoad
```

Then open:

- `http://127.0.0.1:8000`

## 4) Real model timeline (latest checkpoints)

Sorted by timestamp (newest first):

1. `checkpoints/safe/laptop_safe_11L.pt` - 2026-03-20 05:25:10
2. `checkpoints/safe/laptop_safe.pt` - 2026-03-20 03:15:02
3. `checkpoints/ultimate/hatchling_latest.pt` - 2026-03-14 23:02:30
4. `checkpoints/final/latest.pt` - 2026-03-14 15:19:29

## 5) Architecture snapshot (real values)

From checkpoint inspection:

- `safe/laptop_safe_11L.pt`: 11 layers, 256 hidden, 8 heads, FFN 1024, seq 192, step 800.
- `safe/laptop_safe.pt`: 8 layers, 256 hidden, 8 heads, FFN 1024, seq 192, step 17092.
- `ultimate/hatchling_latest.pt`: 12 layers, 256 hidden, 8 heads, FFN 1024, seq 224, step 100.
- `1hour_training/final_model.pt`: 8 layers, 512 hidden, 8 heads, FFN 2048, seq 512.

Tokenizer/model vocab note for latest safe checkpoints:

- model vocab: 248320
- tokenizer vocab: 248077
- padded gap: 243 tokens (handled in webapp backend)

## 6) Real quick benchmark results (sample_count=20, max_len=192, seed=42)

Baseline for all rows: DistilGPT2 perplexity = 35.0488, logic_score = 0.

| Checkpoint | Step | BDH Perplexity | BDH Logic Score | Notes |
|---|---:|---:|---:|---|
| `checkpoints/safe/laptop_safe_11L.pt` | 800 | 60.6972 | 0 | Best perplexity among latest safe models |
| `checkpoints/safe/laptop_safe.pt` | 17092 | 66.3318 | 0 | Slightly worse perplexity than 11L |
| `checkpoints/ultimate/hatchling_latest.pt` | 100 | 247987.9562 | 0 | Not presentation-ready |
| `checkpoints/1hour_training/final_model.pt` | 0 | 108.8107 | 20 | Best logic score in this quick test |

## 7) Which model to present

Use-case recommendation:

- **If asked for latest checkpoint:** use `safe/laptop_safe_11L.pt`.
- **If asked for strongest reasoning demo:** use `1hour_training/final_model.pt`.

Practical default for professor meeting:

- Start with `-Profile latest`.
- If output quality looks unstable for your prompt, switch to `-Profile stable`.

## 8) Differentiating factors (science expo angle)

1. **Biologically-inspired memory mechanism**: explicit multi-timescale Hebbian traces instead of only attention cache.
2. **Architectural experimentation at scale**: 8L/11L/12L variants with different hidden sizes and tokenizer regimes.
3. **Real evaluation tooling**: live side-by-side generation + metrics + benchmark endpoint.
4. **Transparent model introspection**: UI now displays real model cards (layers, heads, logical layers, params, vocab details).

## 9) Questions professor is likely to ask

1. Why does BDH lose on perplexity vs DistilGPT2 in your current quick benchmark?
2. Which metric matters for your expo claim: perplexity, reasoning score, latency, or interpretability?
3. How robust is your logic score with only two questions?
4. Why does the latest 11L model have lower step count than the 8L model?
5. What are the trade-offs between BBPE checkpoints and Qwen-vocab checkpoints?
6. What is your controlled ablation proving multi-scale memory helps?
7. How do you avoid cherry-picking prompts in the live demo?
8. What is your reproducibility protocol (seed, sample_count, max_len, fixed prompts)?
9. Which checkpoint should be called "best" and by what objective function?
10. What is your next concrete step to close the gap vs DistilGPT2 perplexity?

## 10) Tight answers to keep ready

- "I am not claiming universal superiority; I am showing architecture-level trade-offs and where BDH is currently competitive."
- "Perplexity and logic score move differently across checkpoints, which is why I present both."
- "Latest checkpoint is `safe/laptop_safe_11L.pt`; best reasoning in my quick test is `1hour_training/final_model.pt`."
- "I use a fixed quick benchmark config (sample_count=20, max_len=192, seed=42) for reproducibility."

## 11) Direct answers for each likely professor question

1. Why does BDH lose on perplexity vs DistilGPT2 right now?
	- DistilGPT2 is a mature general LM baseline with optimized next-token likelihood, while BDH here is an architecture-exploration project balancing memory design, reasoning behavior, and efficiency trade-offs.

2. Which metric matters most for your expo claim?
	- Primary claim: architecture and memory mechanism novelty. So I report multiple metrics: perplexity for language modeling, logic score for targeted reasoning checks, and live latency/throughput for usability.

3. How robust is logic score with two questions?
	- It is only a quick sanity metric, not a final benchmark. I present it as preliminary and pair it with perplexity plus live prompt demos.

4. Why does latest 11L have lower step count than safe 8L?
	- They are from different training runs and schedules. "Latest" means newest file timestamp, not longest training duration.

5. BBPE vs Qwen-vocab trade-offs?
	- BBPE checkpoints are easier to run locally and show stronger quick logic in this repo. Qwen-vocab checkpoints align with teacher-token space and newer safe runs, but require careful tokenizer/vocab handling.

6. What ablation proves multi-scale memory helps?
	- Baseline to multi-scale comparisons are done via benchmark scripts and reports; the intended ablation is same data and training setup, only changing memory design and measuring retention, perplexity, and behavior.

7. How do you avoid cherry-picking prompts?
	- Use fixed benchmark settings (seed/sample_count/max_len), show exact checkpoint path, and run both pre-selected prompts and one prompt chosen by the professor live.

8. What is your reproducibility protocol?
	- Fixed config: sample_count=20, max_len=192, seed=42 for quick benchmark; explicit checkpoint path and explicit model architecture flags in run commands.

9. Which checkpoint is "best"?
	- Depends on objective: best latest checkpoint is `safe/laptop_safe_11L.pt`; best quick reasoning score in this repo snapshot is `checkpoints/1hour_training/final_model.pt`.

10. Next step to close perplexity gap?
	- Increase controlled training budget on the stronger reasoning checkpoint, expand evaluation set beyond two logic questions, and run systematic hyperparameter sweeps for temperature, sequence length, and distillation settings.

## 12) Exact command to run 1hour model for reasoning demo

From repo root:

```powershell
D:\Projects\BDH\.venv-webapp\Scripts\Activate.ps1
pwsh -File webapp/run_professor_demo.ps1 -Profile stable -EagerLoad
```

Hard-reset start (recommended if output still looks old):

```powershell
cd D:\Projects\BDH
& D:\Projects\BDH\.venv-webapp\Scripts\Activate.ps1
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
pwsh -File webapp/run_professor_demo.ps1 -Profile stable -EagerLoad -Port 8000
```

Direct server command (same model profile):

```powershell
python webapp/server.py --host 127.0.0.1 --port 8000 --checkpoint checkpoints/1hour_training/final_model.pt --data data/tinystories.txt --teacher-model Qwen3.5-0.8B --distil-model distilbert/distilgpt2 --bdh-n-embd 512 --bdh-n-layer 8 --bdh-n-head 8 --bdh-ffn-dim 2048 --bdh-max-seq-len 512 --bdh-hebbian-lr 0.0006 --default-top-k 40 --eager-load
```

Open in browser:

- `http://127.0.0.1:8000`

Reasoning demo settings in UI (recommended):

- Temperature: 0.6
- Top-K: 30
- Max New Tokens: 80

If you ever see outputs with `Ġ` markers, restart the server after pulling latest code so the tokenizer ByteLevel decoder patch is active.
