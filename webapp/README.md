# BDH Web Interface (BDH vs DistilGPT2)

This web app provides a real backend+frontend interface for:

- Prompt input and side-by-side text generation.
- BDH checkpoint output vs DistilGPT2 output.
- Live comparison metrics (latency, tokens/sec, prompt perplexity).
- Quick benchmark endpoint on TinyStories samples.
- Existing benchmark JSON + visualization gallery.

## Start

From repo root:

```bash
python webapp/server.py --host 0.0.0.0 --port 8000 --checkpoint checkpoints/safe/laptop_safe_11L.pt --data data/tinystories.txt --teacher-model Qwen3.5-0.8B --bdh-n-embd 256 --bdh-n-layer 11 --bdh-n-head 8 --bdh-ffn-dim 1024 --bdh-max-seq-len 192
```

Open:

`http://localhost:8000`

## Professor Demo Shortcut

From repo root:

```powershell
pwsh -File webapp/run_professor_demo.ps1 -Profile latest -EagerLoad
```

Profiles:

- `latest`: newest safe 11-layer checkpoint (`checkpoints/safe/laptop_safe_11L.pt`)
- `stable`: BBPE checkpoint with stronger quick-logic score (`checkpoints/1hour_training/final_model.pt`)

## Notes

- This uses real model inference (no fake results).
- First run may take time while models download/load.
- Default baseline model: `distilbert/distilgpt2`.
- BDH architecture defaults are set for the latest safe 11-layer checkpoint:
  - n_embd=256, n_layer=11, n_head=8, ffn_dim=1024, max_seq_len=192.

## API endpoints

- `GET /api/health`
- `GET /api/config`
- `GET /api/artifacts`
- `GET /api/benchmark/latest`
- `POST /api/generate`
- `POST /api/benchmark/quick`

