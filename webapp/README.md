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
python webapp/server.py --host 0.0.0.0 --port 8000 --checkpoint checkpoints/jarvis_l4_super/latest.pt --data data/tinystories.txt
```

Open:

`http://localhost:8000`

## Notes

- This uses real model inference (no fake results).
- First run may take time while models download/load.
- Default baseline model: `distilbert/distilgpt2`.
- BDH architecture defaults are set for Jarvis L4 checkpoint:
  - n_embd=512, n_layer=10, n_head=8, ffn_dim=2048, max_seq_len=192.

## API endpoints

- `GET /api/health`
- `GET /api/config`
- `GET /api/artifacts`
- `GET /api/benchmark/latest`
- `POST /api/generate`
- `POST /api/benchmark/quick`

