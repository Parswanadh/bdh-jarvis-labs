"""
Analyze generated logits chunks for integrity and training readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def analyze_chunk(path: Path) -> Dict[str, float]:
    with np.load(path) as z:
        input_ids = z["input_ids"]
        topi = z["topk_indices"]
        topv = z["topk_logits"]
        amask = z["assistant_mask"]

        if input_ids.ndim != 2 or topi.ndim != 3 or topv.ndim != 3 or amask.ndim != 2:
            raise RuntimeError(f"Bad rank in {path.name}")
        if not (
            input_ids.shape[0] == topi.shape[0] == topv.shape[0] == amask.shape[0]
            and input_ids.shape[1] == topi.shape[1] == topv.shape[1] == amask.shape[1]
            and topi.shape[2] == topv.shape[2]
        ):
            raise RuntimeError(f"Shape mismatch in {path.name}")

        finite = np.isfinite(topv)
        nonfinite = int((~finite).sum())
        valid_tokens = int((input_ids != 0).sum())
        assistant_tokens = int(amask.sum())
        rows = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        top_k = int(topi.shape[2])

        return {
            "rows": rows,
            "seq_len": seq_len,
            "top_k": top_k,
            "nonfinite_logits": nonfinite,
            "assistant_tokens": assistant_tokens,
            "valid_tokens": valid_tokens,
            "assistant_token_ratio": float(assistant_tokens) / float(max(valid_tokens, 1)),
        }


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze logits run quality and integrity")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-nonfinite", type=int, default=0)
    p.add_argument("--min-assistant-ratio", type=float, default=0.01)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        return 1

    chunks = sorted(run_dir.glob("logits_chunk_*.npz"))
    if not chunks:
        print(f"No logits chunks found in: {run_dir}")
        return 2

    per_chunk: List[Dict[str, float]] = []
    total_rows = 0
    total_nonfinite = 0
    total_assistant = 0
    total_valid = 0
    seq_lens = set()
    top_ks = set()

    for chunk in chunks:
        stat = analyze_chunk(chunk)
        stat["file"] = chunk.name
        per_chunk.append(stat)
        total_rows += int(stat["rows"])
        total_nonfinite += int(stat["nonfinite_logits"])
        total_assistant += int(stat["assistant_tokens"])
        total_valid += int(stat["valid_tokens"])
        seq_lens.add(int(stat["seq_len"]))
        top_ks.add(int(stat["top_k"]))

    assistant_ratio = float(total_assistant) / float(max(total_valid, 1))

    out = {
        "run_dir": str(run_dir),
        "chunk_count": len(chunks),
        "rows": total_rows,
        "nonfinite_logits": total_nonfinite,
        "assistant_tokens": total_assistant,
        "valid_tokens": total_valid,
        "assistant_token_ratio": assistant_ratio,
        "seq_lens": sorted(seq_lens),
        "top_ks": sorted(top_ks),
        "per_chunk": per_chunk,
    }
    out_path = run_dir / "analysis_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 70)
    print("LOGITS ANALYSIS")
    print("=" * 70)
    print(f"Run:                 {run_dir}")
    print(f"Chunks:              {len(chunks)}")
    print(f"Rows:                {total_rows}")
    print(f"Non-finite logits:   {total_nonfinite}")
    print(f"Assistant ratio:     {assistant_ratio:.6f}")
    print(f"Seq lens:            {sorted(seq_lens)}")
    print(f"Top-k values:        {sorted(top_ks)}")
    print(f"Report:              {out_path}")
    print("=" * 70)

    if total_nonfinite > int(args.max_nonfinite):
        print(f"[fail] non-finite logits {total_nonfinite} > max {args.max_nonfinite}")
        return 3
    if assistant_ratio < float(args.min_assistant_ratio):
        print(f"[fail] assistant ratio {assistant_ratio:.6f} < min {args.min_assistant_ratio}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
