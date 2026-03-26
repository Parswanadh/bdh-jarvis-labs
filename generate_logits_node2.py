"""
Node 2 deterministic logits generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distribute_logits_core import (
    ChunkWriter,
    DistLogitsConfig,
    GitOperations,
    ProgressTracker,
    ShardRegistry,
    get_node_shard_range,
)


NODE_ID = 2


def load_story_slice(path: Path, start: int, end: int) -> List[str]:
    stories: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            t = line.strip()
            if t:
                stories.append(t)
    return stories


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate logits shard for node2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--git-push", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    cfg = DistLogitsConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        top_k=args.top_k,
        teacher_model=args.teacher_model,
    )

    root = Path(__file__).parent
    data_file = root / cfg.data_file
    out_root = root / cfg.output_root
    node_dir = out_root / f"node{NODE_ID}"
    progress_file = node_dir / "progress.json"
    metadata_file = out_root / "metadata.json"

    registry = ShardRegistry(metadata_file)
    registry.initialize(cfg)
    tracker = ProgressTracker(progress_file, NODE_ID)

    if tracker.state.get("status") == "completed":
        print("Node2 already completed.")
        return 0
    if tracker.state.get("status") == "error" and not args.resume:
        print("Previous run errored. Re-run with --resume.")
        return 2

    try:
        start, end = get_node_shard_range(NODE_ID, cfg.total_stories, cfg.num_nodes)
        stories = load_story_slice(data_file, start, end)
        offset = int(tracker.state.get("last_story_offset", 0)) if args.resume else 0
        if offset > 0:
            stories = stories[offset:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.teacher_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        ).eval()

        writer = ChunkWriter(node_dir, NODE_ID, cfg.chunk_max_bytes, registry)

        for i in range(0, len(stories), cfg.batch_size):
            batch_text = stories[i : i + cfg.batch_size]
            enc = tokenizer(
                batch_text,
                padding="max_length",
                truncation=True,
                max_length=cfg.seq_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with torch.no_grad():
                logits = teacher(input_ids=input_ids, attention_mask=attn).logits
            top_vals, top_idx = torch.topk(logits, k=cfg.top_k, dim=-1)
            flushed = writer.add_batch(
                input_ids.detach().cpu().numpy().astype(np.int32, copy=False),
                top_idx.detach().cpu().numpy().astype(np.int32, copy=False),
                top_vals.detach().cpu().numpy().astype(np.float16, copy=False),
            )
            if flushed is not None:
                tracker.mark_chunk(flushed[0])
            tracker.mark_batch(stories_inc=len(batch_text), batch_inc=1)

        last = writer.finalize()
        if last is not None:
            tracker.mark_chunk(last[0])
        tracker.mark_completed()
        registry.mark_node_complete(NODE_ID)

        if args.git_push:
            git_ops = GitOperations(root, NODE_ID)
            git_ops.stage_commit_push(str(cfg.output_root).replace("\\", "/"))

        summary = {
            "node": NODE_ID,
            "range": [start, end],
            "stories_processed": tracker.state.get("stories_processed", 0),
            "chunks_written": tracker.state.get("chunks_written", 0),
            "status": "completed",
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:
        tracker.mark_error(str(e))
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

