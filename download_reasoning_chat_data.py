"""
Download fresh chat+reasoning data for BDH (new-data run).

Default mix:
- Reasoning: open-r1/OpenR1-Math-220k
- Chat: HuggingFaceH4/ultrachat_200k

Output:
- JSONL with unified "messages" format
- metadata JSON with counts + sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step, show concise chain-of-thought "
    "internally, then provide clear final answer. Use structured reasoning when needed."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in {"assistant", "user", "system"}:
        return r
    if r in {"human"}:
        return "user"
    if r in {"gpt", "model", "bot"}:
        return "assistant"
    return "user"


def normalize_messages(raw_messages: Iterable[Dict]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        role = normalize_role(str(m.get("role", "user")))
        out.append({"role": role, "content": content})

    # Ensure starts with user/system and has assistant response.
    if not out:
        return out
    has_assistant = any(x["role"] == "assistant" for x in out)
    if not has_assistant:
        return []
    return out


def row_to_messages_reasoning(row: Dict) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        msgs = normalize_messages(row["messages"])
        if msgs:
            return msgs

    problem = str(row.get("problem", "")).strip() or str(row.get("question", "")).strip()
    solution = str(row.get("solution", "")).strip() or str(row.get("answer", "")).strip()
    if not problem or not solution:
        return []
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]


def row_to_messages_chat(row: Dict) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        msgs = normalize_messages(row["messages"])
        if msgs:
            return msgs

    prompt = str(row.get("prompt", "")).strip() or str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip() or str(row.get("output", "")).strip()
    if not prompt or not response:
        return []
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def sample_dataset_rows(ds, n: int, seed: int):
    if n <= 0:
        return []

    # Map-style dataset
    if hasattr(ds, "select") and hasattr(ds, "__len__"):
        n = min(n, len(ds))
        idx = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(idx)
        idx = idx[:n]
        return ds.select(idx)

    # Streaming iterable dataset
    rows = []
    rng = random.Random(seed)
    # Keep bounded lookahead so we don't iterate huge corpus.
    max_scan = max(n * 20, 5000)
    for i, row in enumerate(ds):
        # random thinning for diversity
        if rng.random() < 0.3:
            rows.append(row)
        if len(rows) >= n:
            break
        if i >= max_scan:
            break
    return rows[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download fresh reasoning+chat dataset for BDH")
    parser.add_argument("--reasoning-dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--reasoning-split", default="train")
    parser.add_argument("--reasoning-samples", type=int, default=30000)
    parser.add_argument("--chat-dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--chat-split", default="train_sft")
    parser.add_argument("--chat-samples", type=int, default=15000)
    parser.add_argument("--output", default="data/chat_reasoning_new_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_REASONING_SYSTEM_PROMPT)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package missing. Install: pip install datasets")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Guard: do not overwrite old TinyStories file.
    if out_path.name.lower() == "tinystories.txt":
        print("Refusing to write into old TinyStories path. Choose JSONL output path.")
        return 2

    print("=" * 70)
    print("DOWNLOADING FRESH DATA (NEW, NOT OLD TINYSTORIES)")
    print("=" * 70)
    print(f"Reasoning dataset: {args.reasoning_dataset} [{args.reasoning_split}]")
    print(f"Chat dataset:      {args.chat_dataset} [{args.chat_split}]")
    print(f"Output:            {out_path}")
    print()

    started = time.time()
    records: List[Dict] = []
    dedupe = set()

    # Reasoning
    rds = load_dataset(args.reasoning_dataset, split=args.reasoning_split, streaming=args.streaming)
    rds = sample_dataset_rows(rds, args.reasoning_samples, args.seed)
    r_ok = 0
    for row in rds:
        msgs = row_to_messages_reasoning(row)
        if not msgs:
            continue
        if args.inject_system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": args.system_prompt}] + msgs
        key = hashlib.sha1(json.dumps(msgs, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key in dedupe:
            continue
        dedupe.add(key)
        records.append({"messages": msgs, "source": args.reasoning_dataset, "kind": "reasoning"})
        r_ok += 1

    # Chat
    cds = load_dataset(args.chat_dataset, split=args.chat_split, streaming=args.streaming)
    cds = sample_dataset_rows(cds, args.chat_samples, args.seed + 1)
    c_ok = 0
    for row in cds:
        msgs = row_to_messages_chat(row)
        if not msgs:
            continue
        if args.inject_system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": args.system_prompt}] + msgs
        key = hashlib.sha1(json.dumps(msgs, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key in dedupe:
            continue
        dedupe.add(key)
        records.append({"messages": msgs, "source": args.chat_dataset, "kind": "chat"})
        c_ok += 1

    rng = random.Random(args.seed)
    rng.shuffle(records)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    digest = sha256_file(out_path)
    meta = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": str(out_path),
        "output_sha256": digest,
        "reasoning_dataset": args.reasoning_dataset,
        "chat_dataset": args.chat_dataset,
        "reasoning_rows_kept": r_ok,
        "chat_rows_kept": c_ok,
        "total_rows": len(records),
        "seconds": round(time.time() - started, 2),
        "system_prompt_injected": bool(args.inject_system_prompt),
        "system_prompt": args.system_prompt if args.inject_system_prompt else None,
        "streaming": bool(args.streaming),
        "note": "Fresh mixed chat+reasoning JSONL for new logits generation",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 70)
    print("DONE")
    print(f"Rows: {len(records)} (reasoning={r_ok}, chat={c_ok})")
    print(f"SHA256: {digest}")
    print(f"Metadata: {meta_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

