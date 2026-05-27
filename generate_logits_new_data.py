"""
Generate teacher top-k logits from NEW chat+reasoning JSONL data.

Key guarantees:
- Refuses old TinyStories text path.
- Refuses reusing same source data hash unless --allow-reuse set.
- Writes run metadata + source registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
# Work around cublasLt matmul heuristic failures on some RTX setups (8GB consumer GPUs).
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
import torch
from torch.utils.data import DataLoader, Dataset

# Avoid namespace collision with repo-local "tokenizers/" folder.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_CWD = str(Path.cwd())
for _p in ("", _SCRIPT_DIR, _CWD):
    while _p in sys.path:
        sys.path.remove(_p)


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_registry(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sources": []}


def save_registry(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def gpu_mem_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        return {
            "gpu_mem_free_mb": round(float(free) / (1024 * 1024), 2),
            "gpu_mem_total_mb": round(float(total) / (1024 * 1024), 2),
            "gpu_mem_allocated_mb": round(float(torch.cuda.memory_allocated(idx)) / (1024 * 1024), 2),
            "gpu_mem_reserved_mb": round(float(torch.cuda.memory_reserved(idx)) / (1024 * 1024), 2),
        }
    except Exception:
        return {}


def normalize_messages(raw) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(m.get("content", "")).strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def row_to_messages(row: Dict) -> List[Dict[str, str]]:
    msgs = normalize_messages(row.get("messages", []))
    if msgs:
        return msgs

    prompt = str(row.get("problem", "")).strip() or str(row.get("question", "")).strip()
    answer = str(row.get("solution", "")).strip() or str(row.get("answer", "")).strip()
    if prompt and answer:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return []


class EncodedDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, assistant_start: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.assistant_start = assistant_start

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, i: int) -> Dict[str, object]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "assistant_start": self.assistant_start[i],
        }


class ChunkWriter:
    def __init__(self, out_dir: Path, chunk_rows: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = max(1, chunk_rows)
        self.idx = 0
        self.ids_buf: List[np.ndarray] = []
        self.topi_buf: List[np.ndarray] = []
        self.topv_buf: List[np.ndarray] = []
        self.mask_buf: List[np.ndarray] = []
        self.rows = 0
        self.files: List[str] = []

    def add(
        self,
        input_ids: np.ndarray,
        top_idx: np.ndarray,
        top_vals: np.ndarray,
        assistant_mask: np.ndarray,
    ) -> None:
        self.ids_buf.append(input_ids.astype(np.int32, copy=False))
        self.topi_buf.append(top_idx.astype(np.int32, copy=False))
        self.topv_buf.append(top_vals.astype(np.float16, copy=False))
        self.mask_buf.append(assistant_mask.astype(np.uint8, copy=False))
        self.rows += int(input_ids.shape[0])
        if self.rows >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if self.rows == 0:
            return
        ids = np.concatenate(self.ids_buf, axis=0)
        topi = np.concatenate(self.topi_buf, axis=0)
        topv = np.concatenate(self.topv_buf, axis=0)
        amask = np.concatenate(self.mask_buf, axis=0)
        path = self.out_dir / f"logits_chunk_{self.idx:04d}.npz"
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f,
                input_ids=ids,
                topk_indices=topi,
                topk_logits=topv,
                assistant_mask=amask,
            )
        tmp.replace(path)
        self.files.append(path.name)
        self.idx += 1
        self.ids_buf.clear()
        self.topi_buf.clear()
        self.topv_buf.clear()
        self.mask_buf.clear()
        self.rows = 0

    def finalize(self) -> List[str]:
        self.flush()
        return self.files


def _dtype_from_name(name: str) -> torch.dtype:
    n = str(name).strip().lower()
    if n == "fp16":
        return torch.float16
    if n == "bf16":
        return torch.bfloat16
    if n == "fp32":
        return torch.float32
    # auto
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_teacher(
    model_id: str,
    use_4bit: bool = True,
    force_cpu: bool = False,
    gpu_dtype: str = "auto",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chosen_dtype = _dtype_from_name(gpu_dtype)
    if force_cpu and chosen_dtype != torch.float32:
        chosen_dtype = torch.float32

    model = None
    if use_4bit and not force_cpu:
        # Try 4-bit first for 8GB GPU safety.
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=chosen_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            ).eval()
        except Exception as e:
            print(f"[warn] 4bit load failed, fallback to fp16: {e}")

    if model is None:
        load_kwargs = dict(
            device_map="auto" if (torch.cuda.is_available() and not force_cpu) else None,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=chosen_dtype,
            **load_kwargs,
        ).eval()
        if force_cpu:
            model = model.to("cpu")
    return model, tokenizer


def model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def first_assistant_index(messages: List[Dict[str, str]]) -> int:
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return i
    return -1


def main() -> int:
    p = argparse.ArgumentParser(description="Generate teacher top-k logits from NEW chat/reasoning data")
    p.add_argument("--data", default="data/chat_reasoning_new_v1.jsonl")
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output-root", default="data/teacher_logits_new")
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--chunk-rows", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--run-hours", type=float, default=0.0, help="0 = no time limit")
    p.add_argument("--allow-reuse", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument(
        "--reasoning-system-prompt",
        default=(
            "You are a careful reasoning assistant. Think step by step, explore alternatives, "
            "then provide concise final answer."
        ),
    )
    p.add_argument("--disable-system-prompt", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--heartbeat-every-sec", type=float, default=30.0)
    p.add_argument("--tokenize-batch-size", type=int, default=256)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return 1
    if data_path.name.lower() == "tinystories.txt":
        print("Refusing old TinyStories file. Use new JSONL chat+reasoning data.")
        return 2
    if data_path.suffix.lower() != ".jsonl":
        print("Expected .jsonl chat data.")
        return 3

    source_hash = sha256_file(data_path)
    output_root = Path(args.output_root)
    registry_path = output_root / "source_registry.json"
    registry = load_registry(registry_path)
    existing = [x for x in registry.get("sources", []) if x.get("sha256") == source_hash]
    if existing and not args.allow_reuse:
        print("Source hash already used before. Pass --allow-reuse to override.")
        print(f"sha256={source_hash}")
        return 4

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{source_hash[:10]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEW-DATA LOGITS GENERATION")
    print("=" * 70)
    print(f"Data:        {data_path}")
    print(f"Data sha256: {source_hash}")
    print(f"Run dir:     {run_dir}")
    print(f"Teacher:     {args.teacher_model}")
    print()

    teacher, tokenizer = load_teacher(
        args.teacher_model,
        use_4bit=not args.no_4bit,
        force_cpu=bool(args.cpu),
        gpu_dtype=args.gpu_dtype,
    )
    input_dev = model_input_device(teacher)
    if input_dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    samples: List[Dict[str, object]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = row_to_messages(row)
            if not msgs:
                continue
            if not args.disable_system_prompt and (not msgs or msgs[0].get("role") != "system"):
                msgs = [{"role": "system", "content": args.reasoning_system_prompt}] + msgs

            assist_i = first_assistant_index(msgs)
            if assist_i < 0:
                continue
            prefix = msgs[:assist_i]
            if not prefix:
                prefix = [{"role": "system", "content": args.reasoning_system_prompt}]

            full_text = render_chat(tokenizer, msgs, add_generation_prompt=False)
            prompt_text = render_chat(tokenizer, prefix, add_generation_prompt=True)
            if not full_text.strip():
                continue

            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=args.seq_len,
                return_tensors="pt",
            )["input_ids"]
            assistant_start = int(prompt_ids.shape[-1])
            samples.append({"text": full_text, "assistant_start": assistant_start})
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break

    if not samples:
        print("No usable rows parsed from input JSONL.")
        return 5

    texts = [str(x["text"]) for x in samples]
    assistant_start_np = np.array([int(x["assistant_start"]) for x in samples], dtype=np.int32)

    # Pre-tokenize once to remove tokenizer bottleneck from hot path.
    tok_bs = max(1, int(args.tokenize_batch_size))
    ids_parts: List[np.ndarray] = []
    attn_parts: List[np.ndarray] = []
    for i in range(0, len(texts), tok_bs):
        enc = tokenizer(
            texts[i : i + tok_bs],
            padding="max_length",
            truncation=True,
            max_length=args.seq_len,
            return_tensors="np",
        )
        ids_parts.append(enc["input_ids"].astype(np.int32, copy=False))
        attn_parts.append(enc["attention_mask"].astype(np.uint8, copy=False))
    input_ids_np = np.concatenate(ids_parts, axis=0)
    attn_np = np.concatenate(attn_parts, axis=0)
    ds = EncodedDataset(input_ids_np, attn_np, assistant_start_np)
    pin_memory = bool(args.pin_memory or input_dev.type == "cuda")
    nw = max(0, int(args.num_workers))
    dl_kwargs: Dict[str, Any] = {
        "batch_size": max(1, args.batch_size),
        "shuffle": False,
        "num_workers": nw,
        "pin_memory": pin_memory,
    }
    if nw > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **dl_kwargs)
    writer = ChunkWriter(run_dir, chunk_rows=args.chunk_rows)

    start = time.time()
    rows_done = 0
    last_hb = 0.0
    total_assistant_tokens = 0
    total_valid_tokens = 0
    error_path = run_dir / "error.json"
    try:
        with torch.inference_mode():
            for b_idx, batch in enumerate(loader):
                if args.run_hours > 0 and (time.time() - start) / 3600.0 >= args.run_hours:
                    print(f"Timed stop at {args.run_hours}h.")
                    break
                if torch.is_tensor(batch["assistant_start"]):
                    assistant_start = [int(x) for x in batch["assistant_start"].tolist()]
                else:
                    assistant_start = [int(x) for x in batch["assistant_start"]]

                input_ids = batch["input_ids"].to(input_dev, dtype=torch.long, non_blocking=pin_memory)
                attn = batch["attention_mask"].to(input_ids.device, dtype=torch.long, non_blocking=pin_memory)
                logits = teacher(input_ids=input_ids, attention_mask=attn).logits
                k = min(int(args.top_k), int(logits.shape[-1]))
                top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
                if not torch.isfinite(top_vals).all():
                    raise RuntimeError(f"Non-finite logits detected at batch={b_idx}")

                attn_cpu = attn.detach().cpu().numpy()
                mask = np.zeros_like(attn_cpu, dtype=np.uint8)
                for i in range(mask.shape[0]):
                    valid_len = int(attn_cpu[i].sum())
                    start_i = min(max(assistant_start[i], 0), mask.shape[1])
                    end_i = min(valid_len, mask.shape[1])
                    if start_i < end_i:
                        mask[i, start_i:end_i] = 1

                writer.add(
                    input_ids.detach().cpu().numpy(),
                    top_idx.detach().cpu().numpy(),
                    top_vals.detach().to(torch.float16).cpu().numpy(),
                    mask,
                )
                rows_done += int(input_ids.shape[0])
                total_assistant_tokens += int(mask.sum())
                total_valid_tokens += int(attn_cpu.sum())

                now = time.time()
                if b_idx % max(1, int(args.progress_every)) == 0:
                    elapsed = (now - start) / 60.0
                    print(f"batch={b_idx:5d} rows={rows_done:7d}/{len(samples)} elapsed={elapsed:.1f}m")
                if (now - last_hb) >= float(args.heartbeat_every_sec):
                    elapsed_sec = max(1e-6, now - start)
                    rows_per_sec = float(rows_done) / elapsed_sec
                    remaining = max(0, len(samples) - rows_done)
                    hb = {
                        "timestamp_utc": utc_ts(),
                        "status": "running",
                        "rows_done": rows_done,
                        "rows_total": len(samples),
                        "batch_idx": int(b_idx),
                        "elapsed_sec": round(elapsed_sec, 3),
                        "rows_per_sec": round(rows_per_sec, 4),
                        "eta_sec": round(float(remaining) / rows_per_sec, 2) if rows_per_sec > 0 else 0.0,
                        "assistant_tokens": int(total_assistant_tokens),
                        "valid_tokens": int(total_valid_tokens),
                        "assistant_token_ratio": round(
                            float(total_assistant_tokens) / float(max(total_valid_tokens, 1)),
                            6,
                        ),
                    }
                    hb.update(gpu_mem_stats(input_dev))
                    write_json_atomic(run_dir / "heartbeat.json", hb)
                    last_hb = now
    except Exception as e:
        error_payload = {
            "timestamp_utc": utc_ts(),
            "status": "failed",
            "rows_done": rows_done,
            "rows_total": len(samples),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        error_payload.update(gpu_mem_stats(input_dev))
        write_json_atomic(error_path, error_payload)
        raise

    files = writer.finalize()

    meta = {
        "run_id": run_id,
        "created_at_utc": utc_ts(),
        "data_path": str(data_path),
        "data_sha256": source_hash,
        "teacher_model": args.teacher_model,
        "seq_len": int(args.seq_len),
        "top_k": int(args.top_k),
        "batch_size": int(args.batch_size),
        "rows_generated": rows_done,
        "rows_total_loaded": len(samples),
        "chunk_files": files,
        "assistant_mask_saved": True,
        "system_prompt_enabled": not args.disable_system_prompt,
        "system_prompt_text": None if args.disable_system_prompt else args.reasoning_system_prompt,
        "seconds": round(time.time() - start, 2),
        "rows_per_sec": round(float(rows_done) / float(max(time.time() - start, 1e-6)), 4),
        "assistant_tokens": int(total_assistant_tokens),
        "valid_tokens": int(total_valid_tokens),
        "assistant_token_ratio": round(
            float(total_assistant_tokens) / float(max(total_valid_tokens, 1)),
            6,
        ),
        "num_workers": int(nw),
        "pin_memory": bool(pin_memory),
        "progress_every": int(max(1, args.progress_every)),
        "heartbeat_every_sec": float(args.heartbeat_every_sec),
        "tokenize_batch_size": int(tok_bs),
        "heartbeat_path": str(run_dir / "heartbeat.json"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_json_atomic(
        run_dir / "heartbeat.json",
        {
            "timestamp_utc": utc_ts(),
            "status": "completed",
            "rows_done": rows_done,
            "rows_total": len(samples),
            "elapsed_sec": round(time.time() - start, 3),
            "rows_per_sec": round(float(rows_done) / float(max(time.time() - start, 1e-6)), 4),
            "assistant_tokens": int(total_assistant_tokens),
            "valid_tokens": int(total_valid_tokens),
            "assistant_token_ratio": round(
                float(total_assistant_tokens) / float(max(total_valid_tokens, 1)),
                6,
            ),
            **gpu_mem_stats(input_dev),
        },
    )

    registry.setdefault("sources", [])
    if not existing:
        registry["sources"].append(
            {
                "sha256": source_hash,
                "first_path": str(data_path),
                "first_seen_utc": utc_ts(),
                "runs": [run_id],
            }
        )
    else:
        for item in registry["sources"]:
            if item.get("sha256") == source_hash:
                item.setdefault("runs", [])
                item["runs"].append(run_id)
                item["last_seen_utc"] = utc_ts()
                break
    save_registry(registry_path, registry)

    print("=" * 70)
    print("DONE")
    print(f"Rows generated: {rows_done}")
    print(f"Chunks:         {len(files)}")
    print(f"Metadata:       {run_dir / 'metadata.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

