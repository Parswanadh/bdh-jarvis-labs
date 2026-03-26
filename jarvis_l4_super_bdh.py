"""
Jarvis L4 optimized BDH distillation pipeline.

Goals:
- Keep sequence length fixed and fast (default 192).
- Use as much 24GB VRAM as possible.
- Parallelize teacher-logit production and student training.
- Resume safely from checkpoint.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


class StoryDataset(Dataset):
    def __init__(self, path: Path, max_items: int = 400000):
        self.rows: List[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_items:
                    break
                t = line.strip()
                if t:
                    self.rows.append(t)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> str:
        return self.rows[i]


def collate_text(batch: List[str]) -> List[str]:
    return batch


def distill_loss_topk(student_logits: torch.Tensor, top_idx: torch.Tensor, top_vals: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # next-token shift
    shift_s = student_logits[:, :-1, :].contiguous()
    shift_l = input_ids[:, 1:].contiguous()
    shift_idx = top_idx[:, :-1, :].contiguous()
    shift_vals = top_vals[:, :-1, :].contiguous()

    vocab = shift_s.size(-1)
    if shift_idx.numel() > 0:
        shift_idx = shift_idx.clamp(min=0, max=vocab - 1)
    shift_l = shift_l.clamp(min=0, max=vocab - 1)

    s_top = torch.gather(shift_s, dim=-1, index=shift_idx)
    t_probs = F.softmax(shift_vals.float() / 2.0, dim=-1)
    s_log_probs = F.log_softmax(s_top.float() / 2.0, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (2.0**2)

    ce = F.cross_entropy(
        shift_s.reshape(-1, shift_s.size(-1)),
        shift_l.reshape(-1),
        ignore_index=0,
    )
    return 0.8 * kl + 0.2 * ce


def make_student(vocab_size: int, seq_len: int, device: torch.device) -> MultiScaleBDH:
    cfg = MultiScaleBDHConfig(
        vocab_size=vocab_size,
        n_embd=512,
        n_layer=10,
        n_head=8,
        ffn_dim=2048,
        max_seq_len=seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.0006,
    )
    model = MultiScaleBDH(cfg).to(device)
    return model


def load_4bit_teacher(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return teacher, tok


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description="Jarvis L4 super BDH")
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data/tinystories.txt")
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--teacher-replicas", type=int, default=6)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--max-items", type=int, default=500000)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument("--output", default="checkpoints/jarvis_l4_super/latest.pt")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for jarvis_l4_super_bdh.py")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda")

    dataset = StoryDataset(Path(args.data), max_items=args.max_items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_text,
        pin_memory=True,
        drop_last=True,
    )

    # tokenization/teacher
    teachers = []
    tok_ref: Optional[AutoTokenizer] = None
    teacher_vocab_size: Optional[int] = None
    for i in range(args.teacher_replicas):
        t_model, t_tok = load_4bit_teacher(args.teacher_model)
        teachers.append(t_model)
        tok_ref = t_tok
        if teacher_vocab_size is None:
            teacher_vocab_size = int(t_model.config.vocab_size)
        print(f"[teacher] loaded replica {i+1}/{args.teacher_replicas}")

    assert tok_ref is not None
    tokenizer_vocab = int(getattr(tok_ref, "vocab_size", 0))
    tokenizer_len = int(len(tok_ref))
    teacher_vocab = int(teacher_vocab_size or tokenizer_vocab or tokenizer_len)
    student_vocab = max(teacher_vocab, tokenizer_vocab, tokenizer_len)
    print(
        f"[vocab] teacher={teacher_vocab} tokenizer_vocab={tokenizer_vocab} "
        f"tokenizer_len={tokenizer_len} student={student_vocab}"
    )
    student = make_student(vocab_size=student_vocab, seq_len=args.seq_len, device=device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_path = Path(args.output)
    step = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        print(f"[resume] step={step}")

    q: "queue.Queue[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    def producer(replica_idx: int):
        teacher = teachers[replica_idx]
        local_iter = iter(loader)
        while not stop_event.is_set():
            try:
                texts = next(local_iter)
            except StopIteration:
                local_iter = iter(loader)
                texts = next(local_iter)

            enc = tok_ref(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.seq_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attn = enc["attention_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                logits = teacher(input_ids=input_ids, attention_mask=attn).logits
                top_vals, top_idx = torch.topk(logits, k=args.top_k, dim=-1)
                top_idx = top_idx.clamp(min=0, max=student_vocab - 1)

            # Offload to host queue
            q.put(
                (
                    input_ids.detach().cpu(),
                    top_idx.detach().cpu(),
                    top_vals.detach().cpu(),
                )
            )

    workers = [threading.Thread(target=producer, args=(i,), daemon=True) for i in range(len(teachers))]
    for w in workers:
        w.start()

    start_time = time.time()
    try:
        while (time.time() - start_time) < args.hours * 3600:
            ids_cpu, idx_cpu, vals_cpu = q.get(timeout=120)
            input_ids = ids_cpu.to(device, non_blocking=True)
            top_idx = idx_cpu.to(device, non_blocking=True)
            top_vals = vals_cpu.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                s_logits, _ = student(input_ids)
                loss = distill_loss_topk(s_logits, top_idx, top_vals, input_ids)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            step += 1

            if step % 10 == 0:
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB queue={q.qsize()}"
                )

            if step % args.save_every == 0:
                payload = {
                    "step": step,
                    "model": student.state_dict(),
                    "optimizer": optim.state_dict(),
                    "config": {
                        "teacher_model": args.teacher_model,
                        "seq_len": args.seq_len,
                        "top_k": args.top_k,
                        "teacher_replicas": args.teacher_replicas,
                        "batch_size": args.batch_size,
                    },
                }
                atomic_save(ckpt_path, payload)
                print(f"[save] {ckpt_path} step={step}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[oom] caught OOM, saving checkpoint and exiting")
        else:
            raise
    finally:
        stop_event.set()
        payload = {
            "step": step,
            "model": student.state_dict(),
            "optimizer": optim.state_dict(),
            "config": {
                "teacher_model": args.teacher_model,
                "seq_len": args.seq_len,
                "top_k": args.top_k,
                "teacher_replicas": args.teacher_replicas,
                "batch_size": args.batch_size,
            },
        }
        atomic_save(ckpt_path, payload)
        print(f"[final-save] {ckpt_path} step={step}")


if __name__ == "__main__":
    main()