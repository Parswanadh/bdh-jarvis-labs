"""
Minimal BDH sanity training for RTX 4070 8GB.

Purpose:
- Verify end-to-end training works on limited VRAM
- Confirm train/val loss trends down in a short run
- Avoid teacher-model complexity while debugging quality issues
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


class ByteSequenceDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[: self.seq_len]
        y = chunk[1 : self.seq_len + 1]
        return x, y


def load_lines(path: Path, max_lines: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            t = line.strip()
            if t:
                lines.append(t)

    if len(lines) < 200:
        raise ValueError(f"Need at least 200 non-empty lines, found {len(lines)} in {path}")

    return lines


@torch.no_grad()
def eval_loss(
    model: MultiScaleBDH,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 20,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        total += loss.item()
        n += 1

    model.train()
    return total / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal BDH sanity test for RTX 4070 8GB")
    p.add_argument("--data", default="data/tinystories.txt")
    p.add_argument("--max-lines", type=int, default=30000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--max-minutes", type=float, default=15.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--checkpoint", default="checkpoints/rtx4070_minimal/latest.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 72)
    print("RTX4070 MINIMAL BDH SANITY TEST")
    print("=" * 72)
    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size} steps={args.steps}")
    print(f"expected_random_ce≈ln(256)={math.log(256):.4f}")
    print()

    lines = load_lines(Path(args.data), args.max_lines)
    split = int(0.95 * len(lines))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_data = torch.tensor(list(train_text.encode("utf-8", errors="ignore")), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode("utf-8", errors="ignore")), dtype=torch.long)

    train_ds = ByteSequenceDataset(train_data, args.seq_len)
    val_ds = ByteSequenceDataset(val_data, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cfg = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=192,
        n_layer=4,
        n_head=4,
        ffn_dim=768,
        dropout=0.1,
        max_seq_len=args.seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001,
    )
    model = MultiScaleBDH(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = 0
    best_val = float("inf")
    start = time.time()
    train_iter = iter(train_loader)
    first_loss = None
    last_loss = None

    while step < args.steps:
        elapsed_min = (time.time() - start) / 60.0
        if elapsed_min >= args.max_minutes:
            break

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

        if step % args.log_every == 0:
            msg = f"step={step} train_loss={last_loss:.4f} elapsed_min={elapsed_min:.1f}"
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                msg += f" vram_alloc={alloc:.2f}GB vram_reserved={reserved:.2f}GB"
            print(msg)

        if step % args.eval_every == 0:
            v = eval_loss(model, val_loader, device, use_amp, amp_dtype, max_batches=20)
            best_val = min(best_val, v)
            print(f"[eval] step={step} val_loss={v:.4f} best_val={best_val:.4f}")

    final_val = eval_loss(model, val_loader, device, use_amp, amp_dtype, max_batches=30)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "first_train_loss": first_loss,
            "last_train_loss": last_loss,
            "final_val_loss": final_val,
            "config": vars(args),
            "model_config": cfg,
        },
        ckpt_path,
    )

    print()
    print("=" * 72)
    print(f"steps_completed={step}")
    print(f"first_train_loss={first_loss:.4f}" if first_loss is not None else "first_train_loss=n/a")
    print(f"last_train_loss={last_loss:.4f}" if last_loss is not None else "last_train_loss=n/a")
    print(f"final_val_loss={final_val:.4f}")
    if first_loss is not None and last_loss is not None:
        print(f"train_loss_delta={first_loss - last_loss:.4f}")
    print(f"checkpoint={ckpt_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
