"""
Run one autonomous self-improvement cycle for BDH.

Example:
python run_autonomous_cycle.py ^
  --train-jsonl data/distill/world_train.jsonl ^
  --val-jsonl data/distill/world_val.jsonl ^
  --checkpoint checkpoints/safe/laptop_safe_11L.pt ^
  --out-checkpoint checkpoints/autonomous/autonomous_cycle_1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implementation.multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from implementation.autonomous_self_improvement import (
    AutonomousImprover,
    build_loaders,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous BDH self-improvement cycle")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=11)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.0005,
    )
    model = MultiScaleBDH(config)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"Checkpoint not found, starting fresh: {ckpt_path}")

    train_loader, val_loader = build_loaders(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    improver = AutonomousImprover(model, device=device)

    result = improver.improve_one_cycle(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
    )

    print("Cycle result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "cycle_metrics": result,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
