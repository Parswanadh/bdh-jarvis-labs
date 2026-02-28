"""
Train Multi-Scale BDH on Jarvis Labs A100 80GB
====================================================

Optimized for:
- A100 80GB VRAM
- 120GB RAM
- 1 hour time limit (cost efficient)

Performance optimizations:
- Large batch sizes (512-1024)
- Mixed precision (FP16)
- torch.compile
- Multiple data workers
- Memory pinning
- Gradient checkpointing disabled for speed
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import sys
from datetime import datetime
import time

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from stable_config import get_small_model_config


class TeacherDataset(Dataset):
    """Dataset from teacher-generated data."""

    def __init__(self, data_file: str, max_seq_len: int = 512):
        print(f"[LOAD] Loading teacher data from {data_file}")

        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item['generated']
                self.data.append(text)

        self.full_text = "\n".join(self.data)
        self.data_tensor = torch.tensor(
            list(self.full_text.encode('utf-8')),
            dtype=torch.long
        )

        self.max_seq_len = max_seq_len

        print(f"[OK] Loaded {len(self.data)} samples")
        print(f"   Total characters: {len(self.full_text):,}")
        print(f"   Total bytes: {len(self.data_tensor):,}")

    def __len__(self):
        return max(0, len(self.data_tensor) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data_tensor[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]
        y = chunk[1:self.max_seq_len + 1]
        return x, y


def train_epoch(model, dataloader, optimizer, config, device, scaler=None):
    """Train for one epoch with mixed precision."""

    model.train()
    total_loss = 0
    start_time = time.time()

    use_amp = device.type == 'cuda' and scaler is not None

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device, y.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if hasattr(config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            if hasattr(config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_idx + 1) * config.batch_size * config.max_seq_len / elapsed
            avg_loss = total_loss / (batch_idx + 1)

            print(f"[{batch_idx+1}/{len(dataloader)}] loss: {avg_loss:.4f} | tokens/sec: {int(tokens_per_sec)}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, config, iteration, loss, checkpoint_dir, is_best=False):
    """Save training checkpoint."""

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best,
        'task': 'gemma3_distillation'
    }

    torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_iter_{iteration}.pt")
    torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_latest.pt")

    if is_best:
        torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_best.pt")
        print(f"[BEST] New best model! Val loss: {loss:.4f}")

    print(f"[OK] Checkpoint saved: iteration {iteration}, loss: {loss:.4f}")


def train(config, train_config):
    """Main training function optimized for A100 80GB."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Training on device: {device}")

    # Show GPU info
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[GPU] Allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # Create model
    model = MultiScaleBDH(config)
    model = model.to(device)

    # Compile model for speed
    try:
        model = torch.compile(model)
        print("[OK] Model compiled with torch.compile")
    except:
        print("[INFO] torch.compile not available")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created: {num_params:,} parameters")
    print()

    # Setup data with optimizations for A100
    print("[STATS] Loading dataset...")
    train_dataset = TeacherDataset(
        "teacher_data/gemma_training_data.jsonl",
        max_seq_len=config.max_seq_len
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # OPTIMIZED FOR A100: Large batch size, multiple workers, pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=8,  # Multiple workers for A100
        pin_memory=True,  # Pin memory for faster transfer
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[OK] Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Mixed precision for A100
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print("[START] Starting training...")
    print(f"   Max iterations (epochs): {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Mixed precision: {scaler is not None}")
    print(f"   Data workers: 8")
    print("="*70)
    print()

    best_val_loss = float('inf')
    t0 = time.time()

    for iteration in range(train_config.max_iters):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, train_config, device, scaler)

        epoch_time = time.time() - epoch_start

        print()
        print(f"[EPOCH {iteration+1}/{train_config.max_iters}]")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Epoch Time: {epoch_time/60:.1f} minutes")
        print()

        # Save checkpoint after each epoch
        save_checkpoint(
            model, optimizer, train_config, iteration + 1, train_loss,
            train_config.checkpoint_dir, is_best=False
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        perplexity = torch.exp(torch.tensor(val_loss))

        print()
        print("="*70)
        print(f"[STATS] VALIDATION [epoch {iteration + 1}]")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Perplexity: {perplexity:.2f}")
        print("="*70)
        print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, train_config, iteration + 1, val_loss,
                train_config.checkpoint_dir, is_best=True
            )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, train_config, train_config.max_iters, train_loss,
        train_config.checkpoint_dir, is_best=False
    )

    elapsed = time.time() - t0
    print()
    print("="*70)
    print("[DONE] TRAINING COMPLETE!")
    print("="*70)
    print(f"[OK] Training time: {elapsed/60:.1f} minutes")
    print(f"[OK] Final train loss: {train_loss:.4f}")
    print(f"[OK] Best validation loss: {best_val_loss:.4f}")
    print(f"[OK] Checkpoints saved to: {train_config.checkpoint_dir}")
    print("="*70)


def main():
    """Main function."""

    print("="*70)
    print("BDH DISTILLATION - JARVIS LABS A100 80GB")
    print("Teacher: Gemma 3 270M (via Ollama)")
    print("="*70)
    print()

    # Model configuration
    config = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        max_seq_len=512,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    # Training configuration - OPTIMIZED FOR A100 80GB
    train_config = get_small_model_config()
    train_config.max_iters = 5  # 5 epochs
    train_config.batch_size = 512  # LARGE batch for A100!
    train_config.eval_interval = 1
    train_config.save_interval = 1
    train_config.checkpoint_dir = "checkpoints/bdh_a100"
    train_config.resume_from = None

    print("MODEL CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: ~5M")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print()

    print("TRAINING CONFIGURATION:")
    print(f"   Epochs: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size} (A100 optimized!)")
    print(f"   Learning rate: {train_config.learning_rate}")
    print()

    print("TASK:")
    print(f"   Task: Knowledge Distillation")
    print(f"   Teacher: Gemma 3 270M (Ollama)")
    print(f"   Student: Multi-Scale BDH")
    print(f"   Estimated time: 30-40 minutes")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
