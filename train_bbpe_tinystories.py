"""
Train BDH with BBPE Tokenizer on TinyStories Dataset
====================================================
Better tokenizer + Real data = Better model!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time
from datetime import datetime
from functools import partial

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer


class TinyStoriesDataset(Dataset):
    """TinyStories dataset with BBPE tokenization."""

    def __init__(self, data_file: str, tokenizer: BBPETokenizer, max_seq_len: int = 512):
        print(f"[LOAD] Loading TinyStories from {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            self.stories = [line.strip() for line in f if line.strip()]

        print(f"[OK] Loaded {len(self.stories)} stories")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        """Encode story with BBPE tokenizer."""
        story = self.stories[idx]

        # Encode with BBPE
        token_ids = self.tokenizer.encode(story)

        # Truncate if needed
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(token_ids), dtype=torch.long)
        }


def collate_fn(batch, pad_token_id=0):
    """Collate function to pad sequences."""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    # Pad sequences
    max_len = max(ids.size(0) for ids in input_ids)

    padded_ids = []
    padded_masks = []

    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - ids.size(0)
        padded_ids.append(F.pad(ids, (0, pad_len), value=pad_token_id))
        padded_masks.append(F.pad(mask, (0, pad_len), value=0))

    return {
        'input_ids': torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_masks)
    }


def train_epoch(model, dataloader, optimizer, config, device, scaler=None, checkpoint_dir=None, global_batch=0):
    """Train one epoch."""

    model.train()
    total_loss = 0
    start_time = time.time()

    use_amp = device.type == 'cuda' and scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Create targets (shift by 1 for next-token prediction)
        targets = input_ids.roll(shifts=-1, dims=1)
        targets[:, -1] = 0  # Mask last token

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(input_ids)

                # Compute loss (only on non-padded tokens)
                # logits: [batch, seq_len, vocab_size]
                # input_ids: [batch, seq_len]
                shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
                shift_labels = input_ids[..., 1:].contiguous()  # [batch, seq_len-1]

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0  # Padding token
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
            shift_labels = input_ids[..., 1:].contiguous()  # [batch, seq_len-1]

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )

            optimizer.zero_grad()
            loss.backward()

            if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_idx + 1) * config.batch_size * config.max_seq_len / elapsed
            avg_loss = total_loss / (batch_idx + 1)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9

            print(f"[{batch_idx+1}/{len(dataloader)}] "
                  f"loss: {avg_loss:.4f} | "
                  f"tokens/sec: {int(tokens_per_sec)} | "
                  f"GPU: {allocated:.1f}GB" if torch.cuda.is_available() else f"tokens/sec: {int(tokens_per_sec)}")

        # Save checkpoint every 10000 batches
        if (batch_idx + 1) % 10000 == 0 and checkpoint_dir:
            avg_loss = total_loss / (batch_idx + 1)
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': -1,  # Mid-epoch
                'batch_idx': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': avg_loss,
                'timestamp': datetime.now().isoformat(),
                'task': 'bbpe_tinystories_training'
            }
            torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_batch_{batch_idx+1}.pt")
            print(f"[CHECKPOINT] Saved at batch {batch_idx+1}, loss: {avg_loss:.4f}")

    return total_loss / len(dataloader), batch_idx + 1  # Return batch count


def save_checkpoint(model, optimizer, config, epoch, loss, checkpoint_dir, is_best=False):
    """Save checkpoint."""

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best,
        'task': 'bbpe_tinystories_training'
    }

    torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_latest.pt")

    if is_best:
        torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_best.pt")
        print(f"[BEST] New best model! Loss: {loss:.4f}")

    print(f"[OK] Checkpoint saved: epoch {epoch}, loss: {loss:.4f}")


def train(config, train_config):
    """Main training function."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Training on device: {device}")

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()
    print("="*70)
    print("TRAINING BDH WITH BBPE TOKENIZER ON TINYSTORIES")
    print("="*70)
    print()

    # Load or train tokenizer
    tokenizer_path = "tokenizers/bbpe_tokenizer.json"

    if Path(tokenizer_path).exists():
        print(f"[LOAD] Loading BBPE tokenizer from: {tokenizer_path}")
        tokenizer = BBPETokenizer(tokenizer_path=tokenizer_path)
    else:
        print(f"[TRAIN] Training new BBPE tokenizer...")
        tokenizer = BBPETokenizer(vocab_size=train_config.vocab_size)

        # Check for TinyStories data
        data_file = "data/tinystories.txt"
        if not Path(data_file).exists():
            print("[ERROR] TinyStories data not found!")
            print("[INFO] Please download tinystories.txt first")
            return

        tokenizer.train([data_file])

        # Create directory before saving
        Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)

    print(f"[OK] Tokenizer vocab size: {tokenizer.vocab_size_actual}")
    print()

    # Create model with BBPE vocab size
    print("[CREATE] Creating BDH model...")
    config.vocab_size = tokenizer.vocab_size_actual
    config.max_seq_len = train_config.max_seq_len

    model = MultiScaleBDH(config)
    model = model.to(device)

    # Disable torch.compile due to Triton dependency
    # try:
    #     model = torch.compile(model)
    #     print("[OK] Model compiled with torch.compile")
    # except:
    #     print("[INFO] torch.compile not available")
    print("[INFO] torch.compile disabled (requires Triton)")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created: {num_params:,} parameters")
    print()

    # Load dataset
    print("[DATASET] Loading TinyStories...")
    data_file = "data/tinystories.txt"

    if not Path(data_file).exists():
        print("[ERROR] TinyStories not found!")
        print("[INFO] Downloading from HuggingFace...")
        from datasets import load_dataset

        dataset = load_dataset("roneneldan/TinyStories", split="train")
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)

        with open(data_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(example['text'].strip() + '\n')

        print(f"[OK] Downloaded {len(dataset)} stories")

    train_dataset = TinyStoriesDataset(data_file, tokenizer, max_seq_len=train_config.max_seq_len)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, pad_token_id=0)
    )

    print(f"[OK] Dataset loaded: {len(train_dataset)} stories")
    print(f"[OK] Batch size: {train_config.batch_size}")
    print(f"[OK] Workers: {train_config.num_workers}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print("[START] Training BDH with BBPE on TinyStories")
    print("="*70)
    print(f"   Epochs: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Max seq len: {train_config.max_seq_len}")
    print(f"   Vocab size: {tokenizer.vocab_size_actual}")
    print("="*70)
    print()

    best_val_loss = float('inf')
    t0 = time.time()

    global_batch = 0

    for epoch in range(train_config.max_iters):
        epoch_start = time.time()

        # Train
        train_loss, num_batches = train_epoch(
            model, train_loader,
            optimizer, train_config, device, scaler,
            checkpoint_dir=train_config.checkpoint_dir,
            global_batch=global_batch
        )

        global_batch += num_batches

        epoch_time = time.time() - epoch_start

        print()
        print(f"[EPOCH {epoch+1}/{train_config.max_iters}]")
        print(f"   Loss: {train_loss:.4f}")
        print(f"   Epoch Time: {epoch_time/60:.1f} minutes")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            utilization = (allocated / total) * 100
            print(f"   GPU Memory: {allocated:.1f}GB / {total:.0f}GB ({utilization:.0f}%)")
        print()

        # Save checkpoint
        save_checkpoint(
            model, optimizer, train_config, epoch + 1, train_loss,
            train_config.checkpoint_dir, is_best=(train_loss < best_val_loss)
        )

        if train_loss < best_val_loss:
            best_val_loss = train_loss

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
    print(f"[OK] Final loss: {train_loss:.4f}")
    print(f"[OK] Best loss: {best_val_loss:.4f}")
    print(f"[OK] Checkpoints: {train_config.checkpoint_dir}")
    print()


def main():
    """Main function."""

    print("="*70)
    print("BDH TRAINING WITH BBPE TOKENIZER")
    print("Dataset: TinyStories")
    print("="*70)
    print()

    # Model configuration
    config = MultiScaleBDHConfig(
        vocab_size=8192,  # Will be set by tokenizer
        n_embd=512,
        n_layer=8,
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
        max_seq_len=512,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    # Training configuration
    class TrainConfig:
        max_iters = 1  # epochs (reduced to 1)
        batch_size = 32
        learning_rate = 1e-4  # Lower LR to prevent NaN
        weight_decay = 0.01
        max_seq_len = 512
        vocab_size = 8192
        num_workers = 4
        grad_clip = 0.5  # Stronger gradient clipping
        checkpoint_dir = "checkpoints/bbh_bdh_tinystories"

    train_config = TrainConfig()

    print("MODEL CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Embedding dim: {config.n_embd}")
    print(f"   Layers: {config.n_layer}")
    print(f"   Heads: {config.n_head}")
    print(f"   Decay rates: {config.decay_rates}")
    print()

    print("TRAINING CONFIGURATION:")
    print(f"   Epochs: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Max seq len: {train_config.max_seq_len}")
    print(f"   Vocab size: {train_config.vocab_size}")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
