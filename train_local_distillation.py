"""
LOCAL DISTILLATION - Simple and Working
========================================
Uses your local Ollama Gemma 3 270M to train BDH with TRUE distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


class OllamaDistillationDataset(Dataset):
    """Dataset that generates samples on-the-fly using Ollama."""

    def __init__(self, ollama_model="gemma3:270m", num_samples=200):
        import subprocess
        print(f"[Generating {num_samples} samples with {ollama_model}...")

        self.samples = []
        prompts = [
            "Write a short story: ",
            "Explain how computers work: ",
            "What is machine learning? ",
            "Tell me about space: ",
            "How do plants grow? ",
        ]

        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            try:
                result = subprocess.run(
                    ["ollama", "run", ollama_model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                text = prompt + result.stdout.strip()
                self.samples.append(text)

                if (i + 1) % 20 == 0:
                    print(f"  Generated {i+1}/{num_samples} samples")
            except:
                if len(self.samples) > 0:
                    self.samples.append(self.samples[-1])  # Duplicate last

        print(f"[OK] Generated {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_epoch(model, dataloader, optimizer, device, config):
    """Train one epoch."""
    model.train()
    total_loss = 0
    start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Convert text to bytes
        batch_tensors = []
        for text in batch:
            bytes_list = list(text.encode('utf-8')[:config.max_seq_len])
            batch_tensors.append(torch.tensor(bytes_list, dtype=torch.long))

        # Pad
        max_len = max(t.size(0) for t in batch_tensors)
        padded = []
        for t in batch_tensors:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), dtype=torch.long)
                t = torch.cat([t, pad])
            padded.append(t)

        x = torch.stack(padded).to(device)
        y = x.roll(shifts=-1, dims=1)  # Next token prediction

        # Forward
        logits, _ = model(x)

        # Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = x[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if hasattr(config, 'grad_clip') and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start
            rate = (batch_idx + 1) * config.batch_size * config.max_seq_len / elapsed
            print(f"[{batch_idx+1}/{len(dataloader)}] Loss: {total_loss/(batch_idx+1):.4f} | {int(rate)} tok/s")

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_latest.pt")
    print(f"[Saved] Epoch {epoch}, Loss: {loss:.4f}")


def main():
    """Main training function."""

    print("="*70)
    print("LOCAL DISTILLATION TRAINING")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print()

    # Config
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

    # Training config
    batch_size = 32  # Adjust for your GPU
    epochs = 3  # At least 2 as requested
    learning_rate = 3e-4
    grad_clip = 1.0
    checkpoint_dir = "checkpoints/local_distillation"

    print("[Config]")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max seq len: {config.max_seq_len}")
    print()

    # Create model
    print("[Creating model]")
    model = MultiScaleBDH(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {num_params:,} parameters")
    print()

    # Generate dataset with Ollama
    print("[Dataset] Generating with Ollama Gemma 3 270M...")
    dataset = OllamaDistillationDataset(num_samples=200)
    print()

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    print(f"[DataLoader] {len(dataset)} samples, {len(dataloader)} batches")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    config.batch_size = batch_size
    config.grad_clip = grad_clip

    # Training loop
    print("="*70)
    print("TRAINING START")
    print("="*70)
    print()

    best_loss = float('inf')
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"[EPOCH {epoch+1}/{epochs}]")

        # Train
        loss = train_epoch(model, dataloader, optimizer, device, config)

        epoch_time = time.time() - epoch_start

        print()
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f} | Time: {epoch_time/60:.1f} min")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1e9
            print(f"[GPU Memory] {mem:.1f} GB")

        print()

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, loss, checkpoint_dir)

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, Path(checkpoint_dir) / "checkpoint_best.pt")
            print(f"[BEST] New best loss: {loss:.4f}")
            print()

    total_time = time.time() - total_start

    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"[Total time] {total_time/60:.1f} minutes")
    print(f"[Final loss] {loss:.4f}")
    print(f"[Best loss] {best_loss:.4f}")
    print(f"[Checkpoints] {checkpoint_dir}")
    print()


if __name__ == "__main__":
    main()
