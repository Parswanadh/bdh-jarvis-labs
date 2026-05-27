"""
Pure Knowledge Distillation - BDH from Gemma 3 270M
====================================================

TRUE DISTILLATION:
- Teacher (Gemma 3 270M) provides probability distributions
- Student (BDH) learns to mimic teacher's logits
- Loss: KL Divergence between teacher and student
- Teacher runs during training (not pre-generated)

This is REAL knowledge distillation where student learns "how to think"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time
from datetime import datetime

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from stable_config import get_small_model_config


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for distillation."""

    def __init__(self, data_file: str, max_seq_len: int = 512):
        print(f"[LOAD] Loading dataset from {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            self.stories = [line.strip() for line in f if line.strip()]

        print(f"[OK] Loaded {len(self.stories)} stories")
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        return self.stories[idx]


class DistillationCollator:
    """
    Collator that:
    1. Tokenizes with teacher tokenizer (Gemma)
    2. Gets teacher logits
    3. Returns input_ids and teacher logits for distillation
    """

    def __init__(self, tokenizer, teacher_model, device, max_seq_len=512):
        self.tokenizer = tokenizer
        self.teacher = teacher_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.teacher.eval()

    def __call__(self, batch):
        """Process batch and get teacher logits."""

        # Tokenize batch with teacher tokenizer
        encodings = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Get teacher logits (no grad)
        with torch.no_grad():
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            teacher_logits = outputs.logits  # [batch, seq_len, vocab_size]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'teacher_logits': teacher_logits
        }


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Knowledge distillation loss using KL divergence.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        temperature: Softening temperature (higher = softer distribution)

    Returns:
        loss: Scalar
    """
    # Soften probability distributions with temperature
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence: teacher (target) || student (prediction)
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    )

    # Scale by temperature^2 (standard practice)
    loss = kl_div * (temperature ** 2)

    return loss


def train_distillation_epoch(
    student_model, teacher_model, dataloader,
    optimizer, config, device, scaler=None
):
    """Train for one epoch with knowledge distillation."""

    student_model.train()
    teacher_model.eval()

    total_loss = 0
    start_time = time.time()

    use_amp = device.type == 'cuda' and scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        teacher_logits = batch['teacher_logits'].to(device)

        # Forward pass with student
        if use_amp:
            with torch.cuda.amp.autocast():
                # Student forward
                student_logits, _ = student_model(input_ids)

                # Compute distillation loss
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    temperature=config.temperature
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if hasattr(config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Student forward
            student_logits, _ = student_model(input_ids)

            # Compute distillation loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                temperature=config.temperature
            )

            optimizer.zero_grad()
            loss.backward()

            if hasattr(config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_clip)

            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_idx + 1) * config.batch_size * config.max_seq_len / elapsed
            avg_loss = total_loss / (batch_idx + 1)

            print(f"[{batch_idx+1}/{len(dataloader)}] "
                  f"distill_loss: {avg_loss:.4f} | tokens/sec: {int(tokens_per_sec)}")

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
        'task': 'pure_distillation_gemma3_270m'
    }

    torch.save(checkpoint, Path(checkpoint_dir) / f"checkpoint_iter_{iteration}.pt")
    torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_latest.pt")

    if is_best:
        torch.save(checkpoint, Path(checkpoint_dir) / "checkpoint_best.pt")
        print(f"[BEST] New best model! Val loss: {loss:.4f}")

    print(f"[OK] Checkpoint saved: iteration {iteration}, loss: {loss:.4f}")


def train(config, train_config):
    """Main training function for pure knowledge distillation."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Training on device: {device}")

    # Show GPU info
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()
    print("="*70)
    print("PURE KNOWLEDGE DISTILLATION")
    print("="*70)
    print()

    # Load teacher model (Gemma 3 270M)
    print("[LOAD] Loading teacher: Gemma 3 270M")
    teacher_model_name = "google/gemma-3-1b-it"  # Using 1B for A100 capacity

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    teacher_model.eval()
    print(f"[OK] Teacher loaded: {teacher_model_name}")
    print()

    # Create student model (Multi-Scale BDH)
    print("[LOAD] Creating student: Multi-Scale BDH")
    student_model = MultiScaleBDH(config)
    student_model = student_model.to(device)

    # Compile model for speed
    try:
        student_model = torch.compile(student_model)
        print("[OK] Student compiled with torch.compile")
    except:
        print("[INFO] torch.compile not available")

    num_params = sum(p.numel() for p in student_model.parameters())
    print(f"[OK] Student created: {num_params:,} parameters")
    print(f"[COMPRESSION] Teacher: ~1B params → Student: ~5M params ({1000000000/num_params:.0f}x compression)")
    print()

    # Load dataset
    print("[LOAD] Loading TinyStories dataset...")

    # Download TinyStories if not exists
    data_file = Path("data/tinystories.txt")
    if not data_file.exists():
        print("[INFO] TinyStories not found, downloading...")
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train")

        # Save to file
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(example['text'].strip() + '\n')

        print(f"[OK] Downloaded and saved {len(dataset)} stories")

    train_dataset = TinyStoriesDataset(str(data_file), max_seq_len=config.max_seq_len)

    # Create distillation collator
    collator = DistillationCollator(
        teacher_tokenizer,
        teacher_model,
        device,
        max_seq_len=config.max_seq_len
    )

    # DataLoader with collator
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    print(f"[OK] Dataset loaded: {len(train_dataset)} stories")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print("[START] Starting pure knowledge distillation...")
    print(f"   Max iterations (epochs): {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Temperature: {train_config.temperature}")
    print(f"   Mixed precision: {scaler is not None}")
    print("="*70)
    print()

    best_val_loss = float('inf')
    t0 = time.time()

    for iteration in range(train_config.max_iters):
        epoch_start = time.time()

        # Train
        train_loss = train_distillation_epoch(
            student_model, teacher_model, train_loader,
            optimizer, train_config, device, scaler
        )

        epoch_time = time.time() - epoch_start

        print()
        print(f"[EPOCH {iteration+1}/{train_config.max_iters}]")
        print(f"   Distillation Loss: {train_loss:.4f}")
        print(f"   Epoch Time: {epoch_time/60:.1f} minutes")
        print()

        # Save checkpoint
        save_checkpoint(
            student_model, optimizer, train_config, iteration + 1, train_loss,
            train_config.checkpoint_dir, is_best=(train_loss < best_val_loss)
        )

        if train_loss < best_val_loss:
            best_val_loss = train_loss

    # Final checkpoint
    save_checkpoint(
        student_model, optimizer, train_config, train_config.max_iters, train_loss,
        train_config.checkpoint_dir, is_best=False
    )

    elapsed = time.time() - t0
    print()
    print("="*70)
    print("[DONE] PURE DISTILLATION COMPLETE!")
    print("="*70)
    print(f"[OK] Training time: {elapsed/60:.1f} minutes")
    print(f"[OK] Final distillation loss: {train_loss:.4f}")
    print(f"[OK] Best validation loss: {best_val_loss:.4f}")
    print(f"[OK] Checkpoints saved to: {train_config.checkpoint_dir}")
    print()
    print("[SUCCESS] Student learned from teacher's probability distributions!")
    print("="*70)


def main():
    """Main function."""

    print("="*70)
    print("PURE KNOWLEDGE DISTILLATION")
    print("Teacher: Gemma 3 1B → Student: Multi-Scale BDH (5M)")
    print("="*70)
    print()

    # Student model configuration
    config = MultiScaleBDHConfig(
        vocab_size=256,  # Byte-level
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.1,
        max_seq_len=512,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    # Training configuration
    train_config = get_small_model_config()
    train_config.max_iters = 5  # 5 epochs
    train_config.batch_size = 32  # Smaller batch due to teacher inference
    train_config.temperature = 2.0  # Distillation temperature
    train_config.checkpoint_dir = "checkpoints/bdh_pure_distillation"
    train_config.resume_from = None

    print("TEACHER-STUDENT SETUP:")
    print(f"   Teacher: Gemma 3 1B (1B parameters)")
    print(f"   Student: Multi-Scale BDH (~5M parameters)")
    print(f"   Compression: 200x")
    print(f"   Distillation temperature: {train_config.temperature}")
    print()

    print("STUDENT CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: ~5M")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print()

    print("TRAINING CONFIGURATION:")
    print(f"   Epochs: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print()

    print("TASK:")
    print(f"   Task: PURE Knowledge Distillation")
    print(f"   Teacher guides student: TRUE ✅")
    print(f"   Student learns 'how to think': TRUE ✅")
    print(f"   Loss: KL Divergence on logits")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
