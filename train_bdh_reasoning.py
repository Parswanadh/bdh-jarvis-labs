"""
BDH Science Fest - Training on REASONING TASKS
==================================================

This script trains Multi-Scale BDH on reasoning benchmarks:
- GSM8K-style math word problems
- Logical reasoning tasks
- Commonsense reasoning

Task: Next-token prediction on reasoning datasets
Training includes proper checkpointing for science fair demo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Import our implementations
import sys
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from stable_config import BDHStableTrainingConfig, get_small_model_config


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks.
    Includes math problems, logic puzzles, and commonsense reasoning.
    """
    def __init__(self, problems, max_seq_len=512):
        """
        Args:
            problems: List of (question, answer) tuples
            max_seq_len: Maximum sequence length
        """
        self.problems = problems
        self.max_seq_len = max_seq_len

        # Format: "Q: [question] A: [answer]"
        self.text_data = []
        for question, answer in problems:
            formatted = f"Q: {question} A: {answer}"
            self.text_data.append(formatted)

        # Combine all text
        self.full_text = "\n".join(self.text_data)

        # Convert to bytes
        self.data = torch.tensor(list(self.full_text.encode('utf-8')), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = chunk[:self.max_seq_len]      # Input
        y = chunk[1:self.max_seq_len + 1]  # Target (next token)
        return x, y


def create_gsm8k_style_dataset():
    """
    Create GSM8K-style math word problems dataset.

    GSM8K (Grade School Math 8K) is a benchmark for reasoning.
    We create sample problems for the science fair demo.
    """
    problems = [
        # Arithmetic word problems
        (
            "John has 15 apples. He gives 5 to Mary and buys 3 more. How many apples does John have now?",
            "John has 13 apples"
        ),
        (
            "A school has 240 students. If 3/5 of them are girls, how many boys are there?",
            "There are 96 boys"
        ),
        (
            "A train travels 120 km in 2 hours. What is its average speed in km/h?",
            "The train's average speed is 60 km/h"
        ),
        (
            "If 3 books cost $45, how much do 7 books cost?",
            "7 books cost $105"
        ),
        (
            "A rectangle has length 12 and width 8. What is its area?",
            "The area is 96"
        ),

        # Multi-step reasoning
        (
            "Sarah has twice as many marbles as Tom. Tom has 24 marbles. They have 96 marbles altogether. How many more marbles does Sarah have than Tom?",
            "Sarah has 16 more marbles than Tom"
        ),
        (
            "A factory produces 500 toys per day. If it operates 5 days a week, how many toys are produced in 3 weeks?",
            "7,500 toys are produced in 3 weeks"
        ),
        (
            "The sum of three consecutive numbers is 72. What is the largest number?",
            "The largest number is 25"
        ),

        # Logical reasoning
        (
            "All cats are mammals. All mammals have kidneys. Does Fluffy the cat have kidneys?",
            "Yes, Fluffy has kidneys because all cats are mammals and all mammals have kidneys"
        ),
        (
            "If A is taller than B, and B is taller than C, who is the shortest?",
            "C is the shortest"
        ),
        (
            "In a race, Alice finished before Bob but after Carol. Who won the race?",
            "Carol won the race"
        ),

        # Commonsense reasoning
        (
            "If you drop a glass on a concrete floor, it will likely break. If you drop a glass on a carpeted floor, will it break?",
            "It might break but is less likely to break on a carpeted floor"
        ),
        (
            "To stay warm in winter, you should wear a heavy coat or stay indoors. If it's winter and very cold, should you stay indoors?",
            "Yes, you should stay indoors or wear a heavy coat"
        ),
        (
            "Plants need sunlight and water to grow. If a plant is kept in a dark closet without water, will it grow?",
            "No, the plant will not grow without sunlight and water"
        ),

        # Pattern recognition
        (
            "What comes next in the sequence: 2, 4, 8, 16, ?",
            "32"
        ),
        (
            "What comes next: A, C, E, G, ?",
            "I"
        ),
        (
            "What comes next: 1, 1, 2, 3, 5, ?",
            "8"
        ),

        # Word problems with money
        (
            "If a pen costs $1.50 and a notebook costs $3.00, how much do 3 pens and 2 notebooks cost?",
            "3 pens and 2 notebooks cost $10.50"
        ),
        (
            "A shirt costs $20. If it's on sale for 25% off, what is the sale price?",
            "The sale price is $15"
        ),

        # Time problems
        (
            "A movie starts at 2:30 PM and lasts for 2 hours and 15 minutes. What time does it end?",
            "The movie ends at 4:45 PM"
        ),
        (
            "If you bake cookies for 12 minutes at 350°F, but you only bake for 8 minutes, are they done?",
            "The cookies are not fully done"
        ),

        # Fraction problems
        (
            "Half of a pizza is divided equally among 4 people. What fraction does each person get?",
            "Each person gets 1/8 of the pizza"
        ),
        (
            "If 3/4 of a class is girls and there are 30 girls, how many students are in the class?",
            "There are 40 students in the class"
        ),

        # Age problems
        (
            "John is twice as old as Mary. In 5 years, the sum of their ages will be 35. How old is John now?",
            "John is 10 years old now"
        ),
        (
            "Five years ago, Tom was 12. How old will Tom be in 3 years?",
            "Tom will be 20 in 3 years"
        ),
    ]

    # Repeat problems to create enough training data
    problems_repeated = problems * 50  # 800 problems

    return problems_repeated


def load_fallback_data():
    """
    Load fallback reasoning data if no dataset file exists.
    Includes math, logic, and commonsense reasoning.
    """
    print("📚 Creating reasoning dataset with math, logic, and commonsense problems...")

    # Scientific/AI reasoning (relevant to BDH!)
    ai_reasoning = [
        (
            "Neural networks learn from data. More data usually improves performance. Does more data always help?",
            "More data helps up to a point, but data quality and model architecture also matter"
        ),
        (
            "BDH uses Hebbian learning where neurons that fire together wire together. Does this make BDH biologically plausible?",
            "Yes, Hebbian learning is inspired by real biological neural networks"
        ),
        (
            "Transformers have O(N²) attention complexity. BDH has O(N) linear attention. Is BDH faster for long sequences?",
            "Yes, BDH is much faster for long sequences due to linear attention"
        ),
        (
            "A model with 10M parameters can learn simple patterns. Can it learn complex reasoning?",
            "Simple patterns can be learned, but complex reasoning may require larger models or more training"
        ),
        (
            "Regularization helps prevent overfitting. Does dropout help during training?",
            "Yes, dropout randomly disables neurons during training to prevent overfitting"
        ),
        (
            "Learning rate determines step size in gradient descent. Is higher learning rate always better?",
            "No, learning rate that is too high can cause training instability, while too low can be slow"
        ),
    ]

    # Math reasoning
    math_problems = [
        ("What is 15 × 14?", "15 × 14 = 210"),
        ("What is 144 ÷ 12?", "144 ÷ 12 = 12"),
        ("What is 2⁸?", "2⁸ = 256"),
        ("What is √625?", "√625 = 25"),
        ("What is 3.5 + 4.8?", "3.5 + 4.8 = 8.3"),
        ("What is 15% of 200?", "15% of 200 = 30"),
        ("If x + 5 = 12, what is x?", "x = 7"),
        ("What is the prime factorization of 30?", "2 × 3 × 5"),
    ]

    # Logic reasoning
    logic_problems = [
        (
            "All A are B. All B are C. Are all A also C?",
            "Yes, all A are also C (transitive property)"
        ),
        (
            "Some A are B. Some B are C. Are all A also C?",
            "Not necessarily, we cannot conclude that all A are C"
        ),
        (
            "If P implies Q, and Q is false, what can we say about P?",
            "P must be false (contrapositive)"
        ),
    ]

    # Combine all problems
    all_problems = []

    # Add GSM8K-style problems
    all_problems.extend(create_gsm8k_style_dataset())

    # Add AI reasoning
    all_problems.extend(ai_reasoning * 20)

    # Add math problems
    for q, a in math_problems:
        all_problems.append((f"Solve: {q}", a))

    # Add logic problems
    for q, a in logic_problems:
        all_problems.append((f"Reason: {q}", a))

    # Repeat for training data
    all_problems_repeated = all_problems * 30

    return all_problems_repeated


def save_checkpoint(model, optimizer, config, iteration, loss, checkpoint_dir, is_best=False):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best,
        'task': 'reasoning'
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)

    print(f"✅ Checkpoint saved: iteration {iteration}, loss: {loss:.4f}")


def evaluate(model, dataloader, device, max_batches=10):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, _ = model(x)

            # Calculate loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def train(config, train_config, resume_from=None):
    """Main training loop for reasoning task."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training on device: {device}")

    # Create model
    model = MultiScaleBDH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params:,} parameters")

    # Load reasoning dataset
    print("📊 Loading reasoning dataset...")
    problems = load_fallback_data()

    # Split into train/val
    split_idx = int(len(problems) * 0.9)
    train_problems = problems[:split_idx]
    val_problems = problems[split_idx:]

    train_dataset = ReasoningDataset(train_problems, max_seq_len=config.max_seq_len)
    val_dataset = ReasoningDataset(val_problems, max_seq_len=config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"✅ Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    print(f"📊 Training on reasoning problems: math, logic, commonsense, and AI reasoning")

    # Create checkpoint directory
    checkpoint_dir = train_config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save training config
    with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w') as f:
        json.dump({
            'model_config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'training_config': {k: v for k, v in train_config.__dict__.items() if not k.startswith('_')},
            'task': 'reasoning',
            'date': datetime.now().isoformat()
        }, f, indent=2)

    # Resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float('inf')

    if resume_from and os.path.exists(resume_from):
        start_iteration, _ = load_checkpoint(resume_from, model, optimizer)
        print(f"✅ Resumed from iteration {start_iteration}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )

    # Training loop
    print(f"\n🚀 Starting training for {train_config.max_iters} iterations...")
    print(f"   Task: Reasoning (math, logic, commonsense)")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Checkpoint interval: every {train_config.save_interval} iterations")
    print(f"   Eval interval: every {train_config.eval_interval} iterations")
    print("="*60)

    model.train()
    train_iter = iter(train_loader)
    total_loss = 0.0
    t0 = time.time()

    for iteration in range(start_iteration, train_config.max_iters):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _ = model(x)

        # Calculate loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        loss = F.cross_entropy(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Optimizer step
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Print progress
        if (iteration + 1) % 100 == 0:
            t1 = time.time()
            tokens_per_sec = (iteration + 1) * train_config.batch_size * config.max_seq_len / (t1 - t0)
            avg_loss = total_loss / 100

            print(f"[{iteration+1:5d}/{train_config.max_iters}] "
                  f"loss: {avg_loss:.4f} | "
                  f"tokens/sec: {tokens_per_sec:.0f}")

            total_loss = 0.0

        # Save checkpoint
        if (iteration + 1) % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, config, iteration + 1,
                avg_loss, checkpoint_dir, is_best=False
            )

        # Evaluation
        if (iteration + 1) % train_config.eval_interval == 0:
            val_loss, perplexity = evaluate(model, val_loader, device)

            print(f"\n{'='*60}")
            print(f"📊 VALIDATION [iter {iteration+1}]")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"{'='*60}\n")

            # Save if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, iteration + 1,
                    val_loss, checkpoint_dir, is_best=True
                )
                print(f"🏆 New best model! Val loss: {val_loss:.4f}\n")

            model.train()

    # Final checkpoint
    save_checkpoint(
        model, optimizer, config, train_config.max_iters,
        avg_loss, checkpoint_dir, is_best=False
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print(f"✅ Final checkpoint saved to {checkpoint_dir}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Task: Reasoning (math, logic, commonsense)")
    print("="*60)


def main():
    """Main training function."""
    print("="*60)
    print("BDH Science Fest - Training on REASONING TASKS")
    print("="*60)
    print()

    # Model configuration (10M Multi-Scale BDH)
    config = MultiScaleBDHConfig(
        vocab_size=256,              # Byte-level
        n_embd=256,                  # Embedding dimension
        n_layer=6,                   # Number of layers
        n_head=4,                    # Number of attention heads
        ffn_dim=1024,                # FFN dimension
        dropout=0.1,
        max_seq_len=512,             # Context length
        decay_rates=[0.95, 0.99, 0.995],  # Multi-scale!
        hebbian_lr=0.001            # Lower for stability
    )

    # Training configuration (stable!)
    train_config = get_small_model_config()

    # Override for reasoning training
    train_config.max_iters = 1000      # Quick demo training
    train_config.batch_size = 32       # Good for RTX 4070
    train_config.eval_interval = 500   # Evaluate twice during training
    train_config.save_interval = 500   # Save checkpoints frequently
    train_config.checkpoint_dir = "checkpoints/multiscale_bdh_reasoning"
    train_config.resume_from = None    # Set to checkpoint path to resume

    # Print configuration
    print("📋 MODEL CONFIGURATION:")
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: ~10M")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print()

    print("📋 TRAINING CONFIGURATION:")
    print(f"   Max iterations: {train_config.max_iters}")
    print(f"   Batch size: {train_config.batch_size}")
    print(f"   Learning rate: {train_config.learning_rate}")
    print(f"   Warmup steps: {train_config.warmup_steps}")
    print(f"   Gradient clip: {train_config.grad_clip}")
    print()

    print("📋 TASK:")
    print(f"   Task: Reasoning (math, logic, commonsense)")
    print(f"   Dataset: ~800 reasoning problems (repeated)")
    print(f"   Training time: ~20-30 minutes")
    print()

    # Start training
    train(config, train_config)


if __name__ == "__main__":
    main()
