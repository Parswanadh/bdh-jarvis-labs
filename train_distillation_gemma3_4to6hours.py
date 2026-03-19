"""
BDH Training with Knowledge Distillation from Gemma 3 270M
Trains for 4-6 hours with checkpointing every 30 minutes
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time
import json

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

class TinyStoriesDataset(Dataset):
    """TinyStories dataset with checkpoint support"""

    def __init__(self, data_file, tokenizer, max_seq_len=256, max_stories=None):
        print(f"[LOAD] Loading TinyStories from {data_file}")

        self.stories = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_stories and i >= max_stories:
                    break
                story = line.strip()
                if story:  # Skip empty lines
                    self.stories.append(story)
                if (i + 1) % 100000 == 0:
                    print(f"   Loaded {i + 1} stories...")

        print(f"[OK] Loaded {len(self.stories)} stories")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        # Tokenize with teacher tokenizer
        encoded = self.tokenizer(
            story,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


class DistillationTrainer:
    """Knowledge Distillation Trainer with Checkpointing"""

    def __init__(
        self,
        teacher_model,
        teacher_tokenizer,
        student_config,
        checkpoint_dir="checkpoints/distillation_gemma3",
        checkpoint_interval=30  # minutes
    ):
        self.teacher = teacher_model.eval()
        self.teacher_tokenizer = teacher_tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval * 60  # convert to seconds

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize student
        self.student = MultiScaleBDH(student_config).to(self.device)
        self.num_params = sum(p.numel() for p in self.student.parameters())
        print(f"[OK] Student model: {self.num_params:,} parameters")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=2e-4,  # Conservative for stability
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.start_time = None
        self.last_checkpoint_time = None
        self.global_step = 0

    def save_checkpoint(self, epoch, batch_idx, loss, metrics):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"

        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'global_step': self.global_step,
            'config': self.student.config
        }, checkpoint_path)

        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'global_step': self.global_step,
            'config': self.student.config
        }, latest_path)

        print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"

        if not checkpoint_path.exists():
            print("[INFO] No checkpoint found, starting from scratch")
            return 0, 0

        print(f"[LOAD] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']

        print(f"[OK] Resumed from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
        return checkpoint['epoch'], checkpoint['batch_idx']

    def get_teacher_logits(self, input_ids, attention_mask):
        """Get teacher logits for distillation"""
        with torch.no_grad():
            # Teacher forward pass
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            teacher_logits = outputs.logits

        return teacher_logits

    def compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        input_ids,
        attention_mask,
        temperature=2.0,
        alpha=0.5  # Balance between distillation and CE loss
    ):
        """Compute distillation loss"""
        # Shift for next token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        # Reshape
        B, T, V = shift_logits.shape
        shift_logits = shift_logits.view(-1, V)
        shift_labels = shift_labels.view(-1)
        shift_teacher_logits = shift_teacher_logits.view(-1, V)
        shift_attention_mask = shift_attention_mask.view(-1)

        # Filter out padding tokens
        mask = shift_attention_mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        shift_teacher_logits = shift_teacher_logits[mask]

        # Distillation loss (KL divergence)
        student_log_probs = F.log_softmax(shift_logits / temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Combined loss
        loss = alpha * kl_div + (1 - alpha) * ce_loss

        return loss

    def train_epoch(self, dataloader, epoch, start_batch=0, max_duration_hours=6):
        """Train for one epoch with time-based checkpointing"""
        self.student.train()
        total_loss = 0
        num_batches = len(dataloader)

        self.start_time = time.time()
        self.last_checkpoint_time = time.time()

        # Iterate batches
        for batch_idx in range(start_batch, num_batches):
            # Check elapsed time
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= max_duration_hours:
                print(f"\n[TIME] Reached {max_duration_hours} hours training!")
                break

            # Check if we need to save checkpoint
            time_since_checkpoint = time.time() - self.last_checkpoint_time
            if time_since_checkpoint >= self.checkpoint_interval:
                avg_loss = total_loss / (batch_idx - start_batch + 1)
                metrics = {
                    'elapsed_hours': elapsed_hours,
                    'tokens_per_sec': self.global_step * 32 * 256 / (time.time() - self.start_time)
                }
                self.save_checkpoint(epoch, batch_idx, avg_loss, metrics)
                self.last_checkpoint_time = time.time()

            # Get batch
            batch = dataloader.dataset[batch_idx]
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)  # Add batch dim
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)

            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Student forward pass with gradient computation
            with torch.cuda.amp.autocast():
                student_logits, _ = self.student(input_ids)

                # Compute distillation loss
                loss = self.compute_distillation_loss(
                    student_logits,
                    teacher_logits,
                    input_ids,
                    attention_mask,
                    temperature=2.0,
                    alpha=0.7  # More weight on distillation
                )

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx - start_batch + 1)
                elapsed = time.time() - self.start_time
                elapsed_min = elapsed / 60
                elapsed_hr = elapsed / 3600
                tok_per_sec = (batch_idx - start_batch + 1) * 256 / elapsed

                print(f"[Epoch {epoch}] [{batch_idx+1}/{num_batches}] "
                      f"loss: {avg_loss:.4f} | "
                      f"time: {elapsed_hr:.2f}h ({elapsed_min:.1f}min) | "
                      f"tok/s: {tok_per_sec:.0f}")

        # Final checkpoint for epoch
        avg_loss = total_loss / max(1, num_batches - start_batch)
        elapsed_hours = (time.time() - self.start_time) / 3600
        metrics = {
            'elapsed_hours': elapsed_hours,
            'tokens_per_sec': self.global_step * 32 * 256 / (time.time() - self.start_time)
        }
        self.save_checkpoint(epoch, num_batches - 1, avg_loss, metrics)

        return avg_loss

    def train(self, train_loader, num_epochs=3, max_duration_hours=6):
        """Main training loop"""
        print()
        print("="*70)
        print("DISTILLATION TRAINING START")
        print("="*70)
        print(f"Teacher: Gemma 3 270M")
        print(f"Student: BDH {self.num_params:,} params")
        print(f"Max duration: {max_duration_hours} hours")
        print(f"Checkpoint interval: {self.checkpoint_interval // 60} minutes")
        print("="*70)
        print()

        # Try to load checkpoint
        start_epoch, start_batch = self.load_checkpoint()

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print()
            print(f"[EPOCH {epoch + 1}/{num_epochs}]")
            print("-" * 70)

            avg_loss = self.train_epoch(
                train_loader,
                epoch,
                start_batch=start_batch if epoch == start_epoch else 0,
                max_duration_hours=max_duration_hours
            )

            print()
            print(f"[EPOCH {epoch + 1} COMPLETE] Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            epoch_checkpoint = self.checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'config': self.student.config
            }, epoch_checkpoint)
            print(f"[CHECKPOINT] Saved epoch {epoch + 1} to {epoch_checkpoint}")

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'config': self.student.config
        }, final_path)

        print()
        print("="*70)
        print("[SUCCESS] Training complete!")
        print(f"Final model saved to {final_path}")
        print("="*70)


def main():
    print("="*70)
    print("BDH DISTILLATION TRAINING WITH GEMMA 3 270M")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    print()

    # Load teacher
    print("[TEACHER] Loading Gemma 3 270M...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()
    vocab_size = len(teacher_tokenizer)
    num_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"[OK] Teacher loaded: {num_params:,} parameters")
    print(f"   Vocabulary size: {vocab_size:,}")
    print()

    # Student configuration (match teacher vocab)
    print("[STUDENT] Initializing BDH...")
    student_config = MultiScaleBDHConfig(
        vocab_size=vocab_size,  # Match teacher!
        n_embd=512,
        n_layer=8,
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
        max_seq_len=256,  # Shorter for faster training
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )
    print()

    # Load dataset (smaller subset for 4-6 hours)
    print("[DATA] Loading TinyStories...")
    dataset = TinyStoriesDataset(
        "data/tinystories.txt",
        teacher_tokenizer,
        max_seq_len=256,
        max_stories=100000  # 100K stories for faster training
    )
    print()

    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_config=student_config,
        checkpoint_dir="checkpoints/distillation_gemma3_4to6h",
        checkpoint_interval=30  # Save every 30 minutes
    )

    # Train for 4-6 hours
    print("[START] Training for 4-6 hours...")
    trainer.train(
        train_loader=dataset,  # Pass dataset directly (we index it manually)
        num_epochs=3,
        max_duration_hours=6
    )


if __name__ == "__main__":
    main()
