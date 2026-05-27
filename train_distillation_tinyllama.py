"""
BDH Distillation with TinyLlama (1.1B, 32K vocab)
FITS COMFORTABLY in 8GB VRAM!
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time
import gc

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


class TinyStoriesDataset(Dataset):
    """TinyStories dataset with tokenizer"""

    def __init__(self, data_file, tokenizer, max_seq_len=256, max_stories=None):
        print(f"[LOAD] Loading TinyStories...")

        self.stories = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_stories and i >= max_stories:
                    break
                story = line.strip()
                if story:
                    self.stories.append(story)
                if (i + 1) % 50000 == 0:
                    print(f"   Loaded {i + 1} stories...")

        print(f"[OK] Loaded {len(self.stories)} stories")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        encoded = self.tokenizer(
            story,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            pad_to_multiple_of=None
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def clear_memory():
    """Clear CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("="*70)
    print("BDH DISTILLATION WITH TINYLLAMA (8GB VRAM OPTIMIZED)")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    if torch.cuda.is_available():
        print(f"[VRAM] Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load teacher (TinyLlama - 32K vocab, NOT 262K!)
    print("[TEACHER] Loading TinyLlama 1.1B...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Fix pad token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()

    teacher_vocab_size = teacher_model.config.vocab_size

    print(f"[OK] Teacher loaded!")
    print(f"   Parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"   Vocab size: {teacher_vocab_size:,}")
    print()

    # Clear memory
    clear_memory()

    # Student configuration - NOW MUCH SMALLER!
    print("[STUDENT] Creating BDH model...")
    config = MultiScaleBDHConfig(
        vocab_size=teacher_vocab_size,  # 32K instead of 262K!
        n_embd=512,          # Can afford larger now
        n_layer=8,           # Full 8 layers
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
        max_seq_len=256,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    model = MultiScaleBDH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"[OK] Student model: {num_params:,} parameters")
    print(f"   Estimated VRAM: {num_params * 4 / 1e9:.2f} GB")
    print()

    # Load dataset
    print("[DATA] Loading TinyStories (100K stories)...")
    dataset = TinyStoriesDataset(
        "data/tinystories.txt",
        teacher_tokenizer,
        max_seq_len=256,
        max_stories=100000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Can use batch_size=8 now!
        shuffle=True,
        num_workers=0
    )
    print(f"[OK] Batches: {len(dataloader)}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # Checkpoint directory
    checkpoint_dir = Path("checkpoints/distillation_tinyllama")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training settings
    max_hours = 6
    checkpoint_interval = 30 * 60
    start_time = time.time()
    last_checkpoint = time.time()

    print("="*70)
    print(f"TRAINING FOR {max_hours} HOURS")
    print(f"Teacher: TinyLlama 1.1B (32K vocab)")
    print(f"Student: BDH {num_params // 1e6:.0f}M params")
    print(f"Batch size: 8")
    print(f"Checkpoint every: {checkpoint_interval // 60} min")
    print("="*70)
    print()

    # Training loop
    global_step = 0
    total_loss = 0

    try:
        while True:
            elapsed = time.time() - start_time
            elapsed_hours = elapsed / 3600

            if elapsed_hours >= max_hours:
                print(f"\n[TIME] Reached {max_hours} hours! Stopping.")
                break

            # Checkpoint
            if time.time() - last_checkpoint >= checkpoint_interval:
                avg_loss = total_loss / max(1, global_step % 1000)
                checkpoint_path = checkpoint_dir / f"checkpoint_{int(elapsed_hours)}h.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'global_step': global_step,
                    'elapsed_hours': elapsed_hours,
                    'config': config
                }, checkpoint_path)
                print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                clear_memory()
                last_checkpoint = time.time()

            model.train()
            for batch_idx, batch in enumerate(dataloader):
                elapsed = time.time() - start_time
                if elapsed / 3600 >= max_hours:
                    break

                if time.time() - last_checkpoint >= checkpoint_interval:
                    avg_loss = total_loss / max(1, global_step % 1000)
                    elapsed_hours = elapsed / 3600
                    checkpoint_path = checkpoint_dir / f"checkpoint_{int(elapsed_hours)}h.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'global_step': global_step,
                        'elapsed_hours': elapsed_hours,
                        'config': config
                    }, checkpoint_path)
                    print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                    clear_memory()
                    last_checkpoint = time.time()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Teacher forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits

                # Student forward pass
                with torch.cuda.amp.autocast():
                    student_logits, _ = model(input_ids)

                    if global_step == 0:
                        print(f"[DEBUG] teacher_logits: {teacher_logits.shape}")
                        print(f"[DEBUG] student_logits: {student_logits.shape}")

                    B, T, V = student_logits.shape

                    # Shift for next token prediction
                    shift_student_logits = student_logits[:, :-1, :]
                    shift_teacher_logits = teacher_logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    new_T = shift_student_logits.size(1)

                    # Reshape
                    shift_student_logits = shift_student_logits.reshape(B * new_T, V)
                    shift_teacher_logits = shift_teacher_logits.reshape(B * new_T, V)
                    shift_labels = shift_labels.reshape(B * new_T)

                    # Distillation loss
                    temperature = 2.0
                    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
                    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

                    kl_div = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction='batchmean'
                    ) * (temperature ** 2)

                    # Cross-entropy loss
                    ce_loss = F.cross_entropy(
                        shift_student_logits,
                        shift_labels,
                        ignore_index=teacher_tokenizer.pad_token_id
                    )

                    loss = 0.7 * kl_div + 0.3 * ce_loss

                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % 50 == 0:
                    avg_loss = total_loss / 50
                    elapsed_min = (time.time() - start_time) / 60
                    elapsed_hr = elapsed_min / 60
                    tok_per_sec = global_step * 8 * 256 / (time.time() - start_time)

                    # Print memory
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.max_memory_allocated() / 1e9
                        vram_reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"[Step {global_step}] "
                              f"loss: {avg_loss:.4f} | "
                              f"time: {elapsed_hr:.2f}h | "
                              f"tok/s: {tok_per_sec:.0f} | "
                              f"VRAM: {vram_used:.2f}GB / {vram_reserved:.2f}GB")

                    total_loss = 0

    except KeyboardInterrupt:
        print()
        print("[INTERRUPT] Training stopped by user")

    # Final save
    elapsed_hours = (time.time() - start_time) / 3600
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'elapsed_hours': elapsed_hours
    }, final_path)

    print()
    print("="*70)
    print(f"[DONE] Training complete in {elapsed_hours:.2f} hours")
    print(f"Saved to {final_path}")
    print("="*70)


if __name__ == "__main__":
    main()
