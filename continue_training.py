"""
Continue Training from Checkpoint - Train for 12 more hours
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("="*70)
    print("CONTINUE DISTILLATION TRAINING")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Load checkpoint
    checkpoint_path = "checkpoints/distillation_tinyllama/final_model.pt"
    print(f"[LOAD] Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    config = checkpoint['config']
    elapsed_so_far = checkpoint.get('elapsed_hours', 6.0)

    print(f"[OK] Resuming from checkpoint")
    print(f"   Trained for: {elapsed_so_far:.2f} hours")
    print(f"   Student params: {sum(p.numel() for p in checkpoint['model_state_dict'].values()):,}")
    print()

    # Load teacher
    print("[TEACHER] Loading TinyLlama 1.1B...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()
    print(f"[OK] Teacher loaded")
    print()

    # Load student from checkpoint
    print("[STUDENT] Loading student model...")
    model = MultiScaleBDH(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] Student loaded")
    print()

    # Clear memory
    clear_memory()

    # Load dataset
    print("[DATA] Loading TinyStories (more stories)...")
    dataset = TinyStoriesDataset(
        "data/tinystories.txt",
        teacher_tokenizer,
        max_seq_len=256,
        max_stories=200000  # Use more stories this time
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    print(f"[OK] Batches: {len(dataloader)}")
    print()

    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # Lower LR
    scaler = torch.cuda.amp.GradScaler()

    # Training settings
    additional_hours = 12  # Train for 12 more hours
    checkpoint_interval = 30 * 60
    start_time = time.time()
    last_checkpoint = time.time()
    global_step = 0
    total_loss = 0

    print("="*70)
    print(f"TRAINING FOR {additional_hours} MORE HOURS")
    print(f"Total training time will be: {elapsed_so_far + additional_hours:.0f} hours")
    print(f"Learning rate: 1e-4 (lowered for fine-tuning)")
    print(f"Dataset: 200K stories (increased from 100K)")
    print("="*70)
    print()

    try:
        while True:
            elapsed = time.time() - start_time
            elapsed_hours = elapsed / 3600

            if elapsed_hours >= additional_hours:
                print(f"\n[TIME] Reached {additional_hours} additional hours!")
                break

            if time.time() - last_checkpoint >= checkpoint_interval:
                avg_loss = total_loss / max(1, global_step % 1000)
                total_elapsed = elapsed_so_far + elapsed_hours
                checkpoint_path = f"checkpoints/distillation_tinyllama/checkpoint_{int(total_elapsed)}h.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'global_step': global_step,
                    'elapsed_hours': total_elapsed,
                    'config': config
                }, checkpoint_path)
                print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                clear_memory()
                last_checkpoint = time.time()

            model.train()
            for batch_idx, batch in enumerate(dataloader):
                elapsed = time.time() - start_time
                if elapsed / 3600 >= additional_hours:
                    break

                if time.time() - last_checkpoint >= checkpoint_interval:
                    avg_loss = total_loss / max(1, global_step % 1000)
                    total_elapsed = elapsed_so_far + elapsed_hours
                    checkpoint_path = f"checkpoints/distillation_tinyllama/checkpoint_{int(total_elapsed)}h.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'global_step': global_step,
                        'elapsed_hours': total_elapsed,
                        'config': config
                    }, checkpoint_path)
                    print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                    clear_memory()
                    last_checkpoint = time.time()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits

                with torch.cuda.amp.autocast():
                    student_logits, _ = model(input_ids)

                    B, T, V = student_logits.shape

                    shift_student_logits = student_logits[:, :-1, :]
                    shift_teacher_logits = teacher_logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    new_T = shift_student_logits.size(1)

                    shift_student_logits = shift_student_logits.reshape(B * new_T, V)
                    shift_teacher_logits = shift_teacher_logits.reshape(B * new_T, V)
                    shift_labels = shift_labels.reshape(B * new_T)

                    temperature = 2.0
                    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
                    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)

                    kl_div = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction='batchmean'
                    ) * (temperature ** 2)

                    ce_loss = F.cross_entropy(
                        shift_student_logits,
                        shift_labels,
                        ignore_index=teacher_tokenizer.pad_token_id
                    )

                    loss = 0.7 * kl_div + 0.3 * ce_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = total_loss / 50
                    elapsed_min = (time.time() - start_time) / 60
                    elapsed_hr = elapsed_min / 60
                    total_hr = elapsed_so_far + elapsed_hr
                    tok_per_sec = global_step * 8 * 256 / (time.time() - start_time)

                    if torch.cuda.is_available():
                        vram_used = torch.cuda.max_memory_allocated() / 1e9
                        print(f"[Step {global_step}] "
                              f"loss: {avg_loss:.4f} | "
                              f"total time: {total_hr:.1f}h | "
                              f"tok/s: {tok_per_sec:.0f} | "
                              f"VRAM: {vram_used:.2f}GB")

                    total_loss = 0

    except KeyboardInterrupt:
        print()
        print("[INTERRUPT] Training stopped")

    total_elapsed = elapsed_so_far + (time.time() - start_time) / 3600
    final_path = "checkpoints/distillation_tinyllama/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'elapsed_hours': total_elapsed
    }, final_path)

    print()
    print("="*70)
    print(f"[DONE] Training complete in {total_elapsed:.2f} total hours")
    print(f"Saved to {final_path}")
    print("="*70)


if __name__ == "__main__":
    main()
