"""
Generate Teacher Logits for 1 Hour - Then Train Student
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_len=512):
        print(f"[LOAD] Loading TinyStories...")
        stories = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                stories.append(line.strip())
                if i >= 500000:
                    break
        print(f"[OK] Loaded {len(stories)} stories")
        self.stories = stories
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        token_ids = self.tokenizer.encode(story)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        return token_ids, story


def generate_teacher_logits(teacher_model, teacher_tokenizer, student_tokenizer, duration_minutes=60):
    """Generate teacher logits for 1 hour."""

    print("="*70)
    print("GENERATING TEACHER LOGITS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    dataset = TinyStoriesDataset("data/tinystories.txt", student_tokenizer)

    def collate_fn(batch):
        token_ids_list = [item[0] for item in batch]
        max_len = max(len(ids) if isinstance(ids, list) else ids.size(0) for ids in token_ids_list)

        padded = []
        for ids in token_ids_list:
            if isinstance(ids, torch.Tensor):
                ids_list = ids.tolist()
            else:
                ids_list = ids
            if len(ids_list) < max_len:
                ids_list = ids_list + [0] * (max_len - len(ids_list))
            padded.append(torch.tensor(ids_list, dtype=torch.long))

        return torch.stack(padded)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    print(f"[OK] Batches: {len(dataloader)}")
    print()
    print("[START] Generating logits for 1 hour...")
    print("="*70)

    teacher_model.eval()
    all_logits = []
    all_input_ids = []

    start = time.time()
    target_time = duration_minutes * 60
    batch_count = 0

    with torch.no_grad():
        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)

            outputs = teacher_model(input_ids)
            logits = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_input_ids.append(input_ids.cpu().numpy())

            batch_count += 1

            if batch_count % 10 == 0:
                elapsed = time.time() - start
                remaining = max(0, target_time - elapsed)
                print(f"[{batch_count}/{len(dataloader)}] elapsed: {elapsed/60:.1f}min | remaining: {remaining/60:.1f}min")

            elapsed = time.time() - start
            if elapsed >= target_time:
                print(f"\n[TIME] Reached {duration_minutes} minutes!")
                break

    print()
    print("[SAVE] Saving logits...")

    all_logits = np.concatenate(all_logits, axis=0)
    all_input_ids = np.concatenate(all_input_ids, axis=0)

    output_file = f"data/teacher_logits_{duration_minutes}min.h5"
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('logits', data=all_logits, compression='gzip')
        f.create_dataset('input_ids', data=all_input_ids, compression='gzip')
        f.attrs['num_batches'] = batch_count
        f.attrs['vocab_size'] = logits.shape[-1]
        f.attrs['seq_len'] = logits.shape[1]

    print(f"[OK] Saved to {output_file}")
    print(f"   Shape: {all_logits.shape}")
    print(f"   Size: {Path(output_file).stat().st_size / 1e9:.2f} GB")

    return output_file


def train_student_on_logits(logits_file, epochs=3):
    """Train student model on pre-generated logits."""

    print()
    print("="*70)
    print("TRAINING STUDENT ON PRE-GENERATED LOGITS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    print(f"[LOAD] Loading logits from {logits_file}")
    with h5py.File(logits_file, 'r') as f:
        logits = f['logits'][:]
        input_ids = f['input_ids'][:]
        num_batches = f.attrs['num_batches']
        vocab_size = f.attrs['vocab_size']
        seq_len = f.attrs['seq_len']

    print(f"[OK] Loaded {num_batches} batches")
    print(f"   Logits shape: {logits.shape}")
    print()

    config = MultiScaleBDHConfig(
        vocab_size=vocab_size,
        n_embd=512,
        n_layer=8,
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
        max_seq_len=seq_len,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    model = MultiScaleBDH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Student model: {num_params:,} parameters")
    print()

    class LogitsDataset(Dataset):
        def __init__(self, input_ids, logits):
            self.input_ids = torch.tensor(input_ids, dtype=torch.long)
            self.logits = torch.tensor(logits, dtype=torch.float32)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.logits[idx]

    dataset = LogitsDataset(input_ids, logits)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    print(f"[OK] Training batches: {len(dataloader)}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    print("[START] Training student...")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start = time.time()

        for batch_idx, (batch_input_ids, batch_logits) in enumerate(dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_logits = batch_logits.to(device)

            with torch.cuda.amp.autocast():
                student_logits, _ = model(batch_input_ids)

                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(batch_logits, dim=-1)

                kl_div = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='none'
                )

                mask = (batch_input_ids != 0).unsqueeze(-1)
                kl_div = (kl_div * mask).sum() / mask.sum()

                loss = kl_div

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                avg_loss = total_loss / (batch_idx + 1)
                print(f"[Epoch {epoch+1}] [{batch_idx+1}/{len(dataloader)}] loss: {avg_loss:.4f} | time: {elapsed/60:.1f}min")

        epoch_time = time.time() - start
        avg_loss = total_loss / len(dataloader)
        print()
        print(f"[EPOCH {epoch+1} COMPLETE]")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time/60:.1f} minutes")
        print()

        Path("checkpoints/student_from_logits").mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'epoch': epoch + 1,
            'config': config
        }, f"checkpoints/student_from_logits/epoch_{epoch+1}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, "checkpoints/student_from_logits/final_model.pt")

    print("="*70)
    print("[DONE] Student Training Complete!")
    print(f"Saved: checkpoints/student_from_logits/final_model.pt")
    print("="*70)


def main():
    """Main function."""
    print("="*70)
    print("TEACHER LOGIT GENERATION + STUDENT TRAINING")
    print("="*70)
    print()

    print("[TEACHER] Loading TinyLlama...")
    teacher_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()
    print(f"[OK] Teacher loaded: 1.1B parameters")
    print()

    student_tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")
    print(f"[OK] Student tokenizer vocab: {student_tokenizer.vocab_size_actual}")
    print()

    # Generate logits for 1 hour
    logits_file = generate_teacher_logits(
        teacher_model,
        teacher_tokenizer,
        student_tokenizer,
        duration_minutes=60
    )

    # Train student on logits
    train_student_on_logits(logits_file, epochs=3)


if __name__ == "__main__":
    main()
