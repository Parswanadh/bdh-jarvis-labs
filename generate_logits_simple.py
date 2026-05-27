"""
Simplified Teacher Logit Generation - No multiprocessing issues
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("="*70)
    print("TEACHER LOGIT GENERATION + STUDENT TRAINING")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Load teacher
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

    # Load student tokenizer
    student_tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")
    print(f"[OK] Student tokenizer vocab: {student_tokenizer.vocab_size_actual}")
    print()

    # Load data
    print("[LOAD] Loading TinyStories (first 100K for speed)...")
    stories = []
    with open("data/tinystories.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100000:
                break
            stories.append(line.strip())
    print(f"[OK] Loaded {len(stories)} stories")
    print()

    # Generate logits for 10 minutes (faster than 1 hour for demo)
    print("="*70)
    print("GENERATING TEACHER LOGITS (10 minutes)")
    print("="*70)

    batch_size = 32
    all_logits = []
    all_input_ids = []

    start = time.time()
    target_time = 10 * 60  # 10 minutes
    batch_count = 0

    teacher_model.eval()
    story_idx = 0

    with torch.no_grad():
        while time.time() - start < target_time and story_idx < len(stories):
            # Collect batch
            batch_input_ids = []
            for _ in range(batch_size):
                if story_idx >= len(stories):
                    break

                story = stories[story_idx]
                token_ids = student_tokenizer.encode(story)
                if len(token_ids) > 512:
                    token_ids = token_ids[:512]

                # Pad to same length
                if len(token_ids) < 512:
                    token_ids = token_ids + [0] * (512 - len(token_ids))

                batch_input_ids.append(token_ids)
                story_idx += 1

            if len(batch_input_ids) < batch_size:
                # Skip incomplete batch
                break

            # Convert to tensor and forward
            input_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(device)

            outputs = teacher_model(input_tensor)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            all_logits.append(logits.cpu().numpy())
            all_input_ids.append(input_tensor.cpu().numpy())

            batch_count += 1

            if batch_count % 10 == 0:
                elapsed = time.time() - start
                remaining = max(0, target_time - elapsed)
                print(f"[Batch {batch_count}] Stories processed: {story_idx} | elapsed: {elapsed/60:.1f}min | remaining: {remaining/60:.1f}min")

    print()
    print("[SAVE] Saving logits...")

    all_logits = np.concatenate(all_logits, axis=0)
    all_input_ids = np.concatenate(all_input_ids, axis=0)

    # Save as numpy
    np.save("data/teacher_input_ids.npy", all_input_ids)
    np.save("data/teacher_logits.npy", all_logits)

    print(f"[OK] Saved!")
    print(f"   Shape: {all_logits.shape}")
    print(f"   Size: {(all_logits.nbytes + all_input_ids.nbytes) / 1e9:.2f} GB")
    print()

    # Now train student
    print("="*70)
    print("TRAINING STUDENT ON LOGITS")
    print("="*70)

    config = MultiScaleBDHConfig(
        vocab_size=all_logits.shape[-1],
        n_embd=512,
        n_layer=8,
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
        max_seq_len=512,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    model = MultiScaleBDH(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Student model: {num_params:,} parameters")
    print()

    # Simple dataset
    class LogitsDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, logits):
            self.input_ids = torch.tensor(input_ids, dtype=torch.long)
            self.logits = torch.tensor(logits, dtype=torch.float32)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.logits[idx]

    dataset = LogitsDataset(all_input_ids, all_logits)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    print("[START] Training student for 3 epochs...")
    print("="*70)

    for epoch in range(3):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        for batch_idx, (batch_input_ids, batch_logits) in enumerate(dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_logits = batch_logits.to(device)

            with torch.cuda.amp.autocast():
                student_logits, _ = model(batch_input_ids)

                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(batch_logits, dim=-1)

                kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none')

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
                elapsed = time.time() - epoch_start
                avg_loss = total_loss / (batch_idx + 1)
                print(f"[Epoch {epoch+1}] [{batch_idx+1}/{len(dataloader)}] loss: {avg_loss:.4f} | time: {elapsed/60:.1f}min")

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        print()
        print(f"[EPOCH {epoch+1}] Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f}min")
        print()

        Path("checkpoints/student_from_logits").mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'loss': avg_loss
        }, f"checkpoints/student_from_logits/epoch_{epoch+1}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, "checkpoints/student_from_logits/final_model.pt")

    print("="*70)
    print("[DONE] Training Complete!")
    print(f"Saved: checkpoints/student_from_logits/final_model.pt")
    print("="*70)


if __name__ == "__main__":
    main()
