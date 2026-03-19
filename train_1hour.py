"""
1-Hour BDH Training - Better Quality Results
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer
from functools import partial

class TinyStoriesDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_samples=500000, max_seq_len=512):
        print(f"[LOAD] Loading {max_samples} stories...")
        stories = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                stories.append(line.strip())
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
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(token_ids), dtype=torch.long)
        }

def collate_fn(batch, pad_token_id=0):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
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

def main():
    print("="*70)
    print("1-HOUR BDH TRAINING - 500K Stories")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")
    print(f"[OK] Tokenizer vocab: {tokenizer.vocab_size_actual}")
    
    config = MultiScaleBDHConfig(
        vocab_size=tokenizer.vocab_size_actual,
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
    print(f"[OK] Model: {num_params:,} parameters")
    
    dataset = TinyStoriesDataset("data/tinystories.txt", tokenizer, max_samples=500000)
    
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, pad_token_id=0)
    )
    
    print(f"[OK] Batches: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print()
    print("[START] Training for 1 hour")
    print("="*70)
    
    model.train()
    total_loss = 0
    start = time.time()
    target_time = 60 * 60  # 1 hour in seconds
    checkpoint_interval = 60 * 10  # Save checkpoint every 10 minutes
    
    last_checkpoint_time = start
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        
        with torch.cuda.amp.autocast():
            logits, _ = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start
            avg_loss = total_loss / (batch_idx + 1)
            remaining = max(0, target_time - elapsed)
            progress = ((batch_idx + 1) / len(train_loader)) * 100
            print(f"[{batch_idx+1:5d}/{len(train_loader)}] loss: {avg_loss:.4f} | elapsed: {elapsed/60:5.1f}min | remaining: {remaining/60:5.1f}min | progress: {progress:5.1f}%")
        
        # Save checkpoint every 10 minutes
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval:
            Path("checkpoints/1hour_training").mkdir(parents=True, exist_ok=True)
            avg_loss = total_loss / (batch_idx + 1)
            elapsed_min = int((current_time - start) / 60)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'batch': batch_idx + 1,
                'elapsed_minutes': elapsed_min
            }, f"checkpoints/1hour_training/checkpoint_{elapsed_min}min.pt")
            print(f"\n[CHECKPOINT] Saved at {elapsed_min} minutes, loss: {avg_loss:.4f}\n")
            last_checkpoint_time = current_time
        
        # Check if we've hit 1 hour
        elapsed = time.time() - start
        if elapsed >= target_time:
            print(f"\n[TIME] Reached 1 hour! Stopping training.")
            break
    
    final_loss = total_loss / (batch_idx + 1)
    Path("checkpoints/1hour_training").mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': final_loss,
        'config': config
    }, "checkpoints/1hour_training/final_model.pt")
    
    print()
    print("="*70)
    print("[DONE] Training Complete!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Time: {(time.time() - start)/60:.1f} minutes")
    print(f"Batches completed: {batch_idx + 1} / {len(train_loader)}")
    print(f"Saved: checkpoints/1hour_training/final_model.pt")
    print("="*70)

if __name__ == "__main__":
    main()
