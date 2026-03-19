"""
Quick BDH Training - Small Subset for Fast Results
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
from bbpe_tokenizer import BBPETokenizer
from functools import partial

class TinyStoriesDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_samples=50000, max_seq_len=512):
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
    print("QUICK BDH TRAINING - 50K Stories")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    # Load tokenizer
    tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")
    print(f"[OK] Tokenizer vocab: {tokenizer.vocab_size_actual}")
    
    # Create model
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
    
    # Load small dataset
    dataset = TinyStoriesDataset("data/tinystories.txt", tokenizer, max_samples=50000)
    
    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, pad_token_id=0)
    )
    
    print(f"[OK] Batches: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print()
    print("[START] Training for 1 epoch (~20 minutes)")
    print("="*70)
    
    model.train()
    total_loss = 0
    start = time.time()
    
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
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            avg_loss = total_loss / (batch_idx + 1)
            print(f"[{batch_idx+1}/{len(train_loader)}] loss: {avg_loss:.4f} | time: {elapsed/60:.1f}min")
        
        # Save checkpoint mid-training
        if (batch_idx + 1) % 200 == 0:
            Path("checkpoints/quick_training").mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / (batch_idx + 1),
                'batch': batch_idx + 1
            }, f"checkpoints/quick_training/checkpoint_batch_{batch_idx+1}.pt")
            print(f"[CHECKPOINT] Saved at batch {batch_idx+1}")
    
    # Final save
    final_loss = total_loss / len(train_loader)
    Path("checkpoints/quick_training").mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': final_loss,
        'config': config
    }, "checkpoints/quick_training/final_model.pt")
    
    print()
    print("="*70)
    print("[DONE] Training Complete!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Time: {(time.time() - start)/60:.1f} minutes")
    print(f"Saved: checkpoints/quick_training/final_model.pt")
    print("="*70)

if __name__ == "__main__":
    main()
