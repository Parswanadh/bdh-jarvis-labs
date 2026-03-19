"""
CONTINUE TRAINING FROM YOUR CHECKPOINT
=======================================
Loads your previous model (loss ~2.45) and continues improving it
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

print("="*80)
print("CONTINUE TRAINING - FROM YOUR CHECKPOINT")
print("="*80)

device = torch.device('cuda')
print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# LOAD YOUR CHECKPOINT
# ============================================================================
checkpoint_dir = Path("checkpoints/distillation_tinyllama")
checkpoint_file = checkpoint_dir / "final_model.pt"

if not checkpoint_file.exists():
    print(f"\nERROR: Checkpoint not found!")
    print(f"Looking for: {checkpoint_file}")
    print("\nAvailable files:")
    for f in checkpoint_dir.glob("*.pt"):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.0f} MB")
    sys.exit(1)

print(f"\n[LOAD] Loading your checkpoint...")
print(f"  File: {checkpoint_file}")

checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

# Get model state
if 'model_state_dict' in checkpoint:
    model_state = checkpoint['model_state_dict']
    prev_loss = checkpoint.get('loss', 2.45)
elif 'state_dict' in checkpoint:
    model_state = checkpoint['state_dict']
    prev_loss = checkpoint.get('loss', 2.45)
else:
    model_state = checkpoint
    prev_loss = 2.45

print(f"  Previous loss: {prev_loss:.4f}")
print(f"  State dict keys: {len(model_state)}")

# ============================================================================
# CREATE MODEL WITH SAME CONFIG
# ============================================================================

# Infer vocab size from embedding
if 'embed_tokens.weight' in model_state:
    vocab_size = model_state['embed_tokens.weight'].shape[0]
elif 'token_embedding.weight' in model_state:
    vocab_size = model_state['token_embedding.weight'].shape[0]
else:
    vocab_size = 32000

# Infer embedding dim
if 'embed_tokens.weight' in model_state:
    n_embd = model_state['embed_tokens.weight'].shape[1]
elif 'token_embedding.weight' in model_state:
    n_embd = model_state['token_embedding.weight'].shape[1]
else:
    n_embd = 768

# Find number of layers
layer_keys = [k for k in model_state.keys() if 'layers.' in k]
if layer_keys:
    n_layer = max(int(k.split('.')[1]) for k in layer_keys if k.count('.') >= 1) + 1
else:
    n_layer = 6

print(f"\n[MODEL] Inferred config:")
print(f"  vocab_size: {vocab_size:,}")
print(f"  n_embd: {n_embd}")
print(f"  n_layer: {n_layer}")

bdh_config = MultiScaleBDHConfig(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=min(12, n_embd // 64),
    ffn_dim=n_embd * 4,
    dropout=0.1,
    max_seq_len=512,
    decay_rates=[0.95, 0.99, 0.995],
    hebbian_lr=0.001
)

student_model = MultiScaleBDH(bdh_config).to(device)

# Load state (allow partial load)
load_result = student_model.load_state_dict(model_state, strict=False)
print(f"\n[LOAD] Loaded checkpoint:")
print(f"  Missing keys: {len(load_result.missing_keys)}")
print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")

student_model = student_model.train()
student_params = sum(p.numel() for p in student_model.parameters())
print(f"  Parameters: {student_params:,}")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# LOAD TEACHER (TinyLlama - same as before)
# ============================================================================
print(f"\n[TEACHER] Loading TinyLlama 1.1B...")

teacher_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="cuda:0"
).eval()

teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"  OK - {teacher_params:,} parameters")
print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n[DATA] Loading TinyStories...")

data_file = Path("data/tinystories.txt")
with open(data_file, 'r', encoding='utf-8') as f:
    all_stories = [line.strip() for line in f if line.strip()]

print(f"  Total: {len(all_stories):,} stories")
stories = all_stories[:200000]
print(f"  Using: {len(stories):,} stories")

def collate_batch(batch_texts):
    encodings = teacher_tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    return encodings

dataloader = DataLoader(
    stories,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_batch,
    drop_last=True
)

print(f"  Batches: {len(dataloader):,}")

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Starting from loss: {prev_loss:.4f}")
print(f"Target: < 2.0")
print()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

session_start = time.time()
step = 0
checkpoint_freq = 1800

student_model.train()
teacher_model.eval()

try:
    for epoch in range(10):
        print(f"\n--- EPOCH {epoch + 1} ---")
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if time.time() - session_start >= 7200:
                print("\n[SESSION] 2 hours complete!")
                raise StopIteration

            step += 1

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids)
                teacher_logits = teacher_outputs.logits.float()

            student_logits, _ = student_model(input_ids)

            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            shift_labels[shift_mask == 0] = -100

            B, T, V = shift_logits.shape
            shift_logits = shift_logits.view(-1, V)
            shift_labels = shift_labels.view(-1)

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / batches
                elapsed = time.time() - session_start
                vram = torch.cuda.memory_allocated(0) / 1e9

                print(f"  Step {step:4d} | Loss: {avg_loss:.4f} | VRAM: {vram:.2f}GB | {elapsed/60:.0f}m")

                if avg_loss < 2.3:
                    print(f"       ↓ Improving from {prev_loss:.4f}!")
                if avg_loss < 2.0:
                    print(f"       ★ TARGET REACHED!")

            if time.time() - session_start >= checkpoint_freq * (step / 50 + 1):
                Path("checkpoints/continued").mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss
                }, 'checkpoints/continued/latest.pt')
                print(f"\n[CHECKPOINT] Saved")

        avg_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

except StopIteration:
    pass

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Starting: {prev_loss:.4f}")
print(f"Final: {avg_loss:.4f}")
print(f"Improvement: {prev_loss - avg_loss:.4f}")
