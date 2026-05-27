"""
Working Distillation - Fixed vocab mismatch!
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import time

sys.path.insert(0, "implementation")

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading...")
device = torch.device('cuda')

# Load teacher
teacher_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
teacher = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
).eval()
print(f"Teacher loaded - vocab: {len(teacher_tokenizer)}")

# Create student with SAME vocab size
vocab_size = len(teacher_tokenizer)
config = MultiScaleBDHConfig(
    vocab_size=vocab_size,
    n_embd=512,
    n_layer=8,
    n_head=8,
    ffn_dim=2048,
    dropout=0.1,
    max_seq_len=512,
    decay_rates=[0.95, 0.99, 0.995]
)

student = MultiScaleBDH(config).to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
print(f"Student created - {sum(p.numel() for p in student.parameters()):,} params")

# Load data
print("Loading data...")
stories = []
with open("data/tinystories.txt", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        stories.append(line.strip())
print(f"Loaded {len(stories)} stories\n")

# Train for 5 minutes
print("Training with teacher logits (5 minutes)...")
start = time.time()
target = 5 * 60

batch_size = 16  # Smaller batch for speed
student.train()
loss_sum = 0
batch_count = 0
story_idx = 0

while time.time() - start < target and story_idx + batch_size < len(stories):
    # Get batch
    batch_stories = stories[story_idx:story_idx + batch_size]
    story_idx += batch_size
    
    # Tokenize with teacher tokenizer
    inputs = teacher_tokenizer(
        batch_stories,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(device)
    
    # Teacher logits
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits
    
    # Student forward
    student_logits, _ = student(input_ids)
    
    # KL divergence loss
    kl_div = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    
    loss = kl_div
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
    optimizer.step()
    
    loss_sum += loss.item()
    batch_count += 1
    
    if batch_count % 10 == 0:
        elapsed = time.time() - start
        print(f"[{batch_count:3d} batches] loss: {loss_sum/batch_count:.4f} | elapsed: {elapsed:.0f}s | remaining: {target-elapsed:.0f}s")

# Save
final_loss = loss_sum / batch_count
Path("checkpoints/working_distillation").mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': student.state_dict(),
    'config': config,
    'loss': final_loss,
    'vocab_size': vocab_size
}, "checkpoints/working_distillation/model.pt")

print(f"\nDone! Final loss: {final_loss:.4f}")
print(f"Saved: checkpoints/working_distillation/model.pt")
print(f"Parameters: {sum(p.numel() for p in student.parameters()):,}")
