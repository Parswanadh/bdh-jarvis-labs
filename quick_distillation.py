"""
Quick working distillation - Simplified and tested
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import time

sys.path.insert(0, "implementation")

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading models...")
device = torch.device('cuda')

# Load teacher (TinyLlama)
teacher = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
).eval()
print("Teacher loaded")

# Load student tokenizer
tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")

# Create student
config = MultiScaleBDHConfig(
    vocab_size=8192,
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
print("Student created")

# Load some stories
print("Loading data...")
stories = []
with open("data/tinystories.txt", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        stories.append(line.strip())

print(f"Loaded {len(stories)} stories")
print()

# Train for 5 minutes
print("Training for 5 minutes with teacher logits...")
start = time.time()
target = 5 * 60

batch_size = 32
student.train()

story_idx = 0
loss_sum = 0
batch_count = 0

while time.time() - start < target:
    # Get batch
    batch_stories = stories[story_idx:story_idx + batch_size]
    story_idx += batch_size
    
    # Tokenize
    input_ids = []
    for story in batch_stories:
        ids = tokenizer.encode(story)[:512]
        if len(ids) < 512:
            ids = ids + [0] * (512 - len(ids))
        input_ids.append(ids)
    
    input_ids = torch.tensor(input_ids).to(device)
    
    # Teacher logits
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits
    
    # Student forward + loss
    student_logits, _ = student(input_ids)
    
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
        print(f"[Batch {batch_count}] loss: {loss_sum/batch_count:.4f} | time: {elapsed:.0f}s")

# Save
Path("checkpoints/quick_distillation").mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': student.state_dict(),
    'config': config,
    'loss': loss_sum / batch_count
}, "checkpoints/quick_distillation/model.pt")

print()
print(f"Done! Loss: {loss_sum/batch_count:.4f}")
print(f"Saved to checkpoints/quick_distillation/model.pt")
