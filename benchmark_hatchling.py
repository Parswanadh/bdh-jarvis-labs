import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch.nn.functional as F
import math
import time

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

def calculate_perplexity(model, tokenizer, test_stories, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print(f"Calculating Perplexity on {len(test_stories)} test stories...")
    
    with torch.no_grad():
        for story in test_stories:
            enc = tokenizer(story, return_tensors='pt', truncation=True, max_length=192).to(device)
            input_ids = enc['input_ids']
            
            if input_ids.size(1) <= 1: continue
            
            logits, _ = model(input_ids)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss

def evaluate_logic(model, tokenizer, device):
    print("Running Logic Tests...")
    # Test 1: Character Persistence
    prompt = "Timmy has a blue ball. He gave the ball to Sarah. Who has the ball now? "
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, temperature=0.2)
    
    result = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    score = 0
    if "sarah" in result: score += 10
    elif "timmy" in result: score += 2 # Half credit for remembering a character
    
    # Test 2: Basic Object Logic
    prompt = "The sun is very hot. If you touch it, you will feel "
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, temperature=0.2)
    result = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    if "hot" in result or "pain" in result or "hurt" in result: score += 10
    
    return score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("HATCHLING EXCELLENCE BENCHMARK")
    print("="*60)

    # 1. Load Setup
    model_name = "Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = MultiScaleBDHConfig(
        vocab_size=248320, n_embd=256, n_layer=8, n_head=8, 
        ffn_dim=1024, max_seq_len=192
    )
    model = MultiScaleBDH(config).to(device)
    
    checkpoint_path = Path("checkpoints/safe/laptop_safe.pt")
    if not checkpoint_path.exists():
        print("Error: No checkpoint found!")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    step = ckpt.get('step', 0)
    print(f"Benchmarking Model at Step {step}...\n")

    # 2. Perplexity Test (Data Science Score)
    with open("data/tinystories.txt", 'r', encoding='utf-8') as f:
        # Use stories from the end of the file (unseen)
        all_lines = f.readlines()
        test_stories = [l.strip() for l in all_lines[-100:] if l.strip()]

    ppl, loss = calculate_perplexity(model, tokenizer, test_stories, device)
    
    # 3. Logic Test (Intelligence Score)
    logic_points = evaluate_logic(model, tokenizer, device)

    # 4. SCORING
    # Accuracy Score: 50 points max (Loss 1.5 = 50, Loss 4.0 = 0)
    acc_score = max(0, min(50, 50 * (4.0 - loss) / (4.0 - 1.5)))
    
    # Logic Score: 20 points max
    
    # Fluency Score (calculated from Loss stability)
    fluency_score = max(0, min(30, 30 * (100 / ppl)))

    total_score = acc_score + logic_points + fluency_score

    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"Model Step:      {step}")
    print(f"Avg Loss:        {loss:.4f}")
    print(f"Perplexity:      {ppl:.2f}")
    print("-" * 30)
    print(f"Accuracy Score:  {acc_score:.1f}/50")
    print(f"Logic Score:     {logic_points:.1f}/20")
    print(f"Fluency Score:   {fluency_score:.1f}/30")
    print("-" * 30)
    print(f"EXCELLENCE SCORE: {total_score:.1f}/100")
    print("="*60)
    
    if total_score > 80:
        print("RANK: SOTA HATCHLING (Ready for the World!)")
    elif total_score > 60:
        print("RANK: COHERENT JUVENILE (Very Good progress)")
    elif total_score > 40:
        print("RANK: GROWING DRAGON (Learning well)")
    else:
        print("RANK: INFANT (Needs more training)")

if __name__ == "__main__":
    main()
