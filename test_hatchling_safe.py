import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch.nn.functional as F

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

def test_hatchling(prompt, max_tokens=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Tokenizer & Model Head Size
    model_name = "Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Matches the 'train_laptop_safe.py' architecture exactly
    config = MultiScaleBDHConfig(
        vocab_size=248320, # Qwen 3.5 actual head size
        n_embd=256,
        n_layer=8,
        n_head=8,
        ffn_dim=1024,
        max_seq_len=192,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.0005
    )

    # 2. Load Model
    model = MultiScaleBDH(config).to(device)
    checkpoint_path = Path("checkpoints/safe/laptop_safe.pt")
    
    if not checkpoint_path.exists():
        print("Error: No checkpoint found in checkpoints/safe/laptop_safe.pt")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"Model loaded (Trained to Step {ckpt['step']})")
    model.eval()

    # 3. Generation Loop
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Sliding window for 192 seq len
            window = generated[:, -192:]
            logits, _ = model(window)
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / 0.7 # Slightly more creative temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Result:\n{response}")
    print("-" * 40)

if __name__ == "__main__":
    prompts = [
        "Once upon a time, there was a little boy named Tim who loved",
        "One day, Lily went to the park and found a",
        "The sun was shining and the birds were"
    ]
    
    for p in prompts:
        test_hatchling(p)
        print("\n")
