import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch.nn.functional as F

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

def test_model(prompt, max_tokens=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Tokenizer
    model_name = "Qwen3.5-0.8B" if Path("Qwen3.5-0.8B").exists() else "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 2. Setup Model Config
    config = MultiScaleBDHConfig(
        vocab_size=len(tokenizer),
        n_embd=192,
        n_layer=3,
        n_head=4,
        ffn_dim=768,
        dropout=0.0,
        max_seq_len=80,
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    # 3. Load Model
    model = MultiScaleBDH(config).to(device)
    checkpoint_path = Path("checkpoints/final/latest.pt")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 4. Custom Generation with Sliding Window
    print(f"\nPrompt: {prompt}")
    print("-" * 30)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # SLIDING WINDOW: Only take the last 80 tokens
            window = generated[:, -80:]
            
            # Get logits
            logits, _ = model(window)
            
            # Focus on the last token produced
            next_token_logits = logits[:, -1, :] / 0.8 # Temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Result:\n{response}")
    print("-" * 30)

if __name__ == "__main__":
    prompts = [
        "Once upon a time, there was a little boy named Tim who loved",
        "Lily was a very happy girl. One day, she found a",
        "The big cat sat on the",
    ]
    
    for p in prompts:
        test_model(p)
        print("\n")
