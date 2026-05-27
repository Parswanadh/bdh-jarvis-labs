import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch.nn.functional as F

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- PATHS FOR THE FINAL MODEL ---
MODEL_NAME = "Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("checkpoints/safe/laptop_safe.pt")

def generate_story(model, tokenizer, prompt, max_tokens=100, device='cuda'):
    # Encode with Qwen tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Sliding window for 192 context
            window = generated[:, -192:]
            logits, _ = model(window)
            
            # Use Temperature 0.7 for SOTA logic
            next_token_logits = logits[:, -1, :] / 0.7
            
            # Filter out extreme low probability noise
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if the model thinks the story is finished
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("BABY DRAGON HATCHLING - FINAL SOTA DEMO")
    print("="*70)

    # 1. Load Setup (Matching your 40-hour training run)
    print("Loading 8-Layer SOTA Brain...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Architecture MUST match your laptop_safe.pt exactly
    config = MultiScaleBDHConfig(
        vocab_size=248320, 
        n_embd=256, 
        n_layer=8, 
        n_head=8, 
        ffn_dim=1024, 
        max_seq_len=192
    )
    model = MultiScaleBDH(config).to(device)
    
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Could not find final model at {CHECKPOINT_PATH}")
        return

    # Load the Step 7109 weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"[SUCCESS] Hatchling woke up at Step {ckpt['step']}!")
    print(f"Architecture: 8 Layers | 256 Embedding | {sum(p.numel() for p in model.parameters())/1e6:.1f}M Params")

    # 2. Interactive Showcase
    print("\n" + "*"*70)
    print("LIVE INTERACTION MODE")
    print("The model will now generate stories based on your input.")
    print("Type 'exit' to stop.")
    print("*"*70)

    while True:
        try:
            user_prompt = input("\nEnter a story starter >> ")
            if user_prompt.lower() in ['exit', 'quit', 'q']:
                break
            
            if not user_prompt.strip():
                continue
                
            print("\nHatchling is writing...")
            result = generate_story(model, tokenizer, user_prompt, 100, device)
            
            print("\n--- GENERATED STORY ---")
            print(result)
            print("-----------------------")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
