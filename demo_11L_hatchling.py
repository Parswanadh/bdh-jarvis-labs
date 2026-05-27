import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import torch.nn.functional as F

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig

# --- PATHS FOR THE 11-LAYER MODEL ---
MODEL_NAME = "Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("checkpoints/safe/laptop_safe_11L.pt")

def generate_story(model, tokenizer, prompt, max_tokens=100, device='cuda'):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Sliding window for 192 context
            window = generated[:, -192:]
            logits, _ = model(window)
            
            # Use Temperature 0.7 for a balance of logic and creativity
            next_token_logits = logits[:, -1, :] / 0.7
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("BABY DRAGON HATCHLING - 11-LAYER SUPER-HATCHLING DEMO")
    print("="*70)

    print("Waking up the 11-Layer Brain...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # CRITICAL: Architecture must match the new 11-layer RYS hack
    config = MultiScaleBDHConfig(
        vocab_size=248320, 
        n_embd=256, 
        n_layer=11,     # <--- Updated to 11 layers
        n_head=8, 
        ffn_dim=1024, 
        max_seq_len=192
    )
    model = MultiScaleBDH(config).to(device)
    
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Could not find the 11-layer model at {CHECKPOINT_PATH}")
        print("Did the 'rys_surgery.py' or 'train_hatchling_ultimate_hacks.py' script run successfully?")
        return

    # Load the 11-layer weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    step = ckpt.get('step', 'Unknown')
    print(f"\n[SUCCESS] 11-Layer Hatchling is ready! (Trained to Step {step})")
    
    print("\n" + "*"*70)
    print("LIVE INTERACTION MODE")
    print("Type 'exit' to stop.")
    print("*"*70)

    while True:
        try:
            user_prompt = input("\nEnter a story starter >> ")
            if user_prompt.lower() in ['exit', 'quit', 'q']:
                break
            
            if not user_prompt.strip():
                continue
                
            print("\nHatchling is thinking...")
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
