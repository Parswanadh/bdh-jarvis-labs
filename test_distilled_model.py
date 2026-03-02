"""
Test the Distilled BDH Model
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from transformers import AutoTokenizer

def load_model(checkpoint_path="checkpoints/distillation_tinyllama/final_model.pt"):
    """Load the distilled model"""
    print("="*70)
    print("LOADING DISTILLED BDH MODEL")
    print("="*70)
    print()

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    config = checkpoint['config']
    model = MultiScaleBDH(config).to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load TinyLlama tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[OK] Model loaded!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {config.vocab_size:,}")
    print(f"   Trained for: {checkpoint.get('elapsed_hours', 6):.2f} hours")
    print()

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Generate text from prompt"""
    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    # Truncate if too long
    if input_ids.size(1) > 200:
        input_ids = input_ids[:, -200:]

    # Generate
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for i in range(max_tokens):
            logits, _ = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            if generated_ids.size(1) >= model.config.max_seq_len:
                break

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    """Test the distilled model"""
    model, tokenizer = load_model()

    print("="*70)
    print("TESTING TEXT GENERATION")
    print("="*70)
    print()

    # Test prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "One sunny day",
        "In a forest",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"[Test {i}] Prompt: '{prompt}'")
        print("-" * 70)

        try:
            generated = generate_text(model, tokenizer, prompt, max_tokens=40, temperature=0.8)

            # Show only the new part
            if len(generated) > len(prompt):
                new_text = generated[len(prompt):]
                print(f"Generated: {new_text}")
            else:
                print(f"Generated: {generated}")

        except Exception as e:
            print(f"Error: {e}")

        print()
        print()

    print("="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
