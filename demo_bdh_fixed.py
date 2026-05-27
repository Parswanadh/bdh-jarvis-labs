"""
Interactive BDH Demo - Fixed encoding issues!
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer

def load_model(checkpoint_path="checkpoints/20min_training/final_model.pt"):
    """Load the trained model."""
    print("="*70)
    print("LOADING TRAINED BDH MODEL")
    print("="*70)

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    config = checkpoint['config']
    model = MultiScaleBDH(config).to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")

    print(f"[OK] Model loaded!")
    print(f"   Loss: {checkpoint['loss']:.4f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    return model, tokenizer, checkpoint['loss']


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """Generate text from a prompt."""

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to('cuda')

    # Generate tokens
    generated_ids = input_ids.clone()
    tokens_generated = 0

    with torch.no_grad():
        for i in range(max_tokens):
            logits, _ = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            tokens_generated += 1

            if generated_ids.size(1) >= model.config.max_seq_len:
                break

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text, tokens_generated


def safe_print(text):
    """Print text safely, handling encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Print character by character, replacing problematic ones
        for char in text:
            try:
                print(char, end='', flush=True)
            except:
                print('?', end='', flush=True)
        print()


def interactive_demo(model, tokenizer, loss):
    """Run interactive demo."""

    print("="*70)
    print("INTERACTIVE TEXT GENERATION DEMO")
    print("="*70)
    print()
    print(f"Your BDH model is ready! (Loss: {loss:.4f})")
    print()
    print("Try prompts like:")
    print("  - 'Once upon a time'")
    print("  - 'The little girl'")
    print("  - 'In a forest'")
    print()
    print("Type 'quit' to exit")
    print("="*70)
    print()

    while True:
        try:
            prompt = input("\nYour prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nThanks for trying BDH!\n")
                break

            if not prompt:
                continue

            print()
            print("[GENERATING]...")
            generated, tokens = generate_text(model, tokenizer, prompt, max_tokens=40)

            print()
            print("Full output:")
            safe_print(generated)
            print()
            print(f"(Generated {tokens} tokens)")
            print()

        except KeyboardInterrupt:
            print("\n\nDemo stopped!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main demo function."""
    model, tokenizer, loss = load_model()
    interactive_demo(model, tokenizer, loss)


if __name__ == "__main__":
    main()
