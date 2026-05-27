"""
Test the 1-hour trained model
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer

def load_model(checkpoint_path="checkpoints/1hour_training/final_model.pt"):
    print("="*70)
    print("LOADING 1-HOUR TRAINED MODEL")
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
    # Encode
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to('cuda')
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

    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


def safe_print(text):
    try:
        print(text)
    except:
        for char in text:
            try:
                print(char, end='', flush=True)
            except:
                print('?', end='', flush=True)
        print()


def main():
    model, tokenizer, loss = load_model()

    print("="*70)
    print("TESTING 1-HOUR MODEL")
    print("="*70)
    print()

    prompts = [
        "Once upon a time",
        "The little girl",
        "One sunny day",
        "In a forest",
        "The cat sat"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Example {i}:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: ", end="")

        generated = generate_text(model, tokenizer, prompt, max_tokens=30, temperature=0.8)
        new_text = generated[len(prompt):]
        safe_print(new_text)
        print()
        print()


if __name__ == "__main__":
    main()
