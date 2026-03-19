"""
Generate text with trained BDH model from TRUE distillation.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"[LOAD] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')

    if config is None:
        # Use default config if not saved
        config = MultiScaleBDHConfig(
            vocab_size=256,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024,
            dropout=0.1,
            max_seq_len=256,
            decay_rates=[0.95, 0.99, 0.995],
            hebbian_lr=0.001
        )

    # Create model
    model = MultiScaleBDH(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded!")
    print(f"    Task: {checkpoint.get('task', 'unknown')}")
    print(f"    Loss: {checkpoint.get('loss', 'unknown')}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def generate_text(model, prompt_text, max_tokens=100, temperature=0.8, device='cuda'):
    """Generate text using the trained model."""

    # Convert prompt to bytes
    prompt_bytes = prompt_text.encode('utf-8')
    input_tokens = torch.tensor([list(prompt_bytes)], dtype=torch.long).to(device)

    print(f"\n[GENERATE] Prompt: '{prompt_text}'")
    print(f"[GENERATE] Max tokens: {max_tokens}, Temperature: {temperature}")
    print()
    print("[OUTPUT]")

    # Generate tokens
    generated = prompt_text

    with torch.no_grad():
        for i in range(max_tokens):
            # Forward pass
            logits, _ = model(input_tokens)

            # Get next token (sample from distribution)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Convert token to byte and append
            next_byte = next_token.item()
            try:
                next_char = bytes([next_byte]).decode('utf-8')
                generated += next_char
                print(next_char, end='', flush=True)
            except:
                # Invalid UTF-8, stop generation
                break

            # Update input tokens
            # next_token is scalar, reshape to [1, 1] for concatenation
            next_token_expanded = next_token.view(1, 1)  # Shape: [batch_size=1, seq_len=1]
            input_tokens = torch.cat([input_tokens, next_token_expanded], dim=1)

            # Stop if we hit max sequence length
            if input_tokens.size(1) >= model.config.max_seq_len:
                break

    print()
    print()
    return generated


def main():
    """Main generation function."""

    print("="*70)
    print("TEXT GENERATION WITH TRAINED BDH MODEL")
    print("From TRUE Knowledge Distillation (TinyLlama -> BDH)")
    print("="*70)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    print()

    # Load model
    checkpoint_path = "checkpoints/true_distillation/best.pt"
    model, config = load_model(checkpoint_path, device)
    print()

    # Test prompts
    prompts = [
        "The quick brown fox",
        "Machine learning is",
        "Once upon a time",
        "The cat sat on",
        "Artificial intelligence can",
    ]

    print("="*70)
    print("GENERATING TEXT SAMPLES")
    print("="*70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"Sample {i}/{len(prompts)}")
        print(f"{'='*70}")

        output = generate_text(
            model,
            prompt,
            max_tokens=50,
            temperature=0.8,
            device=device
        )

        print(f"\n[FULL OUTPUT]")
        print(f"  {output}")

    print()
    print("="*70)
    print("GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
