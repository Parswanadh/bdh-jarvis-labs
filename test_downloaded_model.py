"""
Test Downloaded Trained Model
================================

Quick script to test a model downloaded from GitHub/Jarvis Labs.
"""

import torch
import sys
from pathlib import Path

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


def load_checkpoint(checkpoint_path: str):
    """Load a checkpoint and display info."""

    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    print("="*70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Display info
    print("\nCHECKPOINT INFO:")
    print(f"  Iteration (Epoch): {checkpoint.get('iteration', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    print(f"  Task: {checkpoint.get('task', 'N/A')}")
    print(f"  Is Best: {checkpoint.get('is_best', False)}")

    # Get config
    config = checkpoint.get('config')
    if config:
        print("\nMODEL CONFIG:")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Heads: {config.n_head}")
        print(f"  Max seq len: {config.max_seq_len}")
        print(f"  Decay rates: {config.decay_rates}")

    # Load model
    print("\n[LOAD] Loading model weights...")
    model = MultiScaleBDH(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded: {num_params:,} parameters")

    return model, config


def test_generation(model, config, device='cpu'):
    """Test text generation."""

    print("\n" + "="*70)
    print("TESTING TEXT GENERATION")
    print("="*70)

    # Move to device
    model = model.to(device)

    # Test prompts
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "In a galaxy far",
    ]

    for prompt in prompts:
        print(f"\n[PROMPT] {prompt}")

        # Encode prompt
        tokens = torch.tensor(
            list(prompt.encode('utf-8')),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                tokens,
                max_new_tokens=50,
                temperature=0.8,
                top_k=10
            )

        # Decode
        text = bytes(generated.cpu().squeeze().tolist()).decode('utf-8', errors='ignore')
        print(f"[OUTPUT] {text}")

    print("\n" + "="*70)


def main():
    """Main function."""

    print("="*70)
    print("DOWNLOADED MODEL TEST")
    print("="*70)
    print()

    # Check checkpoint path
    checkpoint_path = "checkpoints/bdh_a100/checkpoint_best.pt"

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nPlease download the model first:")
        print("  ./download_from_github.sh")
        print("\nOr specify checkpoint path:")
        print("  python test_downloaded_model.py checkpoints/bdh_a100/checkpoint_iter_5.pt")
        sys.exit(1)

    # Allow custom path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    # Load checkpoint
    model, config = load_checkpoint(checkpoint_path)

    # Test generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_generation(model, config, device)

    print("\n[DONE] Test complete!")
    print()


if __name__ == "__main__":
    main()
