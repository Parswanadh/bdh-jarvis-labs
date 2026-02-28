"""
BDH Reasoning Model - Demo Script
===================================

This script demonstrates the trained BDH model solving reasoning problems.
Run this after training completes to show off your model!
"""

import torch
import sys
from pathlib import Path

# Add implementation to path
sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


def load_trained_model(checkpoint_path="checkpoints/multiscale_bdh_reasoning/checkpoint_best.pt"):
    """Load the trained model."""
    print("="*60)
    print("🧠 BDH REASONING MODEL DEMO")
    print("="*60)
    print()

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Please train the model first: python train_bdh_reasoning.py")
        return None, None

    # Create model (must match training config exactly!)
    config = MultiScaleBDHConfig(
        vocab_size=256,
        n_embd=256,
        n_layer=6,
        n_head=4,
        ffn_dim=1024,
        dropout=0.0,  # No dropout for inference
        max_seq_len=512,  # Must match training!
        decay_rates=[0.95, 0.99, 0.995],
        hebbian_lr=0.001
    )

    model = MultiScaleBDH(config)

    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load training info
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', 0)
    timestamp = checkpoint.get('timestamp', 'Unknown')

    print(f"✅ Model loaded successfully!")
    print(f"   Iteration: {iteration}")
    print(f"   Loss: {loss:.4f}")
    print(f"   Trained: {timestamp}")
    print()

    return model, config


def demo_reasoning(model, config):
    """Demonstrate reasoning capabilities."""
    model.eval()

    # Test problems
    test_problems = [
        # Math problems
        "Q: What is 15 × 14? A:",
        "Q: What is 144 ÷ 12? A:",
        "Q: What is 2⁸? A:",
        "Q: What is √625? A:",

        # Logic problems
        "Q: If A > B and B > C, who is shortest? A:",
        "Q: All cats are mammals. Fluffy is a cat. Is Fluffy a mammal? A:",

        # Patterns
        "Q: What comes next: 2, 4, 8, 16, ? A:",
        "Q: What comes next: A, C, E, G, ? A:",

        # AI reasoning
        "Q: Why does BDH use Hebbian learning? A:",
        "Q: What is the advantage of multi-scale memory? A:",
    ]

    print("🎯 REASONING DEMO")
    print("="*60)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for i, problem in enumerate(test_problems, 1):
        print(f"[Problem {i}] {problem}")

        # Encode problem (simple byte encoding for demo)
        input_text = problem
        input_ids = torch.tensor(list(input_text.encode('utf-8')), dtype=torch.long).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits, state = model(input_ids.to(device))

        # Get predictions (top 10)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = torch.topk(probs, 10)

        # Decode top predictions (just show first token for demo)
        print(f"   Top 5 predictions:")
        for j, (prob, token_id) in enumerate(zip(top_probs[:5], top_ids[:5])):
            token_char = chr(token_id.item()) if token_id.item() < 256 else f"[{token_id.item()}]"
            print(f"      {j+1}. '{token_char}' (probability: {prob:.4f})")

        print()


def show_model_info(model, config):
    """Show model information."""
    num_params = sum(p.numel() for p in model.parameters())

    print("📊 MODEL INFORMATION")
    print("="*60)
    print(f"   Architecture: Multi-Scale BDH")
    print(f"   Parameters: {num_params:,}")
    print(f"   Layers: {config.n_layer}")
    print(f"   Embedding dim: {config.n_embd}")
    print(f"   Attention heads: {config.n_head}")
    print(f"   Decay rates: {config.decay_rates}")
    print(f"   Context length: {config.max_seq_len} tokens")
    print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    print()


def interactive_demo(model, config):
    """Interactive demo where user can input problems."""
    print("🎮 INTERACTIVE MODE")
    print("="*60)
    print("Type your reasoning problems (or 'quit' to exit)")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    while True:
        try:
            user_input = input("Your question (or 'quit'): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Prepare input
            problem = f"Q: {user_input} A:"
            input_ids = torch.tensor(list(problem.encode('utf-8')), dtype=torch.long).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                logits, _ = model(input_ids.to(device))

            # Get predictions
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_probs, top_ids = torch.topk(probs, 5)

            print(f"\n🤔 Model's top 5 predictions:")

            for j, (prob, token_id) in enumerate(zip(top_probs, top_ids)):
                token_char = chr(token_id.item()) if token_id.item() < 256 else f"[{token_id.item()}]"
                print(f"   {j+1}. '{token_char}' (confidence: {prob:.4f})")

            print()

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            print()


def main():
    """Main demo function."""
    # Load model
    model, config = load_trained_model()

    if model is None:
        print("\n❌ Cannot run demo - model not trained yet")
        print("   Please run: python train_bdh_reasoning.py")
        return

    # Show model info
    show_model_info(model, config)

    # Run demo
    demo_reasoning(model, config)

    # Interactive mode
    print()
    choice = input("Do you want to try interactive mode? (y/n): ").strip().lower()

    if choice in ['y', 'yes']:
        interactive_demo(model, config)

    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE!")
    print("="*60)
    print()
    print("✅ Your BDH model is ready for the science fair!")
    print()


if __name__ == "__main__":
    main()
