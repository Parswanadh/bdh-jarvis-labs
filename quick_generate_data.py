"""
Fast Ollama Data Generator
===========================

Simpler, faster version using ollama python library.
"""

import json
from pathlib import Path
from datetime import datetime

# Try importing ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("❌ ollama library not found. Install with: pip install ollama")

# Configuration
MODEL = "qwen3-4b-fast"
NUM_SAMPLES_PER_PROMPT = 10  # Reduced for faster testing
OUTPUT_FILE = Path("data/ollama_generated/qwen_training_data.jsonl")


def main():
    """Generate training data using Ollama."""

    if not OLLAMA_AVAILABLE:
        print("Please install ollama: pip install ollama")
        return

    print("="*60)
    print("🚀 FAST OLLAMA DATA GENERATION")
    print(f"Model: {MODEL}")
    print("="*60)
    print()

    # Check if model is available
    print(f"📚 Checking models...")
    try:
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]

        if MODEL in model_names:
            print(f"✅ Found {MODEL}")
        else:
            print(f"❌ {MODEL} not found!")
            print(f"\nAvailable models:")
            for name in model_names:
                if 'qwen' in name.lower() or '4b' in name.lower():
                    print(f"   - {name}")
            return
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return

    # Prompts
    prompts = [
        "Write a short story about AI and humanity.",
        "Explain quantum computing in simple terms.",
        "Describe the history of the internet.",
        "What is machine learning and how does it work?",
        "Write about the future of space exploration.",
        "Explain how the human brain processes language.",
        "Describe the impact of climate change on ecosystems.",
        "What is the meaning of consciousness?",
        "Write a technical explanation of blockchain.",
        "Explain the theory of relativity simply.",
    ]

    dataset = []

    print(f"\n📝 Generating {len(prompts) * NUM_SAMPLES_PER_PROMPT} samples...")
    print()

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

        for j in range(NUM_SAMPLES_PER_PROMPT):
            try:
                print(f"   Generating sample {j+1}/{NUM_SAMPLES_PER_PROMPT}... ", end='', flush=True)

                response = ollama.generate(
                    model=MODEL,
                    prompt=prompt,
                    options={
                        'num_predict': 300,
                        'temperature': 0.8,
                        'top_p': 0.9,
                    }
                )

                generated = response.get('response', '')

                if len(generated) > 100:
                    dataset.append({
                        "prompt": prompt,
                        "generated": generated,
                        "length": len(generated),
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"✅ {len(generated)} chars")
                else:
                    print(f"⚠️  too short: {len(generated)} chars")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    # Save dataset
    print()
    print("="*60)
    print(f"💾 Saving {len(dataset)} samples to {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ Saved!")

    # Statistics
    total_chars = sum(item['length'] for item in dataset)
    print()
    print("📊 Statistics:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average length: {total_chars // len(dataset) if dataset else 0:,} chars")
    print()
    print("="*60)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*60)
    print()
    print("Next step:")
    print("   python train_ollama_distillation.py")
    print()


if __name__ == "__main__":
    main()
