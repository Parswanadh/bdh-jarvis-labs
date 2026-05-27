"""
Generate Training Data with Gemma 3 270M via Ollama
======================================================

This script generates high-quality training data using Gemma 3 270M
running locally via Ollama. The generated data is then used to train
Multi-Scale BDH via knowledge distillation.

Optimized for Jarvis Labs A100 80GB.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

# Configuration
MODEL = "gemma3:270m"
OUTPUT_DIR = Path("teacher_data")
OUTPUT_FILE = OUTPUT_DIR / "gemma_training_data.jsonl"
NUM_SAMPLES_PER_PROMPT = 100  # Adjust based on time
TIMEOUT = 300  # 5 minutes per batch

# Diverse prompts for data generation
PROMPTS = {
    "stories": [
        "Write a short story about a robot that learns to love.",
        "Once upon a time, in a magical forest, there lived",
        "The time machine hummed to life. Professor Johnson stepped through",
        "In the year 2150, humanity finally made contact with",
        "The dragon woke up from its thousand-year slumber",
    ],
    "science": [
        "Explain how photosynthesis works in plants.",
        "What is quantum computing and how does it work?",
        "Describe the theory of evolution by natural selection.",
        "What causes climate change and how can we address it?",
        "Explain how the human brain processes language.",
    ],
    "history": [
        "What were the causes and consequences of World War I?",
        "Describe the Industrial Revolution and its impact on society.",
        "Explain the fall of the Roman Empire.",
        "What was the Renaissance and why was it important?",
        "Describe the Space Race between USA and USSR.",
    ],
    "technology": [
        "Explain how the internet works.",
        "What is artificial intelligence and machine learning?",
        "Describe how blockchain technology works.",
        "What is the difference between RAM and CPU?",
        "Explain how neural networks learn from data.",
    ],
    "philosophy": [
        "What is consciousness and do animals have it?",
        "Explain the trolley problem in ethics.",
        "What is utilitarianism as a moral theory?",
        "Describe the concept of free will.",
        "What is the meaning of life according to different philosophers?",
    ],
}


def check_ollama():
    """Check if Ollama is running and model is available."""
    try:
        # Check if Ollama is running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print("❌ Ollama is not running!")
            print("   Start with: ollama serve &")
            return False

        # Check if model is available
        if MODEL in result.stdout:
            print(f"✅ Found {MODEL}")
            return True
        else:
            print(f"❌ {MODEL} not found!")
            print(f"   Pull it with: ollama pull {MODEL}")
            return False

    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def generate_with_ollama(prompt: str, max_tokens: int = 500) -> str:
    """
    Generate text using Ollama Gemma 3 270M.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
        }
    }

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return ""

    except subprocess.TimeoutExpired:
        print("   ⚠️  Generation timed out")
        return ""
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return ""


def generate_dataset():
    """Generate complete training dataset."""

    print("="*70)
    print("GENERATING TRAINING DATA WITH GEMMA 3 270M")
    print("="*70)
    print()

    # Check Ollama
    if not check_ollama():
        print("\n❌ Please start Ollama and pull the model first:")
        print(f"   ollama serve &")
        print(f"   ollama pull {MODEL}")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = []
    total_samples = 0
    target_samples = NUM_SAMPLES_PER_PROMPT * len(PROMPTS)

    print(f"\n🎯 Target: {target_samples} samples")
    print(f"⏱️  Estimated time: {target_samples * 2 / 60:.1f} minutes")
    print()

    start_time = time.time()

    for category, prompts in PROMPTS.items():
        print(f"\n📂 Category: {category}")

        for i, prompt in enumerate(prompts, 1):
            print(f"   Prompt {i}/{len(prompts)}: '{prompt[:50]}...'")

            batch_count = 0
            failures = 0

            for j in range(NUM_SAMPLES_PER_PROMPT):
                print(f"      Sample {j+1}/{NUM_SAMPLES_PER_PROMPT}... ", end="", flush=True)

                generated = generate_with_ollama(prompt, max_tokens=500)

                if len(generated) > 100:
                    dataset.append({
                        "category": category,
                        "prompt": prompt,
                        "generated": generated,
                        "length": len(generated),
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"✅ {len(generated)} chars")
                    batch_count += 1
                else:
                    print(f"⚠️  too short: {len(generated)} chars")
                    failures += 1

                total_samples += 1
                rate_limit = 0.1  # Rate limiting
                time.sleep(rate_limit)

            print(f"   ✅ Generated {batch_count} samples")

    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"✅ GENERATION COMPLETE!")
    print("="*70)
    print(f"   Total samples: {len(dataset)}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print()

    # Save dataset
    print(f"💾 Saving to {OUTPUT_FILE}")
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
    print(f"   Total words: ~{total_chars // 5:,}")
    print(f"   Average length: {total_chars // len(dataset):,} chars")
    print()

    return dataset


if __name__ == "__main__":
    generate_dataset()
