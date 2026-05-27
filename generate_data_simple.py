"""
Ultra-simple Ollama Data Generator
===================================

Uses ollama CLI via subprocess - most reliable method!
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

MODEL = "qwen3-4b-fast"
OUTPUT_FILE = Path("data/ollama_generated/qwen_training_data.jsonl")

# Simple prompts that work well
PROMPTS = [
    "Write a short story about artificial intelligence.",
    "Explain how machine learning works.",
    "Describe the history of computing.",
    "What is quantum computing?",
    "Explain the theory of evolution.",
    "Write about climate change.",
    "Describe how the internet works.",
    "What is blockchain technology?",
    "Explain consciousness.",
    "Write about space exploration.",
]


def generate_with_ollama(prompt, model=MODEL):
    """Generate text using ollama CLI."""

    cmd = [
        "ollama", "run", model,
        prompt
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
            input=prompt
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            return output
        else:
            print(f"   ⚠️  Ollama error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("   ⚠️  Timeout")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    """Generate training data."""

    print("="*60)
    print("🚀 SIMPLE OLLAMA DATA GENERATOR")
    print(f"Model: {MODEL}")
    print("="*60)
    print()

    # Test ollama first
    print("🔍 Testing ollama...")
    test_result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )

    if test_result.returncode != 0:
        print("❌ Ollama not working!")
        print("   Make sure 'ollama serve' is running")
        return

    print("✅ Ollama is working!")
    print()

    dataset = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {prompt}")
        print(f"   Generating... ", end='', flush=True)

        generated = generate_with_ollama(prompt)

        if generated and len(generated) > 50:
            dataset.append({
                "prompt": prompt,
                "generated": generated,
                "length": len(generated),
                "timestamp": datetime.now().isoformat()
            })
            print(f"✅ {len(generated)} chars")
        else:
            print(f"⚠️  Failed or too short")

        print()

    # Save
    print("="*60)
    print(f"💾 Saving {len(dataset)} samples...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ Saved!")

    # Stats
    total_chars = sum(item['length'] for item in dataset)
    print()
    print("📊 Statistics:")
    print(f"   Samples: {len(dataset)}")
    print(f"   Characters: {total_chars:,}")
    print(f"   Average: {total_chars // len(dataset) if dataset else 0:,} chars/sample")
    print()
    print("="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print()
    print("Next: python train_ollama_distillation.py")


if __name__ == "__main__":
    main()
