"""
Generate Training Data with Ollama Gemma 3 270M
===============================================

This script uses your LOCAL Ollama installation of Gemma 3 270M
to generate training data for BDH distillation.

Requirements:
- Ollama installed and running
- Gemma 3 270M pulled: ollama pull gemma3:2b
"""

import requests
import json
from pathlib import Path
from typing import List, Dict
import random
from datetime import datetime
import time

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-4b-fast"  # Your Qwen 3 4B Fast model!
NUM_SAMPLES = 500  # Samples per category
OUTPUT_DIR = Path("data/ollama_generated")
OUTPUT_FILE = OUTPUT_DIR / "qwen_training_data.jsonl"


class OllamaTeacher:
    """
    Ollama Gemma 3 270M Teacher Model.

    Uses local Ollama installation for text generation.
    """

    def __init__(self, model: str = MODEL):
        """Initialize Ollama teacher."""

        print("="*60)
        print("🤖 Connecting to Ollama")
        print("="*60)
        print()

        self.model = model

        # Test connection
        try:
            response = requests.post("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✅ Connected to Ollama!")
                print(f"📚 Available models:")
                for m in models:
                    print(f"   - {m.get('name', 'unknown')}")
                print()
            else:
                print("❌ Could not connect to Ollama")
                print("   Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("   Start Ollama with: ollama serve")
            raise

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            generated = result.get('response', '')

            return generated

        except Exception as e:
            print(f"⚠️  Error generating: {e}")
            return ""

    def generate_diverse_dataset(
        self,
        num_samples: int = NUM_SAMPLES,
        output_file: Path = OUTPUT_FILE
    ) -> List[Dict]:
        """
        Generate diverse training dataset.

        Categories:
        - Stories and narratives
        - Science explanations
        - Historical facts
        - Conversations
        - Technical documentation
        """

        print("="*60)
        print("📝 Generating Diverse Training Dataset")
        print("="*60)
        print()

        # Prompts for different categories
        prompts = {
            "stories": [
                "Write a short story about a robot learning to feel emotions.",
                "Tell a story about a time traveler who accidentally changes history.",
                "Once upon a time, in a kingdom where dreams could be harvested,",
                "The last human on Earth sat alone in a room. There was a knock on the door.",
            ],
            "science": [
                "Explain how photosynthesis works in simple terms.",
                "Describe the theory of evolution and its evidence.",
                "Explain quantum entanglement like I'm five.",
                "What is machine learning and how does it work?",
                "Describe the structure and function of DNA.",
            ],
            "history": [
                "Tell me about the Industrial Revolution and its impact.",
                "Explain the causes and consequences of World War I.",
                "Describe the Renaissance period and its key figures.",
                "What were the major achievements of ancient Rome?",
            ],
            "technology": [
                "Explain how the internet works.",
                "Describe the history of computing.",
                "What is artificial intelligence and its applications?",
                "Explain how blockchain technology works.",
            ],
            "philosophy": [
                "What is consciousness and does it exist outside the brain?",
                "Explain the concept of utilitarianism in ethics.",
                "What is the meaning of life according to different philosophers?",
                "Describe the trolley problem and possible solutions.",
            ],
        }

        dataset = []
        total_prompts = sum(len(prompts) for prompts in prompts.values())
        samples_per_prompt = num_samples // total_prompts

        for category, category_prompts in prompts.items():
            print(f"📂 Generating {category}...")

            for i, prompt in enumerate(category_prompts):
                print(f"   Prompt {i+1}/{len(category_prompts)}: '{prompt[:50]}...'")

                for j in range(samples_per_prompt):
                    print(f"      Sample {j+1}/{samples_per_prompt}...")

                    try:
                        # Generate with some variation
                        generated = self.generate_text(
                            prompt,
                            temperature=random.uniform(0.7, 0.9),
                            max_tokens=random.randint(200, 512)
                        )

                        if len(generated) > 50:  # Filter short generations
                            dataset.append({
                                "category": category,
                                "prompt": prompt,
                                "generated": generated,
                                "length": len(generated),
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            print(f"      ⚠️  Generated text too short: {len(generated)} chars")

                        # Rate limiting to be nice to Ollama
                        time.sleep(0.5)  # Slower rate limiting

                    except Exception as e:
                        print(f"      ❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                print(f"   ✅ Generated {len([d for d in dataset if d['category'] == category])} samples for {category}")

        print()
        print(f"✅ Generated {len(dataset)} samples total")
        print()

        return dataset

    def save_dataset(self, dataset: List[Dict], output_file: Path = OUTPUT_FILE):
        """Save dataset to file."""

        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving dataset to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ Dataset saved!")

        # Print statistics
        categories = {}
        total_chars = 0
        total_words = 0

        for item in dataset:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
            total_chars += item['length']
            total_words += len(item['generated'].split())

        print()
        print("📊 Dataset Statistics:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words: {total_words:,}")
        print(f"   Average length: {total_chars // len(dataset)} chars")
        print()

        for cat, count in categories.items():
            print(f"   {cat}: {count} samples")


def main():
    """Main function to generate teacher data."""

    print("="*60)
    print("OLLAMA QWEN 2.5 4B - TEACHER DATA GENERATION")
    print("🎯 One of the best open models under 4B!")
    print("="*60)
    print()

    # Initialize teacher
    teacher = OllamaTeacher()

    # Generate dataset
    dataset = teacher.generate_diverse_dataset(num_samples=NUM_SAMPLES)

    # Save dataset
    teacher.save_dataset(dataset)

    print()
    print("="*60)
    print("✅ TEACHER DATA GENERATION COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Check the generated data: cat data/ollama_generated/gemma_training_data.jsonl | head -5")
    print("2. Run: python train_ollama_distillation.py")
    print()


if __name__ == "__main__":
    main()
