"""
Download Public Dataset for BDH Training
==========================================

Instead of generating data with Ollama (unreliable),
let's use a high-quality public dataset!

We'll use the TinyStories dataset - perfect for language modeling.
"""

import json
from pathlib import Path
from datetime import datetime

# Try importing datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

OUTPUT_FILE = Path("data/ollama_generated/tinystories_data.jsonl")
NUM_SAMPLES = 1000  # Number of stories to download


def download_tinystories():
    """Download TinyStories dataset."""

    if not DATASETS_AVAILABLE:
        print("❌ datasets library not found")
        print("   Install with: pip install datasets")
        return None

    print("="*60)
    print("DOWNLOADING TINYSTORIES DATASET")
    print("="*60)
    print()

    print("🔍 Loading TinyStories from Hugging Face...")
    print("   This is a dataset of short stories for training language models")
    print()

    try:
        # Load dataset (small subset for speed)
        dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{NUM_SAMPLES}]")

        print(f"✅ Loaded {len(dataset)} stories!")
        print()

        # Convert to our format
        output_data = []

        for i, example in enumerate(dataset):
            text = example['text']

            output_data.append({
                "category": "story",
                "prompt": "",
                "generated": text,
                "length": len(text),
                "timestamp": datetime.now().isoformat()
            })

            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(dataset)} stories...")

        print()
        print(f"✅ Processed {len(output_data)} stories!")

        # Calculate stats
        total_chars = sum(item['length'] for item in output_data)
        print()
        print("📊 Dataset Statistics:")
        print(f"   Total samples: {len(output_data)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words: ~{total_chars // 5:,}")
        print(f"   Average length: {total_chars // len(output_data):,} chars")
        print()

        return output_data

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_dataset(dataset, output_file=OUTPUT_FILE):
    """Save dataset to file."""

    if not dataset:
        print("❌ No data to save")
        return

    print("="*60)
    print(f"💾 Saving to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ Saved!")
    print()
    print("="*60)
    print("✅ DATA READY FOR TRAINING!")
    print("="*60)
    print()
    print("Next step: python train_ollama_distillation.py")
    print()


def main():
    """Main function."""

    dataset = download_tinystories()

    if dataset:
        save_dataset(dataset)
    else:
        print()
        print("="*60)
        print("❌ FAILED TO DOWNLOAD DATASET")
        print("="*60)
        print()
        print("Alternative: Install datasets library")
        print("   pip install datasets")
        print()


if __name__ == "__main__":
    main()
