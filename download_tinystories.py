"""
Download TinyStories Dataset for BDH Training
==============================================

Downloads the TinyStories dataset from Hugging Face.
This is perfect for language model training!
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not found. Install with: pip install datasets")
    sys.exit(1)

OUTPUT_FILE = Path("data/ollama_generated/tinystories_data.jsonl")
NUM_SAMPLES = 1000


def main():
    print("="*60)
    print("DOWNLOADING TINYSTORIES DATASET")
    print("="*60)
    print()

    print(f"Loading {NUM_SAMPLES} stories from Hugging Face...")
    print()

    try:
        dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{NUM_SAMPLES}]")

        print(f"Loaded {len(dataset)} stories!")
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
        print(f"Processed {len(output_data)} stories!")

        # Calculate stats
        total_chars = sum(item['length'] for item in output_data)
        print()
        print("Dataset Statistics:")
        print(f"   Total samples: {len(output_data)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words: ~{total_chars // 5:,}")
        print(f"   Average length: {total_chars // len(output_data):,} chars")
        print()

        # Save
        print("="*60)
        print(f"Saving to {OUTPUT_FILE}")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print("Saved!")
        print()
        print("="*60)
        print("DATA READY FOR TRAINING!")
        print("="*60)
        print()
        print("Next step: python train_ollama_distillation.py")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
