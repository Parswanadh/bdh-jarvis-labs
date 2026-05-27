"""
Download TinyStories dataset from HuggingFace.
"""

from datasets import load_dataset
from pathlib import Path

print("[DOWNLOAD] Downloading TinyStories dataset...")
print("This may take a while...")

# Download dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

print(f"[OK] Downloaded {len(dataset)} stories")

# Save to file
output_file = Path("data/tinystories.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"[SAVE] Saving to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for example in dataset:
        f.write(example['text'].strip() + '\n')

print(f"[OK] Saved!")
print(f"[INFO] File size: {output_file.stat().st_size / 1e9:.2f} GB")
print()
print("Ready to train!")
