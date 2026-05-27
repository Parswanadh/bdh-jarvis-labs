"""
BBPE Tokenizer Training Script (Fixed)
===============================

Train a Byte-Level BPE (BBPE) tokenizer for BDH to achieve 2-4× token reduction
while maintaining byte-level foundation for biological plausibility.

Author: Tokenization Engineer (BDH Science Fest Sprint)
Date: 2025-02-25
"""

import os
import sys
import argparse
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import BertProcessing, ByteLevel


def create_training_data(input_files, output_file, max_size_mb=10):
    """
    Create training data sample for tokenizer training.
    """
    print(f"Creating training data from {len(input_files)} files...")

    total_bytes = 0
    max_bytes = max_size_mb * 1024 * 1024

    with open(output_file, 'w', encoding='utf-8', errors='ignore') as out_f:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping...")
                continue

            print(f"Processing {input_file}...")
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as in_f:
                for line in in_f:
                    if total_bytes >= max_bytes:
                        print(f"Reached {max_size_mb}MB limit, stopping...")
                        break

                    out_f.write(line)
                    total_bytes += len(line.encode('utf-8'))

            if total_bytes >= max_bytes:
                break

    final_size_mb = total_bytes / (1024 * 1024)
    print(f"Created training data: {output_file} ({final_size_mb:.2f}MB)")
    return output_file


def train_bbpe_tokenizer(
    training_file,
    vocab_size=8192,
    output_dir="tokenizer-model",
    special_tokens=None
):
    """
    Train a Byte-Level BPE tokenizer with proper roundtrip support.
    """
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "<|endoftext|>"]

    print(f"\nTraining BBPE tokenizer...")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training data: {training_file}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Byte-Level BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Use ByteLevel pre-tokenizer for proper byte handling
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train tokenizer
    tokenizer.train([training_file], trainer=trainer)

    # Use ByteLevel post-processor for proper decoding
    from tokenizers.processors import ByteLevel
    tokenizer.post_processor = ByteLevel()

    # Save tokenizer
    tokenizer_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)

    print(f"\nTokenizer saved to: {tokenizer_file}")

    # Print tokenizer info
    print(f"\nTokenizer Information:")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Special tokens: {special_tokens}")

    return tokenizer


def test_tokenizer(tokenizer, test_texts=None):
    """
    Test the trained tokenizer with sample texts.
    """
    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, world!",
            "This is a test of BBPE tokenization for BDH.",
            "Science fair demo: 2-4x token reduction achieved!",
        ]

    print(f"\n" + "="*60)
    print(f"Testing Tokenizer")
    print(f"="*60)

    for i, text in enumerate(test_texts, 1):
        encoding = tokenizer.encode(text)

        # Get byte-level token count (baseline)
        byte_count = len(text.encode('utf-8'))

        print(f"\nTest {i}: {text[:50]}...")
        print(f"  Byte-level tokens: {byte_count}")
        print(f"  BBPE tokens: {len(encoding.ids)}")
        print(f"  Reduction ratio: {byte_count / len(encoding.ids):.2f}x")

    # Test roundtrip
    print(f"\n" + "-"*60)
    print("Roundtrip Test (encode -> decode)")
    print("-"*60)

    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)

    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {test_text == decoded}")

    if test_text != decoded:
        print("WARNING: Roundtrip failed!")
        return False
    else:
        print("SUCCESS: Roundtrip passed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train BBPE tokenizer for BDH")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["paper_text.txt"],
        help="Input text files for training data"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8192,
        help="Vocabulary size (default: 8192 for optimal 2-4x reduction)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer-model",
        help="Output directory for trained tokenizer"
    )
    parser.add_argument(
        "--max_data_mb",
        type=int,
        default=10,
        help="Maximum training data size in MB (default: 10)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test existing tokenizer, don't train"
    )

    args = parser.parse_args()

    # Resolve paths
    input_files = [str(Path(f).resolve()) for f in args.input]
    output_dir = str(Path(args.output_dir).resolve())

    # Check if we're just testing
    if args.test_only:
        tokenizer_file = os.path.join(output_dir, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            print(f"Loading existing tokenizer from: {tokenizer_file}")
            tokenizer = Tokenizer.from_file(tokenizer_file)
            test_tokenizer(tokenizer)
        else:
            print(f"Error: Tokenizer not found at {tokenizer_file}")
            print("Please train first without --test_only flag")
        return

    # Create training data
    training_file = "tokenizer_training_data.txt"
    create_training_data(input_files, training_file, args.max_data_mb)

    # Train tokenizer
    tokenizer = train_bbpe_tokenizer(
        training_file=training_file,
        vocab_size=args.vocab_size,
        output_dir=output_dir
    )

    # Test tokenizer
    success = test_tokenizer(tokenizer)

    # Cleanup training data
    if os.path.exists(training_file):
        os.remove(training_file)
        print(f"\nCleaned up training data: {training_file}")

    print(f"\n" + "="*60)
    print(f"BBPE Tokenizer Training Complete!")
    print(f"="*60)
    print(f"\nTo use this tokenizer:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{output_dir}/tokenizer.json')")
    print(f"  tokens = tokenizer.encode('Your text here').ids")
    print(f"\nNext steps:")
    print(f"  1. Modify BDHConfig to use vocab_size={tokenizer.get_vocab_size()}")
    print(f"  2. Update token embedding layer dimension")
    print(f"  3. Integrate with training pipeline")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
