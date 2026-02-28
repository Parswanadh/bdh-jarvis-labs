"""
Quick Tokenizer Test (No Torch Required)
========================================

Test the BBPE tokenizer without requiring PyTorch.
This validates that the tokenizer is working correctly.

Author: Tokenization Engineer (BDH Science Fest Sprint)
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from tokenizers import Tokenizer

def main():
    print("="*60)
    print("BBPE Tokenizer Test")
    print("="*60)

    # Check for tokenizer
    tokenizer_path = "tokenizer-model/tokenizer.json"

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at: {tokenizer_path}")
        print("Please train the tokenizer first:")
        print("  python implementation/train_tokenizer_v2.py")
        return 1

    # Load tokenizer
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
        "This is a test of BBPE tokenization for BDH.",
        "Science fair demo: 2-4x token reduction achieved!",
        "BDH: The Dragon Hatchling architecture.",
    ]

    print("\n" + "-"*60)
    print("Tokenization Results")
    print("-"*60)

    total_byte_tokens = 0
    total_bbpe_tokens = 0

    for i, text in enumerate(test_texts, 1):
        # Encode
        encoding = tokenizer.encode(text)
        byte_count = len(text.encode('utf-8'))
        bbpe_count = len(encoding.ids)
        reduction = byte_count / bbpe_count if bbpe_count > 0 else 0

        total_byte_tokens += byte_count
        total_bbpe_tokens += bbpe_count

        print(f"\nTest {i}: {text}")
        print(f"  Byte tokens: {byte_count}")
        print(f"  BBPE tokens: {bbpe_count}")
        print(f"  Reduction: {reduction:.2f}x")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    overall_reduction = total_byte_tokens / total_bbpe_tokens if total_bbpe_tokens > 0 else 0
    print(f"Total byte tokens: {total_byte_tokens}")
    print(f"Total BBPE tokens: {total_bbpe_tokens}")
    print(f"Overall reduction: {overall_reduction:.2f}x")

    # Check if target achieved
    if 2 <= overall_reduction <= 4:
        print("\n✅ SUCCESS: Token reduction target (2-4x) achieved!")
        return 0
    else:
        print(f"\n⚠️  WARNING: Reduction {overall_reduction:.2f}x is outside target range (2-4x)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
