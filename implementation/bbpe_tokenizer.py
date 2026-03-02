"""
BBPE (Byte-Level BPE) Tokenizer for BDH
========================================
Maintains byte-level foundation while providing 2-4x efficiency.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path
from typing import List, Optional
import json


class BBPETokenizer:
    """
    Byte-Level BPE Tokenizer.
    Starts with all 256 bytes as base vocabulary.
    Merges most common byte pairs for efficiency.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        tokenizer_path: Optional[str] = None
    ):
        self.vocab_size = vocab_size

        if tokenizer_path and Path(tokenizer_path).exists():
            print(f"[LOAD] Loading tokenizer from: {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"[OK] Loaded tokenizer")
        else:
            print(f"[CREATE] Creating new BBPE tokenizer (vocab_size={vocab_size})")
            self.tokenizer = self._create_tokenizer(vocab_size)
            print(f"[OK] Tokenizer created")

    def _create_tokenizer(self, vocab_size: int) -> Tokenizer:
        """Create a byte-level BPE tokenizer."""

        # Byte-level BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # Pre-tokenizer: split into bytes
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        return tokenizer

    def train(
        self,
        files: List[str],
        special_tokens: Optional[List[str]] = None
    ):
        """Train tokenizer on text files."""

        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        print(f"[TRAIN] Training BBPE tokenizer...")
        print(f"  Files: {len(files)}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Special tokens: {special_tokens}")

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            show_progress=True
        )

        self.tokenizer.train(files, trainer=trainer)
        print(f"[OK] Training complete!")

    def save(self, path: str):
        """Save tokenizer to file."""
        self.tokenizer.save(path)
        print(f"[OK] Tokenizer saved to: {path}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return self.tokenizer.get_vocab_size()


def train_bbpe_tokenizer(
    data_file: str,
    vocab_size: int = 8192,
    output_path: str = "tokenizers/bbpe_tokenizer.json"
):
    """
    Train BBPE tokenizer on a text file.
    """

    print("="*70)
    print("TRAINING BBPE TOKENIZER")
    print("="*70)
    print()

    # Create tokenizer
    tokenizer = BBPETokenizer(vocab_size=vocab_size)

    # Check if data file exists
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        print(f"[INFO] Please provide a text file for training")
        return None

    # Train
    tokenizer.train([str(data_path)])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    print()
    print("="*70)
    print("TOKENIZER READY!")
    print("="*70)
    print(f"Vocab size: {tokenizer.vocab_size_actual}")
    print(f"Saved to: {output_path}")
    print()

    # Test encoding/decoding
    test_text = "Hello, world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"[TEST] Original: {test_text}")
    print(f"[TEST] Encoded: {encoded[:10]}... (showing first 10 tokens)")
    print(f"[TEST] Decoded: {decoded}")
    print()

    return tokenizer


if __name__ == "__main__":
    # Train tokenizer on TinyStories
    tokenizer = train_bbpe_tokenizer(
        data_file="data/tinystories.txt",
        vocab_size=8192,
        output_path="tokenizers/bbpe_tokenizer.json"
    )
