"""
BBPE Training Pipeline Integration
===================================

This module provides the integration layer for using BBPE tokenization
with the BDH training pipeline. It handles data loading, tokenization,
and preparation for training with the BBPE tokenizer.

Author: Tokenization Engineer (BDH Science Fest Sprint)
Date: 2025-02-25
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: tokenizers library not available")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object
    DataLoader = None
    print("Warning: PyTorch not available. Install with: pip install torch")


class BBPETextDataset(Dataset if TORCH_AVAILABLE else object):
    """
    PyTorch Dataset for BBPE-tokenized text data.

    This dataset loads text data and tokenizes it on-the-fly using
    the BBPE tokenizer for efficient memory usage.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 2048,
        sliding_window: bool = True,
        overlap: int = 256
    ):
        """
        Initialize BBPE text dataset.

        Args:
            data_path: Path to text data file
            tokenizer_path: Path to BBPE tokenizer.json
            max_seq_len: Maximum sequence length
            sliding_window: Use sliding window for long texts
            overlap: Overlap between windows (for sliding window)
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library required")

        self.max_seq_len = max_seq_len
        self.sliding_window = sliding_window
        self.overlap = overlap

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load and tokenize data
        self.data = self._load_and_tokenize(data_path)

        print(f"Dataset initialized:")
        print(f"  Sequences: {len(self.data)}")
        print(f"  Max sequence length: {max_seq_len}")

    def _load_and_tokenize(self, data_path: str) -> List[List[int]]:
        """
        Load text data and tokenize into sequences.

        Args:
            data_path: Path to text file

        Returns:
            List of token sequences
        """
        print(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Read entire file
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Tokenize entire text
        encoding = self.tokenizer.encode(text)
        all_tokens = encoding.ids

        print(f"Total tokens: {len(all_tokens)}")

        # Split into sequences
        if self.sliding_window:
            sequences = self._create_sliding_windows(all_tokens)
        else:
            sequences = self._create_fixed_windows(all_tokens)

        return sequences

    def _create_sliding_windows(self, tokens: List[int]) -> List[List[int]]:
        """Create sequences using sliding window with overlap."""
        sequences = []
        stride = self.max_seq_len - self.overlap

        for i in range(0, len(tokens) - self.max_seq_len + 1, stride):
            seq = tokens[i:i + self.max_seq_len]
            sequences.append(seq)

        # Handle remaining tokens
        if len(tokens) > self.max_seq_len and len(tokens) % stride != 0:
            seq = tokens[-self.max_seq_len:]
            sequences.append(seq)

        return sequences

    def _create_fixed_windows(self, tokens: List[int]) -> List[List[int]]:
        """Create sequences using fixed windows without overlap."""
        sequences = []
        for i in range(0, len(tokens), self.max_seq_len):
            seq = tokens[i:i + self.max_seq_len]
            if len(seq) < self.max_seq_len:
                # Pad last sequence
                seq = seq + [0] * (self.max_seq_len - len(seq))
            sequences.append(seq)

        return sequences

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence by index.

        Returns:
            Token tensor [seq_len]
        """
        tokens = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long)


class BBPEDataModule:
    """
    Data module for managing BBPE-tokenized datasets.

    Provides train/validation/test splits and dataloaders.
    """

    def __init__(
        self,
        tokenizer_path: str,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        batch_size: int = 32,
        max_seq_len: int = 2048,
        num_workers: int = 0,
        train_val_split: float = 0.9
    ):
        """
        Initialize BBPE data module.

        Args:
            tokenizer_path: Path to BBPE tokenizer.json
            train_path: Path to training data
            val_path: Path to validation data (optional)
            test_path: Path to test data (optional)
            batch_size: Batch size for dataloaders
            max_seq_len: Maximum sequence length
            num_workers: Number of workers for data loading
            train_val_split: Train/val split if val_path not provided
        """
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        # Load vocab size from tokenizer
        if TOKENIZERS_AVAILABLE:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            self.vocab_size = 5895  # Default from training

        # Create datasets
        self.train_dataset = BBPETextDataset(
            train_path, tokenizer_path, max_seq_len
        )

        # Handle validation split
        if val_path is None:
            # Split train dataset
            total_size = len(self.train_dataset)
            val_size = int(total_size * (1 - train_val_split))
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
        else:
            self.val_dataset = BBPETextDataset(
                val_path, tokenizer_path, max_seq_len
            )

        # Test dataset (optional)
        self.test_dataset = None
        if test_path is not None:
            self.test_dataset = BBPETextDataset(
                test_path, tokenizer_path, max_seq_len
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if TORCH_AVAILABLE else False
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if TORCH_AVAILABLE else False
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader if available."""
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        return None


def create_training_data_splits(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split data into train/val/test sets.

    Args:
        input_file: Input text file
        output_dir: Output directory for splits
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print(f"Splitting data from: {input_file}")
    print(f"Train/Val/Test ratios: {train_ratio}/{val_ratio}/{test_ratio}")

    # Read input file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Shuffle lines
    np.random.shuffle(lines)

    # Calculate split points
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Create splits
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write splits
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    test_path = os.path.join(output_dir, "test.txt")

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print(f"\nCreated splits in: {output_dir}")
    print(f"  Train: {len(train_lines)} lines ({train_path})")
    print(f"  Val:   {len(val_lines)} lines ({val_path})")
    print(f"  Test:  {len(test_lines)} lines ({test_path})")

    return train_path, val_path, test_path


def benchmark_tokenization(
    text_file: str,
    tokenizer_path: str,
    num_samples: int = 1000
) -> dict:
    """
    Benchmark tokenization performance.

    Args:
        text_file: Path to text file for benchmarking
        tokenizer_path: Path to BBPE tokenizer
        num_samples: Number of samples to benchmark

    Returns:
        Dictionary with benchmark results
    """
    import time

    if not TOKENIZERS_AVAILABLE:
        raise ImportError("tokenizers library required")

    print("="*60)
    print("BBPE Tokenization Benchmark")
    print("="*60)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Read samples
    with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[:num_samples]

    print(f"\nBenchmarking with {len(lines)} samples...")
    print(f"Vocabulary size: {vocab_size}")

    # Benchmark encoding
    start_time = time.time()
    total_tokens = 0
    total_bytes = 0

    for line in lines:
        encoding = tokenizer.encode(line.strip())
        total_tokens += len(encoding.ids)
        total_bytes += len(line.strip().encode('utf-8'))

    encoding_time = time.time() - start_time

    # Calculate statistics
    avg_tokens_per_line = total_tokens / len(lines)
    avg_bytes_per_line = total_bytes / len(lines)
    reduction_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    tokens_per_second = total_tokens / encoding_time if encoding_time > 0 else 0

    results = {
        "num_samples": len(lines),
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "avg_tokens_per_line": avg_tokens_per_line,
        "avg_bytes_per_line": avg_bytes_per_line,
        "reduction_ratio": reduction_ratio,
        "encoding_time": encoding_time,
        "tokens_per_second": tokens_per_second,
        "vocab_size": vocab_size
    }

    print(f"\nResults:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total bytes: {total_bytes:,}")
    print(f"  Avg tokens/line: {avg_tokens_per_line:.1f}")
    print(f"  Avg bytes/line: {avg_bytes_per_line:.1f}")
    print(f"  Reduction ratio: {reduction_ratio:.2f}x")
    print(f"  Encoding time: {encoding_time:.3f}s")
    print(f"  Tokens/second: {tokens_per_second:,.0f}")

    return results


# Example usage and testing
def main():
    """Example usage of BBPE training pipeline integration."""

    print("="*60)
    print("BBPE Training Pipeline Integration - Examples")
    print("="*60)

    # Check for tokenizer
    tokenizer_path = "tokenizer-model/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at: {tokenizer_path}")
        return 1

    # Example 1: Benchmark tokenization
    print("\n" + "-"*60)
    print("Example 1: Benchmarking Tokenization")
    print("-"*60)

    if os.path.exists("paper_text.txt"):
        results = benchmark_tokenization(
            "paper_text.txt",
            tokenizer_path,
            num_samples=100
        )

        print(f"\n✅ Benchmark complete!")
        print(f"   Token reduction: {results['reduction_ratio']:.2f}x")
        print(f"   Processing speed: {results['tokens_per_second']:,.0f} tokens/sec")

    # Example 2: Create data splits (if PyTorch available)
    if TORCH_AVAILABLE:
        print("\n" + "-"*60)
        print("Example 2: Creating Data Splits")
        print("-"*60)

        if os.path.exists("paper_text.txt"):
            train_path, val_path, test_path = create_training_data_splits(
                "paper_text.txt",
                "data_splits",
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )

            print(f"\n✅ Data splits created!")
            print(f"   Train: {train_path}")
            print(f"   Val:   {val_path}")
            print(f"   Test:  {test_path}")

        # Example 3: Create data module
        print("\n" + "-"*60)
        print("Example 3: Creating Data Module")
        print("-"*60)

        if os.path.exists("data_splits/train.txt"):
            data_module = BBPEDataModule(
                tokenizer_path=tokenizer_path,
                train_path="data_splits/train.txt",
                val_path="data_splits/val.txt",
                batch_size=32,
                max_seq_len=512
            )

            print(f"\n✅ Data module created!")
            print(f"   Vocab size: {data_module.vocab_size}")
            print(f"   Train samples: {len(data_module.train_dataset)}")
            print(f"   Val samples: {len(data_module.val_dataset)}")

    else:
        print("\n" + "-"*60)
        print("PyTorch not available - skipping data module examples")
        print("Install with: pip install torch")
        print("-"*60)

    print("\n" + "="*60)
    print("BBPE Training Pipeline Integration - Complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
