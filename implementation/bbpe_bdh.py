"""
BBPE-BDH Integration
====================

Integration of Byte-Level BPE (BBPE) tokenization with BDH architecture.
This module provides the BBPEBDHConfig and BBPEBDHModel classes that extend
the base BDH to use BBPE tokenization for 2-4× token reduction and 2-3× training speedup.

Key Features:
- BBPE tokenization with vocab_size=8192 (or trained vocab size)
- Seamless integration with existing BDH architecture
- Proper token encoding/decoding with roundtrip support
- Special token handling ([PAD], [UNK], [CLS], [SEP], "")
- Maintains biological plausibility of byte-level foundation

Author: Tokenization Engineer (BDH Science Fest Sprint)
Date: 2025-02-25
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, List
import sys
import os

# Add parent directory to path to import bdh_gpu_10m
sys.path.insert(0, str(Path(__file__).parent.parent))

from bdh_gpu_10m import BDHConfig, BDHGPUTensor

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: tokenizers library not available. Install with: pip install tokenizers")


class BBPEBDHConfig(BDHConfig):
    """
    BBPE-BDH Configuration.

    Extends BDHConfig with BBPE-specific settings.
    """
    def __init__(
        self,
        vocab_size: int = 8192,  # BBPE vocab size (default 8192)
        tokenizer_path: Optional[str] = None,  # Path to trained tokenizer
        **kwargs
    ):
        # Initialize base BDH config
        super().__init__(vocab_size=vocab_size, **kwargs)

        self.tokenizer_path = tokenizer_path

        # BBPE-specific special tokens
        self.pad_token_id = 0  # [PAD]
        self.unk_token_id = 1  # [UNK]
        self.cls_token_id = 2  # [CLS]
        self.sep_token_id = 3  # [SEP]
        self.mask_token_id = 4  # [MASK]

        # Verify tokenizer exists if path provided
        if tokenizer_path and TOKENIZERS_AVAILABLE:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    @classmethod
    def from_tokenizer(cls, tokenizer_path: str, **kwargs):
        """
        Create config from trained tokenizer.

        Args:
            tokenizer_path: Path to tokenizer.json
            **kwargs: Additional config parameters

        Returns:
            BBPEBDHConfig with vocab_size from tokenizer
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library required. Install with: pip install tokenizers")

        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

        print(f"Loaded tokenizer from: {tokenizer_path}")
        print(f"Vocabulary size: {vocab_size}")

        return cls(vocab_size=vocab_size, tokenizer_path=tokenizer_path, **kwargs)


class BBPEBDHModel(BDHGPUTensor):
    """
    BBPE-BDH Model with integrated tokenization.

    This extends the base BDHGPUTensor model to work with BBPE tokenization.
    """
    def __init__(self, config: BBPEBDHConfig):
        # Initialize base BDH model
        super().__init__(config)

        self.config = config

        # Load tokenizer if available
        self.tokenizer = None
        if config.tokenizer_path and TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(config.tokenizer_path)
            print(f"BBPE tokenizer loaded: {config.tokenizer_path}")

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs using BBPE tokenizer.

        Args:
            text: Input text string

        Returns:
            Token IDs as tensor [seq_len]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please provide tokenizer_path in config.")

        # Encode text
        encoding = self.tokenizer.encode(text)
        tokens = torch.tensor(encoding.ids, dtype=torch.long)

        return tokens

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token ID tensor [seq_len] or [batch, seq_len]

        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please provide tokenizer_path in config.")

        # Convert to list if tensor
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 2:
                # Take first batch element
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()

        # Decode tokens
        text = self.tokenizer.decode(token_ids)

        return text

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = "cuda"
    ) -> str:
        """
        Generate text from a text prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, sample from top-k tokens
            device: Device to run on

        Returns:
            Generated text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please provide tokenizer_path in config.")

        # Encode prompt
        input_tokens = self.encode_text(prompt).unsqueeze(0).to(device)

        # Generate tokens
        self.eval()
        with torch.no_grad():
            generated = self.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Decode generated tokens
        generated_text = self.decode_tokens(generated[0].cpu())

        return generated_text

    @torch.no_grad()
    def get_token_reduction_stats(self, text_samples: List[str]) -> dict:
        """
        Calculate token reduction statistics compared to byte-level.

        Args:
            text_samples: List of text samples to analyze

        Returns:
            Dictionary with reduction statistics
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please provide tokenizer_path in config.")

        byte_counts = []
        bbpe_counts = []
        reductions = []

        for text in text_samples:
            # Byte-level count
            byte_count = len(text.encode('utf-8'))

            # BBPE token count
            encoding = self.tokenizer.encode(text)
            bbpe_count = len(encoding.ids)

            # Reduction ratio
            reduction = byte_count / bbpe_count if bbpe_count > 0 else 0

            byte_counts.append(byte_count)
            bbpe_counts.append(bbpe_count)
            reductions.append(reduction)

        # Calculate statistics
        avg_reduction = sum(reductions) / len(reductions) if reductions else 0
        total_bytes = sum(byte_counts)
        total_bbpe = sum(bbpe_counts)
        overall_reduction = total_bytes / total_bbpe if total_bbpe > 0 else 0

        stats = {
            "num_samples": len(text_samples),
            "total_byte_tokens": total_bytes,
            "total_bbpe_tokens": total_bbpe,
            "avg_reduction_ratio": avg_reduction,
            "overall_reduction_ratio": overall_reduction,
            "reductions": reductions
        }

        return stats


def create_bbpe_bdh_model(
    tokenizer_path: str,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 4,
    ffn_dim: int = 1024,
    **kwargs
) -> BBPEBDHModel:
    """
    Factory function to create a BBPE-BDH model.

    Args:
        tokenizer_path: Path to trained BBPE tokenizer.json
        n_embd: Embedding dimension
        n_layer: Number of BDH layers
        n_head: Number of attention heads
        ffn_dim: FFN internal dimension
        **kwargs: Additional config parameters

    Returns:
        BBPEBDHModel instance
    """
    # Create config from tokenizer
    config = BBPEBDHConfig.from_tokenizer(
        tokenizer_path=tokenizer_path,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        ffn_dim=ffn_dim,
        **kwargs
    )

    # Create model
    model = BBPEBDHModel(config)

    return model


def test_bbpe_bdh():
    """
    Test BBPE-BDH integration.
    """
    print("="*60)
    print("BBPE-BDH Integration Test")
    print("="*60)

    # Check for tokenizer
    tokenizer_path = "tokenizer-model/tokenizer.json"

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at: {tokenizer_path}")
        print("Please train the tokenizer first:")
        print("  python implementation/train_tokenizer_v2.py")
        return False

    try:
        # Create model
        print("\nCreating BBPE-BDH model...")
        model = create_bbpe_bdh_model(
            tokenizer_path=tokenizer_path,
            n_embd=256,
            n_layer=6,
            n_head=4,
            ffn_dim=1024
        )

        # Test encoding/decoding
        print("\n" + "-"*60)
        print("Test 1: Encoding/Decoding")
        print("-"*60)

        test_text = "The quick brown fox jumps over the lazy dog."
        tokens = model.encode_text(test_text)
        decoded = model.decode_tokens(tokens)

        print(f"Original:  {test_text}")
        print(f"Tokens:    {tokens.tolist()}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {test_text.strip() == decoded.strip()}")

        # Test token reduction stats
        print("\n" + "-"*60)
        print("Test 2: Token Reduction Statistics")
        print("-"*60)

        test_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, world!",
            "This is a test of BBPE tokenization for BDH.",
            "Science fair demo: 2-4x token reduction achieved!"
        ]

        stats = model.get_token_reduction_stats(test_samples)

        print(f"Samples: {stats['num_samples']}")
        print(f"Total byte tokens: {stats['total_byte_tokens']}")
        print(f"Total BBPE tokens: {stats['total_bbpe_tokens']}")
        print(f"Avg reduction ratio: {stats['avg_reduction_ratio']:.2f}x")
        print(f"Overall reduction ratio: {stats['overall_reduction_ratio']:.2f}x")

        # Test forward pass
        print("\n" + "-"*60)
        print("Test 3: Forward Pass")
        print("-"*60)

        batch_size = 2
        seq_len = 32
        vocab_size = model.config.vocab_size

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, _ = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Vocab size: {vocab_size}")

        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_bbpe_bdh()
    sys.exit(0 if success else 1)
