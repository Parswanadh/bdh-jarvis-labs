"""Rotary Position Embedding (RoPE)."""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes position by rotating Q/K in the embedding space.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache()

    def _build_cache(self):
        t = torch.arange(self.max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin for rotation. Shape: [1, 1, seq_len, dim]."""
        cos = self.cos_cached[:, :, :seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.dtype)
        return cos, sin

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding: x_rot = x * cos + rotate_half(x) * sin.

        Args:
            x: [B, T, H, D]
            cos: [1, 1, T, D]
            sin: [1, 1, T, D]
        """
        x = x.transpose(1, 2)
        x_rot = (x * cos) + (self.rotate_half(x) * sin)
        return x_rot.transpose(1, 2)
