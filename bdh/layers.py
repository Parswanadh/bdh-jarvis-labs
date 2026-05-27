"""BDH Layer combining attention, FFN, and gating."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from bdh.config import BDHv2Config
from bdh.attention import MultiScaleLinearAttention


class ReLULowRankFFN(nn.Module):
    """ReLU Low-Rank Feed-Forward Network (~95% sparse activations)."""

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.W1 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.W2 = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        h = self.W1(x)
        h = torch.nn.functional.relu(h)
        sparsity = (h == 0).float().mean().item()
        out = self.W2(h)
        out = self.dropout(out)
        return out, sparsity


class BDHv2Layer(nn.Module):
    """Single BDH v2 layer with multi-scale attention, FFN, and gating."""

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiScaleLinearAttention(config)
        self.ffn = ReLULowRankFFN(config)

        if config.use_hybrid_gating:
            self.gating_alpha = nn.Parameter(torch.tensor(config.gating_alpha))

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, ...]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], float]:
        x_norm = self.norm1(x)
        attn_out, states = self.attention(x_norm, states, return_states=return_states)

        x_norm = self.norm2(x)
        ffn_out, sparsity = self.ffn(x_norm)

        if self.config.use_hybrid_gating:
            gate = torch.sigmoid(attn_out + ffn_out)
            residual = attn_out + ffn_out
            out = x + self.gating_alpha * gate * residual
        else:
            gate = torch.sigmoid(attn_out + ffn_out)
            out = x * gate

        return out, states, sparsity
