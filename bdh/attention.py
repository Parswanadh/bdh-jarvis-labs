"""Multi-Scale Linear Attention with Hebbian state matrices."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from bdh.config import BDHv2Config
from bdh.rope import RotaryPositionEmbedding


class MultiScaleLinearAttention(nn.Module):
    """Linear attention with multi-scale state matrices.

    Maintains N independent state matrices with different decay rates:
    - E_fast:  Fast decay (~100 token timescale, gamma=0.95)
    - E_med:   Medium decay (~500 token timescale, gamma=0.99)
    - E_slow:  Slow decay (~2000 token timescale, gamma=0.995)
    """

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        self.Wq = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wk = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wv = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wo = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.rope = RotaryPositionEmbedding(
            self.head_dim, config.max_seq_len, config.rope_base
        )

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("scale_weights", torch.tensor(config.scale_weights))

    def _init_states(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize multi-scale state matrices. Shape: num_scales x [H, D, D]."""
        H, D = self.n_head, self.head_dim
        if self.config.init_states == "xavier":
            states = []
            for _ in range(self.config.num_scales):
                state = torch.empty(H, D, D, device=device, dtype=dtype)
                nn.init.xavier_uniform_(state.view(H, -1))
                states.append(state)
            return tuple(states)
        else:
            return tuple(
                torch.zeros(H, D, D, device=device, dtype=dtype)
                for _ in range(self.config.num_scales)
            )

    def _compute_hebbian_update(self, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian update. Returns per-head state update [H, D, D]."""
        Q_flat = Q.mean(dim=(0, 1))
        V_flat = V.mean(dim=(0, 1))

        if self.config.use_outer_product:
            hebbian = torch.einsum("hd,he->hde", Q_flat, V_flat)
        else:
            hebbian = torch.diag_embed(Q_flat * V_flat)

        return hebbian

    def _update_states(
        self, states: Tuple[torch.Tensor, ...], hebbian: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Update all state matrices with decay and Hebbian learning."""
        updated_states = []
        for i, (state, decay) in enumerate(zip(states, self.config.decay_rates)):
            updated_state = decay * state + self.config.hebbian_lr * hebbian
            updated_states.append(updated_state)
        return tuple(updated_states)

    def _combine_states(self, states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Combine multi-scale states via weighted average."""
        return sum(w * state for w, state in zip(self.scale_weights, states))

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, ...]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass with multi-scale state matrix updates."""
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.Wq(x).view(B, T, H, D)
        K = self.Wk(x).view(B, T, H, D)
        V = self.Wv(x).view(B, T, H, D)

        if self.config.use_rope:
            cos, sin = self.rope(Q, T)
            Q = self.rope.apply_rotary(Q, cos, sin)
            K = self.rope.apply_rotary(K, cos, sin)

        kv = torch.matmul(K.transpose(-2, -1), V)
        attn = torch.matmul(Q, kv)

        attn = attn.contiguous().view(B, T, C)
        out = self.Wo(attn)
        out = self.dropout(out)

        if return_states:
            if states is None:
                states = self._init_states(x.device, x.dtype)

            hebbian = self._compute_hebbian_update(Q, V)
            states = self._update_states(states, hebbian)

            return out, states
        else:
            return out, None
