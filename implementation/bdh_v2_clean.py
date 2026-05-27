"""
BDH v2 Clean — Baseline Architecture
=====================================

Clean rewrite of the Multi-Scale BDH architecture from scratch.
NOT a patch of the old code — built from first principles.

This v2 baseline preserves the current element-wise Hebbian behavior
for comparison purposes, but is structured to easily swap in:
- True outer product Hebbian (Day 2)
- RoPE positional encoding (Day 3)
- Hybrid gated residual (Day 4)
- Vocab projection for distillation (Day 5)

Key improvements over v1:
- Consistent tensor shapes throughout
- Clear separation of concerns
- Unit-testable components
- Configuration flags for each fix
- Proper gradient flow

Author: BDH v2 Architecture
Date: 2026-05-17 (Day 1 — Baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BDHv2Config:
    """Configuration for BDH v2 baseline architecture."""

    # Architecture
    vocab_size: int = 32000  # TinyLlama tokenizer (fixes Bug #4)
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048

    # Multi-scale memory
    decay_rates: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    num_scales: int = 3
    scale_weights: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    hebbian_lr: float = 0.001

    # State initialization
    init_states: str = "zeros"  # "zeros" or "xavier"

    # === FIX FLAGS (toggle each fix on/off for ablation) ===
    use_outer_product: bool = False  # Day 2: True outer product Hebbian
    use_rope: bool = False  # Day 3: Enable RoPE
    rope_base: float = 10000.0  # RoPE base frequency
    use_hybrid_gating: bool = False  # Day 4: Hybrid residual (x + α·gate·residual)
    gating_alpha: float = 0.1  # Learnable gating coefficient
    use_vocab_projection: bool = False  # Day 5: Vocab projection for distillation
    teacher_vocab_size: int = 151936  # Qwen3.5 vocab size

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __post_init__(self):
        assert len(self.decay_rates) == self.num_scales
        assert len(self.scale_weights) == self.num_scales
        assert all(0 < d < 1 for d in self.decay_rates)
        assert abs(sum(self.scale_weights) - 1.0) < 1e-6
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


# ============================================================================
# RoPE — Rotary Position Embedding
# ============================================================================


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position by rotating Q/K in the embedding space.
    Ready to enable on Day 3.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute frequencies
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
        x = x.transpose(1, 2)  # [B, H, T, D] for broadcasting
        x_rot = (x * cos) + (self.rotate_half(x) * sin)
        return x_rot.transpose(1, 2)  # [B, T, H, D]


# ============================================================================
# Multi-Scale Linear Attention
# ============================================================================


class MultiScaleLinearAttention(nn.Module):
    """
    Linear attention with multi-scale state matrices.

    Maintains N independent state matrices with different decay rates:
    - E_fast:  Fast decay (~100 token timescale, γ=0.95)
    - E_med:   Medium decay (~500 token timescale, γ=0.99)
    - E_slow:  Slow decay (~2000 token timescale, γ=0.995)

    Each state is updated via Hebbian learning, then combined by weighted average.
    """

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        # Q, K, V projections
        self.Wq = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wk = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wv = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wo = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # RoPE (ready for Day 3)
        self.rope = RotaryPositionEmbedding(
            self.head_dim, config.max_seq_len, config.rope_base
        )

        self.dropout = nn.Dropout(config.dropout)

        # Scale weights as buffer for efficient computation
        self.register_buffer("scale_weights", torch.tensor(config.scale_weights))

    def _init_states(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize multi-scale state matrices. Shape: num_scales × [H, D, D]."""
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
        """
        Compute Hebbian update. Returns per-head state update [H, D, D].

        BASELINE (v2, element-wise): diagonal matrix — only self-correlations
        OUTER PRODUCT (Day 2): full matrix — all cross-dimensional correlations

        Args:
            Q: [B, T, H, D]
            V: [B, T, H, D]

        Returns:
            [H, D, D] per-head Hebbian update
        """
        Q_flat = Q.mean(dim=(0, 1))  # [H, D]
        V_flat = V.mean(dim=(0, 1))  # [H, D]

        if self.config.use_outer_product:
            # DAY 2: True outer product — captures cross-dim correlations
            hebbian = torch.einsum("hd,he->hde", Q_flat, V_flat)  # [H, D, D]
        else:
            # BASELINE: Element-wise → diagonal matrix (v1 behavior)
            hebbian = torch.diag_embed(Q_flat * V_flat)  # [H, D, D]

        return hebbian

    def _update_states(
        self, states: Tuple[torch.Tensor, ...], hebbian: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Update all state matrices with decay and Hebbian learning.

        For each scale i: E_i ← γ_i · E_i + η · hebbian

        Handles both element-wise [C] and outer product [C, C] hebbian updates.
        """
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
        """
        Forward pass with multi-scale state matrix updates.

        Args:
            x: [B, T, C]
            states: Previous state matrices or None
            return_states: Whether to return updated states

        Returns:
            output: [B, T, C]
            states: Updated states if return_states
        """
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # Project to Q, K, V
        Q = self.Wq(x).view(B, T, H, D)
        K = self.Wk(x).view(B, T, H, D)
        V = self.Wv(x).view(B, T, H, D)

        # DAY 3: Apply RoPE (toggle via config.use_rope)
        if self.config.use_rope:
            cos, sin = self.rope(Q, T)
            Q = self.rope.apply_rotary(Q, cos, sin)
            K = self.rope.apply_rotary(K, cos, sin)

        # Linear attention: Q @ (K^T @ V) — O(N) complexity
        # K^T @ V = [B, H, D, T] @ [B, H, T, D] = [B, H, D, D]
        kv = torch.matmul(K.transpose(-2, -1), V)  # [B, H, D, D]
        attn = torch.matmul(Q, kv)  # [B, T, H, D]

        # Concatenate heads and project
        attn = attn.contiguous().view(B, T, C)
        out = self.Wo(attn)
        out = self.dropout(out)

        # Multi-scale state updates
        if return_states:
            if states is None:
                states = self._init_states(x.device, x.dtype)

            hebbian = self._compute_hebbian_update(Q, V)
            states = self._update_states(states, hebbian)

            return out, states
        else:
            return out, None


# ============================================================================
# ReLU Low-Rank FFN
# ============================================================================


class ReLULowRankFFN(nn.Module):
    """
    ReLU Low-Rank Feed-Forward Network.

    The "inhibitory circuit" in BDH terminology.
    ReLU creates ~95% sparsity (~5% neurons active).
    """

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.W1 = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.W2 = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward pass.

        Returns:
            output: [B, T, C]
            sparsity: fraction of zero activations (for monitoring)
        """
        h = self.W1(x)
        h = F.relu(h)

        # Measure sparsity
        sparsity = (h == 0).float().mean().item()

        out = self.W2(h)
        out = self.dropout(out)

        return out, sparsity


# ============================================================================
# BDH Layer
# ============================================================================


class BDHv2Layer(nn.Module):
    """
    Single BDH v2 layer.

    Combines:
    1. Multi-Scale Linear Attention (excitatory circuit)
    2. ReLU Low-Rank FFN (inhibitory circuit)
    3. Gating (configurable: multiplicative or hybrid)
    4. Pre-norm LayerNorm
    """

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiScaleLinearAttention(config)
        self.ffn = ReLULowRankFFN(config)

        # DAY 4: Learnable gating coefficient (for hybrid residual)
        if config.use_hybrid_gating:
            self.gating_alpha = nn.Parameter(torch.tensor(config.gating_alpha))

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, ...]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], float]:
        """
        Forward pass.

        Returns:
            output: [B, T, C]
            states: Updated states if return_states
            sparsity: FFN sparsity metric
        """
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, states = self.attention(x_norm, states, return_states=return_states)

        # Pre-norm FFN
        x_norm = self.norm2(x)
        ffn_out, sparsity = self.ffn(x_norm)

        # GATING
        if self.config.use_hybrid_gating:
            # DAY 4: Hybrid residual — additive base + multiplicative modulation
            # out = x + α · σ(attn + ffn) · (attn + ffn)
            gate = torch.sigmoid(attn_out + ffn_out)
            residual = attn_out + ffn_out
            out = x + self.gating_alpha * gate * residual
        else:
            # BASELINE: Pure multiplicative gating (v1 behavior)
            gate = torch.sigmoid(attn_out + ffn_out)
            out = x * gate

        return out, states, sparsity


# ============================================================================
# Vocab Projection (Day 5)
# ============================================================================


class VocabProjection(nn.Module):
    """
    Learned projection from teacher vocab to student vocab.

    Maps teacher logits [B, T, teacher_vocab] → [B, T, student_vocab]
    for proper KL divergence computation in distillation.
    """

    def __init__(self, teacher_vocab: int, student_vocab: int):
        super().__init__()
        # Low-rank projection to reduce parameters
        rank = min(teacher_vocab, student_vocab, 512)
        self.down = nn.Linear(teacher_vocab, rank, bias=False)
        self.up = nn.Linear(rank, student_vocab, bias=False)

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Project teacher logits to student vocab space."""
        return self.up(self.down(teacher_logits))


# ============================================================================
# Complete BDH v2 Model
# ============================================================================


class BDHv2(nn.Module):
    """
    Complete BDH v2 model.

    Architecture:
    - Token embedding (vocab_size=32000, TinyLlama tokenizer)
    - 8 BDH v2 layers with multi-scale linear attention
    - ReLU low-rank FFN in each layer
    - Configurable gating (multiplicative or hybrid)
    - Optional RoPE positional encoding
    - Optional vocab projection for distillation

    Expected: ~100M parameters at default config
    """

    def __init__(self, config: BDHv2Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # BDH v2 layers
        self.layers = nn.ModuleList([BDHv2Layer(config) for _ in range(config.n_layer)])

        # Final norm and output projection
        self.norm_f = nn.LayerNorm(config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.output_projection.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

        # DAY 5: Vocab projection for distillation
        if config.use_vocab_projection:
            self.vocab_projection = VocabProjection(
                config.teacher_vocab_size, config.vocab_size
            )
        else:
            self.vocab_projection = None

        # Initialize weights
        self.apply(self._init_weights)

        # Sparsity tracking
        self._sparsity_history = []

        print(f"BDH v2 Model initialized")
        print(f"  Vocab: {config.vocab_size}")
        print(
            f"  Layers: {config.n_layer}, Embedding: {config.n_embd}, Heads: {config.n_head}"
        )
        print(f"  Scales: {config.num_scales}, Decay: {config.decay_rates}")
        print(f"  Outer product: {config.use_outer_product}")
        print(f"  RoPE: {config.use_rope}")
        print(f"  Hybrid gating: {config.use_hybrid_gating}")
        print(f"  Vocab projection: {config.use_vocab_projection}")
        print(f"  Total parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        """Xavier uniform initialization (stable_config.py recommends 0.006 range)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embedding (for fair comparison)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, ...]]]]:
        """
        Forward pass.

        Args:
            idx: [B, T] token indices
            states: List of state tuples per layer, or None
            return_states: Whether to return updated states

        Returns:
            logits: [B, T, vocab_size]
            states: Updated states if return_states
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        # Embedding
        x = self.token_embedding(idx)
        x = self.dropout(x)

        # Initialize states
        if return_states and states is None:
            states = [None] * len(self.layers)

        # Forward through layers
        new_states = []
        sparsities = []
        for i, layer in enumerate(self.layers):
            layer_states = states[i] if states is not None else None
            x, layer_states, sparsity = layer(
                x, layer_states, return_states=return_states
            )
            sparsities.append(sparsity)
            if return_states:
                new_states.append(layer_states)

        # Track average sparsity
        avg_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0
        self._sparsity_history.append(avg_sparsity)

        # Final norm and output
        x = self.norm_f(x)
        logits = self.output_projection(x)

        if return_states:
            return logits, new_states
        else:
            return logits, None

    def project_teacher_logits(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Project teacher logits to student vocab (Day 5)."""
        if self.vocab_projection is None:
            raise RuntimeError(
                "Vocab projection not enabled. Set use_vocab_projection=True."
            )
        return self.vocab_projection(teacher_logits)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive token generation."""
        states = None
        for _ in range(max_new_tokens):
            # Truncate context if too long
            if idx.size(1) > self.config.max_seq_len:
                idx = idx[:, -self.config.max_seq_len :]

            logits, states = self(idx, states=states, return_states=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def get_sparsity_history(self) -> List[float]:
        """Return tracked sparsity values."""
        return self._sparsity_history.copy()


# ============================================================================
# Utility
# ============================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# Main — Quick smoke test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BDH v2 Baseline — Smoke Test")
    print("=" * 60)

    # Default config (baseline: element-wise Hebbian, no RoPE, multiplicative gating)
    config = BDHv2Config(
        vocab_size=32000,
        n_embd=512,
        n_layer=8,
        n_head=8,
        ffn_dim=2048,
        dropout=0.1,
    )

    model = BDHv2(config)
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,} ({total / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")

    # Forward pass
    batch_size, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\nInput shape: {x.shape}")

    logits, states = model(x)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # With states
    logits, states = model(x, return_states=True)
    print(f"States returned: {len(states)} layers × {len(states[0])} scales each")
    print(f"State shape: {states[0][0].shape}")

    # Generation
    context = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(context, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (1, 30)

    print("\n✓ All smoke tests passed!")
