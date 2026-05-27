"""Complete BDH v2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from bdh.config import BDHv2Config
from bdh.layers import BDHv2Layer
from bdh.vocab_projection import VocabProjection


class BDHv2(nn.Module):
    """Complete BDH v2 model with multi-scale Hebbian memory.

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

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([BDHv2Layer(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)

        if config.use_vocab_projection:
            self.vocab_projection = VocabProjection(
                config.teacher_vocab_size, config.vocab_size
            )
        else:
            self.vocab_projection = None

        self._sparsity_history: List[float] = []
        self.apply(self._init_weights)

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
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        x = self.token_embedding(idx)
        x = self.dropout(x)

        if return_states and states is None:
            states = [None] * len(self.layers)

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

        avg_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0
        self._sparsity_history.append(avg_sparsity)

        x = self.norm_f(x)
        logits = self.output_projection(x)

        if return_states:
            return logits, new_states
        else:
            return logits, None

    def project_teacher_logits(self, teacher_logits: torch.Tensor) -> torch.Tensor:
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
        states = None
        for _ in range(max_new_tokens):
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
        return self._sparsity_history.copy()
