"""Configuration for BDH v2 architecture."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BDHv2Config:
    """Configuration for BDH v2 baseline architecture."""

    # Architecture
    vocab_size: int = 32000
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
    init_states: str = "zeros"

    # Fix flags
    use_outer_product: bool = False
    use_rope: bool = False
    rope_base: float = 10000.0
    use_hybrid_gating: bool = False
    gating_alpha: float = 0.1
    use_vocab_projection: bool = False
    teacher_vocab_size: int = 151936

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __post_init__(self):
        assert len(self.decay_rates) == self.num_scales
        assert len(self.scale_weights) == self.num_scales
        assert all(0 < d < 1 for d in self.decay_rates)
        assert abs(sum(self.scale_weights) - 1.0) < 1e-6
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
