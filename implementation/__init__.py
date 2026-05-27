"""BDH — Bio-Distilled Hebbian Network.

Brain-inspired linear attention with multi-scale Hebbian memory.

This module re-exports from the `bdh` package for backward compatibility.
For new code, import directly from `bdh`:

    from bdh import BDHv2, BDHv2Config
"""

from bdh import (
    BDHv2Config,
    BDHv2,
    BDHv2Layer,
    MultiScaleLinearAttention,
    RotaryPositionEmbedding,
    VocabProjection,
    count_parameters,
    __version__,
)

__all__ = [
    "BDHv2Config",
    "BDHv2",
    "BDHv2Layer",
    "MultiScaleLinearAttention",
    "RotaryPositionEmbedding",
    "VocabProjection",
    "count_parameters",
    "__version__",
]
