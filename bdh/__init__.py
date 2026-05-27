"""BDH — Bio-Distilled Hebbian Network.

Brain-inspired linear attention with multi-scale Hebbian memory.
"""

from bdh.config import BDHv2Config
from bdh.rope import RotaryPositionEmbedding
from bdh.attention import MultiScaleLinearAttention
from bdh.layers import BDHv2Layer
from bdh.vocab_projection import VocabProjection
from bdh.model import BDHv2
from bdh.utils import count_parameters

__version__ = "0.2.0"
__author__ = "BDH Team"

__all__ = [
    "BDHv2Config",
    "BDHv2",
    "BDHv2Layer",
    "MultiScaleLinearAttention",
    "RotaryPositionEmbedding",
    "VocabProjection",
    "count_parameters",
]
