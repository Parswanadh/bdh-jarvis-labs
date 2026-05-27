"""Vocabulary projection for cross-architecture distillation."""

import torch
import torch.nn as nn


class VocabProjection(nn.Module):
    """Learned projection from teacher vocab to student vocab.

    Maps teacher logits [B, T, teacher_vocab] -> [B, T, student_vocab]
    for proper KL divergence computation in distillation.
    """

    def __init__(self, teacher_vocab: int, student_vocab: int):
        super().__init__()
        rank = min(teacher_vocab, student_vocab, 512)
        self.down = nn.Linear(teacher_vocab, rank, bias=False)
        self.up = nn.Linear(rank, student_vocab, bias=False)

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Project teacher logits to student vocab space."""
        return self.up(self.down(teacher_logits))
