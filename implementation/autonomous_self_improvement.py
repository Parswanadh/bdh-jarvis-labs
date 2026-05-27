"""
Autonomous Self-Improvement Loop for BDH.

This module adds a practical scaffold for:
1) Distillation from multiple world data sources (web + books).
2) Meta-cognitive scoring (confidence/uncertainty) per sample.
3) Safe model update acceptance with rollback on regression.

The design intentionally avoids online scraping side effects inside training.
You provide curated JSONL exports, and this loop performs controlled updates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class SourceWeights:
    """Reliability priors used to weight training targets."""

    web: float = 0.7
    books: float = 0.95
    papers: float = 0.9
    synthetic: float = 0.5

    def get(self, source: str) -> float:
        return getattr(self, source, 0.5)


@dataclass
class SafetyPolicy:
    """Conservative guardrails for autonomous updates."""

    max_gradient_norm: float = 1.0
    max_allowed_val_loss_increase: float = 0.01
    min_confidence_to_learn: float = 0.55


class DistillationCorpus(Dataset):
    """
    JSONL format (one item per line):
    {
      "prompt": "...",
      "teacher_response": "...",
      "source": "web|books|papers|synthetic",
      "evidence": ["https://..."],
      "quality": 0.0-1.0
    }
    """

    def __init__(self, jsonl_path: str, max_seq_len: int = 256):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus file not found: {jsonl_path}")
        self.max_seq_len = max_seq_len
        self.samples: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "prompt" in item and "teacher_response" in item:
                    self.samples.append(item)
        if not self.samples:
            raise ValueError(f"No valid samples in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _to_bytes(self, text: str) -> torch.Tensor:
        data = list(text.encode("utf-8", errors="ignore"))
        if len(data) < 2:
            data = [32, 32]
        data = data[: self.max_seq_len + 1]
        if len(data) < self.max_seq_len + 1:
            data += [32] * (self.max_seq_len + 1 - len(data))
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        combined = f"{item['prompt']}\n{item['teacher_response']}"
        byte_seq = self._to_bytes(combined)
        x = byte_seq[:-1]
        y = byte_seq[1:]
        source = item.get("source", "synthetic")
        quality = float(item.get("quality", 0.5))
        return x, y, source, quality


class MetaCognitionScorer:
    """
    Lightweight meta-cognitive estimates from logits.
    This is not "proof" of correctness, but a calibrated confidence proxy.
    """

    @staticmethod
    def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)  # [B, T]
        max_entropy = math.log(logits.shape[-1])
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        confidence = 1.0 - norm_entropy
        return confidence.mean(dim=-1)  # [B]

    @staticmethod
    def uncertainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return 1.0 - MetaCognitionScorer.confidence_from_logits(logits)


class AutonomousImprover:
    """
    Orchestrates source-aware distillation with meta-cognitive weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        source_weights: Optional[SourceWeights] = None,
        safety_policy: Optional[SafetyPolicy] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.source_weights = source_weights or SourceWeights()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.meta = MetaCognitionScorer()

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        source: List[str],
        quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # token CE per position
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)
        sample_ce = ce.mean(dim=-1)  # [B]

        confidence = self.meta.confidence_from_logits(logits)  # [B]
        uncertainty = 1.0 - confidence

        source_weight = torch.tensor(
            [self.source_weights.get(s) for s in source],
            device=logits.device,
            dtype=logits.dtype,
        )
        quality = quality.to(logits.device, dtype=logits.dtype).clamp(0.0, 1.0)

        # Learn more from uncertain regions but only above confidence floor.
        learn_mask = (confidence >= self.safety_policy.min_confidence_to_learn).float()
        sample_weight = source_weight * (0.5 + quality) * (0.5 + uncertainty) * learn_mask
        sample_weight = sample_weight.clamp_min(1e-6)

        loss = (sample_ce * sample_weight).sum() / sample_weight.sum()
        stats = {
            "mean_confidence": float(confidence.mean().item()),
            "mean_uncertainty": float(uncertainty.mean().item()),
            "active_sample_ratio": float(learn_mask.mean().item()),
            "mean_source_weight": float(source_weight.mean().item()),
        }
        return loss, stats

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for x, y, _source, _quality in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits, _ = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total += float(loss.item())
            n += 1
        return total / max(1, n)

    def improve_one_cycle(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
    ) -> Dict[str, float]:
        baseline = self.evaluate(val_loader)
        best_snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.model.train()
        last_stats: Dict[str, float] = {}
        for _ in range(epochs):
            for x, y, source, quality in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, _ = self.model(x)
                loss, stats = self._compute_weighted_loss(logits, y, list(source), quality)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.safety_policy.max_gradient_norm)
                optimizer.step()
                last_stats = stats

        new_val = self.evaluate(val_loader)
        delta = new_val - baseline

        # Safety gate: rollback if validation regresses too much.
        accepted = delta <= self.safety_policy.max_allowed_val_loss_increase
        if not accepted:
            self.model.load_state_dict(best_snapshot)

        result = {
            "baseline_val_loss": baseline,
            "new_val_loss": new_val,
            "val_loss_delta": delta,
            "accepted": float(1.0 if accepted else 0.0),
        }
        result.update(last_stats)
        return result


def build_loaders(
    train_jsonl: str,
    val_jsonl: str,
    batch_size: int = 8,
    max_seq_len: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DistillationCorpus(train_jsonl, max_seq_len=max_seq_len)
    val_ds = DistillationCorpus(val_jsonl, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
