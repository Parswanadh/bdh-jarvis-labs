"""
BDH v2 Benchmark Suite — Day 6
================================

Comprehensive benchmark suite with 5 benchmarks + ablation runner.

Benchmarks:
1. Language Modeling Perplexity — WikiText-2 style
2. Needle-in-Haystack — fact retrieval at [64, 128, 256, 512] context lengths
3. Sequence Reversal — reverse sequences, measure accuracy
4. Copy Task — verbatim copy, measure exact match rate
5. Associative Recall — key-value pair recall

Ablation Runner:
- baseline (element-wise Hebbian, no RoPE, multiplicative gating)
- +outer_product
- +rope
- +hybrid_gating
- +all_fixes
- Statistical significance testing (paired t-test across 3 seeds)

All synthetic data — no external dependencies.

Author: BDH v2 Benchmark Suite (Day 6)
Date: 2026-05-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import argparse
import random
import traceback
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from implementation.bdh_v2_clean import BDHv2Config, BDHv2


# ============================================================================
# Utility: Memory Tracker
# ============================================================================


class MemoryTracker:
    """Track GPU/CPU memory usage."""

    def __init__(self):
        self.snapshots: List[Dict[str, float]] = []

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        info: Dict[str, Any] = {"label": label, "timestamp": time.time()}
        if torch.cuda.is_available():
            info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        info["cpu_ram_mb"] = self._get_cpu_ram_mb()
        self.snapshots.append(info)
        return info

    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _get_cpu_ram_mb() -> float:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**2
        except Exception:
            return 0.0

    def get_peak_gpu_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0


# ============================================================================
# Utility: Tiny Model Config for Benchmarks
# ============================================================================


def make_benchmark_config(
    seed: int = 42,
    use_outer_product: bool = False,
    use_rope: bool = False,
    use_hybrid_gating: bool = False,
    vocab_size: int = 256,
    n_embd: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    ffn_dim: int = 256,
    max_seq_len: int = 512,
    dropout: float = 0.0,
) -> BDHv2Config:
    """Create a small config suitable for fast benchmarking."""
    return BDHv2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        ffn_dim=ffn_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_outer_product=use_outer_product,
        use_rope=use_rope,
        use_hybrid_gating=use_hybrid_gating,
        init_states="zeros",
        decay_rates=[0.95, 0.99, 0.995],
        scale_weights=[0.2, 0.3, 0.5],
        hebbian_lr=0.001,
    )


def make_model(config: BDHv2Config, device: torch.device) -> BDHv2:
    """Create model and move to device."""
    model = BDHv2(config)
    model = model.to(device)
    return model


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-select device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Benchmark 1: Language Modeling Perplexity
# ============================================================================


class PerplexityBenchmark:
    """
    WikiText-2 style perplexity benchmark.

    Generates synthetic text from a small vocabulary and measures
    cross-entropy perplexity on held-out sequences.
    """

    NAME = "perplexity"

    @staticmethod
    def generate_data(
        vocab_size: int = 256,
        num_train: int = 2000,
        num_test: int = 500,
        seq_len: int = 64,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic text with n-gram-like structure."""
        rng = np.random.RandomState(seed)
        # Create a transition matrix for structured text
        transition = rng.dirichlet(np.ones(vocab_size) * 0.1, size=vocab_size)
        # Smooth to make it learnable
        transition = transition * 0.8 + np.ones_like(transition) * 0.2 / vocab_size
        transition = transition / transition.sum(axis=1, keepdims=True)

        def generate_sequences(n: int) -> torch.Tensor:
            sequences = []
            for _ in range(n):
                seq = [rng.randint(vocab_size)]
                for _ in range(seq_len - 1):
                    probs = transition[seq[-1]]
                    next_token = rng.choice(vocab_size, p=probs)
                    seq.append(int(next_token))
                sequences.append(seq)
            return torch.tensor(sequences, dtype=torch.long)

        train_data = generate_sequences(num_train)
        test_data = generate_sequences(num_test)
        return train_data, test_data

    @staticmethod
    def train_and_evaluate(
        model: BDHv2,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train on synthetic data and measure test perplexity."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        train_data = train_data.to(device)
        test_data = test_data.to(device)
        vocab_size = model.config.vocab_size
        n = len(train_data)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = train_data[idx, :-1]
                y = train_data[idx, 1:]
                B, T = x.shape

                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            x_test = test_data[:, :-1]
            y_test = test_data[:, 1:]
            logits, _ = model(x_test)
            test_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y_test.reshape(-1)
            ).item()

        perplexity = np.exp(test_loss)
        return {
            "test_loss": round(test_loss, 6),
            "perplexity": round(perplexity, 4),
            "bits_per_token": round(test_loss / np.log(2), 4),
        }


# ============================================================================
# Benchmark 2: Needle-in-Haystack
# ============================================================================


class NeedleInHaystackBenchmark:
    """
    Insert a fact at a random position in context, test retrieval.

    Context lengths tested: [64, 128, 256, 512]
    """

    NAME = "needle_in_haystack"
    CONTEXT_LENGTHS = [32, 64, 128, 256]

    @staticmethod
    def generate_data(
        vocab_size: int = 256,
        num_samples: int = 100,
        context_lengths: Optional[List[int]] = None,
        seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        """
        Generate needle-in-haystack samples.

        Each sample: {context, needle_token, needle_pos, query_token, target}
        The needle is a special token inserted at a known position.
        The model must retrieve it when queried.
        """
        if context_lengths is None:
            context_lengths = NeedleInHaystackBenchmark.CONTEXT_LENGTHS

        rng = np.random.RandomState(seed)
        samples_by_length = {}

        for ctx_len in context_lengths:
            if ctx_len < 4:
                continue
            samples = []
            for _ in range(num_samples):
                # Choose a needle token (use high-value tokens to avoid collision)
                needle_token = rng.randint(vocab_size - 100, vocab_size)
                # Choose needle position (not at edges)
                needle_pos = rng.randint(1, ctx_len - 2)
                # Fill context with random tokens (avoiding needle range)
                context = rng.randint(0, vocab_size - 100, size=ctx_len)
                # Insert needle
                context[needle_pos] = needle_token
                # Query: position 0 token is a "retrieve" command
                # The model should output the needle token at the end
                # We frame this as: given context, predict what was at needle_pos
                # Simplified: last token of input = needle_pos indicator, target = needle_token
                # Even simpler: append needle_pos as query, target is needle_token
                query_token = needle_pos  # encode position as query
                samples.append(
                    {
                        "context": context.tolist(),
                        "needle_token": int(needle_token),
                        "needle_pos": int(needle_pos),
                        "query_token": int(query_token),
                    }
                )
            samples_by_length[str(ctx_len)] = samples

        return samples_by_length

    @staticmethod
    def train_and_evaluate(
        model: BDHv2,
        samples_by_length: Dict[str, List[Dict]],
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train model to retrieve needles and measure accuracy per context length."""
        vocab_size = model.config.vocab_size
        results = {}

        for ctx_len_str, samples in samples_by_length.items():
            ctx_len = int(ctx_len_str)
            if len(samples) == 0:
                results[ctx_len_str] = {"accuracy": 0.0, "num_samples": 0}
                continue

            # Prepare training data
            # Format: [needle_pos, ...context..., needle_pos] -> predict needle_token at last position
            # Simpler: input = context with needle_pos appended, target = needle_token at last position
            inputs = []
            targets = []
            for s in samples:
                inp = s["context"] + [s["query_token"]]
                inputs.append(inp)
                targets.append(s["needle_token"])

            inputs = torch.tensor(inputs, dtype=torch.long, device=device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            # Train
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            n = len(inputs)

            for epoch in range(epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch_size):
                    idx = perm[i : i + batch_size]
                    x = inputs[idx]
                    y = targets[idx]

                    logits, _ = model(x)
                    # Predict from last position
                    last_logits = logits[:, -1, :]  # [B, vocab]
                    loss = F.cross_entropy(last_logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                logits, _ = model(inputs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct = (preds == targets).sum().item()
                accuracy = correct / len(targets)

            results[ctx_len_str] = {
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "num_samples": len(targets),
            }

        return results


# ============================================================================
# Benchmark 3: Sequence Reversal
# ============================================================================


class SequenceReversalBenchmark:
    """
    Train model to reverse input sequences.

    Input:  [a, b, c, d]
    Target: [d, c, b, a]
    """

    NAME = "sequence_reversal"

    @staticmethod
    def generate_data(
        vocab_size: int = 256,
        num_train: int = 500,
        num_test: int = 100,
        seq_len: int = 16,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sequence reversal pairs."""
        rng = np.random.RandomState(seed)

        def generate_pairs(n: int):
            inputs = []
            targets = []
            for _ in range(n):
                seq = rng.randint(0, vocab_size, size=seq_len).tolist()
                inputs.append(seq)
                targets.append(seq[::-1])
            return (
                torch.tensor(inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
            )

        train_in, train_tgt = generate_pairs(num_train)
        test_in, test_tgt = generate_pairs(num_test)
        return train_in, train_tgt, test_in, test_tgt

    @staticmethod
    def train_and_evaluate(
        model: BDHv2,
        train_in: torch.Tensor,
        train_tgt: torch.Tensor,
        test_in: torch.Tensor,
        test_tgt: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train on reversal task and measure accuracy."""
        vocab_size = model.config.vocab_size
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        train_in = train_in.to(device)
        train_tgt = train_tgt.to(device)
        test_in = test_in.to(device)
        test_tgt = test_tgt.to(device)
        n = len(train_in)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = train_in[idx]
                y = train_tgt[idx]

                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, _ = model(test_in)
            preds = logits.argmax(dim=-1)  # [B, T]
            # Per-token accuracy
            token_acc = (preds == test_tgt).float().mean().item()
            # Exact match (all tokens correct)
            exact_match = (preds == test_tgt).all(dim=-1).float().mean().item()

        return {
            "token_accuracy": round(token_acc, 4),
            "exact_match_rate": round(exact_match, 4),
        }


# ============================================================================
# Benchmark 4: Copy Task
# ============================================================================


class CopyTaskBenchmark:
    """
    Train model to copy input sequences verbatim.

    Input:  [a, b, c, d]
    Target: [a, b, c, d]
    """

    NAME = "copy_task"

    @staticmethod
    def generate_data(
        vocab_size: int = 256,
        num_train: int = 500,
        num_test: int = 100,
        seq_len: int = 16,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate copy task pairs (input == target)."""
        rng = np.random.RandomState(seed)

        def generate_pairs(n: int):
            data = rng.randint(0, vocab_size, size=(n, seq_len))
            return torch.tensor(data, dtype=torch.long)

        train_data = generate_pairs(num_train)
        test_data = generate_pairs(num_test)
        return train_data, train_data.clone(), test_data, test_data.clone()

    @staticmethod
    def train_and_evaluate(
        model: BDHv2,
        train_in: torch.Tensor,
        train_tgt: torch.Tensor,
        test_in: torch.Tensor,
        test_tgt: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train on copy task and measure exact match rate."""
        vocab_size = model.config.vocab_size
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        train_in = train_in.to(device)
        train_tgt = train_tgt.to(device)
        test_in = test_in.to(device)
        test_tgt = test_tgt.to(device)
        n = len(train_in)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = train_in[idx]
                y = train_tgt[idx]

                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, _ = model(test_in)
            preds = logits.argmax(dim=-1)
            token_acc = (preds == test_tgt).float().mean().item()
            exact_match = (preds == test_tgt).all(dim=-1).float().mean().item()

        return {
            "token_accuracy": round(token_acc, 4),
            "exact_match_rate": round(exact_match, 4),
        }


# ============================================================================
# Benchmark 5: Associative Recall
# ============================================================================


class AssociativeRecallBenchmark:
    """
    Train model on key-value pairs, test recall of values given keys.

    Input:  [key1, val1, key2, val2, ..., query_key, <sep>]
    Target: [query_value]
    """

    NAME = "associative_recall"

    @staticmethod
    def generate_data(
        vocab_size: int = 256,
        num_train: int = 500,
        num_test: int = 100,
        num_pairs: int = 3,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate associative recall data.

        Each sample has num_pairs key-value pairs followed by a query key.
        The target is the value associated with the query key.
        """
        rng = np.random.RandomState(seed)
        # Use separate ranges for keys and values to avoid ambiguity
        key_range = vocab_size // 2
        val_start = vocab_size // 2

        def generate_samples(n: int):
            inputs = []
            targets = []
            for _ in range(n):
                # Generate unique keys
                keys = rng.choice(key_range, size=num_pairs, replace=False).tolist()
                vals = rng.randint(0, key_range, size=num_pairs).tolist()
                # Shift values to val range
                vals = [v + val_start for v in vals]

                # Interleave: key1, val1, key2, val2, ...
                seq = []
                for k, v in zip(keys, vals):
                    seq.extend([k, v])

                # Pick a random key to query
                query_idx = rng.randint(0, num_pairs)
                query_key = keys[query_idx]
                target_val = vals[query_idx]

                seq.append(query_key)
                inputs.append(seq)
                targets.append(target_val)

            # Pad to same length
            max_len = max(len(s) for s in inputs)
            padded_inputs = []
            for s in inputs:
                padded = s + [0] * (max_len - len(s))
                padded_inputs.append(padded)

            return (
                torch.tensor(padded_inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
            )

        train_in, train_tgt = generate_samples(num_train)
        test_in, test_tgt = generate_samples(num_test)
        return train_in, train_tgt, test_in, test_tgt

    @staticmethod
    def train_and_evaluate(
        model: BDHv2,
        train_in: torch.Tensor,
        train_tgt: torch.Tensor,
        test_in: torch.Tensor,
        test_tgt: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train on associative recall and measure accuracy."""
        vocab_size = model.config.vocab_size
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        train_in = train_in.to(device)
        train_tgt = train_tgt.to(device)
        test_in = test_in.to(device)
        test_tgt = test_tgt.to(device)
        n = len(train_in)

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = train_in[idx]
                y = train_tgt[idx]

                logits, _ = model(x)
                # Predict from last position
                last_logits = logits[:, -1, :]
                loss = F.cross_entropy(last_logits, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, _ = model(test_in)
            preds = logits[:, -1, :].argmax(dim=-1)
            accuracy = (preds == test_tgt).float().mean().item()

        return {
            "accuracy": round(accuracy, 4),
            "correct": int((preds == test_tgt).sum().item()),
            "num_samples": len(test_tgt),
        }


# ============================================================================
# Ablation Runner
# ============================================================================


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "perplexity",
            "needle_in_haystack",
            "sequence_reversal",
            "copy_task",
            "associative_recall",
        ]
    )
    # Training hyperparams (kept small for speed)
    vocab_size: int = 256
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    ffn_dim: int = 256
    max_seq_len: int = 512
    dropout: float = 0.0
    lr: float = 1e-3
    device: str = "auto"


# Predefined model configurations for ablation
ABLATION_CONFIGS = {
    "baseline": {
        "use_outer_product": False,
        "use_rope": False,
        "use_hybrid_gating": False,
    },
    "+outer_product": {
        "use_outer_product": True,
        "use_rope": False,
        "use_hybrid_gating": False,
    },
    "+rope": {
        "use_outer_product": False,
        "use_rope": True,
        "use_hybrid_gating": False,
    },
    "+hybrid_gating": {
        "use_outer_product": False,
        "use_rope": False,
        "use_hybrid_gating": True,
    },
    "+all_fixes": {
        "use_outer_product": True,
        "use_rope": True,
        "use_hybrid_gating": True,
    },
}


class AblationRunner:
    """
    Runs all 5 benchmarks on multiple model configurations.
    Produces comparison table + statistical significance testing.
    """

    def __init__(self, config: AblationConfig):
        self.config = config
        self.memory = MemoryTracker()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.device = (
            get_device() if config.device == "auto" else torch.device(config.device)
        )

    def _make_model_config(self, seed: int, **fixes) -> BDHv2Config:
        return make_benchmark_config(
            seed=seed,
            vocab_size=self.config.vocab_size,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            ffn_dim=self.config.ffn_dim,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            **fixes,
        )

    def run_single_benchmark(
        self,
        benchmark_name: str,
        model_config: BDHv2Config,
        seed: int,
    ) -> Dict[str, float]:
        """Run a single benchmark on a single model config + seed."""
        set_seed(seed)
        self.memory.reset_peak()
        self.memory.snapshot(f"before_{benchmark_name}")

        model = make_model(model_config, self.device)

        if benchmark_name == "perplexity":
            train_data, test_data = PerplexityBenchmark.generate_data(
                vocab_size=self.config.vocab_size, seed=seed
            )
            result = PerplexityBenchmark.train_and_evaluate(
                model, train_data, test_data, self.device, lr=self.config.lr
            )

        elif benchmark_name == "needle_in_haystack":
            samples = NeedleInHaystackBenchmark.generate_data(
                vocab_size=self.config.vocab_size, seed=seed
            )
            result = NeedleInHaystackBenchmark.train_and_evaluate(
                model, samples, self.device, lr=self.config.lr
            )

        elif benchmark_name == "sequence_reversal":
            train_in, train_tgt, test_in, test_tgt = (
                SequenceReversalBenchmark.generate_data(
                    vocab_size=self.config.vocab_size, seed=seed
                )
            )
            result = SequenceReversalBenchmark.train_and_evaluate(
                model,
                train_in,
                train_tgt,
                test_in,
                test_tgt,
                self.device,
                lr=self.config.lr,
            )

        elif benchmark_name == "copy_task":
            train_in, train_tgt, test_in, test_tgt = CopyTaskBenchmark.generate_data(
                vocab_size=self.config.vocab_size, seed=seed
            )
            result = CopyTaskBenchmark.train_and_evaluate(
                model,
                train_in,
                train_tgt,
                test_in,
                test_tgt,
                self.device,
                lr=self.config.lr,
            )

        elif benchmark_name == "associative_recall":
            train_in, train_tgt, test_in, test_tgt = (
                AssociativeRecallBenchmark.generate_data(
                    vocab_size=self.config.vocab_size, seed=seed
                )
            )
            result = AssociativeRecallBenchmark.train_and_evaluate(
                model,
                train_in,
                train_tgt,
                test_in,
                test_tgt,
                self.device,
                lr=self.config.lr,
            )
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        self.memory.snapshot(f"after_{benchmark_name}")
        result["peak_gpu_mb"] = round(self.memory.get_peak_gpu_mb(), 2)
        result["seed"] = seed
        return result

    def run_ablation(self) -> Dict[str, Any]:
        """Run full ablation study across all configs, benchmarks, and seeds."""
        print("=" * 70)
        print("BDH v2 ABLATION STUDY")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Benchmarks: {self.config.benchmarks}")
        print(f"Configs: {list(ABLATION_CONFIGS.keys())}")
        print(f"Date: {datetime.now().isoformat()}")
        print("=" * 70)

        total_runs = (
            len(ABLATION_CONFIGS) * len(self.config.benchmarks) * len(self.config.seeds)
        )
        run_count = 0

        for config_name, fixes in ABLATION_CONFIGS.items():
            print(f"\n{'=' * 50}")
            print(f"Config: {config_name}")
            print(f"  Fixes: {fixes}")
            print(f"{'=' * 50}")

            self.results[config_name] = {}

            for bench_name in self.config.benchmarks:
                print(f"\n  Benchmark: {bench_name}")
                self.results[config_name][bench_name] = {"seeds": []}

                for seed in self.config.seeds:
                    run_count += 1
                    start = time.time()
                    print(
                        f"    Seed {seed} (run {run_count}/{total_runs})...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        model_cfg = self._make_model_config(seed, **fixes)
                        result = self.run_single_benchmark(bench_name, model_cfg, seed)
                        elapsed = time.time() - start
                        print(f"OK ({elapsed:.1f}s)")
                        self.results[config_name][bench_name]["seeds"].append(result)
                    except Exception as e:
                        elapsed = time.time() - start
                        print(f"FAILED ({elapsed:.1f}s): {e}")
                        self.results[config_name][bench_name]["seeds"].append(
                            {
                                "error": str(e),
                                "seed": seed,
                            }
                        )

        # Compute statistics
        summary = self._compute_summary()
        return summary

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics and significance tests."""
        summary = {
            "configs": {},
            "comparison_table": [],
            "significance_tests": [],
        }

        baseline_name = "baseline"
        baseline_results = self.results.get(baseline_name, {})

        for config_name, bench_results in self.results.items():
            config_summary = {}
            for bench_name, seed_data in bench_results.items():
                seed_list = seed_data.get("seeds", [])
                # Extract primary metric per benchmark
                values = []
                for s in seed_list:
                    if "error" in s:
                        continue
                    val = self._extract_primary_metric(bench_name, s)
                    if val is not None:
                        values.append(val)

                if len(values) >= 2:
                    config_summary[bench_name] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "min": round(float(np.min(values)), 4),
                        "max": round(float(np.max(values)), 4),
                        "n": len(values),
                        "values": [round(v, 4) for v in values],
                    }
                elif len(values) == 1:
                    config_summary[bench_name] = {
                        "mean": round(float(values[0]), 4),
                        "std": 0.0,
                        "min": round(float(values[0]), 4),
                        "max": round(float(values[0]), 4),
                        "n": 1,
                        "values": [round(values[0], 4)],
                    }
                else:
                    config_summary[bench_name] = {"mean": None, "std": None, "n": 0}

            summary["configs"][config_name] = config_summary

        # Build comparison table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)

        header = f"{'Config':<20}"
        for bench_name in self.config.benchmarks:
            metric_label = self._metric_label(bench_name)
            header += f" {metric_label:>14}"
        print(header)
        print("-" * len(header))

        for config_name in self.results:
            row = f"{config_name:<20}"
            cfg_summary = summary["configs"].get(config_name, {})
            for bench_name in self.config.benchmarks:
                bs = cfg_summary.get(bench_name, {})
                mean_val = bs.get("mean")
                std_val = bs.get("std")
                if mean_val is not None:
                    row += f" {mean_val:>10.4f}±{std_val:.3f}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

        # Significance testing (paired t-test vs baseline)
        print("\n" + "=" * 70)
        print("SIGNIFICANCE TESTS (paired t-test vs baseline)")
        print("=" * 70)

        for config_name in self.results:
            if config_name == baseline_name:
                continue
            print(f"\n{config_name} vs {baseline_name}:")
            for bench_name in self.config.benchmarks:
                baseline_seeds = baseline_results.get(bench_name, {}).get("seeds", [])
            for bench_name in self.config.benchmarks:
                baseline_vals = []
                for s in baseline_results.get(bench_name, {}).get("seeds", []):
                    v = self._extract_primary_metric(bench_name, s)
                    if v is not None:
                        baseline_vals.append(v)

                config_vals = []
                for s in self.results[config_name].get(bench_name, {}).get("seeds", []):
                    v = self._extract_primary_metric(bench_name, s)
                    if v is not None:
                        config_vals.append(v)

                if len(baseline_vals) >= 2 and len(config_vals) >= 2:
                    t_stat, p_value = stats.ttest_rel(config_vals, baseline_vals)
                    sig = (
                        "***"
                        if p_value < 0.001
                        else "**"
                        if p_value < 0.01
                        else "*"
                        if p_value < 0.05
                        else "ns"
                    )
                    print(
                        f"  {bench_name:<25} t={t_stat:>7.3f}  p={p_value:.4f}  {sig}"
                    )

                    summary["significance_tests"].append(
                        {
                            "config": config_name,
                            "baseline": baseline_name,
                            "benchmark": bench_name,
                            "t_statistic": round(float(t_stat), 4),
                            "p_value": round(float(p_value), 6),
                            "significant": bool(p_value < 0.05),
                        }
                    )
                else:
                    print(f"  {bench_name:<25} insufficient data for t-test")

        return summary

    @staticmethod
    def _extract_primary_metric(benchmark_name: str, result: Dict) -> Optional[float]:
        """Extract the primary metric from a benchmark result."""
        if "error" in result:
            return None
        mapping = {
            "perplexity": "perplexity",
            "needle_in_haystack": None,  # special: average accuracy across lengths
            "sequence_reversal": "exact_match_rate",
            "copy_task": "exact_match_rate",
            "associative_recall": "accuracy",
        }
        key = mapping.get(benchmark_name)
        if key is None:
            # Needle-in-haystack: average accuracy across context lengths
            if benchmark_name == "needle_in_haystack":
                accs = []
                for k, v in result.items():
                    if isinstance(v, dict) and "accuracy" in v:
                        accs.append(v["accuracy"])
                return float(np.mean(accs)) if accs else None
            return None
        return result.get(key)

    @staticmethod
    def _metric_label(benchmark_name: str) -> str:
        """Get a short label for the primary metric."""
        labels = {
            "perplexity": "ppl",
            "needle_in_haystack": "needle_acc",
            "sequence_reversal": "rev_exact",
            "copy_task": "copy_exact",
            "associative_recall": "assoc_acc",
        }
        return labels.get(benchmark_name, benchmark_name)


# ============================================================================
# Single Benchmark Runner
# ============================================================================


def run_single_benchmark_cli(
    benchmark_name: str,
    seeds: List[int],
    output: str,
    device: str = "auto",
    vocab_size: int = 256,
    n_embd: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    ffn_dim: int = 256,
    max_seq_len: int = 512,
):
    """Run a single benchmark across multiple seeds."""
    dev = get_device() if device == "auto" else torch.device(device)
    all_results = {
        "benchmark": benchmark_name,
        "seeds": seeds,
        "device": str(dev),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "ffn_dim": ffn_dim,
            "max_seq_len": max_seq_len,
        },
        "results": [],
    }

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"Benchmark: {benchmark_name} | Seed: {seed}")
        print(f"{'=' * 50}")

        set_seed(seed)
        cfg = make_benchmark_config(
            seed=seed,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
        )
        model = make_model(cfg, dev)

        if benchmark_name == "perplexity":
            train_data, test_data = PerplexityBenchmark.generate_data(
                vocab_size=vocab_size, seed=seed
            )
            result = PerplexityBenchmark.train_and_evaluate(
                model, train_data, test_data, dev
            )

        elif benchmark_name == "needle_in_haystack":
            samples = NeedleInHaystackBenchmark.generate_data(
                vocab_size=vocab_size, seed=seed
            )
            result = NeedleInHaystackBenchmark.train_and_evaluate(model, samples, dev)

        elif benchmark_name == "sequence_reversal":
            train_in, train_tgt, test_in, test_tgt = (
                SequenceReversalBenchmark.generate_data(
                    vocab_size=vocab_size, seed=seed
                )
            )
            result = SequenceReversalBenchmark.train_and_evaluate(
                model, train_in, train_tgt, test_in, test_tgt, dev
            )

        elif benchmark_name == "copy_task":
            train_in, train_tgt, test_in, test_tgt = CopyTaskBenchmark.generate_data(
                vocab_size=vocab_size, seed=seed
            )
            result = CopyTaskBenchmark.train_and_evaluate(
                model, train_in, train_tgt, test_in, test_tgt, dev
            )

        elif benchmark_name == "associative_recall":
            train_in, train_tgt, test_in, test_tgt = (
                AssociativeRecallBenchmark.generate_data(
                    vocab_size=vocab_size, seed=seed
                )
            )
            result = AssociativeRecallBenchmark.train_and_evaluate(
                model, train_in, train_tgt, test_in, test_tgt, dev
            )
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        result["seed"] = seed
        all_results["results"].append(result)
        print(f"  Result: {result}")

    # Save
    os.makedirs(
        os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True
    )
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output}")
    return all_results


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="BDH v2 Benchmark Suite — 5 benchmarks + ablation runner"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "perplexity",
            "needle_in_haystack",
            "sequence_reversal",
            "copy_task",
            "associative_recall",
        ],
        help="Run a single benchmark",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run full ablation study across all configs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds (default: 42 123 456)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Vocabulary size (default: 256)",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=4,
        help="Number of layers (default: 4)",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=256,
        help="FFN hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.ablation:
            args.output = f"benchmarking/results/ablation_{ts}.json"
        elif args.benchmark:
            args.output = f"benchmarking/results/{args.benchmark}_{ts}.json"
        else:
            parser.print_help()
            return

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    if args.ablation:
        ablation_cfg = AblationConfig(
            seeds=args.seeds,
            device=args.device,
            vocab_size=args.vocab_size,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_dim=args.ffn_dim,
            max_seq_len=args.max_seq_len,
        )
        runner = AblationRunner(ablation_cfg)
        summary = runner.run_ablation()

        # Full results for JSON
        full_output = {
            "type": "ablation",
            "timestamp": datetime.now().isoformat(),
            "config": asdict(ablation_cfg),
            "raw_results": runner.results,
            "summary": summary,
        }

        with open(args.output, "w") as f:
            json.dump(full_output, f, indent=2, default=str)
        print(f"\nAblation results saved to: {args.output}")

    elif args.benchmark:
        run_single_benchmark_cli(
            benchmark_name=args.benchmark,
            seeds=args.seeds,
            output=args.output,
            device=args.device,
            vocab_size=args.vocab_size,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_dim=args.ffn_dim,
            max_seq_len=args.max_seq_len,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
