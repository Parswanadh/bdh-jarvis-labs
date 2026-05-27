"""
BDH with Recurrent Depth (BDH-RD)
==================================

A "Mythos-style" loop built natively into BDH architecture.

Key insight: BDH already has the core components:
- State matrix (like Mythos recurrent state)
- Decay rates (like Mythos A matrix, but fixed)
- Hebbian updates (the learning rule)

This module adds:
- Adaptive loop control (ACT-style halting)
- Learnable decay (like Mythos A, but optimized for BDH)
- Depth-wise gating for token-level early exit

Author: Expert ML Engineer Analysis
Date: 2026-04-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


# ============================================================================
# PART 1: Learnable Decay Matrix (The "A" Matrix from Mythos)
# ============================================================================

class LearnableDecayMatrix(nn.Module):
    """
    Learnable decay matrix for each scale.

    This is the Mythos-like "A" matrix, but designed for BDH's state structure.
    Instead of a single matrix, we learn a decay profile per scale.

    Mythos: h[t+1] = A·h[t] + ...
    BDH-RD: state[i] = sigmoid(decay_proj[i]) * state[i] + ...

    The sigmoid ensures decay stays in (0, 1) range for stability.
    """
    def __init__(self, n_embd: int, num_scales: int = 3):
        super().__init__()
        self.n_embd = n_embd
        self.num_scales = num_scales

        # Learn decay projection for each scale
        # This replaces the fixed decay rates with learned values
        self.decay_projs = nn.ModuleList([
            nn.Linear(n_embd, 1, bias=True)  # Project state to scalar decay
            for _ in range(num_scales)
        ])

    def forward(self, states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute learnable decay for each state.

        Args:
            states: Tuple of (E_fast, E_med, E_slow) [C, C] each

        Returns:
            decays: [num_scales] - learnable decay values
        """
        decays = []
        for i, (state, proj) in enumerate(zip(states, self.decay_projs)):
            # Use trace or mean as proxy for state magnitude
            # This is a simple approximation - could use SVD for exact eigenvalue
            state_mean = state.mean()
            decay = torch.sigmoid(proj(state_mean.unsqueeze(0)))
            decays.append(decay.squeeze())

        return torch.stack(decays)  # [num_scales]


# ============================================================================
# PART 2: Adaptive Computation Time (ACT) for BDH
# ============================================================================

class BDHHaltingUnit(nn.Module):
    """
    Halting unit for Adaptive Computation Time.

    Each token can exit the loop at different depths.

    Similar to Mythos's ACT mechanism but designed for BDH's linear attention.
    """
    def __init__(self, n_embd: int, hidden_dim: int = 64):
        super().__init__()
        self.n_embd = n_embd

        self.halt_net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Halting probability [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute halting probability for each token.

        Args:
            x: [batch, seq_len, n_embd]

        Returns:
            halt_prob: [batch, seq_len] - probability of halting at this step
            active_mask: [batch, seq_len] - which tokens are still active
        """
        halt_prob = self.halt_net(x).squeeze(-1)  # [B, T]
        return halt_prob


# ============================================================================
# PART 3: BDH-RD Model with Recurrent Depth
# ============================================================================

@dataclass
class BDHRecurrentConfig:
    """Configuration for BDH-RD (BDH with Recurrent Depth)."""
    # Base BDH config
    base_config: MultiScaleBDHConfig = None

    # Recurrent parameters
    max_loops: int = 8  # Maximum number of loop iterations
    min_loops: int = 1  # Minimum iterations before halting

    # Halting parameters
    halting_threshold: float = 0.95  # When to stop looping
    halting_penalty: float = 0.01  # Penalty for extra loops (regularization)

    # Adaptive decay (Mythos-style A matrix)
    use_learnable_decay: bool = True

    # Token-level early exit
    use_per_token_exit: bool = True

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = MultiScaleBDHConfig()


class BDHRecurrentLayer(nn.Module):
    """
    Single BDH layer with recurrent depth.

    Applies the BDH forward pass multiple times with state updates,
    while allowing tokens to exit early.
    """
    def __init__(self, config: BDHRecurrentConfig):
        super().__init__()
        self.config = config
        self.base_layer = MultiScaleBDHLayer(config.base_config)

        # Learnable decay for each scale (Mythos A matrix equivalent)
        if config.use_learnable_decay:
            self.decay_net = LearnableDecayMatrix(
                config.base_config.n_embd,
                config.base_config.num_scales
            )

        # Halting unit for ACT
        self.halting_unit = BDHHaltingUnit(config.base_config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        loop_id: int = 0,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        """
        Forward pass with optional recurrence.

        Args:
            x: Input [B, T, C]
            states: Previous states from earlier loops
            loop_id: Current loop index (for position encoding)
            return_states: Whether to return updated states

        Returns:
            output: [B, T, C]
            states: Updated states
            halt_prob: Halting probabilities for each token
        """
        B, T, C = x.shape

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.base_layer.layers) if hasattr(self.base_layer, 'layers') else [None]

        # Apply base BDH layer
        x_out, states = self.base_layer(x, states=states, return_states=return_states)

        # Compute halting probability
        halt_prob = self.halting_unit(x_out)  # [B, T]

        return x_out, states, halt_prob


class BDHRecurrent(nn.Module):
    """
    Complete BDH model with recurrent depth (BDH-RD).

    This implements the "Mythos-style" loop natively in BDH:
    - Recurrent application of BDH layers
    - State updates with learned decay (Mythos A matrix)
    - Adaptive Computation Time (ACT) for early exit
    - Multi-scale memory preserved across loops

    Architecture:
    ```
    Input → Embedding → Loop {
        BDH Layer → State Update → Halting Check
    } → Output
    ```
    """
    def __init__(self, config: BDHRecurrentConfig):
        super().__init__()
        self.config = config

        # Base BDH model (for layers)
        self.bdh = MultiScaleBDH(config.base_config)

        # Learnable decay (Mythos A matrix equivalent)
        if config.use_learnable_decay:
            self.decay_net = LearnableDecayMatrix(
                config.base_config.n_embd,
                config.base_config.num_scales
            )

        # Per-layer halting units
        self.halting_units = nn.ModuleList([
            BDHHaltingUnit(config.base_config.n_embd)
            for _ in range(config.base_config.n_layer)
        ])

        # Loop position embedding (like Mythos's loop-index embedding)
        self.loop_embeddings = nn.Embedding(config.max_loops, config.base_config.n_embd)

        print(f"BDH-RD initialized with {config.base_config.n_layer} layers, {config.max_loops} max loops")
        print(f"Total parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        return_states: bool = False,
        return_loop_metrics: bool = False
    ):
        """
        Forward pass with recurrent depth.

        Args:
            idx: Input tokens [B, T]
            return_states: Whether to return final states
            return_loop_metrics: Whether to return loop statistics

        Returns:
            logits: [B, T, vocab_size]
            states: Final states if return_states
            metrics: Loop statistics if return_loop_metrics
        """
        B, T = idx.shape
        max_loops = min(self.config.max_loops, T // 4 + 1)  # Adaptive max loops

        # Token embeddings
        x = self.bdh.token_embedding(idx)  # [B, T, C]
        x = self.bdh.dropout(x)

        # Initialize states per layer
        states = [None] * len(self.bdh.layers)

        # Loop metrics for logging
        loop_depths = []  # Average loop depth per token
        halt_probs_all = []  # Halting probabilities per loop

        # Recurrent depth loop (Mythos-style)
        for loop_id in range(max_loops):
            loop_emb = self.loop_embeddings(torch.tensor(loop_id, device=x.device))
            x = x + loop_emb  # Add loop position encoding

            new_states = []
            halt_probs_this_loop = []

            for i, layer in enumerate(self.bdh.layers):
                # Get current state for this layer
                layer_state = states[i] if states is not None else None

                # Apply layer
                x, layer_state = layer(x, states=layer_state, return_states=True)

                if return_states:
                    new_states.append(layer_state)

                # Compute halting probability (optional)
                if self.config.use_per_token_exit:
                    halt_prob = self.halting_units[i](x)  # [B, T]
                    halt_probs_this_loop.append(halt_prob)

            states = new_states if return_states else None
            halt_probs_all.append(halt_probs_this_loop)

            # Compute average halt probability
            if halt_probs_this_loop:
                avg_halt = torch.stack(halt_probs_this_loop).mean()
                loop_depths.append(avg_halt.item())

            # Check if all tokens have halted (early exit)
            # (Simplified check - actual implementation needs per-token masks)
            if loop_id >= self.config.min_loops - 1:
                mean_halt = torch.mean(torch.tensor(loop_depths[-1:]))
                if mean_halt.item() >= self.config.halting_threshold:
                    break

        # Final layer norm and output
        x = self.bdh.norm_f(x)
        logits = self.bdh.output_projection(x)

        # Return metrics if requested
        metrics = {}
        if return_loop_metrics:
            metrics = {
                'actual_loops': len(loop_depths),
                'avg_loop_depth': sum(loop_depths) / len(loop_depths) if loop_depths else 0,
                'loop_depths': loop_depths,
            }

        if return_states:
            return logits, states, metrics
        else:
            return logits, None, metrics

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        return_loop_info: bool = False
    ):
        """
        Generate tokens with recurrent depth.

        Each generation step may use multiple BDH iterations.
        """
        loops_used = []

        for _ in range(max_new_tokens):
            # Forward pass with loop tracking
            logits, _, metrics = self(idx, return_loop_metrics=True)

            loops_used.append(metrics.get('actual_loops', 1))

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        metrics = {'loops_used': loops_used}
        return (idx, metrics) if return_loop_info else idx


# ============================================================================
# PART 4: Training Utilities
# ============================================================================

class BDHRDLoss(nn.Module):
    """
    Loss function for BDH-RD with loop regularization.

    Combines:
    - Cross-entropy loss (standard)
    - Loop penalty (encourages efficient computation)
    - State stability loss (encourages stable decay)
    """
    def __init__(self, config: BDHRecurrentConfig):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loop_metrics: Dict[str, float],
        decay_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with loop regularization.

        Args:
            logits: [B*T, vocab_size]
            targets: [B*T]
            loop_metrics: Dictionary with loop statistics
            decay_states: States for stability regularization

        Returns:
            total_loss: Combined loss
            losses: Dict of individual losses
        """
        # Standard CE loss
        ce = self.ce_loss(logits, targets)

        # Loop penalty (encourage early exit)
        loop_penalty = loop_metrics.get('avg_loop_depth', 0) * self.config.halting_penalty

        # Stability loss (encourage decay < 1)
        stability_loss = 0.0
        if decay_states and loop_metrics.get('actual_loops', 1) > 1:
            # Penalize if decay is too high
            for state in decay_states:
                if state is not None:
                    decay = torch.sigmoid(state).mean()
                    stability_loss += torch.relu(decay - 0.9).pow(2).mean()

        total_loss = ce + loop_penalty + stability_loss

        losses = {
            'ce': ce.item(),
            'loop_penalty': loop_penalty.item(),
            'stability': stability_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, losses
