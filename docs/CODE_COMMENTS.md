# BDH Code Documentation Standards

## Overview

This document defines the code documentation standards for the BDH Science Fest project. All code changes must follow these standards to ensure clarity, maintainability, and scientific rigor.

---

## Documentation Principles

### Core Philosophy
1. **Explain WHY, not just WHAT** - Why was this design choice made?
2. **Assume intelligent but novice reader** - Reader knows ML/PyTorch but not BDH specifics
3. **Keep comments current** - Comments must match code behavior
4. **Be concise but complete** - Enough detail, no fluff
5. **Include examples** - Show, don't just tell

---

## File-Level Documentation

### Header Template

Every Python file should start with:

```python
"""
BDH [Component Name]
====================

[One-line description of what this file does]

[Detailed paragraph explaining the purpose, motivation, and key design decisions]

Key Features:
- [Feature 1]: [Brief explanation]
- [Feature 2]: [Brief explanation]

Usage:
    [Simple code example showing usage]

Author: [Author name/role]
Date: [Date]
References: [Papers, URLs, etc.]
"""
```

### Example

```python
"""
Multi-Scale BDH Architecture
============================

Implementation of BDH with multi-scale state matrices for extended working memory.

The key innovation is using multiple state matrices with different decay rates,
inspired by multi-timescale plasticity in the brain (STP, LTP, structural changes).

Key Features:
- Multi-scale state matrices: [0.95, 0.99, 0.995] for fast/medium/slow timescales
- Backward compatible with original BDH API
- Supports both byte-level and BBPE tokenization

Usage:
    config = MultiScaleBDHConfig(
        decay_rates=[0.95, 0.99, 0.995],
        n_embd=256,
        n_layer=6
    )
    model = MultiScaleBDH(config)
    output = model(input_tokens)

Author: T1 - Multi-Scale Architect
Date: February 25, 2026
References:
    - BDH Paper: arXiv:2509.26507
    - Kording et al. (2001): Multi-timescale learning
"""
```

---

## Class-Level Documentation

### Template

```python
class ClassName:
    """
    [One-line summary of class purpose]

    [Detailed paragraph explaining what the class does and why it exists]

    [Additional paragraphs as needed for complex classes]

    Args:
        param1 (type): Description of parameter
        param2 (type, optional): Description with default value

    Attributes:
        attr1 (type): Description of important attribute
        attr2 (type): Description of important attribute

    Raises:
        ErrorType: When and why this error occurs

    Example:
        >>> [Simple interactive example]
        >>> result = ClassName(param1=value)
        >>> print(result.method())
    """
```

### Example

```python
class MultiScaleLinearAttention(nn.Module):
    """
    Multi-scale linear attention for BDH.

    Implements Hebbian learning at multiple timescales, inspired by
    fast/medium/slow synaptic plasticity in the brain. Unlike standard
    attention which uses a single KV cache, this maintains separate state
    matrices for different decay rates.

    The multi-scale approach extends effective memory from ~500 tokens to
    ~2000 tokens by combining fast, medium, and slow timescales.

    Args:
        config (MultiScaleBDHConfig): Configuration object with:
            - n_embd: Embedding dimension
            - decay_rates: List of decay rates (default: [0.95, 0.99, 0.995])
            - hebbian_lr: Learning rate for state updates

    Attributes:
        decay_rates (List[float]): Decay rates for each timescale
        states (List[torch.Tensor]): State matrices [n_scales, n_embd, n_embd]
        Wq, Wk, Wv, Wo (nn.Linear): Projection layers

    Raises:
        ValueError: If decay_rates is empty or contains invalid values

    Example:
        >>> config = MultiScaleBDHConfig(
        ...     n_embd=256,
        ...     decay_rates=[0.95, 0.99, 0.995]
        ... )
        >>> attn = MultiScaleLinearAttention(config)
        >>> x = torch.randn(2, 10, 256)  # [batch, seq, dim]
        >>> output, states = attn(x, return_state=True)
        >>> print(output.shape)  # torch.Size([2, 10, 256])
    """

    def __init__(self, config: MultiScaleBDHConfig):
        # Implementation...
        pass
```

---

## Method-Level Documentation

### Template

```python
def method_name(self, param1: type, param2: type = default) -> return_type:
    """
    [One-line summary of what method does]

    [Detailed explanation if needed for complex logic]

    Args:
        param1 (type): Description
        param2 (type, optional): Description

    Returns:
        return_type: Description of return value

    Raises:
        ErrorType: When/why error occurs

    Note:
        [Important details or caveats]

    Example:
        >>> [Simple usage example]
    """
```

### Example

```python
def forward(
    self,
    x: torch.Tensor,
    return_state: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Forward pass with multi-scale state update.

    Processes input through linear attention while updating all state matrices
    using Hebbian learning. Each state matrix captures different timescales.

    Args:
        x (torch.Tensor): Input tensor [batch, seq_len, n_embd]
        return_state (bool): Whether to return updated states

    Returns:
        torch.Tensor: Output tensor [batch, seq_len, n_embd]
        List[torch.Tensor]: State matrices (if return_state=True)

    Note:
        State matrices are updated in-place using Hebbian rule:
        E_new = decay * E_old + lr * (Q ⊗ V)

        This is the core learning mechanism inspired by:
        "Neurons that fire together, wire together" - Donald Hebb

    Example:
        >>> attn = MultiScaleLinearAttention(config)
        >>> x = torch.randn(2, 10, 256)
        >>> output = attn(x)
        >>> output, states = attn(x, return_state=True)
        >>> print(len(states))  # 3 (one per decay rate)
    """
```

---

## Inline Comments

### When to Use Inline Comments

**DO use inline comments for:**
- Non-obvious algorithms or calculations
- Biological inspiration for design choices
- References to papers or equations
- Warnings about edge cases
- Optimization explanations

**DON'T use for:**
- Obvious code (e.g., `i += 1  # increment i`)
- Redundant information
- Outdated explanations

### Examples

**Good:**
```python
# Hebbian state update: "Neurons that fire together, wire together"
# E_new = decay * E_old + lr * (Q ⊗ V)
# Reference: Hebb (1949), Kording et al. (2001)
for i, decay in enumerate(self.decay_rates):
    # Outer product captures co-activation patterns
    hebbian_update = torch.einsum('bti,bti->ij', Q, V)

    # Decay old state, add new learning
    # Decay factor 0.95 corresponds to ~50 token half-life
    # Decay factor 0.99 corresponds to ~500 token half-life
    # Decay factor 0.995 corresponds to ~2000 token half-life
    self.states[i] = decay * self.states[i] + self.config.hebbian_lr * hebbian_update
```

**Bad:**
```python
# Loop through decay rates
for i, decay in enumerate(self.decay_rates):
    # Calculate outer product
    hebbian_update = torch.einsum('bti,bti->ij', Q, V)
    # Update state
    self.states[i] = decay * self.states[i] + self.config.hebbian_lr * hebbian_update
```

---

## Configuration Documentation

### Dataclass Documentation

```python
@dataclass
class MultiScaleBDHConfig:
    """
    Configuration for Multi-Scale BDH model.

    This configuration extends the base BDH config with multi-scale memory
    support. All parameters are backward compatible with original BDH.

    Attributes:
        # ===== Architecture =====
        vocab_size (int): Vocabulary size. Use 256 for byte-level or
            8192 for BBPE tokenization. Must match tokenizer.

        n_embd (int): Embedding dimension. Recommended: 256 (10M model),
            512 (100M model), or 2048 (1B model).

        n_layer (int): Number of BDH layers. More layers = more capacity
            but slower training. Recommended: 6-12.

        # ===== Multi-Scale Memory =====
        decay_rates (List[float]): Decay rates for multi-scale memory.
            Each decay rate creates a separate state matrix with different
            timescale. Recommended: [0.95, 0.99, 0.995] for fast/medium/slow.

            Timescale mapping:
            - 0.90-0.95: Fast synaptic plasticity (STP), ~50-200 tokens
            - 0.97-0.99: Medium-term plasticity (LTP), ~500-1000 tokens
            - 0.995-0.999: Slow structural changes, ~2000+ tokens

        hebbian_lr (float): Hebbian learning rate for state updates.
            Controls how quickly associations form. Too high = unstable,
            too low = slow learning. Recommended: 0.01.

    Example:
        >>> # Small multi-scale model
        >>> config = MultiScaleBDHConfig(
        ...     vocab_size=256,
        ...     n_embd=256,
        ...     n_layer=6,
        ...     decay_rates=[0.95, 0.99, 0.995]
        ... )
        >>> model = MultiScaleBDH(config)
    """

    # Architecture
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 4
    ffn_dim: int = 1024

    # Multi-Scale Memory
    decay_rates: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    hebbian_lr: float = 0.01
```

---

## Math and Algorithm Documentation

### Documenting Mathematical Operations

When implementing algorithms, include:
1. The mathematical equation
2. Reference to paper/source
3. Intuition/explanation

```python
def compute_multi_scale_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Compute multi-scale linear attention.

    Standard attention: O(N²) complexity
        attn = softmax(Q @ K.T / sqrt(d)) @ V

    Linear attention: O(N) complexity
        attn = Q @ (K.T @ V)

    Multi-scale extension:
        attn = sum_i(w_i * (Q @ (K.T @ E_i)))
        where E_i is state matrix for timescale i

    Args:
        Q: Query tensor [batch, seq_len, n_embd]
        K: Key tensor [batch, seq_len, n_embd]
        V: Value tensor [batch, seq_len, n_embd]

    Returns:
        torch.Tensor: Attention output [batch, seq_len, n_embd]

    Reference:
        - BDH Paper: arXiv:2509.26507, Section 3.2
        - Linear Attention: Katharopoulos et al. (2020)
    """
    # Implementation...
```

---

## Type Hints

Always use type hints for function signatures:

```python
from typing import List, Tuple, Optional, Union
import torch
from torch import Tensor

def process_input(
    data: Union[str, bytes, Tensor],
    tokenizer: Optional[Any] = None,
    max_length: int = 512
) -> Tuple[Tensor, Tensor]:
    """
    Process input data into model-ready tensors.

    Args:
        data: Input data (string, bytes, or pre-tokenized tensor)
        tokenizer: Optional tokenizer for text/binary data
        max_length: Maximum sequence length

    Returns:
        Tuple of (input_ids, attention_mask)
    """
```

---

## Performance Comments

Document performance-critical code:

```python
def optimized_state_update(self, Q: Tensor, V: Tensor) -> None:
    """
    Optimized state update using fused operations.

    Performance notes:
    - Uses in-place operations to reduce memory allocations
    - Fused einsum for better cache utilization
    - Benchmark: ~2x faster than naive implementation

    DO NOT modify without benchmarking!
    """
    # Fused operation: decay + hebbian update in one pass
    # This is faster than separate operations due to:
    # 1. Fewer kernel launches
    # 2. Better memory locality
    # 3. Reduced memory bandwidth
    for i, decay in enumerate(self.decay_rates):
        self.states[i].mul_(decay).add_(
            self.hebbian_lr * torch.einsum('bti,bti->ij', Q, V),
            alpha=1.0
        )
```

---

## TODO/FIXME/HACK Comments

Use standardized markers:

```python
# TODO: [T1] Implement adaptive decay rates (Phase 2)
# FIXME: [T3] Training instability on small datasets (< 1MB)
# HACK: [T2] Workaround for tokenizer bug, remove after upgrade
# NOTE: [T11] This is backward compatible with original BDH API
# OPTIMIZE: Could be vectorized for 2-3x speedup
```

---

## Testing Documentation

### Test Function Documentation

```python
def test_multiscale_state_update():
    """
    Test multi-scale state update mechanism.

    Verifies:
    1. Each state matrix updates independently
    2. Decay rates produce expected half-lives
    3. State accumulation is stable over many iterations

    Regression test for: Issue #42 - State explosion with high learning rates

    Expected behavior:
    - After 100 steps, state[0] (decay=0.95) should retain ~0.5% of initial
    - After 100 steps, state[1] (decay=0.99) should retain ~37% of initial
    - After 100 steps, state[2] (decay=0.995) should retain ~60% of initial
    """
    # Test implementation...
```

---

## Code Review Checklist

Before submitting code, verify:
- [ ] All classes have docstrings
- [ ] All public methods have docstrings
- [ ] Complex algorithms have detailed comments
- [ ] Math is documented with equations
- [ ] Type hints are used consistently
- [ ] Examples are provided where appropriate
- [ ] References to papers are included
- [ ] Performance-critical sections are marked
- [ ] TODOs/FIXMEs have ticket references

---

## Documentation Generation

To generate documentation:

```bash
# Generate HTML docs from docstrings
pdoc --html implementation/multiscale_bdh.py -o docs/api/

# Or use pydoc
python -m pydoc implementation.multiscale_bdh

# Check docstring coverage
docstr-coverage implementation/
```

---

**Document Version:** 1.0
**Last Updated:** February 25, 2026
**Author:** T11 - Technical Writer
**Status:** Active Standard
