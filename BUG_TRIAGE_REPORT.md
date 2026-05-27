# BDH Bug Triage Report — Day 1

> **Date:** May 17, 2026
> **Severity Classification:** CRITICAL / HIGH / MEDIUM / LOW
> **Source:** Code audit of `implementation/multiscale_bdh.py`, `implementation/bdh_recurrent.py`, `implementation/stable_config.py`
> **Cross-referenced with:** Research docs 01-08, expert critique (07), critical gaps (06)

---

## Executive Summary

4 critical bugs identified in the BDH implementation. All are theory-implementation gaps — the paper/architecture spec says one thing, the code does another. These bugs invalidate all experimental results from the current codebase.

| # | Bug | Severity | File:Line | Status |
|---|-----|----------|-----------|--------|
| 1 | Element-wise Hebbian (not outer product) | CRITICAL | `multiscale_bdh.py:221` | IDENTIFIED |
| 2 | RoPE disabled — no positional awareness | HIGH | `multiscale_bdh.py:304` | IDENTIFIED |
| 3 | Multiplicative gating causes vanishing gradients | HIGH | `multiscale_bdh.py:431-432` | IDENTIFIED |
| 4 | Vocab mismatch in distillation | HIGH | `multiscale_bdh.py:34` vs trained model | IDENTIFIED |

---

## Bug #1: Element-Wise Hebbian Update (Not Outer Product)

**Severity:** CRITICAL
**File:** `implementation/multiscale_bdh.py`
**Lines:** 201-227

### What the Paper Says

```
E[t] = γ·E[t-1] + η·(Q ⊗ V)
```

Where ⊗ is the **outer product**, producing a [d × d] state matrix that captures cross-dimensional correlations.

### What the Code Does

```python
# Line 217-225
Q_flat = Q.mean(dim=(0, 1))  # [H, D]
V_flat = V.mean(dim=(0, 1))  # [H, D]

# Outer product (element-wise for each head)  ← WRONG COMMENT
hebbian = Q_flat * V_flat  # [H, D]  ← ELEMENT-WISE, NOT OUTER PRODUCT

# Reshape to [C, C] for state matrix update
C = self.n_embd
hebbian = hebbian.reshape(C)  # [C]  ← VECTOR, NOT MATRIX

return hebbian  # Returns [C], not [C, C]
```

### Evidence

| Aspect | Expected | Actual |
|--------|----------|--------|
| Operation | `torch.einsum('hd,he->hde', Q, V)` | `Q_flat * V_flat` |
| Output shape | `[H, D, D]` or `[C, C]` | `[H, D]` → `[C]` |
| Cross-dim correlations | Captured | **Lost entirely** |
| State update | Matrix + Matrix | Matrix + Vector (broadcast) |

### Impact

1. **State matrix loses all cross-dimensional associations.** The [d × d] state should encode "when neuron i fires, neuron j tends to fire too." Element-wise product only captures "when neuron i fires, neuron i fires" — a no-op.
2. **Shape mismatch silently masked by broadcasting.** `_init_states` creates [C, C] matrices (line 190, 197), but `_update_states` adds a [C] vector (line 250). NumPy/PyTorch broadcasts [C] across rows of [C, C], so every row gets the same update. This is wrong but doesn't crash.
3. **All experimental results from the current model are invalid.** The state matrix is not doing what the architecture claims.

### Fix (Day 2)

```python
def _compute_hebbian_update(self, Q: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute Hebbian update: outer product of Q and V."""
    Q_flat = Q.mean(dim=(0, 1))  # [H, D]
    V_flat = V.mean(dim=(0, 1))  # [H, D]

    # TRUE outer product: [H, D, D]
    hebbian = torch.einsum('hd,he->hde', Q_flat, V_flat)

    # Reshape to [C, C] where C = H * D
    H, D = self.n_head, self.head_dim
    C = self.n_embd
    hebbian = hebbian.reshape(H, D, D).reshape(C, C)

    return hebbian  # [C, C]
```

**Memory cost:** O(d) → O(d²) per layer. For d=256: 65,536 vs 256 elements. Manageable.

### Cross-References

- Research doc 06 (Critical Gaps): "Proper outer product Hebbian update (BDH uses element-wise)"
- Research doc 07 (Expert Critique): Section 2.1 — "Hebbian Update is Element-Wise, Not Outer Product"
- Research doc 01 (Architecture): Defines state as edge weights on graph (n×n matrix)

---

## Bug #2: RoPE Disabled — No Positional Awareness

**Severity:** HIGH
**File:** `implementation/multiscale_bdh.py`
**Lines:** 298-304

### What the Code Does

```python
# Lines 298-304
# Apply RoPE position encoding
# TEMPORARILY DISABLED - causing shape mismatch errors
# TODO: Fix RoPE shape compatibility
# cos, sin = self.rope(Q, T)
# Q = self.rope.rotate(Q, cos, sin)
# K = self.rope.rotate(K, cos, sin)
pass  # No positional encoding for now
```

### Evidence

- `RotaryPositionalEmbedding` class exists and is fully implemented (lines 69-134)
- Instance is created: `self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)` (line 165)
- But **never called** in forward pass
- Comment says "TEMPORARILY DISABLED" — this has been in place since the multi-scale implementation

### Impact

1. **Model is permutation-invariant.** It cannot distinguish "A B C" from "C B A" from "B A C".
2. **All sequence-order-dependent tasks fail:** copy, reversal, needle-in-haystack, any task requiring positional reasoning.
3. **Invalidates all results** from the multi-scale model for any task where order matters.

### Root Cause

The comment says "causing shape mismatch errors." The RoPE class expects input `[batch, seq_len, heads, head_dim]` (line 99), but the forward pass provides Q/K in that format (line 294-296). The issue is likely in the `rotate` method's broadcasting with cos/sin cached as `[1, 1, max_seq_len, dim]` (lines 91-92).

### Fix (Day 3)

```python
# Apply RoPE position encoding
cos, sin = self.rope(Q, T)  # [1, 1, T, D]
Q = self.rope.rotate(Q, cos, sin)
K = self.rope.rotate(K, cos, sin)
```

May need to adjust RoPE's `rotate` method to handle the `[B, T, H, D]` shape correctly.

### Cross-References

- Research doc 07 (Expert Critique): Section 2.2 — "RoPE Disabled in Multi-Scale"
- Research doc 02 (Qwen): RoPE is critical for positional awareness in all modern architectures

---

## Bug #3: Multiplicative Gating Causes Vanishing Gradients

**Severity:** HIGH
**File:** `implementation/multiscale_bdh.py`
**Lines:** 428-432

### What the Code Does

```python
# Lines 428-432
# MULTIPLICATIVE GATING (key difference from Transformers!)
# Instead of: x + attn_out + ffn_out
# We use: x * sigmoid(attn_out + ffn_out)
gated = torch.sigmoid(attn_out + ffn_out)
out = x * gated
```

### Analysis

The gradient flow through this gating:

```
out = x * σ(attn_out + ffn_out)

∂out/∂x = σ(attn_out + ffn_out)  ← gate value directly
∂out/∂attn = x * σ'(attn + ffn)  ← depends on x AND gate derivative
∂out/∂ffn = x * σ'(attn + ffn)   ← same
```

**Problem:** When `attn_out + ffn_out` is large negative, `sigmoid → 0`:
- `out ≈ 0` — all information from x is destroyed
- `∂out/∂x ≈ 0` — gradient vanishes
- `∂out/∂attn ≈ 0` and `∂out/∂ffn ≈ 0` — no learning signal to attention or FFN

This is the root cause of "BDH is 20-30% harder to train."

### Evidence

- Research doc 07 (Expert Critique): "Multiplicative gating causes vanishing gradients when sigmoid saturates at 0"
- Research doc 04 (Architecture Landscape): "All successful architectures at scale use ADDITIVE residuals, not multiplicative"
- Research doc 06 (Critical Gaps): Recommends hybrid residual: `x + α·gate·(attn+ffn)`

### Impact

1. **Training instability** — random initialization can push gates to 0, causing dead layers
2. **Vanishing gradients** — deep layers receive no signal
3. **Explains the 20-30% training difficulty** claim
4. **Requires 3× smaller init, 10× longer warmup** (documented in stable_config.py)

### Fix (Day 4)

Option A — Hybrid residual (recommended):
```python
# Hybrid: additive base + multiplicative modulation
gate = torch.sigmoid(attn_out + ffn_out)
out = x + 0.1 * gate * (attn_out + ffn_out)  # α = 0.1 learnable
```

Option B — Softplus gating (never saturates at 0):
```python
gate = F.softplus(attn_out + ffn_out)  # Always > 0
out = x * gate / (1 + gate)  # Normalized to (0, 1)
```

Option C — Gated residual with skip connection:
```python
gate = torch.sigmoid(attn_out + ffn_out)
out = (1 - gate) * x + gate * (attn_out + ffn_out)  # Always has skip path
```

### Cross-References

- Research doc 07: Section 1.3 — "Multiplicative Gating: Theoretically Interesting, Practically Problematic"
- Research doc 06: Proposal 1 (BDH-2) — recommends data-dependent gating
- `stable_config.py`: Documents the workaround (smaller init, longer warmup) but doesn't fix the root cause

---

## Bug #4: Vocab Mismatch in Distillation

**Severity:** HIGH
**File:** `implementation/multiscale_bdh.py` line 34, trained model config

### The Mismatch

| Component | Vocab Size | Tokenizer |
|-----------|-----------|-----------|
| MultiScaleBDHConfig default | 256 | Byte-level (no tokenizer) |
| Trained model (per docs) | 32,000 | TinyLlama tokenizer |
| Teacher (Qwen3.5-0.8B) | ~151,936 | Qwen tokenizer |
| Teacher (TinyLlama 1.1B) | 32,000 | Llama tokenizer |

### Evidence

- `multiscale_bdh.py` line 34: `vocab_size: int = 256`
- `BDH_CORE_ALGORITHM_EXPLAINED.md` line 631: `Vocab size: 32,000 (TinyLlama tokenizer)`
- `README.md` line 48: `--teacher-model Qwen3.5-0.8B` (vocab ~151,936)
- No projection layer exists in any implementation file
- No vocab mapping code exists

### Impact

1. **True logits distillation never works.** Teacher produces logits over 32K-152K tokens, student produces logits over 256-32K tokens. No mapping between them.
2. **Falls back to CE loss only.** The "distillation" is just supervised learning on the student's own vocab.
3. **All dark knowledge is lost.** Temperature-scaled soft targets, relative probabilities between tokens — none of it transfers.
4. **Explains why training needed 18+ hours** — without dark knowledge, the model learns from scratch.

### Fix (Day 5)

Option A — Learned projection (expensive):
```python
class VocabProjection(nn.Module):
    def __init__(self, teacher_vocab: int, student_vocab: int):
        super().__init__()
        self.proj = nn.Linear(teacher_vocab, student_vocab, bias=False)

    def forward(self, teacher_logits):
        return self.proj(teacher_logits)
```

Option B — CKA-based hidden state distillation (recommended):
```python
# Match teacher and student hidden states via CKA
# No vocab alignment needed
def cka_loss(student_h, teacher_h):
    # Centered Kernel Alignment
    # Works for any dimension mismatch
    ...
```

Option C — Shared tokenizer (breaks "no tokenizer" philosophy):
- Use TinyLlama tokenizer for both teacher and student
- Simplest fix but loses byte-level advantage

### Cross-References

- Research doc 07: Section 2.3 — "Vocab Mismatch in Distillation"
- Research doc 05 (Distillation Master): 41 papers on cross-architecture distillation
- Research doc 06: "CKA-based hidden state matching" as novel direction

---

## Bug Severity Summary

| Bug | Severity | Blocks | Fix Complexity | Fix Day |
|-----|----------|--------|---------------|---------|
| #1 Element-wise Hebbian | CRITICAL | All state matrix results | Medium (O(d²) memory) | Day 2 |
| #2 RoPE Disabled | HIGH | All positional tasks | Low (enable + fix shape) | Day 3 |
| #3 Gating Saturation | HIGH | Training stability | Medium (redesign residual) | Day 4 |
| #4 Vocab Mismatch | HIGH | Distillation quality | Medium (projection or CKA) | Day 5 |

---

## Additional Issues (Not Critical)

### A1: State Initialization Shape Inconsistency
- `_init_states` creates `[C, C]` matrices (line 190, 197)
- `_compute_hebbian_update` returns `[C]` vector (line 225)
- `_update_states` adds vector to matrix via broadcasting (line 250)
- **Impact:** Every row of the state matrix gets the same update — loses head-specific information
- **Fix:** Align shapes (part of Bug #1 fix)

### A2: Base BDH vs Multi-Scale BDH Config Mismatch
- `BDH_CORE_ALGORITHM_EXPLAINED.md` says: 8 layers, 512 embedding, 8 heads
- `MultiScaleBDHConfig` defaults: 6 layers, 256 embedding, 4 heads
- **Impact:** Confusion about model size; benchmarks may use different configs
- **Fix:** Standardize config defaults

### A3: No Gradient Clipping in Implementation
- `stable_config.py` specifies `grad_clip=1.0` (line 131)
- But no implementation file actually applies gradient clipping
- **Impact:** Training may diverge with large gradients
- **Fix:** Add `torch.nn.utils.clip_grad_norm_` to training loop

### A4: No Sparsity Measurement
- Claims "~5% active neurons" but no code measures this
- **Impact:** Unverified claim
- **Fix:** Add sparsity tracking: `sparsity = (h == 0).float().mean()`

---

## Bug-to-Fix Mapping

| Bug | Fix File | Fix Function | Test |
|-----|----------|-------------|------|
| #1 Hebbian | `bdh_v2_clean.py` | `_compute_hebbian_update` | Unit test: outer product shape + gradient flow |
| #2 RoPE | `bdh_v2_clean.py` | `forward` (enable RoPE) | Test: "ABC" ≠ "CBA" output |
| #3 Gating | `bdh_v2_clean.py` | `MultiScaleBDHLayer.forward` | Test: gradient norm > 0 for all layers |
| #4 Vocab | `train_distillation_v2.py` | VocabProjection or CKA loss | Test: KL divergence > 0 |

---

## Research Doc Cross-Reference Matrix

| Research Doc | Bug #1 | Bug #2 | Bug #3 | Bug #4 |
|-------------|--------|--------|--------|--------|
| 01 Architecture | ✅ Theory defines outer product | ✅ RoPE in base model | ✅ Describes multiplicative gating | ✅ Mentions vocab 32K |
| 02 Qwen | — | ✅ RoPE implementation | — | — |
| 03 DeepSeek | — | — | — | ✅ Distillation pipeline |
| 04 Landscape | ✅ Compares to DeltaNet | — | ✅ Notes all use additive | — |
| 05 Distillation | — | — | — | ✅ Vocab mismatch solutions |
| 06 Critical Gaps | ✅ "Proper outer product" | ✅ "Enable RoPE" | ✅ "Data-dependent gating" | ✅ "Fix vocab mismatch" |
| 07 Expert Critique | ✅ Section 2.1 | ✅ Section 2.2 | ✅ Section 1.3 | ✅ Section 2.3 |
| 08 SubQ SSA | — | — | — | — |

---

## Conclusion

All 4 bugs are **theory-implementation gaps**, not fundamental architecture flaws. The BDH architecture is sound in principle, but the implementation has drifted from the spec. Fixing these bugs is the highest-priority work and unblocks all downstream research.

**Estimated fix time:** 4 days (Days 1-4 of master plan)
**Risk:** Low — fixes are well-understood and documented in research literature
**Impact:** High — enables valid experiments, fair benchmarks, and NeurIPS-quality results
