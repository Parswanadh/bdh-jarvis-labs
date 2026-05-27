# 🧠 EEI Swarm Report 1: BDH SCALABILITY & PERFORMANCE IMPLEMENTATION
## Engineering Specification for 500M-1B Parameter Model

**Agent:** BDH Scalability Lead
**Focus:** CUDA Kernels, Parallel Associative Scan, and Memory Optimization

---

## 📋 1. ARCHITECTURAL DIMENSIONS (500M - 850M)

To reach the target parameter count with optimal performance, we specify the following dimensions:

| Parameter | 500M Target | 850M Target | 1.2B Target |
|-----------|-------------|-------------|-------------|
| **Embedding Dimension ($d$)** | 2048 | 3072 | 4096 |
| **Num Layers ($L$)** | 12 | 12 | 16 |
| **FFN Intermediate ($d_{ffn}$)** | 8192 | 12288 | 16384 |
| **State Matrices ($d_{state}$)** | 2048 | 3072 | 4096 |
| **Num Multi-scale (K)** | 3 | 4 | 4 |
| **Sequence Length ($N$)** | 32K | 64K | 128K |

**Key Decision:** We will use 12 deep layers rather than 24 shallow layers to maximize the "depth of reasoning" per token, while keeping the state matrix size $d^2$ manageable for the $O(N)$ attention mechanism.

---

## 🚀 2. PARALLEL ASSOCIATIVE SCAN FOR HEBBIAN UPDATES

The core BDH update is:
$$E_t = \lambda E_{t-1} + \eta (Q_t \otimes V_t)$$

In standard recurrent form, this is $O(N \cdot d^2)$, which is slow for long sequences on GPUs. To parallelize training, we use the property of linear recurrence:

$$E_T = \lambda^T E_0 + \sum_{t=1}^T \eta \lambda^{T-t} (Q_t \otimes V_t)$$

### 2.1 Implementation Strategy (PyTorch/CUDA)

We will implement a custom CUDA kernel using **Parallel Associative Scan** (prefix sum logic):

```python
import torch
import torch.nn.functional as F

def bdh_parallel_scan(q, v, lambda_vals, eta):
    """
    q: (B, N, d)
    v: (B, N, d)
    lambda_vals: (B, N, 1) - can be per-step learned decay
    eta: float or (B, N, 1) - Hebbian learning rate
    """
    # 1. Compute outer products: (B, N, d, d)
    # Caution: d=2048 means d*d = 4M. For B=1, N=1024, this is 4B elements.
    # To avoid O(N*d^2) memory, we use the "Kernel Trick" for linear attention.
    
    # Linear Attention Property:
    # y_t = E_t * q_t = [sum_{i=1}^t η λ^(t-i) (Q_i ⊗ V_i)] * q_t
    # y_t = sum_{i=1}^t η λ^(t-i) V_i * (Q_i^T * q_t)
    
    # 2. Optimized Computation (Memory Efficient):
    # Instead of E_t [d x d], we compute weights W_it = η λ^(t-i)
    # This is equivalent to an Exponentially Weighted Moving Average (EWMA)
    # over the projected values.
    
    pass # See CUDA kernel spec below
```

### 2.2 Memory-Efficient CUDA Kernel Spec

To avoid the $O(d^2)$ memory bottleneck, the kernel will:
1. Load $Q_i$ and $V_i$ into shared memory.
2. Maintain a running state $E$ in register/L1 for small chunks.
3. For large $d$, use the **Selective State Update** (only updating the top-K principal components of $E$).

---

## 🧠 3. MULTI-SCALE STATE MANAGEMENT

We will use 4 scales per layer to capture different temporal dependencies:

1. **Short-term ($λ=0.9$)**: Local context, grammar, syntax (last ~10 tokens).
2. **Medium-term ($λ=0.99$)**: Sentence-level context, entities (last ~100 tokens).
3. **Long-term ($λ=0.999$)**: Paragraph-level, story arc (last ~1000 tokens).
4. **Permanent ($λ=0.9999$)**: Document-level, global theme (last ~10,000 tokens).

**Scale Integration:**
$$y_t = \text{LayerNorm}(\sum_{k=1}^4 w_k \cdot \text{BDH}_k(x_t))$$
Where $w_k$ are learned attention-gate weights.

---

## ⚡ 4. SPARSITY & EFFICIENCY

To keep the 1B model efficient, we enforce:
1. **Activation Sparsity (~5%)**: Using Top-K or ReLU with learned thresholds.
2. **Hebbian Pruning**: Synapses in $E$ with values below $\epsilon$ are zeroed out periodically to maintain sparsity in the state matrix.
3. **Multiplicative Gating**: $x_{out} = x_{in} \odot \sigma(\text{Attention}(x_{in}) + \text{FFN}(x_{in}))$. This biological "gating" mechanism is more stable than standard additive residuals for SSMs.

---

## 🛠️ 5. PHASE 1 ACTION ITEMS (MONTH 1)

1. **Kernel Development**: Finalize the `bdh_scan_cuda` kernel for $O(N)$ parallel training.
2. **Scaling Test**: Train 100M version for 10B tokens to verify scaling laws.
3. **Decay Initialization**: Implement the `ScaleInitializer` that sets λ based on the layer depth (deeper layers = longer memory).

---

**Next Report:** Meta-Cognition Subsystem Detail.
