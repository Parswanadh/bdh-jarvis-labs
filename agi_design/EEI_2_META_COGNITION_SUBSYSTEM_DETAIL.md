# 🧘 EEI Swarm Report 2: META-COGNITION SUBSYSTEM DETAIL
## Engineering Specification for Integrated Self-Reflection

**Agent:** Meta-Cognition Architect
**Focus:** Subspace Reservoirs, Confidence Metrics, and Performance Tracking

---

## 🧘 1. THE RESERVED SUBSPACE STRATEGY

Instead of adding separate "meta-cognitive layers," we will reserve specific channels within the 2048-dim embedding for meta-cognitive state.

| Embedding Channel | Signal Type | Purpose |
|-------------------|-------------|---------|
| 0 - 1919 | **Cognitive** | Standard token representations, syntax, semantic meaning. |
| 1920 - 1951 | **Confidence** | Logits of current uncertainty ($u$) per scale. |
| 1952 - 1983 | **Attention Heat**| Entropy of current synaptic updates (meta-attention). |
| 1984 - 2015 | **Resource Load** | Current compute/memory saturation levels. |
| 2016 - 2047 | **Self-Identity** | Persistent "anchor" tokens for the model's self-model. |

---

## 📈 2. UNCERTAINTY QUANTIFICATION MECHANISM

We implement uncertainty as a first-class citizen in the state matrix update.

### 2.1 State Matrix Variance ($u_t$)
We define the uncertainty of a state matrix $E$ as the reciprocal of its spectral density. High entropy in the synaptic weights implies the model has not converged on a specific pattern.

$$u_t = \text{Softplus}\left(\frac{1}{\text{trace}(E_t^T E_t)}\right)$$

### 2.2 Confidence-Gated Output
If $u_t > \tau$ (threshold), the model's output is modulated to be "tentative":
$$y_t = y_{base} \cdot \sigma(\omega - u_t)$$

This allows the model to "speak" with lower confidence or trigger internal retry mechanisms.

---

## 🏗️ 3. PERFORMANCE SELF-EVALUATION (THE "E_PERF" MATRIX)

In Phase 1 (Month 2), we add the $E_{perf}$ matrix. This is a dedicated state matrix that learns to predict the **loss** of the next token *before* seeing it.

### 3.1 Update Rule
$$E_{perf} = \lambda_p E_{perf} + \eta_p (\text{Self-Feature} \otimes \text{Error Signal})$$
Where the **Error Signal** is the cross-entropy loss of the *previous* token prediction.

### 3.2 Predictive Meta-cognition
Before outputting token $x_{t+1}$, the model queries $E_{perf}$:
$$\hat{\mathcal{L}}_{t+1} = q_t^T E_{perf} v_t$$
If predicted loss $\hat{\mathcal{L}}$ is high, the model activates **Search Mode** (increasing the multi-scale decay rate to look for more distant context).

---

## 🧘 4. SELF-MODEL CONSTRUCTION (HIERARCHICAL)

The model maintains three levels of "Self":

1. **Working Self (L1)**: Resets every ~2K tokens. Tracks current topic, speaker intent, and immediate goals.
2. **Contextual Self (L2)**: Resets every session (~32K tokens). Tracks "who I am in this conversation" and "what I have learned so far."
3. **Identity Self (L3)**: Persistent across the model's lifetime (embedded in static weights). Fundamental safety and goal-alignment priors.

---

## 🛠️ 5. PHASE 1 ACTION ITEMS (MONTH 2)

1. **Subspace Masking**: Implement the `ReservedSubspaceWrapper` in PyTorch to prevent cognitive gradients from polluting meta-cognitive channels.
2. **Confidence Head**: Train a small MLP head on top of the state matrix variance to output human-readable confidence scores (0-1).
3. **Uncertainty-Gated Attention**: Implement the mechanism where high uncertainty triggers broader scale attention.

---

**Next Report:** ACN and Safety Validator Logic.
