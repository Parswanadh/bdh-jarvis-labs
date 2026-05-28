# BDH Technical Review & Hardening Report

## 📌 Overview
This document serves as the primary technical audit and remediation roadmap for the BDH project. It captures the vulnerabilities identified during the 'Critique Wave' and the resolved architectural decisions from the 'Technical Summit'.

---

## 🚨 Critical Vulnerabilities & Resolved Solutions

### 1. The Normalization Crisis (Memory Vanishing)
*   **Vulnerability**: The state matrix $E$ decayed by $\lambda < 1$, but the normalization factor $D_t$ was a cumulative sum. This caused the output $\frac{\phi(q)^T E}{D_t}$ to vanish as $t \to \infty$.
*   **Resolved Solution**: **Exponential Moving Average (EMA) Normalization**.
*   **Implementation**:
    $$\text{New Update: } D_t = \lambda D_{t-1} + \phi(k_t)^T \phi(k_t) + \epsilon$$
*   **Impact**: Numerator and denominator now decay at the same rate, preserving signal stability across infinite sequences.

### 2. The Hardware-Sparsity Gap (ReLU Myth)
*   **Vulnerability**: Claimed energy efficiency based on ReLU sparsity, which does not provide actual speedups on standard GPU kernels (zeros are still computed).
*   **Resolved Solution**: **Tiered Efficiency Framework**.
    *   **Capacity Efficiency**: Use sparsity to maximize "intelligence per byte."
    *   **Execution Efficiency**: Focus on **Kernel Fusion** to reduce HBM traffic.
    *   **Roadmap**: Align sparsity patterns with **NVIDIA 2:4 Structured Sparsity**.
*   **Impact**: Shifts the narrative from "TFLOPS win" to "Memory Bandwidth win."

### 3. The Hybrid-Path Dilemma (Architectural Contradiction)
*   **Vulnerability**: The use of a Softmax window contradicts the "Linear-Only" thesis, making the linear path look like a "crutch."
*   **Resolved Solution**: **Multi-Resolution Memory (MRM) / Saccadic Attention**.
*   **Implementation**:
    *   **Syntactic Path**: High-resolution local window for grammar/syntax.
    *   **Structural Path**: Linear state for global context/thematics.
    *   **Dynamic Gating**: A router balances paths based on input entropy.
*   **Impact**: Turns a weakness into a feature: "The model chooses the optimal resolution for the task."

### 4. The Academic Foundation (Lack of Rigor)
*   **Vulnerability**: Missing foundational 2020-2022 citations, leading to a "how-to guide" feel rather than a scientific paper.
*   **Resolved Solution**: **Anchor Paper Integration**.
*   **Core Anchors**:
    *   *Katharopoulos (2020)* $\rightarrow$ Associative Property proof.
    *   *Performer (2021)* $\rightarrow$ $\phi(\cdot)$ feature map legitimacy.
    *   *Transformers are RNNs (2020)* $\rightarrow$ Recursive state update theory.

---

## 🛠️ Technical Action Items for the Dev Team

### Priority: HIGH (Immediate Implementation)
- [ ] **Normalizer Update**: Update `bdh/attention.py` to implement EMA normalization for $D_t$ instead of cumulative sum.
- [ ] **Saccadic Gating**: Implement the entropy-based router to balance the Softmax window and Linear state.
- [ ] **$\epsilon$-Stabilizer**: Add a small epsilon to all division operations in the attention path to prevent underflow.

### Priority: MEDIUM (Optimization)
- [ ] **Kernel Fusion**: Explore fusing the Hebbian update $\lambda E + \phi(k)v^T$ into a single CUDA kernel to reduce memory roundtrips.
- [ ] **Structured Sparsity**: Transition `ReLULowRankFFN` to a format compatible with NVIDIA 2:4 structured sparsity.

### Priority: LOW (Long-term Research)
- [ ] **Fixed-Point Analysis**: Implement a diagnostic tool to measure hidden state convergence $\Delta \mathbf{h}$ during recurrent loops.
- [ ] **Logit Matching**: Move from SFT-proxy to full KL-divergence logit matching in the distillation pipeline.
