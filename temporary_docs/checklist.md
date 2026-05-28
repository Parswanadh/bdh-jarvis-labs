# BDH PoC Mastery Checklist

### 1. The 'PoC' Narrative (The Why)
- [ ] Explain the core objective: Transitioning from static, quadratic-complexity Transformers to dynamic, linear, and recurrent brain-inspired processing.
- [ ] Contrast the "Standard Transformer" paradigm (fixed depth, $O(N^2)$ attention) with the "BDH" paradigm (recurrent depth, $O(N)$ attention, and biological efficiency).
- [ ] Articulate the trade-off between memory efficiency (fixed-size state) and expressive power (iterative refinement).

### 2. Core Architecture (The Foundation)
- [ ] Derive the mathematical shift from Softmax Attention to Linear Attention using the associative property of matrix multiplication.
- [ ] Explain the Memory Recurrence Model (MRM) state update: how the hidden state matrix evolves and summarizes sequence history.
- [ ] Contrast the MRM state update with standard RNNs (e.g., LSTM/GRU) and traditional Transformer KV caches.

### 3. Brain-Inspired Design (The Efficiency)
- [ ] Explain "Sparsity" in the context of neural activations and the biological motivation for energy efficiency and feature separation.
- [ ] Detail the "Gating" mechanisms: how the model selectively controls the flow of information into and out of the recurrent state.
- [ ] Analyze how the combination of sparsity and gating prevents state saturation and catastrophic forgetting during long-sequence processing.

### 4. Recurrent Depth (The Intelligence)
- [ ] Differentiate between "Layer Depth" (stacking different weights) and "Recurrent Depth" (iterating through the same weights).
- [ ] Describe the mechanism of Looped Inference: the process of refining a representation through multiple passes of the same network.
- [ ] Explain Adaptive Computation Time (ACT): the logic used to dynamically determine the number of iteration loops based on the complexity of the input.

### 5. Training & Implementation (The How)
- [ ] Explain Byte-level Byte Pair Encoding (BBPE) and its role in maintaining a compact, efficient vocabulary for the PoC.
- [ ] Describe the Knowledge Distillation pipeline: how a pre-trained Transformer (Teacher) is used to supervise the training of the BDH architecture (Student).
- [ ] Identify the specific optimization challenges of recurrent depth (e.g., gradient stability) and the methods used to mitigate them.

### 6. Expert Verification (The Proof)
- [ ] Prove that the inference time and memory usage scale linearly $O(N)$ with sequence length.
- [ ] Demonstrate that increasing recurrent loops correlates with improved performance on high-complexity reasoning tasks.
- [ ] Verify the sparsity of the model by analyzing activation patterns across different input types.
