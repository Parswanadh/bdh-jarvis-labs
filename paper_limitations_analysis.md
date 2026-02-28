# BDH Paper (arXiv:2509.26507) - Comprehensive Limitations Analysis

**Paper:** "The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"
**Authors:** Adrian Kosowski, Przemyslaw Uznanski, Jan Chorowski, Michal Bartoszkiewicz, Zuzanna Stamirowska
**Date:** September 30, 2025

---

## Executive Summary

This analysis extracts all limitations, future work, and critical findings from the BDH paper. The paper presents a new architecture that matches Transformer performance on tested benchmarks (10M to 1B parameters) but has several important limitations and open questions.

---

## 1. STATED LIMITATIONS OF BDH

### 1.1 Architectural Limitations

**BDH-GPU as a Restricted Version of BDH**
- **Location:** Section 1.3, Line 214
- **Quote:** "In order to train BDH efficiently and analyze its performance, we restrict it, making this restriction the core of a GPU-friendly architecture called BDH-GPU. This restriction is obtained by treating the communication of the n particles as proceeding through a mean-field ('radio network'), rather than a graph ('communication by wire')"
- **Limitation:** BDH-GPU is a restricted/simplified version of the full BDH architecture, sacrificing some graph-based properties for GPU efficiency.

**Parameter Sharing Overhead**
- **Location:** Section 4.3, Line 923
- **Quote:** "The downside of sharing parameters between layers in the Universal Transformer is a slight time overhead for the feed-forward network operations, when measured in FLOPS per parameter. The situation is similar in BDH-GPU."
- **Limitation:** BDH-GPU shares parameters across layers (like Universal Transformer), which introduces computational overhead measured in FLOPS per parameter.

**Negative Weight Edges**
- **Location:** Section 2.1, Line 363
- **Quote:** "We provide a graph distributed systems interpretation only for dynamics on graphs with non-negative matrix entries (positive-weight edges). Negative-weight edges are hard to represent using natural local dynamics based on token distribution or spiking models."
- **Limitation:** The graph-based interpretation only works directly with non-negative (positive) edge weights. Negative weights are difficult to represent in natural local dynamics.

### 1.2 Time Scale Limitations

**Short Time Scale Focus (Minutes, Not Longer)**
- **Location:** Section 1.3, Line 236-237
- **Quote:** "In this work, we provide and validate at scale a plausible explanation of what the predominant dynamics of such a system could look like, taking the system from 'split-second' scale, to the scale of inference during 'minutes', considering the flow of time at the natural rate of thought and language for humans. A complementary discussion of the learning dynamics would aim to provide an explanation of how to take such a lifelong inference system from the scale of 'minutes' into even longer timescales."
- **Limitation:** BDH addresses reasoning at "split-second to minutes" timescales. Longer-term learning and memory transfer (beyond minutes) are NOT addressed in this work.

**Long-Term Memory Not Addressed**
- **Location:** Section 8.2, Line 1714
- **Quote:** "More broadly, this work may potentially serve to support efforts aiming to isolate, from among the many extremely complex electrochemical patterns and signal dynamics occurring in the brain, those that are crucial for solving tasks in-context (based on attention), from those that potentially serve other purposes, such as transfer of information from short-term memory to long-term memory, or long-term improvement of brain function (learning)."
- **Limitation:** The paper does NOT address:
  - Transfer from short-term to long-term memory
  - Long-term learning (synaptic weight changes over extended periods)
  - Structural adaptation of the network over time

**Memory Consolidation Not Solved**
- **Location:** Section 8.2, Line 1717-1720
- **Quote:** "For long time scales, this reduces the question of finding supervised training dynamics from the most general case, to a specific class of local dynamics: an interaction kernel performing 'edge-reweighting' rules... At this point, it seems one natural next step would be to ground the current discussion more deeply in findings of brain science, to refine or simplify the actual kernels used by brain reasoning (which was not the objective of this paper)..."
- **Limitation:** The actual kernels used by the brain for longer-term memory consolidation were not studied in this paper.

### 1.3 Scale and Testing Limitations

**Limited Parameter Scale Testing**
- **Location:** Abstract, Line 15
- **Quote:** "BDH rivals GPT2-architecture Transformer performance on language and translation tasks, at the same number of parameters (10M to 1B)"
- **Location:** Section 4.2, Line 904
- **Quote:** "BDH-GPU retains or strengthens the key advantages of the Transformer... on tests and benchmarks at the model scales we tested (1B parameters)"
- **Limitation:** BDH was only tested up to 1B parameters. Modern state-of-the-art models are in the 7B-400B+ parameter range. Performance at larger scales is unknown.

**Limited Benchmark Diversity**
- **Location:** Section 4.2, Line 912
- **Quote:** "on all next-token prediction tasks we tested, including tasks of language and translation reminiscent of those in the original benchmark set for the Transformer architecture"
- **Limitation:** Only tested on:
  - Next-token prediction tasks
  - Language tasks
  - Translation tasks (specifically mentioned in Figure 7 caption)
- **NOT tested on:** Code generation, reasoning benchmarks (MMLU, GSM8K), instruction following, multi-turn dialogue, etc.

**Translation Task Specific Results**
- **Location:** Figure 7 caption, Line 920
- **Quote:** "BDH-GPU' matches the GPT Transformer at all model sizes we have evaluated."
- **Note:** The base BDH-GPU performs comparably but not necessarily better than GPTXL. The enhanced BDH-GPU' (with conditional gating) matches GPTXL performance.
- **Limitation:** Base BDH-GPU may slightly underperform compared to Transformer on some tasks; the enhanced version with additional features (BDH-GPU') is needed to match Transformer performance exactly.

### 1.4 Theoretical Limitations

**Edge-Rewriting Kernel Complexity**
- **Location:** Section 2.1, Line 418
- **Quote:** "For context, we remark that, in comparison to the strictly simpler dynamics of node-reweighting governed by graph-based replicator dynamics equations, dynamical systems based on the edge-reweighting kernel given by Definition 2 are rather elusive to study."
- **Limitation:** The edge-reweighting kernel dynamics are theoretically complex and "elusive to study" compared to simpler node-reweighting systems.

**Expressiveness Gap**
- **Location:** Section 5.2, Line 962-966
- **Quote:** "for a Transformer with latent dimension D and BDH-GPU with hidden dimension d, we expect their feed-forward networks to be comparably expressive (though usually without strict mathematical equivalence) as function approximators for functions up to some dimension d, d < d < D, between d and D the Transformer can express a richer class of functions, and between D and n, BDH-GPU can approximate some functions, whereas the Transformer does not use such high dimension in its vector representations."
- **Limitation:** There is an expressiveness gap between BDH-GPU and Transformer:
  - For functions between d and D dimensions: Transformer can express a richer class of functions
  - BDH-GPU can approximate some high-dimensional functions that Transformer cannot, but this comes with the constraint that it only works in the positive orthant (R+)^n

**Non-Linear Dynamics**
- **Location:** Line 159
- **Quote:** "For a linear system, temporal behavior would be a direct consequence of the spectral properties of the system. The considered systems dynamics are not linear."
- **Limitation:** BDH dynamics are non-linear, making theoretical analysis more difficult compared to linear systems.

---

## 2. OPEN QUESTIONS AND FUTURE WORK

### 2.1 Explicitly Stated Future Directions

**Grounding in Brain Science**
- **Location:** Section 8.2, Line 1720
- **Quote:** "At this point, it seems one natural next step would be to ground the current discussion more deeply in findings of brain science, to refine or simplify the actual kernels used by brain reasoning (which was not the objective of this paper), and potentially seek validation through experiment."
- **Future Work:** Validate the proposed kernels against actual brain science findings.

**State-to-Weight Transfer**
- **Location:** Section 8.2, Line 1717-1720
- **Quote:** "All natural training approaches, whether based on backpropagation, or any more direct form of relaxation 'from state into weights', appear to bottleneck on the amount of available state space on synapses... Some part of the 'global mystery' of learning in the brain can be reduced to a more 'localized problem' of state-to-operator transfer for some relatively compact form of state-space dynamics."
- **Future Work:** Understand how short-term synaptic state transfers to long-term structural changes.

**Longer Time Scale Dynamics**
- **Location:** Section 8.2, Line 1714-1717
- **Quote:** "For natural systems undergoing continuous learning, the time scales to look at are: language function and reasoning (chain-of-thought inference), then short-to-long memory transfer from state to network weights, adaptation of structure: changes to interconnections, and finally, changes to neuron nodes."
- **Future Work:** Study BDH dynamics across longer time scales (hours, days, years).

### 2.2 Implied Limitations from "What We Did Not Do"

**No L1 Regularization Applied**
- **Location:** Section 1.3, Line 231
- **Quote:** "For the purposes of our experiments, we did not apply any specific training method which would be known to guide the system towards any of the observed emergent properties. (In particular, L1-regularization was disabled.)"
- **Implied Limitation:** The sparsity observed (5%) is emergent, not engineered. Without L1 regularization, sparsity might not be optimal or consistent across all training runs.

**No Specialized Training for Emergent Properties**
- **Location:** Section 1.3, Line 231
- **Quote:** "The observed emergent effects follow naturally from the design choices of the BDH and BDH-GPU architectures"
- **Implied Limitation:** The authors did not optimize training specifically to encourage modularity, monosemanticity, or other emergent properties. Performance might improve with targeted training methods.

---

## 3. COMPUTATIONAL AND PRACTICAL LIMITATIONS

### 3.1 Computational Complexity

**FLOPS per Token**
- **Location:** Section 4.2, Line 914
- **Quote:** "The theoretical count of arithmetic operations per token of BDH-GPU during inference is bounded by O(ndL). Each parameter is accessed O(L) times per token (with the typical sufficient number of layers being smaller than in the Transformer), and each element of state is accessed O(1) times per token, with small hidden constants."
- **Implication:** While asymptotic complexity is given, the constant factors and real-world performance may vary. The "slight time overhead" from parameter sharing is acknowledged.

**State Space Size**
- **Location:** Section 4.1, Line 848-849
- **Quote:** "Similarly to BDH, BDH-GPU maintains a large recurrent state comparable in size with its total number of parameters"
- **Limitation:** BDH-GPU requires O(nd) state space per layer, which for large models can be substantial. This is a trade-off: no context window limit, but large memory requirement for state.

### 3.2 Training Considerations

**Sparsity Level**
- **Location:** Section 4.1, Line 902
- **Quote:** "An empirically observed fact is that the activation pattern of xt rapidly becomes sparse (in a typical training run, only approximately 5% of the n entries of vector xt are non-zero)."
- **Note:** Only ~5% sparsity was achieved without explicit sparsity regularization. Models like Spark Transformer achieve 8% sparsity with specialized methods.
- **Limitation:** Sparsity could potentially be higher with targeted training techniques.

**Context Window Transition**
- **Location:** Section 4.2, Line 915
- **Quote:** "For short contexts BDH-GPU is amenable to parallel training with a causal self-attention kernel. For longer contexts (typically above 4096 tokens for d = 256), a state-space kernel for linear attention is faster and more space-efficient."
- **Practical Limitation:** Different kernels are optimal for different context lengths, adding implementation complexity.

---

## 4. BENCHMARK RESULTS AND COMPARISONS

### 4.1 Where BDH Performs Well

**Translation Tasks**
- **Location:** Figure 7, Line 920
- **Result:** BDH-GPU' matches GPT Transformer performance at all evaluated model sizes (10M to ~1B parameters).
- **Base BDH-GPU:** Performs comparably but may slightly trail the Transformer.

**Scaling Laws**
- **Location:** Section 4.2, Line 912
- **Quote:** "BDH-GPU appears to show improvement of loss reduction per token of data than the Transformer, i.e., learns faster per data token"
- **Advantage:** BDH-GPU learns more efficiently per data token, potentially requiring less training data.

### 4.2 Potential Weaknesses (Not Explicitly Stated but Implied)

**Limited Task Diversity**
- Only tested on language modeling and translation
- No results on:
  - Reasoning benchmarks (MMLU, BIG-Bench, etc.)
  - Code generation (HumanEval, MBPP)
  - Instruction following
  - Multi-turn dialogue
  - Mathematical reasoning
  - Commonsense reasoning

**No Comparison to Modern LLMs**
- Only compared to GPT2-style Transformer
- No comparison to:
  - GPT-3/GPT-4 architecture
  - LLaMA models
  - Mistral/Mixtral
  - Other modern decoder-only models

**Scale Limitations**
- Maximum tested: 1B parameters
- Modern SOTA: 7B to 400B+ parameters
- Unknown: Whether BDH advantages scale to model sizes beyond 1B

---

## 5. THEORETICAL GAPS AND OPEN PROBLEMS

### 5.1 Axiomatic AI

**No PAC-Like Bounds Yet**
- **Location:** Abstract, Line 19
- **Quote:** "We believe BDH opens the door to a new theory of 'Thermodynamic Limit' behavior for language and reasoning models, with the ultimate goal of Probably Approximately Correct (PAC)-like bounds for generalization of reasoning over time."
- **Gap:** PAC-like bounds for reasoning generalization are proposed as a goal but NOT achieved in this paper.

**Thermodynamic Limit Theory**
- **Location:** Abstract, Line 19
- **Quote:** "We believe BDH opens the door to a new theory of 'Thermodynamic Limit' behavior"
- **Gap:** The thermodynamic limit behavior is hypothesized but not fully derived or proven.

### 5.2 Brain Correspondence

**Not Directly Validated Against Brain Data**
- **Location:** Section 8.2, Line 1720
- **Quote:** "refine or simplify the actual kernels used by brain reasoning (which was not the objective of this paper), and potentially seek validation through experiment."
- **Gap:** The proposed correspondence between BDH dynamics and actual brain function has not been experimentally validated.

---

## 6. SUMMARY OF KEY LIMITATIONS

| Category | Limitation | Severity | Source |
|----------|-----------|----------|--------|
| **Time Scale** | Only addresses inference at "minutes" timescale, not longer-term learning | High | Line 236-237 |
| **Memory** | Short-term to long-term memory transfer NOT addressed | High | Line 1714 |
| **Scale** | Only tested up to 1B parameters (modern SOTA is 7B-400B+) | Medium | Abstract |
| **Benchmarks** | Only tested on language + translation; no code, reasoning, instruction-following | Medium | Section 4.2 |
| **Negative Weights** | Graph interpretation only for positive-weight edges | Medium | Line 363 |
| **Computational** | Slight time overhead from parameter sharing (FLOPS/parameter) | Low | Line 923 |
| **Theory** | Edge-reweighting dynamics are "elusive to study" | Low | Line 418 |
| **Validation** | Brain correspondence not experimentally validated | Medium | Line 1720 |

---

## 7. CRITICAL ANALYSIS: What BDH Does NOT Claim

The paper is careful about what it does NOT claim:

1. **Does NOT claim** to solve long-term learning or memory consolidation
2. **Does NOT claim** to be better than Transformer at all tasks (only "rivals" performance)
3. **Does NOT claim** to have experimentally validated brain correspondence
4. **Does NOT claim** to have derived PAC-like bounds (only proposes as future goal)
5. **Does NOT claim** to work at scales beyond 1B parameters (not tested)

The authors position BDH as:
- A proof-of-concept for axiomatic AI
- A bridge between Transformer and brain models
- A practical architecture with Transformer-like performance
- A foundation for future theoretical work

---

## 8. QUOTES ON LIMITATIONS (Direct from Paper)

1. **On Long-Term Memory (Line 1714):** "transfer of information from short-term memory to long-term memory, or long-term improvement of brain function (learning)" - NOT addressed.

2. **On Time Scales (Line 236-237):** "taking the system from 'split-second' scale, to the scale of inference during 'minutes'... A complementary discussion... would aim to provide an explanation of how to take such a lifelong inference system from the scale of 'minutes' into even longer timescales."

3. **On Brain Kernels (Line 1720):** "refine or simplify the actual kernels used by brain reasoning (which was not the objective of this paper)"

4. **On Negative Weights (Line 363):** "Negative-weight edges are hard to represent using natural local dynamics"

5. **On Parameter Sharing Overhead (Line 923):** "slight time overhead for the feed-forward network operations, when measured in FLOPS per parameter"

---

## 9. CONCLUSION

The BDH paper presents an innovative architecture with promising theoretical foundations and competitive empirical results at the 10M-1B parameter scale. However, significant limitations remain:

**Critical Limitations:**
1. Short-term focus only (minutes, not longer-term learning)
2. Limited scale testing (only up to 1B parameters)
3. Limited benchmark diversity (language + translation only)
4. No experimental validation of brain correspondence
5. Positive-only edge weights in graph interpretation

**Open Research Directions:**
1. Long-term memory and learning mechanisms
2. Scaling beyond 1B parameters
3. Broader benchmark evaluation (reasoning, code, etc.)
4. Experimental validation against brain data
5. Derivation of thermodynamic limit theory and PAC bounds
6. Optimization of sparsity beyond emergent 5%

The paper is notably transparent about its limitations and carefully scopes its claims. The authors present BDH as a foundation for "Axiomatic AI" rather than a complete solution to all challenges in language modeling and reasoning.

---

**Analysis Completed:** 2025-02-24
**Source:** arXiv:2509.26507v1
**Total Paper Pages:** 61
**Analysis Sections:** 1-8 (Introduction through Conclusions)
