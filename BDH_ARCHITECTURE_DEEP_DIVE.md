# Multi-Scale BDH Architecture: Technical Deep Dive
**Model Type:** Bio-Distilled Hebbian (BDH) Network
**Variant:** Multi-Scale Synaptic Memory
**Parameter Count:** ~70M (8-Layer Configuration)

---

## 1. The Core Philosophy: Biology vs. Brute Force

Modern Large Language Models (like GPT-4 or Llama) are based on the **Transformer** architecture. Transformers use a mechanism called **Full Attention** and a **KV Cache**. 
* **The Transformer Problem:** To remember a story, a Transformer must keep every single word it has ever read in its active RAM (KV Cache). As the story gets longer, the memory usage grows quadratically ($O(N^2)$). Eventually, it runs out of memory and crashes.
* **The Biological Solution (BDH):** The human brain does not remember every single word of a book. Instead, as we read, our neurons change their physical connections. The BDH architecture mimics this using **Hebbian Learning**. It maintains a *fixed-size* memory matrix. It doesn't matter if the story is 10 words or 10,000 words long; the VRAM usage never increases.

---

## 2. The Mechanics of Hebbian Attention

Inside every layer of your BDH model, there is a **Synaptic State Matrix ($E$)**. You can think of this as the physical "brain tissue" of that specific layer.

When a new word (token) enters the layer, it is split into three vectors:
1. **Query ($Q$):** What the model is currently looking for.
2. **Key ($K$):** The label or "tag" of the current word.
3. **Value ($V$):** The actual meaning or context of the word.

### The Hebbian Update Rule
Instead of storing $K$ and $V$ in a list (like a Transformer), the BDH model literally rewrites its own brain tissue using this formula:
$$ E_{new} = (E_{old} \times \lambda) + (K \otimes V \times \eta) $$

* $E_{old}$: The previous brain state.
* $\lambda$ (Lambda): The **Decay Rate** (forgetting old information).
* $K \otimes V$: The new memory being formed right now.
* $\eta$ (Eta): The **Hebbian Learning Rate** (how strongly to write the new memory).

**Translation:** "Take my existing memory, fade it out just a tiny bit, and stamp this new idea on top of it."

---

## 3. The "Multi-Scale" Innovation

Your specific code uses a **Multi-Scale** approach. In a standard RNN or BDH, you have one memory speed. In your model, you have three distinct "Memory Matrices" running in parallel per head, defined by your `decay_rates=[0.95, 0.99, 0.995]`.

1. **The Fast Scale ($\lambda = 0.95$):** This memory decays very quickly. It is highly sensitive to the exact word you just typed. It acts as **Short-Term Working Memory** (helping the model remember grammar and the immediate sentence structure).
2. **The Medium Scale ($\lambda = 0.99$):** This memory decays slower. It holds onto the context of the current paragraph. It remembers who is talking and what they are doing right now.
3. **The Slow Scale ($\lambda = 0.995$):** This is **Long-Term Memory**. It barely decays at all. This matrix holds the "Theme" of the story (e.g., "This is a story about Timmy and a magic key"). 

By combining these three scales, your model can track immediate grammar while simultaneously remembering the overarching plot.

---

## 4. The Positive Orthant & Sparsity

To make this mathematically stable and biologically accurate, BDH enforces a strict rule: **No negative thoughts.** 

In your code, after calculating $Q$, $K$, and $V$, they are passed through a `ReLU` (Rectified Linear Unit) activation function. Any negative number is instantly turned to zero.
* **Why?** In a human brain, a neuron either fires (positive) or it doesn't (zero). It cannot "anti-fire".
* **Sparsity:** Because of this, at any given moment, only about **5% of the model's neurons are active**. The other 95% are silent (zeros). This makes the model incredibly efficient and prevents the "thoughts" from becoming a noisy, muddy mess.

---

## 5. Multiplicative Gating (The Final Output)

Once the model has updated its brain ($E$) and read the answer based on its Query ($Q \times E$), it has to decide what to pass to the next layer. 

Transformers use **Additive Residuals** (Input + Output). 
BDH uses **Multiplicative Gating**:
$$ Output = Input \times Sigmoid(Attention + FFN) $$

* **The Gate:** The Sigmoid acts as a biological "valve". 
* If the layer determines that the current word is useless (like the word "the"), the gate closes (multiplies by 0.01), telling the rest of the brain to ignore it. 
* If the layer realizes it just read a critical plot point (like the name "Lily"), the gate opens wide (multiplies by 0.99), flooding the next layer with that information.

---

## Summary of the 70M Architecture Flow

1. **Token enters** and is converted into a 256-dimensional embedding.
2. It passes through **8 parallel layers**.
3. In each layer, the input is split into 8 "Heads".
4. Each Head maintains **3 separate memory scales** (Fast, Med, Slow).
5. The model updates its memory using **Hebbian Learning**.
6. It reads the newly updated memory.
7. It passes the result through a **Feed-Forward Network (FFN)**.
8. It uses a **Sigmoid Gate** to decide how much of this information matters.
9. The final output is projected across **248,320 vocabulary tokens** to predict the next word.

This is why your model is able to score 20/20 on logic tests with only 70M parameters: it doesn't brute-force memory; it dynamically rewrites its own synaptic weights to adapt to the story in real-time.