# LLM Architecture Knowledge Base (27-05) - PRODUCTION-GRADE DEEP DIVE

This document is a high-density technical repository for the BDH project. It moves beyond academic summaries to provide "Implementation-Ready" knowledge.

**Focus Areas:**
1. **Mathematical Foundations**: Exact formulas and tensor transformations.
2. **DeepSeek Breakthroughs**: Exhaustive analysis of MLA, DeepSeekMoE, and their efficiency gains.
3. **Production Infrastructure**: Analysis of scaling laws, cluster orchestration, FP8 training, and communication optimization.
4. **Engineering Secrets**: Implementation details that enable SOTA performance in real-world production.

---

## [SOTA FOCUS] DeepSeek-V2/V3: Engineering Specification

### 1. Multi-head Latent Attention (MLA)
MLA is designed to eliminate the KV cache bottleneck of standard Multi-Head Attention (MHA) while maintaining or exceeding its performance.

#### 1.1 Hardware-First Bottleneck Analysis
- **HBM vs SRAM**: KV cache memory traffic reduced from $2 \cdot S \cdot n_{head} \cdot d_{head}$ to $S \cdot d_{latent}$. Bytes/FLOP ratio: $\frac{d_{latent}}{d_{head} \cdot n_{head}}$ reduction.
- **Stall Operation**: `LDSM` latency during KV expansion.
- **TMA Specification**: 2D TMA descriptor for latent vector $\mathbf{c}_{kv} \in \mathbb{R}^{S \times d_{latent}}$ with stride $d_{latent}$ for asynchronous tile loading.

#### 1.2 Mathematical DNA
- **Exhaustive LaTeX**: 
  $\mathbf{c}_{kv} = W_{DKV} \mathbf{x}$
  $\mathbf{k} = W_{UK} \mathbf{c}_{kv}, \quad \mathbf{v} = W_{UV} \mathbf{c}_{kv}$
  $\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{(W_{UQ} \mathbf{c}_{q})^T (W_{UK} \mathbf{c}_{kv})}{\sqrt{d_{head}}}\right) (W_{UV} \mathbf{c}_{kv})$
- **Tensor Flow**: $[B, S, d_{model}] \xrightarrow{W_{DKV}} [B, S, d_{latent}] \xrightarrow{\text{TMA\_SRAM}} [T_x, T_y] \xrightarrow{W_{UK}/W_{UV}} [T_x, n_{head} \cdot d_{head}]$
- **Gradient Stability**: Latent compression $\text{rank}(W_{DKV}) \ll d_{model}$ acts as a spectral filter, bounding the Lipschitz constant of the attention mapping.

#### 1.3 CUDA/Triton Implementation Spec
- **WGMMA logic**: Execute `wgmma.mma_async` on $\mathbf{c}_{kv} \times W_{UK}$ using warpgroup-level cooperation.
- **SRAM Tiling**: $\mathbf{c}_{kv}$ tile $[128, 512]$, $W_{UK}$ tile $[512, 128]$. Total SRAM usage $\approx 128\text{KB}$.
- **Sync Primitives**: `mbarrier` for TMA $\rightarrow$ WGMMA dependency tracking.

#### 1.4 Implementation Algorithm
$\text{TMA Load}(\mathbf{c}_{kv}) \rightarrow \text{WGMMA}(\mathbf{c}_{kv}, W_{UK}) \rightarrow \text{SRAM Store}(\mathbf{k}) \rightarrow \text{WGMMA}(\mathbf{q}, \mathbf{k}) \rightarrow \text{SRAM Store}(\text{score})$.

#### 1.5 BDH la-Steal
Implement `FusedLatentKVExpansion` kernel: $\text{TMA}(\text{latent}) \to \text{WGMMA}(\text{upscale}) \to \text{Register}$ without HBM write-back.

---

## 2. DeepSeekMoE (Fine-Grained Expert Specialization)

### 2.1 Hardware-First Bottleneck Analysis
- **HBM vs SRAM**: Parameter-heavy; weight loading $\gg$ computation. Bottleneck is $\text{HBM\_BW}$.
- **Stall Operation**: `LDG.E` (Global Load) for expert weights.
- **TMA Specification**: 3D TMA descriptor for expert weight tensors $[E, D_{in}, D_{out}]$ to enable asynchronous loading of active expert slices.

### 2.2 Mathematical DNA
- **Exhaustive LaTeX**: 
  $g = \text{softmax}(\mathbf{x} W_g)$
  $\text{TopK}(g, k) \rightarrow \{i_1, \dots, i_k\}$
  $\mathbf{y} = \sum_{j=1}^k g_{i_j} (W_{expert, i_j} \mathbf{x})$
- **Tensor Flow**: $[B, S, D] \xrightarrow{\text{Route}} [B, S, K] \xrightarrow{\text{TMA\_Expert}} [T_x, D_{out}]$
- **Gradient Stability**: Auxiliary loss $\mathcal{L}_{aux} = \alpha \sum_{i=1}^E (\text{count}(i) - \frac{BS}{E})^2$ enforces load balancing to prevent expert collapse.

### 2.3 CUDA/Triton Implementation Spec
- **WGMMA logic**: Grouped GEMM utilizing `wgmma` for multiple active experts per warpgroup.
- **SRAM Tiling**: $[64, 128]$ tiles for expert weights to maximize reuse of input $\mathbf{x}$.
- **Sync Primitives**: `cuda::barrier` to synchronize routing results before weight fetching.

### 2.4 Implementation Algorithm
$\text{Compute\_Route} \rightarrow \text{TMA Load}(\text{Active\_Experts}) \rightarrow \text{WGMMA}(\mathbf{x}, W_{exp}) \rightarrow \text{SRAM Store}(\mathbf{y})$.

### 2.5 BDH la-Steal
Implement `AsyncExpertGather`: Use TMA to pull weights for the top-k experts in parallel with the routing calculation.

---

## 3. Production Scaling & Infrastructure

### 3.1 Training Precision (FP8)
- **HBM vs SRAM**: FP8 reduces traffic by $2\times$ vs BF16. Bottleneck shifts to scaling factor application.
- **Stall Operation**: FP32 conversion for scaling.
- **TMA Specification**: TMA loading of FP8 blocks with 1-byte scaling factors.

### 3.2 Mathematical DNA
- **Exhaustive LaTeX**: 
  $Q_{fp8} = \text{quant}(Q, \text{scale}_q), \quad K_{fp8} = \text{quant}(K, \text{scale}_k)$
  $\text{Score} = (\text{scale}_q \cdot \text{scale}_k) \cdot \text{WGMMA}(Q_{fp8}, K_{fp8})$
- **Tensor Flow**: $\mathbb{R}_{fp32} \xrightarrow{\text{quant}} \mathbb{R}_{fp8} \xrightarrow{\text{WGMMA}} \mathbb{R}_{fp32}$
- **Gradient Stability**: Dynamic per-tensor scaling prevents underflow in FP8 $\text{E4M3}$ range.

### 3.3 CUDA/Triton Implementation Spec
- **WGMMA logic**: `wgmma.mma_async` with `e4m3` input types.
- **SRAM Tiling**: $[128, 128]$ FP8 tiles.
- **Sync Primitives**: `mbarrier` for pipeline stage synchronization.

### 3.4 Implementation Algorithm
$\text{TMA Load}(\text{FP8\_Tensor}, \text{Scale}) \rightarrow \text{WGMMA}(\text{FP8}) \rightarrow \text{Dequant\_SRAM} \rightarrow \text{SRAM Store}$.

### 3.5 BDH la-Steal
Implement `FP8BlockScaling`: A Triton kernel that performs block-wise scaling and casting in a single pass using `mma_async`.

---

## [SOTA FOCUS] Company Articles 1-5

### 1. Llama 3 Herd of Models (405B)

**1. The Physical Bottleneck (First Principles)**
- **The Wall**: HBM Bandwidth (Memory Wall). During auto-regressive decoding, the KV-cache must be loaded from HBM to SRAM for every token generated.
- **SRAM/HBM Ratio**: For a hidden dimension $d=8192$ and batch size $B=32$, moving the KV-cache for one layer exceeds 100 GB/s per token. The ratio is $\approx 1 \text{ TFLOP} : 0.1 \text{ Byte}$ for the attention primitive, leading to severe memory-bound stalls.
- **Stall Operation**: $\text{Softmax}(\frac{QK^T}{\sqrt{d}})V$ load-compute cycle.

**2. Mathematical DNA (GQA Primitive)**
- **Formulae**:
  $$\text{Query: } Q = XW_Q, \quad \text{Key: } K = XW_K, \quad \text{Value: } V = XW_V$$
  $$\text{GQA: } \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \cdot K_{\text{grouped}}^T}{\sqrt{d_k}}\right) V_{\text{grouped}}$$
  $$\text{RoPE: } f_{\text{RoPE}}(x, m) = x \cdot \cos(m\theta) + \text{rotate}(x) \cdot \sin(m\theta)$$
- **Tensor Flow**: $[B, S, H_q, d_h] \times [B, S, H_{kv}, d_h]^T \xrightarrow{\text{Grouped}} [B, S, H_q, d_h]$ where $H_{kv} \ll H_q$.
- **Stability**: RMSNorm: $\bar{a}_i = \frac{a_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d a_j^2 + \epsilon}} \gamma_i$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused GQA kernel. Use `wmma` for $Q \cdot K^T$ GEMM, then fused `exp` and `sum` for Softmax in SRAM.
- **Data Movement**: Store $K, V$ in fragmented blocks (PagedAttention style). Use L2 cache to pin the shared $K, V$ heads across the $H_q/H_{kv}$ group.
- **Tiling Strategy**: Tile size $[128, 64]$ for the attention matrix to ensure the $QK^T$ block fits in shared memory (SRAM).

**4. Implementation Algorithm**
1. Load $Q$-tile $[T_q, d_h]$ into SRAM.
2. Load grouped $K$-tile $[T_{kv}, d_h]$ into SRAM.
3. Compute dot product $\to$ apply RoPE $\to$ store in SRAM.
4. Compute Softmax across the $S$ dimension using an online update rule.
5. Load $V$-tile and accumulate the weighted sum into the output buffer.

**5. BDH Alignment (The la-Steal)**
Implement a `FusedGQA` operator that performs $QK^T$ and $SV$ in a single kernel call, reducing HBM round-trips for $K$ and $V$ by a factor of $H_q/H_{kv}$.

---

### 2. Gemini 1.5 (Massive Context MoE)

**1. The Physical Bottleneck (First Principles)**
- **The Wall**: SRAM capacity and Inter-GPU Bandwidth. Massive contexts ($1\text{M}+$ tokens) exceed the capacity of any single GPU's HBM.
- **SRAM/HBM Ratio**: The attention matrix $S \times S$ grows quadratically. At $1\text{M}$ tokens, a full matrix is $10^{12}$ elements.
- **Stall Operation**: All-to-All communication for MoE routing and Ring-Attention synchronization.

**2. Mathematical DNA (Ring Attention MoE)**
- **Formulae**:
  $$\text{MoE Output: } y = \sum_{i=1}^{k} G(x)_i E_i(x)$$
  $$\text{Ring Attention: } \text{Attn}_j = \text{Attend}(Q_j, K_{j \dots j+S}, V_{j \dots j+S})$$
- **Tensor Flow**: $[B, S_{local}, D] \xrightarrow{\text{Route}} [B, S_{local}, \text{Expert}_k, D] \xrightarrow{\text{Ring}} [B, S_{global}, D]$.
- **Stability**: **Load Balancing Loss** is added to the objective function to prevent "expert collapse" (where one expert takes all the gradient), ensuring stable convergence.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Implement Ring-Attention using `cudaMemcpyAsync` to overlap the transfer of $K, V$ blocks with the computation of the current block.
- **Data Movement**: KV cache is sharded across devices; only the current block being processed is cached in SRAM.
- **Nature**: Shifted from Compute-Bound to **Communication-Bound** due to the massive scale of the ring buffer.

**4. Implementation Algorithm**
1. Partition sequence $S$ across $N$ GPUs.
2. Loop $N$ times:
   a. Compute attention for local $Q$ and current $K, V$ block.
   b. Send current $K, V$ to GPU $i+1$; receive $K, V$ from GPU $i-1$.
   c. Update running Softmax normalization and weighted sum.
3, Route tokens to experts using a fused Top-K routing kernel.

**5. BDH Alignment (The la-Steal)**
Implement a `RingAttention` primitive that decouples sequence length from GPU memory by treating the cluster as a single ring, using asynchronous NCCL primitives for $K, V$ shifting.

---

### 3. GPT-4 (Sparse MoE)

**1. The Physical Bottleneck (First Principles)**
- **The Wall**: HBM Capacity and Weights-to-Compute Ratio. MoE increases parameter count without increasing FLOPs/token.
- **SRAM/HBM Ratio**: High weights-to-compute ratio. Loading expert weights from HBM for a small number of tokens per expert is extremely inefficient.
- **Stall Operation**: Weight loading latency for sparse expert selection (Weight-shuffling).

**2. Mathematical DNA (Sparse MoE Primitive)**
- **Formulae**:
  $$\text{Gating: } G(x) = \text{Softmax}(\text{TopK}(W_g x))$$
  $$\text{Sparse Op: } y = \sum_{j \in \text{TopK}} G(x)_j \cdot \text{MLP}_j(x)$$
- **Tensor Flow**: $[B, S, D] \xrightarrow{\text{Route}} [B, S, K \times D] \xrightarrow{\text{Expert}} [B, S, D]$.
- **Stability**: Z-loss for gating: $\mathcal{L}_z = \sum (\log(G(x)_i))^2$ to prevent logits from drifting.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused MoE kernel. Use "Grouped GEMM" to compute multiple experts of different token counts in one kernel launch.
- **Data Movement**: L2 cache reuse for the gating matrix $W_g$. Weights for selected experts are loaded into SRAM in blocks.
- **Tiling Strategy**: $[16, 16, 16]$ (B, S, D) tiles to fit multiple expert weight blocks in SRAM.

**4. Implementation Algorithm**
1. Compute gating logits $\to$ Top-K selection $\to$ Sort tokens by expert ID.
2. Group tokens into batches for each selected expert.
3. Launch Grouped GEMM kernel:
   a. Load expert weights $W_{exp}$ into SRAM.
   b. Perform GEMM for the token batch.
4. Unsort tokens back to original sequence order and apply gating weights.

**5. BDH Alignment (The la-Steal)**
Implement a `GroupedGEMM` operator in BDH that accepts a list of matrices with different dimensions to compute MoE experts efficiently without padding.

---

### 4. DeepSpeed (ZeRO-3)

**1. The Physical Bottleneck (First Principles)**
- **The Wall**: GPU Memory (VRAM). Optimizer states and gradients consume $\approx 75\%$ of memory in large models.
- **SRAM/HBM Ratio**: Communication-to-compute ratio. All-Gathering parameters for every layer creates a bottleneck.
- **Stall Operation**: $\text{All-Gather} \to \text{GEMM}$ synchronization point.

**2. Mathematical DNA (ZeRO-3 Primitive)**
- **Formulae**:
  $$\text{Partition: } W = \{W_1, W_2, \dots, W_N\} \text{ where } \sum W_i = W$$
  $$\text{Reconstruction: } W = \bigoplus_{i=1}^N \text{AllGather}(\{W_i\}) \rightarrow W \in \mathbb{R}^{M \times D}$$
  $$\text{Gradient Sharding: } \nabla W_i = \text{ReduceScatter}(\nabla W_{full}) \rightarrow \nabla W_i \in \mathbb{R}^{\frac{M}{N} \times D}$$
- **Tensor Flow**: $[W_{part}] \xrightarrow{\text{All-Gather}} [W_{full}] \xrightarrow{\text{GEMM}} [X] \xrightarrow{\text{Reduce-Scatter}} [G_{part}]$.
- **Stability**: Mixed-precision FP16/BF16 with FP32 master weights stored in partitioned state.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused All-Gather and GEMM. Use NCCL primitives integrated into the forward pass to overlap weight reconstruction with computation.
- **Data Movement**: CPU-offloading via PCIe. Use pinned memory for `cudaMemcpyAsync` between GPU and Host RAM.
- **Tiling Strategy**: Parameter tiling $[T_{param}, D]$ to gather weights just-in-time for the current layer.

**4. Implementation Algorithm**
1. Partition Weights, Gradients, and Optimizer states across $N$ GPUs.
2. Forward Pass:
   a. All-Gather parameters for Layer $L$.
   b. Compute $\text{Layer}_L(x)$.
   c. Immediate\_Discard($W_l \setminus W_{local}$).
3. Backward Pass:
   a. All-Gather parameters for Layer $L$.
   b. Compute gradients $G_L$.
   c. Reduce-Scatter $G_L$ to update partitioned states.

**5. BDH Alignment (The la-Steal)**
Implement a `PartitionedParameter` class that triggers a background All-Gather call before the tensor is needed by the compute graph.

---

### 5. Megatron-LM (Tensor Parallelism)

**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Inter-GPU Bandwidth (NVLink). TP requires multiple synchronizations per Transformer block.
- **SRAM/HBM Ratio**: High communication overhead. Every GEMM is followed by an All-Reduce or All-Gather.
- **Stall Operation**: $\text{All-Reduce}$ on the output of the MLP block.

**2. Mathematical DNA (TP Primitive)**
- **Formulae**:
  $$\text{Column Parallel: } Y = X [W_1 | W_2] = [XW_1 | XW_2]$$
  $$\text{Row Parallel: } Y = [X_1 | X_2] \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1 W_1 + X_2 W_2$$
- **Tensor Flow**: $[B, S, D] \xrightarrow{\text{Col}} [B, S, D/N] \xrightarrow{\text{Row}} [B, S, D]$.
- **Stability**: Scaling factors applied to gradients in distributed GEMMs to maintain variance.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused GEMM-AllReduce. Use NCCL `all_reduce` directly on the output buffer of the GEMM kernel.
- **Data Movement**: Use NVLink for direct P2P access to remote GPU memory to avoid host staging.
- **Tiling Strategy**: Tile GEMM across the $D$ dimension $[T_d, T_s]$ to parallelize the column/row split.

**4. Implementation Algorithm**
1. Split Weight $W_{mlp}$ into $N$ columns across GPUs.
2. Local GEMM: Compute $Y_i = X W_{col\_i}$.
3. Split Weight $W_{out}$ into $N$ rows across GPUs.
4. Local GEMM: Compute $Z_i = Y_i W_{row\_i}$.
5. All-Reduce: Sum $Z = \sum Z_i$ across all GPUs to get the final output.

**5. BDH Alignment (The la-Steal)**
Implement `ColumnParallelLinear` and `RowParallelLinear` modules that automatically inject NCCL All-Reduce calls into the output of the row-parallel layer.

---

## [POST-TRANSFORMER] la-Primitives

### 53. Mamba (S6)
**1. The Physical Bottleneck**
- **The Wall**: Transitions from $\mathcal{O}(L^2)$ memory wall (Attention KV-cache) to $\mathcal{O}(L)$ state-wall. The bottleneck shifts from quadratic memory to the sequential dependency of the state update $\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t x_t$.
- **Stall Operation**: The sequential scan. Conventional RNNs stall GPUs because $\mathbf{h}_t$ must be materialized in HBM between steps, causing memory-bandwidth saturation.
- **TMA Specification**: Asynchronous copies of $\Delta, \mathbf{B}, \mathbf{C}$ tensors from HBM to SRAM using Hopper TMA to hide memory latency during the selective scan computation.

**2. Mathematical DNA**
- **Exhaustive LaTeX**:
  $$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t \mathbf{h}_t$$
  $$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A}), \quad \bar{\mathbf{B}}_t = (\Delta_t \mathbf{A})^{-1}(\exp(\Delta_t \mathbf{A}) - \mathbf{I})\mathbf{B}_t$$
- **Tensor Flow**: $[B, L, D] \xrightarrow{\text{Linear Proj}} [B, L, D \cdot 3] \xrightarrow{\text{SRAM Scan}} [B, L, D]$.
- **Stability**: $\mathbf{A}$ is initialized as a diagonal matrix with negative values; $\exp(\Delta \mathbf{A})$ ensures the spectral radius $\rho(\bar{\mathbf{A}}) \leq 1$, preventing divergence.

**3. Hardware-Aware Implementation**
- **Parallel Scan**: Implements an associative scan in Triton. Instead of materializing $\mathbf{h}_t$, it computes the prefix product of $\bar{\mathbf{A}}_t$ and the sum of $\bar{\mathbf{B}}_t x_t$ in SRAM.
- **Tiling Strategy**: Tiles along the sequence length $L$. Each SM processes a block of $L_{tile}$, performing a local scan and then a global synchronization via the scan-reduce pattern.
- **WGMMA Integration**: Linear projections for $\Delta, \mathbf{B}, \mathbf{C}$ are implemented using `wgmma.mma_async` instructions to maximize throughput on H100 Tensor Cores.

**4. Implementation Algorithm**
$\text{HBM Load } (x, \Delta, \mathbf{B}, \mathbf{C}) \xrightarrow{\text{mbarrier}} \text{SRAM Local Scan} \rightarrow \text{Global Prefix Sum} \rightarrow \text{SRAM Update } \mathbf{h}_t \rightarrow \text{SRAM Store } y_t \rightarrow \text{HBM Store } y$.

**5. BDH la-Steal**
`SelectiveStateScan`: A fused CUDA/Triton kernel that performs an input-dependent associative scan without materializing the intermediate state in HBM.

---

### 54. S4 (Structured State Space)
**1. The Physical Bottleneck**
- **The Wall**: $\mathcal{O}(L^2)$ Attention vs $\mathcal{O}(L \log L)$ Convolutional training and $\mathcal{O}(L)$ Recurrent inference.
- **Stall Operation**: The FFT operation during training. While faster than attention, the transition between time-domain and frequency-domain creates synchronization barriers.
- **TMA Specification**: Blocked movement of the SSM kernel $\mathbf{K}$ and input $\mathbf{x}$ into SRAM for point-wise multiplication in the Fourier domain.

**2. Mathematical DNA**
- **Exhaustive LaTeX**:
  $$\dot{\mathbf{h}}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}\mathbf{h}(t)$$
  $$\mathbf{y} = \mathbf{K} * \mathbf{x}, \quad \mathbf{K} = (\mathbf{C} \bar{\mathbf{A}}^0 \bar{\mathbf{B}}, \mathbf{C} \bar{\mathbf{A}}^1 \bar{\mathbf{B}}, \dots, \mathbf{C} \bar{\mathbf{A}}^{L-1} \bar{\mathbf{B}})$$
- **Tensor Flow**: $[B, L, D] \xrightarrow{\text{FFT}} [B, L, D] \xrightarrow{\text{Pointwise}} [B, L, D] \xrightarrow{\text{IFFT}} [B, L, D]$.
- **Stability**: Uses HiPPO (High-Order Polynomial Projection Operators) matrices for $\mathbf{A}$ to ensure long-range memory without gradient vanish/explosion.

**3. Hardware-Aware Implementation**
- **Parallel Scan logic**: Not used; replaced by the Convolution Theorem $\mathcal{F}(\mathbf{K} * \mathbf{x}) = \mathcal{F}(\mathbf{K}) \cdot \mathcal{F}(\mathbf{x})$.
- **Tiling Strategy**: Standard cuFFT tiling. The SSM kernel is pre-computed and stored as a frequency-domain filter to avoid redundant FFTs.
- **WGMMA integration**: Linear projections for the state-space parameters are handled via standard WGMMA matmuls.

**4. Implementation Algorithm**
$\text{HBM Load } (x, \mathbf{K}_{freq}) \xrightarrow{\text{TMA}} \text{SRAM FFT}(x) \rightarrow \text{SRAM Pointwise Mul} \rightarrow \text{SRAM IFFT} \rightarrow \text{HBM Store } y$.

**5. BDH la-Steal**
`SSMConvolution`: A frequency-domain filter implementation that applies a pre-computed HiPPO-based kernel via FFT.

---

### 55. RWKV
**1. Physical Bottleneck**
- **The Wall**: Eliminates the KV-cache entirely. Moves from $\mathcal{O}(L^2)$ to $\mathcal{O}(L)$ time and $\mathcal{O}(1)$ space for inference.
- **Stall Operation**: Element-wise state updates $\mathbf{s}_t = \mathbf{s}_{t-1} \odot \mathbf{w}_{t-1} + \mathbf{k}_t \mathbf{v}_t$ are memory-bound (low arithmetic intensity).
- **TMA Specification**: Asynchronous loading of weight vectors $\mathbf{W}_r, \mathbf{W}_k, \mathbf{W}_v$ to SRAM to overlap with the state update computation.

**2. Mathematical DNA**
- **Exhaustive LaTeX**:
  $$w_t = \sigma(x_t \mathbf{W}_r), \quad k_t = x_t \mathbf{W}_k, \quad v_t = x_t \mathbf{W}_v$$
  $$\mathbf{s}_t = \mathbf{s}_{t-1} \odot \mathbf{w}_{t-1} + \mathbf{k}_t \mathbf{v}_t, \quad y_t = \mathbf{s}_t \mathbf{W}_o$$
- **Tensor Flow**: $[B, L, D] \xrightarrow{\text{Linear Proj}} [B, L, D] \xrightarrow{\text{Recurrent Update}} [B, L, D]$.
- **Stability**: Weight decay on projections and specific initialization of the time-decay vector $\mathbf{w}$ ensure stability over infinite sequences.

**3. Hardware-Aware Implementation**
- **Parallel Scan logic**: Implemented as a Linear Attention form during training $\sum \exp(\dots)$ to allow parallelization.
- **Tiling Strategy**: Tiling along the batch $B$ and dimension $D$. State $\mathbf{s}$ is kept in registers as much as possible.
- **WGMMA integration**: Projections $\mathbf{W}_r, \mathbf{W}_k, \mathbf{W}_v$ are computed using WGMMA to saturate Tensor Core utilization.

**4. Implementation Algorithm**
$\text{HBM Load } (x, \mathbf{W}) \xrightarrow{\text{TMA}} \text{SRAM WGMMA Proj} \rightarrow \text{SRAM Element-wise Recurrence} \rightarrow \text{SRAM Store } y \rightarrow \text{HBM}$.

**5. BDH la-Steal**
`LinearAttentionRecurrence`: A fused kernel that converts a linear attention operation into a constant-memory RNN state update.

---

### 56. RetNet (Retention Network)
**1. The Physical Bottleneck**
- **The Wall**: Breaks the trade-off between Parallel training (Transformer) and Recurrent inference (RNN).
- **Stall Operation**: The decay-masked matrix multiplication in parallel mode $\mathbf{S} = (Q K^T \odot \Gamma) V$ is $\mathcal{O}(L^2)$.
- **TMA Specification**: Moving blocks of $Q, K, V$ into SRAM and applying the $\gamma^{i-j}$ decay factor on-the-fly without materializing the full matrix $\Gamma$.

**2. Mathematical DNA**
- **Exhaustive LaTeX**:
  $$\text{Parallel: } \text{Ret}(X) = (Q K^T \odot \Gamma) V, \quad \Gamma_{ij} = \gamma^{i-j} \text{ for } i \geq j$$
  $$\text{Recurrent: } \mathbf{h}_t = \gamma \mathbf{h}_{t-1} + k_t v_t, \quad y_t = q_t \mathbf{h}_t$$
- **Tensor Flow**: $[B, L, D] \xrightarrow{\text{Proj}} [B, L, D] \xrightarrow{\text{Decay-Accumulate}} [B, L, D]$.
- **Stability**: The decay factor $\gamma \in (0, 1)$ acts as a natural regularizer, preventing the state $\mathbf{h}_t$ from exploding.

**3. Hardware-Aware Implementation**
- **Parallel Scan logic**: Implemented as a masked matrix multiplication where the mask is a geometric progression.
- **Tiling Strategy**: $Q, K, V$ are tiled. The $\gamma$ decay is applied via a shift-and-multiply operation within the SM to avoid HBM reads.
- **WGMMA integration**: Linear projections for $Q, K, V$ use WGMMA.

**4. Implementation Algorithm**
$\text{HBM Load } (Q, K, V) \xrightarrow{\text{TMA}} \text{SRAM WGMMA Proj} \rightarrow \text{SRAM Decayed-Accumulate} \rightarrow \text{SRAM Store } y \rightarrow \text{HBM}$.

**5. BDH la-Steal**
`DecayedMatrixMul`: A kernel that implements $(Q K^T \odot \Gamma) V$ by computing the decay $\gamma^{i-j}$ inside the inner loop of the matmul.

---

### 57. Hyena Hierarchy
**1. The Physical Bottleneck**
- **The Wall**: $\mathcal{O}(L^2)$ Attention replaced by $\mathcal{O}(L \log L)$ Long Convolution.
- **Stall Operation**: FFT/IFFT transforms. The movement of large tensors between the time and frequency domains causes GPU stalls.
- **TMA Specification**: Asynchronous loading of input $x$ and the filtered kernel $\phi(x)$ into SRAM for point-wise multiplication in the Fourier domain.

**2. Mathematical DNA**
- **Exhaustive LaTeX**:
  $$y = \text{conv}(x, \phi(x)) = \mathcal{F}^{-1}(\mathcal{F}(x) \cdot \mathcal{F}(\phi(x)))$$
  $$\phi(x) = \text{MLP}(x) \text{ expanded to length } L$$
- **Tensor Flow**: $[B, L, D] \xrightarrow{\text{MLP}} [B, L, D] \xrightarrow{\text{FFT}} [B, L, D] \xrightarrow{\text{Pointwise}} [B, L, D] \xrightarrow{\text{IFFT}} [B, L, D]$.
- **Stability**: Stability is maintained via normalization layers and the bounded nature of the FFT-based point-wise product.

**3. Hardware-Aware Implementation**
- **Parallel Scan logic**: Not used. Relies on the Convolution Theorem.
- **Tiling Strategy**: FFT is performed over the sequence length $L$. Tiling is used to fit the point-wise multiplication of $\mathcal{F}(x)$ and $\mathcal{F}(\phi(x))$ into SRAM.
- **WGMMA integration**: The MLP generating the filter $\phi(x)$ is implemented using WGMMA.

**4. Implementation Algorithm**
$\text{HBM Load } (x) \xrightarrow{\text{TMA}} \text{SRAM MLP} \rightarrow \text{HBM Store } \phi(x) \rightarrow \text{SRAM FFT}(x, \phi(x)) \rightarrow \text{SRAM Pointwise Mul} \rightarrow \text{SRAM IFFT} \rightarrow \text{HBM Store } y$.

**5. BDH la-Steal**
`FFTConvolution`: A fused operator that computes $\mathcal{F}^{-1}(\mathcal{F}(x) \cdot \mathcal{F}(\phi(x)))$ with optimized memory layouts to minimize FFT overhead.

---

## [SOTA FOCUS] Quantization & Efficiency la-Primitives

### 1. BitNet 1.58b (1.58-bit Ternary LLMs)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Arithmetic Intensity Wall. By reducing weights to $\{-1, 0, 1\}$, we eliminate floating-point multiplications, but the compute-to-memory ratio drops precipitously. The bottleneck shifts from HBM bandwidth to **SRAM-to-Register Unpacking Throughput**.
- **SRAM/HBM Ratio**: Weights are packed (e.g., 2 bits per element). Memory traffic is reduced by $16\times$ vs BF16. However, the GPU must unpack these into a format usable by Tensor Cores (INT8), introducing a latency penalty in the load-compute pipeline.
- **Stall Operation**: `LDSM` (Load Shared Memory) followed by custom bit-shift/masking operations to expand ternary values to INT8.

**2. Mathematical DNA (Ternary GEMM)**
- **Formulae**:
  $$W \in \{-1, 0, 1\}^{M \times N}, \quad X \in \mathbb{R}^{B \times M}$$
  $$Y = \text{quant}_{int8}(X) \cdot W_{int8}$$
  $\text{where } W_{int8} \text{ is the expanded version of } W$.
- **Tensor Flow**: $[B, M] \xrightarrow{\text{quant}} [B, M]_{int8} \xrightarrow{\text{TMA\_Packed\_W}} [T_x, T_y]_{SRAM} \xrightarrow{\text{Unpack}} [T_x, T_y]_{int8} \xrightarrow{\text{WGMMA}} [B, N]_{int32}$.
- **Stability**: Uses a specialized scaling factor $\gamma$ per channel to maintain dynamic range: $Y = \gamma \sum (X_{int8} \cdot W_{int8})$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused Unpack-GEMM. Use TMA to load packed 2-bit weights. In the register file, use bitwise `AND` and `OR` to expand to $\{-1, 0, 1\}$.
- **WGMMA Logic**: Map the ternary GEMM to `wgmma.mma_async` using INT8 inputs. This leverages the high-throughput INT8 Tensor Cores of the H100.
- **Tiling Strategy**: Tile size $[64, 128]$ for weights. Use a shared memory buffer to store the expanded INT8 weights temporarily to avoid repeated unpacking.

**4. Implementation Algorithm**
1. TMA Load packed weights $W_{packed}$ into SRAM.
2. Load and quantize $X \to X_{int8}$.
3. Parallel Unpack: SMs expand $W_{packed} \to W_{int8}$ in registers.
4. Execute `wgmma.mma_async` $(X_{int8}, W_{int8})$.
5. Apply scaling factor $\gamma$ and dequantize to BF16.

**5. BDH la-Steal**
Implement `BitLinearForward`: A Triton kernel that performs 2-bit unpacking and INT8 GEMM in a single fused pass, minimizing SRAM occupancy.

---

### 2. AWQ (Activation-aware Weight Quantization)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Activation Outlier Wall. Standard quantization (e.g., RTN) fails because a small percentage of activation channels have massive magnitudes, causing high quantization error in the corresponding weight channels.
- **SRAM/HBM Ratio**: The bottleneck is the **Scaling Factor Application**. Weights are stored in INT4, but must be scaled by a floating-point vector $s$ before/after the GEMM.
- **Stall Operation**: Per-channel scaling in SRAM.

**2. Mathematical DNA (Weight Scaling)**
- **Formulae**:
  $$\text{Scaling: } W' = W \cdot \text{diag}(s)^{-1}, \quad X' = X \cdot \text{diag}(s)$$
  $$\text{where } s = \text{avg}(|X|) \text{ for each channel.}$$
  $$Y = X' W' = (X \cdot \text{diag}(s)) (W \cdot \text{diag}(s)^{-1})$$
- **Tensor Flow**: $[B, M] \xrightarrow{\text{Scale}} [B, M]_{float} \xrightarrow{\text{WGMMA}} [B, N]_{int} \xrightarrow{\text{Dequant}} [B, N]_{float}$.
- **Stability**: By shifting the "difficulty" of quantization from weights to activations, we preserve the signal of salient channels.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused Scaling-Dequant GEMM. Use TMA to load INT4 weights and the FP16 scale vector $s$.
- **WGMMA Logic**: Perform the GEMM in INT8/INT4. The scaling $X \cdot s$ is fused into the load phase (TMA $\to$ Register).
- **Tiling Strategy**: Store scale vector $s$ in L1 cache/SRAM to avoid HBM round-trips during the inner loop.

**4. Implementation Algorithm**
1. Load $X$ and scale $s$ $\to$ compute $X' = X \cdot s$.
2. TMA Load INT4 weights $W_{int4}$ and scale $s^{-1}$.
3. WGMMA $(X', W_{int4}) \to Y_{int}$.
4. Apply $s^{-1}$ to $Y_{int}$ and cast to BF16.

**5. BDH la-Steal**
Implement `AWQFusedLinear`: A kernel that integrates the channel-wise scaling into the WGMMA pipeline, eliminating a separate scaling pass.

---

### 3. SmoothQuant (Outlier Migration)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Dynamic Range Wall. Activations in LLMs exhibit extreme outliers that make INT8 quantization impossible without severe accuracy loss.
- **SRAM/HBM Ratio**: Similar to AWQ, but the scaling is performed *offline* (pre-computed). The bottleneck is the **Element-wise Scale Application** during the forward pass.
- **Stall Operation**: `FMUL` (Floating point multiply) for every activation element.

**2. Mathematical DNA (Difficulty Migration)**
- **Formulae**:
  $$s = \max(|X|) / \text{threshold}$$
  $$X_{smooth} = X \cdot \text{diag}(s)^{-1}, \quad W_{smooth} = \text{diag}(s) \cdot W$$
- **Tensor Flow**: $[B, M] \xrightarrow{\text{Scale}} [B, M]_{int8} \xrightarrow{\text{WGMMA}} [B, N]_{int8} \xrightarrow{\text{Dequant}} [B, N]_{bf16}$.
- **Stability**: The migration is a mathematically identity operation $Y = (X s^{-1}) (s W)$, but it balances the quantization error across $X$ and $W$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Per-channel scaling fused with quantization.
- **WGMMA Logic**: Since both $X$ and $W$ are now smooth, we can use standard `wgmma.mma_async` with INT8.
- **Tiling Strategy**: $[128, 128]$ tiles. The scale vector $s$ is loaded once per tile and kept in registers.

**4. Implementation Algorithm**
1. Apply pre-computed scale $s^{-1}$ to $X \to X_{int8}$.
2. TMA Load pre-smoothed $W_{int8}$.
3. WGMMA $(X_{int8}, W_{int8}) \to Y_{int32}$.
4. Scale $Y_{int32}$ by the combined scale factor.

**5. BDH la-Steal**
Implement `SmoothQuantLinear`: An INT8 linear operator that assumes pre-smoothed weights, focusing on maximizing Tensor Core utilization via WGMMA.

---

### 4. GPTQ (Post-Training Quantization)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Hessian Computation Wall. Unlike AWQ, GPTQ requires calculating the inverse Hessian of the weights to minimize the MSE of the output.
- **SRAM/HBM Ratio**: High memory pressure during the quantization phase (offline). During inference, it's a standard weight-only quantization problem.
- **Stall Operation**: The layer-wise weight update loop during quantization.

**2. Mathematical DNA (Hessian-based Update)**
- **Formulae**:
  $$W_{new} = W_{quant} + (W - W_{quant}) \cdot (H^{-1})_{ii}^{-1} \cdot (W - W_{quant})^T$$
  $\text{where } H = \mathbb{E}[XX^T]$ (Hessian).
- **Tensor Flow**: $\text{Weights} \xrightarrow{\text{Hessian Update}} \text{INT4 Weights}$.
- **Stability**: Ensures the quantized weight matrix minimizes $\| W X - W_{quant} X \|_2^2$ for a given calibration set $X$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: For inference, use a Weight-Only INT4 GEMM.
- **WGMMA Logic**: Use `wgmma` with a dequantization step (INT4 $\to$ BF16) just before the multiplication.
- **Tiling Strategy**: Tile $W$ in $[T_x, T_y]$ and dequantize on-the-fly in SRAM.

**4. Implementation Algorithm**
1. Load INT4 weights $W_{int4}$ and zero-points $Z$.
2. Dequantize $W_{int4} \to W_{bf16}$ in SRAM.
3. WGMMA $(X_{bf16}, W_{bf16}) \to Y_{bf16}$.

**5. BDH la-Steal**
Implement `GPTQInferenceLinear`: A highly optimized Weight-Only INT4 GEMM that uses TMA to load packed weights and performs on-the-fly dequantization in registers.

---

### 5. LoRA/QLoRA (Low-Rank Adaptation)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Memory Wall for Adapter Weights. In QLoRA, the base model is frozen in 4-bit, and small adapters (A, B) are trained in BF16.
- **SRAM/HBM Ratio**: The bottleneck is the **Dual-Path Compute**. We must compute $X W_0$ and $X A B$ and sum them.
- **Stall Operation**: The synchronization point where the base model output is added to the adapter output.

**2. Mathematical DNA (Low-Rank Update)**
- **Formulae**:
  $$Y = X (W_0 + AB) = X W_0 + X A B$$
  $\text{where } A \in \mathbb{R}^{M \times r}, B \in \mathbb{R}^{r \times N} \text{ with } r \ll M, N$.
- **Tensor Flow**: $[B, M] \xrightarrow{\text{Base GEMM}} [B, N] \text{ and } [B, M] \xrightarrow{A} [B, r] \xrightarrow{B} [B, N]$.
- **Stability**: Since $W_0$ is frozen, gradients only flow through $A$ and $B$, preventing catastrophic forgetting.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Kernel Design**: Fused LoRA Kernel. Instead of two separate GEMMs, fuse $X A B$ into a single operation if $r$ is small.
- **WGMMA Logic**: $X W_0$ uses WGMMA (INT4/BF16). $X A B$ uses smaller WGMMA or standard GEMM if $r < 64$.
- **Tiling Strategy**: Overlap the computation of $X W_0$ with $X A$ using CUDA streams.

**4. Implementation Algorithm**
1. Compute $Y_{base} = \text{WGMMA}(X, W_0)$.
2. Compute $Y_{adapter} = \text{WGMMA}(\text{WGMMA}(X, A), B)$.
3. Element-wise add $Y = Y_{base} + Y_{adapter}$.

**5. BDH la-Steal**
Implement `FusedQLoRALinear`: A kernel that performs the base INT4 GEMM and the BF16 adapter GEMMs in a single pipeline, storing the intermediate $X A$ result in SRAM to avoid HBM write-back.

---

## [SOTA FOCUS] Engineering Deep-Dives (Company Articles)

### 1. Asynchronous FP8 GEMM (TMA-Accelerated)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Global Memory (GMEM) $\rightarrow$ Shared Memory (SMEM) bandwidth. Primary stall: `mbarrier` wait for tensor tile arrival.
- **SRAM/HBM Ratio**: The critical tuning parameter; ratio of tile size to HBM bandwidth determines the overlap efficiency.

**2. Mathematical DNA**
- **Formulae**: $C_{M \times N} = A_{M \times K} \times B_{K \times N} + C_{M \times N}$.
- **Shapes**: Tiled into blocks $bM \times bK$ and $bN \times bK$.
- **Precision**: Inputs $A, B \in \{E4M3, E5M2\}$ (FP8); Accumulator $C \in \{FP16, FP32\}$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **TMA**: Use `cp.async.bulk.tensor` for 2D tiles.
- **WGMMA**: Warp-group level MMA instructions (`wgmma.mma_async`) for compute.
- **SRAM Tiling**: Swizzled layouts in SMEM to eliminate bank conflicts.
- **Warp Specialization**: Producer warp (issues TMA) $\rightarrow$ Consumer warp group (executes WGMMA).

**4. Implementation Algorithm**
1. **Host**: Encode `cuTensorMap` descriptors for $A, B, C$ (Base pointer, shape, stride).
2. **Producer**: Leader thread issues TMA load for next tile into ping-pong SMEM buffer.
3. **Consumer**: Execute `wgmma` on current tile $\rightarrow$ signal `mbarrier` upon completion.
4. **Sync**: Producer waits for consumer signal before overwriting buffer.

**5. BDH la-Steal**
`TMA_FP8_GEMM`: Asynchronous tensor core operator utilizing Warp-Group Specialization for $\sim 2\text{x}$ throughput over cuBLAS FP16 in mid-batch regimes.

---

### 2. Multi-head Latent Attention (MLA)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: KV Cache memory bandwidth (HBM $\rightarrow$ SMEM).
- **Bottleneck**: Quadratic growth of KV cache in standard MHA.

**2. Mathematical DNA**
- **Compression**: $c_{KV} = W_{down} \times [k, v]^T$ (Low-rank projection).
- **Reconstruction**: $k = W_{up\_k} \times c_{KV}$; $v = W_{up\_v} \times c_{KV}$.
- **Attention**: $\text{Softmax}(\frac{q \times k^T}{\sqrt{d}}) \times v$.
- **Shape**: Reduces cache from $n_{heads} \times d_{head}$ to $d_{latent}$ per token.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **KV Store**: Store only $c_{KV}$ in HBM.
- **On-the-fly Projection**: Decompress $c_{KV} \rightarrow [k, v]$ inside SMEM using a small GEMM just before the attention core.

**4. Implementation Algorithm**
1. Compress $K, V$ into a single latent vector per token.
2. During decode, fetch $c_{KV}$ for the sequence.
3. Project $c_{KV} \rightarrow K, V$ using $W_{up}$ weights stored in registers/SMEM.
4. Compute scaled dot-product attention.

**5. BDH la-Steal**
`LATENT_ATTN_KERNEL`: Attention operator achieving $>90\%$ KV cache reduction while maintaining MHA expressivity.

---

### 3. PagedAttention (Block-Based KV Management)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Memory fragmentation and pointer-chasing latency.
- **Stall**: Non-contiguous HBM access during KV gather.

**2. Mathematical DNA**
- **Addressing**: $\text{PhysicalAddr} = \text{BlockTable}[\text{LogicalBlock}] \times \text{BlockSize} + \text{Offset}$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Paging**: KV cache allocated in fixed-size blocks (e.g., 16 tokens).
- **Block Table**: Indirection table mapping logical sequence indices to physical HBM blocks.
- **Gather**: Custom CUDA kernel to pack non-contiguous blocks into contiguous SMEM tiles.

**4. Implementation Algorithm**
1. Allocate physical blocks from a global pool.
2. Map logical tokens to blocks via a per-request block table.
3. In attention kernel, use the table to calculate the address of the $i$-th token's $K, V$.
4. Load blocks asynchronously into SMEM $\rightarrow$ Compute Attention.

**5. BDH la-Steal**
`PAGED_KV_GATHER`: High-efficiency gather-operator for non-contiguous KV cache on H100.

---

### 4. Distributed Shared Memory (DSM) Ring-Copy
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: SM-to-SM communication latency.
- **Bottleneck**: Global memory round-trips for inter-block synchronization.

**2. Mathematical DNA**
- **Addressing**: $Addr_{remote} = \text{ClusterMap}(Addr_{local}, Rank_{target})$.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Cluster**: Thread Block Clusters (guaranteed concurrent scheduling on GPC).
- **Instruction**: `map_shared_rank` (PTX) for direct remote SMEM access.
- **Network**: Hardware-accelerated SM-to-SM network.

**4. Implementation Algorithm**
1. Launch kernel with `cluster_size > 1`.
2. Arrange blocks in a logical ring.
3. Block $R$ writes data to SMEM of Block $(R+1) \pmod{C}$.
4. Use `asynchronous transaction barriers` to signal data arrival across the cluster.

**5. BDH la-Steal**
`DSM_RING_EXCHANGE`: Low-latency inter-SM data exchange primitive for Ring-Attention or MoE routing.

---

### 5. Grouped GEMM (MoE Expert Routing)
**1. The Physical Bottleneck (First Principles)**
- **The Wall**: Load imbalance (Expert Skew) and under-utilization of Tensor Cores for small $M$ (tokens per expert).

**2. Mathematical DNA**
- **Formulae**: $\text{Out} = \text{Concat}(\text{GEMM}_1, \dots, \text{GEMM}_E)$, where $M_i$ is variable.

**3. Hardware-Aware Implementation (CUDA/Triton)**
- **Dynamic Scheduling**: Distribute variable-sized GEMMs across SMs using a global task queue.
- **WGMMA**: Use `wgmma` to maximize throughput for narrow tiles.

**4. Implementation Algorithm**
1. Route tokens to experts $\rightarrow$ Sort/Reorder tokens to create contiguous expert-specific buffers.
2. Calculate $M_i, N_i, K_i$ for each expert.
3. Execute Grouped GEMM: SMs cooperatively process multiple experts, switching tasks once a GEMM is complete.
4. Permute output tokens back to original sequence order.

**5. BDH la-Steal**
`GROUPED_WGMMA_OP`: MoE-specific GEMM operator that handles variable-batch sizes with minimal padding overhead.

---

## [SOTA FOCUS] Fundamental Implementation la-Primitives (Academic/Books)

### 1. Sliding Window Attention (SWA)
- **Physical Bottleneck**: HBM bandwidth (Memory-bound). SRAM capacity limits the window size $W$ per block.
- **Mathematical DNA**: $\text{Attn}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d}}) V$ where $K, V$ are restricted to indices $[i-W, i]$. Shape: $[B, H, L, d] \to [B, H, L, d]$.
- **Hardware-Aware Implementation**: Triton tiling with a diagonal mask. Block-sparse layout to avoid loading zero-valued blocks.
- **Implementation Algorithm**: 
    1. Partition $Q$ into blocks of size $B_q$.
    2. Load $K, V$ blocks within range $[j-W, j+W]$.
    3. Compute dot-product in SRAM.
    4. Apply softmax and weighted sum.
- **BDH la-Steal**: `SWA_Fused_Kernel`

---

### 2. Sparse Attention (Block-Sparse)
- **Physical Bottleneck**: Memory fragmentation and indirect addressing overhead. SRAM stalls during non-contiguous memory access.
- **Mathematical DNA**: $A_{ij} = 0$ if $(i, j) \notin \mathcal{S}$ where $\mathcal{S}$ is the sparsity pattern. $\text{Output} = \text{softmax}(M \odot \frac{QK^T}{\sqrt{d}}) V$.
- **Hardware-Aware Implementation**: Block-sparse matrices (e.g., $16 \times 16$ tiles). Use bitmasks to skip zero blocks in CUDA warps.
- **Implementation Algorithm**:
    1. Generate block-sparsity mask $\mathcal{S}$.
    2. Iterate over non-zero blocks in $\mathcal{S}$.
    3. Load corresponding $Q, K, V$ tiles.
    4. Compute local softmax and accumulate.
- **BDH la-Steal**: `BlockSparse_Attention`

---

### 3. RMSNorm (Root Mean Square Layer Normalization)
- **Physical Bottleneck**: Memory-bound. High HBM R/W ratio for element-wise scaling.
- **Mathematical DNA**: $\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$. Shape: $[L, d] \to [L, d]$.
- **Hardware-Aware Implementation**: Fused CUDA kernel. Single-pass reduction for sum-of-squares using warp shuffles.
- **Implementation Algorithm**:
    1. Load $x$ vector into SRAM.
    2. Compute $\sum x^2$ using `__shfl_down_sync`.
    3. Calculate $1/\sqrt{\text{mean} + \epsilon}$.
    4. Element-wise multiply by scale and $\gamma$.
- **BDH la-Steal**: `Fused_RMSNorm`

---

### 4. DeepNorm
- **Physical Bottleneck**: Gradient instability at scale. Vanishing/exploding gradients in deep residual chains.
- **Mathematical DNA**: $x_{l+1} = \text{LayerNorm}(\alpha \cdot x_l + \text{Sublayer}(x_l))$. $\alpha$ is a scaling constant determined by model depth.
- **Hardware-Aware Implementation**: Integration of $\alpha$ into the residual addition kernel to minimize memory passes.
- **Implementation Algorithm**:
    1. Compute sublayer output.
    2. Scale residual $x_l$ by $\alpha$.
    3. Sum and apply LayerNorm.
- **BDH la-Steal**: `DeepNorm_Residual_Block`

---

### 5. SwiGLU (Swish-Gated Linear Unit)
- **Physical Bottleneck**: Compute-bound (GEMM) followed by memory-bound (Activation). High SRAM pressure from intermediate tensor $x\sigma(x)$.
- **Mathematical DNA**: $\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \otimes (xV + c)$. Shape: $[L, d] \to [L, d_{ff})$.
- **Hardware-Aware Implementation**: Fused GEMM + Activation. Compute $\text{Swish}(xW)$ and $(xV)$ in a single kernel to avoid writing intermediate results to HBM.
- **Implementation Algorithm**:
    1. Compute linear projections $W$ and $V$.
    2. Apply $\text{SiLU}(x) = x \cdot \text{sigmoid}(x)$ to $W$-output.
    3. Element-wise multiply by $V$-output.
- **BDH la-Steal**: `Fused_SwiGLU`

---

### 6. GeLU (Gaussian Error Linear Unit)
- **Physical Bottleneck**: Transcendental function overhead ($\text{erf}, \tanh$). Instruction latency in CUDA cores.
- **Mathematical DNA**: $\text{GeLU}(x) = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
- **Hardware-Aware Implementation**: Approximation using $\tanh$ to avoid expensive $\text{erf}$ calls. Vectorized implementation using `float4` loads.
- **Implementation Algorithm**:
    1. Load $x$ in chunks of 4.
    2. Compute cubic term $x^3$.
    3. Apply $\tanh$ approximation.
    4. Scale and store result.
- **BDH la-Steal**: `Fast_GeLU`

---

### 7. Weight Initialization & Stable Scaling
- **Physical Bottleneck**: Variance shift across layers leading to saturation or divergence.
- **Mathematical DNA**: $\text{Var}(W) = \frac{2}{d_{in} + d_{out}}$ (Xavier) or $\text{Var}(W) = \frac{2}{d_{in}}$ (Kaiming). $\text{Scaling} = \frac{1}{\sqrt{2L}}$ for residual branches.
- **Hardware-Aware Implementation**: Precise float32 initialization before casting to BF16/FP16 to prevent underflow.
- **Implementation Algorithm**:
    1. Sample $W \sim \mathcal{N}(0, \sigma^2)$.
    2. Apply $1/\sqrt{2 \cdot \text{layers}}$ multiplier to weights in residual paths.
    3. Normalize initial gradients via Warmup scheduler.
- **BDH la-Steal**: `Stable_Init_Scaling`

---

### 8. Gradient Checkpointing (Recomputation)
- **Physical Bottleneck**: HBM capacity (Memory-bound). Storage of all activations for backprop.
- **Mathematical DNA**: Storage complexity reduced from $\mathcal{O}(L)$ to $\mathcal{O}(\sqrt{L})$.
- **Hardware-Aware Implementation**: Custom autograd functions. Store only boundary activations; recompute internals during backward pass.
- **Implementation Algorithm**:
    1. During forward pass, discard intermediate activations $a_i$ for $i \in (start, end)$.
    2. Store boundary activations $a_{start}$.
    3. During backward pass, re-run forward pass from $a_{start}$ to recover $a_i$.
    4. Compute gradients $\partial \mathcal{L}/\partial a_i$.
- **BDH la-Steal**: `Selective_Checkpoint`

---

### 9. Recomputation (Triton-style)
- **Physical Bottleneck**: Compute vs. Memory trade-off. ALU utilization vs. HBM bandwidth.
- **Mathematical DNA**: $\text{Cost}(\text{Recomp}) < \text{Cost}(\text{HBM\_Load})$.
- **Hardware-Aware Implementation**: Fused kernels that re-calculate activations on-the-fly using register-level caching.
- **Implementation Algorithm**:
    1. Identify memory-intensive operations (e.g., Softmax).
    2. Implement backward pass to re-derive the activation from the output and input.
    3. Avoid writing the activation to HBM entirely.
- **BDH la-Steal**: `OnTheFly_Recompute`



