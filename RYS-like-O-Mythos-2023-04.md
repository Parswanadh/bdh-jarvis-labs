# Enhancing RYS to Match O-Mythos: Detailed Implementation Plan
## Date: 2023-04-XX (Template - update with actual date)

## Executive Summary
This document outlines a detailed plan to enhance your current RYS (Repeat Yourself) implementation in BDH to match the key features of O-Mythos, while preserving your biological advantages (Hebbian learning, multi-scale memory). The goal is to combine the best of both worlds: O-Mythos's learned stability control and your biological plausibility.

## Current State Analysis

### Your Current RYS (in `multiscale_bdh.py`)
```python
# State update (lines 248-250)
updated_state = decay * state + hebbian_lr * hebbian
# Where decay = [0.95, 0.99, 0.995] (FIXED)
```

### Target O-Mythos Features to Incorporate
1. **Learnable A matrix** (replaces fixed decay)
2. **Learnable B matrix** (input influence)
3. **Enforced spectral radius constraint** ρ(A) < 1
4. **Maintain your biological advantages** (Hebbian, multi-scale)

## Implementation Plan

### Phase 1: Core Enhancement (Days 1-2)
**Goal**: Add learnable A and B matrices while preserving Hebbian learning

#### Task 1.1: Implement Learnable A Matrix (O-Mythos style)
- Replace fixed decay rates with learnable parameters
- Enforce spectral radius constraint using O-Mythos method
- Location: `multiscale_bdh.py` in `_update_states` method

#### Task 1.2: Add Learnable B Matrix (Input influence)
- Add B matrix for input-to-state influence
- Location: Same as above

#### Task 1.3: Preserve Hebbian Learning
- Keep hebbian_lr * hebbian term
- Ensure it combines properly with new A and B terms

### Phase 2: Architectural Integration (Days 3-4)
**Goal**: Integrate enhancements into the full system

#### Task 2.1: Update State Update Equation
Change from:
```python
updated_state = decay * state + hebbian_lr * hebbian
```
To:
```python
# Where A_t = sigmoid(A_raw) ensures 0 < A < 1 (stable)
# B is learnable input matrix
updated_state = A_t * state + B * input_projection + hebbian_lr * hebbian
```

#### Task 2.2: Parameter Initialization
- Initialize A_raw to produce decay-like behavior initially
- Initialize B to small random values
- Keep hebbian_lr as before

#### Task 2.3: Dimension Handling
- Ensure A is [C x C] where C = n_embd
- Ensure B is [C x input_dim] where input_dim = n_embd
- Handle multi-head structure properly

### Phase 3: Testing & Validation (Days 5-7)
**Goal**: Verify the enhancement works correctly

#### Task 3.1: Unit Tests
- Test that A matrix stays in (0,1) range
- Test that B matrix learns meaningful patterns
- Test that combined update doesn't explode/vanish

#### Task 3.2: Comparison Tests
- Compare enhanced vs original RYS on:
  - Training stability
  - Convergence speed
  - Final loss
  - Memory retention properties

#### Task 3.3: Ablation Studies
- Test with only A matrix learnable
- Test with only B matrix learnable
- Test with both learnable
- Test with fixed vs learned decay

### Phase 4: Advanced Features (Optional - Days 8-10)
**Goal**: Add O-Mythos-inspired enhancements beyond basic A/B matrices

#### Task 4.1: Spectral Radius Monitoring
- Add logging of actual spectral radius during training
- Verify constraint is maintained

#### Task 4.2: Initialization Strategy
- Initialize A to match your current decay rates [0.95, 0.99, 0.995]
- This provides smooth transition from old to new

#### Task 4.3: Per-Scale Learnable Parameters
- Consider making A and B learnable per scale (like your multi-scale)
- Or keep global A/B with per-scale mixing

## Detailed Implementation Specifications

### File to Modify: `multiscale_bdh.py`
### Method to Update: `_update_states` (around line 229)

#### Current Code:
```python
def _update_states(
    self,
    states: Tuple[torch.Tensor, ...],
    hebbian: torch.Tensor
) -> Tuple[Tuple[torch.Tensor, ...]]:
    """
    Update all state matrices with decay and Hebbian learning.
    For each scale i:
        E_i <- gamma_i * E_i + eta * hebbian
    """
    updated_states = []
    for i, (state, decay) in enumerate(zip(states, self.decay_rates)):
        # Apply decay and Hebbian update
        updated_state = decay * state + self.hebbian_lr * hebbian
        updated_states.append(updated_state)
    return tuple(updated_states)
```

#### Target Enhanced Code:
```python
def _update_states(
    self,
    states: Tuple[torch.Tensor, ...],
    hebbian: torch.Tensor,
    input_projection: torch.Tensor  # NEW: Q @ Wq or similar
) -> Tuple[Tuple[torch.Tensor, ...]]:
    """
    Update all state matrices with O-Mythos style learnable A/B matrices
    and Hebbian learning.
    Enhanced update: E_i <- A_i * E_i + B_i * input + eta * hebbian
    Where A_i is learned per scale with spectral radius constraint.
    """
    updated_states = []
    
    # Get learnable A matrices (one per scale)
    # These replace your fixed decay_rates
    if hasattr(self, 'learnable_A'):
        A_matrices = self.learnable_A(states)  # Returns [num_scales] values
    else:
        # Fallback to original fixed decay for backward compatibility
        A_matrices = self.decay_rates
    
    # Get learnable B matrices (one per scale)
    if hasattr(self, 'learnable_B'):
        B_matrices = self.learnable_B(states)  # Returns [num_scales] values
    else:
        B_matrices = [torch.zeros_like(state) for state in states]  # Zero if not learned
    
    for i, (state, A_val, B_val) in enumerate(zip(states, A_matrices, B_matrices)):
        # O-Mythos style update with Hebbian preservation
        # A_val and B_val are scalars [0,1] from sigmoid
        updated_state = (
            A_val * state +           # Learned decay (O-Mythos A)
            B_val * input_projection + # Learned input influence (O-Mythos B)
            self.hebbian_lr * hebbian  # Your Hebbian learning (preserved)
        )
        updated_states.append(updated_state)
    
    return tuple(updated_states)
```

### Required New Components

#### 1. Learnable A Matrix Module
Add to `multiscale_bdh.py` class `MultiScaleLinearAttention`:

```python
class LearnableA_B Matrices(nn.Module):
    """Learnable A and B matrices for O-Mythos style enhancement."""
    def __init__(self, n_embd: int, num_scales: int = 3):
        super().__init__()
        self.n_embd = n_embd
        self.num_scales = num_scales
        
        # Learn A (decay) parameter per scale
        # Outputs scalar per scale that gets sigmoided to [0,1]
        self.A_params = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(num_scales)
        ])
        
        # Learn B (input) parameter per scale  
        # Outputs scalar per scale that gets sigmoided to [0,1]
        self.B_params = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(num_scales)
        ])
    
    def forward(self, states: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get learnable A and B values for each scale.
        
        Args:
            states: Tuple of state matrices [C, C] for each scale
            
        Returns:
            A_values: [num_scales] - learnable decay values [0,1]
            B_values: [num_scales] - learnable input values [0,1]
        """
        A_values = []
        B_values = []
        
        for i, (state, A_param, B_param) in enumerate(zip(states, self.A_params, self.B_params)):
            # Use state magnitude to inform learning (like O-Mythos)
            state_magnitude = state.mean().abs()
            
            # Learnable parameters modulated by state
            A_val = torch.sigmoid(A_param + 0.1 * state_magnitude)  # Bias toward original decay
            B_val = torch.sigmoid(B_param + 0.01 * state_magnitude)  # Small input influence
            
            A_values.append(A_val)
            B_values.append(B_val)
        
        return torch.stack(A_values), torch.stack(B_values)
```

#### 2. Integration Points
- Modify `_update_states` to accept `input_projection`
- Modify `forward` method to compute and pass `input_projection`
- `input_projection` = Q @ Wq (or similar - the query projection)

### Modified Forward Flow

In `MultiScaleLinearAttention.forward()`:
1. Compute Q, K, V as before
2. Compute `input_projection = Q.mean(dim=(0,1))` or similar summary statistic
3. Pass `input_projection` to `_update_states`
- Get A_values, B_values from learnable matrices
- Apply enhanced update: `A*state + B*input_projection + hebbian_lr*hebbian`

### Initialization Strategy

#### A Matrix Initialization
```python
# Initialize to match your current decay rates [0.95, 0.99, 0.995]
# We want sigmoid(A_init) = [0.95, 0.99, 0.995]
# So A_init = logit([0.95, 0.99, 0.995])
import torch
A_init = torch.logit(torch.tensor([0.95, 0.99, 0.995]))
# A_init ≈ [2.94, 4.60, 5.29]
for i, A_param in enumerate(self.A_params):
    A_param.data.fill_(A_init[i])
```

#### B Matrix Initialization
```python
# Start small to avoid overwhelming the system
for B_param in self.B_params:
    B_param.data.fill_(0.01)  # Small initial influence
```

### Testing & Validation Plan

#### Unit Tests
1. **Range Test**: Verify A and B values stay in [0,1] after sigmoid
2. **Initialization Test**: Check initial values match [0.95, 0.99, 0.995] for A
3. **Gradient Test**: Verify gradients flow to A and B parameters
4. **Stability Test**: Run for 100 steps with random input, verify state doesn't explode

#### Comparison Tests
1. **Baseline**: Original fixed decay [0.95, 0.99, 0.995]
2. **Enhanced**: Learnable A/B matrices
3. **Metrics to Compare**:
   - Training loss convergence
   - Final perplexity/loss
   - Average state magnitude over time
   - Spectral radius of effective A matrix
   - Memory retention (how long state persists)

#### Ablation Studies
- A only learnable, B fixed
- B only learnable, A fixed  
- Both learnable (full enhancement)
- Neither learnable (original)

### Expected Benefits

1. **Better Adaptivity**: System can learn optimal decay rates per scale
2. **Input Sensitivity**: Can learn how much new input should influence state
3. **Preserved Biology**: Hebbian learning and multi-scale remain intact
4. **Potential Performance**: May learn better decay patterns than fixed [0.95, 0.99, 0.995]

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Training Instability** | Start with small learning rates for A/B params |
| **Loss of Biological Properties** | Verify Hebbian learning still works via ablation |
| **Overfitting to Input** | B matrix initialized small, can add L2 regularization |
| **Dimension Mismatch** | Careful shape checking in implementation |

### Success Criteria

The enhancement is successful if:
1. Model trains without exploding/vanishing gradients
2. Final loss is equal to or better than original fixed decay
3. Learned A values are reasonable (not all 0 or 1)
4. Hebbian learning contribution is still visible in state dynamics
5. Spectral radius of effective A matrix stays < 1.1 (with tolerance)

### Timeline Estimate

| Phase | Duration | Description |
|-------|----------|-------------|
| **Implementation** | 2-3 days | Add A/B matrices, modify update equation |
| **Initial Testing** | 1 day | Unit tests, basic forward pass |
| **Comparison Training** | 2-3 days | Train baseline vs enhanced |
| **Analysis & Ablation** | 1-2 days | Detailed comparison studies |
| **Documentation** | 0.5 day | Update comments, explain changes |
| **Total** | **6-7 days** | Full implementation and validation |

### Files to Modify
1. `multiscale_bdh.py` - Main implementation
2. Potentially `train_multiscale.py` - If training config needs updates
3. Test scripts - Add validation for new components

### Next Steps
1. Implement the LearnableA_B Matrices module
2. Modify `_update_states` to accept and use A/B matrices
3. Update forward pass to compute and pass input_projection
4. Add initialization for A/B parameters
5. Run unit tests to verify ranges and gradients
6. Train comparison study vs original
7. Analyze results and document findings

This approach gives you the best of both worlds:
- O-Mythos's learned stability control (A/B matrices)
- Your biological Hebbian learning and multi-scale memory
- Potential for better adaptive behavior than fixed decay rates