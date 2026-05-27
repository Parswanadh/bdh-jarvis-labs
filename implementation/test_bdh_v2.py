"""
BDH v2 Unit Tests
=================

Unit test scaffolding for the BDH v2 architecture.
Tests each component independently and validates the full forward pass.

Run: python -m pytest implementation/test_bdh_v2.py -v
"""

import torch
import pytest
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation.bdh_v2_clean import (
    BDHv2Config,
    BDHv2,
    BDHv2Layer,
    MultiScaleLinearAttention,
    ReLULowRankFFN,
    RotaryPositionEmbedding,
    VocabProjection,
    count_parameters,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Default BDH v2 config (baseline: element-wise, no RoPE, multiplicative gating)."""
    return BDHv2Config(
        vocab_size=32000,
        n_embd=256,
        n_layer=4,
        n_head=4,
        ffn_dim=512,
        dropout=0.0,  # Disable dropout for deterministic tests
    )


@pytest.fixture
def small_model(default_config):
    """Small BDH v2 model for fast tests."""
    return BDHv2(default_config)


@pytest.fixture
def device():
    """Use CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfig:
    def test_default_config_valid(self):
        """Default config should pass validation."""
        config = BDHv2Config()
        assert config.n_embd % config.n_head == 0
        assert len(config.decay_rates) == config.num_scales
        assert abs(sum(config.scale_weights) - 1.0) < 1e-6

    def test_head_dim_computed_correctly(self):
        """head_dim = n_embd / n_head."""
        config = BDHv2Config(n_embd=512, n_head=8)
        assert config.head_dim == 64

    def test_invalid_n_embd_divisible_by_n_head(self):
        """n_embd must be divisible by n_head."""
        with pytest.raises(AssertionError):
            BDHv2Config(n_embd=256, n_head=7)

    def test_decay_rates_length_mismatch(self):
        """decay_rates length must equal num_scales."""
        with pytest.raises(AssertionError):
            BDHv2Config(decay_rates=[0.95, 0.99], num_scales=3)

    def test_scale_weights_must_sum_to_one(self):
        """scale_weights must sum to 1.0."""
        with pytest.raises(AssertionError):
            BDHv2Config(scale_weights=[0.3, 0.3, 0.3], num_scales=3)


# ============================================================================
# RoPE Tests
# ============================================================================


class TestRoPE:
    def test_rope_output_shape(self):
        """RoPE should return cos/sin with correct shape."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        x = torch.randn(2, 32, 4, 64)  # [B, T, H, D]
        cos, sin = rope(x, seq_len=32)
        assert cos.shape == (1, 1, 32, 64)
        assert sin.shape == (1, 1, 32, 64)

    def test_rope_distinguishes_positions(self):
        """RoPE should produce different rotations for different positions."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        x = torch.ones(1, 2, 1, 64)  # Same input at pos 0 and 1
        cos, sin = rope(x, seq_len=2)
        # cos/sin should differ between positions
        assert not torch.allclose(cos[0, 0, 0, :], cos[0, 0, 1, :], atol=1e-5)

    def test_rope_apply_rotary_shape(self):
        """Rotary application should preserve input shape."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
        x = torch.randn(2, 16, 4, 64)
        cos, sin = rope(x, seq_len=16)
        x_rot = rope.apply_rotary(x, cos, sin)
        assert x_rot.shape == x.shape


# ============================================================================
# Multi-Scale Linear Attention Tests
# ============================================================================


class TestMultiScaleLinearAttention:
    def test_forward_shape(self, default_config, device):
        """Forward pass should preserve shape."""
        config = default_config
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 32, config.n_embd, device=device)
        out, states = attn(x)
        assert out.shape == x.shape

    def test_return_states(self, default_config, device):
        """Should return states when requested."""
        config = default_config
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device)
        out, states = attn(x, return_states=True)
        assert states is not None
        assert len(states) == config.num_scales
        assert states[0].shape == (config.n_head, config.head_dim, config.head_dim)

    def test_state_persistence(self, default_config, device):
        """States should persist across forward passes."""
        config = default_config
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device)

        _, states = attn(x, return_states=True)
        _, states2 = attn(x, states=states, return_states=True)

        # States should be updated (not identical to initial zeros)
        for s in states2:
            assert not torch.allclose(s, torch.zeros_like(s), atol=1e-6)

    def test_hebbian_elementwise_shape(self, default_config, device):
        """Element-wise Hebbian should return [H, D, D] diagonal matrix."""
        config = default_config
        config.use_outer_product = False
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device)
        hebbian = attn._compute_hebbian_update(
            attn.Wq(x).view(2, 16, config.n_head, config.head_dim),
            attn.Wv(x).view(2, 16, config.n_head, config.head_dim),
        )
        assert hebbian.shape == (config.n_head, config.head_dim, config.head_dim)

    def test_hebbian_outer_product_shape(self, default_config, device):
        """Outer product Hebbian should return [H, D, D] full matrix."""
        config = default_config
        config.use_outer_product = True
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device)
        hebbian = attn._compute_hebbian_update(
            attn.Wq(x).view(2, 16, config.n_head, config.head_dim),
            attn.Wv(x).view(2, 16, config.n_head, config.head_dim),
        )
        assert hebbian.shape == (config.n_head, config.head_dim, config.head_dim)

    def test_gradient_flow(self, default_config, device):
        """Gradients should flow through attention."""
        config = default_config
        attn = MultiScaleLinearAttention(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-6)


# ============================================================================
# FFN Tests
# ============================================================================


class TestReLULowRankFFN:
    def test_forward_shape(self, default_config, device):
        """FFN should preserve shape."""
        config = default_config
        ffn = ReLULowRankFFN(config).to(device)
        x = torch.randn(2, 32, config.n_embd, device=device)
        out, sparsity = ffn(x)
        assert out.shape == x.shape
        assert 0.0 <= sparsity <= 1.0

    def test_relu_sparsity(self, default_config, device):
        """ReLU should create sparsity (~50% for random normal input)."""
        config = default_config
        ffn = ReLULowRankFFN(config).to(device)
        x = torch.randn(4, 64, config.n_embd, device=device)
        _, sparsity = ffn(x)
        # With random normal input through linear + ReLU, expect ~50% sparsity
        assert sparsity > 0.3  # At least 30% sparse
        assert sparsity < 0.7  # At most 70% sparse

    def test_gradient_flow(self, default_config, device):
        """Gradients should flow through FFN."""
        config = default_config
        ffn = ReLULowRankFFN(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device, requires_grad=True)
        out, _ = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# BDH Layer Tests
# ============================================================================


class TestBDHv2Layer:
    def test_forward_shape(self, default_config, device):
        """Layer forward should preserve shape."""
        config = default_config
        layer = BDHv2Layer(config).to(device)
        x = torch.randn(2, 32, config.n_embd, device=device)
        out, states, sparsity = layer(x)
        assert out.shape == x.shape

    def test_hybrid_gating_shape(self, default_config, device):
        """Hybrid gating should preserve shape."""
        config = default_config
        config.use_hybrid_gating = True
        layer = BDHv2Layer(config).to(device)
        x = torch.randn(2, 32, config.n_embd, device=device)
        out, states, sparsity = layer(x)
        assert out.shape == x.shape

    def test_hybrid_gating_has_gradient(self, default_config, device):
        """Hybrid gating should have better gradient flow than multiplicative."""
        config = default_config
        config.use_hybrid_gating = True
        layer = BDHv2Layer(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device, requires_grad=True)
        out, _, _ = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Hybrid gating should maintain gradient magnitude
        grad_norm = x.grad.norm().item()
        assert grad_norm > 0.0

    def test_multiplicative_gating_gradient(self, default_config, device):
        """Multiplicative gating gradient magnitude (for comparison)."""
        config = default_config
        config.use_hybrid_gating = False
        layer = BDHv2Layer(config).to(device)
        x = torch.randn(2, 16, config.n_embd, device=device, requires_grad=True)
        out, _, _ = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Full Model Tests
# ============================================================================


class TestBDHv2:
    def test_forward_shape(self, small_model, device):
        """Full model forward should produce correct logits shape."""
        model = small_model.to(device)
        config = model.config
        x = torch.randint(0, config.vocab_size, (2, 64), device=device)
        logits, states = model(x)
        assert logits.shape == (2, 64, config.vocab_size)

    def test_return_states(self, small_model, device):
        """Model should return states when requested."""
        model = small_model.to(device)
        x = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
        logits, states = model(x, return_states=True)
        assert states is not None
        assert len(states) == model.config.n_layer

    def test_generation(self, small_model, device):
        """Autoregressive generation should work."""
        model = small_model.to(device)
        context = torch.randint(0, model.config.vocab_size, (1, 10), device=device)
        generated = model.generate(context, max_new_tokens=20, temperature=0.8)
        assert generated.shape == (1, 30)

    def test_parameter_count(self, small_model):
        """Parameter count should be reasonable."""
        total, trainable = count_parameters(small_model)
        assert total > 0
        assert trainable == total  # All params should be trainable

    def test_weight_tying(self, small_model):
        """Output projection should share weights with embedding."""
        model = small_model
        assert model.output_projection.weight is model.token_embedding.weight

    def test_cuda_forward(self, device):
        """Model should run on CUDA if available."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        config = BDHv2Config(
            vocab_size=1000, n_embd=128, n_layer=2, n_head=2, ffn_dim=256, dropout=0.0
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (1, 32), device=device)
        logits, _ = model(x)
        assert logits.device.type == "cuda"


# ============================================================================
# Vocab Projection Tests (Day 5)
# ============================================================================


class TestVocabProjection:
    def test_projection_shape(self):
        """Vocab projection should map teacher → student vocab."""
        proj = VocabProjection(teacher_vocab=151936, student_vocab=32000)
        teacher_logits = torch.randn(2, 32, 151936)
        student_logits = proj(teacher_logits)
        assert student_logits.shape == (2, 32, 32000)

    def test_projection_gradient(self):
        """Gradients should flow through vocab projection."""
        proj = VocabProjection(teacher_vocab=1000, student_vocab=100)
        teacher_logits = torch.randn(2, 16, 1000, requires_grad=True)
        student_logits = proj(teacher_logits)
        loss = student_logits.sum()
        loss.backward()
        assert teacher_logits.grad is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    def test_rope_enabled_forward(self, device):
        """Model with RoPE enabled should run forward pass."""
        config = BDHv2Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=2,
            n_head=2,
            ffn_dim=256,
            dropout=0.0,
            use_rope=True,
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_outer_product_forward(self, device):
        """Model with outer product Hebbian should run forward pass."""
        config = BDHv2Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=2,
            n_head=2,
            ffn_dim=256,
            dropout=0.0,
            use_outer_product=True,
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        logits, states = model(x, return_states=True)
        assert logits.shape == (2, 32, config.vocab_size)
        # State should be [H, D, D] per head
        assert states[0][0].shape == (config.n_head, config.head_dim, config.head_dim)

    def test_hybrid_gating_forward(self, device):
        """Model with hybrid gating should run forward pass."""
        config = BDHv2Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=2,
            n_head=2,
            ffn_dim=256,
            dropout=0.0,
            use_hybrid_gating=True,
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 32, config.vocab_size)

    def test_all_fixes_enabled(self, device):
        """Model with all fixes enabled should run forward pass."""
        config = BDHv2Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=2,
            n_head=2,
            ffn_dim=256,
            dropout=0.0,
            use_outer_product=True,
            use_rope=True,
            use_hybrid_gating=True,
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        logits, states = model(x, return_states=True)
        assert logits.shape == (2, 32, config.vocab_size)
        assert states is not None

    def test_sparsity_tracking(self, device):
        """Model should track FFN sparsity."""
        config = BDHv2Config(
            vocab_size=1000, n_embd=128, n_layer=2, n_head=2, ffn_dim=256, dropout=0.0
        )
        model = BDHv2(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        model(x)
        model(x)
        history = model.get_sparsity_history()
        assert len(history) == 2
        assert all(0.0 <= s <= 1.0 for s in history)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
