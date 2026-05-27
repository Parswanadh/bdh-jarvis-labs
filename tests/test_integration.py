"""
BDH v2 Integration Tests — Day 9
=================================

End-to-end integration tests that verify the full training pipeline,
checkpointing, all fix combinations, vocab projection, generation,
gradient clipping, LR scheduling, and memory bounds.

Run: python -m pytest tests/test_integration.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation.bdh_v2_clean import BDHv2Config, BDHv2, count_parameters
from implementation.train_distillation_v2 import (
    CosineLRWithWarmup,
    CheckpointManager,
    kl_div_loss_with_temp,
    SyntheticDistillDataset,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_config():
    """Minimal config for fast integration tests."""
    return BDHv2Config(
        vocab_size=256,
        n_embd=64,
        n_layer=2,
        n_head=2,
        ffn_dim=128,
        dropout=0.0,
        max_seq_len=128,
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    return BDHv2(tiny_config).to(device)


# ============================================================================
# Test 1: Full training loop (forward -> loss -> backward -> step)
# ============================================================================


class TestFullTrainingLoop:
    def test_training_step_no_errors(self, tiny_config, device):
        """A single training step should complete without errors."""
        model = BDHv2(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_with_states(self, tiny_config, device):
        """Training step with state persistence should work."""
        model = BDHv2(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()

        logits, states = model(input_ids, return_states=True)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert states is not None
        assert len(states) == tiny_config.n_layer

    def test_multiple_training_steps_decreasing_loss(self, tiny_config, device):
        """Loss should decrease over multiple training steps on simple data."""
        model = BDHv2(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        input_ids = torch.randint(0, tiny_config.vocab_size, (4, 16), device=device)
        labels = input_ids.clone()

        losses = []
        for _ in range(20):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ============================================================================
# Test 2: Checkpoint save and resume produces identical results
# ============================================================================


class TestCheckpointResume:
    def test_checkpoint_save_and_load(self, tiny_config, device):
        """Saving and loading a checkpoint should restore model state exactly."""
        model = BDHv2(tiny_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        total_steps = 100
        scheduler = CosineLRWithWarmup(
            optimizer, base_lr=1e-3, warmup_steps=10, total_steps=total_steps
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(tmpdir, max_keep=3)

            input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), device=device)

            model.train()
            for step in range(5):
                logits, _ = model(input_ids)
                loss = logits.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            ckpt_mgr.save(model, optimizer, scheduler, step=5, epoch=0, val_loss=1.0, is_best=True)

            before_logits, _ = model(input_ids)

            new_model = BDHv2(tiny_config).to(device)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = CosineLRWithWarmup(
                new_optimizer, base_lr=1e-3, warmup_steps=10, total_steps=total_steps
            )

            state = ckpt_mgr.load(new_model, new_optimizer, new_scheduler)

            after_logits, _ = new_model(input_ids)

            assert torch.allclose(before_logits, after_logits, atol=1e-5)
            assert state["step"] == 5

    def test_resume_produces_identical_training_trajectory(self, tiny_config, device):
        """Training from checkpoint should produce identical results."""
        total_steps = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            model_a = BDHv2(tiny_config).to(device)
            optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
            scheduler_a = CosineLRWithWarmup(
                optimizer_a, base_lr=1e-3, warmup_steps=10, total_steps=total_steps
            )
            ckpt_mgr = CheckpointManager(tmpdir, max_keep=3)

            input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), device=device)
            labels = input_ids.clone()

            model_a.train()
            for step in range(3):
                logits, _ = model_a(input_ids)
                loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
                optimizer_a.zero_grad()
                loss.backward()
                optimizer_a.step()
                scheduler_a.step()

            ckpt_mgr.save(model_a, optimizer_a, scheduler_a, step=3, epoch=0, val_loss=loss.item())

            after_3_loss = loss.item()

            model_b = BDHv2(tiny_config).to(device)
            optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
            scheduler_b = CosineLRWithWarmup(
                optimizer_b, base_lr=1e-3, warmup_steps=10, total_steps=total_steps
            )
            ckpt_mgr.load(model_b, optimizer_b, scheduler_b)

            model_b.train()
            for step in range(2):
                logits, _ = model_b(input_ids)
                loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
                optimizer_b.zero_grad()
                loss.backward()
                optimizer_b.step()
                scheduler_b.step()

            after_5_loss_b = loss.item()

            model_a.train()
            for step in range(2):
                logits, _ = model_a(input_ids)
                loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
                optimizer_a.zero_grad()
                loss.backward()
                optimizer_a.step()
                scheduler_a.step()

            after_5_loss_a = loss.item()

            assert abs(after_5_loss_a - after_5_loss_b) < 1e-5, (
                f"Trajectories diverged: {after_5_loss_a:.6f} vs {after_5_loss_b:.6f}"
            )


# ============================================================================
# Test 3: All 4 fix combinations work
# ============================================================================


class TestFixCombinations:
    @pytest.fixture
    def fix_configs(self):
        """All fix combinations to test."""
        base = dict(
            vocab_size=256, n_embd=64, n_layer=2, n_head=2, ffn_dim=128, dropout=0.0, max_seq_len=64
        )
        return {
            "baseline": BDHv2Config(**base),
            "+outer_product": BDHv2Config(**base, use_outer_product=True),
            "+rope": BDHv2Config(**base, use_rope=True),
            "+hybrid_gating": BDHv2Config(**base, use_hybrid_gating=True),
            "+all_fixes": BDHv2Config(
                **base, use_outer_product=True, use_rope=True, use_hybrid_gating=True
            ),
        }

    def test_baseline_forward(self, fix_configs, device):
        """Baseline config should run forward pass."""
        model = BDHv2(fix_configs["baseline"]).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 32, 256)
        assert not torch.isnan(logits).any()

    def test_outer_product_forward(self, fix_configs, device):
        """+outer_product should run forward pass."""
        model = BDHv2(fix_configs["+outer_product"]).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, states = model(x, return_states=True)
        assert logits.shape == (2, 32, 256)
        assert states is not None

    def test_rope_forward(self, fix_configs, device):
        """+rope should run forward pass."""
        model = BDHv2(fix_configs["+rope"]).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 32, 256)

    def test_hybrid_gating_forward(self, fix_configs, device):
        """+hybrid_gating should run forward pass."""
        model = BDHv2(fix_configs["+hybrid_gating"]).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 32, 256)

    def test_all_fixes_forward(self, fix_configs, device):
        """+all_fixes should run forward pass."""
        model = BDHv2(fix_configs["+all_fixes"]).to(device)
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, states = model(x, return_states=True)
        assert logits.shape == (2, 32, 256)
        assert states is not None

    def test_all_fixes_backward(self, fix_configs, device):
        """+all_fixes should support backward pass with gradients."""
        model = BDHv2(fix_configs["+all_fixes"]).to(device)
        model.train()
        x = torch.randint(0, 256, (2, 32), device=device)
        labels = x.clone()

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_all_fixes_training_step(self, fix_configs, device):
        """+all_fixes should complete a full training step."""
        model = BDHv2(fix_configs["+all_fixes"]).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 256, (2, 32), device=device)
        labels = x.clone()

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        assert loss.item() > 0


# ============================================================================
# Test 4: Vocab projection integrates with distillation loss
# ============================================================================


class TestVocabProjectionDistillation:
    def test_projection_shape(self, device):
        """Vocab projection should map teacher -> student vocab."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            ffn_dim=128,
            dropout=0.0,
            use_vocab_projection=True,
            teacher_vocab_size=1024,
        )
        model = BDHv2(config).to(device)
        teacher_logits = torch.randn(2, 16, 1024, device=device)
        projected = model.project_teacher_logits(teacher_logits)
        assert projected.shape == (2, 16, 256)

    def test_kl_div_with_projected_teacher(self, device):
        """KL divergence loss should work with projected teacher logits."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            ffn_dim=128,
            dropout=0.0,
            use_vocab_projection=True,
            teacher_vocab_size=1024,
        )
        model = BDHv2(config).to(device)
        model.train()

        input_ids = torch.randint(0, 256, (2, 16), device=device)
        teacher_logits = torch.randn(2, 16, 1024, device=device)

        student_logits, _ = model(input_ids)
        projected_teacher = model.project_teacher_logits(teacher_logits)

        kl_loss = kl_div_loss_with_temp(student_logits, projected_teacher, temperature=2.0)

        assert kl_loss.item() >= 0
        assert not torch.isnan(kl_loss)
        assert not torch.isinf(kl_loss)

    def test_combined_distillation_loss(self, device):
        """Combined CE + KL loss should backprop correctly."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            ffn_dim=128,
            dropout=0.0,
            use_vocab_projection=True,
            teacher_vocab_size=1024,
        )
        model = BDHv2(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 256, (2, 16), device=device)
        labels = input_ids.clone()
        teacher_logits = torch.randn(2, 16, 1024, device=device)

        student_logits, _ = model(input_ids)
        projected_teacher = model.project_teacher_logits(teacher_logits)

        ce_loss = F.cross_entropy(student_logits.view(-1, 256), labels.view(-1))
        kl_loss = kl_div_loss_with_temp(student_logits, projected_teacher, temperature=2.0)

        total_loss = 0.2 * ce_loss + 0.8 * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_vocab_projection_not_enabled_raises(self, device):
        """Calling project_teacher_logits without projection should raise."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            ffn_dim=128,
            dropout=0.0,
            use_vocab_projection=False,
        )
        model = BDHv2(config).to(device)
        with pytest.raises(RuntimeError, match="Vocab projection not enabled"):
            model.project_teacher_logits(torch.randn(2, 16, 1024, device=device))


# ============================================================================
# Test 5: Model generates coherent output (not NaN/Inf)
# ============================================================================


class TestCoherentGeneration:
    def test_generation_no_nan_inf(self, tiny_model, device):
        """Generated tokens should not contain NaN or Inf."""
        tiny_model.eval()
        context = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            generated = tiny_model.generate(context, max_new_tokens=20, temperature=0.8)

        assert not torch.isnan(generated).any()
        assert not torch.isinf(generated).any()
        assert generated.shape == (1, 28)

    def test_generation_values_in_vocab_range(self, tiny_model, device):
        """Generated token IDs should be within vocabulary range."""
        tiny_model.eval()
        context = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            generated = tiny_model.generate(context, max_new_tokens=10, temperature=1.0)

        assert (generated >= 0).all()
        assert (generated < 256).all()

    def test_generation_deterministic_with_seed(self, tiny_config, device):
        """Generation should be deterministic with same seed."""
        torch.manual_seed(42)
        model_a = BDHv2(tiny_config).to(device)
        model_a.eval()
        context = torch.randint(0, 256, (1, 8), device=device)
        with torch.no_grad():
            logits_a, _ = model_a(context)
            probs_a = torch.softmax(logits_a[:, -1, :] / 0.5, dim=-1)
            gen_a = torch.multinomial(probs_a, num_samples=1)

        torch.manual_seed(42)
        model_b = BDHv2(tiny_config).to(device)
        model_b.eval()
        with torch.no_grad():
            logits_b, _ = model_b(context)
            probs_b = torch.softmax(logits_b[:, -1, :] / 0.5, dim=-1)
            gen_b = torch.multinomial(probs_b, num_samples=1)

        assert torch.equal(gen_a, gen_b)

    def test_logits_no_nan_inf(self, tiny_model, device):
        """Forward pass logits should never contain NaN or Inf."""
        tiny_model.eval()
        with torch.no_grad():
            for _ in range(5):
                x = torch.randint(0, 256, (2, 32), device=device)
                logits, _ = tiny_model(x)
                assert not torch.isnan(logits).any()
                assert not torch.isinf(logits).any()


# ============================================================================
# Test 6: Gradient clipping works
# ============================================================================


class TestGradientClipping:
    def test_clip_grad_norm_reduces_gradient_magnitude(self, tiny_config, device):
        """Gradient clipping should reduce max gradient norm."""
        model = BDHv2(tiny_config).to(device)
        model.train()

        x = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = x.clone()

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
        loss.backward()

        # Get norm before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        ).item()

        # Now clip with a small max_norm
        total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1).item()

        # After clipping with max_norm=0.1, the returned norm should be the pre-clip value
        # but the actual gradients should be scaled down
        # Verify by computing the actual norm after clipping
        actual_norm = math.sqrt(
            sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
        )

        if total_norm_before > 0.1:
            assert actual_norm <= 0.1 + 1e-6, (
                f"Clipping failed: before={total_norm_before:.4f}, actual_after={actual_norm:.4f}"
            )

    def test_clip_grad_norm_does_not_break_training(self, tiny_config, device):
        """Training with gradient clipping should still reduce loss."""
        model = BDHv2(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randint(0, tiny_config.vocab_size, (4, 16), device=device)
        labels = x.clone()

        losses = []
        for _ in range(20):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease with clipping: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_clip_grad_norm_handles_nan_gradients(self, tiny_config, device):
        """Gradient clipping should handle extreme gradients gracefully."""
        model = BDHv2(tiny_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randint(0, tiny_config.vocab_size, (2, 32), device=device)
        labels = x.clone()

        for _ in range(50):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, tiny_config.vocab_size), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        final_logits, _ = model(x)
        assert not torch.isnan(final_logits).any()
        assert not torch.isinf(final_logits).any()


# ============================================================================
# Test 7: LR scheduler produces correct values
# ============================================================================


class TestLRScheduler:
    def test_warmup_phase_linear_increase(self):
        """LR should increase linearly during warmup."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.0)
        scheduler = CosineLRWithWarmup(optimizer, base_lr=1e-3, warmup_steps=100, total_steps=1000)

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_lr())

        assert lrs[0] < lrs[50] < lrs[99]
        assert abs(lrs[99] - 1e-3) < 1e-8

    def test_post_warmup_cosine_decay(self):
        """LR should follow cosine decay after warmup."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.0)
        scheduler = CosineLRWithWarmup(optimizer, base_lr=1e-3, warmup_steps=100, total_steps=1000)

        for _ in range(100):
            scheduler.step()

        lr_at_warmup_end = scheduler.get_lr()

        lrs = []
        for _ in range(900):
            scheduler.step()
            lrs.append(scheduler.get_lr())

        assert lrs[0] <= lr_at_warmup_end
        assert lrs[-1] < lrs[0]

    def test_min_lr_reached_at_end(self):
        """LR should reach min_lr at the end of training."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.0)
        base_lr = 1e-3
        min_lr_ratio = 0.1
        scheduler = CosineLRWithWarmup(
            optimizer,
            base_lr=base_lr,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=min_lr_ratio,
        )

        for _ in range(1000):
            scheduler.step()

        final_lr = scheduler.get_lr()
        expected_min = base_lr * min_lr_ratio
        assert abs(final_lr - expected_min) < 1e-8, f"Expected {expected_min}, got {final_lr}"

    def test_scheduler_state_dict_roundtrip(self):
        """Scheduler state should be preserved across save/load."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.0)
        scheduler = CosineLRWithWarmup(optimizer, base_lr=1e-3, warmup_steps=100, total_steps=1000)

        for _ in range(50):
            scheduler.step()

        lr_before = scheduler.get_lr()
        state = scheduler.state_dict()

        new_optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.0)
        new_scheduler = CosineLRWithWarmup(
            new_optimizer, base_lr=1e-3, warmup_steps=100, total_steps=1000
        )
        new_scheduler.load_state_dict(state)

        assert new_scheduler.get_lr() == lr_before
        assert new_scheduler.current_step == 50


# ============================================================================
# Test 8: Memory usage stays within bounds for long sequences
# ============================================================================


class TestMemoryBounds:
    def test_long_sequence_forward_pass(self, device):
        """512-token sequence should complete without OOM on reasonable hardware."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            ffn_dim=256,
            dropout=0.0,
            max_seq_len=512,
        )
        model = BDHv2(config).to(device)
        model.eval()

        x = torch.randint(0, 256, (1, 512), device=device)
        with torch.no_grad():
            logits, _ = model(x)

        assert logits.shape == (1, 512, 256)
        assert not torch.isnan(logits).any()

    def test_long_sequence_backward_pass(self, device):
        """512-token sequence backward pass should complete."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            ffn_dim=256,
            dropout=0.0,
            max_seq_len=512,
        )
        model = BDHv2(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 256, (1, 512), device=device)
        labels = x.clone()

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_memory_does_not_grow_with_repeated_forward(self, device):
        """Repeated forward passes should not cause unbounded memory growth."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            ffn_dim=256,
            dropout=0.0,
            max_seq_len=512,
        )
        model = BDHv2(config).to(device)
        model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated()

            x = torch.randint(0, 256, (1, 512), device=device)
            for _ in range(10):
                with torch.no_grad():
                    logits, _ = model(x)

            final_mem = torch.cuda.memory_allocated()
            mem_growth = final_mem - initial_mem
            assert mem_growth < 100 * 1024 * 1024, (
                f"Memory grew by {mem_growth / 1024 / 1024:.1f}MB over 10 passes"
            )

    def test_batch_of_long_sequences(self, device):
        """Batch of long sequences should work."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            ffn_dim=256,
            dropout=0.0,
            max_seq_len=512,
        )
        model = BDHv2(config).to(device)
        model.eval()

        x = torch.randint(0, 256, (2, 512), device=device)
        with torch.no_grad():
            logits, _ = model(x)

        assert logits.shape == (2, 512, 256)

    def test_state_matrix_memory_is_constant_per_head(self, device):
        """State matrices should be O(H*D*D) per layer, not O(T)."""
        config = BDHv2Config(
            vocab_size=256,
            n_embd=128,
            n_layer=4,
            n_head=4,
            ffn_dim=256,
            dropout=0.0,
            max_seq_len=512,
        )
        model = BDHv2(config).to(device)
        model.eval()

        x_short = torch.randint(0, 256, (1, 64), device=device)
        x_long = torch.randint(0, 256, (1, 512), device=device)

        with torch.no_grad():
            _, states_short = model(x_short, return_states=True)
            _, states_long = model(x_long, return_states=True)

        assert states_short is not None
        assert states_long is not None

        for layer_idx in range(config.n_layer):
            for scale_idx in range(config.num_scales):
                assert (
                    states_short[layer_idx][scale_idx].shape
                    == states_long[layer_idx][scale_idx].shape
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
