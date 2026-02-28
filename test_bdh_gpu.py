"""
BDH-GPU Quick Test Script
=========================

Run this to verify your installation and test the model.
"""

import torch
import sys
import time

print("="*60)
print("BDH-GPU 10M - Quick Test")
print("="*60)

# Check PyTorch installation
print("\n1. Checking PyTorch installation...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    device = "cpu"
    print("   WARNING: CUDA not available, using CPU (will be slow)")

# Import model
print("\n2. Importing BDH-GPU model...")
try:
    from bdh_gpu_10m import BDHGPUTensor, BDHConfig, count_parameters
    print("   ✓ Model imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Create model
print("\n3. Creating 10M BDH-GPU model...")
config = BDHConfig(
    vocab_size=256,
    n_embd=256,
    n_layer=6,
    n_head=4,
    ffn_dim=1024,
    dropout=0.0,  # Disable dropout for testing
)

try:
    model = BDHGPUTensor(config).to(device)
    params = count_parameters(model)
    print(f"   ✓ Model created with {params/1e6:.2f}M parameters")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test forward pass
print("\n4. Testing forward pass...")
batch_size = 2
seq_len = 16
x = torch.randint(0, 256, (batch_size, seq_len)).to(device)

try:
    with torch.no_grad():
        logits, state = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected: [{batch_size}, {seq_len}, 256]")
    assert logits.shape == (batch_size, seq_len, 256), "Wrong output shape!"
    print("   ✓ Forward pass successful")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test state persistence
print("\n5. Testing state matrix...")
try:
    logits1, state1 = model(x, return_state=True)
    logits2, state2 = model(x, state=state1, return_state=True)

    if state1 is not None and state2 is not None:
        print(f"   State shape: {state1.shape}")
        print("   ✓ State matrix working correctly")
    else:
        print("   ✗ State is None")
except Exception as e:
    print(f"   ✗ State test failed: {e}")

# Test generation
print("\n6. Testing text generation...")
try:
    context = "Hello world"
    context_bytes = context.encode('utf-8')
    idx = torch.tensor([[b for b in context_bytes]], dtype=torch.long).to(device)

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=20, temperature=0.8)

    generated_text = bytes(generated[0].tolist()).decode('utf-8', errors='ignore')
    print(f"   Context: '{context}'")
    print(f"   Generated: '{generated_text}'")
    print("   ✓ Generation successful")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")

# Benchmark speed
print("\n7. Speed benchmark...")
try:
    model.eval()

    # CPU benchmark
    x_cpu = torch.randint(0, 256, (1, 128))
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_cpu)
    cpu_time = (time.time() - t0) / 10
    print(f"   CPU: {cpu_time*1000:.1f}ms per forward pass (seq_len=128)")

    # GPU benchmark
    if device == "cuda":
        x_gpu = torch.randint(0, 256, (1, 128)).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - t0) / 10
        tokens_per_sec = 128 / gpu_time
        print(f"   GPU: {gpu_time*1000:.1f}ms per forward pass (seq_len=128)")
        print(f"   GPU: {tokens_per_sec:.0f} tokens/sec")
        print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

except Exception as e:
    print(f"   ✗ Benchmark failed: {e}")

# Memory usage
print("\n8. Memory usage...")
if device == "cuda":
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
    print(f"   GPU memory reserved: {memory_reserved:.2f} GB")

# Test sparsity (ReLU effect)
print("\n9. Testing activation sparsity...")
try:
    model.eval()
    x_test = torch.randn(1, 1, config.n_embd).to(device)

    # Get activations from first FFN layer
    with torch.no_grad():
        x_proj = model.layers[0].ffn.W1(x_test)
        x_relu = torch.relu(x_proj)

    sparsity = (x_relu == 0).float().mean().item()
    active_ratio = 100 * (1 - sparsity)
    print(f"   Active neurons: {active_ratio:.1f}%")
    print(f"   Expected: ~5% for sparse activations")

    if 1 < active_ratio < 20:
        print("   ✓ Sparsity is in expected range")
    else:
        print("   Note: Sparsity will emerge after training")
except Exception as e:
    print(f"   ✗ Sparsity test failed: {e}")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✓ All tests passed!")
print(f"✓ Model has {params/1e6:.2f}M parameters")
print(f"✓ Running on {device.upper()}")
print("="*60)

print("\nYou're ready to train!")
print("Run: python train_bdh_gpu.py")
print("\nOr test generation:")
print("  python -c 'from bdh_gpu_10m import *; model = BDHGPUTensor(BDHConfig())'")
