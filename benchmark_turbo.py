#!/usr/bin/env python3
"""
Benchmark turbo logits generation to find actual bottlenecks.
Test different batch sizes, top-k values, chunk sizes to find peak throughput.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

def benchmark_batch_sizes():
    """Test batch size impact on throughput."""
    print("\n=== BATCH SIZE BENCHMARK ===")
    
    # Fake Qwen load (just test forward pass timing, not actual model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create fake model output (same shape as Qwen logits)
    vocab_size = 32768
    seq_len = 128
    
    results = {}
    
    for batch_size in [4, 8, 12, 16, 20, 24]:
        try:
            # Simulate forward pass
            dummy_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float16)
            
            # Measure topk time
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(10):
                topk_vals, topk_inds = torch.topk(dummy_logits, k=32, dim=-1)
                topk_vals = topk_vals.cpu()
                topk_inds = topk_inds.cpu()
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = (batch_size * 10) / elapsed
            results[batch_size] = {
                "time_sec": elapsed,
                "throughput_batches_per_sec": throughput,
                "throughput_tokens_per_sec": throughput * seq_len,
            }
            
            print(f"  BS={batch_size:2d}: {throughput:.2f} batches/s, {throughput*seq_len:.1f} tokens/s")
        
        except RuntimeError as e:
            print(f"  BS={batch_size:2d}: OOM - {e}")
            break
    
    return results

def benchmark_topk_values():
    """Test top-k impact on throughput."""
    print("\n=== TOP-K BENCHMARK ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    seq_len = 128
    vocab_size = 32768
    
    dummy_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float16)
    
    results = {}
    
    for top_k in [16, 32, 64, 128]:
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            topk_vals, topk_inds = torch.topk(dummy_logits, k=top_k, dim=-1)
            topk_vals = topk_vals.cpu()
            topk_inds = topk_inds.cpu()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch_size * 10) / elapsed
        results[top_k] = {
            "time_sec": elapsed,
            "throughput_batches_per_sec": throughput,
        }
        
        print(f"  K={top_k:3d}: {throughput:.2f} batches/s")
    
    return results

def check_gpu_utilization():
    """Check actual GPU memory and compute utilization."""
    print("\n=== GPU STATUS ===")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memory: {total_mem:.1f}GB total, {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Compute capability
        props = torch.cuda.get_device_properties(0)
        print(f"Compute Capability: {props.major}.{props.minor}")
        
        # Check for flash attention
        try:
            from flash_attn import flash_attn_func
            print("Flash Attention: ✅ INSTALLED")
        except ImportError:
            print("Flash Attention: ❌ NOT INSTALLED (install: pip install flash-attn)")
    else:
        print("CUDA not available - CPU only")

if __name__ == "__main__":
    check_gpu_utilization()
    batch_results = benchmark_batch_sizes()
    topk_results = benchmark_topk_values()
    
    # Recommendations
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    
    if batch_results:
        best_bs = max(batch_results.items(), key=lambda x: x[1]["throughput_batches_per_sec"])[0]
        print(f"Optimal batch size: {best_bs}")
    
    if topk_results:
        best_k = min(topk_results.items(), key=lambda x: x[1]["time_sec"])[0]
        print(f"Recommended top-k: {best_k} (fastest)")
    
    print("\nKey wins to deploy:")
    print("  1. Install flash-linear-attention for Qwen (3-5x speedup)")
    print("  2. Increase batch size to max non-OOM value")
    print("  3. Reduce top-k from 128 → 32 (memory + speed)")
    print("  4. Use torch.compile on teacher forward pass")
    print("  5. Async I/O writes (don't block on disk)")
