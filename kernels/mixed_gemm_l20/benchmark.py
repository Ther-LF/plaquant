"""
Benchmark: Fused Mixed-Precision GEMM vs Baseline on L20

Usage:
    cd kernels/mixed_gemm_l20
    pip install -e .
    python benchmark.py
"""

import torch
import time
import mixed_gemm_l20

def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function, return average latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1e6  # microseconds


def main():
    # Dimensions (Llama-3.2-1B q_proj)
    M = 128
    N = 2048
    K_high = 256
    K_low = 1792

    device = 'cuda'

    print(f"Mixed-Precision Fused GEMM Benchmark")
    print(f"  M={M}, N={N}, K_high={K_high}, K_low={K_low}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print()

    # Generate random INT8 inputs
    A_low = torch.randint(-128, 127, (M, K_low), dtype=torch.int8, device=device)
    B_low = torch.randint(-128, 127, (K_low, N), dtype=torch.int8, device=device)
    A_high = torch.randint(-128, 127, (M, K_high), dtype=torch.int8, device=device)
    B_high = torch.randint(-128, 127, (K_high, N), dtype=torch.int8, device=device)

    # === Correctness check ===
    print("Correctness check...")
    out_fused = mixed_gemm_l20.fused_mixed_gemm(A_low, B_low, A_high, B_high)

    # Reference: compute in FP32
    ref = (A_low.float() @ B_low.float() + A_high.float() @ B_high.float()).half()

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.flatten().float(), ref.flatten().float(), dim=0).item()
    max_diff = (out_fused.float() - ref.float()).abs().max().item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.2f}")
    print()

    # === Performance benchmark ===
    print("Performance benchmark (100 iterations)...")

    # Fused kernel
    lat_fused = benchmark_fn(mixed_gemm_l20.fused_mixed_gemm, A_low, B_low, A_high, B_high)

    # Baseline: 2x matmul + add (PyTorch/cuBLAS)
    A_low_f = A_low.half()
    B_low_f = B_low.half()
    A_high_f = A_high.half()
    B_high_f = B_high.half()

    def baseline():
        return torch.matmul(A_low_f, B_low_f) + torch.matmul(A_high_f, B_high_f)

    lat_baseline = benchmark_fn(baseline)

    # TOPS calculation
    total_ops = 2 * M * N * (K_low + K_high)  # MAC = 2 ops
    tops_fused = total_ops / (lat_fused * 1e-6) / 1e12
    tops_baseline = total_ops / (lat_baseline * 1e-6) / 1e12

    print(f"  Fused kernel:   {lat_fused:.1f} μs  ({tops_fused:.2f} TOPS)")
    print(f"  Baseline (2x matmul + add): {lat_baseline:.1f} μs  ({tops_baseline:.2f} TOPS)")
    print(f"  Speedup: {lat_baseline / lat_fused:.2f}x")


if __name__ == '__main__':
    main()
