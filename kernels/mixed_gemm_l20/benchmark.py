"""
Benchmark: Fused Mixed-Precision GEMM (INT4 + INT8) vs Baseline on H20

Usage:
    cd kernels/mixed_gemm_l20
    pip install -e .
    python benchmark.py
"""

import torch
import time
import mixed_gemm_l20


def pack_int4(tensor_int8):
    """Pack INT8 tensor (values in [-8, 7]) into INT4 packed format.
    Two int4 values packed into one int8: low_nibble | (high_nibble << 4)
    Input shape: (rows, cols) with cols even
    Output shape: (rows, cols // 2) as int8
    """
    assert tensor_int8.shape[-1] % 2 == 0
    # Clamp to INT4 range
    t = tensor_int8.clamp(-8, 7).to(torch.int8)
    # Split into pairs
    low = t[..., 0::2] & 0x0F   # low nibble
    high = t[..., 1::2] & 0x0F  # high nibble
    packed = low | (high << 4)
    return packed.to(torch.int8)


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

    return (end - start) / iters * 1e6


def main():
    # Dimensions (Llama-3.2-1B q_proj)
    M = 128
    N = 2048
    K_high = 256   # INT8 channels
    K_low = 1792   # INT4 channels

    device = 'cuda'

    print(f"Mixed-Precision Fused GEMM Benchmark (INT4 + INT8)")
    print(f"  M={M}, N={N}, K_high={K_high} (INT8), K_low={K_low} (INT4)")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print()

    # Generate INT8 inputs for high precision path
    A_high = torch.randint(-4, 4, (M, K_high), dtype=torch.int8, device=device)
    B_high = torch.randint(-4, 4, (N, K_high), dtype=torch.int8, device=device)

    # Generate INT4 inputs for low precision path (stored as packed INT8)
    A_low_unpacked = torch.randint(-4, 4, (M, K_low), dtype=torch.int8, device=device)
    B_low_unpacked = torch.randint(-4, 4, (N, K_low), dtype=torch.int8, device=device)
    A_low_packed = pack_int4(A_low_unpacked)  # (M, K_low // 2) as int8
    B_low_packed = pack_int4(B_low_unpacked)  # (N, K_low // 2) as int8

    print(f"  A_high: {A_high.shape} INT8")
    print(f"  B_high: {B_high.shape} INT8")
    print(f"  A_low_packed: {A_low_packed.shape} (packed INT4)")
    print(f"  B_low_packed: {B_low_packed.shape} (packed INT4)")
    print()

    # === Correctness check ===
    print("Correctness check...")
    out_fused = mixed_gemm_l20.fused_mixed_gemm(A_low_packed, B_low_packed, A_high, B_high)

    # Reference: compute in FP32 using unpacked values
    # A_low (M, K_low) @ B_low^T (K_low, N) + A_high (M, K_high) @ B_high^T (K_high, N)
    ref = (A_low_unpacked.float() @ B_low_unpacked.float().t() +
           A_high.float() @ B_high.float().t()).half()

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.flatten().float(), ref.flatten().float(), dim=0).item()
    max_diff = (out_fused.float() - ref.float()).abs().max().item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.2f}")
    print()

    # === Performance benchmark ===
    print("Performance benchmark (100 iterations)...")

    # Fused kernel
    lat_fused = benchmark_fn(mixed_gemm_l20.fused_mixed_gemm,
                             A_low_packed, B_low_packed, A_high, B_high)

    # Baseline 1: CUTLASS SM80 tensor core (same instructions, 2 launch + add)
    lat_cutlass_baseline = benchmark_fn(mixed_gemm_l20.baseline_cutlass_mixed_gemm,
                                         A_low_packed, B_low_packed, A_high, B_high)

    # Baseline 2: cuBLAS FP16 (2x matmul + add) — may use WGMMA on H20
    A_low_f = A_low_unpacked.half()
    B_low_f = B_low_unpacked.half()
    A_high_f = A_high.half()
    B_high_f = B_high.half()

    def baseline_cublas():
        return torch.matmul(A_low_f, B_low_f.t()) + torch.matmul(A_high_f, B_high_f.t())

    lat_cublas = benchmark_fn(baseline_cublas)

    # TOPS calculation
    total_ops = 2 * M * N * (K_low + K_high)
    tops_fused = total_ops / (lat_fused * 1e-6) / 1e12
    tops_cutlass = total_ops / (lat_cutlass_baseline * 1e-6) / 1e12
    tops_cublas = total_ops / (lat_cublas * 1e-6) / 1e12

    print(f"  Fused kernel (INT4+INT8):          {lat_fused:.1f} μs  ({tops_fused:.2f} TOPS)")
    print(f"  CUTLASS baseline (2x launch+add):  {lat_cutlass_baseline:.1f} μs  ({tops_cutlass:.2f} TOPS)")
    print(f"  cuBLAS FP16 (2x matmul+add):       {lat_cublas:.1f} μs  ({tops_cublas:.2f} TOPS)")
    print(f"  Speedup vs CUTLASS baseline:       {lat_cutlass_baseline / lat_fused:.2f}x")
    print(f"  Speedup vs cuBLAS FP16:            {lat_cublas / lat_fused:.2f}x")


if __name__ == '__main__':
    main()
