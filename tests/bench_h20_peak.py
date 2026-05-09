"""
H20 GPU Peak Performance Microbenchmark
========================================
Measure actual INT8 Tensor Core TOPS and HBM bandwidth on our H20.

1. INT8 peak TOPS: large GEMM (10240 x 10240 x 10240) using CUTLASS S8S8
2. HBM bandwidth: large copy kernel

GPU cache is flushed between iterations by writing to a large buffer.
"""

import torch
import time
import mixed_gemm


def flush_gpu_cache(flush_buf: torch.Tensor):
    """Flush L2 cache by writing to a large buffer (> L2 size)."""
    # H20 has ~50MB L2 cache; writing 64MB should flush it
    flush_buf.fill_(1)
    torch.cuda.synchronize()


def benchmark_int8_peak():
    """Measure peak INT8 Tensor Core TOPS with large GEMM."""
    M = N = K = 10240

    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    flush_buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")  # 64MB

    # Warmup
    for _ in range(5):
        mixed_gemm.gemm_s8s8(A, B)
    torch.cuda.synchronize()

    # Benchmark with cache flush
    iters = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # First: without cache flush (best case, data in L2)
    start.record()
    for _ in range(iters):
        mixed_gemm.gemm_s8s8(A, B)
    end.record()
    torch.cuda.synchronize()
    ms_hot = start.elapsed_time(end) / iters

    # Second: with cache flush (cold cache, realistic)
    times_cold = []
    for _ in range(iters):
        flush_gpu_cache(flush_buf)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        mixed_gemm.gemm_s8s8(A, B)
        e.record()
        torch.cuda.synchronize()
        times_cold.append(s.elapsed_time(e))
    ms_cold = sum(times_cold) / len(times_cold)

    flops = 2 * M * N * K
    tops_hot = flops / ms_hot / 1e9
    tops_cold = flops / ms_cold / 1e9

    print(f"=== INT8 Tensor Core Peak (GEMM {M}x{N}x{K}) ===")
    print(f"  Hot cache:  {ms_hot:.3f} ms → {tops_hot:.1f} TOPS")
    print(f"  Cold cache: {ms_cold:.3f} ms → {tops_cold:.1f} TOPS")
    print(f"  Use cold-cache value as realistic peak: {tops_cold:.1f} TOPS")
    return tops_cold


def benchmark_hbm_bandwidth():
    """Measure peak HBM bandwidth with copy kernel."""
    # Use 256MB buffers (much larger than L2)
    nbytes = 256 * 1024 * 1024
    src = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda")
    dst = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda")
    flush_buf = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    # Warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Without cache flush (best case)
    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    ms_hot = start.elapsed_time(end) / iters

    # With cache flush
    times_cold = []
    for _ in range(iters):
        flush_gpu_cache(flush_buf)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dst.copy_(src)
        e.record()
        torch.cuda.synchronize()
        times_cold.append(s.elapsed_time(e))
    ms_cold = sum(times_cold) / len(times_cold)

    # Bandwidth = bytes_read + bytes_written
    total_bytes = nbytes * 2  # read src + write dst
    bw_hot = total_bytes / ms_hot / 1e6  # GB/s
    bw_cold = total_bytes / ms_cold / 1e6

    print(f"\n=== HBM Bandwidth (copy {nbytes//1024//1024}MB) ===")
    print(f"  Hot cache:  {ms_hot:.3f} ms → {bw_hot:.1f} GB/s")
    print(f"  Cold cache: {ms_cold:.3f} ms → {bw_cold:.1f} GB/s")
    print(f"  Use cold-cache value as realistic peak: {bw_cold:.1f} GB/s")
    return bw_cold


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    peak_tops = benchmark_int8_peak()
    peak_bw = benchmark_hbm_bandwidth()

    print(f"\n{'='*50}")
    print(f"  H20 Measured Peak:")
    print(f"    INT8 TOPS:     {peak_tops:.1f} TOPS")
    print(f"    HBM Bandwidth: {peak_bw:.1f} GB/s")
    print(f"{'='*50}")
