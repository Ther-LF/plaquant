"""
Benchmark: Fused Mixed-Precision GEMM (INT4 + INT8) — Multi-config Multi-size

Usage:
    cd kernels/mixed_gemm_l20
    pip install -e .
    python benchmark.py
"""

import torch
import time
import mixed_gemm_l20


def pack_int4(tensor_int8):
    """Pack INT8 tensor (values in [-8, 7]) into INT4 packed format."""
    assert tensor_int8.shape[-1] % 2 == 0
    t = tensor_int8.clamp(-8, 7).to(torch.int8)
    low = t[..., 0::2] & 0x0F
    high = t[..., 1::2] & 0x0F
    packed = low | (high << 4)
    return packed.to(torch.int8)


def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function using CUDA events, return average latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds
    return start_event.elapsed_time(end_event) / iters * 1000  # convert to microseconds


def run_single_config(M, N, K_high, K_low, device='cuda'):
    """Run benchmark for a single problem size.

    Three things are compared, all computing logically equivalent (M, N) outputs
    with K_total = K_high + K_low contraction:
      - Fused INT4+INT8 (our kernel)
      - 2-launch INT (CUTLASS INT4 + INT8 baseline, also SM80 mma.sync)
      - FP16 cuBLAS at full K_total (the real-world replacement target)
    """
    # Generate inputs
    A_high = torch.randint(-4, 4, (M, K_high), dtype=torch.int8, device=device)
    B_high = torch.randint(-4, 4, (N, K_high), dtype=torch.int8, device=device)
    A_low_unpacked = torch.randint(-4, 4, (M, K_low), dtype=torch.int8, device=device)
    B_low_unpacked = torch.randint(-4, 4, (N, K_low), dtype=torch.int8, device=device)
    A_low_packed = pack_int4(A_low_unpacked)
    B_low_packed = pack_int4(B_low_unpacked)

    # FP16 baseline operands at the full equivalent shape
    K_total = K_high + K_low
    A_fp16 = torch.randn(M, K_total, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(N, K_total, dtype=torch.float16, device=device)

    # Correctness check (vs INT reference)
    out_fused = mixed_gemm_l20.fused_mixed_gemm(A_low_packed, B_low_packed, A_high, B_high)
    ref = (A_low_unpacked.float() @ B_low_unpacked.float().t() +
           A_high.float() @ B_high.float().t()).half()

    diff = (out_fused.float() - ref.float())
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.flatten().float(), ref.flatten().float(), dim=0).item()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    mape = (diff.abs() / (ref.float().abs() + 1e-8)).mean().item() * 100

    # Performance
    lat_fused = benchmark_fn(mixed_gemm_l20.fused_mixed_gemm,
                             A_low_packed, B_low_packed, A_high, B_high)
    lat_baseline = benchmark_fn(mixed_gemm_l20.baseline_cutlass_mixed_gemm,
                                A_low_packed, B_low_packed, A_high, B_high)
    lat_fp16 = benchmark_fn(lambda a, b: torch.matmul(a, b.t()), A_fp16, B_fp16)

    total_ops = 2 * M * N * K_total
    tops_fused = total_ops / (lat_fused * 1e-6) / 1e12
    tops_baseline = total_ops / (lat_baseline * 1e-6) / 1e12
    tops_fp16 = total_ops / (lat_fp16 * 1e-6) / 1e12

    return {
        'M': M, 'N': N, 'K_high': K_high, 'K_low': K_low,
        'cos_sim': cos_sim,
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'mape': mape,
        'lat_fused': lat_fused,
        'lat_baseline': lat_baseline,
        'lat_fp16': lat_fp16,
        'tops_fused': tops_fused,
        'tops_baseline': tops_baseline,
        'tops_fp16': tops_fp16,
        'speedup_vs_baseline': lat_baseline / lat_fused,
        'speedup_vs_fp16': lat_fp16 / lat_fused,
    }


def main():
    device = 'cuda'
    print(f"Mixed-Precision Fused GEMM Benchmark (INT4 + INT8)")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print()

    # Test matrix: multiple problem sizes
    # Varying M, N, K_high, K_low
    test_cases = [
        # (M, N, K_high, K_low) — description
        # === Vary M (batch size) with q_proj dims ===
        (1,    2048, 256, 1792),    # bs=1
        (16,   2048, 256, 1792),    # bs=16
        (64,   2048, 256, 1792),    # bs=64
        (128,  2048, 256, 1792),    # bs=128
        (256,  2048, 256, 1792),    # bs=256
        (512,  2048, 256, 1792),    # bs=512
        (1024, 2048, 256, 1792),    # bs=1024
        (2048, 2048, 256, 1792),    # bs=2048
        (4096, 2048, 256, 1792),    # bs=4096
        # === Vary N (output hidden dim) ===
        (128,  4096, 256, 1792),    # larger N
        (128,  8192, 256, 1792),    # gate/up_proj N
        # === Vary K_high/K_low ratio ===
        (128,  2048, 128, 1920),    # 1/16 high fraction
        (128,  2048, 256, 1792),    # 1/8 high fraction (default)
        (128,  2048, 512, 1536),    # 1/4 high fraction
        (128,  2048, 1024, 1024),   # 1/2 high fraction
        # === down_proj (larger K) ===
        (128,  2048, 1024, 7168),   # down_proj
        (256,  2048, 1024, 7168),   # down_proj large M
    ]

    print(f"{'M':>4} {'N':>5} {'K_h':>4} {'K_l':>5} | "
          f"{'Fused':>8} {'IntBase':>8} {'FP16':>8} | "
          f"{'vsInt':>5} {'vsFP16':>6} | "
          f"{'TFLOPS_F':>8} {'TFLOPS16':>8} | "
          f"{'cos':>6} {'MAPE%':>6}")
    print("-" * 120)

    for M, N, K_high, K_low in test_cases:
        try:
            result = run_single_config(M, N, K_high, K_low, device)
            print(f"{result['M']:>4} {result['N']:>5} {result['K_high']:>4} {result['K_low']:>5} | "
                  f"{result['lat_fused']:>8.1f} {result['lat_baseline']:>8.1f} {result['lat_fp16']:>8.1f} | "
                  f"{result['speedup_vs_baseline']:>5.2f} {result['speedup_vs_fp16']:>6.2f} | "
                  f"{result['tops_fused']:>8.1f} {result['tops_fp16']:>8.1f} | "
                  f"{result['cos_sim']:>6.4f} {result['mape']:>6.2f}")
        except Exception as e:
            print(f"{M:>4} {N:>5} {K_high:>4} {K_low:>5} | FAILED: {e}")

    print()
    print("Latency in μs. vsInt = lat_baseline_int / lat_fused. vsFP16 = lat_fp16 / lat_fused.")
    print("Note: fused / baseline both use SM80 mma.sync. FP16 path uses cuBLAS (whatever it picks on this arch).")


if __name__ == '__main__':
    main()
