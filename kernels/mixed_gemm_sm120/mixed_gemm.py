"""C1.3 — 2-launch mixed-precision baseline (NVFP4 low segment + MXFP8 high segment).

Loads both the NVFP4 and MXFP8 SM120 block-scaled GEMM extensions, combines them
into the mixed-precision GEMM the real model needs:

    D = nvfp4_gemm(A_low, B_low) + mxfp8_gemm(A_high, B_high)   # [M,N]
    out[m,n] = D[m,n] * s_x[m] * s_w[n]   -> fp16

Validates cosine >= 0.999 vs a full mixed-precision PyTorch oracle on real LLM
shapes, then benchmarks (a) the 2-launch path vs (b) fp16 cuBLAS (torch.matmul).

Run inside luofan-plaquant on a free GPU.
"""
import os
import csv
import torch

import nvfp4_gemm as nv
import mxfp8_gemm as mx

HERE = os.path.dirname(os.path.abspath(__file__))


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def prep_mixed(M, N, K_low, K_high, seed=0):
    """Generate quantized low(NVFP4)/high(MXFP8) segments + outer scales + oracle deq."""
    torch.manual_seed(seed)
    dev = "cuda"
    # low segment (NVFP4)
    Xl = torch.randn(M, K_low, device=dev)
    Wl = torch.randn(N, K_low, device=dev)
    Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes)
    Bl_packed = nv.pack_fp4(Bl_codes)
    # high segment (MXFP8)
    Xh = torch.randn(M, K_high, device=dev)
    Wh = torch.randn(N, K_high, device=dev)
    Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
    Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
    # outer scales
    s_x = (0.5 + torch.rand(M, device=dev)).float()      # per-token
    s_w = (0.5 + torch.rand(N, device=dev)).float()      # per-channel

    oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t())
    oracle = oracle * s_x.view(M, 1) * s_w.view(1, N)

    pack = dict(Al_packed=Al_packed, SFAl=SFAl, Bl_packed=Bl_packed, SFBl=SFBl,
                Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh,
                s_x=s_x, s_w=s_w)
    return pack, oracle


def mixed_gemm_2launch(p, M, N, K_low, K_high):
    """The 2-launch baseline: NVFP4 GEMM + MXFP8 GEMM + outer scale -> fp16."""
    nvext = nv.get_ext()
    mxext = mx.get_ext()
    D_low = nvext.nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
                             M, N, K_low).float()
    D_high = mxext.mxfp8_gemm(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                              M, N, K_high).float()
    D = D_low + D_high
    D = D * p["s_x"].view(M, 1) * p["s_w"].view(1, N)
    return D.half()


# Llama projection shapes (N, K_low, K_high). K_total = K_low + K_high.
SHAPES_1B = {
    "q_proj_1B":    (2048, 1792, 256),
    "k_proj_1B":    (512,  1792, 256),
    "v_proj_1B":    (512,  1792, 256),
    "o_proj_1B":    (2048, 1792, 256),
    "gate_proj_1B": (8192, 1792, 256),
    "up_proj_1B":   (8192, 1792, 256),
    "down_proj_1B": (2048, 7168, 1024),
}
SHAPES_3B = {
    "q_proj_3B":    (3072, 2688, 384),
    "down_proj_3B": (3072, 7168, 1024),  # 3B down: K_total 8192, 0.125 high -> 7168/1024
}
M_LIST = [1, 16, 256, 2048]


def validate():
    print("=== C1.3 mixed 2-launch correctness ===")
    ok = True
    results = {}
    val_shapes = {
        "q_proj_1B": (2048, 1792, 256),
        "down_proj_1B": (2048, 7168, 1024),
        "q_proj_3B": (3072, 2688, 384),
    }
    for name, (N, K_low, K_high) in val_shapes.items():
        for M in [16, 2048]:
            p, oracle = prep_mixed(M, N, K_low, K_high, seed=M)
            D = mixed_gemm_2launch(p, M, N, K_low, K_high).float()
            cos = cosine(D, oracle)
            rel = ((D - oracle).norm() / (oracle.norm() + 1e-12)).item()
            tag = f"{name} M={M}"
            results[tag] = cos
            status = "PASS" if cos >= 0.999 else "FAIL"
            if cos < 0.999:
                ok = False
            print(f"  {tag:22s} N={N} Klow={K_low} Khigh={K_high}: "
                  f"cosine={cos:.6f} rel={rel:.3e} [{status}]")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark():
    print("\n=== C1.3 benchmark: 2-launch vs fp16 cuBLAS ===")
    rows = []
    all_shapes = {**SHAPES_1B, **SHAPES_3B}
    for name, (N, K_low, K_high) in all_shapes.items():
        K_total = K_low + K_high
        for M in M_LIST:
            p, _ = prep_mixed(M, N, K_low, K_high, seed=1)
            # (a) 2-launch mixed path
            lat_mix = _time_ms(lambda: mixed_gemm_2launch(p, M, N, K_low, K_high))
            # (b) fp16 cuBLAS
            Xf = torch.randn(M, K_total, device="cuda", dtype=torch.float16)
            Wf = torch.randn(N, K_total, device="cuda", dtype=torch.float16)
            lat_fp16 = _time_ms(lambda: torch.matmul(Xf, Wf.t()))
            print(f"  {name:14s} M={M:5d} N={N:5d} K={K_total:5d}: "
                  f"2launch={lat_mix:.4f}ms  fp16={lat_fp16:.4f}ms  "
                  f"ratio={lat_mix/lat_fp16:.2f}x")
            rows.append((name, M, "2launch_mixed", f"{lat_mix:.5f}"))
            rows.append((name, M, "fp16_cublas", f"{lat_fp16:.5f}"))
    csv_path = os.path.join(HERE, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "M", "path", "latency_ms"])
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    print("Loading extensions...")
    nv.get_ext()
    mx.get_ext()
    print("Extensions loaded.\n")
    ok = validate()
    benchmark()
    print("\nDONE" if ok else "\nVALIDATION FAILED")
