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
from torch.utils.cpp_extension import load

import nvfp4_gemm as nv
import mxfp8_gemm as mx

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_fused_ext = None


def get_fused_ext():
    """JIT-load the C2 fused single-host-launch mixed-precision GEMM extension."""
    global _fused_ext
    if _fused_ext is None:
        _fused_ext = load(
            name="fused_gemm_ext",
            sources=[os.path.join(HERE, "fused_gemm_ext.cu")],
            extra_include_paths=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
                os.path.join(CUTLASS, "examples/common"),
            ],
            extra_cuda_cflags=[
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-O3",
            ],
            verbose=True,
        )
    return _fused_ext


def mixed_gemm_fused(p, M, N, K_low, K_high):
    """C2 fused path: ONE host call (both GEMMs + combine + scale, single sync)."""
    ext = get_fused_ext()
    return ext.mixed_gemm_fused(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, K_low, K_high)


def mixed_gemm_fused_b(p, M, N, K_low, K_high):
    """Rung-B-fallback: merged SF-fill + GEMM2 beta=1 accumulate + scale (4 launches)."""
    ext = get_fused_ext()
    return ext.mixed_gemm_fused_b(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, K_low, K_high)


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
    print("=== C2 fused vs 2-launch correctness (both vs oracle) ===")
    ok = True
    results = {}
    val_shapes = {
        "q_proj_1B": (2048, 1792, 256),
        "down_proj_1B": (2048, 7168, 1024),
        "q_proj_3B": (3072, 2688, 384),
    }
    for name, (N, K_low, K_high) in val_shapes.items():
        for M in [1, 16, 256, 2048]:
            p, oracle = prep_mixed(M, N, K_low, K_high, seed=M)
            D2 = mixed_gemm_2launch(p, M, N, K_low, K_high).float()
            Df = mixed_gemm_fused(p, M, N, K_low, K_high).float()
            Db = mixed_gemm_fused_b(p, M, N, K_low, K_high).float()
            cos2 = cosine(D2, oracle)
            cosf = cosine(Df, oracle)
            cosb = cosine(Db, oracle)
            relb = ((Db - oracle).norm() / (oracle.norm() + 1e-12)).item()
            tag = f"{name} M={M}"
            results[tag] = cosb
            status = "PASS" if (cosf >= 0.999 and cosb >= 0.999) else "FAIL"
            if cosf < 0.999 or cosb < 0.999:
                ok = False
            print(f"  {tag:22s} N={N} Klow={K_low} Khigh={K_high}: "
                  f"rungA_vs_oracle={cosf:.6f}  rungB_vs_oracle={cosb:.6f} rel={relb:.3e}  "
                  f"2launch_vs_oracle={cos2:.6f} [{status}]")
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
    print("\n=== C2 benchmark: fused vs 2-launch vs fp16 cuBLAS ===")
    rows = []
    all_shapes = {**SHAPES_1B, **SHAPES_3B}
    for name, (N, K_low, K_high) in all_shapes.items():
        K_total = K_low + K_high
        for M in M_LIST:
            p, _ = prep_mixed(M, N, K_low, K_high, seed=1)
            # (a) 2-launch mixed path
            lat_mix = _time_ms(lambda: mixed_gemm_2launch(p, M, N, K_low, K_high))
            # (b) fused single-host-launch path (Rung A)
            lat_fused = _time_ms(lambda: mixed_gemm_fused(p, M, N, K_low, K_high))
            # (b2) Rung-B-fallback (merged fill + beta accumulate, 4 launches)
            lat_b = _time_ms(lambda: mixed_gemm_fused_b(p, M, N, K_low, K_high))
            # (c) fp16 cuBLAS
            Xf = torch.randn(M, K_total, device="cuda", dtype=torch.float16)
            Wf = torch.randn(N, K_total, device="cuda", dtype=torch.float16)
            lat_fp16 = _time_ms(lambda: torch.matmul(Xf, Wf.t()))
            print(f"  {name:14s} M={M:5d} N={N:5d} K={K_total:5d}: "
                  f"rungB={lat_b:.4f}  rungA={lat_fused:.4f}  2launch={lat_mix:.4f}  fp16={lat_fp16:.4f}ms  "
                  f"B/A={lat_b/lat_fused:.2f}x  B/fp16={lat_b/lat_fp16:.2f}x")
            rows.append((name, M, "rungB_mixed", f"{lat_b:.5f}"))
            rows.append((name, M, "fused_mixed", f"{lat_fused:.5f}"))
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
    get_fused_ext()
    print("Extensions loaded.\n")
    ok = validate()
    benchmark()
    print("\nDONE" if ok else "\nVALIDATION FAILED")
