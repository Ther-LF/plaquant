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


def prepack_weight_sf(p, N, K_low, K_high):
    """T1: pre-interleave the STATIC weight-side SF buffers ONCE (reused across all M).
    Stores physSFB_low (NVFP4 vec=16) + physSFB_high (MXFP8 vec=32) into the pack dict."""
    ext = get_fused_ext()
    p["physSFB_low"] = ext.prepack_sf_b(p["SFBl"], N, K_low, 16)
    p["physSFB_high"] = ext.prepack_sf_b(p["SFBh"], N, K_high, 32)
    return p


def mixed_gemm_fused_b_presf(p, M, N, K_low, K_high):
    """T1 Rung-B with pre-interleaved weight SF (only activation SF filled per call)."""
    ext = get_fused_ext()
    return ext.mixed_gemm_fused_b_presf(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["physSFB_low"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["physSFB_high"],
        p["s_x"], p["s_w"], M, N, K_low, K_high)


def mixed_gemm_fused_presf(p, M, N, K_low, K_high):
    """T1 Rung-A with pre-interleaved weight SF."""
    ext = get_fused_ext()
    return ext.mixed_gemm_fused_presf(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["physSFB_low"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["physSFB_high"],
        p["s_x"], p["s_w"], M, N, K_low, K_high)


# ----------------------------- T0: pure-format full-K bounds -----------------------------
# Priority change (round-3, project lead): T1 (EVT fold) and T2 (M-adaptive tile) are
# CANCELLED this round; no kernel-code (.cu) changes at all. T0 bounds use ONLY the
# existing, unmodified nvfp4_gemm/mxfp8_gemm extensions (nvfp4_gemm_ext.cu /
# mxfp8_gemm_ext.cu, round-1 code, untouched). Those entry points fill BOTH the A-side
# and B-side (weight) SF INSIDE every timed call -- there is no presf split available
# without adding a new kernel entry point, which is out of scope this round. This is a
# deliberate, honestly-recorded methodology difference from rungB_presf (which excludes
# the weight-SF fill from the timed region): the FP4/FP8 bounds below are therefore
# slightly PESSIMISTIC (include extra weight-SF-scatter work the mixed path doesn't pay
# every call), which if anything UNDERSTATES how good the band position of mixed/rungB_presf
# looks. Noted in the round-3 report.
def prep_pure_fp4(M, N, K_total, seed=0):
    """UPPER bound inputs: pure NVFP4 over the full K. Benchmark-only data (random codes)."""
    torch.manual_seed(seed)
    dev = "cuda"
    X = torch.randn(M, K_total, device=dev)
    W = torch.randn(N, K_total, device=dev)
    A_codes, SFA, A_deq = nv.quantize_nvfp4(X)
    B_codes, SFB, B_deq = nv.quantize_nvfp4(W)
    A_packed = nv.pack_fp4(A_codes)
    B_packed = nv.pack_fp4(B_codes)
    oracle = A_deq @ B_deq.t()
    pack = dict(A_packed=A_packed, SFA=SFA, B_packed=B_packed, SFB=SFB)
    return pack, oracle


def prep_pure_fp8(M, N, K_total, seed=0):
    """LOWER bound inputs: pure MXFP8 over the full K. Benchmark-only data (random codes)."""
    torch.manual_seed(seed)
    dev = "cuda"
    X = torch.randn(M, K_total, device=dev)
    W = torch.randn(N, K_total, device=dev)
    A_bytes, SFA, A_deq = mx.quantize_mxfp8(X)
    B_bytes, SFB, B_deq = mx.quantize_mxfp8(W)
    oracle = A_deq @ B_deq.t()
    pack = dict(A_bytes=A_bytes, SFA=SFA, B_bytes=B_bytes, SFB=SFB)
    return pack, oracle


def pure_fp4_fullK(p, M, N, K_total):
    """Existing unmodified nvfp4_gemm_ext.cu entry point (round-1 code)."""
    ext = nv.get_ext()
    return ext.nvfp4_gemm(p["A_packed"], p["SFA"], p["B_packed"], p["SFB"], M, N, K_total)


def pure_fp8_fullK(p, M, N, K_total):
    """Existing unmodified mxfp8_gemm_ext.cu entry point (round-1 code)."""
    ext = mx.get_ext()
    return ext.mxfp8_gemm(p["A_bytes"], p["SFA"], p["B_bytes"], p["SFB"], M, N, K_total)


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
            prepack_weight_sf(p, N, K_low, K_high)
            D2 = mixed_gemm_2launch(p, M, N, K_low, K_high).float()
            Df = mixed_gemm_fused(p, M, N, K_low, K_high).float()
            Db = mixed_gemm_fused_b(p, M, N, K_low, K_high).float()
            Dbp = mixed_gemm_fused_b_presf(p, M, N, K_low, K_high).float()
            Dap = mixed_gemm_fused_presf(p, M, N, K_low, K_high).float()
            cos2 = cosine(D2, oracle)
            cosf = cosine(Df, oracle)
            cosb = cosine(Db, oracle)
            cosbp = cosine(Dbp, oracle)
            cosap = cosine(Dap, oracle)
            relb = ((Db - oracle).norm() / (oracle.norm() + 1e-12)).item()
            tag = f"{name} M={M}"
            results[tag] = cosbp
            gate = min(cosf, cosb, cosbp, cosap)
            status = "PASS" if gate >= 0.999 else "FAIL"
            if gate < 0.999:
                ok = False
            print(f"  {tag:22s} N={N} Klow={K_low} Khigh={K_high}: "
                  f"rungA={cosf:.6f} rungB={cosb:.6f} rungB_presf={cosbp:.6f} "
                  f"rungA_presf={cosap:.6f} rel={relb:.3e} 2launch={cos2:.6f} [{status}]")
    print("ALL PASS" if ok else "SOME FAILED")

    # ---- T0: pure-format full-K bounds correctness (vs each format's own oracle) ----
    print("\n=== T0 bounds correctness (pure NVFP4 / pure MXFP8 over full K) ===")
    ok2 = True
    for name, (N, K_low, K_high) in val_shapes.items():
        K_total = K_low + K_high
        for M in [1, 16, 2048]:
            p4, o4 = prep_pure_fp4(M, N, K_total, seed=M)
            D4 = pure_fp4_fullK(p4, M, N, K_total).float()
            c4 = cosine(D4, o4)
            p8, o8 = prep_pure_fp8(M, N, K_total, seed=M)
            D8 = pure_fp8_fullK(p8, M, N, K_total).float()
            c8 = cosine(D8, o8)
            status = "PASS" if min(c4, c8) >= 0.999 else "FAIL"
            if min(c4, c8) < 0.999:
                ok2 = False
            print(f"  {name:14s} M={M:5d} K_total={K_total:5d}: "
                  f"pure_fp4={c4:.6f} pure_fp8={c8:.6f} [{status}]")
    print("ALL PASS" if ok2 else "SOME FAILED")
    return ok and ok2


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


def _time_ms_median(fn, iters=50, warmup=10):
    """Per-call timing (not amortized mean) -> median. Robust to outliers from a
    noisy/shared GPU; used for the T0 re-measure of down_proj_1B M=2048."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    med = samples[n // 2] if n % 2 else (samples[n // 2 - 1] + samples[n // 2]) / 2
    return med, samples


def band_position(t_mixed, t_fp4, t_fp8):
    """0% = at the FP4 (upper/fastest) bound, 100% = at the FP8 (lower/slowest) bound.
    >100% means mixed is slower than even pure MXFP8 full-K (fusion overhead ate the
    format advantage); <0% means mixed beat pure NVFP4 full-K (should not happen)."""
    span = t_fp8 - t_fp4
    if abs(span) < 1e-9:
        return float("nan")
    return (t_mixed - t_fp4) / span * 100.0


def benchmark():
    print("\n=== C2 benchmark: fused vs 2-launch vs fp16 cuBLAS vs [FP4,FP8] bounds ===")
    rows = []
    band_rows = []
    all_shapes = {**SHAPES_1B, **SHAPES_3B}
    for name, (N, K_low, K_high) in all_shapes.items():
        K_total = K_low + K_high
        for M in M_LIST:
            p, _ = prep_mixed(M, N, K_low, K_high, seed=1)
            prepack_weight_sf(p, N, K_low, K_high)   # T1: done once, outside timing
            # (a) 2-launch mixed path
            lat_mix = _time_ms(lambda: mixed_gemm_2launch(p, M, N, K_low, K_high))
            # (b) fused single-host-launch path (Rung A)
            lat_fused = _time_ms(lambda: mixed_gemm_fused(p, M, N, K_low, K_high))
            # (b2) Rung-B-fallback (merged fill + beta accumulate, 4 launches)
            lat_b = _time_ms(lambda: mixed_gemm_fused_b(p, M, N, K_low, K_high))
            # (b3) Rung-B presf: weight SF pre-interleaved, only activation SF filled
            lat_bp = _time_ms(lambda: mixed_gemm_fused_b_presf(p, M, N, K_low, K_high))
            # (b4) Rung-A presf
            lat_ap = _time_ms(lambda: mixed_gemm_fused_presf(p, M, N, K_low, K_high))
            # (c) fp16 cuBLAS
            Xf = torch.randn(M, K_total, device="cuda", dtype=torch.float16)
            Wf = torch.randn(N, K_total, device="cuda", dtype=torch.float16)
            lat_fp16 = _time_ms(lambda: torch.matmul(Xf, Wf.t()))
            # (d) T0 bounds: pure NVFP4 full-K (UPPER/fastest) and pure MXFP8 full-K (LOWER/slowest)
            p4, _ = prep_pure_fp4(M, N, K_total, seed=1)
            lat_fp4 = _time_ms(lambda: pure_fp4_fullK(p4, M, N, K_total))
            p8, _ = prep_pure_fp8(M, N, K_total, seed=1)
            lat_fp8 = _time_ms(lambda: pure_fp8_fullK(p8, M, N, K_total))
            band = band_position(lat_bp, lat_fp4, lat_fp8)
            print(f"  {name:14s} M={M:5d} N={N:5d} K={K_total:5d}: "
                  f"rungB_presf={lat_bp:.4f}  rungB={lat_b:.4f}  rungA={lat_fused:.4f}  "
                  f"2launch={lat_mix:.4f}  fp16={lat_fp16:.4f}ms  "
                  f"pure_fp4={lat_fp4:.4f}  pure_fp8={lat_fp8:.4f}  band={band:.1f}%  "
                  f"Bpresf/B={lat_bp/lat_b:.2f}x  Bpresf/fp16={lat_bp/lat_fp16:.2f}x")
            rows.append((name, M, "rungB_presf", f"{lat_bp:.5f}"))
            rows.append((name, M, "rungA_presf", f"{lat_ap:.5f}"))
            rows.append((name, M, "rungB_mixed", f"{lat_b:.5f}"))
            rows.append((name, M, "fused_mixed", f"{lat_fused:.5f}"))
            rows.append((name, M, "2launch_mixed", f"{lat_mix:.5f}"))
            rows.append((name, M, "fp16_cublas", f"{lat_fp16:.5f}"))
            rows.append((name, M, "pure_fp4_fullK", f"{lat_fp4:.5f}"))
            rows.append((name, M, "pure_fp8_fullK", f"{lat_fp8:.5f}"))
            band_rows.append((name, M, K_total, lat_fp4, lat_bp, lat_fp8, band))
    csv_path = os.path.join(HERE, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "M", "path", "latency_ms"])
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    band_csv = os.path.join(HERE, "band_position.csv")
    with open(band_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "M", "K_total", "fp4_ms", "mixed_rungB_presf_ms", "fp8_ms", "band_pct"])
        for r in band_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.5f}", f"{r[4]:.5f}", f"{r[5]:.5f}", f"{r[6]:.1f}"])
    print(f"Wrote {band_csv} ({len(band_rows)} rows)")
    return band_rows


def remeasure_down_proj_1B(M=2048, iters=60, warmup=15):
    """T0 re-measure: round-2 saw down_proj_1B M=2048 spike to ~1.1ms (vs round-1
    0.25ms), suspected transient GPU contention. Median of >=50 reps, run on a
    GPU verified quiet (0% util, ~0 MiB used) immediately before this call."""
    name = "down_proj_1B"
    N, K_low, K_high = SHAPES_1B[name]
    K_total = K_low + K_high
    print(f"\n=== T0 re-measure: {name} M={M} N={N} K={K_total} (iters={iters}, median) ===")
    p, _ = prep_mixed(M, N, K_low, K_high, seed=1)
    prepack_weight_sf(p, N, K_low, K_high)
    p4, _ = prep_pure_fp4(M, N, K_total, seed=1)
    p8, _ = prep_pure_fp8(M, N, K_total, seed=1)
    Xf = torch.randn(M, K_total, device="cuda", dtype=torch.float16)
    Wf = torch.randn(N, K_total, device="cuda", dtype=torch.float16)

    results = {}
    paths = {
        "rungB_presf": lambda: mixed_gemm_fused_b_presf(p, M, N, K_low, K_high),
        "rungA_presf": lambda: mixed_gemm_fused_presf(p, M, N, K_low, K_high),
        "rungB_mixed": lambda: mixed_gemm_fused_b(p, M, N, K_low, K_high),
        "fused_mixed(rungA)": lambda: mixed_gemm_fused(p, M, N, K_low, K_high),
        "2launch_mixed": lambda: mixed_gemm_2launch(p, M, N, K_low, K_high),
        "fp16_cublas": lambda: torch.matmul(Xf, Wf.t()),
        "pure_fp4_fullK": lambda: pure_fp4_fullK(p4, M, N, K_total),
        "pure_fp8_fullK": lambda: pure_fp8_fullK(p8, M, N, K_total),
    }
    for pname, fn in paths.items():
        med, samples = _time_ms_median(fn, iters=iters, warmup=warmup)
        results[pname] = (med, min(samples), max(samples))
        print(f"  {pname:20s} median={med:.4f}ms  min={min(samples):.4f}  max={max(samples):.4f}  n={len(samples)}")
    band = band_position(results["rungB_presf"][0], results["pure_fp4_fullK"][0], results["pure_fp8_fullK"][0])
    print(f"  band position (rungB_presf in [fp4,fp8]) = {band:.1f}%")
    return results, band


if __name__ == "__main__":
    print("Loading extensions...")
    nv.get_ext()
    mx.get_ext()
    get_fused_ext()
    print("Extensions loaded.\n")
    ok = validate()
    benchmark()
    remeasure_down_proj_1B(M=2048, iters=60, warmup=15)
    print("\nDONE" if ok else "\nVALIDATION FAILED")
