"""pp_ratio.py — RD PERF-PUZZLE Stage 1/2 + scaling: pure-FP4 vs pure-FP8 GEMM
latency ratio across shapes, to build the waterfall:

  STAGE 1 (hardware peak): large SQUARE GEMM (M=N=K in {4096,8192,16384}) ->
           confirm the FP4/FP8 ratio approaches the ~1.82x dense peak ratio.
  STAGE 2 (LLM shapes):   the ACTUAL 3B/8B Linear shapes at M=2048 (prefill) ->
           show the ratio collapses to ~1.2-1.5x because these skinny/short-K
           shapes don't saturate the tensor cores (achieved TFLOPS << peak).
  SCALING:  sweep M (512..16384) at a fixed square, and sweep K=N (2048..16384)
            at M=2048 -> show the ratio climbing from ~1.3 back toward 1.82 as
            the problem gets bigger (the lever scales with model+batch size).

Each point measures pure-NVFP4 full-K and pure-MXFP8 full-K via the dedicated
SM120 CUTLASS format dispatchers (config_dispatch_test.cu, same path rd_ratio
uses for its endpoints). Reports t_fp4, t_fp8, ratio=t_fp8/t_fp4, and achieved
TFLOPS + %-of-peak (FP4 peak 1469, FP8 peak 807 TFLOPS dense, per
docs/hardware_format_peak.md) so TC saturation is visible per shape.

Run inside luofan-plaquant on a FREE GPU:
  CUDA_VISIBLE_DEVICES=<n> python pp_ratio.py --out /home/luofan/PP_RATIO.csv
"""
import argparse, csv, os, sys, json, torch
from torch.utils.cpp_extension import load

import nvfp4_gemm as nv
import mxfp8_gemm as mx

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
INC = [HERE, CUTLASS + "/include", CUTLASS + "/tools/util/include",
       CUTLASS + "/examples/common"]
BASE = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]

PEAK_FP4 = 1469.0   # TFLOPS dense, docs/hardware_format_peak.md
PEAK_FP8 = 807.0


def build(kind):
    return load(name="cfgdisp_" + kind,
                sources=[os.path.join(HERE, "config_dispatch_test.cu")],
                extra_include_paths=INC,
                extra_cuda_cflags=BASE + [f"-DDISPATCH_ENABLE_{kind.upper()}"],
                verbose=False)


def median_ms(fn, iters=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    s.sort(); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def prep_pure(M, N, K, fmt, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    X = torch.randn(M, K, device=dev); W = torch.randn(N, K, device=dev)
    if fmt == "fp4":
        Ac, SFA, _ = nv.quantize_nvfp4(X); Bc, SFB, _ = nv.quantize_nvfp4(W)
        return dict(A=nv.pack_fp4(Ac), SFA=SFA, B=nv.pack_fp4(Bc), SFB=SFB)
    Ab, SFA, _ = mx.quantize_mxfp8(X); Bb, SFB, _ = mx.quantize_mxfp8(W)
    return dict(A=Ab, SFA=SFA, B=Bb, SFB=SFB)


def tflops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def measure(MNV, MMX, M, N, K):
    p4 = prep_pure(M, N, K, "fp4")
    t4 = median_ms(lambda: MNV.run(p4["A"], p4["SFA"], p4["B"], p4["SFB"], M, N, K))
    del p4; torch.cuda.empty_cache()
    p8 = prep_pure(M, N, K, "fp8")
    t8 = median_ms(lambda: MMX.run(p8["A"], p8["SFA"], p8["B"], p8["SFB"], M, N, K))
    del p8; torch.cuda.empty_cache()
    tf4, tf8 = tflops(M, N, K, t4), tflops(M, N, K, t8)
    return dict(M=M, N=N, K=K, t_fp4_ms=round(t4, 5), t_fp8_ms=round(t8, 5),
                ratio_fp8_over_fp4=round(t8 / t4, 4),
                tflops_fp4=round(tf4, 1), tflops_fp8=round(tf8, 1),
                pct_peak_fp4=round(100 * tf4 / PEAK_FP4, 1),
                pct_peak_fp8=round(100 * tf8 / PEAK_FP8, 1))


# 3B: hidden 3072, inter 8192, 24h/8kv head128.  8B: hidden 4096, inter 14336, 32h/8kv.
STAGE2 = [
    ("3b_qkv",  2048, 3072, 3072),
    ("3b_kv",   2048, 1024, 3072),
    ("3b_o",    2048, 3072, 3072),
    ("3b_gate", 2048, 8192, 3072),
    ("3b_down", 2048, 3072, 8192),
    ("8b_qkv",  2048, 4096, 4096),
    ("8b_kv",   2048, 1024, 4096),
    ("8b_gate", 2048, 14336, 4096),
    ("8b_down", 2048, 4096, 14336),
]
STAGE1 = [("square_%d" % s, s, s, s) for s in (4096, 8192, 16384)]
M_SWEEP = [512, 1024, 2048, 4096, 8192, 16384]   # at square 4096
KN_SWEEP = [2048, 4096, 8192, 16384]             # at M=2048 square


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/luofan/PP_RATIO.csv")
    ap.add_argument("--json", default="/home/luofan/PP_RATIO.json")
    args = ap.parse_args()
    print("Building pure-format dispatchers...", flush=True)
    MNV = build("nvfp4"); MMX = build("mxfp8")
    print("Built.\n", flush=True)

    f = open(args.out, "w", newline=""); w = csv.writer(f)
    cols = ["stage", "tag", "M", "N", "K", "t_fp4_ms", "t_fp8_ms",
            "ratio_fp8_over_fp4", "tflops_fp4", "tflops_fp8",
            "pct_peak_fp4", "pct_peak_fp8"]
    w.writerow(cols); f.flush()
    allrows = []

    def emit(stage, tag, M, N, K):
        r = measure(MNV, MMX, M, N, K)
        r["stage"] = stage; r["tag"] = tag
        w.writerow([r[c] for c in cols]); f.flush()
        allrows.append(r)
        print(f"  [{stage}] {tag:12s} M{M} N{N} K{K}: "
              f"fp4={r['t_fp4_ms']:.4f} fp8={r['t_fp8_ms']:.4f} "
              f"ratio={r['ratio_fp8_over_fp4']:.3f}  "
              f"TF4={r['tflops_fp4']:.0f}({r['pct_peak_fp4']:.0f}%) "
              f"TF8={r['tflops_fp8']:.0f}({r['pct_peak_fp8']:.0f}%)", flush=True)

    print("=== STAGE 1: large square (hardware peak) ===", flush=True)
    for tag, M, N, K in STAGE1:
        emit("stage1_square", tag, M, N, K)
    print("=== STAGE 2: LLM Linear shapes @ M=2048 ===", flush=True)
    for tag, M, N, K in STAGE2:
        emit("stage2_llm", tag, M, N, K)
    print("=== SCALING: sweep M @ square N=K=4096 ===", flush=True)
    for M in M_SWEEP:
        emit("scaling_M", "sqM_%d" % M, M, 4096, 4096)
    print("=== SCALING: sweep K=N @ M=2048 ===", flush=True)
    for S in KN_SWEEP:
        emit("scaling_KN", "sqKN_%d" % S, 2048, S, S)

    json.dump(allrows, open(args.json, "w"), indent=2)
    print(f"WROTE {args.out} and {args.json}", flush=True)
    print("PP_RATIO_DONE", flush=True)


if __name__ == "__main__":
    main()
