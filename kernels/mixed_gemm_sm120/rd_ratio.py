"""rd_ratio.py — RD perf diagnostic: GEMM-level K_high -> latency curve vs ideal.

Isolates the fused NVFP4(low-K)+MXFP8(high-K) SM120 co-design GEMM (the PRODUCTION
path, launch_best_codesign in config_dispatch.hpp) from the full model. For 3 rep
3B Linear shapes, sweeps K_high and measures:
  * auto  = launch_best_codesign (production dispatch; prints chosen tag)
  * t128x128 / t128x64 = the two co-design tile configs forced explicitly (rd_cd_probe)
    -> answers "does a finer per-K_high config change the work?"
  * endpoints: pure NVFP4 full-K (r=0 floor) and pure MXFP8 full-K (r=1 ceiling)
Ideal curve t_ideal(r) = t_fp4*(1-r) + t_fp8*r normalized to the MEASURED endpoints.

Run inside luofan-plaquant on a FREE GPU:  CUDA_VISIBLE_DEVICES=<n> python rd_ratio.py
Rows flushed to rd_ratio.csv per point.
"""
import csv, os, sys, torch
from torch.utils.cpp_extension import load

import nvfp4_gemm as nv
import mxfp8_gemm as mx

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
INC = [HERE, CUTLASS + "/include", CUTLASS + "/tools/util/include", CUTLASS + "/examples/common"]
BASE = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
CSV_PATH = os.path.join(HERE, "rd_ratio.csv")

# (scenario, N, K_total)  -- 3B projection shapes
SCENARIOS = [
    ("qkv_3b",  3072, 3072),   # square-ish
    ("gate_3b", 8192, 3072),   # N-heavy
    ("down_3b", 3072, 8192),   # K-heavy
]
M_LIST = [2048, 4096]


def build(kind):
    return load(name="cfgdisp_" + kind, sources=[os.path.join(HERE, "config_dispatch_test.cu")],
                extra_include_paths=INC, extra_cuda_cflags=BASE + [f"-DDISPATCH_ENABLE_{kind.upper()}"],
                verbose=False)


def build_probe():
    return load(name="rd_cd_probe", sources=[os.path.join(HERE, "rd_cd_probe.cu")],
                extra_include_paths=INC, extra_cuda_cflags=BASE + ["-DDISPATCH_ENABLE_CODESIGN"],
                verbose=False)


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def median_ms(fn, iters=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    s.sort(); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def prep_mixed_raw(M, N, K_low, K_high, seed=1):
    """Low=NVFP4, High=MXFP8. Raw oracle (NO outer scale) to match the kernels."""
    torch.manual_seed(seed); dev = "cuda"
    d = {}
    orc = torch.zeros(M, N, device=dev)
    if K_low > 0:
        Xl = torch.randn(M, K_low, device=dev); Wl = torch.randn(N, K_low, device=dev)
        Alc, SFAl, Ald = nv.quantize_nvfp4(Xl); Blc, SFBl, Bld = nv.quantize_nvfp4(Wl)
        d.update(Al=nv.pack_fp4(Alc), SFAl=SFAl, Bl=nv.pack_fp4(Blc), SFBl=SFBl)
        orc = orc + Ald @ Bld.t()
    if K_high > 0:
        Xh = torch.randn(M, K_high, device=dev); Wh = torch.randn(N, K_high, device=dev)
        Ahb, SFAh, Ahd = mx.quantize_mxfp8(Xh); Bhb, SFBh, Bhd = mx.quantize_mxfp8(Wh)
        d.update(Ah=Ahb, SFAh=SFAh, Bh=Bhb, SFBh=SFBh)
        orc = orc + Ahd @ Bhd.t()
    return d, orc


def prep_pure(M, N, K, fmt, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    X = torch.randn(M, K, device=dev); W = torch.randn(N, K, device=dev)
    if fmt == "fp4":
        Ac, SFA, Ad = nv.quantize_nvfp4(X); Bc, SFB, Bd = nv.quantize_nvfp4(W)
        return dict(A=nv.pack_fp4(Ac), SFA=SFA, B=nv.pack_fp4(Bc), SFB=SFB), Ad @ Bd.t()
    else:
        Ab, SFA, Ad = mx.quantize_mxfp8(X); Bb, SFB, Bd = mx.quantize_mxfp8(W)
        return dict(A=Ab, SFA=SFA, B=Bb, SFB=SFB), Ad @ Bd.t()


def khigh_grid(K_total):
    """Dense step-64 near low-r (region of interest), coarser above r~0.25."""
    lo_cap = K_total // 4
    ks = set(range(64, lo_cap + 1, 64))
    ks |= set(range(lo_cap, K_total, 256 if K_total <= 3072 else 512))
    ks |= {K_total // 16 // 64 * 64, K_total // 8 // 64 * 64, K_total // 4 // 64 * 64, K_total // 2 // 64 * 64}
    return sorted(k for k in ks if 64 <= k <= K_total - 64 and k % 64 == 0 and (K_total - k) % 64 == 0)


def main():
    print("Building dispatcher modules...", flush=True)
    M_NV = build("nvfp4"); M_MX = build("mxfp8"); M_CD = build("codesign")
    PB = build_probe()
    print("Built.\n", flush=True)

    f = open(CSV_PATH, "w", newline=""); w = csv.writer(f)
    w.writerow(["scenario", "M", "N", "K_total", "K_high", "r", "auto_tag",
                "lat_auto_ms", "lat_t128x128_ms", "lat_t128x64_ms", "lat_best_ms",
                "t_fp4_ms", "t_fp8_ms", "t_ideal_ms", "excess_pct", "cos_auto"])
    f.flush()

    for (scen, N, K_total) in SCENARIOS:
        for M in M_LIST:
            print(f"=== {scen} M={M} N={N} K_total={K_total} ===", flush=True)
            # endpoints (measured bounds for the ideal curve)
            p4, o4 = prep_pure(M, N, K_total, "fp4")
            c4 = cosine(M_NV.run(p4["A"], p4["SFA"], p4["B"], p4["SFB"], M, N, K_total).float(), o4)
            t4 = median_ms(lambda: M_NV.run(p4["A"], p4["SFA"], p4["B"], p4["SFB"], M, N, K_total))
            p8, o8 = prep_pure(M, N, K_total, "fp8")
            c8 = cosine(M_MX.run(p8["A"], p8["SFA"], p8["B"], p8["SFB"], M, N, K_total).float(), o8)
            t8 = median_ms(lambda: M_MX.run(p8["A"], p8["SFA"], p8["B"], p8["SFB"], M, N, K_total))
            print(f"  endpoints: t_fp4={t4:.4f} (cos {c4:.5f})  t_fp8={t8:.4f} (cos {c8:.5f})", flush=True)
            # write r=0 and r=1 endpoints as rows
            w.writerow([scen, M, N, K_total, 0, "0.0000", "pure_fp4", f"{t4:.5f}", "", "",
                        f"{t4:.5f}", f"{t4:.5f}", f"{t8:.5f}", f"{t4:.5f}", "0.00", f"{c4:.6f}"])
            w.writerow([scen, M, N, K_total, K_total, "1.0000", "pure_fp8", f"{t8:.5f}", "", "",
                        f"{t8:.5f}", f"{t4:.5f}", f"{t8:.5f}", f"{t8:.5f}", "0.00", f"{c8:.6f}"])
            f.flush()
            del p4, o4, p8, o8; torch.cuda.empty_cache()

            for K_high in khigh_grid(K_total):
                K_low = K_total - K_high
                r = K_high / K_total
                p, orc = prep_mixed_raw(M, N, K_low, K_high)
                args = (p["Al"], p["SFAl"], p["Bl"], p["SFBl"],
                        p["Ah"], p["SFAh"], p["Bh"], p["SFBh"], M, N, K_low, K_high)
                D = M_CD.run(*args).float()
                cos = cosine(D, orc); del D, orc
                tag = M_CD.tag(M, N, K_total, K_high)
                la = median_ms(lambda: M_CD.run(*args))
                lbig = median_ms(lambda: PB.run_t128x128(*args))
                lsml = median_ms(lambda: PB.run_t128x64(*args))
                lbest = min(lbig, lsml)
                t_ideal = t4 * (1 - r) + t8 * r
                excess = (la - t_ideal) / t_ideal * 100.0
                flag = "" if cos >= 0.999 else "  <<COSFAIL"
                print(f"  Kh={K_high:5d} r={r:.3f} [{tag.split(':')[-1]:14s}] "
                      f"auto={la:.4f} 128={lbig:.4f} 64={lsml:.4f} best={lbest:.4f} "
                      f"ideal={t_ideal:.4f} exc={excess:+5.1f}% cos={cos:.4f}{flag}", flush=True)
                w.writerow([scen, M, N, K_total, K_high, f"{r:.4f}", tag,
                            f"{la:.5f}", f"{lbig:.5f}", f"{lsml:.5f}", f"{lbest:.5f}",
                            f"{t4:.5f}", f"{t8:.5f}", f"{t_ideal:.5f}", f"{excess:.2f}", f"{cos:.6f}"])
                f.flush()
                del p; torch.cuda.empty_cache()
            print()
    f.close()
    print("RD RATIO DONE")


if __name__ == "__main__":
    main()
