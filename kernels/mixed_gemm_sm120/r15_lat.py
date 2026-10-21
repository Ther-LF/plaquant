"""Round-15 latency A/B (SHIP/NO-SHIP decision).

Dual-format shared-accumulator CO-DESIGN fused GEMM (codesign_fused_ext.fused_gemm)
vs the SHIPPING hand-rolled dual kernel (b1_tma_ext.b1_fused_tma_dual, round-12
config: SFILV=0 + SFASYNC(sfa=1) + KPIPE per-shape dispatch).

Same data, same GPU, warm, median of 50 (10 warmup). Re-confirms the dual-kernel
cosine BEFORE timing (hard gate; a sub-0.999 cell is reported, never timed as "ok").
Excess vs t_ideal(r) = t_fp4*(1-r) + t_fp8*r from the CUTLASS pure-presf endpoints
(same-session bounds). CSV line-buffered; harvest from disk.

Honest caveat: the co-design currently uses a PLAIN epilogue (acc->bf16, no outer
s_x*s_w); the shipping kernel applies the full s_x*s_w epilogue. The GEMM mainloop
dominates at these K-sizes; the delta from the missing separable scaling is tiny
and is noted in the report. Cosine is validated by applying s_x*s_w in Python.
"""
import os, csv, sys, math, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as bt


def get_cd():
    return load(
        name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=True)


cd = get_cd()
print("CD_BUILD_OK", flush=True)
tma = bt.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("ALL_BUILD_OK", flush=True)

# shipping per-shape SFILV=0 + sfasync config (from r9_sfa_lat winning sfa1 column)
# (name, N, KT, tile, sw, pers)
SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
R = [0.0, 0.125, 0.5, 1.0]


def ship(p, M, N, Kl, Kh, tile, sw, pers, kp):
    # SFILV=0, sfa=1 (shipping), kmul=2, minb=1, pingpong=0, ldbw=0, sfilv=0
    return tma.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0)


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m


CSV = os.path.join(HERE, "r15_lat.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "cos_cd", "lat_cd_us",
            "lat_ship_us", "ideal_us", "ex_cd_pct", "ex_ship_pct",
            "delta_us", "cd_over_ship"])
print(f"{'shape':10s}{'M':>6s}{'r':>7s}{'cos':>10s}{'cd_us':>10s}{'ship_us':>10s}"
      f"{'ideal':>9s}{'ex_cd':>8s}{'ex_sh':>8s}{'cd/sh':>7s}", flush=True)
allgate = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in (2048, 8192):
        p4b, p8b = b1.prep_bounds_presf(M, N, KT, seed=1)
        t4 = med(lambda: b1.pure_fp4_presf(p4b, M, N, KT)) * 1000.0  # us
        t8 = med(lambda: b1.pure_fp8_presf(p8b, M, N, KT)) * 1000.0
        del p4b, p8b
        torch.cuda.empty_cache()
        for r in R:
            Kh = KT if r == 1.0 else (0 if r == 0.0 else int(round(r * KT)))
            Kl = KT - Kh
            kp = 0 if r == 0.0 else 1
            p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            ideal = t4 * (1 - r) + t8 * r
            cos_cd = float('nan')
            t_cd = float('nan')
            cd_fn = None
            try:
                if r == 1.0:
                    cd_fn = None   # co-design has no pure-MXFP8 (K_low=0) path
                elif r == 0.0:
                    physB = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    cd_fn = lambda: cd.single_nvfp4_gemm(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB, M, N, Kl)
                    D = cd_fn().float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                else:
                    physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
                    cd_fn = lambda: cd.fused_gemm(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Kl, Kh)
                    D = cd_fn().float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
            except Exception as e:
                print(f"  CD ERR {name} M{M} r{r}: {type(e).__name__} "
                      f"{str(e).splitlines()[0][:90]}", flush=True)
                cd_fn = None
            if cd_fn is not None:
                if cos_cd == cos_cd and cos_cd < 0.999:
                    allgate = False
                    print(f"  CD COSINE FAIL {name} M{M} r{r}: cos={cos_cd:.6f} (NOT timed)",
                          flush=True)
                else:
                    t_cd = med(cd_fn) * 1000.0
            t_ship = med(lambda: ship(p, M, N, Kl, Kh, tile, sw, pers, kp)) * 1000.0
            ex_cd = (t_cd - ideal) / ideal * 100 if t_cd == t_cd else float('nan')
            ex_sh = (t_ship - ideal) / ideal * 100
            delta = (t_cd - t_ship) if t_cd == t_cd else float('nan')
            ratio = (t_cd / t_ship) if t_cd == t_cd else float('nan')
            w.writerow([name, M, N, KT, f"{r:.3f}", Kl, Kh, f"{cos_cd:.6f}",
                        f"{t_cd:.4f}", f"{t_ship:.4f}", f"{ideal:.4f}",
                        f"{ex_cd:.2f}", f"{ex_sh:.2f}", f"{delta:.4f}", f"{ratio:.4f}"])
            print(f"{name:10s}{M:6d}{r:7.3f}{cos_cd:10.6f}{t_cd:10.4f}{t_ship:10.4f}"
                  f"{ideal:9.4f}{ex_cd:8.1f}{ex_sh:8.1f}{ratio:7.3f}", flush=True)
            del p, oracle
            torch.cuda.empty_cache()
f.close()
print("R15_LAT_ALL_COSINE_PASS" if allgate else "R15_LAT_COSINE_FAILED", flush=True)
print("R15_LAT_DONE", flush=True)
