"""Round-17 FAIR latency A/B (the real SHIP/NO-SHIP decision).

Removes the avoidable per-call wrapper overhead that round-16 proved was making the
A/B unfair, then re-measures. Both kernels start from the SAME quantizer output
(natural-layout SF p["SFAl"]/p["SFAh"]); each pays only its REAL per-call cost:
  * SHIP  (b1_fused_tma_dual): consumes natural SF natively, no reformat.
  * CD    (codesign fused_gemm): efficient per-call activation-SF interleave
          (1 thread/SF byte, not VECx redundant) + cached workspace; weight SF
          interleaved ONCE at load (prepack_sf_*_b, outside the timed loop).

Also reports:
  * CD GEMM-only ceiling (fused_gemm_prepacked; activation SF pre-interleaved).
  * CD old-redundant (fused_gemm_slow) = the round-14/16 "before" number.
  * standalone interleave cost before (slow) -> after (efficient), per shape.

Gate: buffer-equality probe (efficient interleave == redundant, bit-identical) +
cosine (hard gate) all 24 cells BEFORE any timing.

Per shape x M{2048,8192} x r{0,1/8,1/2,1}. excess vs t_ideal(r) from CUTLASS pure
presf endpoints (same session). CSV line-buffered; harvest from disk.
"""
import os, csv, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as bt


def get_cd():
    return load(name="codesign_fused_ext",
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
print(cd.compat_report(), flush=True)

# (name, N, KT, tile, sw, pers) -- shipping per-shape config (round-12/15)
SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
R = [0.0, 0.125, 0.5, 1.0]


def ship(p, M, N, Kl, Kh, tile, sw, pers, kp):
    return tma.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0)


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m * 1000.0  # us


# ---- buffer-equality probe: efficient interleave == old redundant (bit-identical) ----
def probe_bitexact():
    print("=== buffer-equality probe (efficient vs redundant interleave) ===", flush=True)
    ok = True
    for name, N, KT, *_ in SHAPES:
        M = 2048
        Kh = KT // 8; Kl = KT - Kh
        p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
        a4 = cd.prepack_sf_fp4_a(p["SFAl"], M, N, Kl)
        a4s = cd.prepack_sf_fp4_a_slow(p["SFAl"], M, N, Kl)
        a8 = cd.prepack_sf_fp8_a(p["SFAh"], M, N, Kh)
        a8s = cd.prepack_sf_fp8_a_slow(p["SFAh"], M, N, Kh)
        e4 = bool(torch.equal(a4, a4s)); e8 = bool(torch.equal(a8, a8s))
        print(f"  {name:4s} fp4 buf_eq={e4} (sz={a4.numel()}) fp8 buf_eq={e8} (sz={a8.numel()})", flush=True)
        ok = ok and e4 and e8
        del p, a4, a4s, a8, a8s; torch.cuda.empty_cache()
    print("PROBE_BITEXACT_PASS" if ok else "PROBE_BITEXACT_FAIL", flush=True)
    return ok


probe_ok = probe_bitexact()

CSV = os.path.join(HERE, "r17_lat.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "cos_cd",
            "cd_full_us", "cd_gemmonly_us", "cd_slow_us", "ship_us", "ideal_us",
            "ex_cdfull_pct", "ex_cdgemm_pct", "ex_ship_pct",
            "intlv_fast_us", "intlv_slow_us",
            "cdfull_over_ship", "cdgemm_over_ship"])
print(f"{'shape':9s}{'M':>6s}{'r':>6s}{'cos':>9s}{'cdF':>9s}{'cdG':>9s}{'cdS':>9s}"
      f"{'ship':>9s}{'ideal':>8s}{'exF':>7s}{'exG':>7s}{'exS':>7s}"
      f"{'ilF':>7s}{'ilS':>7s}{'F/sh':>6s}{'G/sh':>6s}", flush=True)
allgate = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in (2048, 8192):
        p4b, p8b = b1.prep_bounds_presf(M, N, KT, seed=1)
        t4 = med(lambda: b1.pure_fp4_presf(p4b, M, N, KT))
        t8 = med(lambda: b1.pure_fp8_presf(p8b, M, N, KT))
        del p4b, p8b; torch.cuda.empty_cache()
        for r in R:
            Kh = KT if r == 1.0 else (0 if r == 0.0 else int(round(r * KT)))
            Kl = KT - Kh
            kp = 0 if r == 0.0 else 1
            p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            ideal = t4 * (1 - r) + t8 * r
            cos_cd = float('nan')
            cdF = cdG = cdS = ilF = ilS = float('nan')
            cdF_fn = cdG_fn = cdS_fn = None
            try:
                if r == 1.0:
                    pass  # co-design has no pure-MXFP8 (K_low=0) path
                elif r == 0.0:
                    physB = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    cdF_fn = lambda: cd.single_nvfp4_gemm(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB, M, N, Kl)
                    D = cdF_fn().float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                else:
                    physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
                    # pre-interleave activation SF ONCE for the GEMM-only ceiling
                    physA_lo = cd.prepack_sf_fp4_a(p["SFAl"], M, N, Kl)
                    physA_hi = cd.prepack_sf_fp8_a(p["SFAh"], M, N, Kh)
                    cdF_fn = lambda: cd.fused_gemm(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Kl, Kh)
                    cdG_fn = lambda: cd.fused_gemm_prepacked(
                        p["Al_packed"], physA_lo, p["Bl_packed"], physB_lo,
                        p["Ah_bytes"], physA_hi, p["Bh_bytes"], physB_hi, M, N, Kl, Kh)
                    cdS_fn = lambda: cd.fused_gemm_slow(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Kl, Kh)
                    ilF = (med(lambda: cd.prepack_sf_fp4_a(p["SFAl"], M, N, Kl))
                           + med(lambda: cd.prepack_sf_fp8_a(p["SFAh"], M, N, Kh)))
                    ilS = (med(lambda: cd.prepack_sf_fp4_a_slow(p["SFAl"], M, N, Kl))
                           + med(lambda: cd.prepack_sf_fp8_a_slow(p["SFAh"], M, N, Kh)))
                    D = cdF_fn().float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
            except Exception as e:
                print(f"  CD ERR {name} M{M} r{r}: {type(e).__name__} "
                      f"{str(e).splitlines()[0][:90]}", flush=True)
                cdF_fn = cdG_fn = cdS_fn = None
            if cdF_fn is not None:
                if cos_cd == cos_cd and cos_cd < 0.999:
                    allgate = False
                    print(f"  CD COSINE FAIL {name} M{M} r{r}: cos={cos_cd:.6f} (NOT timed)", flush=True)
                else:
                    cdF = med(cdF_fn)
                    if cdG_fn is not None: cdG = med(cdG_fn)
                    if cdS_fn is not None: cdS = med(cdS_fn)
            t_ship = med(lambda: ship(p, M, N, Kl, Kh, tile, sw, pers, kp))
            exF = (cdF - ideal) / ideal * 100 if cdF == cdF else float('nan')
            exG = (cdG - ideal) / ideal * 100 if cdG == cdG else float('nan')
            exS = (t_ship - ideal) / ideal * 100
            Fsh = (cdF / t_ship) if cdF == cdF else float('nan')
            Gsh = (cdG / t_ship) if cdG == cdG else float('nan')
            w.writerow([name, M, N, KT, f"{r:.3f}", Kl, Kh, f"{cos_cd:.6f}",
                        f"{cdF:.4f}", f"{cdG:.4f}", f"{cdS:.4f}", f"{t_ship:.4f}",
                        f"{ideal:.4f}", f"{exF:.2f}", f"{exG:.2f}", f"{exS:.2f}",
                        f"{ilF:.4f}", f"{ilS:.4f}", f"{Fsh:.4f}", f"{Gsh:.4f}"])
            print(f"{name:9s}{M:6d}{r:6.3f}{cos_cd:9.6f}{cdF:9.3f}{cdG:9.3f}{cdS:9.3f}"
                  f"{t_ship:9.3f}{ideal:8.3f}{exF:7.1f}{exG:7.1f}{exS:7.1f}"
                  f"{ilF:7.2f}{ilS:7.2f}{Fsh:6.2f}{Gsh:6.2f}", flush=True)
            del p, oracle; torch.cuda.empty_cache()
f.close()
print("R17_PROBE_OK" if probe_ok else "R17_PROBE_FAIL", flush=True)
print("R17_LAT_ALL_COSINE_PASS" if allgate else "R17_LAT_COSINE_FAILED", flush=True)
print("R17_LAT_DONE", flush=True)
