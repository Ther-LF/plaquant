"""Round-23 Part A: GEMM-only, clean, matched-clock, ABSOLUTE %-of-peak.

Measures GEMM-ONLY latency (inputs pre-packed; NO quant/pack in the timed region)
for:
  * co-design GEMM      = cd.fused_gemm_prepacked  (activation SF pre-interleaved)
  * shipping GEMM       = tma.b1_fused_tma_dual    (natural SF, single launch, no scatter)
  * pure-NVFP4 presf    = b1.pure_fp4_presf        (CUTLASS block-scaled NVFP4 bound)
  * pure-MXFP8 presf    = b1.pure_fp8_presf        (CUTLASS block-scaled MXFP8 bound)

shapes qkv(K2048,N2048) / gate(K2048,N8192) / down(K8192,N2048)
  x M{512,2048,4096,8192} x r{0, 1/8, 1/2, 1}

achieved TFLOPS = 2*M*N*K_total / latency_s
%-of-peak: against the SM120 hardware peak at the LOCKED clock.

HARDWARE PEAK (RTX PRO 6000 Blackwell Server Edition, SM120):
  Source: HPE QuickSpecs 'NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB PCIe':
    1.6 PFLOP FP4 TC, 811 TFLOPS FP8 TC, 406 TFLOPS FP16 (all DENSE, each a clean 2x).
  Per-SM Blackwell 5th-gen tensor rate (cross-checked vs RTX 5090 FP16=419TF@170SM,2.41GHz):
    FP16 1024, FP8(MXFP8) 2048, FP4(NVFP4) 4096 dense FLOP/SM/clk.
  At the measured/locked clock (read live below) x 188 SM:
    peak_nvfp4 = 188 * clk * 4096 ;  peak_mxfp8 = 188 * clk * 2048.
  Effective mixed peak at ratio r (fraction r of K is MXFP8 half-rate):
    eff_peak(r) = peak_nvfp4 / (1 + r)   [r=0 -> nvfp4, r=1 -> mxfp8].

median-of-N warm; also reports MIN (best = least-contended sample) since the box is shared.
CSV line-buffered; harvest from disk.
"""
import os, csv, sys, subprocess, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1
import b1_tma as bt

SM_COUNT = 188
FLOP_PER_SM_CLK_FP4 = 4096          # dense NVFP4
FLOP_PER_SM_CLK_FP8 = 2048          # dense MXFP8


def read_clock_mhz():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
             "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")]).decode()
        return float(out.strip().splitlines()[0])
    except Exception:
        return float('nan')


def get_cd():
    return load(name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=False)


cd = get_cd(); print("CD_BUILD_OK", flush=True)
tma = bt.get_tma(); b1.get_ext(); b1.get_presf_ext(); nv.get_ext(); mx.get_ext()
print("ALL_BUILD_OK", flush=True)

CLK_MHZ = read_clock_mhz()
CLK_HZ = CLK_MHZ * 1e6
PEAK_FP4 = SM_COUNT * CLK_HZ * FLOP_PER_SM_CLK_FP4 / 1e12   # TFLOPS
PEAK_FP8 = SM_COUNT * CLK_HZ * FLOP_PER_SM_CLK_FP8 / 1e12
print(f"CLK={CLK_MHZ}MHz  PEAK_NVFP4={PEAK_FP4:.1f}TF  PEAK_MXFP8={PEAK_FP8:.1f}TF", flush=True)


def peaks_at(clk_mhz):
    hz = clk_mhz * 1e6
    return (SM_COUNT * hz * FLOP_PER_SM_CLK_FP4 / 1e12,
            SM_COUNT * hz * FLOP_PER_SM_CLK_FP8 / 1e12)


SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1),
          ("down", 2048, 8192, 0, 0, 0)]
MLIST = [512, 2048, 4096, 8192]
R = [0.0, 0.125, 0.5, 1.0]
ITERS = 60
WARMUP = 15


def prep_mixed_codes(M, N, K_low, K_high, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    Xl = torch.randn(M, K_low, device=dev) if K_low else None
    Wl = torch.randn(N, K_low, device=dev) if K_low else None
    pack = {}
    if K_low:
        Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
        Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
        pack.update(Al_codes=Al_codes, Al_packed=nv.pack_fp4(Al_codes), SFAl=SFAl,
                    Bl_packed=nv.pack_fp4(Bl_codes), SFBl=SFBl, Al_deq=Al_deq, Bl_deq=Bl_deq)
    if K_high:
        Xh = torch.randn(M, K_high, device=dev); Wh = torch.randn(N, K_high, device=dev)
        Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
        Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
        pack.update(Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh,
                    Ah_deq=Ah_deq, Bh_deq=Bh_deq)
    s_x = (0.5 + torch.rand(M, device=dev)).float()
    s_w = (0.5 + torch.rand(N, device=dev)).float()
    pack["s_x"] = s_x; pack["s_w"] = s_w
    oracle = torch.zeros(M, N, device=dev)
    if K_low:  oracle += pack["Al_deq"] @ pack["Bl_deq"].t()
    if K_high: oracle += pack["Ah_deq"] @ pack["Bh_deq"].t()
    oracle = oracle * s_x.view(M, 1) * s_w.view(1, N)
    return pack, oracle


def med_min(fn):
    m, samples = mg._time_ms_median(fn, iters=ITERS, warmup=WARMUP)
    return m * 1000.0, min(samples) * 1000.0  # us


def interleave(named_fns):
    """Round-robin timed sampling of several kernels so they all experience the
    SAME contention distribution -> clean ratios AND clean min-absolute on a
    shared box. Returns {name:(median_us, min_us)}."""
    for _, fn in named_fns:
        for _ in range(WARMUP):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name, _ in named_fns}
    for _ in range(ITERS):
        for name, fn in named_fns:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            samples[name].append(s.elapsed_time(e))
    out = {}
    for name, sl in samples.items():
        sl.sort(); n = len(sl)
        med = sl[n // 2] if n % 2 else (sl[n // 2 - 1] + sl[n // 2]) / 2
        out[name] = (med * 1000.0, sl[0] * 1000.0)
    return out


def tflops(M, N, Kt, us):
    if us != us or us <= 0:
        return float('nan')
    return 2.0 * M * N * Kt / (us * 1e-6) / 1e12


CSV = os.path.join(HERE, "r23_partA.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "clk_mhz",
            "cos_cd", "cos_ship",
            "cd_us", "cd_min_us", "ship_us", "ship_min_us", "fp4_us", "fp8_us",
            "cd_tflops", "cd_pct_peak", "ship_tflops", "ship_pct_peak",
            "fp4_tflops", "fp4_pct_peak", "fp8_tflops", "fp8_pct_peak",
            "eff_peak_tf", "winner", "ship_over_cd"])
print(f"{'shape':6s}{'M':>6s}{'r':>6s}{'cdTF':>8s}{'cd%':>6s}{'shTF':>8s}{'sh%':>6s}"
      f"{'fp4%':>6s}{'fp8%':>6s}{'win':>6s}{'sh/cd':>7s}", flush=True)
allgate = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in MLIST:
        for r in R:
            Kh = KT if r == 1.0 else (0 if r == 0.0 else int(round(r * KT)))
            Kl = KT - Kh
            kp = 0 if r == 0.0 else 1
            cos_cd = cos_ship = float('nan')
            res = {}
            try:
                # pure-format bounds (full KT) prepped per-r so they interleave with
                # cd/ship in the SAME contention window -> self-consistent row.
                p4b, p8b = b1.prep_bounds_presf(M, N, KT, seed=1)
                p, oracle = prep_mixed_codes(M, N, Kl, Kh, seed=1)
                c4 = b1.cosine(b1.pure_fp4_presf(p4b, M, N, KT).float(), p4b["oracle"])
                c8 = b1.cosine(b1.pure_fp8_presf(p8b, M, N, KT).float(), p8b["oracle"])
                if min(c4, c8) < 0.999:
                    allgate = False; print(f"  BOUND COS FAIL {name} M{M} r{r}: {c4:.6f} {c8:.6f}", flush=True)
                named = [("fp4", lambda: b1.pure_fp4_presf(p4b, M, N, KT)),
                         ("fp8", lambda: b1.pure_fp8_presf(p8b, M, N, KT))]
                if r == 0.0:
                    physB = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    D = cd.single_nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB, M, N, Kl).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle); del D
                    named.append(("cd", lambda: cd.single_nvfp4_gemm(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], physB, M, N, Kl)))
                elif r < 1.0:
                    physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
                    o4i = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
                    o8i = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
                    D = cd.fused_gemm_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                M, N, Kl, Kh).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle); del D
                    Ds = tma.b1_fused_tma_dual(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
                        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0).float()
                    cos_ship = b1.cosine(Ds, oracle); del Ds
                    named.append(("cd", lambda: cd.fused_gemm_prepacked(
                        o4i[0], o4i[1], p["Bl_packed"], physB_lo, o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                        M, N, Kl, Kh)))
                    named.append(("ship", lambda: tma.b1_fused_tma_dual(
                        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
                        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0)))
                res = interleave(named)
            except Exception as e:
                print(f"  ERR {name} M{M} r{r}: {type(e).__name__} {str(e).splitlines()[0][:80]}", flush=True)
            if cos_cd == cos_cd and cos_cd < 0.999:
                allgate = False; print(f"  CD COS FAIL {name} M{M} r{r}: {cos_cd:.6f}", flush=True)
            if cos_ship == cos_ship and cos_ship < 0.999:
                allgate = False; print(f"  SHIP COS FAIL {name} M{M} r{r}: {cos_ship:.6f}", flush=True)
            cd_us, cd_min = res.get("cd", (float('nan'), float('nan')))
            ship_us, ship_min = res.get("ship", (float('nan'), float('nan')))
            _, t4m = res.get("fp4", (float('nan'), float('nan')))
            _, t8m = res.get("fp8", (float('nan'), float('nan')))
            clk = read_clock_mhz()
            pk4, pk8 = peaks_at(clk if clk == clk and clk > 0 else CLK_MHZ)
            ep = pk4 / (1.0 + r)
            cdTF = tflops(M, N, KT, cd_min); shTF = tflops(M, N, KT, ship_min)
            f4TF = tflops(M, N, KT, t4m); f8TF = tflops(M, N, KT, t8m)
            cdP = cdTF / ep * 100 if cdTF == cdTF else float('nan')
            shP = shTF / ep * 100 if shTF == shTF else float('nan')
            f4P = f4TF / pk4 * 100 if f4TF == f4TF else float('nan')
            f8P = f8TF / pk8 * 100 if f8TF == f8TF else float('nan')
            win = ""; soc = float('nan')
            if cd_min == cd_min and ship_min == ship_min:
                win = "ship" if ship_min < cd_min else "codesign"
                soc = ship_min / cd_min
            w.writerow([name, M, N, KT, f"{r:.4f}", Kl, Kh, f"{clk:.0f}",
                        f"{cos_cd:.6f}", f"{cos_ship:.6f}",
                        f"{cd_us:.3f}", f"{cd_min:.3f}", f"{ship_us:.3f}", f"{ship_min:.3f}",
                        f"{t4m:.3f}", f"{t8m:.3f}",
                        f"{cdTF:.1f}", f"{cdP:.1f}", f"{shTF:.1f}", f"{shP:.1f}",
                        f"{f4TF:.1f}", f"{f4P:.1f}", f"{f8TF:.1f}", f"{f8P:.1f}",
                        f"{ep:.1f}", win, f"{soc:.4f}"])
            print(f"{name:6s}{M:6d}{r:6.3f}{cdTF:8.0f}{cdP:6.1f}{shTF:8.0f}{shP:6.1f}"
                  f"{f4P:6.1f}{f8P:6.1f}{win:>6s}{soc:7.3f}", flush=True)
            try:
                del p, oracle, p4b, p8b
            except Exception:
                pass
            torch.cuda.empty_cache()
f.close()
print("R23_PARTA_ALL_COSINE_PASS" if allgate else "R23_PARTA_COSINE_FAILED", flush=True)
print("R23_PARTA_DONE", flush=True)
