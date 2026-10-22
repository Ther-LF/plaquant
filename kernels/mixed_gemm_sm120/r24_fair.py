"""Round-24 FAIR comparison regen: best-config FP4 / FP8 / co-design per shape x M x r.

Reloads the CACHED round-24 grid-search extensions (presf_cfg_* and cd_cfg24_*; no rebuild)
and, per (shape, M, r), interleave-times (median + min, live clock, same contention window --
r23_partA methodology) every viable config for each kernel, then reports:
  * fair FP4 endpoint = min over {128x128_auto, 128x128_pp, 256x128_s3, 128x64_auto}
  * fair FP8 endpoint = min over {128x128_auto, 128x128_pp, 128x64_auto}
  * fair co-design    = min over {128x128_nv4mx2, 128x64_nv4mx2}
  * OLD under-tuned pure_presf endpoint = ps_128x128_auto (== pure_presf_ext.cu config)
  * improvement = old / fair_best  (how much the fair endpoint beat the under-tuned baseline)
plus achieved TFLOPS and %-of-hardware-peak (same peak basis as round-23).

cosine gated (>=0.999) on every config before timing. Clocks locked 2430 MHz (idle GPU).
"""
import os, csv, sys, subprocess, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import mixed_gemm as mg
import b1_fused as b1

SM_COUNT = 188
FP4_PER = 4096
FP8_PER = 2048
INC = [HERE, CUTLASS+"/include", CUTLASS+"/tools/util/include", CUTLASS+"/examples/common"]
BASE = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]


def read_clk():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")]).decode()
        return float(out.strip().splitlines()[0])
    except Exception:
        return 2430.0


def load_presf(lbl, TM, TN, TK, STG, SCH):
    flags = BASE + [f"-DPSCFG_TM={TM}", f"-DPSCFG_TN={TN}", f"-DPSCFG_TK={TK}",
                    "-DPSCFG_CM=1", "-DPSCFG_CN=1", "-DPSCFG_CK=1",
                    f"-DPSCFG_STAGES={STG}", f"-DPSCFG_SCHED={SCH}"]
    return load(name="presf_cfg_" + lbl, sources=[os.path.join(HERE, "presf_cfg_ext.cu")],
                extra_include_paths=INC, extra_cuda_cflags=flags, verbose=False)


def load_cd(lbl, TM, TN, TK, NVST, MXST):
    flags = BASE + [f"-DCDCFG_TM={TM}", f"-DCDCFG_TN={TN}", f"-DCDCFG_TK={TK}",
                    "-DCDCFG_CM=1", "-DCDCFG_CN=1", "-DCDCFG_CK=1",
                    f"-DCDCFG_NVST={NVST}", f"-DCDCFG_MXST={MXST}"]
    return load(name="cd_cfg24_" + lbl, sources=[os.path.join(HERE, "cd_cfg_ext.cu")],
                extra_include_paths=INC, extra_cuda_cflags=flags, verbose=False)


print("Reloading cached grid extensions...", flush=True)
NVCFG = {  # tag -> (ext, label)
    "128x128_auto": load_presf("ps_128x128_auto", 128, 128, 128, 0, 0),
    "128x128_pp":   load_presf("ps_128x128_pp", 128, 128, 128, 0, 2),
    "256x128_s3":   load_presf("ps_256x128_s3", 256, 128, 128, 3, 0),
    "128x64_auto":  load_presf("ps_128x64_auto", 128, 64, 128, 0, 0),
}
# MXFP8 uses the same TUs (both formats compiled in each); 256x128_s3 MXFP8 overflows -> excluded.
MXCFG = {"128x128_auto": NVCFG["128x128_auto"], "128x128_pp": NVCFG["128x128_pp"], "128x64_auto": NVCFG["128x64_auto"]}
CDCFG = {
    "128x128_nv4mx2": load_cd("t128x128_nv4mx2", 128, 128, 128, 4, 2),
    "128x64_nv4mx2":  load_cd("t128x64_nv4mx2", 128, 64, 128, 4, 2),
}
print("ALL_RELOADED", flush=True)

CLK = read_clk()
PK4 = SM_COUNT * CLK * 1e6 * FP4_PER / 1e12
PK8 = SM_COUNT * CLK * 1e6 * FP8_PER / 1e12
print(f"CLK={CLK} PK4={PK4:.1f} PK8={PK8:.1f}", flush=True)

SHAPES = [("qkv", 2048, 2048), ("gate", 8192, 2048), ("down", 2048, 8192)]
MLIST = [512, 2048, 4096, 8192]
RLIST = [0.0, 0.125, 0.5]
ITERS = 50
WARMUP = 15


def interleave(named):
    for _, fn in named:
        for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    samp = {n: [] for n, _ in named}
    for _ in range(ITERS):
        for n, fn in named:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            samp[n].append(s.elapsed_time(e))
    out = {}
    for n, sl in samp.items():
        sl.sort(); out[n] = sl[0] * 1000.0  # min us
    return out


def tflops(M, N, Kt, us):
    return 2.0 * M * N * Kt / (us * 1e-6) / 1e12 if us and us > 0 else float('nan')


CSV = os.path.join(HERE, "r24_fair.csv")
f = open(CSV, "w", newline="", buffering=1); w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "clk",
            "fp4_old_us", "fp4_best_us", "fp4_best_cfg", "fp4_improve",
            "fp8_old_us", "fp8_best_us", "fp8_best_cfg", "fp8_improve",
            "cd_best_us", "cd_best_cfg",
            "fp4_best_tflops", "fp4_best_pct", "fp8_best_tflops", "fp8_best_pct",
            "cd_best_tflops", "cd_best_pct_effpeak",
            "fp4_old_pct", "fp8_old_pct"])
allgate = True
print(f"{'shape':6}{'M':>6}{'r':>6}  {'fp4best':>9}{'fp4cfg':>13}{'impr':>6}  {'cdbest':>9}{'cdcfg':>14}", flush=True)

for (sname, N, KT) in SHAPES:
    for M in MLIST:
        # pure-format data (full KT) -- r-independent, prep once per (shape,M)
        p4, o4 = mg.prep_pure_fp4(M, N, KT, seed=1)
        p8, o8 = mg.prep_pure_fp8(M, N, KT, seed=1)
        phys4 = {t: NVCFG[t].prepack_sf_fp4_b(p4["SFB"], N, KT) for t in NVCFG}
        phys8 = {t: MXCFG[t].prepack_sf_fp8_b(p8["SFB"], N, KT) for t in MXCFG}
        # cosine gate pure
        for t in NVCFG:
            c = b1.cosine(NVCFG[t].nvfp4_gemm_presf(p4["A_packed"], p4["SFA"], p4["B_packed"], phys4[t], M, N, KT).float(), o4)
            if c < 0.999: allgate = False; print(f"  FP4 COSFAIL {sname} M{M} {t}: {c:.6f}", flush=True)
        for t in MXCFG:
            c = b1.cosine(MXCFG[t].mxfp8_gemm_presf(p8["A_bytes"], p8["SFA"], p8["B_bytes"], phys8[t], M, N, KT).float(), o8)
            if c < 0.999: allgate = False; print(f"  FP8 COSFAIL {sname} M{M} {t}: {c:.6f}", flush=True)

        for r in RLIST:
            Kh = 0 if r == 0.0 else int(round(r * KT)); Kl = KT - Kh
            # co-design data
            torch.manual_seed(1); dev = "cuda"
            Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
            Al_c, SFAl, Al_d = nv.quantize_nvfp4(Xl); Bl_c, SFBl, Bl_d = nv.quantize_nvfp4(Wl)
            Al_p = nv.pack_fp4(Al_c); Bl_p = nv.pack_fp4(Bl_c)
            s_x = (0.5 + torch.rand(M, device=dev)).float(); s_w = (0.5 + torch.rand(N, device=dev)).float()
            ocd = Al_d @ Bl_d.t()
            if Kh > 0:
                Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
                Ah_b, SFAh, Ah_d = mx.quantize_mxfp8(Xh); Bh_b, SFBh, Bh_d = mx.quantize_mxfp8(Wh)
                ocd = ocd + Ah_d @ Bh_d.t()
            ocd = ocd * s_x.view(M, 1) * s_w.view(1, N)

            named = []
            # pure endpoints (full KT, r-independent)
            for t in NVCFG:
                named.append(("fp4_" + t, (lambda tt: (lambda: NVCFG[tt].nvfp4_gemm_presf(p4["A_packed"], p4["SFA"], p4["B_packed"], phys4[tt], M, N, KT)))(t)))
            for t in MXCFG:
                named.append(("fp8_" + t, (lambda tt: (lambda: MXCFG[tt].mxfp8_gemm_presf(p8["A_bytes"], p8["SFA"], p8["B_bytes"], phys8[tt], M, N, KT)))(t)))
            # co-design
            cd_fns = {}
            for t in CDCFG:
                ext = CDCFG[t]
                if r == 0.0:
                    physB = ext.prepack_sf_fp4_b(SFBl, N, Kl)
                    D = ext.single_nvfp4_gemm(Al_p, SFAl, Bl_p, physB, M, N, Kl).float()
                    c = b1.cosine(D * s_x.view(M, 1) * s_w.view(1, N), ocd); del D
                    cd_fns[t] = (lambda e, pb: (lambda: e.single_nvfp4_gemm(Al_p, SFAl, Bl_p, pb, M, N, Kl)))(ext, physB)
                else:
                    pBlo = ext.prepack_sf_fp4_b(SFBl, N, Kl); pBhi = ext.prepack_sf_fp8_b(SFBh, N, Kh)
                    pAlo = ext.prepack_sf_fp4_a(SFAl, M, N, Kl); pAhi = ext.prepack_sf_fp8_a(SFAh, M, N, Kh)
                    D = ext.fused_gemm_prepacked(Al_p, pAlo, Bl_p, pBlo, Ah_b, pAhi, Bh_b, pBhi, M, N, Kl, Kh).float()
                    c = b1.cosine(D * s_x.view(M, 1) * s_w.view(1, N), ocd); del D
                    cd_fns[t] = (lambda e, a, b, c2, d: (lambda: e.fused_gemm_prepacked(Al_p, a, Bl_p, b, Ah_b, c2, Bh_b, d, M, N, Kl, Kh)))(ext, pAlo, pBlo, pAhi, pBhi)
                if c < 0.999: allgate = False; print(f"  CD COSFAIL {sname} M{M} r{r} {t}: {c:.6f}", flush=True)
                named.append(("cd_" + t, cd_fns[t]))

            res = interleave(named)
            # pick bests
            fp4b = min((t for t in NVCFG), key=lambda t: res["fp4_" + t]); fp4u = res["fp4_" + fp4b]
            fp8b = min((t for t in MXCFG), key=lambda t: res["fp8_" + t]); fp8u = res["fp8_" + fp8b]
            cdb = min((t for t in CDCFG), key=lambda t: res["cd_" + t]); cdu = res["cd_" + cdb]
            fp4_old = res["fp4_128x128_auto"]; fp8_old = res["fp8_128x128_auto"]
            ep = PK4 / (1.0 + r)
            f4tf = tflops(M, N, KT, fp4u); f8tf = tflops(M, N, KT, fp8u); cdtf = tflops(M, N, KT, cdu)
            row = [sname, M, N, KT, f"{r:.4f}", Kl, Kh, f"{CLK:.0f}",
                   f"{fp4_old:.3f}", f"{fp4u:.3f}", fp4b, f"{fp4_old/fp4u:.4f}",
                   f"{fp8_old:.3f}", f"{fp8u:.3f}", fp8b, f"{fp8_old/fp8u:.4f}",
                   f"{cdu:.3f}", cdb,
                   f"{f4tf:.1f}", f"{f4tf/PK4*100:.1f}", f"{f8tf:.1f}", f"{f8tf/PK8*100:.1f}",
                   f"{cdtf:.1f}", f"{cdtf/ep*100:.1f}",
                   f"{tflops(M,N,KT,fp4_old)/PK4*100:.1f}", f"{tflops(M,N,KT,fp8_old)/PK8*100:.1f}"]
            w.writerow(row)
            print(f"{sname:6}{M:>6}{r:>6.3f}  {fp4u:9.2f}{fp4b:>13}{fp4_old/fp4u:6.3f}  {cdu:9.2f}{cdb:>14}", flush=True)
            torch.cuda.empty_cache()
        del p4, o4, p8, o8; torch.cuda.empty_cache()
f.close()
print("R24_FAIR_ALL_COSINE_PASS" if allgate else "R24_FAIR_COSINE_FAILED", flush=True)
print("R24_FAIR_DONE", flush=True)
