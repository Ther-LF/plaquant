"""Round-25 DENSE-r FAIR best-config sweep (GEMM-only, run-only, reuse round-24 binaries).

For EACH (shape in {qkv,gate,down}, M in {512,2048,4096,8192}), sweep co-design GEMM
latency across the FULL dense r = {0, 1/16, ..., 16/16} (17 points) at that cell's BEST
config (round-24 dispatch):
  * qkv  : M<=512 -> 128x64 NV4/MX2 ; M>=2048 -> 128x128 NV4/MX2
  * gate : all M  -> 128x128 NV4/MX2
  * down : M<=512 -> 128x64 NV4/MX2 ; M>=2048 -> 128x128 NV4/MX2
Fixed per-shape×M endpoints (r-independent, measured once, reused across all 17 r):
  * FP4 endpoint = best over {128x128_auto, 128x128_pp, 256x128_s3, 128x64_auto} (pure NVFP4)
  * FP8 endpoint = best over {128x128_auto, 128x128_pp, 128x64_auto}             (pure MXFP8)
r=1 (Kl=0) has NO NVFP4 phase -> dual kernel cannot run -> recorded N/A; the FP8 endpoint
IS the r=1 value (noted in CSV cfg field).

K-split snap: Kt in {2048,8192}; r=k/16 -> Kh=k*(Kt/16) (mult of 32), Kl=Kt-Kh (mult of 64).
All 17 land exactly for both Kt (confirmed in-script; actual r reported).

cosine gated (>=0.999) on every measured config before timing. Clocks locked (idle GPU).
%-peak basis identical to round-23/24: eff_peak(r)=peak_nvfp4/(1+r), peak_nvfp4=188*clk*4096.
Reloads CACHED round-24 extensions (presf_cfg_*, cd_cfg24_*) -- no rebuild.
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


print("Reloading cached round-24 grid extensions (no rebuild expected)...", flush=True)
NVCFG = {
    "128x128_auto": load_presf("ps_128x128_auto", 128, 128, 128, 0, 0),
    "128x128_pp":   load_presf("ps_128x128_pp", 128, 128, 128, 0, 2),
    "256x128_s3":   load_presf("ps_256x128_s3", 256, 128, 128, 3, 0),
    "128x64_auto":  load_presf("ps_128x64_auto", 128, 64, 128, 0, 0),
}
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
RLIST = [i / 16.0 for i in range(17)]  # 0, 1/16, ..., 16/16
ITERS = 50
WARMUP = 15


def cd_best_cfg(sname, M):
    # round-24 co-design dispatch: small-M (<=512) qkv/down -> 128x64; everything else -> 128x128
    if sname in ("qkv", "down") and M <= 512:
        return "128x64_nv4mx2"
    return "128x128_nv4mx2"


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


CSV = os.path.join(HERE, "r25_fair_dense.csv")
f = open(CSV, "w", newline="", buffering=1); w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "clk",
            "cd_best_us", "cd_best_tflops", "cd_best_pct_peak", "cd_best_cfg",
            "fp4_best_tflops", "fp8_best_tflops"])
allgate = True
min_cos = 1.0
print(f"{'shape':6}{'M':>6}{'r':>7}{'Kl':>6}{'Kh':>6}  {'cd_us':>9}{'cd_tf':>8}{'%pk':>6}  {'cfg':>15}", flush=True)

for (sname, N, KT) in SHAPES:
    STEP = KT // 16
    for M in MLIST:
        # ---- fixed pure endpoints (full KT, r-independent, measured once) ----
        p4, o4 = mg.prep_pure_fp4(M, N, KT, seed=1)
        p8, o8 = mg.prep_pure_fp8(M, N, KT, seed=1)
        phys4 = {t: NVCFG[t].prepack_sf_fp4_b(p4["SFB"], N, KT) for t in NVCFG}
        phys8 = {t: MXCFG[t].prepack_sf_fp8_b(p8["SFB"], N, KT) for t in MXCFG}
        for t in NVCFG:
            c = b1.cosine(NVCFG[t].nvfp4_gemm_presf(p4["A_packed"], p4["SFA"], p4["B_packed"], phys4[t], M, N, KT).float(), o4)
            if c < 0.999: allgate = False; print(f"  FP4 COSFAIL {sname} M{M} {t}: {c:.6f}", flush=True)
        for t in MXCFG:
            c = b1.cosine(MXCFG[t].mxfp8_gemm_presf(p8["A_bytes"], p8["SFA"], p8["B_bytes"], phys8[t], M, N, KT).float(), o8)
            if c < 0.999: allgate = False; print(f"  FP8 COSFAIL {sname} M{M} {t}: {c:.6f}", flush=True)
        nv_named = [("fp4_" + t, (lambda tt: (lambda: NVCFG[tt].nvfp4_gemm_presf(p4["A_packed"], p4["SFA"], p4["B_packed"], phys4[tt], M, N, KT)))(t)) for t in NVCFG]
        mx_named = [("fp8_" + t, (lambda tt: (lambda: MXCFG[tt].mxfp8_gemm_presf(p8["A_bytes"], p8["SFA"], p8["B_bytes"], phys8[tt], M, N, KT)))(t)) for t in MXCFG]
        ep_res = interleave(nv_named + mx_named)
        fp4b = min(NVCFG, key=lambda t: ep_res["fp4_" + t]); fp4u = ep_res["fp4_" + fp4b]
        fp8b = min(MXCFG, key=lambda t: ep_res["fp8_" + t]); fp8u = ep_res["fp8_" + fp8b]
        fp4_tf = tflops(M, N, KT, fp4u); fp8_tf = tflops(M, N, KT, fp8u)
        print(f"  [endpoints] {sname} M{M}: FP4 {fp4b} {fp4_tf:.1f}TF  FP8 {fp8b} {fp8_tf:.1f}TF", flush=True)

        cdtag = cd_best_cfg(sname, M)
        cd_ep_fn = {
            "fp4": (lambda: NVCFG[fp4b].nvfp4_gemm_presf(p4["A_packed"], p4["SFA"], p4["B_packed"], phys4[fp4b], M, N, KT)),
            "fp8": (lambda: MXCFG[fp8b].mxfp8_gemm_presf(p8["A_bytes"], p8["SFA"], p8["B_bytes"], phys8[fp8b], M, N, KT)),
        }

        for r in RLIST:
            Kh = int(round(r * KT)); Kl = KT - Kh
            assert Kh % 32 == 0 and Kl % 64 == 0 and abs(r - Kh / KT) < 1e-9, f"snap fail {sname} r{r} Kh{Kh} Kl{Kl}"
            ract = Kh / KT  # actual snapped r
            if Kl == 0:
                # r=1 pure MXFP8: dual kernel has no NVFP4 phase -> N/A. FP8 endpoint IS the r=1 value.
                w.writerow([sname, M, N, KT, f"{ract:.4f}", Kl, Kh, f"{CLK:.0f}",
                            "NA", "NA", "NA", "NA(=FP8_endpoint)",
                            f"{fp4_tf:.1f}", f"{fp8_tf:.1f}"])
                print(f"{sname:6}{M:>6}{ract:>7.4f}{Kl:>6}{Kh:>6}  {'N/A':>9}{'--':>8}{'--':>6}  {'=FP8ep':>15}", flush=True)
                continue
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

            ext = CDCFG[cdtag]
            if Kh == 0:
                physB = ext.prepack_sf_fp4_b(SFBl, N, Kl)
                D = ext.single_nvfp4_gemm(Al_p, SFAl, Bl_p, physB, M, N, Kl).float()
                c = b1.cosine(D * s_x.view(M, 1) * s_w.view(1, N), ocd); del D
                cd_fn = (lambda e, pb: (lambda: e.single_nvfp4_gemm(Al_p, SFAl, Bl_p, pb, M, N, Kl)))(ext, physB)
            else:
                pBlo = ext.prepack_sf_fp4_b(SFBl, N, Kl); pBhi = ext.prepack_sf_fp8_b(SFBh, N, Kh)
                pAlo = ext.prepack_sf_fp4_a(SFAl, M, N, Kl); pAhi = ext.prepack_sf_fp8_a(SFAh, M, N, Kh)
                D = ext.fused_gemm_prepacked(Al_p, pAlo, Bl_p, pBlo, Ah_b, pAhi, Bh_b, pBhi, M, N, Kl, Kh).float()
                c = b1.cosine(D * s_x.view(M, 1) * s_w.view(1, N), ocd); del D
                cd_fn = (lambda e, a, b_, c2, d: (lambda: e.fused_gemm_prepacked(Al_p, a, Bl_p, b_, Ah_b, c2, Bh_b, d, M, N, Kl, Kh)))(ext, pAlo, pBlo, pAhi, pBhi)
            if c < 0.999: allgate = False; print(f"  CD COSFAIL {sname} M{M} r{ract:.4f} {cdtag}: {c:.6f}", flush=True)
            min_cos = min(min_cos, c)

            # interleave cd best with fixed endpoints (same clock/contention window)
            res = interleave([("cd", cd_fn), ("fp4", cd_ep_fn["fp4"]), ("fp8", cd_ep_fn["fp8"])])
            cdu = res["cd"]
            ep = PK4 / (1.0 + ract)
            cdtf = tflops(M, N, KT, cdu)
            w.writerow([sname, M, N, KT, f"{ract:.4f}", Kl, Kh, f"{CLK:.0f}",
                        f"{cdu:.3f}", f"{cdtf:.1f}", f"{cdtf/ep*100:.1f}", cdtag,
                        f"{fp4_tf:.1f}", f"{fp8_tf:.1f}"])
            print(f"{sname:6}{M:>6}{ract:>7.4f}{Kl:>6}{Kh:>6}  {cdu:9.2f}{cdtf:8.1f}{cdtf/ep*100:6.1f}  {cdtag:>15}", flush=True)
            torch.cuda.empty_cache()
        del p4, o4, p8, o8; torch.cuda.empty_cache()
f.close()
print(f"R25_MIN_COSINE={min_cos:.6f}", flush=True)
print("R25_FAIR_DENSE_ALL_COSINE_PASS" if allgate else "R25_FAIR_DENSE_COSINE_FAILED", flush=True)
print("R25_FAIR_DENSE_DONE", flush=True)
