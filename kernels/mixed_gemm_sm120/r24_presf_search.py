"""Round-24 PURE-FORMAT (NVFP4 / MXFP8) config GRID SEARCH driver (detached).

Builds presf_cfg_ext.cu once per config (TileShape / ClusterShape / stages / schedule)
as a separate named torch extension via -D macros, gates cosine>=0.999, and measures the
GEMM-only latency of the pure single-format full-K block-scaled GEMM (weight SF
pre-interleaved, activation SF filled per call -- identical methodology to the PROTECTED
pure_presf_ext.cu). Pure formats are r-INDEPENDENT (full-K single format) so we sweep
only shape x M for each config, both formats.

Goal: find the BEST config per (format, shape, M) -> FAIR FP4/FP8 endpoints, vs the
under-tuned hardcoded 128x128x128 auto-carveout of pure_presf_ext.cu.

Compile / can_implement / smem-overflow / static_assert failures are CAUGHT and recorded
-- NO silent truncation. Each config = one independent compile; box overloaded -> DETACHED,
line-buffered CSV, harvest from disk.
"""
import os, csv, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import mixed_gemm as mg
import b1_fused as b1

# (label, TM,TN,TK, CM,CN,CK, STAGES, SCHED)  STAGES=0 => auto-carveout; SCHED 0 auto/1 coop/2 pingpong
CONFIGS = [
    ("ps_128x128_auto",    128, 128, 128, 1, 1, 1, 0, 0),   # == pure_presf baseline
    ("ps_256x128_auto",    256, 128, 128, 1, 1, 1, 0, 0),
    ("ps_128x256_auto",    128, 256, 128, 1, 1, 1, 0, 0),
    ("ps_256x256_auto",    256, 256, 128, 1, 1, 1, 0, 0),
    ("ps_128x64_auto",     128,  64, 128, 1, 1, 1, 0, 0),
    ("ps_64x128_auto",      64, 128, 128, 1, 1, 1, 0, 0),
    ("ps_64x64_auto",       64,  64, 128, 1, 1, 1, 0, 0),
    ("ps_128x128k256_auto",128, 128, 256, 1, 1, 1, 0, 0),
    ("ps_128x128_pp",      128, 128, 128, 1, 1, 1, 0, 2),   # pingpong schedule
    ("ps_256x128_s3",      256, 128, 128, 1, 1, 1, 3, 0),   # explicit stages on wide tile
    ("ps_128x256_s3",      128, 256, 128, 1, 1, 1, 3, 0),
]

SHAPES = [("qkv", 2048, 2048), ("gate", 8192, 2048), ("down", 2048, 8192)]
MLIST = [512, 2048, 4096, 8192]
ITERS = 40
WARMUP = 12


def med_min(fn):
    m, samples = mg._time_ms_median(fn, iters=ITERS, warmup=WARMUP)
    return m * 1000.0, min(samples) * 1000.0


CSV = os.path.join(HERE, "r24_presf_search.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["cfg", "TM", "TN", "TK", "stages_macro", "sched",
            "fmt", "shape", "M", "N", "K", "build",
            "nv_stages", "nv_smem", "mx_stages", "mx_smem",
            "cos", "med_us", "min_us"])
print("R24_PRESF_SEARCH_START", flush=True)
allgate = True

for (lbl, TM, TN, TK, CM, CN, CK, STG, SCH) in CONFIGS:
    t0 = time.time()
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3",
             f"-DPSCFG_TM={TM}", f"-DPSCFG_TN={TN}", f"-DPSCFG_TK={TK}",
             f"-DPSCFG_CM={CM}", f"-DPSCFG_CN={CN}", f"-DPSCFG_CK={CK}",
             f"-DPSCFG_STAGES={STG}", f"-DPSCFG_SCHED={SCH}"]
    ext = None; build = "OK"
    nv_st = nv_sm = mx_st = mx_sm = -1
    try:
        ext = load(name="presf_cfg_" + lbl, sources=[os.path.join(HERE, "presf_cfg_ext.cu")],
                   extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                                        os.path.join(CUTLASS, "tools/util/include"),
                                        os.path.join(CUTLASS, "examples/common")],
                   extra_cuda_cflags=flags, verbose=False)
        rep = ext.compat_report()
        for line in rep.splitlines():
            ls = line.strip()
            if ls.startswith("NV:"):
                nv_st = int(ls.split("stages=")[1].split()[0]); nv_sm = int(ls.split("mainloop_smem=")[1].split()[0])
            if ls.startswith("MX:"):
                mx_st = int(ls.split("stages=")[1].split()[0]); mx_sm = int(ls.split("mainloop_smem=")[1].split()[0])
        print(f"[{lbl}] BUILD_OK {time.time()-t0:.0f}s NVsmem={nv_sm}(st{nv_st}) MXsmem={mx_sm}(st{mx_st})", flush=True)
    except Exception as e:
        build = "BUILD_FAIL: " + str(e).splitlines()[-1][:120]
        print(f"[{lbl}] BUILD_FAIL {time.time()-t0:.0f}s :: {str(e).splitlines()[-1][:160]}", flush=True)

    for (sname, N, K) in SHAPES:
        for M in MLIST:
            for fmt in ("fp4", "fp8"):
                cos = float('nan'); med = mn = float('nan'); st = build
                if ext is not None:
                    try:
                        if fmt == "fp4":
                            p, oracle = mg.prep_pure_fp4(M, N, K, seed=1)
                            physB = ext.prepack_sf_fp4_b(p["SFB"], N, K)
                            D = ext.nvfp4_gemm_presf(p["A_packed"], p["SFA"], p["B_packed"], physB, M, N, K).float()
                            cos = b1.cosine(D, oracle); del D
                            med, mn = med_min(lambda: ext.nvfp4_gemm_presf(p["A_packed"], p["SFA"], p["B_packed"], physB, M, N, K))
                        else:
                            p, oracle = mg.prep_pure_fp8(M, N, K, seed=1)
                            physB = ext.prepack_sf_fp8_b(p["SFB"], N, K)
                            D = ext.mxfp8_gemm_presf(p["A_bytes"], p["SFA"], p["B_bytes"], physB, M, N, K).float()
                            cos = b1.cosine(D, oracle); del D
                            med, mn = med_min(lambda: ext.mxfp8_gemm_presf(p["A_bytes"], p["SFA"], p["B_bytes"], physB, M, N, K))
                        st = "OK" if (cos == cos and cos >= 0.999) else f"COSFAIL {cos:.5f}"
                        if cos == cos and cos < 0.999:
                            allgate = False
                        del p, oracle; torch.cuda.empty_cache()
                    except Exception as e:
                        st = "RUN_FAIL: " + str(e).splitlines()[-1][:90]
                w.writerow([lbl, TM, TN, TK, STG, SCH, fmt, sname, M, N, K, st,
                            nv_st, nv_sm, mx_st, mx_sm,
                            f"{cos:.6f}", f"{med:.3f}", f"{mn:.3f}"])
                print(f"  {lbl:20s}{fmt:4s}{sname:5s}M{M:<5d}{st[:14]:14s}cos={cos:.6f} min={mn:8.2f}", flush=True)
    del ext; torch.cuda.empty_cache()
    import gc; gc.collect()

f.close()
print("R24_PRESF_ALL_COSINE_PASS" if allgate else "R24_PRESF_COSINE_FAILED", flush=True)
print("R24_PRESF_SEARCH_DONE", flush=True)
