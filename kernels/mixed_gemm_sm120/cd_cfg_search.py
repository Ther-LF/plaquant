"""Round-19 CO-DESIGN GRID CONFIG SEARCH driver (detached).

Builds cd_cfg_ext.cu once per config (TileShape / ClusterShape / stages) as a separate
named torch extension via -D macros, gates cosine=1.0, and measures the GEMM-only
latency (cd_gemmonly = fused_gemm_prepacked, the config discriminator; pack is
config-independent). Representative cells only: 3 shapes x M{2048,8192} x r{1/8,1/2}
(+ r=0 single-NVFP4 path), per the lead's "config optimum varies by shape x M, not
fine r". Each config = one independent compile; machine overloaded -> run DETACHED,
line-buffered CSV, harvest from disk. Compile/can_implement failures (smem over the
~101KB SM120 budget, or cooperative Tile-M>=128 static_assert) are caught and recorded
-- NO silent truncation.

NO mainloop logic change: cd_cfg_ext.cu #includes the SAME protected headers; only the
CollectiveBuilder config template params vary. Cooperative kept (shared accumulator).
"""
import os, csv, sys, time, traceback, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1
import mixed_gemm as mg

nv.get_ext(); mx.get_ext(); b1.get_presf_ext()

# (label, TM,TN,TK, CM,CN,CK, NVST,MXST)   -- all cooperative; Tile-M must be >=128.
CONFIGS = [
    ("t128x128_nv4mx2", 128,128,128, 1,1,1, 4,2),   # baseline (shipping) same-session anchor
    ("t128x256_nv4mx2", 128,256,128, 1,1,1, 4,2),
    ("t256x128_nv4mx2", 256,128,128, 1,1,1, 4,2),
    ("t128x64_nv4mx2",  128,64,128,  1,1,1, 4,2),
    ("t128x128_nv5mx3", 128,128,128, 1,1,1, 5,3),
    ("t128x128_nv3mx2", 128,128,128, 1,1,1, 3,2),
    ("t128x256_nv3mx2", 128,256,128, 1,1,1, 3,2),
    ("t256x128_nv3mx2", 256,128,128, 1,1,1, 3,2),
    ("t128x128_nv6mx4", 128,128,128, 1,1,1, 6,4),
    ("t128x64_nv6mx4",  128,64,128,  1,1,1, 6,4),
]

# (name, N, KT)
SHAPES = [("qkv", 2048, 2048), ("down", 2048, 8192), ("gate", 8192, 2048)]
MLIST = [2048, 8192]
RLIST = [0.0, 0.125, 0.5]
ITERS = 30
WARMUP = 10


def prep(M, N, Kl, Kh, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
    Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    s_x = (0.5 + torch.rand(M, device=dev)).float()
    s_w = (0.5 + torch.rand(N, device=dev)).float()
    d = dict(Al_packed=Al_packed, SFAl=SFAl, Bl_packed=Bl_packed, SFBl=SFBl, s_x=s_x, s_w=s_w)
    if Kh > 0:
        Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
        Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
        Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
        oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t())
        d.update(Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh)
    else:
        oracle = Al_deq @ Bl_deq.t()
    oracle = oracle * s_x.view(M, 1) * s_w.view(1, N)
    return d, oracle


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=ITERS, warmup=WARMUP)
    return m * 1000.0


CSV = os.path.join(HERE, "cd_cfg_search.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["cfg", "TM", "TN", "TK", "NVST", "MXST", "smem_dual",
            "shape", "M", "N", "Kt", "r", "Kl", "Kh", "build", "cos", "cd_gemmonly_us"])
print("CFG_SEARCH_START", flush=True)

for (lbl, TM, TN, TK, CM, CN, CK, NVST, MXST) in CONFIGS:
    t0 = time.time()
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3",
             f"-DCDCFG_TM={TM}", f"-DCDCFG_TN={TN}", f"-DCDCFG_TK={TK}",
             f"-DCDCFG_CM={CM}", f"-DCDCFG_CN={CN}", f"-DCDCFG_CK={CK}",
             f"-DCDCFG_NVST={NVST}", f"-DCDCFG_MXST={MXST}"]
    ext = None; build = "OK"; smem = -1
    try:
        ext = load(name="cd_cfg_" + lbl, sources=[os.path.join(HERE, "cd_cfg_ext.cu")],
                   extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                                        os.path.join(CUTLASS, "tools/util/include"),
                                        os.path.join(CUTLASS, "examples/common")],
                   extra_cuda_cflags=flags, verbose=False)
        rep = ext.compat_report()
        for line in rep.splitlines():
            if "SharedStorageSize(dual)" in line:
                smem = int(line.split("=")[1].strip().split()[0])
        print(f"[{lbl}] BUILD_OK {time.time()-t0:.0f}s smem={smem}\n{rep}", flush=True)
    except Exception as e:
        build = "BUILD_FAIL: " + str(e).splitlines()[-1][:120]
        print(f"[{lbl}] BUILD_FAIL {time.time()-t0:.0f}s :: {str(e).splitlines()[-1][:160]}", flush=True)

    for (sname, N, KT) in SHAPES:
        for M in MLIST:
            for r in RLIST:
                Kh = 0 if r == 0.0 else int(round(r * KT))
                Kl = KT - Kh
                cos = float('nan'); lat = float('nan'); st = build
                if ext is not None:
                    try:
                        d, oracle = prep(M, N, Kl, Kh)
                        if r == 0.0:
                            physB = ext.prepack_sf_fp4_b(d["SFBl"], N, Kl)
                            D = ext.single_nvfp4_gemm(d["Al_packed"], d["SFAl"], d["Bl_packed"], physB, M, N, Kl).float()
                            cos = b1.cosine(D * d["s_x"].view(M,1) * d["s_w"].view(1,N), oracle); del D
                            lat = med(lambda: ext.single_nvfp4_gemm(d["Al_packed"], d["SFAl"], d["Bl_packed"], physB, M, N, Kl))
                        else:
                            pBlo = ext.prepack_sf_fp4_b(d["SFBl"], N, Kl)
                            pBhi = ext.prepack_sf_fp8_b(d["SFBh"], N, Kh)
                            pAlo = ext.prepack_sf_fp4_a(d["SFAl"], M, N, Kl)
                            pAhi = ext.prepack_sf_fp8_a(d["SFAh"], M, N, Kh)
                            D = ext.fused_gemm_prepacked(d["Al_packed"], pAlo, d["Bl_packed"], pBlo,
                                                         d["Ah_bytes"], pAhi, d["Bh_bytes"], pBhi,
                                                         M, N, Kl, Kh).float()
                            cos = b1.cosine(D * d["s_x"].view(M,1) * d["s_w"].view(1,N), oracle); del D
                            lat = med(lambda: ext.fused_gemm_prepacked(d["Al_packed"], pAlo, d["Bl_packed"], pBlo,
                                                                       d["Ah_bytes"], pAhi, d["Bh_bytes"], pBhi,
                                                                       M, N, Kl, Kh))
                        st = "OK" if (cos == cos and cos >= 0.999) else f"COSFAIL {cos:.5f}"
                        del d, oracle; torch.cuda.empty_cache()
                    except Exception as e:
                        st = "RUN_FAIL: " + str(e).splitlines()[-1][:100]
                        print(f"[{lbl}] RUN_FAIL {sname} M{M} r{r}: {str(e).splitlines()[-1][:120]}", flush=True)
                w.writerow([lbl, TM, TN, TK, NVST, MXST, smem, sname, M, N, KT,
                            f"{r:.4f}", Kl, Kh, st, f"{cos:.6f}", f"{lat:.4f}"])
                print(f"  {lbl:18s}{sname:5s}M{M:<5d}r{r:<6.3f}{st[:14]:14s}cos={cos:.6f} lat={lat:8.2f}", flush=True)
    del ext; torch.cuda.empty_cache()
    import gc; gc.collect()

f.close()
print("CFG_SEARCH_DONE", flush=True)
