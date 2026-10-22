"""Round-24 CO-DESIGN config GRID SEARCH over the FULL M grid (detached).

Reuses cd_cfg_ext.cu (macro-parameterized config-only copy of the PROTECTED co-design
mainloop headers; zero logic change). Round-19 already proved the SM120-viable co-design
config space is tiny (smem cap ~101376B + cooperative Tile-M>=128 static_assert leave only
128x128). Round-24 EXTENDS that verdict to the FULL M grid {512,2048,4096,8192} and adds an
r-sensitivity sweep r{0,1/8,1/2} on every cell -- so the dispatcher's co-design entries are
backed by measured best-config per shape x M (not just M{2048,8192} from round-19).

ALL tiles enumerated (fits / smem-overflow / static_assert-fail recorded; NO silent skips).
A config that fails can_implement/initialize on its first cell is marked dead and its
remaining cells are skipped (status recorded) to save build-box time.
"""
import os, csv, sys, time, torch
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

# (label, TM,TN,TK, CM,CN,CK, NVST,MXST)  all cooperative; enumerate full tile set.
CONFIGS = [
    ("t128x128_nv4mx2", 128, 128, 128, 1, 1, 1, 4, 2),   # shipping/baseline
    ("t128x128_nv3mx2", 128, 128, 128, 1, 1, 1, 3, 2),
    ("t128x64_nv4mx2",  128,  64, 128, 1, 1, 1, 4, 2),
    ("t128x256_nv4mx2", 128, 256, 128, 1, 1, 1, 4, 2),   # expect smem-overflow
    ("t256x128_nv4mx2", 256, 128, 128, 1, 1, 1, 4, 2),   # expect smem-overflow
    ("t128x128_nv5mx3", 128, 128, 128, 1, 1, 1, 5, 3),   # expect smem-overflow
    ("t64x128_nv4mx2",   64, 128, 128, 1, 1, 1, 4, 2),   # expect static_assert Tile-M>=128
    ("t64x64_nv4mx2",    64,  64, 128, 1, 1, 1, 4, 2),   # expect static_assert Tile-M>=128
]

SHAPES = [("qkv", 2048, 2048), ("gate", 8192, 2048), ("down", 2048, 8192)]
MLIST = [512, 2048, 4096, 8192]
RLIST = [0.0, 0.125, 0.5]
ITERS = 40
WARMUP = 12


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


def med_min(fn):
    m, samples = mg._time_ms_median(fn, iters=ITERS, warmup=WARMUP)
    return m * 1000.0, min(samples) * 1000.0


CSV = os.path.join(HERE, "r24_cd_search.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["cfg", "TM", "TN", "TK", "NVST", "MXST", "smem_dual",
            "shape", "M", "N", "Kt", "r", "Kl", "Kh", "build", "cos", "med_us", "min_us"])
print("R24_CD_SEARCH_START", flush=True)
allgate = True

for (lbl, TM, TN, TK, CM, CN, CK, NVST, MXST) in CONFIGS:
    t0 = time.time()
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3",
             f"-DCDCFG_TM={TM}", f"-DCDCFG_TN={TN}", f"-DCDCFG_TK={TK}",
             f"-DCDCFG_CM={CM}", f"-DCDCFG_CN={CN}", f"-DCDCFG_CK={CK}",
             f"-DCDCFG_NVST={NVST}", f"-DCDCFG_MXST={MXST}"]
    ext = None; build = "OK"; smem = -1
    try:
        ext = load(name="cd_cfg24_" + lbl, sources=[os.path.join(HERE, "cd_cfg_ext.cu")],
                   extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                                        os.path.join(CUTLASS, "tools/util/include"),
                                        os.path.join(CUTLASS, "examples/common")],
                   extra_cuda_cflags=flags, verbose=False)
        rep = ext.compat_report()
        for line in rep.splitlines():
            if "SharedStorageSize(dual)" in line:
                smem = int(line.split("=")[1].strip().split()[0])
        print(f"[{lbl}] BUILD_OK {time.time()-t0:.0f}s smem={smem}", flush=True)
    except Exception as e:
        build = "BUILD_FAIL(static_assert?): " + str(e).splitlines()[-1][:100]
        print(f"[{lbl}] BUILD_FAIL {time.time()-t0:.0f}s :: {str(e).splitlines()[-1][:160]}", flush=True)

    dead = (ext is None)
    for (sname, N, KT) in SHAPES:
        for M in MLIST:
            for r in RLIST:
                Kh = 0 if r == 0.0 else int(round(r * KT))
                Kl = KT - Kh
                cos = float('nan'); med = mn = float('nan'); st = build
                if ext is not None and not dead:
                    try:
                        d, oracle = prep(M, N, Kl, Kh)
                        if r == 0.0:
                            physB = ext.prepack_sf_fp4_b(d["SFBl"], N, Kl)
                            D = ext.single_nvfp4_gemm(d["Al_packed"], d["SFAl"], d["Bl_packed"], physB, M, N, Kl).float()
                            cos = b1.cosine(D * d["s_x"].view(M,1) * d["s_w"].view(1,N), oracle); del D
                            med, mn = med_min(lambda: ext.single_nvfp4_gemm(d["Al_packed"], d["SFAl"], d["Bl_packed"], physB, M, N, Kl))
                        else:
                            pBlo = ext.prepack_sf_fp4_b(d["SFBl"], N, Kl)
                            pBhi = ext.prepack_sf_fp8_b(d["SFBh"], N, Kh)
                            pAlo = ext.prepack_sf_fp4_a(d["SFAl"], M, N, Kl)
                            pAhi = ext.prepack_sf_fp8_a(d["SFAh"], M, N, Kh)
                            D = ext.fused_gemm_prepacked(d["Al_packed"], pAlo, d["Bl_packed"], pBlo,
                                                         d["Ah_bytes"], pAhi, d["Bh_bytes"], pBhi,
                                                         M, N, Kl, Kh).float()
                            cos = b1.cosine(D * d["s_x"].view(M,1) * d["s_w"].view(1,N), oracle); del D
                            med, mn = med_min(lambda: ext.fused_gemm_prepacked(d["Al_packed"], pAlo, d["Bl_packed"], pBlo,
                                                                               d["Ah_bytes"], pAhi, d["Bh_bytes"], pBhi,
                                                                               M, N, Kl, Kh))
                        st = "OK" if (cos == cos and cos >= 0.999) else f"COSFAIL {cos:.5f}"
                        if cos == cos and cos < 0.999:
                            allgate = False
                        del d, oracle; torch.cuda.empty_cache()
                    except Exception as e:
                        st = "RUN_FAIL(smem-overflow?): " + str(e).splitlines()[-1][:80]
                        print(f"[{lbl}] RUN_FAIL {sname} M{M} r{r}: {str(e).splitlines()[-1][:120]}", flush=True)
                        dead = True  # smem-overflow configs fail every cell; record once, skip rest
                w.writerow([lbl, TM, TN, TK, NVST, MXST, smem, sname, M, N, KT,
                            f"{r:.4f}", Kl, Kh, st, f"{cos:.6f}", f"{med:.3f}", f"{mn:.3f}"])
                print(f"  {lbl:18s}{sname:5s}M{M:<5d}r{r:<6.3f}{st[:18]:18s}cos={cos:.6f} min={mn:8.2f}", flush=True)
    del ext; torch.cuda.empty_cache()
    import gc; gc.collect()

f.close()
print("R24_CD_ALL_COSINE_PASS" if allgate else "R24_CD_COSINE_FAILED", flush=True)
print("R24_CD_SEARCH_DONE", flush=True)
