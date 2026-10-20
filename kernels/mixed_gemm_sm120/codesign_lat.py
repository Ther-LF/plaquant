"""Latency A/B: co-design dual-format fused GEMM vs the hand-rolled mixed kernel.

Per shape x M: time codesign fused_gemm (shared-accumulator dual collective) vs
mixed_gemm_fused_presf (the existing hand-rolled single-warpgroup mixed consumer,
alu/tc~3.5, full s_x*s_w epilogue) and mixed_gemm_fused_b_presf. Same data, same GPU.
Honest caveat: the co-design currently uses a PLAIN epilogue (no outer s_x*s_w);
the reference applies them. The GEMM mainloop dominates; delta noted.
"""
import os, torch
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

ext = load(name="codesign_fused_ext",
    sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
    extra_include_paths=[HERE, os.path.join(CUTLASS,"include"),
        os.path.join(CUTLASS,"tools/util/include"), os.path.join(CUTLASS,"examples/common")],
    extra_cuda_cflags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3"],
    verbose=False)
print("LAT_BUILD_OK", flush=True)
mg.get_fused_ext()  # warm hand-rolled mixed kernel ext

SHAPES = [("qkv", 2048, 1792, 256), ("down", 2048, 7168, 1024), ("gate", 8192, 1792, 256)]

def med(fn):
    m,_ = mg._time_ms_median(fn, iters=50, warmup=10); return m

print(f"{'shape':12s} {'M':>6s} {'codesign_ms':>12s} {'handroll_A_ms':>14s} {'handroll_B_ms':>14s} {'cd/A':>7s}", flush=True)
for name, N, Klo, Khi in SHAPES:
    for M in (2048, 8192):
        p, _ = mg.prep_mixed(M, N, Klo, Khi, seed=1)
        mg.prepack_weight_sf(p, N, Klo, Khi)                 # hand-roll weight SF (once)
        physB_lo = ext.prepack_sf_fp4_b(p["SFBl"], N, Klo)   # codesign weight SF (once)
        physB_hi = ext.prepack_sf_fp8_b(p["SFBh"], N, Khi)
        cd = lambda: ext.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                                    p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Klo, Khi)
        hrA = lambda: mg.mixed_gemm_fused_presf(p, M, N, Klo, Khi)
        hrB = lambda: mg.mixed_gemm_fused_b_presf(p, M, N, Klo, Khi)
        t_cd = med(cd); t_a = med(hrA); t_b = med(hrB)
        print(f"{name:12s} {M:6d} {t_cd:12.4f} {t_a:14.4f} {t_b:14.4f} {t_cd/t_a:7.3f}", flush=True)
        del p, physB_lo, physB_hi; torch.cuda.empty_cache()
print("LAT_DONE", flush=True)
