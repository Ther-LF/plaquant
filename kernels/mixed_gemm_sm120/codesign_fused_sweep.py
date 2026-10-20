import os, torch, threading, time
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

# MUST match codesign_trace.py's load config exactly to hit the cached .so (no rebuild)
ext = load(name="codesign_fused_trace",
    sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
    extra_include_paths=[HERE, os.path.join(CUTLASS,"include"),
        os.path.join(CUTLASS,"tools/util/include"), os.path.join(CUTLASS,"examples/common")],
    extra_cuda_cflags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3","-DCD_TRACE"],
    verbose=False)
print("SWEEP_BUILD_OK", flush=True)

SHAPES = [("qkv", 2048, 1792, 256), ("down", 2048, 7168, 1024), ("gate", 8192, 1792, 256)]

allok = True
print("=== GATE A: single-phase NVFP4 (forked glue) vs pure-NVFP4 oracle ===", flush=True)
for name, N, Klo, Khi in SHAPES:
    K = Klo + Khi
    for M in (2048, 8192):
        p4, o4 = mg.prep_pure_fp4(M, N, K, seed=1)
        physB = ext.prepack_sf_fp4_b(p4["SFB"], N, K)
        D = ext.single_nvfp4_gemm(p4["A_packed"], p4["SFA"], p4["B_packed"], physB, M, N, K).float()
        c = b1.cosine(D, o4); tag = "PASS" if c>=0.999 else "FAIL"; allok &= c>=0.999
        print(f"  single-NVFP4 {name:4s} M{M:5d}: cos={c:.6f} [{tag}]", flush=True)
        del D, p4, o4, physB; torch.cuda.empty_cache()

print("=== GATE B: dual-phase fused vs (D_low+D_high) and full oracle ===", flush=True)
for name, N, Klo, Khi in SHAPES:
    for M in (2048, 8192):
        p, oracle = mg.prep_mixed(M, N, Klo, Khi, seed=M)
        D_low = nv.get_ext().nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"], M, N, Klo).float()
        D_high = mx.get_ext().mxfp8_gemm(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"], M, N, Khi).float()
        core_ref = D_low + D_high
        physB_lo = ext.prepack_sf_fp4_b(p["SFBl"], N, Klo)
        physB_hi = ext.prepack_sf_fp8_b(p["SFBh"], N, Khi)
        D = ext.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                           p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Klo, Khi).float()
        c_core = b1.cosine(D, core_ref)
        c_full = b1.cosine(D * p["s_x"].view(M,1) * p["s_w"].view(1,N), oracle)
        tag = "PASS" if (c_core>=0.999 and c_full>=0.999) else "FAIL"; allok &= (c_core>=0.999 and c_full>=0.999)
        print(f"  fused {name:4s} M{M:5d} Klo{Klo} Khi{Khi}: cos_core={c_core:.6f} cos_full={c_full:.6f} [{tag}]", flush=True)
        del D, D_low, D_high, core_ref, p, oracle, physB_lo, physB_hi; torch.cuda.empty_cache()

print("CDFUSED_SWEEP_ALL_PASS" if allok else "CDFUSED_SWEEP_SOME_FAILED", flush=True)
