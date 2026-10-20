"""STAGE-2 dual-format shared-accumulator co-design kernel: build + cosine gate.

Gate (hard, cosine=1.0):
  (A) single-phase NVFP4 through the FORKED glue  vs  pure-NVFP4 oracle
      -> proves the glue FORK reproduces STAGE-1 before MXFP8 is stitched.
  (B) dual-phase fused NVFP4(K_low)+MXFP8(K_high)  vs  D_low + D_high
      (the two proven round-1 single-format kernels summed = the exact
       shared-accumulator result). Also validates the full prep_mixed oracle
       by applying the separable outer scales s_x*s_w in Python.

Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python codesign_fused_check.py
"""
import os
import torch
from torch.utils.cpp_extension import load

import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="codesign_fused_ext",
            sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
            extra_include_paths=[
                HERE,
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
                os.path.join(CUTLASS, "examples/common"),
            ],
            extra_cuda_cflags=[
                "-std=c++17", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "-O3",
            ],
            verbose=True,
        )
    return _ext


# three b1 shapes: (name, N, K_low, K_high) at high_fraction = 1/8
SHAPES = [("qkv", 2048, 1792, 256), ("down", 2048, 7168, 1024), ("gate", 8192, 1792, 256)]


def gate_single(ext):
    print("=== GATE A: single-phase NVFP4 through FORKED glue vs pure-NVFP4 oracle ===", flush=True)
    allok = True
    for name, N, Klo, Khi in SHAPES:
        K = Klo + Khi  # full K for the pure test
        for M in (2048, 8192):
            p4, o4 = mg.prep_pure_fp4(M, N, K, seed=1)
            physB = ext.prepack_sf_fp4_b(p4["SFB"], N, K)
            D = ext.single_nvfp4_gemm(p4["A_packed"], p4["SFA"], p4["B_packed"], physB, M, N, K).float()
            c = b1.cosine(D, o4)
            tag = "PASS" if c >= 0.999 else "FAIL"
            if c < 0.999:
                allok = False
            print(f"  single-NVFP4 {name:4s} M{M:5d} N{N} K{K}: cos={c:.6f} [{tag}]", flush=True)
            del D, p4, o4, physB
            torch.cuda.empty_cache()
    return allok


def gate_dual(ext):
    print("=== GATE B: dual-phase fused vs (D_low + D_high) and full oracle ===", flush=True)
    allok = True
    for name, N, Klo, Khi in SHAPES:
        for M in (2048, 8192):
            p, oracle = mg.prep_mixed(M, N, Klo, Khi, seed=M)
            # proven round-1 single-format cores
            D_low = nv.get_ext().nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"], M, N, Klo).float()
            D_high = mx.get_ext().mxfp8_gemm(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"], M, N, Khi).float()
            core_ref = D_low + D_high
            # my fused kernel
            physB_lo = ext.prepack_sf_fp4_b(p["SFBl"], N, Klo)
            physB_hi = ext.prepack_sf_fp8_b(p["SFBh"], N, Khi)
            D = ext.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                               p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi,
                               M, N, Klo, Khi).float()
            c_core = b1.cosine(D, core_ref)
            # full oracle via separable outer scales
            D_scaled = (D * p["s_x"].view(M, 1) * p["s_w"].view(1, N))
            c_full = b1.cosine(D_scaled, oracle)
            tag = "PASS" if (c_core >= 0.999 and c_full >= 0.999) else "FAIL"
            if c_core < 0.999 or c_full < 0.999:
                allok = False
            print(f"  fused {name:4s} M{M:5d} N{N} Klo{Klo} Khi{Khi}: "
                  f"cos_core={c_core:.6f} cos_full={c_full:.6f} [{tag}]", flush=True)
            del D, D_low, D_high, core_ref, p, oracle, physB_lo, physB_hi, D_scaled
            torch.cuda.empty_cache()
    return allok


def main():
    ext = get_ext()
    print("BUILD_OK", flush=True)
    print("=== compat/build report ===", flush=True)
    print(ext.compat_report(), flush=True)
    a = gate_single(ext)
    b = gate_dual(ext)
    print("CODESIGN_FUSED_GATE_ALL_PASS" if (a and b) else "CODESIGN_FUSED_GATE_SOME_FAILED", flush=True)


if __name__ == "__main__":
    main()
