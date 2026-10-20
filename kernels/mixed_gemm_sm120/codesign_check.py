"""STAGE-1 co-design NVFP4 kernel (b1-opt round-13): loader + cosine gate + compat report.

Builds codesign_nvfp4_ext.cu (stock SM120 block-scaled collective, our packed
A/B/SF data contract) and validates cosine=1.0 vs the mixed oracle's NVFP4 path,
on the three b1 shapes x M{2048,8192}. Also prints the Path-A shared-accumulator
compile-time compatibility report (NVFP4 vs MXFP8 accumulator-fragment identity).

Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python codesign_check.py
"""
import os
import torch
from torch.utils.cpp_extension import load

import mixed_gemm as mg
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="codesign_nvfp4_ext",
            sources=[os.path.join(HERE, "codesign_nvfp4_ext.cu")],
            extra_include_paths=[
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


# three b1 shapes (N, K_total); pure NVFP4 => full K
SHAPES = [("qkv", 2048, 2048), ("down", 2048, 8192), ("gate", 8192, 2048)]


def main():
    ext = get_ext()
    print("BUILD_OK", flush=True)
    # Path-A shared-accumulator compile-time compatibility report
    print("=== Path-A shared-accumulator compat (compile-time) ===", flush=True)
    print(ext.compat_report(), flush=True)

    print("=== cosine gate: codesign NVFP4 vs oracle (our data contract) ===", flush=True)
    allok = True
    for name, N, K in SHAPES:
        for M in (2048, 8192):
            p4, o4 = mg.prep_pure_fp4(M, N, K, seed=1)
            physB = ext.prepack_sf_fp4_b(p4["SFB"], N, K)
            D = ext.codesign_nvfp4_gemm(
                p4["A_packed"], p4["SFA"], p4["B_packed"], physB, M, N, K).float()
            c = b1.cosine(D, o4)
            rel = ((D - o4).norm() / (o4.norm() + 1e-12)).item()
            tag = "PASS" if c >= 0.999 else "FAIL"
            if c < 0.999:
                allok = False
            print(f"  codesign-NVFP4 {name:4s} M{M:5d} N{N} K{K}: "
                  f"cos={c:.6f} rel={rel:.2e} [{tag}]", flush=True)
            del D, p4, o4, physB
            torch.cuda.empty_cache()
    print("CODESIGN_GATE_ALL_PASS" if allok else "CODESIGN_GATE_SOME_FAILED", flush=True)


if __name__ == "__main__":
    main()
