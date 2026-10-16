"""B1 Phase-2 (MXFP8) TMA kernel — validator + benchmark.
Compares the TMA kernel to the cp.async WS kernel (b1_fused, K_low=0 pure phase-2)
and the pure-MXFP8 presf bound at the ncu-flagged down-class r=1 case.
Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python b1_tma.py
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load

import mixed_gemm as mg
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_tma = None
def get_tma():
    global _tma
    if _tma is None:
        _tma = load(
            name="b1_tma_ext",
            sources=[os.path.join(HERE, "b1_tma_ext.cu")],
            extra_include_paths=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
                os.path.join(CUTLASS, "examples/common"),
            ],
            extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                               "--expt-extended-lambda", "-O3", "-Xptxas", "-v"],
            verbose=True,
        )
    return _tma


def tma_mx(p, M, N, K_high, stages=0):
    return get_tma().b1_fused_tma_mx(
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, K_high, stages)


def median_ms(fn):
    med, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return med


def main():
    ext = get_tma()
    b1.get_ext()
    b1.get_presf_ext()
    print("Built.\n")

    # pure MXFP8 (K_low=0) cases; include the ncu baseline down r=1.
    cases = [
        ("down_r1", 2048, 2048, 8192),
        ("qkv_r1",  2048, 2048, 2048),
        ("gate_r1", 2048, 8192, 2048),
        ("down_r1_M256", 256, 2048, 8192),
        ("down_r1_M16",  16, 2048, 8192),
    ]
    print("=== cosine gate (TMA phase-2 vs mixed oracle, K_low=0) ===")
    ok = True
    for name, M, N, K in cases:
        p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
        for st in [3, 4, 5]:
            D = tma_mx(p, M, N, K, stages=st).float()
            cos = b1.cosine(D, oracle)
            rel = ((D - oracle).norm() / (oracle.norm() + 1e-12)).item()
            tag = "PASS" if cos >= 0.999 else "FAIL"
            if cos < 0.999:
                ok = False
            print(f"  {name:14s} M={M:5d} N={N:5d} K={K:5d} st={st}: cos={cos:.6f} rel={rel:.2e} [{tag}]")
    print("TMA COSINE ALL PASS\n" if ok else "TMA COSINE SOME FAILED\n")
    if not ok:
        print("ABORT benchmark (correctness gate failed)")
        return

    print("=== benchmark: TMA vs cp.async WS vs pure-fp8 bound (excess over bound) ===")
    for name, M, N, K in cases:
        p, oracle = mg.prep_mixed(M, N, 0, K, seed=1)
        p4, p8 = b1.prep_bounds_presf(M, N, K, seed=1)
        t8 = median_ms(lambda: b1.pure_fp8_presf(p8, M, N, K))
        # cp.async WS (existing b1 auto) for the pure phase-2 case
        lat_cp = median_ms(lambda: b1.b1_single(p, M, N, 0, K))
        best_tma, best_st = 1e9, -1
        for st in [3, 4, 5]:
            lt = median_ms(lambda: tma_mx(p, M, N, K, stages=st))
            if lt < best_tma:
                best_tma, best_st = lt, st
        ex_cp = (lat_cp - t8) / t8 * 100.0
        ex_tma = (best_tma - t8) / t8 * 100.0
        print(f"  {name:14s} M={M:5d} N={N:5d} K={K:5d}: bound={t8:.4f}  "
              f"cp.async={lat_cp:.4f}({ex_cp:+.1f}%)  TMA(st{best_st})={best_tma:.4f}({ex_tma:+.1f}%)")
    print("\nB1 TMA DONE")


if __name__ == "__main__":
    main()
