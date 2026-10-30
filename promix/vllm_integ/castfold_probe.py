#!/usr/bin/env python3
"""vllm-perf r5 probe (lever 2): GEMM epilogue output-cast fold bit gate +
graph-replay bench. fused_gemm_prepacked_bias (zero bias) must be
TORCH.EQUAL to fused_gemm_prepacked(...).to(fp16) on every shape (the r26
lever-A epilogue rounds acc->bf16 FIRST, so the fold is bit-identical to the
two-step store+cast), and the fold must win the [GEMM + cast] chain.

Run: CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.castfold_probe [--out o.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_qpfm, load_cd_bias
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph
from promix.quantize.mixed_linear import MixedQuantLinear

MS = (64, 256, 1024, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    qpfm = load_qpfm()
    cdb = load_cd_bias()
    assert cdb is not None

    results = {"shapes": {}}
    all_eq = True
    for tag, N, K, K_high in SHAPES:
        pk = make_packed(N, K, K_high, cd)
        mql = MixedQuantLinear(pk, nv=None, mx=None, cd=cd, out_dtype=torch.float16,
                               qpf=qpf, qpfm=qpfm, cd_bias=cdb).cuda()
        res = {"m": {}}
        x = (torch.randn(max(MS), K, device="cuda") * 0.5).half()
        for M in MS:
            xm = x[:M].contiguous()
            A = mql._pack_activation(xm)
            # reference: validated bf16 GEMM + torch cast
            D_ref = cd.fused_gemm_prepacked(
                A[0], A[1], pk.W_lo_packed, pk.physSFB_lo,
                A[2], A[3], pk.W_hi_bytes, pk.physSFB_hi,
                M, N, K - K_high, K_high, pk.g_w).to(torch.float16)
            D_new = cdb.fused_gemm_prepacked_bias(
                A[0], A[1], pk.W_lo_packed, pk.physSFB_lo,
                A[2], A[3], pk.W_hi_bytes, pk.physSFB_hi,
                mql._bias_or_zero(xm.device), M, N, K - K_high, K_high, pk.g_w)
            eq = torch.equal(D_ref, D_new)
            # -0.0 vs +0.0: torch.equal treats them equal, so a False here is real
            all_eq &= eq
            t_old = bench_graph(lambda: cd.fused_gemm_prepacked(
                A[0], A[1], pk.W_lo_packed, pk.physSFB_lo,
                A[2], A[3], pk.W_hi_bytes, pk.physSFB_hi,
                M, N, K - K_high, K_high, pk.g_w).to(torch.float16), reps=10, inner=5)
            t_new = bench_graph(lambda: cdb.fused_gemm_prepacked_bias(
                A[0], A[1], pk.W_lo_packed, pk.physSFB_lo,
                A[2], A[3], pk.W_hi_bytes, pk.physSFB_hi,
                mql._bias_or_zero(xm.device), M, N, K - K_high, K_high, pk.g_w),
                reps=10, inner=5)
            res["m"][M] = {"equal": eq, "t_gemm_cast_us": t_old,
                           "t_fold_us": t_new, "speedup": t_old / t_new}
            print(f"{tag} M={M}: equal={eq} {t_old:.1f}->{t_new:.1f}us "
                  f"({t_old/t_new:.3f}x)", flush=True)
        results["shapes"][tag] = res
    results["verdict"] = "CASTFOLD_BIT_PASS" if all_eq else "CASTFOLD_BIT_FAIL"
    print(results["verdict"], flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
