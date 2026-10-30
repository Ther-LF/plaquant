#!/usr/bin/env python3
"""vllm-perf r2 probe (lever 2): activation pack fusion numerics + bench.

Gates and graph-timed deltas for the two pack-path changes:
  A. qpfm  quant_pack_float_mixed (ONE launch, both segments) vs the two-launch
           quant_pack_float_fp4/fp8 — torch.equal on all four outputs (hard gate).
  B. hpk   hadamard_pack_mixed / hadamard28_pack_mixed (fused Hadamard+pack, ONE
           kernel) for the down_proj input vs:
             bitref: pack(matmul_hadU_cuda(x.float(), ...).half())  [fp32-trait
                     transform — torch.equal expected, the hpk gate contract]
             cur:    pack(matmul_hadU_cuda(x.fp16, ...))            [the current
                     vLLM chain; fp16-input library call, different stage order
                     -> NOT bit-equal; report GEMM-output cosine]
  Bench (CUDA-graph replay, r1 lesson): per down shape x M, the three chains
  (cur 3-launch, qpfm 2-launch, fused 1-kernel); per non-down shape, qpf vs qpfm.

Run: CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.pack_probe [--out o.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_qpfm, load_had_pack
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph

MS = (8, 16, 64, 256, 1024, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    qpfm = load_qpfm()
    hpk, hpk28 = load_had_pack()
    from promix.quantize.hadamard import get_hadK, matmul_hadU_cuda

    results = {}
    for tag, N, K, K_high in SHAPES:
        K_low = K - K_high
        is_down = tag.endswith("down")
        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        x = (torch.randn(max(MS), K, device="cuda") * 0.5).half()
        had_K, Kfac = get_hadK(K) if is_down else (None, 1)
        had_cpu = None if had_K is None else had_K.cpu()
        had_dev = None if had_K is None else had_K.cuda()   # graph-capture-safe
        pk = make_packed(N, K, K_high, cd)

        def gemm(a, M):
            return cd.fused_gemm_prepacked(a[0], a[1], pk.W_lo_packed, pk.physSFB_lo,
                                           a[2], a[3], pk.W_hi_bytes, pk.physSFB_hi,
                                           M, N, K_low, K_high, pk.g_w)

        def bits_eq(a, b, M):
            """Bit gate robust to the UNINITIALIZED SFA atom padding (the phys SFA
            buffer is padded to 128-row atoms; rows >= M are never written — at
            M%128!=0 raw torch.equal compares garbage). Gate = packed DATA bytes
            exactly equal AND the consuming GEMM outputs exactly equal (the GEMM
            reads the SF only where it multiplies real data)."""
            return (torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
                    and torch.equal(gemm(a, M), gemm(b, M)))

        for M in MS:
            xm = x[:M].contiguous()
            row = {}

            def pack2(inp):
                a_lo = qpf.quant_pack_float_fp4(inp, M, N, K_low, row_stride=K, col_off=0)
                a_hi = qpf.quant_pack_float_fp8(inp, M, N, K_high, row_stride=K, col_off=K_low)
                return a_lo[0], a_lo[1], a_hi[0], a_hi[1]

            # ---- gate A: qpfm bit-exact vs two-launch ----
            ref = pack2(xm)
            got = qpfm.quant_pack_float_mixed(xm, M, N, K_low, K_high)
            row["qpfm_bit"] = bits_eq(ref, got, M)

            # ---- bench: qpf(2) vs qpfm(1) ----
            row["pack_qpf2"] = bench_graph(lambda: pack2(xm))
            row["pack_qpfm"] = bench_graph(
                lambda: qpfm.quant_pack_float_mixed(xm, M, N, K_low, K_high))

            if is_down:
                def had_cur(inp):
                    return matmul_hadU_cuda(inp, had_dev, Kfac,
                                            fold_scale=True).to(torch.half)

                def fused(inp):
                    if K == 14336:
                        return hpk28.hadamard28_pack_mixed(inp, had_cpu, M, N, K_low, K_high)
                    return hpk.hadamard_pack_mixed(inp, M, N, K_low, K_high)

                # gate B1: fused == pack(fp32-trait hadamard) bit-exact
                y32 = matmul_hadU_cuda(xm.float(), had_dev, Kfac, fold_scale=True).half()
                bitref = pack2(y32.contiguous())
                got = fused(xm)
                row["hpk_bit_vs_fp32ref"] = bits_eq(bitref, got, M)
                # gate B2: GEMM-output cosine fused vs the CURRENT chain
                cur = pack2(had_cur(xm).contiguous())
                D_cur = gemm(cur, M)
                D_new = gemm(got, M)
                row["hpk_gemm_cos_vs_cur"] = float(torch.nn.functional.cosine_similarity(
                    D_cur.float().flatten(), D_new.float().flatten(), dim=0))
                # bench: current chain vs had+qpfm vs fused
                row["down_cur3"] = bench_graph(lambda: pack2(had_cur(xm).contiguous()))
                row["down_qpfm2"] = bench_graph(lambda: qpfm.quant_pack_float_mixed(
                    had_cur(xm).contiguous(), M, N, K_low, K_high))
                row["down_fused1"] = bench_graph(lambda: fused(xm))
                print(f"{tag:12s} M={M:5d} qpfm_bit={row['qpfm_bit']} "
                      f"hpk_bit={row['hpk_bit_vs_fp32ref']} "
                      f"cos={row['hpk_gemm_cos_vs_cur']:.8f}  "
                      f"cur3={row['down_cur3']:6.1f} qpfm2={row['down_qpfm2']:6.1f} "
                      f"fused1={row['down_fused1']:6.1f}us "
                      f"({row['down_cur3']/row['down_fused1']:.2f}x)", flush=True)
            else:
                print(f"{tag:12s} M={M:5d} qpfm_bit={row['qpfm_bit']} "
                      f"qpf2={row['pack_qpf2']:6.1f} qpfm={row['pack_qpfm']:6.1f}us "
                      f"({row['pack_qpf2']/row['pack_qpfm']:.2f}x)", flush=True)
            res["m"][M] = row
        results[tag] = res

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
