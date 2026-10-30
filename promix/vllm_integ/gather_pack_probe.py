#!/usr/bin/env python3
"""vllm-perf r3 probe (lever 3): gather-folded o_proj activation pack.

Gates per o_proj shape x M:
  1. BIT   quant_pack_float_mixed_gather(x, order) outputs torch.equal to
           quant_pack_float_mixed(x[:, order]) (data bytes + consuming-GEMM
           output; raw SFA compare only meaningful at M%128==0 — padding).
  2. BENCH graph-replay: [torch gather + qpfm] chain vs the gather-folded
           single launch, per M.

Run (venv, idle GPU):
  CUDA_VISIBLE_DEVICES=7 /home/luofan/baselines/venv/bin/python \
      -m promix.vllm_integ.gather_pack_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_qpfm
from promix.vllm_integ.decode_probe import make_packed, bench_graph

# o_proj shapes: (tag, N, K, K_high)
SITES = [
    ("3b_o", 3072, 3072, 384),
    ("8b_o", 4096, 4096, 512),
]
MS = (4, 8, 16, 64, 256, 512, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    qpfm = load_qpfm()

    results = {"sites": {}}
    allpass = True
    for tag, N, K, K_high in SITES:
        K_low = K - K_high
        pk = make_packed(N, K, K_high, cd)
        order = torch.randperm(K, device="cuda", dtype=torch.int64)
        order_i32 = order.to(torch.int32).contiguous()
        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        for M in MS:
            x = (torch.randn(M, K, device="cuda") * 0.5).half()
            xg = x[:, order].contiguous()
            ref = qpfm.quant_pack_float_mixed(xg, M, N, K_low, K_high)
            fus = qpfm.quant_pack_float_mixed_gather(x, order_i32, M, N, K_low, K_high)
            code_eq = bool(torch.equal(ref[0], fus[0]) and torch.equal(ref[2], fus[2]))
            d_r = cd.fused_gemm_prepacked(ref[0], ref[1], pk.W_lo_packed, pk.physSFB_lo,
                                          ref[2], ref[3], pk.W_hi_bytes, pk.physSFB_hi,
                                          M, N, K_low, K_high, pk.g_w)
            d_f = cd.fused_gemm_prepacked(fus[0], fus[1], pk.W_lo_packed, pk.physSFB_lo,
                                          fus[2], fus[3], pk.W_hi_bytes, pk.physSFB_hi,
                                          M, N, K_low, K_high, pk.g_w)
            gemm_eq = bool(torch.equal(d_r, d_f))
            ok = code_eq and gemm_eq
            allpass = allpass and ok

            def chain():
                xr = x[:, order].contiguous()
                qpfm.quant_pack_float_mixed(xr, M, N, K_low, K_high)

            t_chain = bench_graph(chain)
            t_fused = bench_graph(
                lambda: qpfm.quant_pack_float_mixed_gather(x, order_i32, M, N, K_low, K_high))
            res["m"][M] = {"code_eq": code_eq, "gemm_eq": gemm_eq,
                           "t_chain": t_chain, "t_fused": t_fused}
            print(f"{tag:6s} M={M:5d} bit={'PASS' if ok else 'FAIL'} "
                  f"chain={t_chain:6.1f}us fused={t_fused:6.1f}us "
                  f"({t_chain/t_fused:.2f}x)", flush=True)
        results["sites"][tag] = res

    print(f"GATHER_PACK_GATE_{'ALL_PASS' if allpass else 'FAIL'}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
