#!/usr/bin/env python3
"""vllm-perf r2 probe (lever 1): small-M fused-GEMM config selection.

Per real merged-Linear shape (3B/8B), on synthetic packed weights + a fixed
pre-packed activation:
  1. BIT gate   cd_smallm_ext.fused_gemm_prepacked_cfg(cfg) output vs the
                shipping cd.fused_gemm_prepacked (torch.equal expected: identical
                math, same K-chunk order; cosine reported regardless).
  2. BENCH      GEMM-only, under CUDA-GRAPH REPLAY timing (the serving regime;
                r1 lesson: eager timing picks wrong thresholds), per cfg x M.

Run (container, idle GPU):
  CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.gemm_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_smallm
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph

CFGS = {0: "128x128/4/2", 1: "128x64/4/2", 2: "128x64/6/2", 3: "128x64/6/3"}
MS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=None, help="comma-separated M list override")
    args = ap.parse_args()
    ms = tuple(int(m) for m in args.ms.split(",")) if args.ms else MS
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    sm = load_smallm()
    print(sm.report(), flush=True)

    results = {"cfgs": CFGS, "shapes": {}}
    for tag, N, K, K_high in SHAPES:
        pk = make_packed(N, K, K_high, cd)
        K_low = K - K_high
        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        x = (torch.randn(max(ms), K, device="cuda") * 0.5).half()
        for M in ms:
            xm = x[:M].contiguous()
            a_lo = qpf.quant_pack_float_fp4(xm, M, N, K_low, row_stride=K, col_off=0)
            a_hi = qpf.quant_pack_float_fp8(xm, M, N, K_high, row_stride=K, col_off=K_low)
            A_lo, pSFA_lo, A_hi, pSFA_hi = a_lo[0], a_lo[1], a_hi[0], a_hi[1]

            ref = cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                M, N, K_low, K_high, pk.g_w)
            row = {"cfg": {}, "bit": {}, "cos": {}}
            for cfg in CFGS:
                D = sm.fused_gemm_prepacked_cfg(
                    A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                    A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                    M, N, K_low, K_high, pk.g_w, cfg)
                row["bit"][cfg] = bool(torch.equal(D, ref))
                row["cos"][cfg] = float(torch.nn.functional.cosine_similarity(
                    D.float().flatten(), ref.float().flatten(), dim=0))

            row["ship"] = bench_graph(lambda: cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                M, N, K_low, K_high, pk.g_w))
            for cfg in CFGS:
                row["cfg"][cfg] = bench_graph(lambda cfg=cfg: sm.fused_gemm_prepacked_cfg(
                    A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                    A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                    M, N, K_low, K_high, pk.g_w, cfg))
            res["m"][M] = row
            best = min(row["cfg"], key=row["cfg"].get)
            bits = "".join("T" if row["bit"][c] else "f" for c in CFGS)
            print(f"{tag:12s} M={M:5d} ship={row['ship']:7.1f}us  " +
                  " ".join(f"c{c}={row['cfg'][c]:7.1f}" for c in CFGS) +
                  f"  best=c{best} ({row['ship']/row['cfg'][best]:.2f}x) bit={bits} "
                  f"mincos={min(row['cos'].values()):.8f}", flush=True)
        results["shapes"][tag] = res

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
