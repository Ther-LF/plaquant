#!/usr/bin/env python3
"""vllm-perf r3 probe (lever 1): stream-K/split-K dual-format GEMM for the
latency-bound small-M regime (decode b8/b16 + chunked-prefill tails).

Per real merged-Linear shape (3B/8B), on synthetic packed weights + a fixed
pre-packed activation:
  1. BIT gate   cd_splitk_ext decomp=3 (DataParallel: one unit per output tile,
                full serial K) vs the shipping cd.fused_gemm_prepacked —
                torch.equal expected (identical math + K order). Proves the
                forked glue itself is sound before any K splitting.
  2. NUM gate   split decompositions (heuristic / stream-K / split-K s=2,4,8)
                vs the shipping kernel: cosine + max|diff| (NOT torch.equal —
                the FP32 partial sums are reduced in a different order).
  3. BENCH      GEMM-only under CUDA-GRAPH REPLAY timing (the serving regime),
                per variant x M, vs the shipping kernel and the r2 small-M
                cfg1 (128x64).

Run (container, idle GPU):
  CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.splitk_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_smallm, load_splitk
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph

# (label, cfg, splits, decomp)
VARIANTS = [
    ("c0_heur", 0, 1, 0),
    ("c0_sk", 0, 1, 2),
    ("c0_s2", 0, 2, 1),
    ("c0_s4", 0, 4, 1),
    ("c1_heur", 1, 1, 0),
    ("c1_sk", 1, 1, 2),
    ("c1_s2", 1, 2, 1),
    ("c1_s4", 1, 4, 1),
    ("c1_s8", 1, 8, 1),
]
MS = (8, 16, 32, 64, 128, 256, 384, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=None, help="comma-separated M list override")
    ap.add_argument("--quick", action="store_true", help="numerics gates only, no bench")
    args = ap.parse_args()
    ms = tuple(int(m) for m in args.ms.split(",")) if args.ms else MS
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    sm = load_smallm()
    sk = load_splitk()
    print(sk.report(), flush=True)

    results = {"variants": [v[0] for v in VARIANTS], "shapes": {}}
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
            gemm_args = (A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                         A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                         M, N, K_low, K_high, pk.g_w)

            ref = cd.fused_gemm_prepacked(*gemm_args)
            row = {"num": {}, "t": {}}

            # 1. bit gate: DataParallel (serial per-tile K) must be torch.equal
            dp = sk.fused_gemm_prepacked_sk(*gemm_args, 0, 1, 3)
            row["bit_dp"] = bool(torch.equal(dp, ref))

            # 2. numerics of split decompositions
            for lbl, cfg, splits, decomp in VARIANTS:
                D = sk.fused_gemm_prepacked_sk(*gemm_args, cfg, splits, decomp)
                cos = float(torch.nn.functional.cosine_similarity(
                    D.float().flatten(), ref.float().flatten(), dim=0))
                mx = float((D.float() - ref.float()).abs().max())
                eq = bool(torch.equal(D, ref))
                row["num"][lbl] = {"cos": cos, "maxdiff": mx, "eq": eq}

            if not args.quick:
                row["t"]["ship"] = bench_graph(lambda: cd.fused_gemm_prepacked(*gemm_args))
                row["t"]["sm_c1"] = bench_graph(lambda: sm.fused_gemm_prepacked_cfg(*gemm_args, 1))
                for lbl, cfg, splits, decomp in VARIANTS:
                    row["t"][lbl] = bench_graph(
                        lambda cfg=cfg, splits=splits, decomp=decomp:
                        sk.fused_gemm_prepacked_sk(*gemm_args, cfg, splits, decomp))

            res["m"][M] = row
            mincos = min(v["cos"] for v in row["num"].values())
            line = f"{tag:12s} M={M:5d} bit_dp={'T' if row['bit_dp'] else 'F'} mincos={mincos:.8f}"
            if not args.quick:
                best = min((l for l in row["t"] if l not in ("ship", "sm_c1")),
                           key=row["t"].get)
                line += (f"  ship={row['t']['ship']:7.1f} sm_c1={row['t']['sm_c1']:7.1f}  "
                         + " ".join(f"{l}={row['t'][l]:6.1f}" for l, *_ in VARIANTS)
                         + f"  best={best} ({min(row['t']['ship'], row['t']['sm_c1'])/row['t'][best]:.2f}x vs r2-best)")
            print(line, flush=True)
        results["shapes"][tag] = res

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
