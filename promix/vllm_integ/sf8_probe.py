#!/usr/bin/env python3
"""vllm-perf r2 probe (secondary lever): 1-byte decode-scale GEMV variants.

Per shape: torch.equal gate of decode_gemv_w4a16_sf8 / _mb_sf8 (raw E4M3/UE8M0
scale bytes + in-kernel g) against the fp32-scale r1 kernels, then graph-replay
timing at the GEMV-routed M (1, 2, 4).

Run: CUDA_VISIBLE_DEVICES=6 python -m promix.vllm_integ.sf8_probe [--out o.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_decode_gemv
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, _ = load_kernels()
    dg = load_decode_gemv()
    assert dg is not None and hasattr(dg, "decode_gemv_w4a16_sf8")

    results = {}
    for tag, N, K, K_high in SHAPES:
        pk = make_packed(N, K, K_high, cd)
        K_low = K - K_high
        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        x = (torch.randn(16, K, device="cuda") * 0.5).half()

        # ---- bit gates ----
        bit1 = torch.equal(
            dg.decode_gemv_w4a16(x[0], pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec,
                                 pk.S_hi_dec, N, K_low, K_high, 8),
            dg.decode_gemv_w4a16_sf8(x[0], pk.W_lo_dec, pk.sf_lo_raw, pk.W_hi_dec,
                                     pk.sf_hi_raw, N, K_low, K_high, 8, pk.g_w))
        bit_mb = True
        for M in (2, 3, 4, 8, 16):
            xm = x[:M].contiguous()
            a = dg.decode_gemv_w4a16_mb(xm, pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec,
                                        pk.S_hi_dec, N, K_low, K_high, 32)
            b = dg.decode_gemv_w4a16_mb_sf8(xm, pk.W_lo_dec, pk.sf_lo_raw, pk.W_hi_dec,
                                            pk.sf_hi_raw, N, K_low, K_high, 32, pk.g_w)
            if not torch.equal(a, b):
                bit_mb = False
        res["bit_m1"], res["bit_mb"] = bool(bit1), bool(bit_mb)

        # ---- graph-replay timing at the GEMV-routed M ----
        for M in (1, 2, 4):
            xm = x[:M].contiguous()
            if M == 1:
                t_fp32 = bench_graph(lambda: dg.decode_gemv_w4a16(
                    xm[0], pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec, pk.S_hi_dec,
                    N, K_low, K_high, 24 if N >= 4096 else 8))
                t_sf8 = bench_graph(lambda: dg.decode_gemv_w4a16_sf8(
                    xm[0], pk.W_lo_dec, pk.sf_lo_raw, pk.W_hi_dec, pk.sf_hi_raw,
                    N, K_low, K_high, 24 if N >= 4096 else 8, pk.g_w))
            else:
                t_fp32 = bench_graph(lambda: dg.decode_gemv_w4a16_mb(
                    xm, pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec, pk.S_hi_dec,
                    N, K_low, K_high, 32))
                t_sf8 = bench_graph(lambda: dg.decode_gemv_w4a16_mb_sf8(
                    xm, pk.W_lo_dec, pk.sf_lo_raw, pk.W_hi_dec, pk.sf_hi_raw,
                    N, K_low, K_high, 32, pk.g_w))
            res["m"][M] = {"fp32": t_fp32, "sf8": t_sf8}
        print(f"{tag:12s} bit_m1={bit1} bit_mb={bit_mb}  " +
              "  ".join(f"M={M}: {v['fp32']:6.1f}->{v['sf8']:6.1f}us "
                        f"({v['fp32']/v['sf8']:.2f}x)" for M, v in res["m"].items()),
              flush=True)
        results[tag] = res

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
