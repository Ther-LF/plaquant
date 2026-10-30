#!/usr/bin/env python3
"""vllm-perf r5 probe (lever 1): MXFP8-marlin high-segment companion vs the
r4 hi_add GEMV inside the Marlin W4A16 small-M route.

r4's named bottleneck: the hi_add GEMV scales ~linearly in M (7x the marlin
low's per-byte cost at 3B-qkv M=16) and closes every window by M=16-24. vLLM
0.32's marlin kernel natively supports MXFP8 (fp8 weights + e8m0 group-32
scales — exactly our checkpoint high-segment layout), so the companion is a
second marlin_gemm call + a g-folding alpha-add.

Per real merged-Linear shape (3B/8B), on synthetic packed weights:
  1. NUM gate   full marlin route with the MXFP8-marlin companion vs the fp32
                dequant reference x @ W_deq.T (cosine + max|rel|), plus a
                cross-check vs the r4 hi_add route (same W4A16 math, different
                high-segment accumulation order).
  2. BENCH      CUDA-graph replay per M: hi-companion only (hi_add best-warp
                vs marlin_hi gemm_add) -> _MARLIN_HI_TABLE crossovers; full
                route r4 (low+hi_add) vs r5 (low+marlin_hi); and the current
                W4A4 GEMM route (marlin masked) vs r5 at M=8..64 to re-derive
                the _MARLIN_TABLE window tops (o_proj rows use the
                gather-inclusive serve chains like r4).

Run (container, idle GPU):
  CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.marlin_hi_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import (load_kernels, load_decode_gemv,
                                       load_smallm, load_qpfm, load_splitk)
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph
from promix.vllm_integ.marlin_w4a16 import prepack_marlin_low, prepack_marlin_high
from promix.quantize.mixed_linear import MixedQuantLinear

MS = (2, 4, 8, 16, 24, 32, 48, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=None)
    ap.add_argument("--quick", action="store_true", help="numerics gates only")
    args = ap.parse_args()
    ms = tuple(int(m) for m in args.ms.split(",")) if args.ms else MS
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    qpfm = load_qpfm()
    dg = load_decode_gemv()
    cd_sm = load_smallm()
    cd_sk = load_splitk()
    assert dg is not None and hasattr(dg, "decode_gemv_w8a16_hi_add")

    results = {"shapes": {}}
    all_pass = True
    for tag, N, K, K_high in SHAPES:
        pk = make_packed(N, K, K_high, cd)
        K_low = K - K_high
        sf_lo = (pk.S_lo_dec / pk.g_w).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        ml = prepack_marlin_low(pk.W_lo_packed, sf_lo, pk.g_w, N, K_low)
        mh = prepack_marlin_high(pk.W_hi_bytes, pk.sf_hi_raw, pk.g_w, N, K_high)

        mql = MixedQuantLinear(pk, nv=None, mx=None, cd=cd, out_dtype=torch.float16,
                               qpf=qpf, qpfm=qpfm, dg=dg, cd_sm=cd_sm, cd_sk=cd_sk).cuda()
        assert mql.attach_marlin(ml, hi_warps=32 if N >= 4096 else 16,
                                 marlin_high=mh), tag

        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        x = (torch.randn(max(ms), K, device="cuda") * 0.5).half()
        gidx = torch.randperm(K, device="cuda", dtype=torch.int32).contiguous()
        for M in ms:
            xm = x[:M].contiguous()
            row = {}
            # ---- numerics: force the marlin-hi companion ----
            mql._marlin_hi_mlo = 1
            y_r5 = mql._marlin_forward(xm)
            ref = (xm.float() @ pk.W_deq.T)
            cos = torch.nn.functional.cosine_similarity(
                y_r5.float().flatten(), ref.flatten(), dim=0).item()
            md = (y_r5.float() - ref).abs().max().item()
            sc = ref.abs().max().item()
            row["cos_vs_deq"] = cos
            row["maxdiff_rel"] = md / max(sc, 1e-30)
            if M <= 32:
                mql._marlin_hi = None       # force the r4 hi_add route
                y_r4 = mql._marlin_forward(xm)
                mql._marlin_hi = mh
                row["cos_vs_hiadd_route"] = torch.nn.functional.cosine_similarity(
                    y_r5.float().flatten(), y_r4.float().flatten(), dim=0).item()
            if cos < 0.9999:
                all_pass = False
            if args.quick:
                res["m"][M] = row
                print(f"{tag} M={M}: cos_deq={cos:.7f}", flush=True)
                continue
            # ---- bench: hi companion only ----
            Y = y_r5.clone()
            x_hi = xm[:, K_low:].to(torch.bfloat16)
            t_mar_hi = bench_graph(lambda: mh.gemm_add(Y, x_hi))
            row["t_marlin_hi_us"] = t_mar_hi
            if M <= 32:
                t_hi = {w: bench_graph(lambda w=w: dg.decode_gemv_w8a16_hi_add(
                    Y, xm, pk.W_hi_dec, pk.S_hi_dec, N, K_low, K_high, w))
                    for w in (8, 16, 32)}
                row["t_hi_add_us"] = t_hi
                row["hi_speedup"] = min(t_hi.values()) / t_mar_hi
            # ---- bench: full routes ----
            mql._marlin_hi_mlo = 1
            t_r5 = bench_graph(lambda: mql._marlin_forward(xm))
            row["t_route_r5_us"] = t_r5
            if M <= 32:
                mql._marlin_hi = None
                t_r4 = bench_graph(lambda: mql._marlin_forward(xm))
                mql._marlin_hi = mh
                row["t_route_r4_us"] = t_r4
                row["route_speedup_vs_r4"] = t_r4 / t_r5
            # current W4A4 GEMM route (marlin masked off)
            mar_save = mql._marlin
            mql._marlin = None
            t_cur = bench_graph(lambda: mql(xm))
            mql._marlin = mar_save
            row["t_cur_us"] = t_cur
            row["route_speedup_vs_w4a4"] = t_cur / t_r5
            if "o" in tag.split("_")[-1]:
                t_gp = bench_graph(lambda: mql.forward_gather_pack(xm, gidx))
                t_gmar = bench_graph(
                    lambda: mql._marlin_forward(xm[:, gidx.long()]))
                row.update(t_cur_gatherpack_us=t_gp, t_gather_marlin_us=t_gmar,
                           speedup_o_chain=t_gp / t_gmar)
            res["m"][M] = row
            hi_cmp = (f"hi_add={min(row['t_hi_add_us'].values()):.1f}us "
                      if M <= 32 else "")
            print(f"{tag} M={M}: cos_deq={cos:.7f} mar_hi={t_mar_hi:.1f}us "
                  f"{hi_cmp}r5={t_r5:.1f}us "
                  f"vs_r4={row.get('route_speedup_vs_r4', float('nan')):.2f}x "
                  f"vs_w4a4={t_cur / t_r5:.2f}x", flush=True)
        results["shapes"][tag] = res

    results["verdict"] = "MARLIN_HI_GATE_ALL_PASS" if all_pass else "MARLIN_HI_GATE_FAIL"
    print(results["verdict"], flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
