#!/usr/bin/env python3
"""vllm-perf r4 probe (lever 1): Marlin W4A16 tensor-core route for the
small-M decode GEMM regime (M=8..32), vs the shipping W4A4 GEMM route
(r2 small-M + r3 split-K tables INCLUDED, plus its qpfm pack + fp16 cast).

Per real merged-Linear shape (3B/8B), on synthetic packed weights:
  1. NUM gate   marlin_forward (Marlin fp4 low GEMM + decode_gemv_w8a16_hi_add
                high companion) vs the fp32 dequant reference x @ W_deq.T —
                cosine + max|rel|. W4A16 consumes the EXACT activation, so this
                is the same numerics class as the shipped GEMV routes (NOT
                byte-identical to the W4A4 GEMM route; T2-decode gates e2e).
                Also cross-checks the mb GEMV (M<=16) vs marlin (same W4A16
                math, different accumulation orders).
  2. BENCH      full-route cost under CUDA-GRAPH REPLAY timing per M:
                current forward() (pack + table-dispatched W4A4 GEMM + cast)
                vs marlin forward() (x_lo slice + marlin_gemm + hi_add);
                o_proj additionally compares the gather-inclusive chains
                (forward_gather_pack vs torch-gather + marlin). hi_add warp
                sweep (8/16/32) picks the per-shape companion config.

Run (container, idle GPU):
  CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.marlin_probe [--out out.json]
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import (load_kernels, load_decode_gemv,
                                       load_smallm, load_qpfm, load_splitk)
from promix.vllm_integ.decode_probe import SHAPES, make_packed, bench_graph
from promix.vllm_integ.marlin_w4a16 import prepack_marlin_low
from promix.quantize.mixed_linear import MixedQuantLinear

MS = (4, 8, 16, 24, 32)


def build_mql(pk, cd, qpf, qpfm, dg, cd_sm, cd_sk):
    mql = MixedQuantLinear(pk, nv=None, mx=None, cd=cd, out_dtype=torch.float16,
                           qpf=qpf, qpfm=qpfm, dg=dg, cd_sm=cd_sm, cd_sk=cd_sk)
    return mql.cuda()


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
        # recover the raw E4M3 low-scale bytes from the fp32 decode scales
        # (exact: each S_lo_dec value is e4m3*g; nearest-e4m3 recovers the byte)
        sf_lo = (pk.S_lo_dec / pk.g_w).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        ml = prepack_marlin_low(pk.W_lo_packed, sf_lo, pk.g_w, N, K_low)

        mql = build_mql(pk, cd, qpf, qpfm, dg, cd_sm, cd_sk)
        ok = mql.attach_marlin(ml)
        assert ok, tag

        res = {"N": N, "K": K, "K_high": K_high, "m": {}}
        x = (torch.randn(max(ms), K, device="cuda") * 0.5).half()
        gidx = torch.randperm(K, device="cuda", dtype=torch.int32).contiguous()
        for M in ms:
            xm = x[:M].contiguous()
            row = {}
            # ---- numerics ----
            y_mar = mql._marlin_forward(xm)
            ref = (xm.float() @ pk.W_deq.T)
            cos = torch.nn.functional.cosine_similarity(
                y_mar.float().flatten(), ref.flatten(), dim=0).item()
            md = (y_mar.float() - ref).abs().max().item()
            sc = ref.abs().max().item()
            row["cos_vs_deq"] = cos
            row["maxdiff_rel"] = md / max(sc, 1e-30)
            if M <= 16:
                y_gemv = dg.decode_gemv_w4a16_mb(xm, pk.W_lo_dec, pk.S_lo_dec,
                                                 pk.W_hi_dec, pk.S_hi_dec,
                                                 N, K_low, K_high, 32)
                row["cos_vs_gemv"] = torch.nn.functional.cosine_similarity(
                    y_mar.float().flatten(), y_gemv.flatten(), dim=0).item()
            if cos < 0.9999:
                all_pass = False
            if args.quick:
                res["m"][M] = row
                print(f"{tag} M={M}: cos_deq={cos:.7f}", flush=True)
                continue
            # ---- bench: full current route (marlin masked off) vs marlin ----
            mar_save = mql._marlin
            mql._marlin = None
            t_cur = bench_graph(lambda: mql(xm))
            mql._marlin = mar_save
            assert mql.marlin_route(M) == (8 <= M <= 32)
            t_mar = bench_graph(lambda: mql._marlin_forward(xm))
            t_lo = bench_graph(lambda: ml.gemm(xm[:, :K_low].contiguous()))
            y_fix = mql._marlin_forward(xm)
            t_hi = {w: bench_graph(lambda w=w: dg.decode_gemv_w8a16_hi_add(
                y_fix, xm, pk.W_hi_dec, pk.S_hi_dec, N, K_low, K_high, w))
                for w in (8, 16, 32)}
            row.update(t_cur_us=t_cur, t_marlin_route_us=t_mar,
                       t_marlin_lo_us=t_lo, t_hi_add_us=t_hi,
                       speedup=t_cur / t_mar)
            if "o" in tag.split("_")[-1]:
                # o_proj serve chains: gather-folded pack + W4A4 GEMM vs
                # torch gather + marlin W4A16
                t_gp = bench_graph(lambda: mql.forward_gather_pack(xm, gidx))
                t_gmar = bench_graph(
                    lambda: mql._marlin_forward(xm[:, gidx.long()]))
                row.update(t_cur_gatherpack_us=t_gp, t_gather_marlin_us=t_gmar,
                           speedup_o_chain=t_gp / t_gmar)
            res["m"][M] = row
            print(f"{tag} M={M}: cos_deq={cos:.7f} cur={t_cur:.1f}us "
                  f"marlin={t_mar:.1f}us ({t_cur/t_mar:.2f}x) "
                  f"[lo={t_lo:.1f} hi={min(t_hi.values()):.1f}]", flush=True)
        results["shapes"][tag] = res

    results["verdict"] = "MARLIN_GATE_ALL_PASS" if all_pass else "MARLIN_GATE_FAIL"
    print(results["verdict"], flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
