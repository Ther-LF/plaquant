#!/usr/bin/env python3
"""rd-r14: per-shape fused-GEMM latency at the r=1/8 vs r=1/4 K_high splits,
under CUDA-graph replay (bench_graph), at the decode-route Ms (64..384).

Per 3B shape x r x M, measures every candidate config the route can pick:
  cfg0   shipping 128x128 (cd.fused_gemm_prepacked)
  cfg1   128x64 small-M config (sm.fused_gemm_prepacked_cfg cfg=1)
  sk_*   split-K/stream-K variants where a _SPLITK_TABLE window exists
plus `route` = what the mixed_linear tables actually pick at that M (r-blind).
Also records the activation pack cost (qpf fp4+fp8) per r.

Run: docker exec, CUDA_VISIBLE_DEVICES=<idle-gpu>, PYTHONPATH=worktree.
"""
import argparse
import json

import torch

from promix.vllm_integ.kernels import load_kernels, load_smallm, load_splitk
from promix.vllm_integ.decode_probe import make_packed, bench_graph
from promix.quantize.mixed_linear import _SMALLM_TABLE, _SPLITK_TABLE

SHAPES3B = [("qkv", 5120, 3072), ("o", 3072, 3072),
            ("gate_up", 16384, 3072), ("down", 3072, 8192)]
MS = (64, 128, 192, 256, 384)
RS = {"r125": 8, "r25": 4}          # K_high = K // div


def route_pick(N, K, M):
    for m_lo, m_hi, cfg, splits, decomp in _SPLITK_TABLE.get((N, K), ()):
        if m_lo <= M <= m_hi:
            return ("sk", cfg, splits, decomp)
    win = _SMALLM_TABLE.get((N, K))
    if win and win[0] <= M <= win[1]:
        return ("cfg1",)
    return ("cfg0",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    sm = load_smallm()
    sk = load_splitk()
    res = {"ms": MS, "shapes": {}}
    for tag, N, K in SHAPES3B:
        res["shapes"][tag] = {"N": N, "K": K}
        for rname, dv in RS.items():
            K_high = K // dv
            K_low = K - K_high
            pk = make_packed(N, K, K_high, cd)
            x = (torch.randn(max(MS), K, device="cuda") * 0.5).half()
            rows = {}
            for M in MS:
                xm = x[:M].contiguous()
                a_lo = qpf.quant_pack_float_fp4(xm, M, N, K_low, row_stride=K, col_off=0)
                a_hi = qpf.quant_pack_float_fp8(xm, M, N, K_high, row_stride=K, col_off=K_low)
                A_lo, pSFA_lo, A_hi, pSFA_hi = a_lo[0], a_lo[1], a_hi[0], a_hi[1]
                gemm_args = (A_lo, pSFA_lo, pk.W_lo_packed, pk.physSFB_lo,
                             A_hi, pSFA_hi, pk.W_hi_bytes, pk.physSFB_hi,
                             M, N, K_low, K_high, pk.g_w)
                e = {}
                e["pack_us"] = bench_graph(lambda: (
                    qpf.quant_pack_float_fp4(xm, M, N, K_low, row_stride=K, col_off=0),
                    qpf.quant_pack_float_fp8(xm, M, N, K_high, row_stride=K, col_off=K_low)))
                e["cfg0"] = bench_graph(lambda: cd.fused_gemm_prepacked(*gemm_args))
                e["cfg1"] = bench_graph(lambda: sm.fused_gemm_prepacked_cfg(*gemm_args, 1))
                for m_lo, m_hi, cfg, splits, decomp in _SPLITK_TABLE.get((N, K), ()):
                    key = "sk_c%d_s%d_d%d" % (cfg, splits, decomp)
                    if key not in e:
                        e[key] = bench_graph(
                            lambda cfg=cfg, splits=splits, decomp=decomp:
                            sk.fused_gemm_prepacked_sk(*gemm_args, cfg, splits, decomp))
                pick = route_pick(N, K, M)
                e["route_pick"] = "_".join(str(p) for p in pick)
                if pick[0] == "cfg1":
                    e["route_us"] = e["cfg1"]
                elif pick[0] == "sk":
                    e["route_us"] = e["sk_c%d_s%d_d%d" % (pick[1], pick[2], pick[3])]
                else:
                    e["route_us"] = e["cfg0"]
                e["best"] = min((v, k) for k, v in e.items()
                                if isinstance(v, float) and k.startswith(("cfg", "sk")))[1]
                rows[M] = e
                print(tag, rname, M,
                      {k: (round(v, 1) if isinstance(v, float) else v)
                       for k, v in e.items()}, flush=True)
            res["shapes"][tag][rname] = {"K_high": K_high, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
