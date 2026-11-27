"""rd-ringsize (P0-3) — group-width b (ring_size) sensitivity: alloc analysis + op cost.

Two subcommands, both reading the rd_acctune --stats_cache files produced by the
P0-3 accuracy sweep (stats_{model}_b{B}.pt, one per model x ring_size):

  alloc  (no GPU): recompute the EXACT shipped allocation (rv.measured_alloc +
         rv.clamp_chosen at budget_frac=0.125 — byte-identical inputs to the
         accuracy runs, deterministic) and report, per (model, b):
           - non-monotone layer fraction: Linears whose chosen FP8 ring set is
             NOT the contiguous tail {nr-k..nr-1}, i.e. the runtime needs an
             online group gather (column order != arange in rv.build_eval);
             o_proj is never reallocated (always positional, Pi == I).
           - realized FP8 ring fraction (pre-eval cross-check of avg_fp8)
           - per-proj K_high (channels) median/min/max across layers — feeds
             the op bench's representative K_high per b.

  op     (GPU, idle): representative-shape operator cost at M=8192 through the
         SHIPPING path (pack_weight_mixed + MixedQuantLinear, full kernel set,
         qpfm fused mixed pack), per b at that b's median K_high from `alloc`:
           - t_pack       pack_only(x)              (monotone layer, Pi == I)
           - t_gather_pk  fused gather+pack (qpfm.quant_pack_float_mixed_gather)
                          when pad-eligible, else torch gather + pack chain
                          (what a non-monotone layer actually pays)
           - t_gemm       mql._gemm (castfold 128x128 at M=8192)
         bench_graph CUDA-graph replay timing (serving regime, us/iter).

Usage:
  python promix/scripts/rd_ringsize_harvest.py alloc --stats_dir /home/luofan/ringsize \
      --out /home/luofan/ringsize/alloc_analysis.json
  CUDA_VISIBLE_DEVICES=7 python promix/scripts/rd_ringsize_harvest.py op \
      --alloc_json /home/luofan/ringsize/alloc_analysis.json \
      --out /home/luofan/ringsize/op_cost.json
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))

BUDGET = 0.125


def recompute_chosen(stats):
    import rd_validity as rv
    total = sum(stats[n]["nr"] for n in stats)
    budget = int(round(BUDGET * total))
    return rv.clamp_chosen(rv.measured_alloc(stats, budget, total), stats), total


def alloc_main(args):
    out = {}
    for path in sorted(glob.glob(os.path.join(args.stats_dir, "stats_*_b*.pt"))):
        m = re.match(r"stats_(\w+)_b(\d+)\.pt", os.path.basename(path))
        model, b = m.group(1), int(m.group(2))
        stats = torch.load(path, map_location="cpu", weights_only=False)
        chosen, total = recompute_chosen(stats)
        nsel = sum(len(v) for v in chosen.values())
        nonmono, nlin = 0, 0
        per_proj = {}
        for name, S in chosen.items():
            nr = stats[name]["nr"]
            k = len(S)
            tail = set(range(nr - k, nr))
            nlin += 1
            is_nm = S != tail
            nonmono += int(is_nm)
            proj = name.split(".")[-1]
            d = per_proj.setdefault(proj, {"n": 0, "nonmono": 0, "khigh": []})
            d["n"] += 1
            d["nonmono"] += int(is_nm)
            d["khigh"].append(k * b)
        for proj, d in per_proj.items():
            v = torch.tensor(sorted(d["khigh"]), dtype=torch.float64)
            d["khigh_median"] = int(v.median())
            d["khigh_min"], d["khigh_max"] = int(v.min()), int(v.max())
            del d["khigh"]
        out[f"{model}_b{b}"] = dict(
            model=model, ring_size=b, total_rings=total, sel_rings=nsel,
            realized_frac=nsel / total, alloc_linears=nlin,
            nonmono_linears=nonmono, nonmono_frac=nonmono / nlin,
            per_proj=per_proj)
        print(f"{model} b={b:4d}: nonmono {nonmono}/{nlin} "
              f"({nonmono / nlin:.1%}) frac={nsel / total:.4f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.out}", flush=True)


# Representative shapes for the op bench (8B, per spec): per-projection GEMMs.
OP_SHAPES = {"gate_up": ("gate_proj", 14336, 4096), "down": ("down_proj", 4096, 14336)}


def op_main(args):
    from promix.eval.full_network_realkernel import load_kernels
    from promix.vllm_integ.kernels import (load_smallm, load_splitk, load_cd_bias,
                                           load_fk_smallk, load_fk_spg, load_qpfm)
    from promix.vllm_integ.decode_probe import bench_graph
    from promix.quantize.weight_pack import pack_weight_mixed
    from promix.quantize.mixed_linear import MixedQuantLinear

    with open(args.alloc_json) as f:
        alloc = json.load(f)
    nv, mx, cd, qpf = load_kernels()
    qpfm = load_qpfm()
    kw = dict(qpf=qpf, qpfm=qpfm, cd_bias=load_cd_bias(), cd_sm=load_smallm(),
              cd_sk=load_splitk(), cd_fk=load_fk_smallk(), cd_fk_spg=load_fk_spg())
    M = args.m
    rows = []
    for key, a in sorted(alloc.items(), key=lambda kv: kv[1]["ring_size"]):
        if a["model"] != args.model:
            continue
        b = a["ring_size"]
        for shape, (proj, N, K) in OP_SHAPES.items():
            K_high = a["per_proj"][proj]["khigh_median"]
            K_low = K - K_high
            torch.manual_seed(0)
            W = torch.randn(N, K, device="cuda") * 0.08
            x = (torch.randn(M, K, device="cuda") * 0.5).half()
            packed = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
            mql = MixedQuantLinear(packed, nv, mx, cd, **kw)
            # ring-level non-identity gather order at granularity b (Pi != I)
            nr = K // b
            g = torch.Generator().manual_seed(1)
            ring_perm = torch.randperm(nr, generator=g)
            order = (ring_perm[:, None] * b + torch.arange(b)).flatten().cuda()
            order_i32 = order.to(torch.int32).contiguous()

            t_pack = bench_graph(lambda: mql.pack_only(x))
            fused_ok = (qpfm is not None and K_high > 0
                        and mql.K_low_pad == mql.K_low
                        and mql.K_high_pad == mql.K_high)
            if K_high == 0:
                t_gather, gather_path = None, "n/a (pure NVFP4, 0 FP8 rings)"
            elif fused_ok:
                t_gather = bench_graph(lambda: qpfm.quant_pack_float_mixed_gather(
                    x, order_i32, M, N, K_low, K_high))
                gather_path = "fused_gather_pack"
            else:
                t_gather = bench_graph(
                    lambda: mql.pack_only(x[:, order].contiguous()))
                gather_path = "torch_gather_then_pack"
            A = mql.pack_only(x)
            t_gemm = bench_graph(lambda: mql._gemm(A[0], A[1], A[2], A[3], M))
            rec = dict(model=args.model, shape=shape, ring_size=b, M=M, N=N, K=K,
                       K_high=K_high, K_low=K_low,
                       K_low_pad=mql.K_low_pad, K_high_pad=mql.K_high_pad,
                       khigh_mod128=K_high % 128, khigh_mod64=K_high % 64,
                       pack_us=t_pack, gather_pack_us=t_gather,
                       gather_path=gather_path, gemm_us=t_gemm)
            rows.append(rec)
            gstr = f"{t_gather:7.1f}us" if t_gather is not None else "    n/a"
            print(f"b={b:4d} {shape:7s} K_high={K_high:5d} pack={t_pack:7.1f}us "
                  f"gather+pack={gstr} ({gather_path}) "
                  f"gemm={t_gemm:7.1f}us", flush=True)
            del W, x, packed, mql, A
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"WROTE {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alloc")
    a.add_argument("--stats_dir", required=True)
    a.add_argument("--out", required=True)
    o = sub.add_parser("op")
    o.add_argument("--alloc_json", required=True)
    o.add_argument("--model", default="8b")
    o.add_argument("--m", type=int, default=8192)
    o.add_argument("--out", required=True)
    args = ap.parse_args()
    (alloc_main if args.cmd == "alloc" else op_main)(args)


if __name__ == "__main__":
    main()
