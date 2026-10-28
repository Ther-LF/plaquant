"""Task 6 sub-lever — RING-SIZE ablation on ring-R-on @ 1/8 (3B).

The +0.0478 ring-vs-dense rotation cost (ring64 uniform 8.3452 vs dense uniform
8.2974) is a MIXING-WIDTH effect: the dense R1 baseline is a RANDOM dense
orthogonal (two QRs, not trained), so a wider Hadamard ring (128/256) mixes
2-4x more coordinates and should shrink ε while staying usable for the allocator
(3B: 24 rings @128, 12 @256; both keep K_high%32=0 / K_low%64=0).

For each ring_size S it reports, at matched 1/8 FP8 budget (R-on, ring-R1):
  - ring-S uniform   (chosen=None)
  - ring-S measured  (measured-MSE allocation on S-ring deltaD)

Decision rule (vs dense-R-on uniform 8.2974):
  if  ring-S measured  <  8.2974  -> flip SUCCEEDS at ring S (OVERTURNED-at-S)
  else ε(S)=ring-S uniform-8.2974 and gain(S)=ring-S uniform-measured; if ε still
  > gain even at 256 -> verdict stands, mixing-width lever also exhausted.

Reuses rd_validity build/collect/alloc/eval (now ring_size-parametrized).
"""
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))

import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.data import get_wikitext2

import rd_validity as rv
import rd_ron


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ring_sizes", default="128,256")
    ap.add_argument("--dense_ron_uniform", type=float, default=8.2974,
                    help="in-harness dense-R-on uniform-1/8 PPL (the flip target)")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--dense_rotation", action="store_true",
                    help="DENSE-LIMIT probe: use the SHIPPED dense R1 rotation "
                         "(ring_r1=False) instead of the ring substrate; ring_size "
                         "is then only the measured-allocation atom (K alignment).")
    ap.add_argument("--out", default="/home/luofan/RD_RON_RINGSIZE.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    ring_r1 = not args.dense_rotation          # dense-limit probe => dense R1
    ring_sizes = [int(x) for x in args.ring_sizes.split(",")]
    dense = args.dense_ron_uniform
    print(f"START ringsize model={config['model']['name']} sizes={ring_sizes} "
          f"ring_r1={ring_r1} dense_uniform={dense}", flush=True)
    results = {"meta": dict(model=config["model"]["name"], ring_sizes=ring_sizes,
                            ring_r1=ring_r1, dense_ron_uniform=dense), "by_ring": {}}

    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)

    def run(tag, chosen, S):
        t = time.time()
        model = rv.build_eval(config, r1_off=False, chosen=chosen,
                              ring_r1=ring_r1, ring_size=S)
        frac, hi, tot = rv.avg_fp8_fraction(model)
        ppl = rv.eval_ppl(model, config)
        print(f"RESULT {tag} ppl={ppl:.4f} avg_fp8={frac:.4f} ({hi}/{tot}) "
              f"({time.time()-t:.0f}s)", flush=True)
        del model
        cleanup_memory()
        return dict(ppl=ppl, avg_fp8=frac, hi_ch=hi, tot_ch=tot)

    for S in ring_sizes:
        cmodel = rv.build_base(config, r1_off=False, ring_r1=ring_r1, ring_size=S)
        cmodel.to(DEV)
        stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens, ring=S)
        del cmodel
        cleanup_memory()
        total_rings = sum(stats[n]["nr"] for n in stats)
        budget = int(round(0.125 * total_rings))
        chosen = rv.measured_alloc(stats, budget, total_rings)
        print(f"RING S={S} total_rings={total_rings} budget@1/8={budget}", flush=True)

        uni = run(f"ring{S}_uniform", None, S)
        meas = run(f"ring{S}_measured", chosen, S)
        eps = uni["ppl"] - dense                 # rotation cost vs dense
        gain = uni["ppl"] - meas["ppl"]          # allocation gain at this S
        beats = meas["ppl"] < dense
        results["by_ring"][str(S)] = dict(
            total_rings=total_rings, budget=budget,
            uniform=uni, measured=meas, eps_vs_dense=eps, alloc_gain=gain,
            net_vs_dense=meas["ppl"] - dense, beats_dense=beats)
        print(f"RING S={S}: uniform={uni['ppl']:.4f} measured={meas['ppl']:.4f} "
              f"eps={eps:+.4f} gain={gain:+.4f} net={meas['ppl']-dense:+.4f} "
              f"{'OVERTURNED' if beats else 'stands'}", flush=True)

    # overall verdict
    any_beat = any(v["beats_dense"] for v in results["by_ring"].values())
    results["verdict"] = ("OVERTURNED at some ring size" if any_beat else
                          "STANDS: mixing-width lever exhausted (eps>gain at all S)")
    print("=" * 64, flush=True)
    print(f"RINGSIZE VERDICT: {results['verdict']}", flush=True)
    for S, v in results["by_ring"].items():
        print(f"  S={S}: ring-uniform={v['uniform']['ppl']:.4f} "
              f"ring-measured={v['measured']['ppl']:.4f} "
              f"eps={v['eps_vs_dense']:+.4f} gain={v['alloc_gain']:+.4f} "
              f"net_vs_dense={v['net_vs_dense']:+.4f}", flush=True)
    print("=" * 64, flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {args.out}", flush=True)
    print("RINGSIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
