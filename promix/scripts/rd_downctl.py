"""rd-downctl — the explicit FORCED-FP8-on-down_proj control for the
"all-FP4 down_proj is measured-OPTIMAL, not a convenience" claim.

Context (docs/rd_whydecay_report.md §5b): down_proj is a deltaD desert — it
holds 35-44 % of all rings but <1 % of total deltaD, because its input passes
through the online Hadamard which flattens outliers. The measured-MSE
allocator strips down_proj to ZERO FP8 rings and reinvests the freed budget in
the attn/mlp streams, and that config is the committed measured-1/8 anchor.
What was missing is the explicit control where down_proj is FORCED to carry
FP8. Three points, same substrate as the committed anchor (dense-R-on,
ring=64, o_proj held uniform 1/8, matched protocol = rd_validity build path):

  anchor   : measured-MSE global-deltaD-threshold allocation at avg FP8 = 1/8
             (down_proj gets ~0 FP8 naturally). Must reproduce the committed
             3B numbers: wikitext 8.2678 / C4 11.3194 (docs/rd_metrics_report.md).
  controlA : BUDGET-MATCHED — down_proj FORCED to per-Linear uniform-1/8 FP8
             (its k best rings by measured deltaD, i.e. the most charitable
             selection for down_proj), and the equivalent ring budget REMOVED
             from the attn/mlp allocation (global-deltaD threshold over the
             remaining 5 Linears at the reduced budget). Global avg stays 1/8.
             Expectation: WORSE than anchor (budget wasted on the desert).
  controlB : ADDITIVE — down_proj forced to uniform-1/8 FP8 ON TOP of the full
             measured allocation (global budget rises above 1/8).
             Expectation: ~= anchor — the extra FP8 on down_proj buys ~nothing
             even for free.

Protocol bridges (must reproduce): bf16 3B wikitext 7.8148; anchor as above.

Run (inside docker luofan-plaquant, GPU 6/7 lane):
  python promix/scripts/rd_downctl.py \
      --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
      --c4 --out /home/luofan/downctl/RD_DOWNCTL_3b.json
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

from promix.utils import cleanup_memory
from promix.eval import ptq as ptqmod
import rd_validity as rv
import rd_ron
from rd_acctune import eval_c4_ppl


def is_down(name):
    return name.split(".")[-1] == "down_proj"


def forced_down_set(stats):
    """Per-down_proj uniform ring COUNT (k = round(nr/8)), ring SELECTION by
    measured deltaD (the most charitable pick for down_proj)."""
    ch = {}
    for n, st in stats.items():
        if not is_down(n):
            continue
        k = int(round(0.125 * st["nr"]))
        v = torch.as_tensor(st["deltaD"])
        ch[n] = set(torch.topk(v, k).indices.tolist()) if k > 0 else set()
    return ch


def group_counts(chosen, stats):
    """(down_fp8_rings, down_rings, rest_fp8_rings, rest_rings)."""
    dh = dt = rh = rt = 0
    for n, st in stats.items():
        sel = len(chosen.get(n, set()))
        if is_down(n):
            dh, dt = dh + sel, dt + st["nr"]
        else:
            rh, rt = rh + sel, rt + st["nr"]
    return dh, dt, rh, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--points", default="anchor,controlA,controlB")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    results = {"meta": dict(model=config["model"]["name"],
                            basis=os.path.basename(config["paths"]["basis"]),
                            substrate="dense", ring_size=64,
                            collect_tokens=args.collect_tokens,
                            points=args.points)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"START rd_downctl {json.dumps(results['meta'])}", flush=True)

    # --- protocol bridge: bf16 (3B must reproduce 7.8148) ---------------------
    t0 = time.time()
    ppl_bf16 = rv.bf16_bridge(config)
    print(f"BF16_BRIDGE ppl={ppl_bf16:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["bf16_bridge"] = ppl_bf16
    cleanup_memory()
    save()

    # --- collect dense-R-on rotated activation stats (once) -------------------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rv.build_base(config, r1_off=False)
    from promix.utils import DEV
    cmodel.to(DEV)
    stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()

    total_rings = sum(stats[n]["nr"] for n in stats)
    budget = int(round(0.125 * total_rings))

    # --- the three allocations -------------------------------------------------
    anchor = rv.measured_alloc(stats, budget, total_rings)
    forced = forced_down_set(stats)
    forced_n = sum(len(v) for v in forced.values())

    # controlA: forced down + global-deltaD over the REST at the reduced budget
    rest_stats = {n: st for n, st in stats.items() if not is_down(n)}
    rest_total = sum(st["nr"] for st in rest_stats.values())
    remaining = budget - forced_n
    assert remaining > 0
    restA = rv.measured_alloc(rest_stats, remaining, rest_total)
    controlA = dict(restA)
    controlA.update(forced)

    # controlB: full measured allocation + forced down ON TOP (budget rises)
    controlB = {n: set(s) for n, s in anchor.items()}
    for n, s in forced.items():
        controlB[n] = controlB.get(n, set()) | s

    alloc_meta = {}
    for tag, chosen in (("anchor", anchor), ("controlA", controlA),
                        ("controlB", controlB)):
        dh, dt, rh, rt = group_counts(chosen, stats)
        tot_sel = dh + rh
        alloc_meta[tag] = dict(
            fp8_rings=tot_sel, total_rings=total_rings,
            global_frac=tot_sel / total_rings,
            down_fp8_rings=dh, down_rings=dt, down_frac=dh / dt,
            rest_fp8_rings=rh, rest_rings=rt, rest_frac=rh / rt)
        print(f"ALLOC {tag}: fp8={tot_sel}/{total_rings} "
              f"(global {tot_sel/total_rings:.4f}) | down {dh}/{dt} "
              f"({dh/dt:.4f}) | rest {rh}/{rt} ({rh/rt:.4f})", flush=True)
    results["alloc"] = alloc_meta
    results["alloc"]["budget_rings"] = budget
    results["alloc"]["forced_down_rings"] = forced_n
    save()

    # --- build + eval each point ------------------------------------------------
    todo = [t.strip() for t in args.points.split(",") if t.strip()]
    configs = {"anchor": anchor, "controlA": controlA, "controlB": controlB}
    for tag in todo:
        chosen = configs[tag]
        t0 = time.time()
        model = rv.build_eval(config, r1_off=False, chosen=chosen)
        frac, hi, tot = rv.avg_fp8_fraction(model)
        ppl_wiki = rv.eval_ppl(model, config)
        row = dict(wikitext=ppl_wiki, avg_fp8=frac, hi_ch=hi, tot_ch=tot)
        print(f"RESULT {tag} wikitext={ppl_wiki:.4f} avg_fp8={frac:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if args.c4:
            t0 = time.time()
            ppl_c4 = eval_c4_ppl(model, config)
            row["c4"] = ppl_c4
            print(f"RESULT {tag} c4={ppl_c4:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        results[tag] = row
        save()
        del model
        cleanup_memory()

    if all(t in results for t in ("anchor", "controlA", "controlB")):
        a = results["anchor"]["wikitext"]
        ca = results["controlA"]["wikitext"]
        cb = results["controlB"]["wikitext"]
        print("=" * 64, flush=True)
        print(f"DOWNCTL  bf16={ppl_bf16:.4f} | anchor={a:.4f}  "
              f"controlA={ca:.4f} (dA={ca-a:+.4f})  "
              f"controlB={cb:.4f} (dB={cb-a:+.4f})", flush=True)
        print(f"DOWNCTL  expect: dA > 0 (matched-budget cost), "
              f"|dB| small (free FP8 on down buys ~nothing)", flush=True)
        print("=" * 64, flush=True)
    save()
    print("RD_DOWNCTL_DONE", flush=True)


if __name__ == "__main__":
    main()
