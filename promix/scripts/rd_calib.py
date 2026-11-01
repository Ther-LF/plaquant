"""rd-calib: calibration-set robustness ablation for the rate-distortion
allocation pipeline (standard quant-paper ablation).

Worry being tested: ALL of our offline calibration artifacts — the Step-0 PCA
basis U/E, the GPTQ Hessians, and the measured-ΔD allocation stats — come from
wikitext-2 train windows. If the measured-vs-uniform advantage (and the
pipeline's accuracy in general) is calibration-set-specific, it evaporates on
other eval text.

Design (2x2 + isolation cell), dense-R-on shipping regime (ring=64, matched
avg FP8 = 1/8, o_proj uniform — the exact rd_metrics recipe):

  calib axis : wikitext-512 (committed artifacts, row already measured by
               rd_metrics → docs/rd_metrics/RD_METRICS2_*.json)
               vs C4-512 (THIS script: full offline pipeline re-calibrated on
               allenai/c4 train — Step-0 basis U-c4-512-*.bin, GPTQ on C4
               windows, ΔD collected on C4-rotated activations)
  config axis: uniform-1/8 vs measured-MSE-1/8 (does the ALLOCATION gain
               specifically survive calibration shift?)
  eval axis  : wikitext-2 PPL + C4 PPL (committed protocol, rd_metrics
               loaders) + optional 0-shot AVG spot check.

Isolation cell (--basis_calib wikitext --gptq_calib wikitext --alloc_calib c4,
--configs measured): ONLY the ΔD measurement distribution is swapped, holding
basis + GPTQ at wikitext — separates allocation sensitivity from basis/GPTQ
sensitivity.

Ring-overlap mode (--mode overlap): same (wikitext-basis) base model, collect
per-ring ΔD stats under BOTH calib sets → two measured allocations in the SAME
rotated coordinates → ring-set overlap % per unit + global (the "is ΔD itself
calib-stable" statistic; random-baseline overlap = 12.5%).

The three calibration entry points are controlled independently:
  --basis_calib : which U-{calib}-512-{short}.bin the model is built on
                  (C4 basis must be generated first via
                   python -m promix.quantize.basis ... --calib_dataset c4;
                   the R file is REUSED (seed-fixed random) so rotation is a
                   controlled constant — only U/E change)
  --gptq_calib  : GPTQ Hessian calibration text. run_gptq_if_enabled
                  late-imports promix.eval.data.get_wikitext2, so a scoped
                  module-attribute patch reroutes ONLY the GPTQ trainloader.
  --alloc_calib : the 16x2048 ΔD collection text for measured_alloc.

Act-MSE / weight-MSE scale search are runtime searches (no offline calib
artifact), so they need no swapping. Reuses rd_validity's proven dense-R-on
build/alloc machinery + rd_metrics' eval suite unchanged.
"""
import argparse
import contextlib
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
from promix.eval.data import get_wikitext2, get_c4
import promix.eval.data as data_mod

import rd_validity as rv
import rd_ron
import rd_metrics as rm


def calib_loader(dataset, nsamples, seed, seqlen, tokenizer):
    """Train-mode calibration windows from the named dataset (same (inp, tar)
    tuple format from both loaders)."""
    fn = get_wikitext2 if dataset == "wikitext" else get_c4
    return fn(nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer,
              eval_mode=False)


@contextlib.contextmanager
def gptq_calib_dataset(dataset):
    """Scoped reroute of the GPTQ trainloader to C4.

    ptq.run_gptq_if_enabled late-imports promix.eval.data.get_wikitext2 at
    call time, so patching the module attribute inside this context affects
    exactly that import and nothing already bound (this script's eval loaders
    are bound at module import)."""
    if dataset == "wikitext":
        yield
        return
    orig = data_mod.get_wikitext2

    def _c4_trainloader(nsamples=128, seed=0, seqlen=2048, tokenizer=None,
                        eval_mode=False):
        assert not eval_mode, "gptq calib patch must only serve train mode"
        return get_c4(nsamples=nsamples, seed=seed, seqlen=seqlen,
                      tokenizer=tokenizer, eval_mode=False)

    data_mod.get_wikitext2 = _c4_trainloader
    try:
        yield
    finally:
        data_mod.get_wikitext2 = orig


def resolve_calib_paths(config, basis_calib):
    """rd_ron model-aware path resolve, then swap the basis to the chosen
    calibration set's Step-0 artifact. Rotation R stays the wikitext-era file
    (it is seed-fixed RANDOM, not data-fit → controlled constant)."""
    config = rd_ron.resolve_paths(config)
    if basis_calib != "wikitext":
        b = config["paths"]["basis"]
        nb = b.replace("U-wikitext-", f"U-{basis_calib}-")
        assert nb != b, f"basis path has no wikitext tag to swap: {b}"
        config["paths"]["basis"] = nb
    return config


def chosen_to_json(chosen):
    return {n: sorted(int(r) for r in s) for n, s in chosen.items()}


def ring_overlap(chosen_a, chosen_b):
    """Per-unit and global overlap of two {name: set(ring idx)} allocations."""
    per_unit = {}
    inter_tot = na_tot = 0
    for n in sorted(set(chosen_a) | set(chosen_b)):
        a, b = chosen_a.get(n, set()), chosen_b.get(n, set())
        inter = len(a & b)
        per_unit[n] = dict(n_a=len(a), n_b=len(b), inter=inter,
                           frac=inter / max(len(a), 1))
        inter_tot += inter
        na_tot += len(a)
    return per_unit, inter_tot / max(na_tot, 1)


def deltaD_spearman(stats_a, stats_b):
    """Mean per-unit Spearman rank correlation of the two ΔD vectors."""
    rhos = {}
    for n in stats_a:
        da, db = stats_a[n]["deltaD"].float(), stats_b[n]["deltaD"].float()
        ra = torch.argsort(torch.argsort(da)).float()
        rb = torch.argsort(torch.argsort(db)).float()
        ra = (ra - ra.mean()) / ra.std()
        rb = (rb - rb.mean()) / rb.std()
        rhos[n] = float((ra * rb).mean())
    return rhos, sum(rhos.values()) / max(len(rhos), 1)


def measured_from_stats(stats):
    total_rings = sum(stats[n]["nr"] for n in stats)
    budget = int(round(0.125 * total_rings))
    return rv.measured_alloc(stats, budget, total_rings), budget, total_rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["run", "overlap"], default="run")
    ap.add_argument("--basis_calib", choices=["wikitext", "c4"],
                    default="wikitext")
    ap.add_argument("--gptq_calib", choices=["wikitext", "c4"],
                    default="wikitext")
    ap.add_argument("--alloc_calib", choices=["wikitext", "c4"],
                    default="wikitext")
    ap.add_argument("--configs", default="uniform,measured",
                    help="comma subset of uniform,measured (run mode)")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--ring_size", type=int, default=64)
    ap.add_argument("--zs_configs", default="",
                    help="comma subset of configs to also run 0-shot on")
    ap.add_argument("--zs_limit", type=int, default=None)
    ap.add_argument("--no_c4_eval", action="store_true")
    ap.add_argument("--out", default="/home/luofan/RD_CALIB.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_calib_paths(config, args.basis_calib)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    seed = config["calibration"].get("seed", 0)
    transformers.set_seed(seed)
    S = args.ring_size
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])

    results = {"meta": dict(
        model=config["model"]["name"],
        basis=os.path.basename(config["paths"]["basis"]),
        rotation=os.path.basename(config["paths"]["rotation"]),
        ring_size=S, mode=args.mode,
        basis_calib=args.basis_calib, gptq_calib=args.gptq_calib,
        alloc_calib=args.alloc_calib,
        regime="dense-R-on (ring_r1=False), matched avg FP8=0.125",
    )}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"START rd_calib mode={args.mode} basis={args.basis_calib} "
          f"gptq={args.gptq_calib} alloc={args.alloc_calib} "
          f"basis_file={results['meta']['basis']}", flush=True)

    # ---------------- overlap mode: ΔD stability in SAME coordinates --------
    if args.mode == "overlap":
        cmodel = rv.build_base(config, r1_off=False, ring_r1=False,
                               ring_size=S)
        cmodel.to(DEV)
        stats = {}
        for ds in ("wikitext", "c4"):
            t = time.time()
            calib = calib_loader(ds, args.calib_samples, seed, 2048, tok)
            stats[ds] = rv.collect_stats(cmodel, calib,
                                         max_tokens=args.collect_tokens,
                                         ring=S)
            print(f"[overlap] collected {ds} stats "
                  f"({time.time()-t:.0f}s)", flush=True)
        del cmodel
        cleanup_memory()

        chosen, budgets = {}, {}
        for ds in ("wikitext", "c4"):
            chosen[ds], budget, total = measured_from_stats(stats[ds])
            budgets[ds] = dict(budget_rings=budget, total_rings=total)
        per_unit, global_frac = ring_overlap(chosen["wikitext"], chosen["c4"])
        rhos, mean_rho = deltaD_spearman(stats["wikitext"], stats["c4"])
        results["budgets"] = budgets
        results["ring_overlap"] = dict(
            global_frac=global_frac, random_baseline=0.125,
            per_unit={n: v for n, v in per_unit.items()})
        results["deltaD_spearman"] = dict(mean=mean_rho, per_unit=rhos)
        results["chosen"] = {ds: chosen_to_json(chosen[ds])
                             for ds in chosen}
        print(f"RING OVERLAP global={global_frac:.4f} "
              f"(random baseline 0.125)  deltaD spearman mean={mean_rho:.4f}",
              flush=True)
        save()
        print(f"WROTE {args.out}", flush=True)
        print("RD_CALIB_DONE", flush=True)
        return

    # ---------------- run mode: build + eval the requested cells ------------
    want = [c for c in args.configs.split(",") if c]
    zs_want = set(c for c in args.zs_configs.split(",") if c)

    if "uniform" in want:
        t = time.time()
        with gptq_calib_dataset(args.gptq_calib):
            m = rv.build_eval(config, r1_off=False, chosen=None,
                              ring_r1=False, ring_size=S)
        frac, hi, tot = rv.avg_fp8_fraction(m)
        print(f"[uniform] built avg_fp8={frac:.4f} ({hi}/{tot}) "
              f"({time.time()-t:.0f}s)", flush=True)
        r = rm.eval_all_metrics(m, config, tok, "uniform" in zs_want,
                                args.zs_limit, do_c4=not args.no_c4_eval)
        r["avg_fp8"] = frac
        results["uniform_1_8"] = r
        del m
        cleanup_memory()
        save()

    if "measured" in want:
        t = time.time()
        calib = calib_loader(args.alloc_calib, args.calib_samples, seed,
                             2048, tok)
        cmodel = rv.build_base(config, r1_off=False, ring_r1=False,
                               ring_size=S)
        cmodel.to(DEV)
        stats = rv.collect_stats(cmodel, calib,
                                 max_tokens=args.collect_tokens, ring=S)
        del cmodel
        cleanup_memory()
        chosen, budget, total = measured_from_stats(stats)
        results["chosen_measured"] = chosen_to_json(chosen)
        with gptq_calib_dataset(args.gptq_calib):
            m = rv.build_eval(config, r1_off=False, chosen=chosen,
                              ring_r1=False, ring_size=S)
        frac, hi, tot = rv.avg_fp8_fraction(m)
        print(f"[measured] built avg_fp8={frac:.4f} ({hi}/{tot}) "
              f"budget_rings={budget}/{total} ({time.time()-t:.0f}s)",
              flush=True)
        r = rm.eval_all_metrics(m, config, tok, "measured" in zs_want,
                                args.zs_limit, do_c4=not args.no_c4_eval)
        r["avg_fp8"] = frac
        results["measured_1_8"] = r
        del m
        cleanup_memory()
        save()

    save()
    print(f"WROTE {args.out}", flush=True)
    print("RD_CALIB_DONE", flush=True)


if __name__ == "__main__":
    main()
