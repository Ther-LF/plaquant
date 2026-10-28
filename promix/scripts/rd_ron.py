"""Task 6 — R-ON INTEGRATION: measured-MSE allocation on the uniform 64-ring
block-diagonal R1 (spec 2026-10-28-nested-r1-per-layer-allocation-design §3).

Where rd_validity.py ran everything R1-OFF (identity main rotation, T = U), this
driver builds the SHIPPED R-on substrate with the ring rotation: T = U @ ring_R1
(fixed Hadamard per 64-ring, block-diagonal over all of hidden_dim). Only the
hidden-dim residual-stream rotation changes; per-head R2 / value / key rotations
and the down_proj online Hadamard are untouched. It reuses rd_validity's proven
build + collect + allocate + eval machinery (build_base / build_eval now take a
`ring_r1` kwarg) so the ONLY thing that varies vs the shipped path is the ring
rotation and WHICH 64-rings each Linear routes to FP8.

It answers the flip-lever question (rounds 11/13 concluded 4x that per-layer
allocation cannot beat R-on uniform — but always with allocation forced R1-OFF;
this is the first test with R preserved via the ring substrate):

  (i)   does ring-R-on + measured-MSE @ 1/8 beat dense-R-on uniform (3B 8.293)?
  (ii)  iso-accuracy (PPL <= dense-R-on-uniform) MINIMUM budget on ring-R-on.
  (iii) best PPL at budget 1/8 (iso-speed reading).

GATES (3B first):
  (a) bf16 bridge reproduces the floor (3B 7.82) — protocol sanity.
  (b) ring-rotation-only, NO quant == plain fp16 (orthogonal rotation lossless
      -> catches fusing bugs).
  (c) quantized uniform-1/8 with ring-R1 lands between R1-off (8.3924) and
      dense-R-on (8.293) — report the ring-vs-dense rotation cost honestly.

COMPARATORS @ matched 1/8 (same derive_khigh/tau machinery rd_validity used):
  uniform, variance-proxy, measured-MSE  -> re-verify the 3-way ranking under
  ring-R-on (the ring Hadamard mixes each 64-ring internally and changes
  per-ring outlier statistics, so the R1-off ranking may not carry over).

o_proj is held at uniform 1/8 in every config (controlled constant, as in
rd_validity).
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
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.models.loader import load_model

import rd_validity as rv

RING = rv.RING


def resolve_paths(config):
    """Model-aware artifact + local-model resolution (generalizes
    rd_validity.resolve_paths, which was 3B-only, to 1B/3B/8B).

    Forces the EXISTING, proven per-head PCA basis + R-on rotation artifacts
    and the local model dir for whichever Llama size the config names, so the
    same R-on substrate the shipped uniform path uses is what we build on.
    """
    name = config["model"]["name"]
    lname = name.lower()
    if "3b" in lname:
        short, local = "Llama-3.2-3B", "/home/luofan/models/Llama-3.2-3B"
    elif "8b" in lname:
        short, local = "llama-3-8b", "/home/luofan/models/llama-3-8b"
    elif "1b" in lname:
        # artifacts for 1B are the Instruct variant; no local dir (HF cache).
        short, local = "Llama-3.2-1B-Instruct", "/home/luofan/models/Llama-3.2-1B-Instruct"
    else:
        short, local = os.path.basename(name), None
    if local and os.path.isdir(local):
        config["model"]["name"] = local
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{short}.bin")
    config["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{short}.bin")
    config["quantize"]["o_proj_pca"] = "per_head"   # matches the existing basis
    config["quantize"]["high_fraction"] = 0.125
    return config


def fp16_bridge(config):
    """Plain fp16 baseline (no rotation, no quant) — the lossless-rotation
    reference for gate (b). Loaded in the SAME dtype build_base uses so the
    comparison isolates the rotation, not an fp16-vs-bf16 dtype gap."""
    model = load_model(config["model"]["name"], dtype=torch.float16).to(DEV)
    return rv.eval_ppl(model, config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--dense_ron_uniform_ref", type=float, default=8.293,
                    help="published dense-R-on uniform-1/8 floor for this model")
    # budget grid (as fractions). Priority order: downward iso-accuracy hunt
    # first (1/12,1/16,1/24,1/32), then mission's 3/16, then the UPPER wing
    # (1/4,3/8,1/2,1.0 = spend-latency-buy-accuracy side, to see the knee of
    # the final publishable R-on curve). 1/8 is measured in the @1/8 block and
    # merged into the curve, so it is not repeated here.
    ap.add_argument("--budgets",
                    default="0.0833333,0.0625,0.0416667,0.03125,0.1875,0.25,0.375,0.5,1.0")
    ap.add_argument("--out", default="/home/luofan/RD_RON.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    budgets = [float(x) for x in args.budgets.split(",")]
    print(f"START rd_ron model={config['model']['name']} "
          f"basis={os.path.basename(config['paths']['basis'])} budgets={budgets}",
          flush=True)

    results = {"meta": dict(model=config["model"]["name"],
                            basis=os.path.basename(config["paths"]["basis"]),
                            dense_ron_uniform_ref=args.dense_ron_uniform_ref,
                            budgets=budgets)}

    # --- bf16 / fp16 bridges (protocol sanity) -------------------------------
    t0 = time.time()
    ppl_bf16 = rv.bf16_bridge(config)
    print(f"BF16_BRIDGE ppl={ppl_bf16:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["bf16_bridge"] = ppl_bf16
    cleanup_memory()

    t0 = time.time()
    ppl_fp16 = fp16_bridge(config)
    print(f"FP16_BRIDGE ppl={ppl_fp16:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["fp16_bridge"] = ppl_fp16
    cleanup_memory()

    # --- GATE (b): ring-rotation-only, NO quant (lossless probe) -------------
    t0 = time.time()
    mdl = rv.build_base(config, r1_off=False, ring_r1=True)  # no configure => no quant
    mdl.to(DEV)
    ppl_ring_noquant = rv.eval_ppl(mdl, config)
    del mdl
    cleanup_memory()
    gate_b_delta = ppl_ring_noquant - ppl_fp16
    print(f"GATE_B ring_noquant ppl={ppl_ring_noquant:.4f} "
          f"(fp16={ppl_fp16:.4f} delta={gate_b_delta:+.4f}) ({time.time()-t0:.0f}s)",
          flush=True)
    results["gate_b_ring_noquant"] = dict(ppl=ppl_ring_noquant, fp16=ppl_fp16,
                                          delta=gate_b_delta,
                                          lossless=abs(gate_b_delta) < 0.05)

    # --- collect ring-R-on rotated activation stats (per-ring deltaD) --------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rv.build_base(config, r1_off=False, ring_r1=True)
    cmodel.to(DEV)
    stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()

    total_rings = sum(stats[n]["nr"] for n in stats)
    uni = rv.uniform_rings(stats)
    budget_rings_18 = sum(uni.values())      # exact matched 1/8 budget
    print(f"ALLOC units={len(stats)} total_rings={total_rings} "
          f"budget_fp8_rings@1/8={budget_rings_18} "
          f"(frac={budget_rings_18/total_rings:.4f})", flush=True)
    results["alloc_meta"] = dict(total_rings=total_rings,
                                 budget_rings_18=budget_rings_18)

    def run(tag, chosen, ring_r1):
        t = time.time()
        model = rv.build_eval(config, r1_off=False, chosen=chosen, ring_r1=ring_r1)
        frac, hi, tot = rv.avg_fp8_fraction(model)
        ppl = rv.eval_ppl(model, config)
        dt = time.time() - t
        print(f"RESULT {tag} ppl={ppl:.4f} avg_fp8={frac:.4f} "
              f"({hi}/{tot} ch) ({dt:.0f}s)", flush=True)
        del model
        cleanup_memory()
        return dict(ppl=ppl, avg_fp8=frac, hi_ch=hi, tot_ch=tot)

    # --- @1/8 comparators: dense-R-on uniform, ring-R-on {uniform, variance,
    #     measured} — the 3-way ranking re-verification + ring-vs-dense cost ---
    results["at_1_8"] = {}
    results["at_1_8"]["dense_ron_uniform"] = run("dense_ron_uniform", None, ring_r1=False)
    results["at_1_8"]["ring_ron_uniform"] = run("ring_ron_uniform", None, ring_r1=True)   # GATE (c)

    chosen_var, tau = rv.variance_alloc(stats, budget_rings_18)
    chosen_mse_18 = rv.measured_alloc(stats, budget_rings_18, total_rings)
    results["at_1_8"]["ring_ron_variance"] = run("ring_ron_variance", chosen_var, ring_r1=True)
    results["at_1_8"]["ring_ron_measured"] = run("ring_ron_measured", chosen_mse_18, ring_r1=True)
    results["at_1_8"]["variance_tau"] = tau

    # --- measured-MSE budget sweep (ring-R-on) -------------------------------
    sweep = {}
    for frac in budgets:
        br = int(round(frac * total_rings))
        chosen = rv.measured_alloc(stats, br, total_rings)
        n_sel = sum(len(v) for v in chosen.values())
        tag = f"measured_b{frac:.4f}"
        r = run(tag, chosen, ring_r1=True)
        r["budget_frac"] = frac
        r["budget_rings"] = br
        r["sel_rings"] = n_sel
        sweep[f"{frac:.5f}"] = r
    # merge the 1/8 measured point (computed in the @1/8 block) into the curve
    # so the iso-accuracy hunt and the Pareto table see a complete sweep.
    m18 = dict(results["at_1_8"]["ring_ron_measured"])
    m18.update(budget_frac=0.125, budget_rings=budget_rings_18,
               sel_rings=sum(len(v) for v in chosen_mse_18.values()))
    sweep[f"{0.125:.5f}"] = m18
    results["measured_sweep"] = sweep
    all_budgets = sorted(set(budgets) | {0.125})

    # --- three key answers ---------------------------------------------------
    dense_uni = results["at_1_8"]["dense_ron_uniform"]["ppl"]
    ring_uni = results["at_1_8"]["ring_ron_uniform"]["ppl"]
    ring_var = results["at_1_8"]["ring_ron_variance"]["ppl"]
    ring_mse_18 = results["at_1_8"]["ring_ron_measured"]["ppl"]
    floor = args.dense_ron_uniform_ref
    ring_vs_dense_cost = ring_uni - dense_uni

    # (i) flip lever: does ring-R-on measured @1/8 beat dense-R-on uniform?
    beats_published = ring_mse_18 < floor
    beats_measured = ring_mse_18 < dense_uni

    # (ii) iso-accuracy minimum budget on measured sweep (PPL <= floor)
    iso_budget = None
    for frac in sorted(all_budgets):   # ascending -> smallest feasible first
        key = f"{frac:.5f}"
        if sweep[key]["ppl"] <= floor:
            iso_budget = frac
            break

    # (iii) best PPL at budget 1/8 (iso-speed): min over {uniform,variance,measured}@1/8
    best_at_18 = min(ring_uni, ring_var, ring_mse_18)

    # 3-way ranking @1/8 (measured vs variance vs uniform under ring-R-on)
    ranking = sorted([("uniform", ring_uni), ("variance", ring_var),
                      ("measured", ring_mse_18)], key=lambda t: t[1])

    answers = dict(
        dense_ron_uniform_measured=dense_uni,
        dense_ron_uniform_published=floor,
        ring_ron_uniform=ring_uni,
        ring_vs_dense_rotation_cost=ring_vs_dense_cost,
        ring_ron_measured_at_1_8=ring_mse_18,
        q_i_flip_lever=dict(
            beats_published_floor=beats_published,
            beats_measured_dense_uniform=beats_measured,
            epsilon_vs_published=ring_mse_18 - floor,
            epsilon_vs_measured=ring_mse_18 - dense_uni,
            verdict=("OVERTURNED: ring-R-on+measured@1/8 beats dense-R-on uniform"
                     if beats_published else
                     "STANDS: ring-R-on+measured@1/8 does not beat dense-R-on uniform; "
                     "flip lever exhausted")),
        q_ii_iso_accuracy_min_budget=iso_budget,
        q_iii_best_ppl_at_1_8=best_at_18,
        ranking_at_1_8=ranking,
    )
    results["answers"] = answers

    print("=" * 70, flush=True)
    print(f"ANSWERS bf16={ppl_bf16:.3f} fp16={ppl_fp16:.3f}", flush=True)
    print(f"  dense-R-on uniform (measured)={dense_uni:.4f} "
          f"(published {floor})", flush=True)
    print(f"  ring-R-on uniform={ring_uni:.4f}  "
          f"ring-vs-dense cost={ring_vs_dense_cost:+.4f}", flush=True)
    print(f"  ring-R-on @1/8: uniform={ring_uni:.4f} variance={ring_var:.4f} "
          f"measured={ring_mse_18:.4f}", flush=True)
    print(f"  ranking@1/8 (best->worst): "
          f"{' < '.join(f'{k}={v:.4f}' for k,v in ranking)}", flush=True)
    print(f"  (i)   flip lever: measured@1/8={ring_mse_18:.4f} vs floor {floor} "
          f"-> {answers['q_i_flip_lever']['verdict']}", flush=True)
    print(f"  (ii)  iso-accuracy min budget (PPL<={floor}): {iso_budget}", flush=True)
    print(f"  (iii) best PPL @1/8: {best_at_18:.4f}", flush=True)
    print("=" * 70, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {args.out}", flush=True)
    print("RD_RON_DONE", flush=True)


if __name__ == "__main__":
    main()
