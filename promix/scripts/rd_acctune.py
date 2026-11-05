"""rd-acctune — ACCURACY TUNING CAMPAIGN single-point driver.

Runs ONE named configuration per invocation (so process-level env knobs —
PROMIX_ACT_MSE_STEPS[_UP], PROMIX_WEIGHT_MSE_STEPS — isolate cleanly) and
writes one JSON with wikitext (+optional C4) PPL. Reuses the committed
rd_validity build/collect/alloc/eval machinery and rd_ron.resolve_paths, so a
default invocation reproduces the shipped protocol digit-for-digit
(bridges 1B 13.1569 / 3B 7.8148 / 8B 6.1368; dense uniform 14.5483 / 8.2974 /
6.5882; dense measured 14.4717 / 8.2678 / 6.6072).

Levers exposed (mission docs/rd_accuracy_tuning.md):
  1. --substrate ring --ring_size 512   (T = U @ ring_R1, fixed Hadamard rings)
  2. --alloc var_uniform | dd_uniform   (per-Linear uniform ring COUNT, ring
     SELECTION by measured ring variance / deltaD — the rd_whydecay explicit
     ring-set that beat the eigenvalue-tail uniform by 0.12 on 1B)
  3. --basis_nsamples N --gptq_nsamples N --collect_tokens T  (calibration size)
  4. env PROMIX_ACT_MSE_STEPS / PROMIX_ACT_MSE_STEPS_UP      (act scale search)
  5. env PROMIX_WEIGHT_MSE_STEPS + --percdamp --act_order    (weight/GPTQ)

Examples:
  python promix/scripts/rd_acctune.py --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
      --point 3b_ring512_measured --substrate ring --ring_size 512 --alloc measured \
      --out /home/luofan/acctune/P_3b_ring512_measured.json
  PROMIX_ACT_MSE_STEPS_UP=2 python promix/scripts/rd_acctune.py ... --alloc measured
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
from promix.eval.data import get_wikitext2, get_c4
from promix.quantize import quant_utils as qu
from promix.models.loader import load_model

import rd_validity as rv
import rd_ron


def per_linear_uniform_chosen(stats, metric):
    """Per-Linear uniform ring COUNT (k = round(nr/8)), ring SELECTION by the
    measured per-ring statistic (`ring_var` or `deltaD`). This is the
    rd_whydecay `uniform_chosen` explicit ring-set, generalized to both
    selection metrics; it goes through the same reorder_before_quant path as
    the measured allocation (apples-to-apples vs the positional tail)."""
    ch = {}
    for n, st in stats.items():
        k = int(round(0.125 * st["nr"]))
        v = torch.as_tensor(st[metric])
        ch[n] = set(torch.topk(v, k).indices.tolist()) if k > 0 else set()
    return ch


def eval_c4_ppl(model, config, eval_seed=None):
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    if eval_seed is None:
        eval_seed = config["calibration"].get("seed", 0)
    c4 = get_c4(nsamples=256, seed=eval_seed,
                seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    return float(evaluate_ppl(model, c4, DEV,
                              batch_size=config["eval"]["batch_size"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--point", required=True, help="unique point tag")
    ap.add_argument("--substrate", choices=["dense", "ring"], default="dense")
    ap.add_argument("--ring_size", type=int, default=64,
                    help="ring substrate width (and the allocation atom)")
    ap.add_argument("--alloc", default="measured",
                    choices=["uniform", "measured", "var_uniform", "dd_uniform",
                             "none"],
                    help="uniform = shipped positional tail (chosen=None); "
                         "measured = global deltaD threshold; var_uniform / "
                         "dd_uniform = per-Linear uniform count + measured ring "
                         "selection; none = bf16 bridge only")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    # rd-densesweep: budget knob for the measured allocation (default = the
    # shipped 1/8; any other value sweeps the accuracy/bits curve). High
    # budgets are clamped to >=1 NVFP4 ring per Linear (rv.clamp_chosen).
    ap.add_argument("--budget_frac", type=float, default=0.125)
    ap.add_argument("--stats_cache", default=None,
                    help="optional path to cache the collected ring stats "
                         "(identical across budget_frac points of one model/"
                         "substrate — saves the collect pass per point)")
    ap.add_argument("--basis_nsamples", type=int, default=None,
                    help="override PCA basis to U-wikitext-<N>-<short>.bin "
                         "(lever 3; keep the shipped R)")
    ap.add_argument("--gptq_nsamples", type=int, default=None,
                    help="override calibration.nsamples for GPTQ (lever 3)")
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--act_order", action="store_true")
    ap.add_argument("--calib_seed", type=int, default=None,
                    help="E6 error bars: override the CALIBRATION seed "
                         "(set_seed + calib sampling + GPTQ loader + alloc "
                         "collect). Eval sets stay at the config seed (wiki "
                         "eval is the full test set; C4 eval windows pinned "
                         "to the original seed) so only calibration varies.")
    ap.add_argument("--c4", action="store_true", help="also eval C4 PPL")
    ap.add_argument("--kv4", action="store_true",
                    help="rd-kv4: quantize the KV cache too (k_bits/v_bits = "
                         "nvfp4, per-head grouping, MXFP8-aligned high "
                         "segment). Off = byte-identical to the acctune "
                         "anchor runs.")
    ap.add_argument("--kv_high", choices=["up", "zero"], default="up",
                    help="FP KV per-head high-segment policy (with --kv4): "
                         "'up' rounds int(head_dim*hf) UP to 32 (option a); "
                         "'zero' drops the KV high segment -> pure-NVFP4 KV "
                         "(option b ablation)")
    ap.add_argument("--bridge", action="store_true",
                    help="also run the bf16 bridge (protocol gate)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    if args.basis_nsamples is not None:
        rotdir = os.path.dirname(config["paths"]["rotation"])
        short = os.path.basename(config["paths"]["basis"])
        short = short[len("U-wikitext-512-"):-len(".bin")]
        config["paths"]["basis"] = os.path.join(
            rotdir, f"U-wikitext-{args.basis_nsamples}-{short}.bin")
    if args.gptq_nsamples is not None:
        config["calibration"]["nsamples"] = args.gptq_nsamples
    if args.percdamp is not None:
        config["quantize"]["percdamp"] = args.percdamp
    if args.act_order:
        config["quantize"]["act_order"] = True
    if args.kv4:
        # rd-kv4 lever-6 fix: enable FP KV-cache quantization on top of any
        # acctune stack. Any positive k_groupsize selects the per-head K
        # path in QKRotationWrapper (the reshape uses the model's real
        # head_dim); v grouping always uses head_dim. kv_fp_high is consumed
        # by kv_fp_high_length (quant_utils) in configure_quantizers +
        # setup_k_quant.
        config["quantize"].update(
            k_bits="nvfp4", v_bits="nvfp4", k_asym=True, v_asym=True,
            k_groupsize=64, v_groupsize=64, kv_fp_high=args.kv_high,
        )
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    eval_seed = config["calibration"].get("seed", 0)
    if args.calib_seed is not None:
        config["calibration"]["seed"] = args.calib_seed
    transformers.set_seed(config["calibration"].get("seed", 0))

    S = args.ring_size
    ring_r1 = args.substrate == "ring"
    meta = dict(point=args.point, model=config["model"]["name"],
                calib_seed=config["calibration"].get("seed", 0),
                basis=os.path.basename(config["paths"]["basis"]),
                substrate=args.substrate, ring_size=S, alloc=args.alloc,
                budget_frac=args.budget_frac,
                collect_tokens=args.collect_tokens,
                gptq_nsamples=config["calibration"].get("nsamples", 128),
                percdamp=config["quantize"].get("percdamp", 0.01),
                act_order=config["quantize"].get("act_order", False),
                act_mse_steps=qu.ACT_MSE_STEPS,
                act_mse_steps_up=qu.ACT_MSE_STEPS_UP,
                weight_mse_steps=qu._MSE_STEPS,
                kv_bits=(config["quantize"].get("k_bits"),
                         config["quantize"].get("v_bits")),
                kv_fp_high=config["quantize"].get("kv_fp_high"))
    results = {"meta": meta}
    print(f"START rd_acctune {json.dumps(meta)}", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.bridge:
        t0 = time.time()
        ppl = rv.bf16_bridge(config)
        print(f"BF16_BRIDGE ppl={ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["bf16_bridge"] = ppl
        cleanup_memory()
        save()
    if args.alloc == "none":
        save()
        print("ACCTUNE_DONE", flush=True)
        return

    # --- allocation ----------------------------------------------------------
    chosen = None
    if args.alloc != "uniform":
        if args.stats_cache and os.path.exists(args.stats_cache):
            stats = torch.load(args.stats_cache, weights_only=False)
            print(f"STATS_CACHE loaded {args.stats_cache}", flush=True)
        else:
            tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
            calib = get_wikitext2(nsamples=args.calib_samples,
                                  seed=config["calibration"].get("seed", 0),
                                  seqlen=2048, tokenizer=tok, eval_mode=False)
            cmodel = rv.build_base(config, r1_off=False, ring_r1=ring_r1,
                                   ring_size=S)
            cmodel.to(DEV)
            stats = rv.collect_stats(cmodel, calib,
                                     max_tokens=args.collect_tokens, ring=S)
            del cmodel
            cleanup_memory()
            if args.stats_cache:
                tmp = args.stats_cache + f".tmp{os.getpid()}"
                torch.save(stats, tmp)
                os.replace(tmp, args.stats_cache)
                print(f"STATS_CACHE saved {args.stats_cache}", flush=True)
        total_rings = sum(stats[n]["nr"] for n in stats)
        if args.alloc == "measured":
            budget = int(round(args.budget_frac * total_rings))
            chosen = rv.clamp_chosen(
                rv.measured_alloc(stats, budget, total_rings), stats)
        elif args.alloc == "var_uniform":
            chosen = per_linear_uniform_chosen(stats, "ring_var")
        elif args.alloc == "dd_uniform":
            chosen = per_linear_uniform_chosen(stats, "deltaD")
        nsel = sum(len(v) for v in chosen.values())
        print(f"ALLOC total_rings={total_rings} sel={nsel} "
              f"(frac={nsel/total_rings:.4f})", flush=True)
        results["alloc_meta"] = dict(total_rings=total_rings, sel_rings=nsel)

    # --- build + eval --------------------------------------------------------
    t0 = time.time()
    model = rv.build_eval(config, r1_off=False, chosen=chosen,
                          ring_r1=ring_r1, ring_size=S)
    frac, hi, tot = rv.avg_fp8_fraction(model)
    print(f"BUILT avg_fp8={frac:.4f} ({hi}/{tot} ch) "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["avg_fp8"] = frac

    t0 = time.time()
    ppl_wiki = rv.eval_ppl(model, config)
    print(f"RESULT {args.point} wikitext={ppl_wiki:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["wikitext"] = ppl_wiki
    save()

    if args.c4:
        t0 = time.time()
        ppl_c4 = eval_c4_ppl(model, config, eval_seed=eval_seed)
        print(f"RESULT {args.point} c4={ppl_c4:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["c4"] = ppl_c4
        save()

    del model
    cleanup_memory()
    save()
    print(f"ACCTUNE_RESULT point={args.point} wiki={ppl_wiki:.4f} "
          f"c4={results.get('c4')}", flush=True)
    print("ACCTUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
