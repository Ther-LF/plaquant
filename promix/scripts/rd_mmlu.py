"""rd-mmlu (paper gap E1): MMLU 5-shot + NEW-recipe 0-shot for the accuracy
main table.

Fills the two accuracy-metric gaps in docs/paper_experiments.md block 1:

  1. MMLU 5-shot (ARCQuant reports it) for {bf16, ProMix NEW recipe} on our
     models, via the SAME lm-eval version/config as the committed rd_metrics
     runs (lm-eval 0.4.5, HFLM, GPU-pinned).  ARCQuant/SharQ MMLU runs go
     through rd_newacc_peer.py --lmeval_tasks mmlu (peer venvs);
     vLLM-NVFP4 goes through lm-eval's vllm backend on the llm-compressor
     checkpoints (see docs/rd_mmlu_report.md).
  2. NEW-recipe 0-shot 6-task suite (the committed 0-shot numbers in
     rd_metrics_report.md are the OLD dense anchors; the paper anchors are the
     acctune ring512 recipes).

One point per invocation (process env knobs PROMIX_ACT_MSE_STEPS_UP /
PROMIX_WEIGHT_MSE_STEPS isolate cleanly, exactly like rd_acctune.py).  The
NEW-recipe build reuses the committed rd_validity/rd_ron machinery with the
rd_accuracy_tuning.md shipping recipes; the wikitext PPL printed after the
build is the reproduction gate (must match 1B 14.1694 / 3B 8.2180 /
8B 6.5642 digit-for-digit before the lm-eval numbers are trusted).

Examples (inside container, CUDA_VISIBLE_DEVICES pinned by the caller):

  # 3B NEW recipe: gate + MMLU 5-shot + 0-shot suite
  PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 \
  python promix/scripts/rd_mmlu.py \
      --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --point 3b_new \
      --substrate ring --ring_size 512 --alloc measured \
      --gptq_nsamples 512 --percdamp 0.025 --mmlu --zeroshot \
      --expect_wiki 8.2180 --out /home/luofan/mmlu/pm_3b.json

  # 3B bf16 MMLU anchor
  python promix/scripts/rd_mmlu.py --config ... --point 3b_bf16 \
      --alloc none --mmlu --out /home/luofan/mmlu/bf16_3b.json
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
from promix.quantize import quant_utils as qu

import rd_validity as rv
import rd_ron
import rd_metrics as rm

MMLU_GROUPS = ("mmlu", "mmlu_humanities", "mmlu_other",
               "mmlu_social_sciences", "mmlu_stem")


def make_lm(model, tokenizer, batch_size):
    """HFLM with the rd_metrics GPU-pinning fix (evaluate_ppl offloads layers
    to CPU on exit; the quantized models' online Hadamard is CUDA-only)."""
    from lm_eval.models.huggingface import HFLM
    import datasets as hf_datasets
    hf_datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    model.eval()
    model.to(DEV)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    lm._device = DEV
    try:
        lm.model.to(DEV)
    except Exception:
        pass
    return lm


def extract_mmlu(res):
    """Group accs (lm-eval's own aggregation = the headline number everyone
    reports) + all 57 subtask accs + our own macro/micro cross-checks."""
    out = {"groups": {}, "subtasks": {}}
    for k, v in res["results"].items():
        if k == "mmlu" or k.startswith("mmlu_"):
            acc = v.get("acc,none")
            if k in MMLU_GROUPS:
                out["groups"][k] = acc
            else:
                out["subtasks"][k] = acc
    subs = [a for a in out["subtasks"].values() if a is not None]
    out["macro_avg"] = (sum(subs) / len(subs)) if subs else None
    ns = res.get("n-samples") or {}
    wsum = wtot = 0.0
    for k, a in out["subtasks"].items():
        n = (ns.get(k) or {}).get("effective") or (ns.get(k) or {}).get(
            "original")
        if a is not None and n:
            wsum += a * n
            wtot += n
    out["micro_avg"] = (wsum / wtot) if wtot else None
    out["n_subtasks"] = len(out["subtasks"])
    return out


def eval_mmlu(model, tokenizer, batch_size, limit=None):
    from lm_eval import evaluator as lm_evaluator
    lm = make_lm(model, tokenizer, batch_size)
    res = lm_evaluator.simple_evaluate(
        lm, tasks=["mmlu"], num_fewshot=5, batch_size=batch_size, limit=limit)
    return extract_mmlu(res)


def build_newrecipe(config, args):
    """ring512 + measured-MSE (+ GPTQ/percdamp knobs from the caller; act/weight
    MSE knobs from process env) — the rd_acctune build path, verbatim."""
    S = args.ring_size
    ring_r1 = args.substrate == "ring"
    chosen = None
    if args.alloc == "measured":
        tok = transformers.AutoTokenizer.from_pretrained(
            config["model"]["name"])
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
        total_rings = sum(stats[n]["nr"] for n in stats)
        budget = int(round(0.125 * total_rings))
        chosen = rv.measured_alloc(stats, budget, total_rings)
        nsel = sum(len(v) for v in chosen.values())
        print(f"ALLOC total_rings={total_rings} sel={nsel} "
              f"(frac={nsel/total_rings:.4f})", flush=True)
    model = rv.build_eval(config, r1_off=False, chosen=chosen,
                          ring_r1=ring_r1, ring_size=S)
    frac, hi, tot = rv.avg_fp8_fraction(model)
    print(f"BUILT avg_fp8={frac:.4f} ({hi}/{tot} ch)", flush=True)
    return model, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--point", required=True)
    ap.add_argument("--substrate", choices=["dense", "ring"], default="ring")
    ap.add_argument("--ring_size", type=int, default=512)
    ap.add_argument("--alloc", default="measured",
                    choices=["measured", "uniform", "none"],
                    help="measured = NEW-recipe allocation; none = bf16")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--gptq_nsamples", type=int, default=None)
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--mmlu", action="store_true")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--skip_wiki", action="store_true",
                    help="skip the wikitext gate (debug only)")
    ap.add_argument("--expect_wiki", type=float, default=None,
                    help="reproduction gate: abort lm-eval if the built "
                         "model's wikitext PPL is off by >0.005")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="lm-eval batch size (default: config eval.batch_size)")
    ap.add_argument("--limit", type=int, default=None,
                    help="lm-eval per-task doc limit (smoke only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    if args.gptq_nsamples is not None:
        config["calibration"]["nsamples"] = args.gptq_nsamples
    if args.percdamp is not None:
        config["quantize"]["percdamp"] = args.percdamp
    if args.alloc != "none":
        for p in (config["paths"]["basis"], config["paths"]["rotation"]):
            assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    bs = args.batch_size or config["eval"]["batch_size"]

    meta = dict(point=args.point, model=config["model"]["name"],
                substrate=args.substrate, ring_size=args.ring_size,
                alloc=args.alloc,
                gptq_nsamples=config["calibration"].get("nsamples", 128),
                percdamp=config["quantize"].get("percdamp", 0.01),
                act_mse_steps=qu.ACT_MSE_STEPS,
                act_mse_steps_up=qu.ACT_MSE_STEPS_UP,
                weight_mse_steps=qu._MSE_STEPS,
                lm_eval_batch_size=bs, limit=args.limit,
                zs_tasks=rm.ZS_TASKS)
    results = {"meta": meta}
    print(f"START rd_mmlu {json.dumps(meta)}", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])

    t0 = time.time()
    if args.alloc == "none":
        model = rm.build_bf16(config)
        print(f"[bf16] built ({time.time()-t0:.0f}s)", flush=True)
    else:
        model, frac = build_newrecipe(config, args)
        results["avg_fp8"] = frac
        print(f"[{args.point}] built ({time.time()-t0:.0f}s)", flush=True)
    save()

    if not args.skip_wiki:
        t0 = time.time()
        ppl = rv.eval_ppl(model, config)
        results["wikitext"] = ppl
        print(f"GATE wikitext={ppl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        save()
        if args.expect_wiki is not None and abs(ppl - args.expect_wiki) > 5e-3:
            print(f"GATE FAIL expected {args.expect_wiki:.4f} got {ppl:.4f} "
                  f"— aborting before lm-eval", flush=True)
            save()
            sys.exit(2)

    if args.mmlu:
        t0 = time.time()
        results["mmlu_5shot"] = eval_mmlu(model, tok, bs, limit=args.limit)
        print(f"MMLU acc={results['mmlu_5shot']['groups'].get('mmlu')} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save()

    if args.zeroshot:
        t0 = time.time()
        results["zeroshot"] = rm.eval_zeroshot(model, tok, bs,
                                               limit=args.limit)
        print(f"ZEROSHOT avg={results['zeroshot']['average']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save()

    save()
    print(f"RD_MMLU_RESULT {args.point} "
          f"wiki={results.get('wikitext')} "
          f"mmlu={(results.get('mmlu_5shot') or {}).get('groups', {}).get('mmlu')} "
          f"zs={(results.get('zeroshot') or {}).get('average')}", flush=True)
    print("RD_MMLU_DONE", flush=True)


if __name__ == "__main__":
    main()
