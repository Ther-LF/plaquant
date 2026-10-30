"""rd-metrics: broaden the rate-distortion allocation evaluation from WikiText-2
PPL alone to the full ResQ/QuaRot/SpinQuant metric suite, on the SHIPPING
dense-R-on regime.

For each model it builds exactly three comparison points and evaluates ALL
metrics on each (so the 3B overturn / 8B break-even findings can be checked
across metrics, not just wikitext):

  (a) bf16        — no rotation, no quant (sanity bridge)
  (b) uniform-1/8 — dense-R-on uniform split (the shipped anchor)
  (c) measured-1/8— dense-R-on + per-ring measured-MSE deltaD allocation at the
                    matched 1/8 FP8 budget (the "dense-limit" from rd_ron, e=0;
                    this is our shipped allocation)

Metrics per config:
  * WikiText-2 PPL (evaluate_ppl)
  * C4 PPL         (evaluate_ppl on the QuaRot 256x seqlen validation stream)
  * zero-shot acc/acc_norm via lm-eval 0.4.x on:
      piqa, arc_easy, arc_challenge, hellaswag, winogrande, lambada_openai
    + the zero-shot AVERAGE (acc_norm where defined, else acc).

Reuses rd_validity's proven dense-R-on build+alloc machinery (build_base /
collect_stats / measured_alloc / build_eval with ring_r1=False, ring_size=64 =
the exact dense-limit recipe rd_ron_ringsize probes), so the wikitext column
reproduces the committed numbers (3B uniform 8.2974 / measured 8.2678; 8B
uniform 6.5882 / measured 6.6072) as a built-in gate.

Writes an incremental JSON (one per model) so partial progress survives.
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
from promix.models.loader import load_model

import rd_validity as rv
import rd_ron

# The ResQ/QuaRot/SpinQuant standard 0-shot suite.
ZS_TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag",
            "winogrande", "lambada_openai"]
# tasks that have no acc_norm -> the average uses plain acc for these.
ZS_ACC_ONLY = {"winogrande", "lambada_openai"}


def eval_zeroshot(model, tokenizer, batch_size, tasks=ZS_TASKS, limit=None):
    """lm-eval 0.4.x zero-shot. Fixes the CLAUDE.md device-mismatch bug by
    pinning the HFLM device to the model's ACTUAL parameter device (the model
    is loaded single-device via load_model().cuda(), no device_map) instead of
    leaving HFLM to guess a default 'cuda:0'.
    """
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator as lm_evaluator
    import datasets as hf_datasets
    hf_datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    model.eval()
    # evaluate_ppl offloads every layer back to CPU when it finishes; zero-shot
    # needs the WHOLE model resident on GPU (the quantized models' online
    # Hadamard is fast_hadamard_transform = CUDA-only, and CPU zero-shot is
    # ~100x slower). Force-resident on DEV instead of trusting the
    # (post-offload, CPU) parameter device.
    model.to(DEV)
    dev = DEV
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    # Device fix: force every device surface HFLM uses onto the real device.
    lm._device = dev
    try:
        lm.model.to(dev)
    except Exception:
        pass

    res = lm_evaluator.simple_evaluate(
        lm, tasks=list(tasks), num_fewshot=0, batch_size=batch_size,
        limit=limit,
    )
    out = {}
    preferred = []
    for t in tasks:
        m = res["results"].get(t, {})
        acc = m.get("acc,none")
        accn = m.get("acc_norm,none")
        out[t] = dict(acc=acc, acc_norm=accn)
        # average convention: acc_norm where it exists (arc/hella/piqa),
        # else plain acc (winogrande/lambada).
        p = acc if (t in ZS_ACC_ONLY or accn is None) else accn
        if p is not None:
            preferred.append(p)
    out["average"] = (sum(preferred) / len(preferred)) if preferred else None
    out["average_acc"] = _avg([out[t]["acc"] for t in tasks])
    return out


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def eval_all_metrics(model, config, tokenizer, do_zeroshot, zs_limit=None,
                     do_c4=True):
    """Run wikitext PPL + C4 PPL + zero-shot on an already-built model."""
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    seed = config["calibration"].get("seed", 0)
    bs = config["eval"]["batch_size"]

    t = time.time()
    wt = get_wikitext2(seed=seed, seqlen=model.seqlen, tokenizer=tokenizer,
                       eval_mode=True)
    ppl_wt = float(evaluate_ppl(model, wt, DEV, batch_size=bs))
    print(f"    wikitext PPL={ppl_wt:.4f} ({time.time()-t:.0f}s)", flush=True)

    ppl_c4 = None
    if do_c4:
        t = time.time()
        c4 = get_c4(nsamples=256, seed=seed, seqlen=model.seqlen,
                    tokenizer=tokenizer, eval_mode=True)
        ppl_c4 = float(evaluate_ppl(model, c4, DEV, batch_size=bs))
        print(f"    c4 PPL={ppl_c4:.4f} ({time.time()-t:.0f}s)", flush=True)

    zs = None
    if do_zeroshot:
        t = time.time()
        zs = eval_zeroshot(model, tokenizer, bs, limit=zs_limit)
        print(f"    zeroshot avg={zs['average']} ({time.time()-t:.0f}s)",
              flush=True)
    return dict(wikitext=ppl_wt, c4=ppl_c4, zeroshot=zs)


def build_bf16(config):
    """bf16 bridge model (no rotation, no quant)."""
    m = transformers.AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], torch_dtype=torch.bfloat16).to(DEV)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--ring_size", type=int, default=64)
    ap.add_argument("--configs", default="bf16,uniform,measured",
                    help="comma subset of bf16,uniform,measured")
    ap.add_argument("--no_zeroshot", action="store_true")
    ap.add_argument("--no_c4", action="store_true",
                    help="skip C4 PPL (e.g. when allenai/c4 is not "
                         "downloadable from the run host)")
    ap.add_argument("--zs_limit", type=int, default=None,
                    help="optional lm-eval per-task doc limit (debug/smoke)")
    ap.add_argument("--out", default="/home/luofan/RD_METRICS.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)   # 1b/3b/8b aware path+model resolve
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    S = args.ring_size
    want = set(args.configs.split(","))
    do_zs = not args.no_zeroshot

    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    results = {"meta": dict(model=config["model"]["name"],
                            basis=os.path.basename(config["paths"]["basis"]),
                            ring_size=S, zs_tasks=ZS_TASKS,
                            regime="dense-R-on (ring_r1=False)")}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"START rd_metrics model={config['model']['name']} "
          f"basis={results['meta']['basis']} configs={sorted(want)}", flush=True)

    # ---- (a) bf16 bridge ----------------------------------------------------
    if "bf16" in want:
        t = time.time()
        m = build_bf16(config)
        print(f"[bf16] built ({time.time()-t:.0f}s)", flush=True)
        results["bf16"] = eval_all_metrics(m, config, tok, do_zs, args.zs_limit,
                 do_c4=not args.no_c4)
        del m
        cleanup_memory()
        save()

    # ---- (b) dense-R-on uniform-1/8 (shipped anchor) ------------------------
    if "uniform" in want:
        t = time.time()
        m = rv.build_eval(config, r1_off=False, chosen=None,
                          ring_r1=False, ring_size=S)
        frac, hi, tot = rv.avg_fp8_fraction(m)
        print(f"[uniform] built avg_fp8={frac:.4f} ({hi}/{tot}) "
              f"({time.time()-t:.0f}s)", flush=True)
        r = eval_all_metrics(m, config, tok, do_zs, args.zs_limit,
                 do_c4=not args.no_c4)
        r["avg_fp8"] = frac
        results["uniform_1_8"] = r
        del m
        cleanup_memory()
        save()

    # ---- (c) dense-R-on measured-MSE-1/8 (shipped allocation, dense-limit) --
    if "measured" in want:
        t = time.time()
        calib = get_wikitext2(nsamples=args.calib_samples,
                              seed=config["calibration"].get("seed", 0),
                              seqlen=2048, tokenizer=tok, eval_mode=False)
        cmodel = rv.build_base(config, r1_off=False, ring_r1=False, ring_size=S)
        cmodel.to(DEV)
        stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens,
                                 ring=S)
        del cmodel
        cleanup_memory()
        total_rings = sum(stats[n]["nr"] for n in stats)
        budget = int(round(0.125 * total_rings))
        chosen = rv.measured_alloc(stats, budget, total_rings)
        m = rv.build_eval(config, r1_off=False, chosen=chosen,
                          ring_r1=False, ring_size=S)
        frac, hi, tot = rv.avg_fp8_fraction(m)
        print(f"[measured] built avg_fp8={frac:.4f} ({hi}/{tot}) "
              f"budget_rings={budget}/{total_rings} ({time.time()-t:.0f}s)",
              flush=True)
        r = eval_all_metrics(m, config, tok, do_zs, args.zs_limit,
                 do_c4=not args.no_c4)
        r["avg_fp8"] = frac
        results["measured_1_8"] = r
        del m
        cleanup_memory()
        save()

    save()
    print(f"WROTE {args.out}", flush=True)
    print("RD_METRICS_DONE", flush=True)


if __name__ == "__main__":
    main()
