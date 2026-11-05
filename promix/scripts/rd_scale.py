"""rd-scale (paper E2) — BIG-MODEL-SAFE accuracy-point driver (70B-capable).

Llama-3-70B is 140 GB in fp16 and can never be GPU-resident on one 96 GB
RTX Pro 6000, so this is `rd_acctune.py` with every full-model `.to(DEV)`
removed; everything that defines the committed protocol (rd_ron.resolve_paths,
rd_validity build/alloc/GPTQ machinery, meta knobs, quantizer-search defaults)
is reused unchanged:

  * PPL eval      : `evaluate_ppl` already layer-offloads (one layer on GPU at
                    a time) — the model stays CPU-resident, we just skip the
                    pre-`.to(DEV)` that rd_validity.eval_ppl does.
  * bf16 bridge   : build on CPU, evaluate through the same layer-offload path.
  * stats collect : the exact per-ring MSE/variance hooks of
                    `rd_validity.collect_stats`, but the forward pass is driven
                    by `evaluate_ppl`'s layer-offload loop (batch_size=1 keeps
                    the token order identical to rd_validity's per-sample loop,
                    so the first `collect_tokens` tokens — and hence the
                    allocation — are the same).

Small-model invocations must reproduce the committed acctune points
digit-for-digit (gate on 1B before any 70B run):
  1b ring512 uniform  wiki 14.8557   |  1b ring512 measured  wiki 14.2392
  (docs/rd_acctune_data/1b_ring512_{uniform,measured}.json)

Zero-shot is deliberately NOT offered here: lm-eval needs the whole fake-quant
model GPU-resident (fp16 weights, 140 GB for 70B) — out of scope for E2
(wiki + C4 PPL only), noted as a limitation in docs/rd_scale_report.md.

Example (70B, GPU lane 0-2):
  python promix/scripts/rd_scale.py \
      --config promix/configs/llama-3-70b-instruct-mxfp8-nvfp4.yaml \
      --point 70b_ring512_measured --substrate ring --ring_size 512 \
      --alloc measured --c4 --out /home/luofan/rdscale/70b_ring512_measured.json
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

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
from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper

import rd_validity as rv
import rd_ron


# ---------------------------------------------------------------------------
# Offload-safe evaluation (no full-model .to(DEV); evaluate_ppl layer-offloads)
# ---------------------------------------------------------------------------
def _park_cpu(model):
    """Move the head/tail pieces evaluate_ppl leaves on GPU back to CPU."""
    model.lm_head = model.lm_head.cpu()
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()


def eval_ppl_offload(model, config, which="wikitext"):
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    seed = config["calibration"].get("seed", 0)
    if which == "wikitext":
        data = get_wikitext2(seed=seed, seqlen=model.seqlen, tokenizer=tok,
                             eval_mode=True)
    else:
        data = get_c4(nsamples=256, seed=seed, seqlen=model.seqlen,
                      tokenizer=tok, eval_mode=True)
    ppl = float(evaluate_ppl(model, data, DEV,
                             batch_size=config["eval"]["batch_size"]))
    _park_cpu(model)
    return ppl


def bf16_bridge_offload(config, do_c4):
    """bf16 bridge built on CPU; layer-offload eval."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], torch_dtype=torch.bfloat16)
    out = {"wikitext": eval_ppl_offload(model, config, "wikitext")}
    if do_c4:
        out["c4"] = eval_ppl_offload(model, config, "c4")
    del model
    cleanup_memory()
    return out


# ---------------------------------------------------------------------------
# Offload-safe stats collection: rd_validity.collect_stats's hooks verbatim,
# forward driven by evaluate_ppl's layer-offload loop instead of model(ids).
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_stats_offload(model, calib, max_tokens=4096, ring=64):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    acc = {}
    handles = []

    def make_hook(name):
        def hook(_module, inp):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()
            st = acc.get(name)
            d = x.shape[1]
            if st is None:
                nr = d // ring
                st = dict(n=0, e4=torch.zeros(nr), e8=torch.zeros(nr),
                          csum=torch.zeros(d), csq=torch.zeros(d), d=d, nr=nr)
                acc[name] = st
            if st["n"] >= max_tokens:
                return
            take = min(x.shape[0], max_tokens - st["n"])
            x = x[:take]
            q4 = qu.fake_quantize_nvfp4(x)
            q8 = qu.fake_quantize_mxfp8(x)
            e4 = (x - q4).pow(2).reshape(take, st["nr"], ring).sum(dim=(0, 2))
            e8 = (x - q8).pow(2).reshape(take, st["nr"], ring).sum(dim=(0, 2))
            st["e4"] += e4.cpu()
            st["e8"] += e8.cpu()
            st["csum"] += x.sum(0).cpu()
            st["csq"] += x.pow(2).sum(0).cpu()
            st["n"] += take
        return hook

    for name, w in qlayers.items():
        if rv._is_alloc_target(name):
            handles.append(w.module.register_forward_pre_hook(make_hook(name)))

    # Layer-offload forward: concatenate the calib samples (in rd_validity's
    # order) into one input_ids stream and drive it through evaluate_ppl with
    # batch_size=1, so every hook sees the same tokens in the same order as
    # rd_validity.collect_stats's per-sample model(ids) loop. The returned PPL
    # is ignored — this is only a forward-pass driver.
    model.seqlen = calib[0][0].shape[-1]
    flat = torch.cat([b[0].reshape(1, -1) for b in calib], dim=1)
    evaluate_ppl(model, SimpleNamespace(input_ids=flat), DEV, batch_size=1)
    _park_cpu(model)

    for h in handles:
        h.remove()

    stats = {}
    for name, st in acc.items():
        n = max(st["n"], 1)
        mse4 = st["e4"] / (n * ring)
        mse8 = st["e8"] / (n * ring)
        deltaD = (mse4 - mse8)
        mean = st["csum"] / n
        var = (st["csq"] / n - mean.pow(2)).clamp_min(0)
        ring_var = var.reshape(st["nr"], ring).mean(dim=1)
        stats[name] = dict(deltaD=deltaD, ring_var=ring_var, nr=st["nr"],
                           d=st["d"], ntok=st["n"])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--point", required=True, help="unique point tag")
    ap.add_argument("--substrate", choices=["dense", "ring"], default="dense")
    ap.add_argument("--ring_size", type=int, default=64,
                    help="ring substrate width (and the allocation atom)")
    ap.add_argument("--alloc", default="measured",
                    choices=["uniform", "measured", "none"],
                    help="uniform = shipped positional tail (chosen=None); "
                         "measured = global deltaD threshold; "
                         "none = bf16 bridge only")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    # rd-densesweep: budget knob + stats cache (see rd_acctune.py)
    ap.add_argument("--budget_frac", type=float, default=0.125)
    ap.add_argument("--stats_cache", default=None)
    ap.add_argument("--gptq_nsamples", type=int, default=None)
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--act_order", action="store_true")
    ap.add_argument("--c4", action="store_true", help="also eval C4 PPL")
    ap.add_argument("--bridge", action="store_true",
                    help="also run the bf16 bridge (protocol gate)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    if args.gptq_nsamples is not None:
        config["calibration"]["nsamples"] = args.gptq_nsamples
    if args.percdamp is not None:
        config["quantize"]["percdamp"] = args.percdamp
    if args.act_order:
        config["quantize"]["act_order"] = True
    if args.alloc != "none":
        for p in (config["paths"]["basis"], config["paths"]["rotation"]):
            assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    S = args.ring_size
    ring_r1 = args.substrate == "ring"
    meta = dict(point=args.point, model=config["model"]["name"],
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
                driver="rd_scale(offload)")
    results = {"meta": meta}
    print(f"START rd_scale {json.dumps(meta)}", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.bridge:
        t0 = time.time()
        bridge = bf16_bridge_offload(config, do_c4=args.c4)
        print(f"BF16_BRIDGE {json.dumps(bridge)} ({time.time()-t0:.0f}s)",
              flush=True)
        results["bf16_bridge"] = bridge
        save()
    if args.alloc == "none":
        save()
        print("RDSCALE_DONE", flush=True)
        return

    # --- allocation ----------------------------------------------------------
    chosen = None
    if args.alloc == "measured":
        t0 = time.time()
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
            stats = collect_stats_offload(cmodel, calib,
                                          max_tokens=args.collect_tokens, ring=S)
            del cmodel
            cleanup_memory()
            if args.stats_cache:
                tmp = args.stats_cache + f".tmp{os.getpid()}"
                torch.save(stats, tmp)
                os.replace(tmp, args.stats_cache)
                print(f"STATS_CACHE saved {args.stats_cache}", flush=True)
        total_rings = sum(stats[n]["nr"] for n in stats)
        budget = int(round(args.budget_frac * total_rings))
        chosen = rv.clamp_chosen(
            rv.measured_alloc(stats, budget, total_rings), stats)
        nsel = sum(len(v) for v in chosen.values())
        print(f"ALLOC total_rings={total_rings} sel={nsel} "
              f"(frac={nsel/total_rings:.4f}) ({time.time()-t0:.0f}s)",
              flush=True)
        results["alloc_meta"] = dict(total_rings=total_rings, sel_rings=nsel,
                                     budget_frac=args.budget_frac)
        save()

    # --- build + eval --------------------------------------------------------
    t0 = time.time()
    model = rv.build_eval(config, r1_off=False, chosen=chosen,
                          ring_r1=ring_r1, ring_size=S)
    frac, hi, tot = rv.avg_fp8_fraction(model)
    print(f"BUILT avg_fp8={frac:.4f} ({hi}/{tot} ch) "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["avg_fp8"] = frac
    save()

    t0 = time.time()
    ppl_wiki = eval_ppl_offload(model, config, "wikitext")
    print(f"RESULT {args.point} wikitext={ppl_wiki:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["wikitext"] = ppl_wiki
    save()

    if args.c4:
        t0 = time.time()
        ppl_c4 = eval_ppl_offload(model, config, "c4")
        print(f"RESULT {args.point} c4={ppl_c4:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["c4"] = ppl_c4
        save()

    del model
    cleanup_memory()
    save()
    print(f"RDSCALE_RESULT point={args.point} wiki={ppl_wiki:.4f} "
          f"c4={results.get('c4')}", flush=True)
    print("RDSCALE_DONE", flush=True)


if __name__ == "__main__":
    main()
