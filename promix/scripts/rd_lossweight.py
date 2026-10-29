"""Loss-weighted rate-distortion allocation (spec §9 fallback / §4.5 refinement).

The shipped allocator (rd_ron / rd_validity) ranks 64-rings by pure LOCAL ΔD
(= D_nvfp4 − D_mxfp8 of the rotated activation). That implicitly assumes every
channel's quantization error costs the final LM loss equally. It doesn't — and
on 8B the pure-ΔD dense-limit pick was actually WRONG (measured 6.6072 vs
uniform 6.5882, gain −0.019; see docs/rd_ron_report.md). This driver tests the
refinement: weight each ring's ΔD by its sensitivity to the final cross-entropy
loss, estimated with an empirical Fisher-diagonal proxy.

Fisher proxy (exact form, see adaptive.ring_fisher_weight / ring_deltaD_weighted):
  one backward pass of the LM cross-entropy loss per calibration sequence;
  a tensor grad-hook on the input of each target Linear captures dL/da_c;
  accumulate F_c = (1/T) Σ_t (dL/da_c)^2 per channel, then w_j = Σ_{c∈ring j} F_c.
  Loss-weighted score_j = w_j · ΔD_j  (raw, globally ranked — the raw Fisher is
  comparable across layers so a ring in a loss-sensitive layer can outrank a
  higher-ΔD ring elsewhere).

THREE allocations compared at a MATCHED 1/8 FP8-ring budget, in the SHIPPING
regime (dense-R-on = shipped dense R1, ring_r1=False, reorder-before-quant K):
  uniform      : every Linear K_high = 1/8·d          (the shipped baseline)
  pure-ΔD      : rv.measured_alloc  (what rd_ron used; 8B pick is negative)
  loss-weighted: allocate_by_budget on w_j·ΔD_j         (the refinement)

Gates: bf16 bridge must reproduce the floor (3B 7.82 / 8B 6.14); uniform and
pure-ΔD must reproduce docs/rd_ron_report.md dense-limit numbers (3B 8.2974 /
8.2678; 8B 6.5882 / 6.6072) before the loss-weighted number is trusted.

Reuses rd_validity's proven build/collect/allocate/eval machinery unchanged; the
ONLY new logic is the Fisher collection and the weighted score. --r1_off runs
the same comparison on the R1-off substrate as a cross-check.
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
from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper
from promix.quantize.adaptive import (
    allocate_by_budget, ring_fisher_weight, ring_deltaD_weighted,
)

import rd_validity as rv
import rd_ron

RING = rv.RING


# ---------------------------------------------------------------------------
# Fisher-diagonal proxy collection: per-channel Σ_t (dL/da_c)^2 at each target
# Linear's input, via LM-loss backward. Same hook point as rv.collect_stats
# (w.module forward-pre), so Fisher channels line up 1:1 with the ΔD channels.
# ---------------------------------------------------------------------------
def collect_fisher(model, calib, max_tokens=4096, ring=RING):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    targets = {n: w for n, w in qlayers.items() if rv._is_alloc_target(n)}
    acc = {}  # name -> dict(g2=cpu Tensor(d), n=int)

    def make_pre_hook(name):
        def pre_hook(_m, inp):
            st = acc.get(name)
            if st is not None and st["n"] >= max_tokens:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not x.requires_grad:
                return

            def grad_hook(g):
                s = acc.get(name)
                if s is not None and s["n"] >= max_tokens:
                    return
                gg = g.detach()
                if gg.dim() == 3:
                    gg = gg.reshape(-1, gg.shape[-1])
                gg = torch.nan_to_num(gg.float(), nan=0.0, posinf=0.0, neginf=0.0)
                d = gg.shape[1]
                if s is None:
                    s = dict(g2=torch.zeros(d), n=0)
                    acc[name] = s
                s["g2"] += gg.pow(2).sum(0).cpu()
                s["n"] += gg.shape[0]

            x.register_hook(grad_hook)
        return pre_hook

    handles = [w.module.register_forward_pre_hook(make_pre_hook(n))
               for n, w in targets.items()]

    # Minimal-memory activation graph: freeze every param, re-enable grad ONLY
    # on the embedding so every downstream activation is in the autograd graph
    # (the grad-hooks need that) while param-grad storage stays tiny.
    frozen = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    emb = model.model.embed_tokens.weight
    emb.requires_grad_(True)
    prev_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    model.eval()

    seqs = 0
    for batch in calib:
        if all(acc.get(n, {"n": 0})["n"] >= max_tokens for n in targets):
            break
        ids = batch[0].to(DEV)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        emb.grad = None
        seqs += 1

    for h in handles:
        h.remove()
    for p, rg in frozen:
        p.requires_grad_(rg)
    if prev_cache is not None:
        model.config.use_cache = prev_cache

    fisher = {}
    for n, s in acc.items():
        fisher[n] = s["g2"] / max(s["n"], 1)   # per-channel mean Fisher F_c
    ntok = {n: acc.get(n, {"n": 0})["n"] for n in targets}
    print(f"FISHER seqs={seqs} tokens~{max(ntok.values()) if ntok else 0} "
          f"units={len(fisher)}", flush=True)
    return fisher


def weighted_alloc(stats, fisher, budget_rings, total_rings, ring=RING):
    """Global single-threshold allocation on the loss-weighted score w_j·ΔD_j."""
    names = list(stats)
    scores = []
    for n in names:
        w = ring_fisher_weight(fisher[n], ring) if fisher[n].numel() != stats[n]["nr"] \
            else fisher[n]
        scores.append(ring_deltaD_weighted(stats[n]["deltaD"], w))
    target = budget_rings / total_rings
    masks = allocate_by_budget(scores, target_mean_fp8=target)
    chosen = {}
    for n, m in zip(names, masks):
        chosen[n] = set(torch.nonzero(m).flatten().tolist())
    return chosen


def fisher_to_ring(stats, fisher, ring=RING):
    """Per-ring Fisher weight dict (channel -> ring reduction) for diagnostics."""
    return {n: ring_fisher_weight(fisher[n], ring) for n in stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r1_off", action="store_true",
                    help="cross-check on the R1-off substrate instead of dense-R-on")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--fisher_tokens", type=int, default=4096)
    ap.add_argument("--fisher_seqlen", type=int, default=1024)
    ap.add_argument("--fisher_samples", type=int, default=16)
    ap.add_argument("--out", default="/home/luofan/RD_LOSSW.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    ring_r1 = False          # dense-R-on (shipped rotation) unless --r1_off
    r1_off = bool(args.r1_off)
    regime = "r1_off" if r1_off else "dense_ron"
    print(f"START rd_lossweight model={config['model']['name']} regime={regime} "
          f"basis={os.path.basename(config['paths']['basis'])}", flush=True)

    results = {"meta": dict(model=config["model"]["name"], regime=regime,
                            basis=os.path.basename(config["paths"]["basis"]),
                            ring=RING, fisher_tokens=args.fisher_tokens,
                            fisher_seqlen=args.fisher_seqlen)}

    # --- bf16 bridge (sanity: 3B 7.82 / 8B 6.14) -----------------------------
    t0 = time.time()
    ppl_bf16 = rv.bf16_bridge(config)
    print(f"BF16_BRIDGE ppl={ppl_bf16:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["bf16_bridge"] = ppl_bf16
    cleanup_memory()

    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])

    # --- collect per-ring ΔD (clean fp16 rotated activations) ----------------
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rv.build_base(config, r1_off=r1_off, ring_r1=ring_r1)
    cmodel.to(DEV)
    stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens, ring=RING)
    del cmodel
    cleanup_memory()

    # --- collect Fisher-diagonal proxy (LM-loss backward) --------------------
    fcalib = get_wikitext2(nsamples=args.fisher_samples,
                           seed=config["calibration"].get("seed", 0),
                           seqlen=args.fisher_seqlen, tokenizer=tok, eval_mode=False)
    fmodel = rv.build_base(config, r1_off=r1_off, ring_r1=ring_r1)
    fmodel.to(DEV)
    fisher = collect_fisher(fmodel, fcalib, max_tokens=args.fisher_tokens, ring=RING)
    del fmodel
    cleanup_memory()

    total_rings = sum(stats[n]["nr"] for n in stats)
    uni = rv.uniform_rings(stats)
    budget_rings = sum(uni.values())
    print(f"ALLOC units={len(stats)} total_rings={total_rings} "
          f"budget_fp8_rings={budget_rings} "
          f"(frac={budget_rings/total_rings:.4f})", flush=True)

    chosen_mse = rv.measured_alloc(stats, budget_rings, total_rings)
    chosen_lw = weighted_alloc(stats, fisher, budget_rings, total_rings, ring=RING)
    n_mse = sum(len(v) for v in chosen_mse.values())
    n_lw = sum(len(v) for v in chosen_lw.values())
    overlap = sum(len(chosen_mse[n] & chosen_lw[n]) for n in stats)
    print(f"PURE-ΔD fp8_rings={n_mse}  LOSS-W fp8_rings={n_lw}  "
          f"MSE∩LW overlap={overlap}/{budget_rings} "
          f"({overlap/max(budget_rings,1)*100:.1f}%)", flush=True)

    # Fisher diagnostics: how non-uniform is the loss sensitivity?
    fr = fisher_to_ring(stats, fisher, ring=RING)
    allw = torch.cat([fr[n].flatten() for n in stats]).double()
    results["fisher_diag"] = dict(
        w_mean=float(allw.mean()), w_std=float(allw.std(unbiased=False)),
        w_cv=float(allw.std(unbiased=False) / allw.mean().clamp_min(1e-12)),
        w_max_over_mean=float(allw.max() / allw.mean().clamp_min(1e-12)))
    results["alloc"] = dict(total_rings=total_rings, budget_rings=budget_rings,
                            n_mse=n_mse, n_lw=n_lw, overlap=overlap)

    def run(tag, chosen):
        t = time.time()
        model = rv.build_eval(config, r1_off=r1_off, chosen=chosen,
                              ring_r1=ring_r1, ring_size=RING)
        frac, hi, tot = rv.avg_fp8_fraction(model)
        ppl = rv.eval_ppl(model, config)
        print(f"RESULT {tag} ppl={ppl:.4f} avg_fp8={frac:.4f} ({hi}/{tot} ch) "
              f"({time.time()-t:.0f}s)", flush=True)
        del model
        cleanup_memory()
        return dict(ppl=ppl, avg_fp8=frac, hi_ch=hi, tot_ch=tot)

    results["uniform"] = run("uniform", None)
    results["pure_deltaD"] = run("pure_deltaD", chosen_mse)
    results["loss_weighted"] = run("loss_weighted", chosen_lw)

    u = results["uniform"]["ppl"]
    p = results["pure_deltaD"]["ppl"]
    l = results["loss_weighted"]["ppl"]
    verdict = dict(
        uniform=u, pure_deltaD=p, loss_weighted=l, bf16=ppl_bf16,
        lw_beats_pure=l < p - 1e-4,
        lw_beats_uniform=l < u - 1e-4,
        pure_beats_uniform=p < u - 1e-4,
        lw_vs_pure=l - p, lw_vs_uniform=l - u, pure_vs_uniform=p - u,
    )
    verdict["summary"] = (
        ("LOSS-WEIGHTING HELPS: beats pure-ΔD" if verdict["lw_beats_pure"]
         else "loss-weighting does NOT beat pure-ΔD")
        + (" and beats uniform" if verdict["lw_beats_uniform"]
           else "; does not beat uniform"))
    results["verdict"] = verdict

    print("=" * 70, flush=True)
    print(f"VERDICT [{regime}] bf16={ppl_bf16:.3f} | uniform={u:.4f} "
          f"pure-ΔD={p:.4f} loss-weighted={l:.4f}", flush=True)
    print(f"  lw vs pure-ΔD: {l-p:+.4f} ({'BEATS' if l<p-1e-4 else 'no'})  "
          f"lw vs uniform: {l-u:+.4f} ({'BEATS' if l<u-1e-4 else 'no'})  "
          f"pure vs uniform: {p-u:+.4f}", flush=True)
    print(f"  {verdict['summary']}", flush=True)
    print("=" * 70, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {args.out}", flush=True)
    print("RD_LOSSW_DONE", flush=True)


if __name__ == "__main__":
    main()
