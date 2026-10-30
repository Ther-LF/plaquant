"""rd-whydecay: WHY does the measured-MSE allocation gain decay to ~0 with
model scale?  (wiki dPPL measured-uniform @1/8, dense-R-on: 1B -0.077 /
3B -0.030 / 8B +0.019 / 13B -0.001)

Two competing mechanisms:
  UPSTREAM   — the per-ring dD / variance structure flattens with scale, so
               there is less concentration for the allocator to exploit.
  DOWNSTREAM — the structure is scale-invariant, but the model's total
               quantization headroom (uniform-W4A4 gap vs bf16) shrinks and/or
               the per-ring PPL sensitivity flattens (redundancy), so the same
               reallocation buys less PPL.

Measurements per model (1b/3b/8b/13b), SHIPPING dense-R-on regime
(ring_size=64, o_proj held uniform, same build as rd_metrics):

  1. STRUCTURE: per-ring dD and per-ring variance concentration — Gini,
     top-1/32 share, top-1/8 share, CV, Spearman(dD, var), per-layer dD CV,
     top-1 / top-16 ring dD shares.  (cheap: calibration forward only)
  2. TOP-RING FLIP (decisive): from the explicit uniform-1/8 ring-set, demote
     the globally top-dD ring (flip1) and the top-16 dD rings (flip16) from
     MXFP8 back to NVFP4; measure wikitext PPL of uniform / flip1 / flip16.
     Per-ring PPL sensitivity = dPPL (and dlogPPL) per flipped dD-share.
     If the same-rank top ring moves PPL much less on bigger models while its
     dD-share is scale-invariant, the decay is DOWNSTREAM.

Headroom (uniform gap vs bf16) and gain/headroom fractions come from the
committed rd_metrics suite (docs/rd_metrics_report.md) — no re-run needed.

Writes an incremental JSON per model.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transformers

import rd_validity as rdv
from promix.eval import ptq as ptqmod
from promix.eval.data import get_wikitext2
from promix.eval.evaluator import evaluate_ppl
from promix.utils import DEV, cleanup_memory

MODELS = {
    "1b": dict(short="Llama-3.2-1B-Instruct",
               model="unsloth/Llama-3.2-1B-Instruct",
               local="/home/luofan/models/Llama-3.2-1B-Instruct",
               config="promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml"),
    "3b": dict(short="Llama-3.2-3B",
               model="unsloth/Llama-3.2-3B",
               local="/home/luofan/models/Llama-3.2-3B",
               config="promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml"),
    "8b": dict(short="llama-3-8b",
               model="unsloth/llama-3-8b",
               local="/home/luofan/models/llama-3-8b",
               config="promix/configs/llama-3-8b-mxfp8-nvfp4.yaml"),
    "13b": dict(short="Llama-2-13b-hf",
                model="NousResearch/Llama-2-13b-hf",
                local="/home/luofan/models/Llama-2-13b-hf",
                config="promix/configs/llama-2-13b-mxfp8-nvfp4.yaml"),
}


def resolve_paths(config, mk):
    m = MODELS[mk]
    config["model"]["name"] = m["local"] if os.path.isdir(m["local"]) else m["model"]
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["basis"] = os.path.join(
        rotdir, f"U-wikitext-512-{m['short']}.bin")
    config["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{m['short']}.bin")
    config["quantize"]["o_proj_pca"] = "per_head"
    config["quantize"]["high_fraction"] = 0.125
    return config


# ---------------------------------------------------------------------------
# concentration metrics
# ---------------------------------------------------------------------------
def gini(x):
    """Gini coefficient of a non-negative 1-D tensor."""
    x = torch.as_tensor(x, dtype=torch.float64).clamp_min(0).sort().values
    n = x.numel()
    s = x.sum()
    if s <= 0:
        return float("nan")
    cum = torch.cumsum(x, 0)
    # G = 1 - 2/(n-1) * (sum(cum)/cum[-1] - 1) alt form; use standard:
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2 * idx - n - 1) * x).sum() / (n * s))


def top_share(x, frac):
    """Fraction of sum(x) held by the top `frac` of entries."""
    x = torch.as_tensor(x, dtype=torch.float64).clamp_min(0)
    k = max(1, int(round(frac * x.numel())))
    return float(x.topk(k).values.sum() / x.sum()), k


def cv(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return float(x.std(unbiased=False) / x.mean()) if x.mean() > 0 else float("nan")


def spearman(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = (ra - ra.mean()) / ra.std(unbiased=False)
    rb = (rb - rb.mean()) / rb.std(unbiased=False)
    return float((ra * rb).mean())


def anova_within_frac(dd, groups):
    """1-way variance decomposition of per-ring dD: fraction of total variance
    that lives WITHIN groups (vs between group means)."""
    dd = torch.as_tensor(dd, dtype=torch.float64)
    mu = dd.mean()
    ssb = ssw = 0.0
    for g in set(groups):
        xs = dd[torch.tensor([i for i, gg in enumerate(groups) if gg == g])]
        ssb += xs.numel() * float((xs.mean() - mu) ** 2)
        ssw += float(((xs - xs.mean()) ** 2).sum())
    tot = ssb + ssw
    return (ssw / tot if tot > 0 else float("nan")), (ssb / tot if tot > 0 else float("nan"))


def layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def structure_metrics(stats):
    names = sorted(stats)
    dd = torch.cat([torch.as_tensor(stats[n]["deltaD"]).double() for n in names])
    vv = torch.cat([torch.as_tensor(stats[n]["ring_var"]).double() for n in names])
    # global ring index -> (name, local ring)
    ring_map = []
    for n in names:
        ring_map += [(n, r) for r in range(stats[n]["nr"])]

    s32, k32 = top_share(dd, 1 / 32)
    s8, k8 = top_share(dd, 1 / 8)
    v32, _ = top_share(vv, 1 / 32)
    v8, _ = top_share(vv, 1 / 8)

    by_layer = {}
    for n in names:
        li = layer_idx(n)
        by_layer[li] = by_layer.get(li, 0.0) + float(
            torch.as_tensor(stats[n]["deltaD"]).clamp_min(0).sum())
    layer_cv = cv(torch.tensor(sorted(by_layer.values())))

    order = dd.argsort(descending=True)
    dsum = float(dd.clamp_min(0).sum())
    top1_g = int(order[0])
    top16 = [int(i) for i in order[:16]]
    # ---- structural breakdown (coordinator add-on) ----
    # ring-level group labels
    g_layer = []
    g_type = []
    g_unit = []
    for n in names:
        li, ty = layer_idx(n), n.split(".")[-1]
        g_layer += [li] * stats[n]["nr"]
        g_type += [ty] * stats[n]["nr"]
        g_unit += [n] * stats[n]["nr"]
    within_layer_frac, between_layer_frac = anova_within_frac(dd, g_layer)
    within_type_frac, between_type_frac = anova_within_frac(dd, g_type)
    within_unit_frac, between_unit_frac = anova_within_frac(dd, g_unit)

    dsum_all = float(dd.clamp_min(0).sum())
    # measured allocation at 1/8 (for per-type / per-layer budget shift)
    budget = sum(int(round(0.125 * stats[n]["nr"])) for n in names)
    total_r = sum(stats[n]["nr"] for n in names)
    mea = rdv.measured_alloc(stats, budget, total_r)

    by_type = {}
    for n in names:
        ty = n.split(".")[-1]
        t = by_type.setdefault(ty, dict(rings=0, dd_sum=0.0, fp8_measured=0,
                                        within_top32=[]))
        t["rings"] += stats[n]["nr"]
        sdd = torch.as_tensor(stats[n]["deltaD"]).double().clamp_min(0)
        t["dd_sum"] += float(sdd.sum())
        t["fp8_measured"] += len(mea.get(n, set()))
        k = max(1, int(round(stats[n]["nr"] / 32)))
        t["within_top32"].append(float(sdd.topk(k).values.sum() / sdd.sum().clamp_min(1e-30)))
    for ty, t in by_type.items():
        t["dd_share"] = t["dd_sum"] / dsum_all
        t["ring_share"] = t["rings"] / total_r
        t["dd_density"] = t["dd_share"] / t["ring_share"]  # 1.0 = pro-rata
        t["fp8_frac_measured"] = t["fp8_measured"] / t["rings"]
        t["within_top1_32_share"] = sum(t["within_top32"]) / len(t["within_top32"])
        del t["within_top32"]

    by_layer_full = {}
    for n in names:
        li = layer_idx(n)
        L = by_layer_full.setdefault(li, dict(rings=0, dd_sum=0.0, fp8_measured=0))
        L["rings"] += stats[n]["nr"]
        L["dd_sum"] += float(torch.as_tensor(stats[n]["deltaD"]).clamp_min(0).sum())
        L["fp8_measured"] += len(mea.get(n, set()))
    for li, L in by_layer_full.items():
        L["dd_share"] = L["dd_sum"] / dsum_all
        L["fp8_frac_measured"] = L["fp8_measured"] / L["rings"]
    # within-layer top-1/32 concentration (avg over layers)
    wl32 = []
    for li in sorted(by_layer_full):
        xs = torch.cat([torch.as_tensor(stats[n]["deltaD"]).double().clamp_min(0)
                        for n in names if layer_idx(n) == li])
        k = max(1, int(round(xs.numel() / 32)))
        wl32.append(float(xs.topk(k).values.sum() / xs.sum().clamp_min(1e-30)))
    within_layer_top1_32 = sum(wl32) / len(wl32)

    m = dict(
        total_rings=int(dd.numel()),
        breakdown=dict(
            within_layer_var_frac=within_layer_frac,
            between_layer_var_frac=between_layer_frac,
            within_type_var_frac=within_type_frac,
            between_type_var_frac=between_type_frac,
            within_unit_var_frac=within_unit_frac,
            between_unit_var_frac=between_unit_frac,
            within_layer_top1_32_share=within_layer_top1_32,
            by_type=by_type,
            by_layer={int(k): v for k, v in sorted(by_layer_full.items())},
        ),
        dd_sum=dsum,
        dd_gini=gini(dd),
        dd_top1_32_share=s32, dd_top1_32_k=k32,
        dd_top1_8_share=s8,
        dd_cv=cv(dd),
        dd_neg_frac=float((dd < 0).double().mean()),
        var_gini=gini(vv),
        var_top1_32_share=v32,
        var_top1_8_share=v8,
        var_cv=cv(vv),
        spearman_dd_var=spearman(dd, vv),
        per_layer_dd_cv=layer_cv,
        top1=dict(name=ring_map[top1_g][0], ring=ring_map[top1_g][1],
                  dd=float(dd[top1_g]), dd_share=float(dd[top1_g]) / dsum),
        top16_dd_share=float(dd[top16].sum()) / dsum,
        top16_rings=[dict(name=ring_map[i][0], ring=ring_map[i][1],
                          dd_share=float(dd[i]) / dsum) for i in top16],
    )
    return m, ring_map, order, dd


def uniform_chosen(stats):
    """Explicit uniform-1/8: each Linear protects its own top round(nr/8)
    variance rings (= the shipped last-K_high tail, built via the same
    reorder path the flips use so the comparison is apples-to-apples)."""
    ch = {}
    for n, st in stats.items():
        k = int(round(0.125 * st["nr"]))
        rv = torch.as_tensor(st["ring_var"])
        ch[n] = set(torch.topk(rv, k).indices.tolist()) if k > 0 else set()
    return ch


def flip_chosen(uni, ring_map, order, k):
    """Uniform minus the global top-k dD rings (only those uniform protects)."""
    ch = {n: set(s) for n, s in uni.items()}
    flipped = []
    missed = []
    for gi in order[:k].tolist():
        n, r = ring_map[gi]
        if r in ch[n]:
            ch[n].remove(r)
            flipped.append((n, r))
        else:
            missed.append((n, r))
    return ch, flipped, missed


def eval_ppl(model, config, testenc):
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    return float(evaluate_ppl(model, testenc, DEV,
                              batch_size=config["eval"]["batch_size"]))


def run(mk, args):
    config = ptqmod.load_config(os.path.join(REPO, MODELS[mk]["config"]))
    config = resolve_paths(config, mk)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    S = args.ring_size

    res = dict(model=mk, hf=config["model"]["name"], ring_size=S,
               regime="dense-R-on")

    def save():
        json.dump(res, open(args.out, "w"), indent=2, default=str)

    # --- collect dense-R-on rotated activation stats --------------------------
    t0 = time.time()
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, r1_off=False, ring_r1=False, ring_size=S)
    cmodel.to(DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens,
                              ring=S)
    del cmodel
    cleanup_memory()
    print(f"[{mk}] stats collected ({time.time()-t0:.0f}s)", flush=True)

    m, ring_map, order, dd = structure_metrics(stats)
    res["structure"] = m
    print(f"[{mk}] STRUCTURE dd_gini={m['dd_gini']:.4f} "
          f"top1/32={m['dd_top1_32_share']*100:.1f}% top1/8={m['dd_top1_8_share']*100:.1f}% "
          f"CV={m['dd_cv']:.3f} rho={m['spearman_dd_var']:.4f} "
          f"layerCV={m['per_layer_dd_cv']:.4f} "
          f"top1_share={m['top1']['dd_share']*100:.3f}% "
          f"top16_share={m['top16_dd_share']*100:.2f}%", flush=True)
    save()
    if args.structure_only:
        return res

    # --- flips ----------------------------------------------------------------
    uni = uniform_chosen(stats)
    f1, fl1, miss1 = flip_chosen(uni, ring_map, order, 1)
    f16, fl16, miss16 = flip_chosen(uni, ring_map, order, 16)
    res["flip1_rings"] = fl1
    res["flip1_missed"] = miss1
    res["flip16_n"] = len(fl16)
    res["flip16_missed"] = miss16
    res["flip1_dd_share"] = m["top1"]["dd_share"] if not miss1 else float(
        sum(dd[i] for i, (nm, r) in enumerate(ring_map) if (nm, r) in fl1)
        / m["dd_sum"])
    res["flip16_dd_share"] = float(
        sum(float(dd[gi]) for gi in order[:16].tolist()
            if ring_map[gi] in fl16) / m["dd_sum"])
    print(f"[{mk}] flip1={fl1} missed={miss1}; flip16 n={len(fl16)} "
          f"missed={len(miss16)} dd_share={res['flip16_dd_share']*100:.2f}%",
          flush=True)

    testenc = get_wikitext2(seed=config["calibration"].get("seed", 0),
                            seqlen=config["eval"]["max_length"], tokenizer=tok,
                            eval_mode=True)
    res["ppl"] = {}
    for tag, ch in [("uniform", uni), ("flip1", f1), ("flip16", f16)]:
        t0 = time.time()
        model = rdv.build_eval(config, r1_off=False, chosen=ch,
                               ring_r1=False, ring_size=S)
        frac, hi, tot = rdv.avg_fp8_fraction(model)
        ppl = eval_ppl(model, config, testenc)
        res["ppl"][tag] = dict(ppl=ppl, avg_fp8=frac,
                               n_fp8=sum(len(v) for v in ch.values()))
        print(f"[{mk}] PPL {tag}={ppl:.4f} avg_fp8={frac:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        cleanup_memory()
        save()

    u = res["ppl"]["uniform"]["ppl"]
    import math
    for tag in ("flip1", "flip16"):
        p = res["ppl"][tag]["ppl"]
        res[f"{tag}_dppl"] = p - u
        res[f"{tag}_dlogppl"] = math.log(p) - math.log(u)
        share = res[f"{tag}_dd_share"]
        res[f"{tag}_sens"] = (math.log(p) - math.log(u)) / share if share > 0 else None
    print(f"[{mk}] FLIP1 dPPL={res['flip1_dppl']:+.4f} "
          f"dlogPPL={res['flip1_dlogppl']:+.5f} sens={res['flip1_sens']:.4f} | "
          f"FLIP16 dPPL={res['flip16_dppl']:+.4f} "
          f"dlogPPL={res['flip16_dlogppl']:+.5f} sens={res['flip16_sens']:.4f}",
          flush=True)
    save()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--ring_size", type=int, default=64)
    ap.add_argument("--structure_only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.model, args)
    print("RD_WHYDECAY_DONE", flush=True)


if __name__ == "__main__":
    main()
