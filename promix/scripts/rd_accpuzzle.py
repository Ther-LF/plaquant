"""rd-accpuzzle: WHY the first ~1/32 FP8 budget erases the pure-NVFP4 PPL cliff.

Offline concentration analysis of the per-ring upgrade benefit ΔD = D_fp4 - D_fp8
and its tie to the PCA variance spectrum, on the SAME R1-off rotated calibration
activations and the SAME global-ΔD ranking that rd_sweep's measured allocation uses.

For each model it emits, to a JSON:
  * total_rings over the 6 reallocated Linears (q/k/v/gate/up/down_proj) -- the exact
    pool rd_validity.measured_alloc ranks; o_proj excluded (held uniform, as in sweep).
  * ΔD concentration: cumulative fraction of total (and positive) ΔD captured by the
    top-k rings, at k = round(frac * total_rings) for frac in the low-end grid
    {1/256,1/128,1/96,1/64,1/48,1/32,1/24,1/16,1/8,1/4,1/2,1}. Because measured_alloc
    upgrades exactly the top-(frac*total_rings) ΔD rings, this cumΔD(frac) curve is
    the "fraction of ALL available mixed-precision benefit captured at budget=frac".
  * variance concentration: same curve on per-ring measured variance (trace fraction).
  * ΔD vs variance coupling: Spearman rho + top-1/32 ring-set overlap.
  * PCA eigenvalue spectrum (E file): top-frac-of-channels share of trace, per
    representative basis (self_attn_mlp, down_proj), averaged over layers -- an
    independent corroboration that the rotated variance is outlier-concentrated.

No model edit, no GPTQ, no eval -- just calibration forward hooks (cheap).
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transformers

import rd_validity as rdv
import rd_sweep as rds
from promix.eval import ptq as ptqmod
from promix.eval.data import get_wikitext2
from promix.utils import cleanup_memory

FRACS = [1/256, 1/128, 1/96, 1/64, 1/48, 1/32, 1/24, 1/16, 1/8, 1/4, 1/2, 1.0]


def concentration_curve(per_ring_values, fracs):
    """Cumulative fraction of the summed value captured by the top-k rings,
    at k = round(frac*total). Returns list of dicts. Reports both the total
    sum (signed) and the positive-only sum (the realizable benefit)."""
    v = torch.as_tensor(per_ring_values, dtype=torch.float64).flatten()
    total = int(v.numel())
    v_desc, _ = torch.sort(v, descending=True)
    cum = torch.cumsum(v_desc, dim=0)
    sum_all = float(v.sum())
    sum_pos = float(v.clamp_min(0).sum())
    out = []
    for f in fracs:
        k = int(round(f * total))
        k = max(1, min(total, k))
        c = float(cum[k - 1])
        out.append(dict(frac=f, k=k,
                        cum_over_total=c / sum_all if sum_all != 0 else float('nan'),
                        cum_over_positive=c / sum_pos if sum_pos != 0 else float('nan')))
    return out, total, sum_all, sum_pos


def spearman(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm())
    return float((ra * rb).sum() / denom) if float(denom) > 0 else float('nan')


def eig_spectrum(E_path, fracs):
    """For each representative basis family, average over layers the top-frac-of-
    channels share of eigenvalue trace (eigenvalues stored ascending -> take tail)."""
    if not os.path.exists(E_path):
        return {"error": f"missing {E_path}"}
    ev = torch.load(E_path, map_location="cpu")
    fams = {}
    for key, e in ev.items():
        if not torch.is_tensor(e):
            continue
        e = e.to(torch.float64).flatten().clamp_min(0)
        if e.numel() < 8:
            continue
        # family = strip the "layer.N." prefix
        fam = ".".join(p for p in key.split(".") if not p.isdigit())
        fams.setdefault(fam, []).append(e)
    summary = {}
    for fam, elist in fams.items():
        rows = []
        for f in fracs:
            shares = []
            for e in elist:
                n = e.numel()
                k = max(1, int(round(f * n)))
                top = torch.sort(e, descending=True).values[:k].sum()
                shares.append(float(top / e.sum()))
            rows.append(dict(frac=f, top_share=sum(shares) / len(shares)))
        summary[fam] = dict(n_layers=len(elist), dim=int(elist[0].numel()), curve=rows)
    return summary


def run_model(model_key, config_path, args):
    config = ptqmod.load_config(os.path.join(REPO, config_path))
    config = rds.resolve_paths(config, model_key)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    config["eval"]["batch_size"] = 16

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    print(f"START accpuzzle model={model_key} basis={os.path.basename(config['paths']['basis'])}",
          flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, r1_off=True)
    cmodel.to(rdv.DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()

    names = list(stats)                       # the 6 ALLOC Linears x layers
    all_dd = torch.cat([torch.as_tensor(stats[n]["deltaD"]).flatten() for n in names])
    all_var = torch.cat([torch.as_tensor(stats[n]["ring_var"]).flatten() for n in names])
    total_rings = int(all_dd.numel())
    assert total_rings == sum(stats[n]["nr"] for n in names)

    dd_curve, _, dd_sum_all, dd_sum_pos = concentration_curve(all_dd, FRACS)
    var_curve, _, var_sum_all, _ = concentration_curve(all_var, FRACS)

    # ΔD vs variance coupling
    rho = spearman(all_dd, all_var)
    k32 = max(1, int(round(total_rings / 32)))
    top_dd = set(torch.argsort(all_dd, descending=True)[:k32].tolist())
    top_var = set(torch.argsort(all_var, descending=True)[:k32].tolist())
    overlap32 = len(top_dd & top_var) / k32

    E_summary = eig_spectrum(config["paths"]["basis"].replace("/U-", "/E-"), FRACS)

    result = dict(
        model=model_key, name=config["model"]["name"],
        basis=os.path.basename(config["paths"]["basis"]),
        total_rings=total_rings, n_units=len(names),
        collect_tokens=args.collect_tokens, calib_samples=args.calib_samples,
        dd_sum_all=dd_sum_all, dd_sum_positive=dd_sum_pos,
        dd_frac_negative=float((all_dd < 0).float().mean()),
        var_sum=var_sum_all,
        deltaD_concentration=dd_curve,
        variance_concentration=var_curve,
        spearman_dd_var=rho, top32_ring_overlap=overlap32,
        eig_spectrum=E_summary,
        per_unit_nr={n: int(stats[n]["nr"]) for n in names},
    )
    print(f"  total_rings={total_rings} dd_sum_pos={dd_sum_pos:.3e} "
          f"frac_neg={result['dd_frac_negative']:.3f} rho={rho:.3f} "
          f"overlap(top1/32)={overlap32:.3f}", flush=True)
    for row in dd_curve:
        print(f"  cumΔD frac={row['frac']:.5f} k={row['k']:4d} "
              f"over_pos={row['cum_over_positive']:.4f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--config", default="promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--out", default="/home/luofan/RD_ACCPUZZLE_concentration.json")
    args = ap.parse_args()

    out = {"fracs": FRACS, "models": {}}
    for mk in args.models.split(","):
        out["models"][mk] = run_model(mk, args.config, args)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        cleanup_memory()
    print(f"WROTE {args.out}", flush=True)
    print("RD_ACCPUZZLE_DONE", flush=True)


if __name__ == "__main__":
    main()
