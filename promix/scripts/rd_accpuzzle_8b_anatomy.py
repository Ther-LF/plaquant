"""rd-accpuzzle: 8B break-even anatomy.

WHY does the measured-MSE allocation stay ~break-even vs uniform on 8B @1/8?
Three diagnostics, all R1-off (same regime as the cliff analysis / the loaded
machinery; R-on shifts absolute PPL ~0.1 but not the selection structure):

  1. SELECTION-DIFFERENCE: measured vs uniform ring-sets at matched 1/8 budget.
     overlap %, |added|, |dropped|. Then the TRUE per-ring PPL impact:
       X = PPL(uniform - dropped) - PPL(uniform)   (cost of dropping uniform's rings)
       Y = PPL(uniform) - PPL(uniform + added)     (benefit of adding measured's rings)
     uniform - dropped = uniform ∩ measured (overlap-only);
     uniform + added   = uniform ∪ measured.
     Also PPL(uniform) and PPL(measured) for reference.
  2. GRANULARITY: per-LAYER total ΔD (summed over each layer's rings); CV across
     layers, 8B vs 3B — is there cross-layer structure a global ring threshold averages out?
  3. DATASET: PPL(uniform) vs PPL(measured) on C4 (if loadable) — does break-even hold off wikitext?

Writes a JSON; a local step folds it into docs/rd_accpuzzle_report.md.
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transformers
import rd_validity as rdv
import rd_sweep as rds
from rd_validity import RING
from promix.quantize.adaptive import allocate_by_budget
from promix.eval import ptq as ptqmod
from promix.eval.data import get_wikitext2
from promix.eval.evaluator import evaluate_ppl
from promix.utils import DEV, cleanup_memory


def layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def per_layer_deltaD_cv(stats):
    by_layer = {}
    for n, st in stats.items():
        li = layer_idx(n)
        by_layer.setdefault(li, 0.0)
        by_layer[li] += float(torch.as_tensor(st["deltaD"]).clamp_min(0).sum())
    vals = torch.tensor(list(by_layer.values()), dtype=torch.float64)
    cv = float(vals.std(unbiased=False) / vals.mean()) if vals.mean() > 0 else float("nan")
    return cv, {int(k): float(v) for k, v in sorted(by_layer.items())}


def uniform_chosen(stats):
    """Per-layer uniform: each Linear protects its own top round(nr/8) variance rings
    (= the shipped last-K_high tail)."""
    ch = {}
    for n, st in stats.items():
        nr = st["nr"]
        k = int(round(0.125 * nr))
        rv = torch.as_tensor(st["ring_var"])
        ch[n] = set(torch.topk(rv, k).indices.tolist()) if k > 0 else set()
    return ch


def measured_chosen(stats, budget_rings, total_rings):
    names = list(stats)
    masks = allocate_by_budget([stats[n]["deltaD"] for n in names],
                               target_mean_fp8=budget_rings / total_rings)
    return {n: set(torch.nonzero(m).flatten().tolist()) for n, m in zip(names, masks)}


def get_c4(nsamples, seqlen, tokenizer):
    import datasets
    try:
        d = datasets.load_dataset("allenai/c4", "en", split="validation",
                                   streaming=True)
    except Exception as ex:
        print(f"C4 load failed: {ex}", flush=True)
        return None
    texts, ntok_target = [], nsamples * seqlen + seqlen
    got = 0
    for row in d:
        texts.append(row["text"])
        got += len(row["text"].split())
        if got > ntok_target * 1.5:
            break
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    ids = enc.input_ids[:, : (nsamples * seqlen)]
    return type("E", (), {"input_ids": ids})()


def eval_on(model, config, testenc):
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    return float(evaluate_ppl(model, testenc, DEV, batch_size=config["eval"]["batch_size"]))


def run(model_key, args):
    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rds.resolve_paths(config, model_key)
    config["eval"]["batch_size"] = args.eval_bs
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])

    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, r1_off=True); cmodel.to(DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel; cleanup_memory()

    total_rings = sum(stats[n]["nr"] for n in stats)
    budget_rings = sum(int(round(0.125 * stats[n]["nr"])) for n in stats)
    uni = uniform_chosen(stats)
    mea = measured_chosen(stats, budget_rings, total_rings)
    n_uni = sum(len(v) for v in uni.values())
    n_mea = sum(len(v) for v in mea.values())
    overlap = sum(len(uni[n] & mea[n]) for n in stats)
    added = {n: (mea[n] - uni[n]) for n in stats}
    dropped = {n: (uni[n] - mea[n]) for n in stats}
    n_added = sum(len(v) for v in added.values())
    n_dropped = sum(len(v) for v in dropped.values())
    cv, per_layer = per_layer_deltaD_cv(stats)

    res = dict(model=model_key, total_rings=total_rings, budget_rings=budget_rings,
               n_uniform=n_uni, n_measured=n_mea, overlap=overlap,
               overlap_pct=overlap / max(budget_rings, 1) * 100,
               n_added=n_added, n_dropped=n_dropped,
               per_layer_deltaD_cv=cv, per_layer_deltaD=per_layer)
    print(f"[{model_key}] total={total_rings} budget={budget_rings} overlap={overlap} "
          f"({res['overlap_pct']:.1f}%) added={n_added} dropped={n_dropped} CV={cv:.4f}",
          flush=True)

    if args.cv_only:
        return res

    # --- PPL flips (wikitext) ---
    testenc = get_wikitext2(seed=config["calibration"].get("seed", 0),
                            seqlen=config["eval"]["max_length"], tokenizer=tok,
                            eval_mode=True)
    overlap_only = {n: (uni[n] & mea[n]) for n in stats}   # uniform - dropped
    union = {n: (uni[n] | mea[n]) for n in stats}          # uniform + added
    configs = [("uniform", uni), ("measured", mea),
               ("overlap_only", overlap_only), ("union", union)]
    wk = {}
    for tag, ch in configs:
        t0 = time.time()
        model = rdv.build_eval(config, r1_off=True, chosen=ch)
        frac, hi, tot = rdv.avg_fp8_fraction(model)
        ppl = eval_on(model, config, testenc)
        wk[tag] = dict(ppl=ppl, avg_fp8=frac, n_fp8=sum(len(v) for v in ch.values()))
        print(f"  [wk] {tag} ppl={ppl:.4f} avg_fp8={frac:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        del model; cleanup_memory()
    res["wikitext"] = wk
    res["X_drop_cost"] = wk["overlap_only"]["ppl"] - wk["uniform"]["ppl"]
    res["Y_add_benefit"] = wk["uniform"]["ppl"] - wk["union"]["ppl"]
    res["measured_minus_uniform"] = wk["measured"]["ppl"] - wk["uniform"]["ppl"]
    print(f"  X(drop_cost)={res['X_drop_cost']:.4f}  Y(add_benefit)={res['Y_add_benefit']:.4f}  "
          f"measured-uniform={res['measured_minus_uniform']:.4f}", flush=True)

    # --- C4 dataset check ---
    if not args.skip_c4:
        c4 = get_c4(args.c4_samples, config["eval"]["max_length"], tok)
        if c4 is not None:
            c4r = {}
            for tag, ch in [("uniform", uni), ("measured", mea)]:
                model = rdv.build_eval(config, r1_off=True, chosen=ch)
                ppl = eval_on(model, config, c4)
                c4r[tag] = ppl
                print(f"  [c4] {tag} ppl={ppl:.4f}", flush=True)
                del model; cleanup_memory()
            c4r["measured_minus_uniform"] = c4r["measured"] - c4r["uniform"]
            res["c4"] = c4r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b")
    ap.add_argument("--config", default="promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--c4_samples", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--cv_only", action="store_true")
    ap.add_argument("--skip_c4", action="store_true")
    ap.add_argument("--out", default="/home/luofan/RD_ACCPUZZLE_8b_anatomy.json")
    args = ap.parse_args()
    out = {}
    for mk in args.model.split(","):
        out[mk] = run(mk, args)
        json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"WROTE {args.out}", flush=True)
    print("RD_ANATOMY_DONE", flush=True)


if __name__ == "__main__":
    main()
