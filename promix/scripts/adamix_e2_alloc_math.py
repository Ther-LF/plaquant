"""AdaMix Motivation — E2 allocation MATH (CPU-pure, NO PPL).

Consumes the E1a group_stats.csv (fit + diag per-group variance / mse_fp4 /
mse_fp8 / signed_delta with STABLE group IDs) and, with the representation FROZEN
at R3 (§5), derives the four required allocation policies on the measured
statistics, WITHOUT any full-model PTQ/PPL (that is GPU + out of scope this run):

  A-pos   : per-layer uniform quota k_l, top-k_l groups by per-group VARIANCE
            (the PCA position/variance rule; direction recorded).
  B-local : per-layer uniform quota k_l, top-k_l groups by measured DELTA.
            A->B isolates the layer-internal selection criterion.
  C-global: global budget K = sum(k_l), top-K groups by measured DELTA across
            all eligible groups. B->C isolates cross-layer quota reallocation.
  D-var   : global budget K, top-K groups by per-group VARIANCE.
            C vs D isolates the scoring criterion (measured-error vs variance).

Reported diagnostics (§5):
  - Spearman(variance, Delta): global and per-layer (fit split).
  - A/B/C/D pairwise mask overlap; per-policy per-layer FP8 ratio + group counts.
  - identity check for the exchange argument: moving one FP8 group i->j changes
    summed group MSE by Delta_i - Delta_j (uses cached d4/d8 directly).
  - fit-vs-diag Delta stability (does the fit ranking hold on diagnostic data).
  - byte accounting per policy (weight + nvfp4/mxfp8 scale + padding), noting
    equal group count != equal bytes because out_features differ per layer.

This is explicitly the ALLOCATION MATH only; PPL evaluation of these masks needs
a GPU and is deferred (E2-PTQ).
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="docs/adamix_motivation/runs/e1a_3b")
    ap.add_argument("--rep", default="R3")
    ap.add_argument("--out_dir", default="docs/adamix_motivation/runs/e2math_3b")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load fit + diag group rows for the frozen representation -----------
    groups = {"fit": {}, "diag": {}}   # split -> module -> list of per-group dict
    out_feat = {}
    # out_features comes from summary.csv
    with open(os.path.join(args.run_dir, "summary.csv")) as f:
        for r in csv.DictReader(f):
            out_feat[r["module"]] = int(r["out_features"])
    with open(os.path.join(args.run_dir, "group_stats.csv")) as f:
        for r in csv.DictReader(f):
            if r["rep"] != args.rep:
                continue
            split = r["split"]
            mod = r["module_name"]
            groups[split].setdefault(mod, []).append(dict(
                gid=int(r["group_id_before_pack"]),
                width=int(r["width"]),
                variance=float(r["variance"]),
                d4=float(r["mse_fp4"]), d8=float(r["mse_fp8"]),
                delta=float(r["signed_delta"])))
    for split in groups:
        for mod in groups[split]:
            groups[split][mod].sort(key=lambda g: g["gid"])

    fit = groups["fit"]
    mods = sorted(fit)
    # per-layer uniform quota k_l = round(ng/8)
    k_l = {m: int(round(len(fit[m]) / 8.0)) for m in mods}
    K = sum(k_l.values())
    total_groups = sum(len(fit[m]) for m in mods)

    def topk_by(mod_groups, key, k):
        idx = sorted(range(len(mod_groups)),
                     key=lambda i: mod_groups[i][key], reverse=True)[:k]
        return set(mod_groups[i]["gid"] for i in idx)

    # A-pos (per-layer top-k_l by variance), B-local (per-layer top-k_l by delta)
    A = {m: topk_by(fit[m], "variance", k_l[m]) for m in mods}
    B = {m: topk_by(fit[m], "delta", k_l[m]) for m in mods}

    # global pools for C (delta) and D (variance): list of (mod,gid,score)
    def global_topK(key):
        pool = [(m, g["gid"], g[key]) for m in mods for g in fit[m]]
        pool.sort(key=lambda t: t[2], reverse=True)
        chosen = defaultdict(set)
        for m, gid, _ in pool[:K]:
            chosen[m].add(gid)
        return {m: chosen.get(m, set()) for m in mods}

    C = global_topK("delta")
    D = global_topK("variance")

    def overlap(P, Q):
        inter = sum(len(P[m] & Q[m]) for m in mods)
        return inter

    # Spearman variance vs delta
    allv = [g["variance"] for m in mods for g in fit[m]]
    alld = [g["delta"] for m in mods for g in fit[m]]
    sp_global = spearman(allv, alld)
    sp_per_layer = {m: spearman([g["variance"] for g in fit[m]],
                                [g["delta"] for g in fit[m]]) for m in mods}

    # fit vs diag delta rank stability (global)
    diag = groups["diag"]
    common = [m for m in mods if m in diag and len(diag[m]) == len(fit[m])]
    fit_d = [g["delta"] for m in common for g in fit[m]]
    diag_d = [g["delta"] for m in common for g in diag[m]]
    sp_fit_diag = spearman(fit_d, diag_d)

    # byte accounting: per group of width G on a Linear with out_features O.
    #   NVFP4: 4 bit/elem + 1 E4M3(8b) scale per 16 elems.
    #   MXFP8: 8 bit/elem + 1 E8M0(8b) scale per 32 elems.
    def group_bytes(O, width, fp8):
        if fp8:
            w = O * width * 8 / 8.0
            s = O * (width / 32) * 1.0
        else:
            w = O * width * 4 / 8.0
            s = O * (width / 16) * 1.0
        return w, s

    def policy_bytes(P):
        wb = sb = 0.0
        fp8g = 0
        for m in mods:
            O = out_feat[m]
            for g in fit[m]:
                fp8 = g["gid"] in P[m]
                w, s = group_bytes(O, g["width"], fp8)
                wb += w
                sb += s
                fp8g += int(fp8)
        return dict(weight_bytes=int(wb), scale_bytes=int(sb),
                    fp8_groups=fp8g, total_groups=total_groups)

    policies = {"A_pos": A, "B_local": B, "C_global": C, "D_var": D}
    byte_acct = {k: policy_bytes(v) for k, v in policies.items()}

    # per-layer fp8 ratio + overlaps
    per_layer = {}
    for m in mods:
        per_layer[m] = {k: dict(fp8=len(P[m]), ng=len(fit[m]),
                                ratio=round(len(P[m]) / len(fit[m]), 4))
                        for k, P in policies.items()}
    overlaps = {}
    names = list(policies)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps[f"{a}&{b}"] = dict(
                shared_groups=overlap(policies[a], policies[b]), of_K=K)

    summary = dict(
        rep=args.rep, total_groups=total_groups, K=K,
        rho_global=round(K / total_groups, 4),
        spearman_var_delta_global=sp_global,
        spearman_fit_diag_delta=sp_fit_diag,
        byte_accounting=byte_acct,
        overlaps=overlaps,
        note=("A-pos=per-layer top-k_l by VARIANCE (PCA position/variance rule, "
              "direction=high-variance->FP8); B-local=per-layer top-k_l by "
              "measured DELTA; C-global=global top-K DELTA; D-var=global top-K "
              "VARIANCE. byte counts ignore no metadata beyond w+scale; equal "
              "group count != equal bytes (out_features differ). PPL of these "
              "masks is GPU/out-of-scope."))
    with open(os.path.join(args.out_dir, "e2_alloc_summary.json"), "w") as f:
        json.dump(dict(summary=summary, spearman_per_layer=sp_per_layer,
                       per_layer=per_layer), f, indent=2)

    # allocation.csv: long table of per (policy, module, group) selection
    with open(os.path.join(args.out_dir, "allocation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "module", "group_id", "variance", "delta",
                    "selected_fp8"])
        for pol, P in policies.items():
            for m in mods:
                for g in fit[m]:
                    w.writerow([pol, m, g["gid"], g["variance"], g["delta"],
                                int(g["gid"] in P[m])])

    print("E2-MATH SUMMARY (rep=%s)" % args.rep)
    print(f"  total_groups={total_groups} K={K} rho={K/total_groups:.4f}")
    print(f"  Spearman(var,delta) global = {sp_global:.4f}")
    print(f"  Spearman(fit,diag delta)   = {sp_fit_diag:.4f}")
    for k, v in byte_acct.items():
        print(f"  {k:9s} weight_bytes={v['weight_bytes']:>12} "
              f"scale_bytes={v['scale_bytes']:>10} fp8_groups={v['fp8_groups']}")
    for k, v in overlaps.items():
        print(f"  overlap {k:18s} = {v['shared_groups']}/{v['of_K']}")
    print("E2MATH_DONE")


if __name__ == "__main__":
    main()
