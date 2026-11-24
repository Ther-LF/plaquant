#!/usr/bin/env python
"""plot.py — Motivation §3.1 figures A/B/C from runs.csv / layer_errors.csv.

Run from repo root:  python docs/motivation_3_1/plot.py
Outputs motivation-allocation-within.{pdf,png}, motivation-allocation-across.{pdf,png},
motivation-allocation-diag.{pdf,png} into docs/motivation_3_1/.

Caption facts (encoded here, restated in the paper): x = actual FP8 channel
fraction; PPL seqlen 2048 (wiki test full / C4 val 256 win); error band =
BETWEEN-LAYER std (ddof=0) of per-linear relative squared output error on the
held-out teacher-input set, NOT a confidence interval; RND PPL = mean of 3
assignment seeds (band = seed min/max), RND error band = layer std of
seed-averaged per-layer E; AV uses the PCA (ResQ-style) representation.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Okabe-Ito (CVD-safe), fixed assignment — never cycled
C = dict(rnd="#8C8C8C", ss="#0072B2", ma="#E69F00", av="#009E73")
LBL = dict(rnd="Random", ss="Squared-sum", ma="Mean-abs",
           av="Act-variance (PCA repr.)")
MODELS = {"3b": "Llama-3.2-3B", "8b": "Llama-3-8B"}


def load_runs():
    rows = list(csv.DictReader(open(os.path.join(HERE, "runs.csv"))))
    for r in rows:
        for k in ("target_ratio", "actual_ratio", "wiki_ppl", "c4_ppl",
                  "error_mean_layers", "error_std_layers"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = None
    return [r for r in rows if str(r["status"]).startswith(("ok", "reused"))]


def load_layer_errors():
    out = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "layer_errors.csv"))):
        out[r["run_id"]][r["module_id"]] = float(r["relative_squared_error"])
    return out


def series(rows, model, strat, scope, seed="0"):
    pts = [(r["actual_ratio"], r) for r in rows
           if r["model"] == model and r["strategy"] == strat
           and r["scope"] == scope and r["seed"] == seed
           and r["actual_ratio"] is not None]
    return sorted(pts)


def rnd_aggregated(rows, model, lerr):
    """PPL: per-ratio mean + min/max over seeds. Error: per-layer E averaged
    over seeds first, then layer mean/std (caption-documented)."""
    byr = defaultdict(list)
    for s in ("0", "1", "2"):
        for x, r in series(rows, model, "rnd", "within", s):
            byr[round(x, 6)].append(r)
    out = []
    for x in sorted(byr):
        rs = byr[x]
        wik = [r["wiki_ppl"] for r in rs]
        c4 = [r["c4_ppl"] for r in rs]
        mods = defaultdict(list)
        for r in rs:
            for m, e in lerr.get(r["run_id"], {}).items():
                mods[m].append(e)
        es = [sum(v) / len(v) for v in mods.values()]
        em = sum(es) / len(es) if es else None
        sd = (sum((e - em) ** 2 for e in es) / len(es)) ** 0.5 if es else None
        out.append(dict(x=x, wiki=sum(wik) / len(wik), wiki_lo=min(wik),
                        wiki_hi=max(wik), c4=sum(c4) / len(c4),
                        c4_lo=min(c4), c4_hi=max(c4), e=em, e_std=sd))
    return out


def style_ax(ax, xlabel, ylabel, title=None):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_a(rows, lerr):
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(10.5, 3.2 * len(MODELS)))
    if len(MODELS) == 1:
        axes = [axes]
    for mi, (mk, mname) in enumerate(MODELS.items()):
        axw, axc, axe = axes[mi]
        for strat in ("rnd", "ss", "ma", "av"):
            if strat == "rnd":
                agg = rnd_aggregated(rows, mk, lerr)
                if not agg:
                    continue
                xs = [a["x"] for a in agg]
                axw.plot(xs, [a["wiki"] for a in agg], color=C[strat], lw=1.6,
                         label=LBL[strat])
                axw.fill_between(xs, [a["wiki_lo"] for a in agg],
                                 [a["wiki_hi"] for a in agg],
                                 color=C[strat], alpha=0.18, lw=0)
                axc.plot(xs, [a["c4"] for a in agg], color=C[strat], lw=1.6)
                axc.fill_between(xs, [a["c4_lo"] for a in agg],
                                 [a["c4_hi"] for a in agg],
                                 color=C[strat], alpha=0.18, lw=0)
                axe.plot(xs, [a["e"] for a in agg], color=C[strat], lw=1.6)
                axe.fill_between(xs, [a["e"] - a["e_std"] for a in agg],
                                 [a["e"] + a["e_std"] for a in agg],
                                 color=C[strat], alpha=0.18, lw=0)
            else:
                pts = series(rows, mk, strat, "within")
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                axw.plot(xs, [r["wiki_ppl"] for _, r in pts], color=C[strat],
                         lw=1.6, label=LBL[strat])
                axc.plot(xs, [r["c4_ppl"] for _, r in pts], color=C[strat],
                         lw=1.6)
                e = [r["error_mean_layers"] for _, r in pts]
                sd = [r["error_std_layers"] for _, r in pts]
                axe.plot(xs, e, color=C[strat], lw=1.6)
                axe.fill_between(xs, [a - b for a, b in zip(e, sd)],
                                 [a + b for a, b in zip(e, sd)],
                                 color=C[strat], alpha=0.18, lw=0)
        style_ax(axw, "actual FP8 channel fraction", "WikiText-2 PPL",
                 f"{mname} — WikiText-2")
        style_ax(axc, "actual FP8 channel fraction", "C4 PPL",
                 f"{mname} — C4")
        style_ax(axe, "actual FP8 channel fraction",
                 "layer rel. sq. output error (mean ± layer std)",
                 f"{mname} — local error")
        if mi == 0:
            axw.legend(fontsize=8, frameon=False)
    fig.suptitle("Within-layer channel-selection strategies (M31-A); "
                 "RND band = 3 assignment seeds; error band = between-layer std",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"motivation-allocation-within.{ext}"),
                    dpi=200)
    plt.close(fig)


def fig_b(rows):
    crits = ("ss", "ma", "av")
    fig, axes = plt.subplots(3, len(crits) * len(MODELS),
                             figsize=(3.2 * len(crits) * len(MODELS), 8.4))
    col = 0
    for mk, mname in MODELS.items():
        for ci, strat in enumerate(crits):
            for row, (metric, ylab) in enumerate((
                    ("wiki_ppl", "WikiText-2 PPL"), ("c4_ppl", "C4 PPL"),
                    ("error_mean_layers", "layer E mean ± layer std"))):
                ax = axes[row][col + ci]
                for scope, ls in (("within", "-"), ("across", "--")):
                    pts = series(rows, mk, strat, scope)
                    if not pts:
                        continue
                    xs = [x for x, _ in pts]
                    ys = [r[metric] for _, r in pts]
                    ax.plot(xs, ys, ls, color=C[strat], lw=1.6,
                            label=f"{scope}")
                    if metric == "error_mean_layers":
                        sd = [r["error_std_layers"] for _, r in pts]
                        ax.fill_between(xs, [a - b for a, b in zip(ys, sd)],
                                        [a + b for a, b in zip(ys, sd)],
                                        color=C[strat],
                                        alpha=0.18 if scope == "within" else 0.10,
                                        lw=0)
                style_ax(ax, "actual FP8 fraction" if row == 2 else "",
                         ylab if ci == 0 and col == 0 else "",
                         f"{mname} · {LBL[strat]}" if row == 0 else None)
                if row == 0 and ci == 0 and col == 0:
                    ax.legend(fontsize=8, frameon=False)
        col += len(crits)
    fig.suptitle("Within vs Across-layer allocation per criterion (M31-B); "
                 "solid=within, dashed=across; same total FP8 budget H(r)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"motivation-allocation-across.{ext}"),
                    dpi=200)
    plt.close(fig)


def fig_c(rows, lerr):
    """Diagnostic: per-layer E and per-layer FP8 fraction at r≈0.1/0.5/0.9."""
    targets = ("0.1", "0.5", "0.9")
    allocs = {}
    for p in sorted(os.listdir(os.path.join(HERE, "data"))):
        if p.endswith(".json"):
            d = json.load(open(os.path.join(HERE, "data", p)))
            if d.get("alloc"):
                allocs[d["run_id"]] = d["alloc"]
    fig, axes = plt.subplots(2 * len(MODELS), len(targets),
                             figsize=(11, 5.6 * len(MODELS)))
    for mi, mk in enumerate(MODELS):
        for ti, rt in enumerate(targets):
            axE = axes[2 * mi][ti]
            axF = axes[2 * mi + 1][ti]
            for strat in ("rnd", "ss", "ma", "av"):
                for scope, ls in (("within", "-"), ("across", "--")):
                    if strat == "rnd" and scope == "across":
                        continue
                    rid = f"{mk}_{strat}_{scope}_s0_r{rt}"
                    if rid not in lerr:
                        continue
                    mods = sorted(lerr[rid])
                    es = [lerr[rid][m] for m in mods]
                    axE.plot(range(len(mods)), es, ls, color=C[strat], lw=0.9,
                             alpha=0.85,
                             label=f"{LBL[strat]} {scope}"
                             if ti == 0 and mi == 0 else None)
                    a = allocs.get(rid)
                    if a:
                        fr = [a[m]["h"] / a[m]["d"] for m in mods]
                        axF.plot(range(len(mods)), fr, ls, color=C[strat],
                                 lw=0.9, alpha=0.85)
            style_ax(axE, "", "per-linear E",
                     f"{MODELS[mk]} r_target={rt}")
            style_ax(axF, "linear index (layer-major)", "FP8 fraction")
    axes[0][0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Diagnostic: per-linear relative sq. output error and FP8 "
                 "fraction at preset r=0.1/0.5/0.9 (actual quotas)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"motivation-allocation-diag.{ext}"),
                    dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_runs()
    lerr = load_layer_errors()
    fig_a(rows, lerr)
    fig_b(rows)
    fig_c(rows, lerr)
    print("wrote figures to", HERE)
