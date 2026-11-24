#!/usr/bin/env python
"""plot.py — Motivation §3.2 figures A/B from runs.csv / magnitude_stats.csv /
layer_metrics.csv + rotated run jsons (quantile curves of representative layers).

Fig A (motivation-representation-distribution): at preset r=0.1 (actual aligned
point), representative shallow/mid/deep q_proj & down_proj: abs-activation
quantile curves Direct vs Rotated per group (H/L), + all-layer block max/RMS
summary + per-group A. Color = condition (Direct blue, Rotated orange — fixed).
Fig B (motivation-representation-quality): vs actual FP8 fraction: A_H, A_L,
E (mean ± between-layer std), wiki/C4 PPL, Direct solid vs Rotated dashed.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M31DATA = os.path.join(os.path.dirname(HERE), "motivation_3_1", "data")
CD = {"direct": "#0072B2", "rotated": "#E69F00"}   # fixed condition colors
MODELS = {"3b": "Llama-3.2-3B", "8b": "Llama-3-8B"}


def load_runs():
    rows = list(csv.DictReader(open(os.path.join(HERE, "runs.csv"))))
    for r in rows:
        for k in ("ratio_target", "ratio_actual", "wiki_ppl", "c4_ppl",
                  "A_high_mean", "A_high_std", "A_low_mean", "A_low_std",
                  "E_mean", "E_std"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = None
    return rows


def style(ax, xl, yl, t=None):
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    if t:
        ax.set_title(t, fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def groups_for(mk, variant, rt, rot_seed=0):
    """Return {module: {group: statdict}} for one point."""
    if variant == "rotated":
        p = os.path.join(HERE, "data", f"{mk}_rotated_rot{rot_seed}_r{rt:g}.json")
        if not os.path.exists(p):
            return None, None
        d = json.load(open(p))
        return d["groups"], d.get("rep_modules")
    p = os.path.join(HERE, "data", f"{mk}_direct_stats.json")
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))["ratios"].get(f"{rt:g}")
    return (d["groups"], d.get("rep_modules")) if d else (None, None)


def fig_a(rt=0.1):
    for mk, mname in MODELS.items():
        gd, repd = groups_for(mk, "direct", rt)
        gr, repr_ = groups_for(mk, "rotated", rt)
        if not gd or not gr:
            continue
        reps = [m for m in (repd or []) if m in gr]
        fig, axes = plt.subplots(3, len(reps),
                                 figsize=(2.6 * len(reps), 8.2), squeeze=False)
        for ci, mod in enumerate(reps):
            for gi, grp in enumerate(("high", "low")):
                ax = axes[gi][ci]
                for variant, g in (("direct", gd), ("rotated", gr)):
                    s = g[mod].get(grp)
                    if not s or "abs_quantile_curve" not in s:
                        continue
                    q = s["abs_quantile_curve"]
                    xs = [i / (len(q) - 1) for i in range(len(q))]
                    ax.plot(xs, q, color=CD[variant], lw=1.5,
                            label=variant.capitalize()
                            if ci == 0 and gi == 0 else None)
                style(ax, "quantile", "|z|",
                      f"{mod.split('model.layers.')[-1]}\n"
                      f"{'H (MXFP8)' if grp == 'high' else 'L (NVFP4)'}")
                ax.set_yscale("log")
            # per-group A bars for this module
            ax = axes[2][ci]
            xs, hs, cs = [], [], []
            for vi, (variant, g) in enumerate((("direct", gd), ("rotated", gr))):
                for gi2, grp in enumerate(("high", "low")):
                    s = g[mod].get(grp)
                    if s:
                        xs.append(gi2 * 2.4 + vi)
                        hs.append(s["A"])
                        cs.append(CD[variant])
            ax.bar(xs, hs, color=cs, width=0.8)
            ax.set_xticks([0.5, 2.9])
            ax.set_xticklabels(["H", "L"], fontsize=7)
            style(ax, "", "A (act. quant. err.)")
            ax.set_yscale("log")
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{mname} r_target={rt}: abs-activation quantile curves "
                     "(log y) + per-group activation quant error; "
                     "Direct vs Within-group rotation", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                HERE, f"motivation-representation-distribution-{mk}.{ext}"),
                dpi=200)
        plt.close(fig)


def fig_b():
    rows = load_runs()
    fig, axes = plt.subplots(len(MODELS), 5, figsize=(16, 3.1 * len(MODELS)),
                             squeeze=False)
    for mi, (mk, mname) in enumerate(MODELS.items()):
        panels = (("A_high_mean", "A_high_std", "A — H group (MXFP8)"),
                  ("A_low_mean", "A_low_std", "A — L group (NVFP4)"),
                  ("E_mean", "E_std", "layer output E"),
                  ("wiki_ppl", None, "WikiText-2 PPL"),
                  ("c4_ppl", None, "C4 PPL"))
        for pi, (key, sdk, title) in enumerate(panels):
            ax = axes[mi][pi]
            for variant, ls in (("direct", "-"), ("rotated", "--")):
                pts = sorted((r["ratio_actual"], r) for r in rows
                             if r["model"] == mk and r["variant"] == variant
                             and str(r.get("rotation_seed", "")) in ("", "0")
                             and r["ratio_actual"] is not None
                             and r[key] is not None)
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                ys = [r[key] for _, r in pts]
                ax.plot(xs, ys, ls, color=CD[variant], lw=1.6,
                        label=variant.capitalize() if pi == 0 and mi == 0
                        else None)
                if sdk:
                    sd = [r[sdk] for _, r in pts]
                    ax.fill_between(xs, [a - b for a, b in zip(ys, sd)],
                                    [a + b for a, b in zip(ys, sd)],
                                    color=CD[variant], alpha=0.15, lw=0)
            style(ax, "actual FP8 channel fraction", "", f"{mname} — {title}")
        axes[mi][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Fixed AV-within allocation: Direct vs within-precision-group "
                 "rotation; bands = between-layer std (ddof=0), seed0 pairs",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(
            HERE, f"motivation-representation-quality.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_a(0.1)
    fig_b()
    print("wrote figures to", HERE)
