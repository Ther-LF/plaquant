#!/usr/bin/env python
"""rd_m31_ext_report.py — M31-ext gates + report.md generator.

Reads docs/motivation_3_1/ (legacy, read-only) and docs/motivation_3_1_ext/
(new runs, backfill, dd scores), checks Gates 1-3, and writes
docs/motivation_3_1_ext/report.md with the headline tables
(spec: docs/paper_motivation_3_1_ext_experiments.md section 6/7).
"""
import csv
import glob
import json
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEG = os.path.join(REPO, "docs", "motivation_3_1")
EXT = os.path.join(REPO, "docs", "motivation_3_1_ext")

REPORT_R = ("0.02", "0.04", "0.06", "0.08", "0.1", "0.2", "0.5")
STRATS = (("rnd", "within"), ("ma", "within"), ("ss", "within"),
          ("dd", "within"), ("ma", "across"), ("ss", "across"),
          ("dd", "across"))


def load_runs(data_dir):
    out = {}
    for p in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "run_id" in d:
            out[d["run_id"]] = d
    return out


def main():
    leg = load_runs(os.path.join(LEG, "data"))
    ext = load_runs(os.path.join(EXT, "data"))
    bf = {}
    p = os.path.join(EXT, "act_mse_backfill.csv")
    if os.path.exists(p):
        for row in csv.DictReader(open(p)):
            bf[row["run_id"]] = row

    def point(mk, strat, scope, r, seed=0):
        """(wiki, c4, act_mse_global, E_mean, src) for a logical point."""
        rid = f"{mk}_{strat}_{scope}_s{seed}_r{r}"
        d = ext.get(rid) or leg.get(rid)
        if d is None:
            return None
        am = d.get("act_mse_global")
        if am is None and rid in bf:
            am = float(bf[rid]["act_mse_global"])
        return dict(wiki=d["wiki"]["ppl"], c4=d["c4"]["ppl"], act_mse=am,
                    emean=d["error_mean_layers"],
                    src="ext" if rid in ext else "legacy", d=d)

    lines = []
    w = lines.append
    w("# M31-ext report — format-aware cross-layer criterion (DD) + "
      "densified low-budget grid")
    w("")
    w("Protocol: docs/paper_motivation_3_1_ext_experiments.md; runner "
      "promix/scripts/rd_m31.py (ext mode). All new artifacts live in "
      "docs/motivation_3_1_ext/; legacy docs/motivation_3_1/ untouched.")
    w("")

    # ---------------- Gate 1 -----------------------------------------------
    w("## Gate 1 — endpoint consistency (new runner vs legacy, tol 1e-4)")
    w("")
    w("| model | r | d_wiki | d_c4 | d_E_mean | pass |")
    w("|---|---|---|---|---|---|")
    g1 = True
    for mk in ("3b", "8b"):
        for r in ("0", "1"):
            rid = f"{mk}_ss_within_s0_r{r}"
            if rid not in ext or rid not in leg:
                w(f"| {mk} | {r} | - | - | - | MISSING |")
                g1 = False
                continue
            dw = abs(ext[rid]["wiki"]["ppl"] - leg[rid]["wiki"]["ppl"])
            dc = abs(ext[rid]["c4"]["ppl"] - leg[rid]["c4"]["ppl"])
            de = abs(ext[rid]["error_mean_layers"] - leg[rid]["error_mean_layers"])
            ok = dw <= 1e-4 and dc <= 1e-4 and de <= 1e-4
            g1 &= ok
            w(f"| {mk} | {r} | {dw:.2e} | {dc:.2e} | {de:.2e} | "
              f"{'PASS' if ok else 'FAIL'} |")
    w("")
    w(f"Gate 1: **{'PASS' if g1 else 'FAIL'}**")
    w("")

    # ---------------- Gate 2 -----------------------------------------------
    w("## Gate 2 — exact quota conservation (H identical across strategies/"
      "scopes at each (model, r))")
    w("")
    g2 = True
    bad = []
    byr = {}
    for d in list(ext.values()) + [v for v in leg.values()
                                   if v.get("representation") == "raw"]:
        if d.get("strategy") in (None, "float"):
            continue
        key = (d["model"], str(d.get("target_ratio")))
        byr.setdefault(key, set()).add(d["H"])
    for key, hs in sorted(byr.items()):
        if len(hs) != 1:
            g2 = False
            bad.append((key, sorted(hs)))
    if bad:
        for key, hs in bad:
            w(f"- MISMATCH {key}: H set {hs}")
    w(f"Gate 2 over {len(byr)} (model, r) groups "
      f"(raw-representation runs, legacy+ext): "
      f"**{'PASS' if g2 else 'FAIL'}**")
    w("")

    # ---------------- Gate 3 -----------------------------------------------
    nan_bad = [rid for rid, d in ext.items()
               if d.get("status") == "ok" and d.get("nan_error_layers")]
    incomplete = [rid for rid, d in ext.items()
                  if not str(d.get("status", "")).startswith(("ok", "reused"))]
    w("## Gate 3 — no NaN / completeness")
    w("")
    w(f"- ext runs: {len(ext)}; with NaN layer errors: {len(nan_bad)} "
      f"{nan_bad or ''}")
    w(f"- non-ok ext runs: {len(incomplete)} {incomplete or ''}")
    fails = os.path.join(EXT, "failures.csv")
    nf = 0
    if os.path.exists(fails):
        nf = max(0, len(open(fails).read().strip().splitlines()) - 1)
    w(f"- failures.csv rows: {nf}")
    w(f"Gate 3: **{'PASS' if not nan_bad and not incomplete else 'FAIL'}**")
    w("")

    # ---------------- DD score diagnostics ---------------------------------
    w("## DD score diagnostics")
    w("")
    for mk in ("3b", "8b"):
        p = os.path.join(EXT, "scores", f"{mk}_dd_raw.pt")
        if not os.path.exists(p):
            continue
        dd = torch.load(p, weights_only=False)
        meta = dd["meta"]
        w(f"- {mk}: negative-score channel fraction overall = "
          f"{meta['neg_fraction_overall']:.4%} (signed scores kept; "
          f"per-module max = "
          f"{max(meta['neg_fraction_per_module'].values()):.4%})")
    w("")

    # ---------------- headline tables --------------------------------------
    for mk, label in (("3b", "Llama-3.2-3B"), ("8b", "Llama-3-8B")):
        w(f"## {label} — WikiText-2 PPL / C4 PPL / act_mse_global / E_mean")
        w("")
        e0 = point(mk, "ss", "within", "0")
        e1 = point(mk, "ss", "within", "1")
        if e0 and e1:
            w(f"Endpoints: pure NVFP4 (r=0) wiki {e0['wiki']:.4f}, c4 "
              f"{e0['c4']:.4f}, act_mse {e0['act_mse']:.6g}; pure MXFP8 "
              f"(r=1) wiki {e1['wiki']:.4f}, c4 {e1['c4']:.4f}, act_mse "
              f"{e1['act_mse']:.6g}")
            w("")
        w("| r | metric | " + " | ".join(f"{s} {sc}" for s, sc in STRATS) + " |")
        w("|---" * (len(STRATS) + 2) + "|")
        for r in REPORT_R:
            for met, fmt in (("wiki", "{:.4f}"), ("c4", "{:.4f}"),
                             ("act_mse", "{:.3e}"), ("emean", "{:.3e}")):
                cells = []
                for s, sc in STRATS:
                    if s == "rnd":
                        vals = [point(mk, s, sc, r, seed)
                                for seed in (0, 1, 2)]
                        vals = [v[met] for v in vals
                                if v is not None and v[met] is not None]
                        cells.append(
                            fmt.format(sum(vals) / len(vals))
                            + f" (n={len(vals)})" if vals else "-")
                    else:
                        v = point(mk, s, sc, r)
                        cells.append(fmt.format(v[met])
                                     if v and v[met] is not None else "-")
                w(f"| {r} | {met} | " + " | ".join(cells) + " |")
        w("")

    # ---------------- allocation audit: down_proj protection ----------------
    w("## Allocation audit — down_proj share and layers.1.mlp.down_proj")
    w("")
    w("FP8 channel share given to each projection type under the ACROSS "
      "scope (fraction of that projection's channels set to MXFP8), plus the "
      "quota of layers.1.mlp.down_proj (the layer MA starved in M31).")
    w("")
    for mk in ("3b", "8b"):
        w(f"### {mk}")
        w("")
        w("| r | strategy | q | k | v | gate | up | down | "
          "layers.1.mlp.down_proj h/d |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in ("0.02", "0.08", "0.1", "0.2", "0.5"):
            for s in ("ma", "ss", "dd"):
                v = point(mk, s, "across", r)
                if not v:
                    continue
                alloc = v["d"].get("alloc") or {}
                agg = {}
                crit = None
                for n, a in alloc.items():
                    t = n.split(".")[-1]
                    h0, d0 = agg.get(t, (0, 0))
                    agg[t] = (h0 + a["h"], d0 + a["d"])
                    if ".layers.1." in n and t == "down_proj":
                        crit = f"{a['h']}/{a['d']}"
                cells = []
                for t in ("q_proj", "k_proj", "v_proj", "gate_proj",
                          "up_proj", "down_proj"):
                    h0, d0 = agg.get(t, (0, 1))
                    cells.append(f"{h0 / d0:.3f}")
                w(f"| {r} | {s} ({v['src']}) | " + " | ".join(cells)
                  + f" | {crit} |")
        w("")

    out = os.path.join(EXT, "report.md")
    w("## Findings (as-measured, incl. results running against the spec's "
      "expectations)")
    w("")
    w("1. **DD (across) does NOT fix the cross-layer collapse.** On 3B it "
      "tracks MA (across): wiki PPL 17.6/18.3/17.1/18.6 at r=0.02-0.08 and "
      "14.8-17.8 at 0.1-0.5, all worse than pure NVFP4 (14.33). On 8B the "
      "same pattern is milder (7.35-7.90 vs r=0 baseline 7.78, while "
      "within-scope reaches 6.48-6.58). SS (across) remains the only "
      "non-collapsing across criterion (3B 8.11-8.23; 8B 6.44-6.53).")
    w("2. **DD (across) also loses on its own same-unit objective**: its "
      "held-out act_mse_global is HIGHER than MA/SS (across) at every r "
      "(e.g. 3B r=0.1: 3.30e-3 vs 2.73e-3), and DD (within) is likewise "
      "worse than MA/SS (within). The spec's expectation (sec 6.4.1: DD "
      "across significantly lower act_mse_global) is contradicted. Cause: "
      "the per-channel DD score is measured independently on the ORIGINAL "
      "block layout, so it ignores the dominant interaction - moving a "
      "high-magnitude channel OUT of an NVFP4 block16 shrinks that block's "
      "scale and reduces every remaining element's error. MA/SS target "
      "exactly those scale-poisoning outlier channels and therefore "
      "minimize the realized post-packing error better than the "
      "format-aware but interaction-blind DD.")
    w("3. **Critical-layer protection**: DD (across) gives "
      "layers.1.mlp.down_proj 0 channels at every reported r (same as MA); "
      "SS (across) consistently gives it exactly 32 channels - and that "
      "single 32-channel difference separates collapse (MA/DD) from "
      "stability (SS). All three across criteria starve the down_proj "
      "projection type as a whole (<=0.2% share).")
    w("4. **Densified cliff**: on 3B the within-scope targeted criteria "
      "recover almost the entire NVFP4->MXFP8 PPL gap with the FIRST 2% of "
      "channels (14.33 -> 8.22-8.27 at r=0.02); RND (within) is far behind "
      "and high-variance in the cliff region (seed-mean 10.5-15.2 over "
      "0.02-0.08). Same shape on 8B (7.78 -> 6.53-6.58 at r=0.02 vs RND "
      "7.36-7.80).")
    w("5. **act-MSE-optimal != PPL-optimal**: at r>=0.2 the across criteria "
      "all have clearly lower act_mse_global than within (e.g. 3B r=0.5: "
      "1.02-1.11e-3 vs 1.71-1.86e-3), yet MA/DD (across) are "
      "catastrophically worse in PPL - allocation quality is NOT summarized "
      "by the global activation-MSE objective alone; layer-coverage "
      "constraints matter.")
    w("6. **P2 (1B) not started**: the 3B/8B cliff region is fully resolved "
      "with large dynamic range between RND / targeted-within / across at "
      "r=0.02-0.08, so the optional larger-headroom model was unnecessary. "
      "DD_out (optional diagnostic variant) was likewise not run; the main "
      "DD criterion already shows the failure mode is packing interaction, "
      "not output weighting.")
    w("")
    txt = "\n".join(lines) + "\n"
    open(out, "w").write(txt)
    print(txt)
    print(f"[report] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
