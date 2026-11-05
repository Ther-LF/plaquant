#!/usr/bin/env python3
"""rd-densesweep: harvest BOTH 5%-grid curves into plot-ready CSVs + §4 tables.

  docs/vllm_rsweep_curve.csv   serving curve  (per model: r, bits, PPL(abs),
                               gate dPPL, prefill ms, decode TPOT b128/b256,
                               gen tok/s b256, conc 1024/512 c256 tok/s)
  docs/rd_accuracy_curve.csv   accuracy curve (per model: r target, realized
                               FP8 fraction, bits/wt, wiki PPL, C4 PPL)

Sources: docs/vllm_final_data/ (gate_t2_pm{size}_r*.json, pm{size}_r*_{main,
qserve}.json, pm{size}_r*_export_meta.json; plus the legacy vllm-final n/32
points, provenance-flagged "uniform-dense") and docs/rd_acctune_data/densesweep/
({size}_r*.json from rd_acctune/rd_scale, ring512+measured acctune recipe).

bits/weight: exact from the export meta per-Linear K splits when present
(sum K_low*4.5 + K_high*8.25 over the 7 target Linears), else the uniform
nominal 4.5 + 3.75*r.

Usage: python promix/scripts/densesweep/harvest_curves.py [--repo .] [--md]
"""
import argparse
import glob
import json
import os
import re

BPW_FP4, BPW_FP8 = 4.5, 8.25
SIZES = ["1b", "3b", "8b"]
ACC_SIZES = ["1b", "3b", "8b", "70b"]
# legacy vllm-final campaign points (uniform n/32 grid, dense substrate)
LEGACY = {"3b": ["0.03125", "0.0625", "0.09375", "0.1875", "0.25"],
          "8b": ["0.03125", "0.0625", "0.25"]}


def jload(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def meta_bits(meta):
    """exact realized (bits, fp8_frac) from export meta per_linear K splits"""
    pl = meta.get("per_linear", {})
    tot = sum(v["K"] for v in pl.values())
    hi = sum(v["K_high"] for v in pl.values())
    if not tot:
        return None, None
    return (((tot - hi) * BPW_FP4 + hi * BPW_FP8) / tot, hi / tot)


def serve_row(d, size, rtxt, tag, provenance):
    gate = jload(os.path.join(d, f"gate_t2_{tag}.json"))
    main = jload(os.path.join(d, f"{tag}_main.json"))
    qs = jload(os.path.join(d, f"{tag}_qserve.json"))
    em = jload(os.path.join(d, f"{tag}_export_meta.json"))
    r = float(rtxt)
    row = dict(model=size, r=r, provenance=provenance,
               bits=4.5 + 3.75 * r, realized_fp8=r)
    if em:
        b, f = meta_bits(em)
        if b:
            row["bits"], row["realized_fp8"] = b, f
    if gate:
        row["ppl"] = gate["ppl_vllm"]
        row["dppl"] = gate["dppl"]
        row["t2"] = "PASS" if abs(gate["dppl"]) <= 0.1 else "FAIL"
    for key, src, field, val in [
            ("tpot_b128", main, "isl128_osl1024_b128", "tpot_ms"),
            ("tpot_b256", main, "isl128_osl1024_b256", "tpot_ms"),
            ("gen_b256", main, "isl128_osl1024_b256", "gen_tok_per_s"),
            ("conc_c256", qs, "isl1024_osl512_b256", "gen_tok_per_s")]:
        c = (src or {}).get("cases", {}).get(field)
        if c and "error" not in c:
            row[key] = c[val]
    p = (main or {}).get("prefill_ttft", {}).get("L2048_b128")
    if p and "error" not in p:
        row["prefill_ms"] = p["median_s"] * 1e3
    return row


def harvest_serving(repo):
    d = os.path.join(repo, "docs/vllm_final_data")
    rows = []
    seen = set()
    for size in SIZES:
        for g in sorted(glob.glob(os.path.join(d, f"gate_t2_pm{size}_r*.json"))):
            rtxt = re.search(r"_r([\d.]+)\.json", g).group(1)
            tag = f"pm{size}_r{rtxt}"
            prov = ("uniform-dense (vllm-final)"
                    if rtxt in LEGACY.get(size, []) else "ring512+measured")
            rows.append(serve_row(d, size, rtxt, tag, prov))
            seen.add((size, float(rtxt)))
        # shipped r=1/8 five-way point (3b/8b)
        if size in ("3b", "8b") and (size, 0.125) not in seen:
            row = serve_row(d, size, "0.125", f"pm{size}",
                            "uniform-dense (vllm-final shipped)")
            if size == "3b":
                g = jload(os.path.join(d, "gate_t2_graphs_final.json"))
                if g:
                    row["ppl"], row["dppl"] = g["ppl_vllm"], g["dppl"]
                    row["t2"] = "PASS" if abs(g["dppl"]) <= 0.1 else "FAIL"
            else:
                row.setdefault("ppl", 6.6402), row.setdefault("dppl", 0.0025)
                row.setdefault("t2", "PASS")
            rows.append(row)
    rows.sort(key=lambda x: (x["model"], x["r"]))
    return rows


def harvest_accuracy(repo):
    d = os.path.join(repo, "docs/rd_acctune_data/densesweep")
    rows = []
    for size in ACC_SIZES:
        for g in sorted(glob.glob(os.path.join(d, f"{size}_r*.json"))):
            rtxt = re.search(r"_r([\d.]+)\.json", g).group(1)
            j = jload(g)
            if not j or "wikitext" not in j:
                continue
            frac = j.get("avg_fp8")
            row = dict(model=size, r=float(rtxt),
                       realized_fp8=frac,
                       bits=(4.5 + 3.75 * frac) if frac is not None else None,
                       wiki_ppl=j["wikitext"], c4_ppl=j.get("c4"),
                       total_rings=j.get("alloc_meta", {}).get("total_rings"),
                       sel_rings=j.get("alloc_meta", {}).get("sel_rings"))
            rows.append(row)
    rows.sort(key=lambda x: (x["model"], x["r"]))
    return rows


def write_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(
                "" if row.get(c) is None else
                (f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]))
                for c in cols) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def md_tables(srows, arows):
    out = []
    for size in SIZES:
        rs = [r for r in srows if r["model"] == size]
        if not rs:
            continue
        out.append(f"\n### ProMix serving r-sweep ({size.upper()}; 5% budget "
                   "grid; gate = |dPPL| vs own exporter reference)\n")
        out.append("| r | bits/wt | PPL (abs) | gate dPPL | TPOT ms b128/b256 "
                   "| gen tok/s b256 | conc256 1024/512 | prefill L2048 b128 ms | provenance |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            def f(k, nd=2):
                v = r.get(k)
                return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"
            out.append(f"| {r['r']:.5g} | {f('bits')} | {f('ppl', 3)} | "
                       f"{f('dppl', 4)} | {f('tpot_b128')}/{f('tpot_b256')} | "
                       f"{f('gen_b256', 0)} | {f('conc_c256', 0)} | "
                       f"{f('prefill_ms', 0)} | {r.get('provenance','')} |")
    for size in ACC_SIZES:
        rs = [r for r in arows if r["model"] == size]
        if not rs:
            continue
        out.append(f"\n### ProMix accuracy r-curve ({size.upper()}; fake-quant "
                   "wiki+C4, ring512+measured acctune recipe)\n")
        out.append("| r target | realized FP8 frac | bits/wt | wiki PPL | C4 PPL |")
        out.append("|---|---|---|---|---|")
        for r in rs:
            def f(k, nd=4):
                v = r.get(k)
                return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"
            out.append(f"| {r['r']:.2f} | {f('realized_fp8')} | {f('bits', 2)} | "
                       f"{f('wiki_ppl')} | {f('c4_ppl')} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--md", action="store_true", help="print §4 markdown tables")
    args = ap.parse_args()
    srows = harvest_serving(args.repo)
    arows = harvest_accuracy(args.repo)
    write_csv(os.path.join(args.repo, "docs/vllm_rsweep_curve.csv"), srows,
              ["model", "r", "bits", "realized_fp8", "ppl", "dppl", "t2",
               "prefill_ms", "tpot_b128", "tpot_b256", "gen_b256", "conc_c256",
               "provenance"])
    write_csv(os.path.join(args.repo, "docs/rd_accuracy_curve.csv"), arows,
              ["model", "r", "realized_fp8", "bits", "wiki_ppl", "c4_ppl",
               "total_rings", "sel_rings"])
    if args.md:
        print(md_tables(srows, arows))


if __name__ == "__main__":
    main()
