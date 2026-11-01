#!/usr/bin/env python3
"""Harvest the ProMix r-sweep into the accuracy-vs-throughput curve table.

Reads per-point gate JSONs (gate_t2_pm{size}_r{r}.json -> absolute-protocol
PPL + |dPPL| vs own exporter reference) and sweep serve JSONs
(pm{size}_r{r}_{main,qserve}.json), plus the five-way method JSONs for the
single-point rivals. bits/weight = (1-r)*4.5 + r*8.25.

Usage: python -m promix.vllm_integ.harvest_rsweep --dir docs/vllm_final_data --size 3b
"""
import argparse
import glob
import json
import os
import re


def jload(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", required=True)
    args = ap.parse_args()
    d, size = args.dir, args.size

    pts = []
    for g in sorted(glob.glob(os.path.join(d, f"gate_t2_pm{size}_r*.json"))):
        r = float(re.search(r"_r([\d.]+)\.json", g).group(1))
        pts.append((r, g))
    # the shipped r=1/8 point rides the five-way files + the campaign gate
    rows = []
    for r, g in sorted(pts) + [(0.125, os.path.join(d, "gate_t2_graphs_final.json"))]:
        gate = jload(g)
        tag = f"pm{size}" if r == 0.125 else f"pm{size}_r{r}"
        main = jload(os.path.join(d, f"{tag}_main.json"))
        qs = jload(os.path.join(d, f"{tag}_qserve.json"))
        row = {"r": r, "bits": 4.5 + 3.75 * r}
        if gate:
            row["ppl"] = gate["ppl_vllm"]
            row["dppl_vs_own_ref"] = gate["dppl"]
        for key, src, field in [
            ("tpot_b128", main, "isl128_osl1024_b128"),
            ("tpot_b256", main, "isl128_osl1024_b256"),
            ("gen_b256", main, "isl128_osl1024_b256"),
            ("pf128_b128", main, "isl1024_osl128_b128"),
        ]:
            c = (src or {}).get("cases", {}).get(field)
            if c and "error" not in c:
                row[key] = c["gen_tok_per_s"] if key.startswith("gen") else c["tpot_ms"]
        c = (qs or {}).get("cases", {}).get("isl1024_osl512_b256")
        if c and "error" not in c:
            row["qserve_c256"] = c["gen_tok_per_s"]
        p = (main or {}).get("prefill_ttft", {}).get("L2048_b128")
        if p and "error" not in p:
            row["l2048_b128"] = p["median_s"] * 1e3
        rows.append(row)
    rows.sort(key=lambda x: x["r"])

    # single-point rivals at the same cells
    rivals = []
    for m, name, ppl3, ppl8 in [("bf", "BF16", 7.82, 6.14),
                                ("fp8", "FP8 W8A8", 7.880, 6.191),
                                ("gptq", "GPTQ-Marlin W4A16", 8.20, 6.52),
                                ("nv", "NVFP4 W4A4", 8.63, 6.87)]:
        main = jload(os.path.join(d, f"{m}{size}_main.json"))
        qs = jload(os.path.join(d, f"{m}{size}_qserve.json"))
        if not main:
            continue
        row = {"name": name, "ppl": ppl3 if size == "3b" else ppl8}
        c = main["cases"].get("isl128_osl1024_b256")
        if c and "error" not in c:
            row["tpot_b256"] = c["tpot_ms"]
            row["gen_b256"] = c["gen_tok_per_s"]
        c = (qs or {}).get("cases", {}).get("isl1024_osl512_b256")
        if c and "error" not in c:
            row["qserve_c256"] = c["gen_tok_per_s"]
        p = main.get("prefill_ttft", {}).get("L2048_b128")
        if p and "error" not in p:
            row["l2048_b128"] = p["median_s"] * 1e3
        rivals.append(row)

    def f(v, nd=2):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

    print(f"### ProMix r-sweep curve ({size.upper()}; gate = |dPPL| vs own "
          "exporter reference, all PASS <= 0.1)\n")
    print("| r | bits/wt | PPL (abs protocol) | gate dPPL | decode TPOT ms b128/b256 | gen tok/s b256 | QServe c256 tok/s | prefill L2048 b128 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['r']:.5g} | {row['bits']:.2f} | {f(row.get('ppl'),3)} | "
              f"{f(row.get('dppl_vs_own_ref'),4)} | "
              f"{f(row.get('tpot_b128'))}/{f(row.get('tpot_b256'))} | "
              f"{f(row.get('gen_b256'),0)} | {f(row.get('qserve_c256'),0)} | "
              f"{f(row.get('l2048_b128'),0)} |")
    print("\n**Single-point rivals (same cells):**\n")
    print("| method | PPL | decode TPOT b256 | gen tok/s b256 | QServe c256 | prefill L2048 b128 ms |")
    print("|---|---|---|---|---|---|")
    for row in rivals:
        print(f"| {row['name']} | {f(row.get('ppl'),3)} | {f(row.get('tpot_b256'))} | "
              f"{f(row.get('gen_b256'),0)} | {f(row.get('qserve_c256'),0)} | "
              f"{f(row.get('l2048_b128'),0)} |")


if __name__ == "__main__":
    main()
