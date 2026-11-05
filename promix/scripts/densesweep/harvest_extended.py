#!/usr/bin/env python3
"""rd-densesweep: harvest the EXTENDED serving protocol into markdown tables.

  ext_{method}{size}.json   fixed-length long-ctx/big-batch grid (bench_serve)
  sgpt_{method}{size}.json  ShareGPT real-distribution run (vllm bench serve)

Prints: per size, (1) gen tok/s grid tables (one per ISL:OSL case, methods x
batch), (2) max cleanly-servable batch per context, (3) the ShareGPT table.
Paste into docs/vllm_serving_extended.md.

Usage: python promix/scripts/densesweep/harvest_extended.py [--repo .]
"""
import argparse
import glob
import json
import os
import re

SIZES = ["3b", "8b"]
METHODS = ["bf", "fp8", "gptq", "awq", "nv", "pm", "pmr0.0625", "pmr0.25",
           "sharq", "fpquant"]
NAME = {"bf": "BF16", "fp8": "FP8 W8A8", "gptq": "GPTQ-Marlin W4A16",
        "awq": "AWQ-Marlin W4A16", "nv": "NVFP4 W4A4", "pm": "ProMix r=1/8",
        "pmr0.0625": "ProMix r=1/16", "pmr0.25": "ProMix r=1/4",
        "sharq": "SharQ", "fpquant": "FP-Quant NVFP4"}
ISLS = [128, 1024, 2048, 4096]
OSLS = [128, 512, 1024]
BATCHES = [16, 64, 128, 256, 512]


def jload(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    d = os.path.join(args.repo, "docs/vllm_final_data")

    for size in SIZES:
        ext = {m: jload(os.path.join(d, f"ext_{m}{size}.json"))
               for m in METHODS}
        if any(ext.values()):
            print(f"\n## {size.upper()} fixed-length grid (gen tok/s; "
                  "'ERR' = case failed, blank = not run)")
            for isl in ISLS:
                for osl in OSLS:
                    rows = []
                    for m in METHODS:
                        if not ext[m]:
                            continue
                        cells = []
                        for b in BATCHES:
                            c = ext[m]["cases"].get(f"isl{isl}_osl{osl}_b{b}")
                            if c is None:
                                cells.append("")
                            elif "error" in c:
                                cells.append("ERR")
                            else:
                                cells.append(f"{c['gen_tok_per_s']:.0f}")
                        rows.append((NAME[m], cells))
                    if not rows:
                        continue
                    print(f"\n### ISL {isl} / OSL {osl}\n")
                    print("| method | " + " | ".join(f"b{b}" for b in BATCHES) + " |")
                    print("|---|" + "---|" * len(BATCHES))
                    for name, cells in rows:
                        print(f"| {name} | " + " | ".join(cells) + " |")
            # max servable batch per context length
            print(f"\n### {size.upper()} max cleanly-served batch per ISL "
                  "(no per-case error, OSL any)\n")
            print("| method | " + " | ".join(f"ISL {i}" for i in ISLS) + " |")
            print("|---|" + "---|" * len(ISLS))
            for m in METHODS:
                if not ext[m]:
                    continue
                cells = []
                for isl in ISLS:
                    mx = 0
                    for b in BATCHES:
                        ok = all(
                            (c := ext[m]["cases"].get(f"isl{isl}_osl{o}_b{b}"))
                            is not None and "error" not in c for o in OSLS)
                        if ok:
                            mx = b
                    cells.append(str(mx) if mx else "—")
                print(f"| {NAME[m]} | " + " | ".join(cells) + " |")

        rows = []
        for m in METHODS:
            j = jload(os.path.join(d, f"sgpt_{m}{size}.json"))
            if not j:
                continue
            rows.append((NAME[m], j))
        if rows:
            print(f"\n## {size.upper()} ShareGPT real-distribution "
                  "(512 prompts, vllm bench serve)\n")
            print("| method | req/s | out tok/s | total tok/s | mean TTFT ms | mean TPOT ms |")
            print("|---|---|---|---|---|---|")
            for name, j in rows:
                def g(*keys, nd=2):
                    for k in keys:
                        if k in j and isinstance(j[k], (int, float)):
                            return f"{j[k]:.{nd}f}"
                    return "—"
                print(f"| {name} | {g('request_throughput')} | "
                      f"{g('output_throughput')} | {g('total_token_throughput')} | "
                      f"{g('mean_ttft_ms')} | {g('mean_tpot_ms')} |")


if __name__ == "__main__":
    main()
