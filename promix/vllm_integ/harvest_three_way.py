#!/usr/bin/env python3
"""Render the three-way (promix / nvfp4 / bf16) markdown tables from
bench_serve.py JSONs.

  python -m promix.vllm_integ.harvest_three_way --dir bench --size 3b
"""
import argparse
import json
import os


def load(d, tag):
    with open(os.path.join(d, f"{tag}.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", required=True, help="e.g. 3b / 8b (file suffix)")
    args = ap.parse_args()
    d = {t: load(args.dir, f"{t}_{args.size}") for t in ("promix", "nvfp4", "bf16")}

    print(f"### {args.size.upper()} — ISL/OSL cases "
          "(TTFT/TPOT: ratio = other/promix, >1 means ProMix faster; "
          "tok/s: ratio = promix/other)\n")
    print("| case | TTFT pm/nv/bf16 (ms) | TPOT pm/nv/bf16 (ms) | gen tok/s pm/nv/bf16 "
          "| pm/bf16 | pm/nvfp4 | nv/bf16 |")
    print("|---|---|---|---|---|---|---|")
    for key, pm in d["promix"]["cases"].items():
        nv, bf = d["nvfp4"]["cases"][key], d["bf16"]["cases"][key]
        if "error" in pm or "error" in nv or "error" in bf:
            print(f"| {key} | ERROR | | | | | |")
            continue
        print(f"| {key} | {pm['ttft_s']*1e3:.1f} / {nv['ttft_s']*1e3:.1f} / "
              f"{bf['ttft_s']*1e3:.1f} | {pm['tpot_ms']:.2f} / {nv['tpot_ms']:.2f} / "
              f"{bf['tpot_ms']:.2f} | {pm['gen_tok_per_s']:.0f} / "
              f"{nv['gen_tok_per_s']:.0f} / {bf['gen_tok_per_s']:.0f} "
              f"| {pm['gen_tok_per_s']/bf['gen_tok_per_s']:.2f}x "
              f"| {pm['gen_tok_per_s']/nv['gen_tok_per_s']:.2f}x "
              f"| {nv['gen_tok_per_s']/bf['gen_tok_per_s']:.2f}x |")

    print("\n| prefill TTFT | promix | nvfp4 | bf16 | pm/bf16 | pm/nvfp4 | nv/bf16 |")
    print("|---|---|---|---|---|---|---|")
    for key, pm in d["promix"].get("prefill_ttft", {}).items():
        nv = d["nvfp4"]["prefill_ttft"][key]
        bf = d["bf16"]["prefill_ttft"][key]
        if "median_s" not in pm or "median_s" not in nv or "median_s" not in bf:
            print(f"| {key} | ERROR | | | | | |")
            continue
        p, n, b = pm["median_s"], nv["median_s"], bf["median_s"]
        print(f"| {key} | {p*1e3:.1f} ms | {n*1e3:.1f} ms | {b*1e3:.1f} ms "
              f"| {b/p:.2f}x | {n/p:.2f}x | {b/n:.2f}x |")


if __name__ == "__main__":
    main()
