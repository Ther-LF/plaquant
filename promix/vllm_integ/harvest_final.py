#!/usr/bin/env python3
"""Harvest the FINAL five-way (+extensions) campaign JSONs into markdown tables.

Input dir layout (from docs/vllm_final_scripts/run_final.sh):
  {tag}_main.json   cases 1024:128 + 128:1024, b {1,16,32,64,128,256,512} + prefill L2048 sweep
  {tag}_qserve.json case 1024:512, b {64,128,256}
  {tag}_kv8.json    cases 2048:128 + 4096:128, b {16,64}, kv fp8
  {tag}_main.log    (optional) engine logs -> KV cache capacity line

tag = <method><size>, e.g. pm3b, nv8b, bf3b, fp83b, gptq8b, awq3b, int83b.

Usage: python -m promix.vllm_integ.harvest_final --dir <data-dir> --size 3b
"""
import argparse
import glob
import json
import os
import re

METHOD_ORDER = ["bf", "fp8", "gptq", "awq", "int8", "nv", "pm"]
METHOD_NAME = {
    "bf": "BF16", "fp8": "FP8 W8A8", "gptq": "GPTQ-Marlin W4A16",
    "awq": "AWQ-Marlin W4A16", "int8": "INT8 W8A8",
    "nv": "NVFP4 W4A4", "pm": "ProMix W4A4",
}


def load(dirname, tag, kind):
    p = os.path.join(dirname, f"{tag}_{kind}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def kv_capacity(dirname, tag):
    """Parse 'Available KV cache memory: X GiB' from the engine log."""
    p = os.path.join(dirname, f"{tag}_main.log")
    if not os.path.exists(p):
        return None
    gib = None
    with open(p, errors="replace") as f:
        for line in f:
            m = re.search(r"Available KV cache memory: ([\d.]+) GiB", line)
            if m:
                gib = float(m.group(1))
                break  # first engine (main sweep)
    return gib


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def cell(case):
    if case is None:
        return None
    if "error" in case:
        err = case["error"]
        return {"err": "OOM" if "memory" in err.lower() or "OutOfMemory" in err
                else err[:40]}
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", required=True, help="3b or 8b")
    args = ap.parse_args()

    tags = []
    for m in METHOD_ORDER:
        t = f"{m}{args.size}"
        if glob.glob(os.path.join(args.dir, f"{t}_*.json")):
            tags.append((m, t))

    data = {t: {k: load(args.dir, t, k) for k in ("main", "qserve", "kv8")}
            for _, t in tags}

    def batches(kind, case_prefix):
        bs = set()
        for _, t in tags:
            d = data[t][kind]
            if not d:
                continue
            for key in d.get("cases", {}):
                if key.startswith(case_prefix):
                    bs.add(int(key.split("_b")[-1]))
        return sorted(bs)

    print(f"## {args.size.upper()} — methods: "
          f"{', '.join(METHOD_NAME[m] for m, _ in tags)}\n")

    for isl, osl, label, metric in [
        (1024, 128, "prefill-heavy 1024/128 — TTFT ms", "ttft"),
        (1024, 128, "prefill-heavy 1024/128 — TPOT ms", "tpot"),
        (1024, 128, "prefill-heavy 1024/128 — gen tok/s", "toks"),
        (128, 1024, "decode-heavy 128/1024 — TPOT ms", "tpot"),
        (128, 1024, "decode-heavy 128/1024 — gen tok/s", "toks"),
    ]:
        pref = f"isl{isl}_osl{osl}_b"
        bs = batches("main", pref)
        if not bs:
            continue
        print(f"### {label}")
        print("| method | " + " | ".join(f"b{b}" for b in bs) + " |")
        print("|---" * (len(bs) + 1) + "|")
        for m, t in tags:
            row = [METHOD_NAME[m]]
            for b in bs:
                c = cell((data[t]["main"] or {}).get("cases", {}).get(f"{pref}{b}"))
                if c is None:
                    row.append("—")
                elif "err" in c:
                    row.append(c["err"])
                elif metric == "ttft":
                    row.append(fmt(c["ttft_s"] * 1e3))
                elif metric == "tpot":
                    row.append(fmt(c["tpot_ms"], 2))
                else:
                    row.append(fmt(c["gen_tok_per_s"], 0))
            print("| " + " | ".join(row) + " |")
        print()

    bs = batches("main", "")
    print("### prefill TTFT L2048 (ms)")
    pb = set()
    for _, t in tags:
        d = data[t]["main"]
        if d:
            pb |= {int(k.split("_b")[-1]) for k in d.get("prefill_ttft", {})}
    pb = sorted(pb)
    print("| method | " + " | ".join(f"b{b}" for b in pb) + " |")
    print("|---" * (len(pb) + 1) + "|")
    for m, t in tags:
        row = [METHOD_NAME[m]]
        for b in pb:
            c = (data[t]["main"] or {}).get("prefill_ttft", {}).get(f"L2048_b{b}")
            if c is None or "error" in c:
                row.append("—" if c is None else "ERR")
            else:
                row.append(fmt(c["median_s"] * 1e3))
        print("| " + " | ".join(row) + " |")
    print()

    print("### QServe-protocol continuous batching 1024/512 — gen tok/s (TPOT ms)")
    qb = batches("qserve", "isl1024_osl512_b")
    print("| method | " + " | ".join(f"c{b}" for b in qb) + " |")
    print("|---" * (len(qb) + 1) + "|")
    for m, t in tags:
        row = [METHOD_NAME[m]]
        for b in qb:
            c = cell((data[t]["qserve"] or {}).get("cases", {}).get(
                f"isl1024_osl512_b{b}"))
            row.append("—" if c is None else c.get("err") or
                       f"{c['gen_tok_per_s']:.0f} ({c['tpot_ms']:.2f})")
        print("| " + " | ".join(row) + " |")
    print()

    print("### long-ISL decode, fp8 KV (TPOT ms @ b16 / b64)")
    print("| method | isl2048 b16 | isl2048 b64 | isl4096 b16 | isl4096 b64 |")
    print("|---|---|---|---|---|")
    for m, t in tags:
        row = [METHOD_NAME[m]]
        for isl in (2048, 4096):
            for b in (16, 64):
                c = cell((data[t]["kv8"] or {}).get("cases", {}).get(
                    f"isl{isl}_osl128_b{b}"))
                row.append("—" if c is None else c.get("err") or
                           fmt(c["tpot_ms"], 2))
        print("| " + " | ".join(row) + " |")
    print()

    print("### capacity (GPU KV cache tokens @ gmu 0.90, mml 4096) "
          "+ peak reserved GiB")
    print("| method | KV cache GiB | max reserved GiB |")
    print("|---|---|---|")
    for m, t in tags:
        kv = kv_capacity(args.dir, t)
        mem = (data[t]["main"] or {}).get("memory", {})
        print(f"| {METHOD_NAME[m]} | {kv if kv else '—'} | "
              f"{fmt(mem.get('max_memory_reserved_gib'), 1)} |")
    print()

    # ratio block: pm vs each
    if any(m == "pm" for m, _ in tags):
        pmt = f"pm{args.size}"
        print("### ProMix ratios (other_ms / pm_ms for latency; pm/other for tok/s; >1 = ProMix wins)")
        for isl, osl in [(1024, 128), (128, 1024)]:
            pref = f"isl{isl}_osl{osl}_b"
            bs2 = batches("main", pref)
            print(f"**case {isl}/{osl} TPOT ratio (x/pm):**")
            print("| vs | " + " | ".join(f"b{b}" for b in bs2) + " |")
            print("|---" * (len(bs2) + 1) + "|")
            for m, t in tags:
                if m == "pm":
                    continue
                row = [METHOD_NAME[m]]
                for b in bs2:
                    a = cell((data[t]["main"] or {}).get("cases", {}).get(f"{pref}{b}"))
                    p = cell((data[pmt]["main"] or {}).get("cases", {}).get(f"{pref}{b}"))
                    ok = a and p and "err" not in a and "err" not in p
                    row.append(fmt(a["tpot_ms"] / p["tpot_ms"], 2) if ok else "—")
                print("| " + " | ".join(row) + " |")
            print()


if __name__ == "__main__":
    main()
