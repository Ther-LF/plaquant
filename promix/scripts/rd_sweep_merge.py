#!/usr/bin/env python3
"""Merge RD_SWEEP_{model}_{ppl,lat}.json -> docs/rd_pareto_sweep.csv + tables.

Run locally after scp-ing the remote JSONs into a dir. Usage:
    python3 merge_sweep.py <json_dir> <out_csv>
"""
import csv
import json
import os
import sys

MODEL_ORDER = ["1b", "3b", "8b"]
MODEL_D = {"1b": 2048, "3b": 3072, "8b": 4096}
COLS = ["model", "alloc", "budget", "avg_fp8_realized", "bits_per_weight",
        "ppl", "prefill_ms", "decode_ms",
        "prefill_x_vs_uniform18", "decode_x_vs_uniform18"]


def load(jdir):
    """Return {(model,alloc,budget): row}, plus {model: bf16}."""
    merged = {}
    bf16 = {}
    for fn in sorted(os.listdir(jdir)):
        if not fn.startswith("RD_SWEEP_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(jdir, fn)) as f:
            d = json.load(f)
        model = d["meta"]["model"]
        if "bf16_bridge" in d:
            bf16[model] = d["bf16_bridge"]
        for key, row in d.get("rows", {}).items():
            k = (row["model"], row["alloc"], round(float(row["budget"]), 6))
            tgt = merged.setdefault(k, {"model": row["model"], "alloc": row["alloc"],
                                        "budget": float(row["budget"])})
            for c in ("ppl", "avg_fp8_realized", "bits_per_weight", "prefill_ms",
                      "decode_ms", "prefill_x_vs_uniform18",
                      "decode_x_vs_uniform18"):
                if c in row and row[c] not in (None, ""):
                    # prefer PPL-file's bpw/avg_fp8 (reorder path) if present
                    if c in ("avg_fp8_realized", "bits_per_weight") and c in tgt \
                            and "_ppl" in fn:
                        tgt[c] = row[c]
                    elif c not in tgt:
                        tgt[c] = row[c]
                    elif "_ppl" in fn and c in ("avg_fp8_realized", "bits_per_weight"):
                        tgt[c] = row[c]
    return merged, bf16


def fmt(v, nd=4):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    jdir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "rd_pareto_sweep.csv"
    merged, bf16 = load(jdir)

    def sort_key(k):
        m, alloc, b = k
        return (MODEL_ORDER.index(m) if m in MODEL_ORDER else 9,
                0 if alloc == "measured" else 1, b)

    keys = sorted(merged, key=sort_key)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for k in keys:
            r = merged[k]
            w.writerow([
                r["model"], r["alloc"], fmt(r["budget"], 5),
                fmt(r.get("avg_fp8_realized")), fmt(r.get("bits_per_weight")),
                fmt(r.get("ppl")), fmt(r.get("prefill_ms"), 3),
                fmt(r.get("decode_ms"), 3),
                fmt(r.get("prefill_x_vs_uniform18"), 4),
                fmt(r.get("decode_x_vs_uniform18"), 4)])
    print(f"wrote {out_csv} ({len(keys)} rows)")
    print("bf16 bridges:", {m: round(v, 4) for m, v in bf16.items()})

    # quick per-model pareto readings
    for m in MODEL_ORDER:
        pts = {k: merged[k] for k in merged if k[0] == m}
        if not pts:
            continue
        uni18 = next((v for k, v in pts.items()
                      if k[1] == "uniform" and abs(k[2] - 0.125) < 1e-9), None)
        if not uni18 or uni18.get("ppl") in (None, ""):
            continue
        ref = float(uni18["ppl"])
        meas = [(k[2], v) for k, v in pts.items()
                if k[1] == "measured" and v.get("ppl") not in (None, "")]
        meas.sort(key=lambda t: t[0])
        iso = [(b, v) for b, v in meas if float(v["ppl"]) <= ref + 1e-9]
        print(f"\n[{m}] uniform-1/8 PPL={ref:.4f}")
        if iso:
            b0, v0 = iso[0]
            print(f"  ISO-ACCURACY: smallest budget b={b0:.5f} PPL={float(v0['ppl']):.4f}"
                  f" (<= {ref:.4f}) prefill_x={v0.get('prefill_x_vs_uniform18')}"
                  f" decode_x={v0.get('decode_x_vs_uniform18')}")


if __name__ == "__main__":
    main()
