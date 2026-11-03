#!/usr/bin/env python3
"""rd-r14: compare re-measure sessions against the campaign numbers."""
import json
import sys

D = "/home/luofan/plaquant-r14probe/docs/rd_r14_data"
C = "/home/luofan/vllm_integ/bench_final"

CAMPAIGN = {
    "bf16": ("bf3b_main.json", "bf3b_qserve.json"),
    "r125": ("pm3b_main.json", "pm3b_qserve.json"),
    "r25": ("pm3b_r0.25_main.json", "pm3b_r0.25_qserve.json"),
}
CELLS = ["isl1024_osl128_b16", "isl1024_osl128_b128", "isl1024_osl128_b256",
         "isl128_osl1024_b16", "isl128_osl1024_b128", "isl128_osl1024_b256"]


def cases(path):
    try:
        return json.load(open(path))["cases"]
    except Exception:
        return None


def fmt(c, k, fld="tpot_ms"):
    if c is None or k not in c:
        return "   -  "
    return "%6.2f" % c[k][fld]


def main():
    sessions = sys.argv[1:] or ["sA", "sB"]
    for cfg in ("bf16", "r125", "r25"):
        camp = cases(f"{C}/{CAMPAIGN[cfg][0]}")
        rows = {s: cases(f"{D}/r14_{s}_{cfg}_main.json") for s in sessions}
        print(f"== {cfg} TPOT ms (campaign | " + " | ".join(sessions) + ")")
        for k in CELLS:
            print(f"  {k:24s} {fmt(camp, k)} | " +
                  " | ".join(fmt(rows[s], k) for s in sessions))
        print(f"  {'prefill L2048 b128 TTFT':24s} "
              f"{fmt(camp, 'prefill2048_b128', 'ttft_s') if camp and 'prefill2048_b128' in camp else '   -  '} | " +
              " | ".join(fmt(rows[s], "prefill2048_b128", "ttft_s") for s in sessions))
        campq = cases(f"{C}/{CAMPAIGN[cfg][1]}")
        qs = {s: cases(f"{D}/r14_{s}_{cfg}_qserve.json") for s in sessions}
        k = "isl1024_osl512_b256"
        print(f"  {'qserve c256 tok/s':24s} " +
              ("%6.0f" % campq[k]["gen_tok_per_s"] if campq and k in campq else "   -  ") + " | " +
              " | ".join(("%6.0f" % qs[s][k]["gen_tok_per_s"]) if qs[s] and k in qs[s] else "   -  "
                         for s in sessions))
        # full runs spread for the anomalous decode cells
        for s in sessions:
            c = rows[s]
            if c and "isl128_osl1024_b256" in c:
                fr = c["isl128_osl1024_b256"]["full_runs_s"]
                print(f"  {s} 128:1024 b256 full_runs: {[round(x,3) for x in fr]}")


if __name__ == "__main__":
    main()
