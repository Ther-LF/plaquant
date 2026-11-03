"""rd-buildup harvest — print the E5a/E5b/E6 tables from /home/luofan/buildup.

Usage: python promix/scripts/rd_buildup_harvest.py [--dir /home/luofan/buildup]
"""
import argparse
import glob
import json
import os
import statistics

ROWS_3B = [
    ("3b_r1_norot", "1. RTN pure-NVFP4, no rotation"),
    ("3b_r2_pca_u", "2. + PCA U rotation (no R1)"),
    ("3b_r3_ring512", "3. + ring512 R1"),
    ("3b_r4_uniform18", "4. + uniform-1/8 mixed precision"),
    ("3b_r5_gptq", "5. + GPTQ"),
    ("3b_r6_wscale", "6. + two-level weight scale (micro-MSE)"),
    ("3b_r7_actmse", "7. + act-MSE scale search (d3u2)"),
    ("3b_r8_measured", "8. + measured-dD allocation"),
    ("3b_c1_norot_full", "C1 control: no-rotation + full weight/act stack"),
    ("3b_c2_norot_gw", "C2 control: no-rotation + GPTQ + wscale (no actMSE)"),
    ("3b_c3_norot_g", "C3 control: no-rotation + GPTQ only (plain scales)"),
]
PERF_STEPS = [
    ("s2_fused_r0", "2. single-launch fused GEMM (levers off)"),
    ("s3_dispatch", "3. + per-shape dispatch / small-M"),
    ("s4_splitk", "4. + split-K"),
    ("s5_gemv", "5. + GEMV/marlin routes (+hi)"),
    ("s6_fusions", "6. + pack/gather/cast fusions"),
]


def j(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def acc_table(d):
    print("== E5a accuracy build-up (3B, wiki / C4) ==")
    for tag, label in ROWS_3B:
        r = j(d, tag)
        if r is None:
            print(f"{label:45s}  PENDING")
            continue
        w, c = r.get("wikitext"), r.get("c4")
        print(f"{label:45s}  wiki {w if w is None else round(w,4)}  "
              f"c4 {c if c is None else round(c,4)}  avg_fp8 "
              f"{round(r.get('avg_fp8', -1), 4)}")
        if "w4a16_wikitext" in r:
            print(f"{'9. + decode W4A16 (act exact, decode route)':45s}  "
                  f"wiki {round(r['w4a16_wikitext'],4)}  "
                  f"c4 {round(r.get('w4a16_c4',-1),4)}")
    for tag in ("1b_r8_measured", "8b_r8_measured"):
        r = j(d, tag)
        if r is None:
            print(f"{tag:45s}  PENDING")
        else:
            print(f"{tag:45s}  wiki {round(r['wikitext'],4)}  "
                  f"c4 {round(r.get('c4',-1),4)}  "
                  f"w4a16 {round(r.get('w4a16_wikitext',-1),4)}/"
                  f"{round(r.get('w4a16_c4',-1),4)}")


def perf_table(d):
    for m in ("3b", "8b"):
        for b, pre in (("b128", ""), ("b16", "_b16")):
            print(f"== E5b serving build-up ({m}, decode {b} TPOT ms / "
                  f"prefill L2048 TTFT ms) ==")
            for st, label in PERF_STEPS:
                r = j(d, f"pb_{m}{pre}_{st}")
                if r is None:
                    print(f"{label:45s}  PENDING")
                    continue
                case = next((c for c in r["cases"].values()
                             if "tpot_ms" in c), None)
                pf = [v["median_s"] for v in r.get("prefill_ttft", {}).values()
                      if "median_s" in v]
                tp = round(case["tpot_ms"], 2) if case else "ERR"
                pv = round(pf[0] * 1e3, 1) if pf else "ERR"
                print(f"{label:45s}  TPOT {tp}  prefill {pv}")


def naive_table(d):
    """Step-1 naive 2-launch: GEMM-level measurement (torch-native mix2 =
    native NVFP4 GEMM + native MXFP8 GEMM + add) + analytic serving estimate:
    serving_naive = serving_s2 + n_layers * n_steps * sum_shapes(mix2 - fused),
    i.e. replace every fused-GEMM call in the step-2 engine by the 2-launch
    torch-native rendering, all else equal. Decode: 1 step/token at M=batch;
    prefill b128xL2048: 262144 tokens / 16384-token chunks = 16 chunked-prefill
    steps at M=16384 (vLLM max_num_batched_tokens=16384, log-verified)."""
    rows = []
    for name in ("h2h_naive", "h2h_naive_m16", "h2h_naive_m16384"):
        r = j(d, name)
        if r is None:
            print(f"{name}: PENDING")
            continue
        rows += r["rows"]
    if not rows:
        return
    print("== E5b step-1 naive 2-launch (GEMM-only, per merged Linear, ms) ==")
    for row in rows:
        print(f"  {row['model']} {row['shape']:8s} M={row['M']:5d} "
              f"fused {row['fused_ms']*1e3:7.1f}us  mix2 "
              f"{row['torchmixed2_ms']*1e3:7.1f}us  mix2/fused "
              f"{row['torchmixed2_ms']/row['fused_ms']:.2f}x")
    NL = {"3B": 28, "8B": 32}
    print("== analytic naive serving deltas (add to the s2 row) ==")
    for m in ("3B", "8B"):
        for M, tag in ((128, "decode b128 TPOT"), (16, "decode b16 TPOT"),
                       (16384, "prefill b128 L2048 (x16 chunks)")):
            sel = [r for r in rows if r["model"] == m and r["M"] == M]
            if not sel:
                continue
            delta = sum(r["torchmixed2_ms"] - r["fused_ms"] for r in sel)
            mult = NL[m] * (16 if M == 16384 else 1)
            print(f"  {m} {tag}: +{delta*mult:.2f} ms "
                  f"(per-layer-set delta {delta*1e3:.1f} us x {mult})")


def seeds_table(d):
    print("== E6 seeds ==")
    groups = {
        "3b_final": ["3b_r8_measured", "3b_final_s1", "3b_final_s2"],
        "8b_final": ["8b_r8_measured", "8b_final_s1", "8b_final_s2"],
    }
    for g, tags in groups.items():
        vals_w, vals_c = [], []
        for t in tags:
            r = j(d, t)
            if r and "wikitext" in r:
                vals_w.append(r["wikitext"])
                if "c4" in r:
                    vals_c.append(r["c4"])
        if vals_w:
            mw = statistics.mean(vals_w)
            sw = statistics.stdev(vals_w) if len(vals_w) > 1 else 0.0
            print(f"{g}: n={len(vals_w)} wiki {mw:.4f}±{sw:.4f} "
                  f"({[round(v,4) for v in vals_w]})")
            if vals_c:
                mc = statistics.mean(vals_c)
                sc = statistics.stdev(vals_c) if len(vals_c) > 1 else 0.0
                print(f"{g}: C4 {mc:.4f}±{sc:.4f} "
                      f"({[round(v,4) for v in vals_c]})")
        else:
            print(f"{g}: PENDING")
    for s in (0, 1, 2):
        r = j(d, f"validity_s{s}")
        if r and "gate" in r:
            gt = r["gate"]
            print(f"validity seed {s}: A={gt['A']:.4f} B={gt['B']:.4f} "
                  f"C={gt['C']:.4f} -> {gt['verdict']}")
        else:
            print(f"validity seed {s}: PENDING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/luofan/buildup")
    args = ap.parse_args()
    acc_table(args.dir)
    print()
    perf_table(args.dir)
    print()
    naive_table(args.dir)
    print()
    seeds_table(args.dir)


if __name__ == "__main__":
    main()
