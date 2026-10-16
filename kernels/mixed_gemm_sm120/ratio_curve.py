"""T1 round-2 — THE RATIO CURVE for B1 (lead requirement).

Characterizes B1 fused-kernel latency as a function of the high-precision ratio
r = K_high / K_total at fixed compute-saturated scenarios, against same-session
pure-format presf bounds and the ideal interpolation
    t_ideal(r) = t_fp4 * (1 - r) + t_fp8 * r.

excess_pct = (t_b1 - t_ideal) / t_ideal * 100 — bumps expose fusion overhead.

Endpoints: K_high = 0 (NVFP4-only, Phase 2 skipped) and K_low = 0 (MXFP8-only,
Phase 1 skipped) run through the SAME b1_fused entry (loop bounds n1/n2 go to 0)
and are cosine-gated like every other point.

Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python ratio_curve.py
Rows are flushed to ratio_curve.csv per point (survives interruption).
"""
import csv
import os
import sys
import torch

import mixed_gemm as mg
import b1_fused as b1
import b1_tma as bt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "ratio_curve.csv")

# (scenario, M, N, K_total)
SCENARIOS = [
    ("qkv_class",  2048, 2048, 2048),
    ("down_class", 2048, 2048, 8192),
    ("gate_class", 2048, 8192, 2048),
]


def khigh_list(K_total):
    # Densified grid (project-lead directive, round-4):
    #  K_total=2048 (qkv/gate): step 128 -> r every 1/16 (17 points).
    #  K_total=8192 (down):     step 512 (17 points) + low-end 128/256/384.
    if K_total == 2048:
        ks = set(range(0, K_total + 1, 128))
    else:
        ks = set(range(0, K_total + 1, 512)) | {128, 256, 384}
    return sorted(k for k in ks
                  if 0 <= k <= K_total and k % 64 == 0 and (K_total - k) % 64 == 0)


def median_ms(fn, iters=50, warmup=10):
    med, _ = mg._time_ms_median(fn, iters=iters, warmup=warmup)
    return med


def best_dispatch(p, M, N, K_low, K_high):
    """Product-form best dispatch per point. Non-TMA auto (b1_single) is the
    baseline for every point; for pure-MXFP8 points (K_low==0, K_high large) the
    TMA phase-2 kernel is also a candidate (round-3: wins down-class r=1 large-K).
    Returns (latency_ms, variant_label, cosine_of_chosen)."""
    lat_b = median_ms(lambda: b1.b1_single(p, M, N, K_low, K_high))
    cand = [(lat_b, "nonTMA_auto", lambda: b1.b1_single(p, M, N, K_low, K_high))]
    if K_low == 0 and K_high >= 4096:
        lat_t = median_ms(lambda: bt.tma_mx(p, M, N, K_high, stages=3))
        cand.append((lat_t, "TMA_mx_st3", lambda: bt.tma_mx(p, M, N, K_high, stages=3)))
    cand.sort(key=lambda x: x[0])
    lat, lab, fn = cand[0]
    return lat, lab


def main():
    print("Building extensions...")
    b1.get_ext()
    b1.get_presf_ext()
    bt.get_tma()
    print("Built.\n")

    f = open(CSV_PATH, "w", newline="")   # OVERWRITE: densified version
    w = csv.writer(f)
    w.writerow(["scenario", "M", "N", "K_total", "K_high", "r",
                "latency_ms", "variant", "t_fp4", "t_fp8", "t_ideal",
                "excess_pct", "cosine"])
    f.flush()

    all_pass = True
    for (scen, M, N, K_total) in SCENARIOS:
        print(f"=== {scen}: M={M} N={N} K_total={K_total} ===")
        # Same-session pure-format presf bounds (cosine-gated vs their own oracle).
        p4, p8 = b1.prep_bounds_presf(M, N, K_total, seed=1)
        c4 = b1.cosine(b1.pure_fp4_presf(p4, M, N, K_total).float(), p4["oracle"])
        c8 = b1.cosine(b1.pure_fp8_presf(p8, M, N, K_total).float(), p8["oracle"])
        print(f"  bounds gate: fp4_presf={c4:.6f} fp8_presf={c8:.6f}")
        if min(c4, c8) < 0.999:
            print("  BOUNDS GATE FAILED — skipping scenario")
            all_pass = False
            continue
        t4 = median_ms(lambda: b1.pure_fp4_presf(p4, M, N, K_total))
        t8 = median_ms(lambda: b1.pure_fp8_presf(p8, M, N, K_total))
        print(f"  bounds: t_fp4={t4:.4f}ms t_fp8={t8:.4f}ms")
        del p4, p8
        torch.cuda.empty_cache()

        for K_high in khigh_list(K_total):
            K_low = K_total - K_high
            r = K_high / K_total
            p, oracle = mg.prep_mixed(M, N, K_low, K_high, seed=1)
            D = b1.b1_single(p, M, N, K_low, K_high).float()
            cos = b1.cosine(D, oracle)
            del D, oracle
            if cos < 0.999:
                print(f"  K_high={K_high:5d} r={r:.3f}: COSINE GATE FAIL "
                      f"({cos:.6f}) — not timing")
                w.writerow([scen, M, N, K_total, K_high, f"{r:.4f}",
                            "", "", f"{t4:.5f}", f"{t8:.5f}", "", "", f"{cos:.6f}"])
                f.flush()
                all_pass = False
                continue
            lat, variant = best_dispatch(p, M, N, K_low, K_high)
            t_ideal = t4 * (1 - r) + t8 * r
            excess = (lat - t_ideal) / t_ideal * 100.0
            print(f"  K_high={K_high:5d} r={r:.3f}: cos={cos:.6f} "
                  f"lat={lat:.4f}ms [{variant}] ideal={t_ideal:.4f}ms excess={excess:+.1f}%")
            w.writerow([scen, M, N, K_total, K_high, f"{r:.4f}",
                        f"{lat:.5f}", variant, f"{t4:.5f}", f"{t8:.5f}",
                        f"{t_ideal:.5f}", f"{excess:.2f}", f"{cos:.6f}"])
            f.flush()
            del p
            torch.cuda.empty_cache()
        print()
    f.close()
    print("RATIO CURVE DONE" if all_pass else "RATIO CURVE DONE (SOME GATES FAILED)")


if __name__ == "__main__":
    main()
