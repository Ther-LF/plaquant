"""T3 round-2 — M=256 mid-range config sweep.

Round-1 weakness: M=256 bands q_proj_3B 162% / down_proj_3B 232% (tile config
tuned for M=2048). Sweeps every B1 config (incl. new cfg7 = 64x64 4-warp KMUL2)
x stages at M=256, cosine-gated, median-timed, with same-session presf bounds.
Also re-times the current auto dispatch (cfg=0) as the before-number.

Run inside luofan-plaquant on a FREE GPU. Appends to t3_sweep.csv per row.
"""
import csv
import os
import torch

import mixed_gemm as mg
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "t3_sweep.csv")

SHAPES = {   # name: (N, K_low, K_high)
    "q_proj_1B":    (2048, 1792, 256),
    "q_proj_3B":    (3072, 2688, 384),
    "down_proj_3B": (3072, 7168, 1024),
    "gate_proj_1B": (8192, 1792, 256),
}
M = 256
CFGS = [3, 4, 5, 6, 7, 10]
STAGES = [1, 2, 3]


def median_ms(fn, iters=50, warmup=10):
    med, _ = mg._time_ms_median(fn, iters=iters, warmup=warmup)
    return med


def main():
    b1.get_ext()
    b1.get_presf_ext()
    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["shape", "M", "cfg", "stages", "latency_ms", "cosine",
                    "t_fp4", "t_fp8", "band_pct"])
        f.flush()

    for name, (N, K_low, K_high) in SHAPES.items():
        K_total = K_low + K_high
        p, oracle = mg.prep_mixed(M, N, K_low, K_high, seed=1)
        p4, p8 = b1.prep_bounds_presf(M, N, K_total, seed=1)
        t4 = median_ms(lambda: b1.pure_fp4_presf(p4, M, N, K_total))
        t8 = median_ms(lambda: b1.pure_fp8_presf(p8, M, N, K_total))
        print(f"=== {name}: M={M} N={N} Kl={K_low} Kh={K_high} "
              f"t_fp4={t4:.4f} t_fp8={t8:.4f} ===")

        # current auto dispatch (before-number)
        lat0 = median_ms(lambda: b1.b1_single(p, M, N, K_low, K_high))
        band0 = mg.band_position(lat0, t4, t8)
        print(f"  auto(cfg0/st0): {lat0:.4f}ms band={band0:.0f}%")
        w.writerow([name, M, 0, 0, f"{lat0:.5f}", "", f"{t4:.5f}",
                    f"{t8:.5f}", f"{band0:.1f}"])
        f.flush()

        for cfg in CFGS:
            for st in STAGES:
                try:
                    D = b1.b1_single(p, M, N, K_low, K_high,
                                     stages=st, cfg=cfg).float()
                except RuntimeError as e:
                    print(f"  cfg={cfg} st={st}: LAUNCH FAIL {e}")
                    continue
                cos = b1.cosine(D, oracle)
                del D
                if cos < 0.999:
                    print(f"  cfg={cfg} st={st}: COSINE FAIL {cos:.6f}")
                    w.writerow([name, M, cfg, st, "", f"{cos:.6f}",
                                f"{t4:.5f}", f"{t8:.5f}", ""])
                    f.flush()
                    continue
                lat = median_ms(lambda: b1.b1_single(p, M, N, K_low, K_high,
                                                     stages=st, cfg=cfg))
                band = mg.band_position(lat, t4, t8)
                print(f"  cfg={cfg} st={st}: {lat:.4f}ms cos={cos:.6f} "
                      f"band={band:.0f}%")
                w.writerow([name, M, cfg, st, f"{lat:.5f}", f"{cos:.6f}",
                            f"{t4:.5f}", f"{t8:.5f}", f"{band:.1f}"])
                f.flush()
        del p, oracle, p4, p8
        torch.cuda.empty_cache()
        print()
    f.close()
    print("T3 SWEEP DONE")


if __name__ == "__main__":
    main()
