"""Round-3 T1 — gate-class before-numbers + existing-cfg sweep (quiet GPU).

Same-session presf bounds; every timed point is cosine-gated first.
Rows flushed to r3_t1_sweep.csv per point (survives interruption).

Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python r3_t1_sweep.py
"""
import csv
import os
import torch

import mixed_gemm as mg
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "r3_t1_sweep.csv")

# (case, M, N, K_low, K_high)
CASES = [
    ("gate_r0", 2048, 8192, 2048, 0),      # pure-FP4 path, the structural target
    ("gate_1B", 2048, 8192, 1792, 256),    # production gate/up shape (r=1/8)
]


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m


def main():
    print("Building extensions...")
    b1.get_ext()
    b1.get_presf_ext()
    print("Built.\n")
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["case", "M", "N", "K_low", "K_high", "cfg", "stages",
                    "latency_ms", "cosine", "t_fp4", "t_fp8"])
        f.flush()

    for (name, M, N, Kl, Kh) in CASES:
        Kt = Kl + Kh
        p4, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
        c4 = b1.cosine(b1.pure_fp4_presf(p4, M, N, Kt).float(), p4["oracle"])
        c8 = b1.cosine(b1.pure_fp8_presf(p8, M, N, Kt).float(), p8["oracle"])
        assert min(c4, c8) >= 0.999, f"bounds gate fail {name}: {c4} {c8}"
        t4 = med(lambda: b1.pure_fp4_presf(p4, M, N, Kt))
        t8 = med(lambda: b1.pure_fp8_presf(p8, M, N, Kt))
        del p4, p8
        torch.cuda.empty_cache()
        print(f"=== {name} M={M} N={N} Kl={Kl} Kh={Kh}: "
              f"t_fp4={t4:.4f} t_fp8={t8:.4f} ===")

        p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
        for cfg in [4, 5, 6, 7, 10]:
            for st in ([2, 3] if cfg == 10 else [1, 2, 3]):
                D = b1.b1_single(p, M, N, Kl, Kh, stages=st, cfg=cfg).float()
                cos = b1.cosine(D, oracle)
                del D
                if cos < 0.999:
                    print(f"  cfg{cfg:2d} st{st}: COSINE FAIL {cos:.6f} — not timing")
                    w.writerow([name, M, N, Kl, Kh, cfg, st, "", f"{cos:.6f}",
                                f"{t4:.5f}", f"{t8:.5f}"])
                    f.flush()
                    continue
                lat = med(lambda: b1.b1_single(p, M, N, Kl, Kh, stages=st, cfg=cfg))
                print(f"  cfg{cfg:2d} st{st}: {lat:.4f} ms  cos={cos:.6f}  "
                      f"vs_fp4 {lat / t4:.3f}x")
                w.writerow([name, M, N, Kl, Kh, cfg, st, f"{lat:.5f}",
                            f"{cos:.6f}", f"{t4:.5f}", f"{t8:.5f}"])
                f.flush()
        lat = med(lambda: b1.b1_single(p, M, N, Kl, Kh))
        print(f"  auto    : {lat:.4f} ms  vs_fp4 {lat / t4:.3f}x")
        w.writerow([name, M, N, Kl, Kh, 0, 0, f"{lat:.5f}", "",
                    f"{t4:.5f}", f"{t8:.5f}"])
        f.flush()
        del p, oracle
        torch.cuda.empty_cache()
        print()
    f.close()
    print("R3 T1 SWEEP DONE")


if __name__ == "__main__":
    main()
