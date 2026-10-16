"""T2 round-2 re-bench with quiet-GPU sentinel gate.

A contending CI pytest loop on this host inflated the first T2c bench by a
~0.19ms constant. This script polls a sentinel (decode q_proj_1B M=1, known
~0.020ms on a quiet GPU) and only benchmarks once the sentinel passes, so the
numbers are honest. Correctness was already gated in t2_bench.py (all PASS).

Run inside luofan-plaquant: CUDA_VISIBLE_DEVICES=<free> python t2_bench2.py
"""
import time
import torch
import mixed_gemm as mg
import b1_fused as b1

SENTINEL_MS = 0.023
MAX_WAIT_MIN = 90


def median_ms(fn, iters=50, warmup=10):
    med, _ = mg._time_ms_median(fn, iters=iters, warmup=warmup)
    return med


def main():
    b1.get_ext()
    b1.get_presf_ext()
    ps, _ = mg.prep_mixed(1, 2048, 1792, 256, seed=1)
    sentinel = lambda: median_ms(
        lambda: b1.b1_single(ps, 1, 2048, 1792, 256), iters=30, warmup=5)

    deadline = time.time() + MAX_WAIT_MIN * 60
    while True:
        s = sentinel()
        print(f"sentinel={s:.4f} ms", flush=True)
        if s <= SENTINEL_MS:
            break
        if time.time() > deadline:
            print("TIMED OUT waiting for quiet GPU — aborting (no numbers).")
            return
        time.sleep(60)
    print("GPU quiet — benchmarking.\n", flush=True)

    shapes = {
        "gate_proj_1B": (8192, 1792, 256),
        "up_proj_1B":   (8192, 1792, 256),
        "down_proj_1B": (2048, 7168, 1024),
        "down_proj_3B": (3072, 7168, 1024),
        "q_proj_3B":    (3072, 2688, 384),
        "q_proj_1B":    (2048, 1792, 256),
    }
    for name, (N, Kl, Kh) in shapes.items():
        Kt = Kl + Kh
        for M in [256, 2048]:
            p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            lat = median_ms(lambda: b1.b1_single(p, M, N, Kl, Kh))
            p4, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
            t4 = median_ms(lambda: b1.pure_fp4_presf(p4, M, N, Kt))
            t8 = median_ms(lambda: b1.pure_fp8_presf(p8, M, N, Kt))
            band = mg.band_position(lat, t4, t8)
            tf = 2.0 * M * N * Kt / (lat * 1e-3) / 1e12
            print(f"  {name:14s} M={M:5d}: b1={lat:.4f}ms fp4p={t4:.4f} "
                  f"fp8p={t8:.4f} band={band:.0f}%  ({tf:.0f} TFLOPS)",
                  flush=True)
            del p, p4, p8
            torch.cuda.empty_cache()
        s = sentinel()
        print(f"  [sentinel recheck: {s:.4f} ms]", flush=True)
    for name, (N, Kl, Kh) in [("q_proj_1B", (2048, 1792, 256)),
                              ("gate_proj_1B", (8192, 1792, 256))]:
        for M in [1, 16]:
            p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            lat = median_ms(lambda: b1.b1_single(p, M, N, Kl, Kh))
            print(f"  {name:14s} M={M:5d}: b1={lat:.4f}ms (decode spot)", flush=True)
            del p
    print("T2 BENCH2 DONE")


if __name__ == "__main__":
    main()
