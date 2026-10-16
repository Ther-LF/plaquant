"""T2 round-2 quick gate+bench: WS-kernel changes (T2a early release, T2c
smem-staged stg.128 epilogue). Gates every shape that routes to cfg10, then
median-times the T2-relevant shapes with same-session presf bounds.

Run inside luofan-plaquant on a FREE GPU."""
import torch
import mixed_gemm as mg
import b1_fused as b1


def median_ms(fn, iters=50, warmup=10):
    med, _ = mg._time_ms_median(fn, iters=iters, warmup=warmup)
    return med


def main():
    b1.get_ext()
    b1.get_presf_ext()

    # ---- correctness gates (incl. ragged M on cfg10 and endpoints) ----
    print("=== T2 gate ===")
    ok = True
    gate_cases = [
        ("gate_proj_1B", 8192, 1792, 256, [49, 256, 2048]),
        ("q_proj_1B",    2048, 1792, 256, [2048]),
        ("down_proj_3B", 3072, 7168, 1024, [49, 2048]),
        ("fp4_only_ep",  2048, 2048, 0,    [2048]),
        ("fp8_only_ep",  2048, 0, 2048,    [2048]),
    ]
    for name, N, Kl, Kh, Ms in gate_cases:
        for M in Ms:
            p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=M)
            for st in ([2, 3] if (Kl % 128 == 0 and Kh % 64 == 0) else [1]):
                D = b1.b1_single(p, M, N, Kl, Kh, stages=st, cfg=10).float() \
                    if (Kl % 128 == 0 and Kh % 64 == 0) else \
                    b1.b1_single(p, M, N, Kl, Kh, stages=st).float()
                cos = b1.cosine(D, oracle)
                stat = "PASS" if cos >= 0.999 else "FAIL"
                if cos < 0.999:
                    ok = False
                print(f"  {name:14s} M={M:5d} cfg10 st={st}: cos={cos:.6f} [{stat}]")
            del p, oracle
    if not ok:
        print("T2 GATE FAILED — NOT BENCHMARKING")
        return

    # ---- benchmark ----
    print("\n=== T2 bench (median, same-session presf bounds) ===")
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
            print(f"  {name:14s} M={M:5d}: b1={lat:.4f}ms fp4p={t4:.4f} "
                  f"fp8p={t8:.4f} band={band:.0f}%")
            del p, p4, p8
            torch.cuda.empty_cache()
    # decode spot check (mw kernels untouched, confirm no regress)
    for name, (N, Kl, Kh) in [("q_proj_1B", (2048, 1792, 256)),
                              ("gate_proj_1B", (8192, 1792, 256))]:
        for M in [1, 16]:
            p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            lat = median_ms(lambda: b1.b1_single(p, M, N, Kl, Kh))
            print(f"  {name:14s} M={M:5d}: b1={lat:.4f}ms (decode spot)")
            del p
    print("T2 BENCH DONE")


if __name__ == "__main__":
    main()
