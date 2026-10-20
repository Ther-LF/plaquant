"""Round-7 B1 sweep: persistent tile-scheduler knob vs round-6 non-persistent.

Self-contained, interruption-robust (detached + per-row CSV flush). Compares the
persistent scheduler (PERSIST=1: grid=#SM, CTA strides over tiles, continuous ring
-> pipeline fill paid once/CTA) against the round-6 non-persistent path, over the
128x128 / 128x128-w42 km2 family x swz{0,4,8} x stages{3,4}.

Target: gate (N8192 K2048) M in {2048,8192}, dense ratio grid. Regression guard:
qkv (N2048 K2048) + down (N2048 K8192) on a coarse grid (persistent must be <=
or ~= non-persistent; else keep round-6 best).

excess vs SAME-SESSION CUTLASS pure-NVFP4/MXFP8 presf bounds:
  ideal(r) = t_fp4*(1-r) + t_fp8*r.

Run detached inside luofan-plaquant on a FREE GPU:
  docker exec -d -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant \
    -e TORCH_CUDA_ARCH_LIST=12.0a -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES=0 luofan-plaquant \
    bash -c 'python kernels/mixed_gemm_sm120/r7_sweep.py > \
             kernels/mixed_gemm_sm120/logs/r7_sweep.log 2>&1'
"""
import csv
import os
import sys
import time
import traceback

import torch

import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "r7_sweep.csv")

# (tile, stages, swz, persist). km=2, minb=1 fixed. tile 0=128x128(WM2WN4), 6=w42(WM4WN2).
def cfg_set(full):
    cfgs = []
    tiles = (0, 6)
    sts = (3, 4)
    sws = (0, 4, 8)
    for tile in tiles:
        for st in sts:
            for sw in sws:
                cfgs.append((tile, st, sw, 0))   # non-persistent (round-6)
                cfgs.append((tile, st, sw, 1))   # persistent (round-7)
    if not full:
        # regression guard: smaller set (sw 0/8 only)
        cfgs = [c for c in cfgs if c[2] in (0, 8) and c[1] == 3] + \
               [c for c in cfgs if c[2] == 0 and c[1] == 4]
    return cfgs

# gate = target (full grid, dense r). qkv/down = regression guard (coarse r).
SHAPES = [("gate", 8192, 2048, True), ("qkv", 2048, 2048, False),
          ("down", 2048, 8192, False)]
M_LIST = [2048, 8192]
COS_GATE = 0.999


def ratio_grid(K_total, full):
    if full:
        if K_total == 8192:
            return sorted(set([0, 128, 384] + list(range(0, 8193, 512))))
        return list(range(0, 2049, 128))
    # coarse regression grid
    if K_total == 8192:
        return [0, 1024, 4096, 8192]
    return [0, 256, 1024, 2048]


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=30, warmup=8)
    return m


def main():
    print("[r7] building b1_tma_ext (persistent knob)...", flush=True)
    t0 = time.time()
    ext = t.get_tma()
    b1.get_ext(); b1.get_presf_ext()
    print(f"[r7] COMPILE_OK in {time.time()-t0:.0f}s", flush=True)

    def dual(p, M, N, Kl, Kh, cfg):
        tile, st, sw, pers = cfg
        return ext.b1_fused_tma_dual(
            p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
            p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
            p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, sw, 2, 1, pers)

    header = ["shape", "N", "K_total", "M", "r", "K_low", "K_high",
              "tile", "stages", "swz", "persist", "latency_ms", "cosine",
              "gated", "t_fp4", "t_fp8", "ideal_ms", "excess_pct", "status"]
    new = not os.path.exists(CSV_PATH)
    fcsv = open(CSV_PATH, "a", newline="", buffering=1)
    w = csv.writer(fcsv)
    if new:
        w.writerow(header); fcsv.flush()

    cos_cache = {}
    n_rows = 0
    for shape, N, KT, full in SHAPES:
        cfgs = cfg_set(full)
        for M in M_LIST:
            try:
                p4, p8 = b1.prep_bounds_presf(M, N, KT, seed=1)
                t_fp4 = med(lambda: b1.pure_fp4_presf(p4, M, N, KT))
                t_fp8 = med(lambda: b1.pure_fp8_presf(p8, M, N, KT))
                del p4, p8
            except Exception as e:
                print(f"[r7] BOUND FAIL {shape} M={M}: {e}", flush=True)
                t_fp4 = t_fp8 = float("nan")
            print(f"=== {shape} M={M} N={N} K={KT} t_fp4={t_fp4:.4f} "
                  f"t_fp8={t_fp8:.4f} ({len(cfgs)} cfgs) ===", flush=True)

            for Kh in ratio_grid(KT, full):
                Kl = KT - Kh
                r = Kh / KT
                phase = "fp4" if Kh == 0 else ("fp8" if Kl == 0 else "mix")
                try:
                    p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
                except Exception as e:
                    print(f"  prep fail r={r:.3f}: {e}", flush=True)
                    continue
                ideal = t_fp4 * (1 - r) + t_fp8 * r

                for cfg in cfgs:
                    tile, st, sw, pers = cfg
                    key = (tile, st, sw, pers, shape, phase)
                    gated = 0
                    cosv = cos_cache.get(key, None)
                    status = "ok"
                    try:
                        if cosv is None:
                            D = dual(p, M, N, Kl, Kh, cfg).float()
                            cosv = b1.cosine(D, oracle)
                            del D
                            cos_cache[key] = cosv
                            gated = 1
                        if cosv < COS_GATE:
                            status = "LOWCOS"; lat = float("nan")
                        else:
                            lat = med(lambda: dual(p, M, N, Kl, Kh, cfg))
                    except Exception as e:
                        status = "ERR:" + type(e).__name__
                        lat = float("nan")
                        cos_cache[key] = cosv if cosv is not None else -1.0
                        msg = str(e).splitlines()[0][:120] if str(e) else ""
                        print(f"  cfg t{tile} st{st} sw{sw} P{pers} r={r:.3f}: "
                              f"{status} {msg}", flush=True)
                    excess = ((lat - ideal) / ideal * 100.0) if (lat == lat and ideal == ideal and ideal > 0) else float("nan")
                    w.writerow([shape, N, KT, M, f"{r:.5f}", Kl, Kh, tile, st, sw,
                                pers, f"{lat:.5f}",
                                f"{cosv:.7f}" if cosv is not None else "", gated,
                                f"{t_fp4:.5f}", f"{t_fp8:.5f}", f"{ideal:.5f}",
                                f"{excess:.2f}", status])
                    fcsv.flush(); n_rows += 1
                del p, oracle
                torch.cuda.empty_cache()
                print(f"  r={r:.3f} K_high={Kh} done ({n_rows} rows)", flush=True)
    fcsv.close()
    print(f"[r7] SWEEP_DONE {n_rows} rows -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
