"""Round-6 B1 multi-axis grid sweep (self-contained, interruption-robust).

Sweeps the dual-phase NVFP4+MXFP8 TMA kernel across INDEPENDENT axes:
  - CTA tile (BM x BN):  128x128 / 128x256 / 256x128 / 64x128 / 128x64
  - pipeline stages:     2,3,4,5,6,8   (first-class axis, per tile, per r)
  - consumer-warp layout: WM2xWN4 (tile0) vs WM4xWN2 (tile6)  [8 warps, re-laid]
  - CTA rasterization/swizzle (group-M): swz = 0 / 2 / 4 / 8  (gate large-N L2 reuse)
  - BK per mainloop iter (kmul): 1 (BK32) / 2 (BK64) / 4 (BK128)
  - launch_bounds min-blocks/SM (minb): 1 / 2

Shapes: qkv N2048K2048 / down N2048K8192 / gate N8192K2048; M in {2048,8192}.
Dense ratio grid. excess vs same-session CUTLASS pure-NVFP4/MXFP8 presf bounds
with linear interpolation: ideal(r) = t_fp4*(1-r) + t_fp8*r.

Each (shape,M,r,config) is TIMED; cosine is gated once per (config,shape,phase)
family (phase = pure-fp4 / mixed / pure-fp8) and cached. Every row is appended +
flushed to sweep_r6.csv IMMEDIATELY so a controller can harvest partial CSVs.

SF staging is the round-5 width-generic vectorized path (sf_copy_row<NB>); epilogue
is the fixed half2 store. Those two axes are single-valued here (documented, would
need kernel work to parametrize).

Run detached inside luofan-plaquant on a FREE GPU:
  docker exec -d -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant \
    -e TORCH_CUDA_ARCH_LIST=12.0a -e CUDA_VISIBLE_DEVICES=0 luofan-plaquant \
    bash -c 'python kernels/mixed_gemm_sm120/sweep_r6.py > \
             kernels/mixed_gemm_sm120/logs/sweep_r6.log 2>&1'
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
CSV_PATH = os.path.join(HERE, "sweep_r6.csv")

# ---- config catalog: EXACTLY the tuples the b1_tma_ext dispatch supports ----
# (tile, stages, swz, kmul, minb, warps, need_N256, need_M256)
WARPS = {0: 8, 6: 8, 1: 16, 2: 16, 3: 4, 4: 4}
TILE_NAME = {0: "128x128", 6: "128x128w42", 1: "128x256", 2: "256x128",
             3: "64x128", 4: "128x64"}
CONFIGS = []
def _add(tile, st, swz, km, mb):
    CONFIGS.append((tile, st, swz, km, mb, WARPS[tile],
                    tile == 1, tile == 2))
# tile0 128x128 WM2WN4 (8 warps) -- baseline tile, full stage + axis sweep
# st6 @128x128/km2 is >99KB smem (illegal), pruned; st6 survives only at km1 (BK32).
for st in (2, 3, 4, 5):
    _add(0, st, 0, 2, 1)
_add(0, 3, 0, 2, 2)                      # minb axis
for sw in (2, 4, 8):                      # rasterization axis
    _add(0, 3, sw, 2, 1); _add(0, 4, sw, 2, 1)
for st in (4, 6, 8):                      # BK=32 (kmul=1): longer K-pipe
    _add(0, st, 0, 1, 1)
_add(0, 2, 0, 4, 1)                       # BK=128 (kmul=4): fat K-iters
# tile6 128x128 WM4WN2 (warp-layout variant)
for sw in (0, 4):
    _add(6, 3, sw, 2, 1); _add(6, 4, sw, 2, 1)
# tile1 128x256 WM2WN8 (16 warps, large-N / gate) -- needs N%256
for sw in (0, 4):
    _add(1, 2, sw, 2, 1); _add(1, 3, sw, 2, 1)
# tile2 256x128 WM4WN4 (16 warps, large-M) -- needs M%256
_add(2, 2, 0, 2, 1); _add(2, 3, 0, 2, 1)
# tile3 64x128 WM1WN4 (4 warps, small-M / more CTAs)
for st in (3, 4, 5):
    _add(3, st, 0, 2, 1)
# tile4 128x64 WM2WN2 (4 warps, small-N)
for st in (3, 4, 5):
    _add(4, st, 0, 2, 1)

SHAPES = [("qkv", 2048, 2048), ("down", 2048, 8192), ("gate", 8192, 2048)]
M_LIST = [2048, 8192]
COS_GATE = 0.999   # record exact cosine; PASS if clean (>=0.999, round-5 convention)


def ratio_grid(K_total):
    if K_total == 8192:
        pts = sorted(set([0, 128, 384] + list(range(0, 8193, 512))))
    else:  # 2048
        pts = list(range(0, 2049, 128))
    return pts  # K_high values


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=30, warmup=8)
    return m


def main():
    print("[sweep_r6] building b1_tma_ext (may take many minutes)...", flush=True)
    t0 = time.time()
    ext = t.get_tma()
    b1.get_ext(); b1.get_presf_ext()
    print(f"[sweep_r6] COMPILE_OK in {time.time()-t0:.0f}s; {len(CONFIGS)} configs",
          flush=True)

    def dual(p, M, N, Kl, Kh, cfg):
        tile, st, swz, km, mb = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
        return ext.b1_fused_tma_dual(
            p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
            p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
            p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, swz, km, mb)

    header = ["shape", "N", "K_total", "M", "r", "K_low", "K_high",
              "tile", "tile_name", "stages", "warps", "swz", "kmul", "bk",
              "minb", "sf_mode", "epi", "latency_ms", "cosine", "gated",
              "t_fp4", "t_fp8", "ideal_ms", "excess_pct", "status"]
    new = not os.path.exists(CSV_PATH)
    fcsv = open(CSV_PATH, "a", newline="", buffering=1)  # line-buffered
    w = csv.writer(fcsv)
    if new:
        w.writerow(header); fcsv.flush()

    cos_cache = {}   # (tile,st,swz,km,mb,shape,phase) -> cosine (or -1 if failed)
    n_rows = 0

    for shape, N, KT in SHAPES:
        for M in M_LIST:
            # same-session CUTLASS presf bounds (full-K pure formats)
            try:
                p4, p8 = b1.prep_bounds_presf(M, N, KT, seed=1)
                t_fp4 = med(lambda: b1.pure_fp4_presf(p4, M, N, KT))
                t_fp8 = med(lambda: b1.pure_fp8_presf(p8, M, N, KT))
                del p4, p8
            except Exception as e:
                print(f"[sweep_r6] BOUND FAIL {shape} M={M}: {e}", flush=True)
                t_fp4 = t_fp8 = float("nan")
            print(f"=== {shape} M={M} N={N} K={KT} t_fp4={t_fp4:.4f} "
                  f"t_fp8={t_fp8:.4f} ===", flush=True)

            for Kh in ratio_grid(KT):
                # align to the coarsest kmul requirement handled per-config below
                Kl = KT - Kh
                r = Kh / KT
                phase = "fp4" if Kh == 0 else ("fp8" if Kl == 0 else "mix")
                try:
                    p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
                except Exception as e:
                    print(f"  prep fail r={r:.3f}: {e}", flush=True)
                    continue
                ideal = t_fp4 * (1 - r) + t_fp8 * r

                for cfg in CONFIGS:
                    tile, st, swz, km, mb, warps, needN, needM = cfg
                    if needN and (N % 256):
                        continue
                    if needM and (M % 256):
                        continue
                    # kmul divisibility (dispatch TORCH_CHECKs these)
                    if Kl % (km * 64) != 0 or Kh % (km * 32) != 0:
                        continue
                    bk = 32 * km
                    key = (tile, st, swz, km, mb, shape, phase)
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
                            status = "LOWCOS"
                            lat = float("nan")
                        else:
                            lat = med(lambda: dual(p, M, N, Kl, Kh, cfg))
                    except Exception as e:
                        status = "ERR:" + type(e).__name__
                        lat = float("nan")
                        cos_cache[key] = cosv if cosv is not None else -1.0
                        # first-line only to keep log compact
                        msg = str(e).splitlines()[0][:120] if str(e) else ""
                        print(f"  cfg t{tile} st{st} sw{swz} km{km} mb{mb} "
                              f"r={r:.3f}: {status} {msg}", flush=True)
                    excess = ((lat - ideal) / ideal * 100.0) if (lat == lat and ideal == ideal and ideal > 0) else float("nan")
                    w.writerow([shape, N, KT, M, f"{r:.5f}", Kl, Kh,
                                tile, TILE_NAME[tile], st, warps, swz, km, bk,
                                mb, "vec", "half2", f"{lat:.5f}",
                                f"{cosv:.7f}" if cosv is not None else "", gated,
                                f"{t_fp4:.5f}", f"{t_fp8:.5f}", f"{ideal:.5f}",
                                f"{excess:.2f}", status])
                    fcsv.flush()
                    n_rows += 1
                del p, oracle
                torch.cuda.empty_cache()
                print(f"  r={r:.3f} K_high={Kh} done ({n_rows} rows)", flush=True)

    fcsv.close()
    print(f"[sweep_r6] SWEEP_DONE {n_rows} rows -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
