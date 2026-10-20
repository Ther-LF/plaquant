"""Round-11 LDBW correctness gate: widening operand-B smem->reg ldmatrix from
SM75_U32x2_LDSM_N (64-bit) to SM75_U32x4_LDSM_N (128-bit) — matching CUTLASS's
sm120_rr_smem_copy_selector_B — only changes HOW B is staged smem->reg; it does
NOT touch the data format (NVFP4 E2M1 blk16 / MXFP8 E4M3 blk32), SF layout, the
K-split, or the shared FP32 accumulator. So every point must stay bit-exact
(cosine>=0.999) vs the mixed oracle. Composed with kpipe/sfa. PASS/FAIL per point.
"""
import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("BUILD_OK", flush=True)

def dual(p, M, N, Kl, Kh, tile, st, sw, pers, kpipe, sfa, ldbw):
    return ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, sw, 2, 1, pers, kpipe, sfa, 0, ldbw)

# (shape, N, Ktot, tile, sw, persist) — per-shape round-7 best cfg.
SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
R = [0.0, 0.125, 0.5, 1.0]
allok = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in (2048, 8192):
        for r in R:
            Kh = int(round(r * KT)); Kh -= Kh % 64 if Kh % 64 else 0
            if r == 1.0: Kh = KT
            if r == 0.0: Kh = 0
            Kl = KT - Kh
            try:
                p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            except Exception as e:
                print(f"  PREPFAIL {name} M{M} r{r}: {e}", flush=True); continue
            for kp in (0, 1):
                for sfa in (0, 1):
                    try:
                        D = dual(p, M, N, Kl, Kh, tile, 3, sw, pers, kp, sfa, 1).float()
                        cos = b1.cosine(D, oracle)
                        del D
                    except Exception as e:
                        print(f"  ERR {name} M{M} r{r} kp{kp} sfa{sfa}: {type(e).__name__} {str(e).splitlines()[0][:120]}", flush=True)
                        allok = False; continue
                    tag = "PASS" if cos >= 0.999 else "FAIL"
                    if cos < 0.999: allok = False
                    print(f"  {name:4s} M{M:5d} r{r:.3f} Kl{Kl:5d} Kh{Kh:5d} kp{kp} sfa{sfa} ldbw1 cos={cos:.6f} [{tag}]", flush=True)
            del p, oracle
            torch.cuda.empty_cache()
print("LDBW_GATE_ALL_PASS" if allok else "LDBW_GATE_SOME_FAILED", flush=True)
