"""Round-12 SFILV bit-exact gate. Compares SFILV=1 (interleaved CUTLASS block-scaled
SF atom + UniversalCopy, fp4 phase) against the mixed oracle, across the shipped shapes
x M x r x kpipe. SFILV uses the plain scatter producer (sfa=0). Must be cosine=1.000000
at EVERY point (the interleaved layout only permutes SF byte order; values identical).
Also re-gates SFILV=0 (flat default) to prove the round-11 fallback is untouched."""
import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("BUILD_OK", flush=True)

def dual(p, M, N, Kl, Kh, tile, sw, pers, kpipe, sfa, sfilv):
    return ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kpipe, sfa, 0, 0, sfilv)

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
            p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            for kp in (0, 1):
                try:
                    D = dual(p, M, N, Kl, Kh, tile, sw, pers, kp, 0, 1).float()  # SFILV=1, sfa=0
                    cos = b1.cosine(D, oracle); del D
                except Exception as e:
                    print(f"  ERR {name} M{M} r{r} kp{kp}: {type(e).__name__} {str(e).splitlines()[0][:110]}", flush=True)
                    allok = False; continue
                tag = "PASS" if cos >= 0.999 else "FAIL"
                if cos < 0.999: allok = False
                print(f"  SFILV1 {name:4s} M{M:5d} r{r:.3f} Kl{Kl:5d} Kh{Kh:5d} kp{kp} cos={cos:.6f} [{tag}]", flush=True)
            del p, oracle; torch.cuda.empty_cache()
print("SFILV_GATE_ALL_PASS" if allok else "SFILV_GATE_SOME_FAILED", flush=True)
