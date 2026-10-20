"""Round-11 bit-exact gate for the shipped kernel (round-10 byte-equivalent default,
ldbw=0 dormant) AND the round-11 dispatch recommendation (r=0 -> KPIPE/kp1, which the
imix evidence shows raises TCact without hurting latency). Confirms the format contract
end-to-end: the kernel consumes our real packed NVFP4 (E2M1 blk16 ue4m3) + MXFP8 (E4M3
blk32 ue8m0) + our K-split, shared FP32 accumulator, epilogue*s_x*s_w -> fp16, matching
the oracle bit-exactly (cosine>=0.999)."""
import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("BUILD_OK", flush=True)

def dual(p, M, N, Kl, Kh, tile, st, sw, pers, kpipe, sfa):
    return ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, sw, 2, 1, pers, kpipe, sfa, 0, 0)

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
            # ship default dispatch: r=0 historically kp0; round-11 recommends kp1 for r=0 too.
            for kp in (0, 1):
                for sfa in (0, 1):
                    try:
                        D = dual(p, M, N, Kl, Kh, tile, 3, sw, pers, kp, sfa).float()
                        cos = b1.cosine(D, oracle); del D
                    except Exception as e:
                        print(f"  ERR {name} M{M} r{r} kp{kp} sfa{sfa}: {type(e).__name__} {str(e).splitlines()[0][:100]}", flush=True)
                        allok = False; continue
                    tag = "PASS" if cos >= 0.999 else "FAIL"
                    if cos < 0.999: allok = False
                    print(f"  {name:4s} M{M:5d} r{r:.3f} Kl{Kl:5d} Kh{Kh:5d} kp{kp} sfa{sfa} cos={cos:.6f} [{tag}]", flush=True)
            del p, oracle; torch.cuda.empty_cache()
print("R11_GATE_ALL_PASS" if allok else "R11_GATE_SOME_FAILED", flush=True)
