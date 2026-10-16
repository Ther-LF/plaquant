"""T1 dual-phase TMA validator + bench. First gates the pure-NVFP4 (K_high=0)
path (de-risks the SU4 ldmatrix consume), then r=1/2, r=1/8, r=1, then the
down-class ratio curve excess over the ideal interpolation.
    CUDA_VISIBLE_DEVICES=<free> python t1_probe.py
"""
import os, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

ext = t.get_tma()
b1.get_ext(); b1.get_presf_ext()
print("Built.\n")

def dual(p, M, N, Kl, Kh, st=0):
    return ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, st)

# K_total=8192 down-class, M2048 N2048. r = Kh/Ktot.
M, N, KT = 2048, 2048, 8192
cases = [
    ("pure_nvfp4", KT, 0),       # SU4 de-risk: only consume4 runs
    ("r=1/8",      7168, 1024),
    ("r=1/4",      6144, 2048),
    ("r=1/2",      4096, 4096),
    ("r=1",        0,    8192),
]
print("=== cosine gate (dual TMA vs mixed oracle) ===")
ok = True
for name, Kl, Kh in cases:
    p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    for st in [3, 4]:
        D = dual(p, M, N, Kl, Kh, st).float()
        cos = b1.cosine(D, oracle)
        rel = ((D - oracle).norm() / (oracle.norm() + 1e-12)).item()
        tag = "PASS" if cos >= 0.999 else "FAIL"
        if cos < 0.999: ok = False
        print(f"  {name:11s} Kl={Kl:5d} Kh={Kh:5d} st={st}: cos={cos:.6f} rel={rel:.2e} [{tag}]")
print("T1 COSINE ALL PASS\n" if ok else "T1 COSINE SOME FAILED\n")
if not ok:
    raise SystemExit("correctness gate failed - abort bench")

print("=== down-class ratio curve: dual-TMA excess over ideal interp ===")
# ideal(r) = t_fp4*(1-r) + t_fp8*r ; excess = (t_dual - ideal)/ideal
for name, Kl, Kh in cases:
    r = Kh / KT
    p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    p4, p8 = b1.prep_bounds_presf(M, N, KT, seed=1)
    t4 = t.median_ms(lambda: b1.pure_fp4_presf(p4, M, N, KT))
    t8 = t.median_ms(lambda: b1.pure_fp8_presf(p8, M, N, KT))
    ideal = t4 * (1 - r) + t8 * r
    best, bst = 1e9, -1
    for st in [3, 4, 5]:
        lt = t.median_ms(lambda: dual(p, M, N, Kl, Kh, st))
        if lt < best: best, bst = lt, st
    # also cp.async WS dual (existing b1) for comparison
    lcp = t.median_ms(lambda: b1.b1_single(p, M, N, Kl, Kh))
    exd = (best - ideal) / ideal * 100.0
    excp = (lcp - ideal) / ideal * 100.0
    print(f"  {name:11s} r={r:.3f} ideal={ideal:.4f}  cp.async={lcp:.4f}({excp:+.1f}%)  dualTMA(st{bst})={best:.4f}({exd:+.1f}%)")
print("\nT1 DONE")
