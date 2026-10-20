"""ncu driver: run ONE dual-kernel config (3 warmups + 1 captured launch) for L2 study.
Usage: prof_l2.py <M> <N> <Kl> <Kh> <st> <tile> <swz> <persist>
Capture the 3rd matching launch with:  ncu -k regex:b1_tma_dual_kernel -c 1 -s 2 ...
"""
import sys, torch
import mixed_gemm as mg
import b1_tma as t

M, N, Kl, Kh, st, tile, swz, pers = (int(x) for x in sys.argv[1:9])
ext = t.get_tma()
p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
fn = lambda: ext.b1_fused_tma_dual(
    p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
    p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
    p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, swz, 2, 1, pers)
for _ in range(3):
    fn()
torch.cuda.synchronize()
fn()
torch.cuda.synchronize()
print("prof done", M, N, Kl, Kh, "st", st, "tile", tile, "swz", swz, "P", pers)
