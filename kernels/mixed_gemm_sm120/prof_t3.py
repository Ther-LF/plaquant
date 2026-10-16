"""ncu driver for T3 swizzle L2 evidence. Usage: prof_t3.py <M> <N> <K> <st> <swz>"""
import sys, torch
import mixed_gemm as mg
import b1_tma as t
M, N, K, st, sw = (int(x) for x in sys.argv[1:6])
ext = t.get_tma()
p, _ = mg.prep_mixed(M, N, 0, K, seed=1)
fn = lambda: ext.b1_fused_tma_mx(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                                 p["s_x"], p["s_w"], M, N, K, st, 1, 0, sw)
for _ in range(3):
    fn()
torch.cuda.synchronize()
fn()
torch.cuda.synchronize()
print("prof done", M, N, K, st, "swz", sw)
