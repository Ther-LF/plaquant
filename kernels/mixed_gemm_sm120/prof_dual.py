"""ncu driver: profile ONE target kernel once (after 2 warmups).
Usage: prof_dual.py <which> <M> <N> <Kl> <Kh> <st>
  which: dual | cpasync | tma_mx | fp4p | fp8p
Capture the 3rd matching launch with:  ncu -k <kname> -c 1 -s 2 ...
"""
import sys, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

which = sys.argv[1]
M, N, Kl, Kh, st = (int(x) for x in sys.argv[2:7])
ext = t.get_tma(); b1.get_ext(); b1.get_presf_ext()
p, oracle = mg.prep_mixed(M, N, Kl, Kh, seed=1)

def dual():
    return ext.b1_fused_tma_dual(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, st)
def cpasync():
    return b1.b1_single(p, M, N, Kl, Kh)
def tma_mx():
    return t.tma_mx(p, M, N, Kh, stages=st)

if which == "dual": fn = dual
elif which == "cpasync": fn = cpasync
elif which == "tma_mx": fn = tma_mx
elif which == "fp4p":
    p4, _ = b1.prep_bounds_presf(M, N, Kl+Kh, seed=1); fn = lambda: b1.pure_fp4_presf(p4, M, N, Kl+Kh)
elif which == "fp8p":
    _, p8 = b1.prep_bounds_presf(M, N, Kl+Kh, seed=1); fn = lambda: b1.pure_fp8_presf(p8, M, N, Kl+Kh)
for _ in range(3):
    fn()
torch.cuda.synchronize()
fn()
torch.cuda.synchronize()
print("prof done", which, M, N, Kl, Kh, st)
