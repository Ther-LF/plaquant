"""Round-8 TC-duty profiling runner (one kernel instance under ncu).
Usage:
  prof_tc.py build                      # build exts, exit (no ncu)
  prof_tc.py dual M N Kl Kh tile st sw km mb pers   # our fused dual kernel
  prof_tc.py fp4  M N Kl Kh             # CUTLASS pure-NVFP4 presf bound (Kt=Kl+Kh)
  prof_tc.py fp8  M N Kl Kh             # CUTLASS pure-MXFP8 presf bound
Does 5 warmup launches + 2 profiled launches; drive ncu with `-s 5 -c 1 -k regex:...`
so the captured instance is warm and uniquely the target kernel.
"""
import sys
import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

kind = sys.argv[1]
if kind == "build":
    t.get_tma(); b1.get_presf_ext(); b1.get_ext()
    print("BUILD_OK"); sys.exit(0)

M, N, Kl, Kh = (int(x) for x in sys.argv[2:6])
Kt = Kl + Kh

if kind == "dual":
    tile, st, sw, km, mb, pers = (int(x) for x in sys.argv[6:12])
    kpipe = int(sys.argv[12]) if len(sys.argv) > 12 else 0
    sfasync = int(sys.argv[13]) if len(sys.argv) > 13 else 0
    ext = t.get_tma()
    p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    fn = lambda: ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, st, tile, sw, km, mb, pers, kpipe, sfasync)
elif kind == "fp4":
    b1.get_presf_ext()
    p4, _ = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = lambda: b1.pure_fp4_presf(p4, M, N, Kt)
elif kind == "fp8":
    b1.get_presf_ext()
    _, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = lambda: b1.pure_fp8_presf(p8, M, N, Kt)
else:
    print("bad kind", kind); sys.exit(1)

for _ in range(5):
    fn()
torch.cuda.synchronize()
for _ in range(2):
    fn()
torch.cuda.synchronize()
print("prof done", kind, M, N, Kl, Kh)
