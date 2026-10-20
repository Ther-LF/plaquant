"""Round-13 ncu runner: ONE codesign-NVFP4 kernel instance under ncu.
Usage:
  prof_codesign.py build                 # build ext, exit (no ncu)
  prof_codesign.py cd M N K              # codesign NVFP4 GEMM (full K)
  prof_codesign.py cut M N K             # CUTLASS pure-NVFP4 endpoint (pure_presf)
5 warmup + 2 profiled launches; drive ncu with `-s 5 -c 1 -k regex:device_kernel`.
Both codesign and the CUTLASS endpoint launch a CUTLASS `device_kernel`; each
process launches exactly one, so the -s5 -c1 capture is unambiguous per label.
"""
import sys
import torch
import mixed_gemm as mg
import b1_fused as b1
import codesign_check as cc

kind = sys.argv[1]
if kind == "build":
    cc.get_ext(); b1.get_presf_ext()
    print("BUILD_OK"); sys.exit(0)

M, N, K = (int(x) for x in sys.argv[2:5])

if kind == "cd":
    ext = cc.get_ext()
    p4, _ = mg.prep_pure_fp4(M, N, K, seed=1)
    physB = ext.prepack_sf_fp4_b(p4["SFB"], N, K)
    fn = lambda: ext.codesign_nvfp4_gemm(p4["A_packed"], p4["SFA"], p4["B_packed"], physB, M, N, K)
elif kind == "cut":
    b1.get_presf_ext()
    p4, _ = b1.prep_bounds_presf(M, N, K, seed=1)
    fn = lambda: b1.pure_fp4_presf(p4, M, N, K)
else:
    print("bad kind", kind); sys.exit(1)

for _ in range(5):
    fn()
torch.cuda.synchronize()
for _ in range(2):
    fn()
torch.cuda.synchronize()
print("prof done", kind, M, N, K)
