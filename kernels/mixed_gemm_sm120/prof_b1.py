"""ncu harness: run ONE kernel case a few times.
Usage: prof_b1.py <shape> <M> <stages> [path]
  path: b1 (default) | fp4p (pure NVFP4 full-K presf CUTLASS bound) | fp8p
"""
import sys, torch, mixed_gemm as mg, b1_fused as b1
shape, M, stages = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path = sys.argv[4] if len(sys.argv) > 4 else "b1"
all_shapes = {**mg.SHAPES_1B, **mg.SHAPES_3B}
N, Kl, Kh = all_shapes[shape]
Kt = Kl + Kh
if path == "b1":
    b1.get_ext()
    p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    fn = lambda: b1.b1_single(p, M, N, Kl, Kh, stages=stages)
else:
    b1.get_presf_ext()
    p4, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = (lambda: b1.pure_fp4_presf(p4, M, N, Kt)) if path == "fp4p" \
        else (lambda: b1.pure_fp8_presf(p8, M, N, Kt))
for _ in range(3):   # warmup
    fn()
torch.cuda.synchronize()
for _ in range(3):   # profiled region (ncu skips via --launch-skip)
    fn()
torch.cuda.synchronize()
print("prof done", shape, M, stages, path)
