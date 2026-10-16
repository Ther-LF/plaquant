"""ncu harness (round-3): ONE case with explicit dims, run a few times.
Usage: prof_t1.py <path> <M> <N> <K_low> <K_high>
  path: b1 (auto cfg) | b1cfg<cfg>s<st> (e.g. b1cfg10s3) | fp4p | fp8p
For fp4p/fp8p the pure bound runs at K_total = K_low + K_high.
"""
import sys
import torch
import mixed_gemm as mg
import b1_fused as b1

path = sys.argv[1]
M, N, Kl, Kh = (int(x) for x in sys.argv[2:6])
Kt = Kl + Kh
if path.startswith("b1"):
    b1.get_ext()
    cfg, st = 0, 0
    if path != "b1":
        body = path[len("b1cfg"):]
        cfg, st = (int(x) for x in body.split("s"))
    p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    fn = lambda: b1.b1_single(p, M, N, Kl, Kh, stages=st, cfg=cfg)
else:
    b1.get_presf_ext()
    p4, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = (lambda: b1.pure_fp4_presf(p4, M, N, Kt)) if path == "fp4p" \
        else (lambda: b1.pure_fp8_presf(p8, M, N, Kt))
for _ in range(3):   # warmup (ncu skips via --launch-skip / -k filter)
    fn()
torch.cuda.synchronize()
for _ in range(3):   # profiled region
    fn()
torch.cuda.synchronize()
print("prof done", path, M, N, Kl, Kh)
