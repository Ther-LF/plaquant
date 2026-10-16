"""ncu prof driver for the TMA phase-2 kernel. Usage: prof_tma.py <M> <N> <K> <stages>"""
import sys
import torch
import mixed_gemm as mg
import b1_tma as t

M, N, K, st = (int(x) for x in sys.argv[1:5])
t.get_tma()
p, _ = mg.prep_mixed(M, N, 0, K, seed=1)
fn = lambda: t.tma_mx(p, M, N, K, stages=st)
for _ in range(3):
    fn()
torch.cuda.synchronize()
for _ in range(3):
    fn()
torch.cuda.synchronize()
print("prof done tma", M, N, K, st)
