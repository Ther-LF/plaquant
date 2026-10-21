"""Round-15 ncu runner: ONE dual-format co-design fused GEMM instance under ncu.

Usage:
  prof_r15dual.py build                  # build ext, exit (no ncu)
  prof_r15dual.py cd M N Kl Kh           # dual-format fused_gemm (NVFP4 Kl + MXFP8 Kh)

10 warmup + 8 profiled launches; drive ncu with `-s 5 -c 1 -k regex:device_kernel`.
fused_gemm launches a per-call SFA scatter kernel (cd_fill_sf*, NOT matched by the
device_kernel regex) plus exactly one CUTLASS `device_kernel` (the fused GEMM), so
the -s5 -c1 capture lands unambiguously on the fused mainloop+epilogue kernel.
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg


def get_cd():
    return load(
        name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=True)


kind = sys.argv[1]
cd = get_cd()
if kind == "build":
    print("BUILD_OK")
    sys.exit(0)

M, N, Kl, Kh = (int(x) for x in sys.argv[2:6])
p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
fn = lambda: cd.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                           p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Kl, Kh)
for _ in range(10):
    fn()
torch.cuda.synchronize()
for _ in range(8):
    fn()
torch.cuda.synchronize()
print("prof done", kind, M, N, Kl, Kh)
