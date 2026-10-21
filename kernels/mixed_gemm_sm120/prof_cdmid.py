"""Round-20 ncu root-cause runner: ONE kernel instance under ncu, at a MID-r cell.

Confirms (or refutes) round-19's "mid-r structural phase-seam" hypothesis for the
serial dual-format co-design GEMM (codesign_fused_ext.fused_gemm_prepacked) BEFORE
any mainloop redesign. Drives the GEMM-only path (activation SF pre-interleaved, no
pack in the timed region) so ncu sees exactly the fused mainloop + epilogue.

Usage:
  prof_cdmid.py build                 # build/prime exts, exit (no ncu)
  prof_cdmid.py cddual M N Klo Khi    # serial dual-format fused GEMM (one launch)
  prof_cdmid.py fp4    M N K          # CUTLASS pure-NVFP4 endpoint (presf)
  prof_cdmid.py fp8    M N K          # CUTLASS pure-MXFP8 endpoint (presf)

5 warmup + 2 profiled launches; drive ncu with `-s 5 -c 1 -k regex:device_kernel`.
All three launch exactly one CUTLASS `device_kernel` in the timed loop (prep kernels
run once, before the loop, and are not device_kernel), so the capture is unambiguous.
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1


def get_cd():
    return load(name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=True)


kind = sys.argv[1]
if kind == "build":
    get_cd(); b1.get_presf_ext()
    print("BUILD_OK"); sys.exit(0)

if kind == "cddual":
    M, N, Klo, Khi = (int(x) for x in sys.argv[2:6])
    cd = get_cd()
    torch.manual_seed(1); dev = "cuda"
    Xl = torch.randn(M, Klo, device=dev); Wl = torch.randn(N, Klo, device=dev)
    Al_codes, SFAl, _ = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, _ = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    Xh = torch.randn(M, Khi, device=dev); Wh = torch.randn(N, Khi, device=dev)
    Ah_bytes, SFAh, _ = mx.quantize_mxfp8(Xh)
    Bh_bytes, SFBh, _ = mx.quantize_mxfp8(Wh)
    physB_lo = cd.prepack_sf_fp4_b(SFBl, N, Klo)
    physB_hi = cd.prepack_sf_fp8_b(SFBh, N, Khi)
    o4i = cd.quant_pack_fp4(Al_codes, SFAl, M, N, Klo, True)
    o8i = cd.quant_pack_fp8(Ah_bytes, SFAh, M, N, Khi, True)
    fn = lambda: cd.fused_gemm_prepacked(o4i[0], o4i[1], Bl_packed, physB_lo,
                                         o8i[0], o8i[1], Bh_bytes, physB_hi,
                                         M, N, Klo, Khi)
elif kind == "fp4":
    M, N, K = (int(x) for x in sys.argv[2:5])
    b1.get_presf_ext()
    p4, _ = b1.prep_bounds_presf(M, N, K, seed=1)
    fn = lambda: b1.pure_fp4_presf(p4, M, N, K)
elif kind == "fp8":
    M, N, K = (int(x) for x in sys.argv[2:5])
    b1.get_presf_ext()
    _, p8 = b1.prep_bounds_presf(M, N, K, seed=1)
    fn = lambda: b1.pure_fp8_presf(p8, M, N, K)
else:
    print("bad kind", kind); sys.exit(1)

for _ in range(5):
    fn()
torch.cuda.synchronize()
for _ in range(2):
    fn()
torch.cuda.synchronize()
print("prof done", kind, sys.argv[2:])
