"""Round-23 ncu runner: one kernel instance under ncu for TC-duty + registers + occupancy.

Usage:
  prof_optc.py build
  prof_optc.py <kind> M N Kl Kh
    kind: base  = codesign serial shared-accum fused_gemm_prepacked (co-design GEMM)
          optc  = codesign two-accumulator fused_gemm_prepacked_optc (register probe)
          cd    = alias of base
          ship  = shipping b1_tma dual GEMM (per-shape cfg via SH env tile/sw/pers)
          fp4   = pure-NVFP4 presf bound (Kt=Kl+Kh)
          fp8   = pure-MXFP8 presf bound
10 warmup + 6 profiled; drive ncu with -s10 -c1 -k regex:device_kernel (co-design/pure)
or -k regex:b1_tma for ship.
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
import b1_tma as bt


def get_optc():
    return load(name="codesign_optc_ext",
        sources=[os.path.join(HERE, "codesign_optc_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"], verbose=False)


kind = sys.argv[1]
oc = get_optc()
if kind == "build":
    bt.get_tma(); b1.get_presf_ext(); b1.get_ext(); nv.get_ext(); mx.get_ext()
    print("BUILD_OK"); sys.exit(0)

M, N, Kl, Kh = (int(x) for x in sys.argv[2:6])
Kt = Kl + Kh
torch.manual_seed(1); dev = "cuda"


def prep_cd():
    Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
    Al_codes, SFAl, _ = nv.quantize_nvfp4(Xl); Bl_codes, SFBl, _ = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
    Ah_bytes, SFAh, _ = mx.quantize_mxfp8(Xh); Bh_bytes, SFBh, _ = mx.quantize_mxfp8(Wh)
    pBl = oc.prepack_sf_fp4_b(SFBl, N, Kl); pBh = oc.prepack_sf_fp8_b(SFBh, N, Kh)
    o4 = oc.quant_pack_fp4(Al_codes, SFAl, M, N, Kl, True)
    o8 = oc.quant_pack_fp8(Ah_bytes, SFAh, M, N, Kh, True)
    return o4, o8, Bl_packed, Bh_bytes, pBl, pBh


if kind in ("base", "cd", "optc"):
    o4, o8, Bl, Bh, pBl, pBh = prep_cd()
    if kind == "optc":
        fn = lambda: oc.fused_gemm_prepacked_optc(o4[0], o4[1], Bl, pBl, o8[0], o8[1], Bh, pBh, M, N, Kl, Kh)
    else:
        fn = lambda: oc.fused_gemm_prepacked(o4[0], o4[1], Bl, pBl, o8[0], o8[1], Bh, pBh, M, N, Kl, Kh)
elif kind == "ship":
    tile = int(os.environ.get("SH_TILE", "0")); sw = int(os.environ.get("SH_SW", "0")); pers = int(os.environ.get("SH_PERS", "0"))
    tma = bt.get_tma()
    p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
    kp = 1
    fn = lambda: tma.b1_fused_tma_dual(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
                                       p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
                                       p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0)
elif kind == "fp4":
    b1.get_presf_ext(); p4, _ = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = lambda: b1.pure_fp4_presf(p4, M, N, Kt)
elif kind == "fp8":
    b1.get_presf_ext(); _, p8 = b1.prep_bounds_presf(M, N, Kt, seed=1)
    fn = lambda: b1.pure_fp8_presf(p8, M, N, Kt)
else:
    print("bad kind", kind); sys.exit(1)

for _ in range(10):
    fn()
torch.cuda.synchronize()
for _ in range(6):
    fn()
torch.cuda.synchronize()
print("prof done", kind, M, N, Kl, Kh)
