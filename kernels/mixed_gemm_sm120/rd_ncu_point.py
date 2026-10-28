"""rd_ncu_point.py — single-launch codesign GEMM for ncu profiling.
Usage: python rd_ncu_point.py <N> <K_low> <K_high> <M>
Does warmups then ONE final profiled M_CD.run launch (profile the last kernel).
"""
import sys, os, torch
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv, mxfp8_gemm as mx

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
INC = [HERE, CUTLASS + "/include", CUTLASS + "/tools/util/include", CUTLASS + "/examples/common"]
BASE = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]

N, K_low, K_high, M = (int(x) for x in sys.argv[1:5])
K_total = K_low + K_high
M_CD = load(name="cfgdisp_codesign", sources=[os.path.join(HERE, "config_dispatch_test.cu")],
            extra_include_paths=INC, extra_cuda_cflags=BASE + ["-DDISPATCH_ENABLE_CODESIGN"], verbose=False)
torch.manual_seed(1); dev = "cuda"
Xl = torch.randn(M, K_low, device=dev); Wl = torch.randn(N, K_low, device=dev)
Alc, SFAl, _ = nv.quantize_nvfp4(Xl); Blc, SFBl, _ = nv.quantize_nvfp4(Wl)
Al = nv.pack_fp4(Alc); Bl = nv.pack_fp4(Blc)
Xh = torch.randn(M, K_high, device=dev); Wh = torch.randn(N, K_high, device=dev)
Ah, SFAh, _ = mx.quantize_mxfp8(Xh); Bh, SFBh, _ = mx.quantize_mxfp8(Wh)
args = (Al, SFAl, Bl, SFBl, Ah, SFAh, Bh, SFBh, M, N, K_low, K_high)
for _ in range(5):
    M_CD.run(*args)
torch.cuda.synchronize()
M_CD.run(*args)          # <-- profiled launch
torch.cuda.synchronize()
print(f"done N={N} Klow={K_low} Khigh={K_high} M={M}")
