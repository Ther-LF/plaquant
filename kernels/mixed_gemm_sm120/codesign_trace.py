import os, torch, threading, time, sys
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

ext = load(name="codesign_fused_trace",
    sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
    extra_include_paths=[HERE, os.path.join(CUTLASS,"include"),
        os.path.join(CUTLASS,"tools/util/include"), os.path.join(CUTLASS,"examples/common")],
    extra_cuda_cflags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3","-DCD_TRACE"],
    verbose=True)
print("BUILD_OK", flush=True)

def wd():
    time.sleep(45); print("WATCHDOG_TIMEOUT: fused kernel hung", flush=True); os._exit(7)

N, Klo, Khi = 2048, 1792, 256
M = 2048
p, oracle = mg.prep_mixed(M, N, Klo, Khi, seed=M)
print("prep_mixed done", flush=True)
D_low = nv.get_ext().nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"], M, N, Klo).float()
torch.cuda.synchronize(); print("D_low done", flush=True)
D_high = mx.get_ext().mxfp8_gemm(p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"], M, N, Khi).float()
torch.cuda.synchronize(); print("D_high done", flush=True)
core_ref = D_low + D_high
physB_lo = ext.prepack_sf_fp4_b(p["SFBl"], N, Klo)
torch.cuda.synchronize(); print("physB_lo done", flush=True)
physB_hi = ext.prepack_sf_fp8_b(p["SFBh"], N, Khi)
torch.cuda.synchronize(); print("physB_hi done", flush=True)
print("calling fused_gemm...", flush=True)
threading.Thread(target=wd, daemon=True).start()
D = ext.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                   p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi, M, N, Klo, Khi).float()
torch.cuda.synchronize()
c_core = b1.cosine(D, core_ref)
print(f"fused qkv M{M}: cos_core={c_core:.6f}", flush=True)
print("TRACE_DONE", flush=True)
