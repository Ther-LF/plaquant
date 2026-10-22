"""Micro-bench + ncu target for quant_pack_float kernels. CUDA-event min-of-N latency
and a single-launch target for ncu BW profiling. Run on a FREE GPU."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv, mxfp8_gemm as mx
nv.get_ext(); mx.get_ext()
inc = [HERE, os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools/util/include"), os.path.join(CUTLASS, "examples/common")]
flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
qpf = load(name="quant_pack_float_ext", sources=[os.path.join(HERE, "quant_pack_float_ext.cu")],
           extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)

dev = "cuda"
MODE = sys.argv[1] if len(sys.argv) > 1 else "lat"

def tmin(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    best = 1e9
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best

# representative shapes (M, N, K_low, K_high) for down (big low) and qkv (small)
cases = [("down8192", 8192, 2048, 7168, 1024), ("qkv8192", 8192, 2048, 1792, 256),
         ("down2048", 2048, 2048, 7168, 1024)]
if MODE == "ncu":
    cases = [("down8192", 8192, 2048, 7168, 1024)]

for dt in (torch.float32, torch.float16):
    for nm, M, N, Kl, Kh in cases:
        xl = torch.randn(M, Kl, device=dev, dtype=dt)
        xh = torch.randn(M, Kh, device=dev, dtype=dt) * 3.0
        if MODE == "ncu":
            qpf.quant_pack_float_fp4(xl, M, N, Kl)
            qpf.quant_pack_float_fp8(xh, M, N, Kh)
            torch.cuda.synchronize()
            continue
        t4 = tmin(lambda: qpf.quant_pack_float_fp4(xl, M, N, Kl))
        t8 = tmin(lambda: qpf.quant_pack_float_fp8(xh, M, N, Kh))
        # bytes moved: fp4 read M*Kl*elsize + write M*Kl/2 ; fp8 read M*Kh*elsize + write M*Kh
        el = 4 if dt == torch.float32 else 2
        bw4 = (M*Kl*el + M*Kl//2) / (t4*1e-3) / 1e12
        bw8 = (M*Kh*el + M*Kh) / (t8*1e-3) / 1e12
        print(f"{str(dt):17s} {nm:9s} fp4={t4*1e3:7.1f}us ({bw4:.2f}TB/s)  fp8={t8*1e3:7.1f}us ({bw8:.2f}TB/s)", flush=True)
print("QPF_LAT_DONE", flush=True)
