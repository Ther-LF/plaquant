"""Lean min-of-N e2e latency for the single-layer (7 Linears), robust to shared-GPU
contention. Measures pack_fast / GEMM-only / e2e_fast / FP16 with CUDA events, min over
N iters (min ~ least-contended iteration ~ clean). Torch-pack timed briefly for 'before'."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv, mxfp8_gemm as mx
nv.get_ext(); mx.get_ext()
inc = [HERE, os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools/util/include"), os.path.join(CUTLASS, "examples/common")]
flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
cd = load(name="codesign_fused_ext", sources=[os.path.join(HERE, "codesign_fused_ext.cu")], extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
qpf = load(name="quant_pack_float_ext", sources=[os.path.join(HERE, "quant_pack_float_ext.cu")], extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.quantize.weight_pack import split_k

LAYER = [("q",2048,2048),("k",512,2048),("v",512,2048),("o",2048,2048),("gate",8192,2048),("up",8192,2048),("down",2048,8192)]
HF = 0.125
dev = "cuda"
DTYPE = torch.float16 if (len(sys.argv) > 1 and sys.argv[1] == "fp16") else torch.float32

def tmin(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); best = 1e9
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
    return best

print(f"dtype={DTYPE}", flush=True)
for M in (2048, 4096, 8192):
    tp_fast = tg = te = tf = tp_slow = 0.0
    for nm, N, K in LAYER:
        Kl, Kh = split_k(K, HF)
        torch.manual_seed(M + hash(nm) % 1000)
        W = torch.randn(N, K, device=dev) * 0.08
        x = torch.randn(M, K, device=dev, dtype=DTYPE); x[:, Kl:] *= 3.0
        mql_f = MixedQuantLinear.from_linear(type("L",(),{"weight":type("T",(),{"data":W})(),"bias":None})(), nv, mx, cd, high_fraction=HF, qpf=qpf)
        mql_s = MixedQuantLinear.from_linear(type("L",(),{"weight":type("T",(),{"data":W})(),"bias":None})(), nv, mx, cd, high_fraction=HF, qpf=None)
        Wh = W.half(); xh = x.half()
        tp_fast += tmin(lambda: mql_f.pack_only(x))
        A_lo,pS_lo,A_hi,pS_hi = mql_f.pack_only(x)
        tg += tmin(lambda: cd.fused_gemm_prepacked(A_lo,pS_lo,mql_f.W_lo_packed,mql_f.physSFB_lo,A_hi,pS_hi,mql_f.W_hi_bytes,mql_f.physSFB_hi,M,N,Kl,Kh))
        te += tmin(lambda: mql_f(x))
        tf += tmin(lambda: xh @ Wh.t())
        tp_slow += tmin(lambda: mql_s.pack_only(x), iters=10, warmup=3)
    print(f"M={M:5d}: pack torch={tp_slow:6.3f}ms -> fused={tp_fast:5.3f}ms ({tp_slow/tp_fast:4.0f}x) | "
          f"GEMM={tg:5.3f}ms ({tf/tg:.2f}x FP16) | e2e={te:5.3f}ms vs FP16={tf:5.3f}ms = {tf/te:.2f}x "
          f"| pack/GEMM={tp_fast/tg:.2f}", flush=True)
print("QPF_E2E_DONE", flush=True)
