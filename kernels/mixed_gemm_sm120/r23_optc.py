"""Round-23 Part B: Option-C register-feasibility probe.

Builds codesign_optc_ext (two-accumulator cooperative fork) with ptxas -v, prints
the smem/thread report, GATES cosine (two-accumulator sum == shared-accumulator,
bit-exact), and times baseline (shared accum) vs Option-C (2x accum) GEMM-only.
This measures the REGISTER cost of holding two full 128x128 FP32 partials
(64 floats/thread EACH -> 128/thread) that round-20 inferred would blow the
64K-reg/SM budget. ptxas regs + launch success + latency delta = the answer.

NO change to PROTECTED files or the shipping/working codesign kernel.
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


def get_optc():
    return load(name="codesign_optc_ext",
        sources=[os.path.join(HERE, "codesign_optc_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3", "-Xptxas", "-v"],
        verbose=True)


oc = get_optc()
print("OPTC_BUILD_OK", flush=True)
print(oc.compat_report(), flush=True)


def prep(M, N, Kl, Kh, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
    Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
    Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
    Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
    s_x = (0.5 + torch.rand(M, device=dev)).float()
    s_w = (0.5 + torch.rand(N, device=dev)).float()
    oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t()) * s_x.view(M, 1) * s_w.view(1, N)
    physB_lo = oc.prepack_sf_fp4_b(SFBl, N, Kl)
    physB_hi = oc.prepack_sf_fp8_b(SFBh, N, Kh)
    o4i = oc.quant_pack_fp4(Al_codes, SFAl, M, N, Kl, True)
    o8i = oc.quant_pack_fp8(Ah_bytes, SFAh, M, N, Kh, True)
    return dict(o4=o4i, o8=o8i, Bl=Bl_packed, Bh=Bh_bytes, pBl=physB_lo, pBh=physB_hi,
                s_x=s_x, s_w=s_w), oracle


CASES = [("down", 2048, 8192, 7168, 1024),   # down r=1/8
         ("gate", 8192, 2048, 1024, 1024),    # gate r=1/2 (round-20 bulge cell)
         ("qkv", 2048, 2048, 1792, 256)]      # qkv r=1/8


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=15)
    return m * 1000.0


print(f"{'case':6s}{'M':>6s}{'Kl':>6s}{'Kh':>6s}{'cos_base':>10s}{'cos_optc':>10s}"
      f"{'base_us':>9s}{'optc_us':>9s}{'optc/base':>10s}", flush=True)
allok = True
for name, N, M, Kl, Kh in [(c[0], c[2], c[1], c[3], c[4]) for c in CASES]:
    try:
        p, oracle = prep(M, N, Kl, Kh)
        base = lambda: oc.fused_gemm_prepacked(p["o4"][0], p["o4"][1], p["Bl"], p["pBl"],
                                               p["o8"][0], p["o8"][1], p["Bh"], p["pBh"], M, N, Kl, Kh)
        optc = lambda: oc.fused_gemm_prepacked_optc(p["o4"][0], p["o4"][1], p["Bl"], p["pBl"],
                                                    p["o8"][0], p["o8"][1], p["Bh"], p["pBh"], M, N, Kl, Kh)
        Db = base().float() * p["s_x"].view(M, 1) * p["s_w"].view(1, N)
        Do = optc().float() * p["s_x"].view(M, 1) * p["s_w"].view(1, N)
        cb = b1.cosine(Db, oracle); co = b1.cosine(Do, oracle)
        if min(cb, co) < 0.999:
            allok = False
        tb = med(base); to = med(optc)
        print(f"{name:6s}{M:6d}{Kl:6d}{Kh:6d}{cb:10.6f}{co:10.6f}{tb:9.1f}{to:9.1f}{to/tb:10.3f}", flush=True)
        del p, oracle, Db, Do; torch.cuda.empty_cache()
    except Exception as e:
        allok = False
        print(f"{name:6s}{M:6d}  ERR {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)
print("R23_OPTC_COSINE_PASS" if allok else "R23_OPTC_COSINE_OR_RUN_FAIL", flush=True)
print("R23_OPTC_DONE", flush=True)
