"""rd-bench: ProMix fused NVFP4+MXFP8 GEMM at M=8192 on the per-projection
(unmerged) 3B/8B Linear shapes, vs our pure-NVFP4 single-format GEMM and fp16
cuBLAS. GEMM-only, cudaEvent median, warm, idle GPU.

Shapes (N, K): 3B qkvo 3072x3072, gate_up 8192x3072, down 3072x8192;
               8B qkvo 4096x4096, gate_up 14336x4096, down 4096x14336.
Run inside container `luofan-plaquant` with PYTHONPATH=<bench clone>.
"""
import os, sys, json, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.eval.full_network_realkernel import load_kernels, HIGH_FRACTION
from promix.quantize.weight_pack import split_k
from promix.quantize.mixed_linear import MixedQuantLinear
import nvfp4_gemm as nvmod

DEV = "cuda"

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]


def median_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort(); n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M = args.m

    nv, mx, cd, qpf = load_kernels()
    nvext = nvmod.get_ext()
    print("kernels loaded", flush=True)

    rows = []
    for mdl, name, N, K in SHAPES:
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(0)
        W = (torch.randn(N, K, device=DEV) * 0.08)
        lin = type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})()
        mql = MixedQuantLinear.from_linear(lin, nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)

        x = torch.randn(M, K, device=DEV)
        x[:, K_low:] *= 3.0
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

        def fused():
            return cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi,
                M, N, K_low, K_high)
        y = fused()
        ref = x.float() @ W.float().t()
        cos = cosine(y, ref)
        t_fused, min_fused = median_ms(fused)

        # pure NVFP4 over full K (r=0 structural reference)
        Wn = torch.randn(N, K, device=DEV)
        Bc, SFB, _ = nvmod.quantize_nvfp4(Wn)
        Bp = nvmod.pack_fp4(Bc)
        Xn = torch.randn(M, K, device=DEV)
        Ac, SFA, _ = nvmod.quantize_nvfp4(Xn)
        Ap = nvmod.pack_fp4(Ac)

        def purenv():
            return nvext.nvfp4_gemm(Ap, SFA, Bp, SFB, M, N, K)
        t_nv, min_nv = median_ms(purenv)

        Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
        Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
        t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))

        flops = 2.0 * M * N * K
        rec = dict(method="promix_fused", model=mdl, shape=name, M=M, N=N, K=K,
                   K_low=K_low, K_high=K_high, cosine_vs_fp32=cos,
                   gemm_ms=t_fused, gemm_min_ms=min_fused,
                   pure_nvfp4_ms=t_nv, pure_nvfp4_min_ms=min_nv,
                   fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                   gemm_tflops=flops / (t_fused * 1e-3) / 1e12,
                   pure_nvfp4_tflops=flops / (t_nv * 1e-3) / 1e12,
                   fp16_tflops=flops / (t_fp16 * 1e-3) / 1e12,
                   fused_vs_fp16=t_fp16 / t_fused)
        rows.append(rec)
        print(f"{mdl} {name:8s} M={M} N={N:6d} K={K:6d}: fused {t_fused:.4f}ms "
              f"({rec['gemm_tflops']:.0f} TF, cos={cos:.6f}) | pureNV {t_nv:.4f}ms | "
              f"fp16 {t_fp16:.4f}ms | fused/fp16 {rec['fused_vs_fp16']:.2f}x", flush=True)
        del mql, W, Wn, Bc, SFB, Bp, Xn, Ac, SFA, Ap, Xf, Wf, x, y, ref
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nOURS_DONE", flush=True)


if __name__ == "__main__":
    main()
