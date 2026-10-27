"""Round-8 GEMM head-to-head (OUR side): shipped fused dual-format NVFP4+MXFP8
GEMM vs our pure-NVFP4 single-format GEMM (r=0 equivalent) vs fp16 cuBLAS, on the
EXACT per-layer shapes of Llama-3.2-3B and Llama-3-8B. GEMM-only, warm, median,
idle GPU. Writes JSON for merge with the vLLM-side script.

Shapes use vLLM's actual GEMM decomposition (merged qkv, merged gate_up, o_proj,
down_proj) so the three kernels time IDENTICAL (M,N,K).
"""
import os, sys, json, argparse, time
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

# (name, N, K) — vLLM GEMM decomposition (qkv/gate_up merged).
def shapes_for(H, I, nheads, nkv):
    hd = H // nheads
    q = nheads * hd; kv = nkv * hd
    return [
        ("qkv",     q + 2 * kv, H),
        ("o_proj",  H,          q),
        ("gate_up", 2 * I,      H),
        ("down",    H,          I),
    ]

MODELS = {
    "3B": dict(H=3072, I=8192, nheads=24, nkv=8),
    "8B": dict(H=4096, I=14336, nheads=32, nkv=8),
}


def median_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort()
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3B,8B")
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]

    nv, mx, cd, qpf = load_kernels()
    nvext = nvmod.get_ext()
    print("kernels loaded", flush=True)

    rows = []
    for mdl in args.models.split(","):
        c = MODELS[mdl]
        for name, N, K in shapes_for(c["H"], c["I"], c["nheads"], c["nkv"]):
            K_low, K_high = split_k(K, HIGH_FRACTION)
            # build our packed weight (random, perf-only)
            torch.manual_seed(0)
            W = (torch.randn(N, K, device=DEV) * 0.08)
            lin = type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})()
            mql = MixedQuantLinear.from_linear(lin, nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)
            # pure NVFP4 full-K weight
            Wn = torch.randn(N, K, device=DEV)
            Bc, SFB, _ = nvmod.quantize_nvfp4(Wn)
            Bp = nvmod.pack_fp4(Bc)
            for M in ms:
                x = torch.randn(M, K, device=DEV)
                x[:, K_low:] *= 3.0
                A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

                def fused():
                    return cd.fused_gemm_prepacked(
                        A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                        A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi,
                        M, N, K_low, K_high)
                t_fused, min_fused = median_ms(fused)

                # pure NVFP4 over full K (r=0 structural reference)
                Xn = torch.randn(M, K, device=DEV)
                Ac, SFA, _ = nvmod.quantize_nvfp4(Xn)
                Ap = nvmod.pack_fp4(Ac)

                def purenv():
                    return nvext.nvfp4_gemm(Ap, SFA, Bp, SFB, M, N, K)
                t_nv, min_nv = median_ms(purenv)

                # fp16 cuBLAS
                Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
                Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)

                def fp16():
                    return torch.matmul(Xf, Wf.t())
                t_fp16, min_fp16 = median_ms(fp16)

                flops = 2.0 * M * N * K
                rec = dict(model=mdl, shape=name, N=N, K=K, K_low=K_low, K_high=K_high, M=M,
                           fused_ms=t_fused, fused_min_ms=min_fused,
                           pure_nvfp4_ms=t_nv, pure_nvfp4_min_ms=min_nv,
                           fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                           fused_tflops=flops / (t_fused * 1e-3) / 1e12,
                           pure_nvfp4_tflops=flops / (t_nv * 1e-3) / 1e12,
                           fp16_tflops=flops / (t_fp16 * 1e-3) / 1e12,
                           fused_vs_fp16=t_fp16 / t_fused,
                           pure_nvfp4_vs_fp16=t_fp16 / t_nv,
                           fused_vs_purenv=t_nv / t_fused)
                rows.append(rec)
                print(f"{mdl} {name:8s} M={M:5d} N={N:6d} K={K:6d}: "
                      f"fused {t_fused:.4f}ms ({rec['fused_tflops']:.0f} TF) | "
                      f"pureNV {t_nv:.4f}ms ({rec['pure_nvfp4_tflops']:.0f} TF) | "
                      f"fp16 {t_fp16:.4f}ms ({rec['fp16_tflops']:.0f} TF) | "
                      f"fused/fp16 {rec['fused_vs_fp16']:.2f}x pureNV/fp16 {rec['pure_nvfp4_vs_fp16']:.2f}x "
                      f"pureNV/fused {rec['fused_vs_purenv']:.2f}x", flush=True)
                del x, Xn, Ac, SFA, Ap, Xf, Wf
            del mql, W, Wn, Bc, SFB, Bp
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nOURS_DONE", flush=True)


if __name__ == "__main__":
    main()
