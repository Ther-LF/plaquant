"""rd-bench: vLLM 0.32 NVFP4 W4A4 GEMM (FlashInferCutlassNvFp4LinearKernel path)
at M=8192 on the per-projection 3B/8B shapes. Run with the baselines venv python
inside container `luofan-plaquant`.
"""
import json, argparse
import torch
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale, pad_nvfp4_weight_for_cutlass)

DEV = "cuda"
FP4_MAX = 6.0
E4M3_MAX = 448.0

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

    rows = []
    for mdl, name, N, K in SHAPES:
        torch.manual_seed(0)
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.08
        w_gs = (E4M3_MAX * FP4_MAX) / W.abs().amax().float().clamp_min(1e-6)
        w_gs = w_gs.reshape(1)
        # NB: weight SF must be produced in LINEAR layout then swizzled ONCE
        # (modelopt checkpoints store linear SF; scaled_fp4_quant defaults to
        # swizzled). The old round-8 harness double-swizzled here.
        w_fp4, w_sf = scaled_fp4_quant(W, w_gs, is_sf_swizzled_layout=False)
        w_sf_sw = swizzle_blockscale(w_sf)
        w_pad, _ = pad_nvfp4_weight_for_cutlass(w_fp4)

        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.08
        x_gs = ((E4M3_MAX * FP4_MAX) / x.abs().amax().float().clamp_min(1e-6)).reshape(1)
        alpha = (1.0 / (w_gs * x_gs)).float().reshape(1)
        x_fp4, x_sf = scaled_fp4_quant(x, x_gs, is_sf_swizzled_layout=True,
                                       backend="flashinfer-cutlass", padded_n=x.shape[-1])

        def gemm_only():
            return flashinfer_scaled_fp4_mm(x_fp4, w_pad, x_sf, w_sf_sw, alpha,
                                            torch.bfloat16, backend="cutlass")
        y = gemm_only()
        cos = cosine(y, x.float() @ W.float().t())
        t_g, min_g = median_ms(gemm_only)

        def quant_gemm():
            xf, xs = scaled_fp4_quant(x, x_gs, is_sf_swizzled_layout=True,
                                      backend="flashinfer-cutlass", padded_n=x.shape[-1])
            return flashinfer_scaled_fp4_mm(xf, w_pad, xs, w_sf_sw, alpha,
                                            torch.bfloat16, backend="cutlass")
        t_qg, min_qg = median_ms(quant_gemm)

        Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
        Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
        t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))

        flops = 2.0 * M * N * K
        rec = dict(method="vllm_nvfp4", model=mdl, shape=name, M=M, N=N, K=K,
                   cosine_vs_fp32=cos,
                   gemm_ms=t_g, gemm_min_ms=min_g,
                   quantgemm_ms=t_qg, quantgemm_min_ms=min_qg,
                   fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                   gemm_tflops=flops / (t_g * 1e-3) / 1e12,
                   fp16_tflops=flops / (t_fp16 * 1e-3) / 1e12,
                   gemm_vs_fp16=t_fp16 / t_g)
        rows.append(rec)
        print(f"{mdl} {name:8s} M={M} N={N:6d} K={K:6d}: vLLM gemm {t_g:.4f}ms "
              f"({rec['gemm_tflops']:.0f} TF, cos={cos:.6f}) | q+g {t_qg:.4f}ms | "
              f"fp16 {t_fp16:.4f}ms | gemm/fp16 {rec['gemm_vs_fp16']:.2f}x", flush=True)
        del W, w_fp4, w_sf, w_sf_sw, w_pad, x, x_fp4, x_sf, Xf, Wf, y
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nVLLM_DONE", flush=True)


if __name__ == "__main__":
    main()
