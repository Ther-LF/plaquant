"""Round-8 GEMM head-to-head (vLLM side): time vLLM's NVFP4 linear GEMM on the
EXACT same per-layer shapes as the ours-side script. Runs in the baselines venv
(python3.12, vLLM 0.32, flashinfer 0.7). GEMM-only AND quant+GEMM, warm, median,
idle GPU. Faithful to FlashInferCutlassNvFp4LinearKernel.apply_weights (SM120
cutlass fp4 path): scaled_fp4_quant(activation) + flashinfer_scaled_fp4_mm(cutlass).
"""
import os, sys, json, argparse
import torch
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale, pad_nvfp4_weight_for_cutlass)

DEV = "cuda"
FP4_MAX = 6.0
E4M3_MAX = 448.0


def shapes_for(H, I, nheads, nkv):
    hd = H // nheads
    q = nheads * hd; kv = nkv * hd
    return [("qkv", q + 2 * kv, H), ("o_proj", H, q), ("gate_up", 2 * I, H), ("down", H, I)]


MODELS = {"3B": dict(H=3072, I=8192, nheads=24, nkv=8),
          "8B": dict(H=4096, I=14336, nheads=32, nkv=8)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3B,8B")
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]

    rows = []
    for mdl in args.models.split(","):
        c = MODELS[mdl]
        for name, N, K in shapes_for(c["H"], c["I"], c["nheads"], c["nkv"]):
            torch.manual_seed(0)
            W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.08
            w_gs = (E4M3_MAX * FP4_MAX) / W.abs().amax().float().clamp_min(1e-6)
            w_gs = w_gs.reshape(1)
            w_fp4, w_sf = scaled_fp4_quant(W, w_gs)
            w_sf_sw = swizzle_blockscale(w_sf)
            w_pad, _ = pad_nvfp4_weight_for_cutlass(w_fp4)
            for M in ms:
                x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.08
                x_gs = (E4M3_MAX * FP4_MAX) / x.abs().amax().float().clamp_min(1e-6)
                x_gs_inv = (1.0 / x_gs).reshape(1)
                alpha = (1.0 / (w_gs * x_gs)).float().reshape(1)
                # pre-quantize activation once (GEMM-only path)
                x_fp4, x_sf = scaled_fp4_quant(x, x_gs_inv, is_sf_swizzled_layout=True,
                                               backend="flashinfer-cutlass", padded_n=x.shape[-1])

                def gemm_only():
                    return flashinfer_scaled_fp4_mm(x_fp4, w_pad, x_sf, w_sf_sw, alpha,
                                                    torch.bfloat16, backend="cutlass")

                def quant_gemm():
                    xf, xs = scaled_fp4_quant(x, x_gs_inv, is_sf_swizzled_layout=True,
                                              backend="flashinfer-cutlass", padded_n=x.shape[-1])
                    return flashinfer_scaled_fp4_mm(xf, w_pad, xs, w_sf_sw, alpha,
                                                    torch.bfloat16, backend="cutlass")

                t_g, min_g = median_ms(gemm_only)
                t_qg, min_qg = median_ms(quant_gemm)
                # fp16 cublas reference on same shape
                Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
                Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
                t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))

                flops = 2.0 * M * N * K
                rec = dict(model=mdl, shape=name, N=N, K=K, M=M,
                           vllm_gemm_ms=t_g, vllm_gemm_min_ms=min_g,
                           vllm_quantgemm_ms=t_qg, vllm_quantgemm_min_ms=min_qg,
                           fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                           vllm_gemm_tflops=flops / (t_g * 1e-3) / 1e12,
                           vllm_quantgemm_tflops=flops / (t_qg * 1e-3) / 1e12,
                           fp16_tflops=flops / (t_fp16 * 1e-3) / 1e12,
                           vllm_gemm_vs_fp16=t_fp16 / t_g,
                           vllm_quantgemm_vs_fp16=t_fp16 / t_qg)
                rows.append(rec)
                print(f"{mdl} {name:8s} M={M:5d} N={N:6d} K={K:6d}: "
                      f"vLLM-gemm {t_g:.4f}ms ({rec['vllm_gemm_tflops']:.0f} TF) | "
                      f"vLLM-quant+gemm {t_qg:.4f}ms ({rec['vllm_quantgemm_tflops']:.0f} TF) | "
                      f"fp16 {t_fp16:.4f}ms | gemm/fp16 {rec['vllm_gemm_vs_fp16']:.2f}x "
                      f"q+g/fp16 {rec['vllm_quantgemm_vs_fp16']:.2f}x", flush=True)
                del x, x_fp4, x_sf, Xf, Wf
            del W, w_fp4, w_sf, w_sf_sw, w_pad
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nVLLM_DONE", flush=True)


if __name__ == "__main__":
    main()
