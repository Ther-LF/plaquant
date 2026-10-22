"""Decode-GEMV prototype: validate + benchmark the memory-bound W4A4 decode GEMV
(decode_gemv_ext.cu) against the FP32 twin reference, the real fused kernel, and
FP16 cuBLAS GEMV, across the 7 Llama-3.2-1B Linear shapes at M=1.

Builds a decode-friendly pack (row-major 2-codes/byte NVFP4 + FP32 per-16 scale;
FP8 bytes + FP32 per-32 scale) from the SAME spec quantizers the fused kernel /
fake-twin use, so the dequantized weight is bit-identical -> the only thing being
tested is that the streaming GEMV reproduces it while reading ~3.5x fewer bytes.

Usage: python -m decode_gemv   (run inside the container, on a free GPU)
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
import json
import torch
from torch.utils.cpp_extension import load
from promix.utils import DEV
from promix.quantize.weight_pack import quantize_nvfp4_spec, quantize_mxfp8_spec, split_k
from promix.quantize.quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.eval.full_network_realkernel import load_kernels, LAYER_SHAPES, HIGH_FRACTION

HIGH = HIGH_FRACTION


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def time_min(fn_, iters=200, warmup=30):
    torch.cuda.synchronize()
    for _ in range(warmup): fn_()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter(); fn_(); torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def build_decode_pack(W, K_low, K_high):
    N, K = W.shape
    codes_lo, SFB_lo, deq_lo = quantize_nvfp4_spec(W[:, :K_low].contiguous())
    bytes_hi, SFB_hi, deq_hi = quantize_mxfp8_spec(W[:, K_low:].contiguous())
    # 2 NVFP4 codes/byte: even k -> low nibble, odd k -> high nibble.
    cl = codes_lo.view(N, K_low // 2, 2)
    Wlo = (cl[:, :, 0] | (cl[:, :, 1] << 4)).to(torch.uint8).contiguous()
    Slo = SFB_lo.view(torch.float8_e4m3fn).float().contiguous()       # (N, K_low/16)
    Whi = bytes_hi.contiguous()                                       # (N, K_high)
    Shi = torch.exp2(SFB_hi.float() - 127.0).contiguous()             # (N, K_high/32)
    W_deq = torch.cat([deq_lo, deq_hi], dim=-1).contiguous()
    packed_bytes = Wlo.numel() + Whi.numel() + Slo.numel() * 4 + Shi.numel() * 4
    return Wlo, Slo, Whi, Shi, W_deq, packed_bytes


def main():
    nv, mx, cd, qpf = load_kernels()
    flags = ["-std=c++17", "-O3"]
    dg = load(name="decode_gemv_ext", sources=[os.path.join(KERN, "decode_gemv_ext.cu")],
              extra_cuda_cflags=flags, verbose=False)

    # empirical HBM ceiling (big D2D copy)
    xb = torch.empty(1 << 28, device=DEV, dtype=torch.float16); yb = torch.empty_like(xb)
    t_cp = time_min(lambda: yb.copy_(xb), iters=50, warmup=10)
    hbm = xb.numel() * 2 * 2 / (t_cp / 1e3) / 1e12
    del xb, yb; torch.cuda.empty_cache()
    print(f"HBM ceiling (copy) = {hbm:.2f} TB/s", flush=True)

    rows = []
    sum_gemv = sum_fused = sum_fp16 = 0.0
    worst_cos_twin = worst_cos_fused = 1.0
    for name, N, K in LAYER_SHAPES:
        K_low, K_high = split_k(K, HIGH)
        torch.manual_seed(hash(name) % 1000)
        W = torch.randn(N, K, device=DEV) * 0.08
        x = torch.randn(1, K, device=DEV); x[:, K_low:] *= 3.0

        Wlo, Slo, Whi, Shi, W_deq, pb = build_decode_pack(W, K_low, K_high)
        fp16_wb = N * K * 2

        # activation: spec fake-quant (same as the twin / real kernel), fp32 (K,)
        xq = torch.cat([fake_quantize_nvfp4(x[:, :K_low].float()),
                        fake_quantize_mxfp8(x[:, K_low:].float())], dim=-1)[0].contiguous()
        # fast in-kernel activation quant (bit-exact target)
        xq_k = dg.quant_act(x, K_low, K_high)
        qact_cos = cos(xq_k, xq)
        qact_maxdiff = (xq_k - xq).abs().max().item()
        t_qact = time_min(lambda: dg.quant_act(x, K_low, K_high))

        # references
        D_twin = (xq @ W_deq.t()).float()                       # (N,) fp32 twin
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH, qpf=qpf)
        D_fused = mql(x).float().reshape(-1)                    # real fused kernel

        D_gemv = dg.decode_gemv(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high)
        D_gemv2 = dg.decode_gemv_v2(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high)
        D_fusedk = dg.decode_gemv_fused(x, Wlo, Slo, Whi, Shi, N, K_low, K_high)  # single-launch
        c_twin = cos(D_gemv, D_twin)
        c_twin2 = cos(D_gemv2, D_twin)
        c_twinf = cos(D_fusedk, D_twin)
        c_fused = cos(D_fusedk, D_fused)
        worst_cos_twin = min(worst_cos_twin, c_twin, c_twin2, c_twinf)
        worst_cos_fused = min(worst_cos_fused, c_fused)

        # timing
        t_gemv = time_min(lambda: dg.decode_gemv(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high))
        t_gemv2 = time_min(lambda: dg.decode_gemv_v2(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high))
        t_fusedk = time_min(lambda: dg.decode_gemv_fused(x, Wlo, Slo, Whi, Shi, N, K_low, K_high))
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)
        t_fused = time_min(lambda: cd.fused_gemm_prepacked(
            A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
            A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi, 1, N, K_low, K_high))
        Wh = W.half(); xh = x.half()
        t_fp16 = time_min(lambda: xh @ Wh.t())

        t_full = t_fusedk                        # single-launch W4A4 decode (the real path)
        bw_gemv = pb / (t_full / 1e3) / 1e12
        sum_gemv += t_full; sum_fused += t_fused; sum_fp16 += t_fp16
        print(f"  {name:10s} N={N:5d} K={K:5d}: gemv-only v2 {t_gemv2*1e3:5.1f} "
              f"FUSED1launch {t_fusedk*1e3:5.1f}us ({bw_gemv:.3f} TB/s, {bw_gemv/hbm*100:4.1f}% HBM)  "
              f"fused-TC {t_fused*1e3:5.1f} fp16 {t_fp16*1e3:5.1f}us | "
              f"cos(twin)={c_twinf:.6f} qact_cos={qact_cos:.6f} "
              f"vs fused-TC={t_fused/t_full:.1f}x vs fp16={t_fp16/t_full:.2f}x", flush=True)
        rows.append(dict(name=name, N=N, K=K, packed_bytes=pb, fp16_bytes=fp16_wb,
                         gemv_only_v2_us=t_gemv2*1e3, fused1_us=t_fusedk*1e3, qact_us=t_qact*1e3,
                         fused_tc_us=t_fused*1e3, fp16_us=t_fp16*1e3,
                         bw_gemv_tbps=bw_gemv, hbm_frac=bw_gemv/hbm,
                         cos_twin=c_twinf, cos_fused=c_fused,
                         qact_cos=qact_cos, qact_maxdiff=qact_maxdiff,
                         su_vs_fused=t_fused/t_full, su_vs_fp16=t_fp16/t_full))
        del mql, W, x

    print(f"\n  SUM/layer (7 shapes): GEMV={sum_gemv*1e3:.1f}us  fused={sum_fused*1e3:.1f}us  "
          f"fp16={sum_fp16*1e3:.1f}us", flush=True)
    print(f"  GEMV vs fused M=1 = {sum_fused/sum_gemv:.1f}x faster | "
          f"GEMV vs FP16 GEMV = {sum_fp16/sum_gemv:.2f}x faster", flush=True)
    print(f"  worst cosine: vs twin={worst_cos_twin:.6f}  vs fused={worst_cos_fused:.6f} "
          f"-> {'PASS' if min(worst_cos_twin, worst_cos_fused) >= 0.999 else 'FAIL'}", flush=True)
    out = dict(hbm_tbps=hbm, shapes=rows,
               sum_gemv_us=sum_gemv*1e3, sum_fused_us=sum_fused*1e3, sum_fp16_us=sum_fp16*1e3,
               su_gemv_vs_fused=sum_fused/sum_gemv, su_gemv_vs_fp16=sum_fp16/sum_gemv,
               worst_cos_twin=worst_cos_twin, worst_cos_fused=worst_cos_fused,
               gate_pass=bool(min(worst_cos_twin, worst_cos_fused) >= 0.999))
    with open("/home/luofan/decode_gemv.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("RESULTS_WRITTEN /home/luofan/decode_gemv.json", flush=True)
    print("DECODE_GEMV_DONE", flush=True)


if __name__ == "__main__":
    main()
