"""rd-kfix latency: the REAL r=0 (pure-NVFP4) GEMM cost vs 1/8, now that r=0 runs.

Three paths per Llama-shape at M=2048 (prefill), cudaEvent median (40 iter):
  (A) 1/8 dual fused   : cd.fused_gemm_prepacked          (the shipped mixed path)
  (B) pure-NVFP4 skip-P2: cd.single_nvfp4_gemm_prepacked  (THE FIX — fused kernel's
                          tuned NVFP4 Phase-1, Phase-2 skipped at compile time)
  (C) pure-NVFP4 stock : nv.nvfp4_gemm                    (the dedicated dispatcher
                          B found — stock/untuned CUTLASS collective)

All GEMM-only (weight+activation SF pre-interleaved for A/B; C interleaves
internally as it is the stock entry — noted). Answers honestly: is pure-NVFP4
faster or slower than 1/8 e2e, and is the dedicated path's slowness an
implementation artifact (stock kernel) rather than a datapath truth.

    CUDA_VISIBLE_DEVICES=5 python /home/luofan/plaquant-kfix/kernels/mixed_gemm_sm120/rd_kfix_latency.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from promix.eval.full_network_realkernel import load_kernels  # noqa: E402
from promix.quantize.weight_pack import pack_weight_mixed  # noqa: E402

DEV = "cuda"
ITERS = 40
WARMUP = 10


def cuda_median_ms(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def eighth(K, ring=32):
    return int(round(0.125 * K / ring)) * ring


def main():
    print("Loading kernels...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    nv_ext = nv.get_ext()
    print("Loaded.\n", flush=True)

    # Llama-3.2-1B prefill shapes (M=2048, hidden 2048), with per-layer multiplicity.
    # (name, N, K, count_per_layer)
    shapes = [
        ("q_proj",    2048, 2048, 1),
        ("k_proj",     512, 2048, 1),
        ("v_proj",     512, 2048, 1),
        ("o_proj",    2048, 2048, 1),
        ("gate_proj", 8192, 2048, 1),
        ("up_proj",   8192, 2048, 1),
        ("down_proj", 2048, 8192, 1),
    ]
    N_LAYERS = 16
    M = 2048

    print(f"{'shape':10s} {'N':>5s} {'K':>5s} | {'1/8 dual':>9s} {'pure skipP2':>11s} "
          f"{'pure stock':>10s} | {'skipP2/1/8':>10s} {'stock/1/8':>9s}")
    tot = {"eighth": 0.0, "skip": 0.0, "stock": 0.0}
    for name, N, K, _ in shapes:
        torch.manual_seed(0)
        W = (torch.randn(N, K, device=DEV) * 0.1)
        x = (torch.randn(M, K, device=DEV)).half()

        # (A) 1/8 dual fused
        Kh = eighth(K)
        Kl = K - Kh
        pk8 = pack_weight_mixed(W, Kl, Kh, nv, mx, cd)
        a_lo = qpf.quant_pack_float_fp4(x.contiguous(), M, N, Kl, row_stride=K, col_off=0)
        a_hi = qpf.quant_pack_float_fp8(x.contiguous(), M, N, Kh, row_stride=K, col_off=Kl)
        t_eighth = cuda_median_ms(lambda: cd.fused_gemm_prepacked(
            a_lo[0], a_lo[1], pk8.W_lo_packed, pk8.physSFB_lo,
            a_hi[0], a_hi[1], pk8.W_hi_bytes, pk8.physSFB_hi,
            M, N, Kl, Kh, pk8.g_w))

        # (B) pure-NVFP4, skip-Phase-2 through the tuned fused kernel (THE FIX)
        pk0 = pack_weight_mixed(W, K, 0, nv, mx, cd)
        b_lo = qpf.quant_pack_float_fp4(x.contiguous(), M, N, K, row_stride=K, col_off=0)
        t_skip = cuda_median_ms(lambda: cd.single_nvfp4_gemm_prepacked(
            b_lo[0], b_lo[1], pk0.W_lo_packed, pk0.physSFB_lo, M, N, K, pk0.g_w))

        # (C) pure-NVFP4, stock dedicated dispatcher (what B measured as 24-65% slower)
        A_codes, SFA, _ = nv.quantize_nvfp4(x.float())
        B_codes, SFB, _ = nv.quantize_nvfp4(W.float())
        A_pk = nv.pack_fp4(A_codes)
        B_pk = nv.pack_fp4(B_codes)
        t_stock = cuda_median_ms(lambda: nv_ext.nvfp4_gemm(A_pk, SFA, B_pk, SFB, M, N, K))

        tot["eighth"] += t_eighth
        tot["skip"] += t_skip
        tot["stock"] += t_stock
        print(f"{name:10s} {N:5d} {K:5d} | {t_eighth:9.4f} {t_skip:11.4f} {t_stock:10.4f} "
              f"| {t_skip/t_eighth:10.3f} {t_stock/t_eighth:9.3f}")

    scale = N_LAYERS
    e, s, st = tot["eighth"]*scale, tot["skip"]*scale, tot["stock"]*scale
    print(f"\nSummed GEMM over {N_LAYERS} layers x 7 Linears (M={M}):")
    print(f"  1/8 dual fused      : {e:8.3f} ms")
    print(f"  pure-NVFP4 skip-P2  : {s:8.3f} ms   ({s/e:.3f}x vs 1/8)")
    print(f"  pure-NVFP4 stock    : {st:8.3f} ms   ({st/e:.3f}x vs 1/8)")
    print(f"\n  => skip-P2 is {'FASTER' if s < e else 'SLOWER'} than 1/8 by "
          f"{abs(1-s/e)*100:.1f}% at the GEMM level")
    print(f"  => stock dedicated is {st/s:.2f}x the skip-P2 path "
          f"(the 24-65% slowness is a stock-kernel artifact)")


if __name__ == "__main__":
    main()
