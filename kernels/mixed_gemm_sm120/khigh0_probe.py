"""rd-kfix probe: K_high==0 pure-NVFP4 real-kernel correctness + zero-regression.

Proves the fix for the silent-fallback / crash bug: a Linear configured with
K_high==0 (pure-NVFP4, 0 MXFP8 rings) now (a) PACKS without crashing, (b) runs
the single-phase NVFP4 GEMM (Phase-2 skipped) instead of the two-phase fused
kernel, and (c) produces output matching the fake-quant pure-NVFP4 reference
(cosine >= 0.999) — NOT a 1/8 mix.

Also checks zero-regression: a normal K_high>0 (1/8) Linear is unchanged, and the
symmetric K_low==0 (pure-MXFP8) config raises a CLEAR error (no silent clamp).

Run inside the luofan-plaquant container on a free GPU:
    CUDA_VISIBLE_DEVICES=5 python kernels/mixed_gemm_sm120/khigh0_probe.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from promix.eval.full_network_realkernel import load_kernels, cosine  # noqa: E402
from promix.quantize.weight_pack import pack_weight_mixed  # noqa: E402
from promix.quantize.mixed_linear import MixedQuantLinear  # noqa: E402
from promix.quantize.quant_utils import (  # noqa: E402
    fake_quantize_nvfp4, fake_quantize_mxfp8,
)

DEV = "cuda"


def _ref(x, Wdeq, K_low):
    """Fake-quant twin: fake_quantize the activation (NVFP4 low | MXFP8 high) and
    FP32-matmul against the SAME dequantized weight the kernel accumulates."""
    x = x.float()
    K = x.shape[1]
    xq_lo = fake_quantize_nvfp4(x[:, :K_low])
    if K_low < K:
        xq_hi = fake_quantize_mxfp8(x[:, K_low:])
        xq = torch.cat([xq_lo, xq_hi], dim=-1)
    else:
        xq = xq_lo
    return xq @ Wdeq.t().float()


def run_shape(name, M, N, K, K_high, nv, mx, cd, qpf, seed=0):
    torch.manual_seed(seed)
    W = (torch.randn(N, K, device=DEV) * 0.1)
    x = (torch.randn(M, K, device=DEV))
    K_low = K - K_high
    pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
    assert pk.K_high == K_high and pk.K_low == K_low
    mql = MixedQuantLinear(pk, nv, mx, cd, out_dtype=torch.float16, qpf=qpf).to(DEV)
    assert mql.is_pure_nvfp4 == (K_high == 0)
    D = mql(x.half()).float()
    Dref = _ref(x, pk.W_deq, K_low)
    cos = cosine(D, Dref)
    path = "single-phase NVFP4 (Phase-2 skipped)" if K_high == 0 else "two-phase fused"
    print(f"  [{name:14s}] M={M} N={N} K={K} K_high={K_high:4d}  "
          f"cosine={cos:.6f}  [{path}]", flush=True)
    return cos


def main():
    print("Loading kernels (first build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    assert hasattr(cd, "single_nvfp4_gemm_prepacked"), \
        "cd module missing single_nvfp4_gemm_prepacked — rebuild codesign_fused_ext"
    print("Kernels loaded.\n", flush=True)

    ok = True
    # Llama-3.2-1B target shapes (hidden 2048). K_high=256 == 1/8 split.
    shapes = [
        ("qkv",      2048, 2048, 2048),
        ("gate/up",  2048, 8192, 2048),
        ("down",     2048, 2048, 8192),
    ]
    print("== K_high==0 pure-NVFP4 (THE FIX): must match fake-quant pure-NVFP4 ==")
    for nm, M, N, K in shapes:
        cos = run_shape(nm, M, N, K, 0, nv, mx, cd, qpf)
        ok = ok and cos >= 0.999

    print("\n== zero-regression: K_high=1/8 two-phase fused unchanged ==")
    for nm, M, N, K in shapes:
        kh = (round(0.125 * K / 32)) * 32
        cos = run_shape(nm, M, N, K, kh, nv, mx, cd, qpf)
        ok = ok and cos >= 0.999

    print("\n== K_low==0 pure-MXFP8: must raise a CLEAR error (no silent clamp) ==")
    raised = False
    try:
        torch.manual_seed(0)
        W = torch.randn(2048, 2048, device=DEV) * 0.1
        pack_weight_mixed(W, 0, 2048, nv, mx, cd)
    except ValueError as e:
        raised = "K_low==0" in str(e)
        print(f"  raised ValueError (good): {str(e)[:90]}...", flush=True)
    ok = ok and raised

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
