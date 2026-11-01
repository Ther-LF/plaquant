"""vllm-perf r4 lever 1: Marlin-style W4A16 tensor-core route for small-M decode.

The r3 verdict: the qkv/o-class decode GEMM at M=8..32 is the remaining latency
floor (3B qkv 19.8 us vs a ~6.1 us weight-byte floor) and split-K cannot win
there (K too short). This module reuses vLLM's OWN Marlin fp4 kernel — already
compiled, running and cos=1.0-correct on SM120 (r4 feasibility probe) — for the
NVFP4 LOW segment of our dual-format weights:

  * our checkpoint low-segment layout (uint8 (N, K_low/2) codes with
    low-nibble=even-k + uint8 (N, K_low/16) E4M3 block scales + per-tensor g)
    is EXACTLY the modelopt-NVFP4 layout `prepare_fp4_layer_for_marlin`
    consumes (nibble order verified against `rand_marlin_weight_nvfp4_like`);
  * the repack (`ops.gptq_marlin_repack` tile-interleave + S0E5M3 scale
    re-encode + power-of-2 global-scale fold) runs ONCE at load;
  * at apply time `torch.ops.vllm.marlin_gemm` streams the repacked weights
    L2-resident and feeds m16n8k16 HMMA tiles, so M=16 costs ~M=1;
  * the MXFP8 HIGH segment rides the companion `decode_gemv_w8a16_hi_add`
    kernel (decode_gemv_ext), which adds into Marlin's fp16 output in place.

W4A16 numerics: like the shipped M<=4 GEMV routes, this path SKIPS activation
quantization — the decode-route gate moves in the improvement direction (more
accurate), it is NOT byte-identical to the W4A4 GEMM route. Prefill M never
routes here (measured windows close by M=16; hi_add kernel bound M<=32).

MEMORY COST (honest): the Marlin repack is a second copy of the low-segment
weight bytes (~ +44% quantized-weight bytes; measured 3.36 GiB on 8B — available KV 64.53 -> 61.17 GiB) because the fused
prefill GEMM still needs the original packed layout. PROMIX_MARLIN=0 skips the
prepack and the route entirely.
"""

import os

import torch

_WS = {}


def marlin_enabled():
    return os.environ.get("PROMIX_MARLIN", "1") != "0"


def marlin_hi_enabled():
    """vllm-perf r5 lever 1: MXFP8-marlin high-segment companion (replaces the
    hi_add GEMV inside its measured M-window). PROMIX_MARLIN_HI=0 restores the
    r4 hi_add-only behavior (the lever's serve A/B control)."""
    return os.environ.get("PROMIX_MARLIN_HI", "1") != "0"


def marlin_workspace(device):
    """Shared per-device Marlin workspace (zero-initialized global-sync buffer;
    the kernel leaves it re-zeroed, so sharing + CUDA-graph replay are safe —
    the same contract vLLM's own layers rely on)."""
    key = (device.type, device.index)
    if key not in _WS:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new)
        _WS[key] = marlin_make_workspace_new(device)
    return _WS[key]


class MarlinLow:
    """Load-time repacked Marlin state for one Linear's NVFP4 low segment."""

    def __init__(self, qweight, scales, global_scale, N, K_low, workspace):
        self.qweight = qweight
        self.scales = scales
        self.global_scale = global_scale
        self.N = N
        self.K_low = K_low
        self.workspace = workspace
        from vllm.scalar_type import scalar_types
        self.type_id = scalar_types.float4_e2m1f.id

    def gemm(self, x_lo):
        """x_lo: contiguous (M, K_low) fp16 -> (M, N) fp16 = x_lo @ W_lo_deq.T
        (W_lo_deq = fp4(code) * E4M3(scale) * g, the exact promix low-segment
        dequant). use_atomic_add=False + use_fp32_reduce=True for deterministic
        fp32 cross-CTA reduction."""
        return torch.ops.vllm.marlin_gemm(
            a=x_lo, b_q_weight=self.qweight, b_bias=None, b_scales=self.scales,
            a_scales=None, global_scale=self.global_scale, b_zeros=None,
            workspace=self.workspace, b_q_type_id=self.type_id,
            size_m=x_lo.shape[0], size_n=self.N, size_k=self.K_low,
            use_atomic_add=False, use_fp32_reduce=True)


def prepack_marlin_low(W_lo_packed, SF_lo_bytes, g, N, K_low):
    """One-time load-side repack of the checkpoint low-segment buffers into
    Marlin format. W_lo_packed: uint8 (N, K_low/2) (low nibble = even k);
    SF_lo_bytes: uint8 (N, K_low/16) raw E4M3; g: per-tensor fp32 global.

    Dequant convention check: Marlin computes
        W = fp4(code) * scale * global_scale
    with `nvfp4_marlin_process_scales` re-encoding scales to S0E5M3 and
    returning a power-of-2 `scale_factor` that the global scale is divided by
    — so passing global_scale=g reproduces our fp4*E4M3*g exactly (feasibility
    probe cos=1.0 vs the dequant reference)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        marlin_padded_nk, nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales)
    from vllm import _custom_ops as ops

    assert W_lo_packed.dtype == torch.uint8 and W_lo_packed.shape == (N, K_low // 2)
    assert SF_lo_bytes.dtype == torch.uint8 and SF_lo_bytes.shape == (N, K_low // 16)
    padded_n, padded_k = marlin_padded_nk(N, K_low, 16)
    # All promix Llama shapes are already tile-aligned; refuse the pad path so
    # the hot route never silently pads (keep apply() allocation-free beyond
    # the marlin output).
    assert (padded_n, padded_k) == (N, K_low), \
        f"marlin W4A16: unpadded shape required, got N={N} K_low={K_low} -> {padded_n}x{padded_k}"
    dev = W_lo_packed.device
    qweight = ops.gptq_marlin_repack(
        b_q_weight=W_lo_packed.view(torch.int32).T.contiguous(),
        size_k=K_low, size_n=N, num_bits=4)
    scales = SF_lo_bytes.view(torch.float8_e4m3fn).T.contiguous().to(torch.half)
    scales = marlin_permute_scales(s=scales, size_k=K_low, size_n=N, group_size=16)
    scales, scale_factor = nvfp4_marlin_process_scales(scales, a_dtype=torch.half)
    gscale = nvfp4_marlin_process_global_scale(
        torch.tensor(float(g), dtype=torch.float32, device=dev), torch.half)
    gscale = (gscale / scale_factor).to(torch.float32)
    return MarlinLow(qweight, scales, gscale, N, K_low, marlin_workspace(dev))


class MarlinHigh:
    """vllm-perf r5 lever 1: load-time repacked MXFP8-marlin state for one
    Linear's high segment — the tensor-core companion that replaces the
    linear-in-M hi_add GEMV (r4's named bottleneck: 7x the per-byte cost of
    the marlin low at M=16, closing every window by M=16-24)."""

    def __init__(self, qweight, scales, g, N, K_high, workspace):
        self.qweight = qweight
        self.scales = scales
        self.g = float(g)
        self.N = N
        self.K_high = K_high
        self.workspace = workspace
        from vllm.scalar_type import scalar_types
        self.type_id = scalar_types.float8_e4m3fn.id

    def gemm(self, x_hi_bf16):
        """x_hi_bf16: contiguous (M, K_high) bf16 — marlin's e8m0-scale (MXFP8)
        kernels are compiled for bf16 activations ONLY (feasibility probe:
        fp16-activation group-32 fp8 kernels do not exist in vLLM 0.32), so the
        caller casts the fp16 high slice to bf16 (the W4A4 reference quantizes
        the same slice to E4M3, so bf16 is still ~5 extra mantissa bits).
        Returns H = x_hi @ (E4M3(w) * 2^(e-127)).T (bf16, g NOT applied)."""
        return torch.ops.vllm.marlin_gemm(
            a=x_hi_bf16, b_q_weight=self.qweight, b_bias=None, b_scales=self.scales,
            a_scales=None, global_scale=None, b_zeros=None,
            workspace=self.workspace, b_q_type_id=self.type_id,
            size_m=x_hi_bf16.shape[0], size_n=self.N, size_k=self.K_high,
            use_atomic_add=False, use_fp32_reduce=True)

    def gemm_add(self, Y, x_hi_bf16, axpy=None):
        """Y (fp16, M,N) += g * gemm(x_hi_bf16) — the dequant convention of the
        hi_add GEMV / fused-GEMM high phase (g folded in the fp32-opmath
        accumulate; e8m0 scales cannot hold it). `axpy` = the decode ext's
        fused axpy_bf16_into_half (ONE kernel); None falls back to the torch
        cast + alpha-add pair."""
        H = self.gemm(x_hi_bf16)
        if axpy is not None:
            axpy(Y, H, self.g)
        else:
            Y.add_(H.to(torch.float16), alpha=self.g)
        return Y


def prepack_marlin_high(W_hi_bytes, SF_hi_bytes, g, N, K_high):
    """One-time load-side repack of the checkpoint high-segment buffers into
    marlin's native MXFP8 format. W_hi_bytes: uint8 (N, K_high) raw E4M3 codes;
    SF_hi_bytes: uint8 (N, K_high/32) raw UE8M0 block scales; g: per-tensor
    fp32 global (NOT folded into the scales — e8m0 only holds powers of two —
    it rides the companion's add alpha, see MarlinHigh.gemm_add).

    Mirrors vLLM's `prepare_mxfp8_layer_for_marlin` exactly (pack_fp8_to_int32
    + gptq_marlin_repack num_bits=8 + e8m0 scale permute +
    mxfp8_marlin_process_scales); like the low-segment prepack we refuse the
    pad path (all promix shapes are tile-aligned: K_high in {384,512,1024,1792}
    is a multiple of 128). The apply consumes bf16 activations (the only
    a_dtype marlin's e8m0 kernels are compiled for)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_padded_nk, marlin_permute_scales)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        mxfp8_marlin_process_scales, pack_fp8_to_int32)
    from vllm import _custom_ops as ops

    assert W_hi_bytes.dtype == torch.uint8 and W_hi_bytes.shape == (N, K_high)
    assert SF_hi_bytes.dtype == torch.uint8 and SF_hi_bytes.shape == (N, K_high // 32)
    padded_n, padded_k = marlin_padded_nk(N, K_high, 32)
    assert (padded_n, padded_k) == (N, K_high), \
        f"marlin hi: unpadded shape required, got N={N} K_high={K_high} -> {padded_n}x{padded_k}"
    dev = W_hi_bytes.device
    qweight = ops.gptq_marlin_repack(
        b_q_weight=pack_fp8_to_int32(
            W_hi_bytes.view(torch.float8_e4m3fn), size_k_first=False).T.contiguous(),
        size_k=K_high, size_n=N, num_bits=8)
    scales = SF_hi_bytes.view(torch.float8_e8m0fnu).to(torch.float32).T.contiguous()
    scales = marlin_permute_scales(s=scales, size_k=K_high, size_n=N, group_size=32)
    scales = mxfp8_marlin_process_scales(scales)
    return MarlinHigh(qweight, scales, g, N, K_high, marlin_workspace(dev))
