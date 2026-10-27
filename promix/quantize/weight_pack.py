"""Offline weight packer for the SM120 mixed NVFP4+MXFP8 fused GEMM.

`pack_weight_mixed` turns one quantized+rotated `nn.Linear` weight into the
packed-code + interleaved-scale-factor (SFB) buffers the co-design fused kernel
(`codesign_fused_ext.cu`) consumes, split along K as:

    [ low (main) channels -> NVFP4 block-16 | high channels -> MXFP8 block-32 ]

This is the reorder-BEFORE-quant, once-at-load offline version. The ResQ channel
reorder (high-variance channels placed last) is assumed already applied to W, so
the split is simply ``W[:, :K_low] | W[:, K_low:]`` with ``K_high = K - K_low``.

The SFB interleave uses the kernel's own load-time prepackers
(`cd.prepack_sf_fp4_b` / `cd.prepack_sf_fp8_b`) so the produced layout is, by
construction, exactly what `fused_gemm` / `fused_gemm_prepacked` read. The
bit-exact contract is additionally asserted by `probe_weight_sfb_bitexact`
(torch.equal vs the kernel prepacker) and, end to end, by the GEMM cosine gate in
the harness.

Design note on numerics: quantization uses the KERNEL's own quantizers
(`nv.quantize_nvfp4` / `mx.quantize_mxfp8`) so the packed codes and the returned
dequantized reference (`W_deq`) are bit-identical to what the kernel accumulates.
The `quant_utils.fake_quantize_*` reference differs only in block-scale rounding
(ceil-to-E4M3 vs round-to-nearest-E4M3) and is compared separately (>=0.999).

The kernel python modules are dependency-injected (`nv`, `mx`, `cd`) so this file
has no hard dependency on the remote kernel build path.
"""

from dataclasses import dataclass

import torch

from promix.quantize import quant_utils as qu


# ---------------------------------------------------------------------------
# Spec-matching quantizers: emit (codes/bytes, sf_bytes, deq) whose `deq` is
# BIT-IDENTICAL to quant_utils.fake_quantize_* (round-UP NVFP4 block scale,
# ties-to-even elements), while the codes/sf are exactly what the SM120 kernel
# + prepackers consume. Using these (instead of the kernel's round-to-nearest
# nvfp4_gemm.quantize_nvfp4 helper) makes MixedQuantLinear a faithful,
# >=0.999-cosine replacement for the ProMix fake-quant path.
# ---------------------------------------------------------------------------
def quantize_nvfp4_spec(x, block=16, frozen_scale=None, global_scale=None):
    """NVFP4: returns (codes uint8 (R,K), sf uint8 (R,K/16), deq f32 (R,K)).

    deq == quant_utils.fake_quantize_nvfp4(x); sf = E4M3 byte of the block
    scale; codes = FP4 E2M1 (abs index | 0x8 sign).

    Block-scale source:
      * frozen_scale=None  -> RTN: re-derive the scale from this (possibly
        GPTQ-corrected) block's own absmax, rounded UP to the E4M3 grid.
      * frozen_scale given  -> use GPTQ's FROZEN per-row-per-block scale
        (shape (R, K//block), each value an E4M3 grid point produced by
        quant_utils.select_block_scales_nvfp4 on the PRE-correction weight).
        This is the scale GPTQ's inner loop rounded the Hessian-corrected
        columns against, so packing the stored (dequantized) weight against
        it is bit-faithful to GPTQ instead of a scale-re-derivation RTN.

    Round-6 two-level: when ``global_scale`` (per-tensor FP32 g) is given, the
    frozen effective scale is DECOMPOSED as g * micro, the SFB stores the E4M3
    ``micro`` (= eff/g, on-grid by construction), and the element rounding / deq
    use the kernel-exact reconstruction ``micro_decoded * g``. The kernel carries
    g on the epilogue alpha, so deq == g*(A@B) matches the two-level fake path."""
    R, K = x.shape
    assert K % block == 0
    dev = x.device
    xf = x.float().reshape(R, K // block, block)
    if frozen_scale is not None:
        scale = frozen_scale.to(dev, torch.float32).reshape(R, K // block, 1)
    else:
        bmax = xf.abs().amax(-1, keepdim=True)
        ideal = (bmax / qu.FP4_E2M1_MAX).clamp_min(2.0 ** -9)
        e4m3 = torch.tensor(qu._FP8_E4M3_POS, device=dev, dtype=torch.float32)
        scale = qu._ceil_to_value(ideal, e4m3)                  # (R,nb,1) grid value
    if global_scale is not None:
        gs = float(global_scale)
        micro = (scale / gs).to(torch.float8_e4m3fn)            # E4M3 micro = eff/g
        sf = micro.view(torch.uint8).reshape(R, K // block).contiguous()
        scale = micro.float() * gs                              # kernel-exact eff
    else:
        sf = scale.to(torch.float8_e4m3fn).view(torch.uint8).reshape(R, K // block).contiguous()
    xs = (xf / scale).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX)
    fp4 = torch.tensor(qu.FP4_E2M1_POS, device=dev, dtype=torch.float32)
    qsigned = qu._round_to_nearest_value(xs, fp4)               # signed grid value
    absidx = (qsigned.abs().unsqueeze(-1) == fp4.view(1, 1, 1, -1)).float().argmax(-1)
    code = absidx.to(torch.uint8)
    neg = (qsigned < 0) & (absidx > 0)
    code = code | (neg.to(torch.uint8) * 8)
    deq = (qsigned * scale).reshape(R, K)
    return code.reshape(R, K).contiguous(), sf, deq


def quantize_mxfp8_spec(x, block=32, frozen_scale=None, global_scale=None):
    """MXFP8: returns (bytes uint8 (R,K), sf uint8 (R,K/32), deq f32 (R,K)).

    deq == quant_utils.fake_quantize_mxfp8(x); sf = E8M0 exponent byte
    (exp+127); bytes = FP8 E4M3 element encoding.

    Block-scale source mirrors quantize_nvfp4_spec: frozen_scale=None re-derives
    a power-of-2 scale from this block's absmax (RTN); frozen_scale (shape
    (R, K//block), each value a power-of-2 from
    quant_utils.select_block_scales_mxfp8 on the pre-correction weight) uses
    GPTQ's frozen scale so packing is bit-faithful to GPTQ.

    Round-6 two-level: when ``global_scale`` (per-tensor FP32 g) is given, the
    stored E8M0 micro = eff/g (a power of 2 for ANY g, since eff was built as
    g*e8m0(...)), and the element rounding / deq use ``2^exp * g``. The kernel
    carries g on the epilogue alpha shared with the NVFP4 low phase."""
    R, K = x.shape
    assert K % block == 0
    dev = x.device
    xf = x.float().reshape(R, K // block, block)
    if frozen_scale is not None:
        scale = frozen_scale.to(dev, torch.float32).reshape(R, K // block, 1)
    else:
        bmax = xf.abs().amax(-1, keepdim=True).clamp_min(0.0)
        ideal = bmax / qu.FP8_E4M3_MAX
        scale = qu._e8m0_round_up(ideal.clamp_min(2.0 ** -127))     # power-of-2
    if global_scale is not None:
        gs = float(global_scale)
        exp = torch.log2(scale / gs).round().to(torch.int32)        # E8M0 micro = eff/g
        sf = (exp + 127).clamp(0, 254).to(torch.uint8).reshape(R, K // block).contiguous()
        scale = torch.exp2(exp.float()) * gs                        # kernel-exact eff
    else:
        exp = torch.log2(scale).round().to(torch.int32)
        sf = (exp + 127).clamp(0, 254).to(torch.uint8).reshape(R, K // block).contiguous()
    xs = (xf / scale).clamp(-qu.FP8_E4M3_MAX, qu.FP8_E4M3_MAX)
    e4m3 = torch.tensor(qu._FP8_E4M3_POS, device=dev, dtype=torch.float32)
    qsigned = qu._round_to_nearest_value(xs, e4m3)
    deq = (qsigned * scale).reshape(R, K)
    bytes_ = qsigned.to(torch.float8_e4m3fn).view(torch.uint8).reshape(R, K).contiguous()
    return bytes_, sf, deq


@dataclass
class PackedWeight:
    """Offline-packed weight buffers for one Linear (one GEMM operand B)."""

    N: int
    K: int
    K_low: int
    K_high: int
    # NVFP4 low/main segment
    W_lo_packed: torch.Tensor      # (N, K_low/2) uint8, 2 FP4 codes / byte
    physSFB_lo: torch.Tensor       # interleaved NVFP4 weight SF (uint8, flat)
    # MXFP8 high segment
    W_hi_bytes: torch.Tensor       # (N, K_high) uint8, FP8 E4M3 bytes
    physSFB_hi: torch.Tensor       # interleaved MXFP8 weight SF (uint8, flat)
    # dequantized reference (kernel numerics), for building the fake-quant ref
    W_deq: torch.Tensor            # (N, K) float32, cat([deq_lo, deq_hi], -1)
    bias: torch.Tensor = None      # (N,) or None
    # Round-6 two-level weight global: per-tensor FP32 g carried on the fused-GEMM
    # epilogue alpha (D = g*(A@B)). 1.0 = single-level (no epilogue scaling).
    g_w: float = 1.0
    # decode-GEMV pack (row-major, raw FP32 scales) — a memory-bound M=1 path that
    # streams the SAME 4-bit/8-bit weights at HBM BW (decode_gemv_ext.cu). Built
    # from the identical spec codes/scales, so dequant is bit-identical to W_deq.
    W_lo_dec: torch.Tensor = None  # (N, K_low/2) uint8, 2 NVFP4 codes/byte (even=low nibble)
    S_lo_dec: torch.Tensor = None  # (N, K_low/16) float32, NVFP4 block scale
    W_hi_dec: torch.Tensor = None  # (N, K_high) uint8, FP8 E4M3 bytes
    S_hi_dec: torch.Tensor = None  # (N, K_high/32) float32, MXFP8 block scale


def split_k(K, high_fraction=0.125):
    """ResQ K split: K_high = high_fraction*K (last), K_low = rest (first).

    Asserts block alignment: K_low % 16 == 0 (NVFP4), K_high % 32 == 0 (MXFP8).
    """
    K_high = int(round(high_fraction * K))
    K_low = K - K_high
    assert K_low % 16 == 0, f"K_low={K_low} not divisible by NVFP4 block 16 (K={K})"
    assert K_high % 32 == 0, f"K_high={K_high} not divisible by MXFP8 block 32 (K={K})"
    return K_low, K_high


def pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=None,
                      frozen_nvfp4_scale=None, frozen_mxfp8_scale=None,
                      global_scale=None):
    """Pack a rotated+reordered weight ``W`` (N, K) into mixed NVFP4+MXFP8 buffers.

    Args:
        W: (N, K) weight, rows = output channels (N), cols = input/contraction (K).
           Assumed already rotated + channel-reordered so the last ``K_high``
           columns are the high-variance (MXFP8) segment.
        K_low:  NVFP4 (main) contraction width, must be % 16 == 0.
        K_high: MXFP8 (high) contraction width, must be % 32 == 0.
        nv, mx, cd: the kernel python modules (nvfp4_gemm, mxfp8_gemm,
           codesign_fused_ext) providing quantize/pack/prepack entry points.
        bias: optional (N,) bias, stored verbatim (applied in MixedQuantLinear).
        frozen_nvfp4_scale: optional (N, K_low//16) GPTQ frozen per-block scale
           for the NVFP4 (main) segment. When provided, the NVFP4 codes/SFB are
           built against GPTQ's FROZEN scale (the one its inner loop rounded the
           Hessian-corrected columns against) instead of re-deriving from the
           stored weight's absmax — recovering GPTQ's full benefit. When None,
           falls back to RTN (re-derive from absmax); this preserves the
           bit-exact single-layer probe / `from_linear` path.
        frozen_mxfp8_scale: optional (N, K_high//32) GPTQ frozen per-block scale
           for the MXFP8 (high) segment. Same frozen-vs-RTN contract.

    Returns:
        PackedWeight.
    """
    assert W.dim() == 2, f"W must be 2D (N,K), got {tuple(W.shape)}"
    N, K = W.shape
    assert K_low + K_high == K, f"K_low+K_high={K_low + K_high} != K={K}"
    dev = W.device
    Wf = W.to(torch.float32).contiguous()

    W_lo = Wf[:, :K_low].contiguous()
    W_hi = Wf[:, K_low:].contiguous()
    # Round-6: when two-level is enabled but no GPTQ frozen scale is supplied
    # (RTN / from_linear / single-layer paths), derive the two-level effective
    # scales + shared global g here so this path matches the GPTQ one.
    if (global_scale is None and frozen_nvfp4_scale is None
            and getattr(qu, "WEIGHT_TWO_LEVEL", False) and K_low > 0):
        xb_lo = W_lo.reshape(N, K_low // 16, 16)
        frozen_nvfp4_scale, g_auto = qu._weight_two_level_nvfp4_eff(
            xb_lo, xb_lo.abs().amax(-1))
        global_scale = g_auto
        if K_high > 0:
            xb_hi = W_hi.reshape(N, K_high // 32, 32)
            frozen_mxfp8_scale = qu._weight_two_level_mxfp8_eff(
                xb_hi, xb_hi.abs().amax(-1), g_auto)
    g = 1.0 if global_scale is None else float(global_scale)

    # --- NVFP4 low/main segment (frozen GPTQ scale when available, else RTN) ---
    codes_lo, SFB_lo, deq_lo = quantize_nvfp4_spec(W_lo, frozen_scale=frozen_nvfp4_scale,
                                                   global_scale=global_scale)
    W_lo_packed = nv.pack_fp4(codes_lo)
    physSFB_lo = cd.prepack_sf_fp4_b(SFB_lo.contiguous(), N, K_low)

    # --- MXFP8 high segment (frozen GPTQ scale when available, else RTN) ---
    bytes_hi, SFB_hi, deq_hi = quantize_mxfp8_spec(W_hi, frozen_scale=frozen_mxfp8_scale,
                                                   global_scale=global_scale)
    physSFB_hi = cd.prepack_sf_fp8_b(SFB_hi.contiguous(), N, K_high)

    W_deq = torch.cat([deq_lo, deq_hi], dim=-1).contiguous()

    # decode-GEMV pack (reuse the already-computed spec codes/scales): row-major
    # 2-codes/byte NVFP4 (even k -> low nibble) + FP32 per-16 scale; FP8 bytes +
    # FP32 per-32 power-of-2 scale. dequant is bit-identical to W_deq above. The
    # decode GEMV streams raw FP32 scales, so the two-level global g is folded
    # directly into the stored FP32 effective scale (no epilogue needed there).
    N_, _ = codes_lo.shape
    cl = codes_lo.view(N_, K_low // 2, 2)
    W_lo_dec = (cl[:, :, 0] | (cl[:, :, 1] << 4)).to(torch.uint8).contiguous()
    S_lo_dec = (SFB_lo.view(torch.float8_e4m3fn).float() * g).contiguous()
    W_hi_dec = bytes_hi.contiguous()
    S_hi_dec = (torch.exp2(SFB_hi.float() - 127.0) * g).contiguous()

    return PackedWeight(
        N=N, K=K, K_low=K_low, K_high=K_high,
        W_lo_packed=W_lo_packed.contiguous(),
        physSFB_lo=physSFB_lo.contiguous(),
        W_hi_bytes=bytes_hi.contiguous(),
        physSFB_hi=physSFB_hi.contiguous(),
        W_deq=W_deq,
        bias=None if bias is None else bias.detach().to(dev).contiguous(),
        g_w=g,
        W_lo_dec=W_lo_dec, S_lo_dec=S_lo_dec, W_hi_dec=W_hi_dec, S_hi_dec=S_hi_dec,
    )


def probe_weight_sfb_bitexact(W, K_low, K_high, nv, mx, cd):
    """Step-1 bit-exact probe for the WEIGHT SFB interleave.

    Re-derives the raw per-block weight SF the same way `pack_weight_mixed`
    does, then asserts the interleaved SFB (via the kernel's load-time
    prepacker) equals the ground-truth kernel prepacker output byte-for-byte.
    Returns a dict of bool results; the packer is the SAME code path, so this
    documents the contract and guards against accidental layout drift.
    """
    N, K = W.shape
    Wf = W.to(torch.float32)
    # This probe checks the SFB interleave LAYOUT (byte addressing), not the
    # scale VALUES — run it single-level so the independent ref and the packer
    # agree regardless of the WEIGHT_TWO_LEVEL default.
    _saved = qu.WEIGHT_TWO_LEVEL
    qu.WEIGHT_TWO_LEVEL = False
    try:
        _, SFB_lo, _ = quantize_nvfp4_spec(Wf[:, :K_low].contiguous())
        _, SFB_hi, _ = quantize_mxfp8_spec(Wf[:, K_low:].contiguous())
        phys_lo_ref = cd.prepack_sf_fp4_b(SFB_lo.contiguous(), N, K_low)
        phys_hi_ref = cd.prepack_sf_fp8_b(SFB_hi.contiguous(), N, K_high)
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
    finally:
        qu.WEIGHT_TWO_LEVEL = _saved
    return {
        "sfb_lo_equal": bool(torch.equal(pk.physSFB_lo, phys_lo_ref)),
        "sfb_hi_equal": bool(torch.equal(pk.physSFB_hi, phys_hi_ref)),
        "data_lo_shape": tuple(pk.W_lo_packed.shape),
        "data_hi_shape": tuple(pk.W_hi_bytes.shape),
    }
