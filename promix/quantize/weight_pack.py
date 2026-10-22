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
def quantize_nvfp4_spec(x, block=16):
    """NVFP4: returns (codes uint8 (R,K), sf uint8 (R,K/16), deq f32 (R,K)).

    deq == quant_utils.fake_quantize_nvfp4(x); sf = E4M3 byte of the round-UP
    block scale; codes = FP4 E2M1 (abs index | 0x8 sign)."""
    R, K = x.shape
    assert K % block == 0
    dev = x.device
    xf = x.float().reshape(R, K // block, block)
    bmax = xf.abs().amax(-1, keepdim=True)
    ideal = (bmax / qu.FP4_E2M1_MAX).clamp_min(2.0 ** -9)
    e4m3 = torch.tensor(qu._FP8_E4M3_POS, device=dev, dtype=torch.float32)
    scale = qu._ceil_to_value(ideal, e4m3)                      # (R,nb,1) grid value
    xs = (xf / scale).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX)
    fp4 = torch.tensor(qu.FP4_E2M1_POS, device=dev, dtype=torch.float32)
    qsigned = qu._round_to_nearest_value(xs, fp4)               # signed grid value
    absidx = (qsigned.abs().unsqueeze(-1) == fp4.view(1, 1, 1, -1)).float().argmax(-1)
    code = absidx.to(torch.uint8)
    neg = (qsigned < 0) & (absidx > 0)
    code = code | (neg.to(torch.uint8) * 8)
    deq = (qsigned * scale).reshape(R, K)
    sf = scale.to(torch.float8_e4m3fn).view(torch.uint8).reshape(R, K // block).contiguous()
    return code.reshape(R, K).contiguous(), sf, deq


def quantize_mxfp8_spec(x, block=32):
    """MXFP8: returns (bytes uint8 (R,K), sf uint8 (R,K/32), deq f32 (R,K)).

    deq == quant_utils.fake_quantize_mxfp8(x); sf = E8M0 exponent byte
    (exp+127); bytes = FP8 E4M3 element encoding."""
    R, K = x.shape
    assert K % block == 0
    dev = x.device
    xf = x.float().reshape(R, K // block, block)
    bmax = xf.abs().amax(-1, keepdim=True).clamp_min(0.0)
    ideal = bmax / qu.FP8_E4M3_MAX
    scale = qu._e8m0_round_up(ideal.clamp_min(2.0 ** -127))     # power-of-2
    xs = (xf / scale).clamp(-qu.FP8_E4M3_MAX, qu.FP8_E4M3_MAX)
    e4m3 = torch.tensor(qu._FP8_E4M3_POS, device=dev, dtype=torch.float32)
    qsigned = qu._round_to_nearest_value(xs, e4m3)
    deq = (qsigned * scale).reshape(R, K)
    bytes_ = qsigned.to(torch.float8_e4m3fn).view(torch.uint8).reshape(R, K).contiguous()
    exp = torch.log2(scale).round().to(torch.int32)
    sf = (exp + 127).clamp(0, 254).to(torch.uint8).reshape(R, K // block).contiguous()
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


def split_k(K, high_fraction=0.125):
    """ResQ K split: K_high = high_fraction*K (last), K_low = rest (first).

    Asserts block alignment: K_low % 16 == 0 (NVFP4), K_high % 32 == 0 (MXFP8).
    """
    K_high = int(round(high_fraction * K))
    K_low = K - K_high
    assert K_low % 16 == 0, f"K_low={K_low} not divisible by NVFP4 block 16 (K={K})"
    assert K_high % 32 == 0, f"K_high={K_high} not divisible by MXFP8 block 32 (K={K})"
    return K_low, K_high


def pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=None):
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

    # --- NVFP4 low/main segment (spec-matching quant -> faithful to fake-quant) ---
    codes_lo, SFB_lo, deq_lo = quantize_nvfp4_spec(W_lo)
    W_lo_packed = nv.pack_fp4(codes_lo)
    physSFB_lo = cd.prepack_sf_fp4_b(SFB_lo.contiguous(), N, K_low)

    # --- MXFP8 high segment ---
    bytes_hi, SFB_hi, deq_hi = quantize_mxfp8_spec(W_hi)
    physSFB_hi = cd.prepack_sf_fp8_b(SFB_hi.contiguous(), N, K_high)

    W_deq = torch.cat([deq_lo, deq_hi], dim=-1).contiguous()

    return PackedWeight(
        N=N, K=K, K_low=K_low, K_high=K_high,
        W_lo_packed=W_lo_packed.contiguous(),
        physSFB_lo=physSFB_lo.contiguous(),
        W_hi_bytes=bytes_hi.contiguous(),
        physSFB_hi=physSFB_hi.contiguous(),
        W_deq=W_deq,
        bias=None if bias is None else bias.detach().to(dev).contiguous(),
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
    _, SFB_lo, _ = quantize_nvfp4_spec(Wf[:, :K_low].contiguous())
    _, SFB_hi, _ = quantize_mxfp8_spec(Wf[:, K_low:].contiguous())
    phys_lo_ref = cd.prepack_sf_fp4_b(SFB_lo.contiguous(), N, K_low)
    phys_hi_ref = cd.prepack_sf_fp8_b(SFB_hi.contiguous(), N, K_high)

    pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
    return {
        "sfb_lo_equal": bool(torch.equal(pk.physSFB_lo, phys_lo_ref)),
        "sfb_hi_equal": bool(torch.equal(pk.physSFB_hi, phys_hi_ref)),
        "data_lo_shape": tuple(pk.W_lo_packed.shape),
        "data_hi_shape": tuple(pk.W_hi_bytes.shape),
    }
