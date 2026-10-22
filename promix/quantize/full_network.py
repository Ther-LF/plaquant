"""Full-network wiring: replace every decoder layer's 7 Linears' fake-quant
forward with the real packed NVFP4+MXFP8 fused-kernel path (MixedQuantLinear),
scaling the validated single-layer integration to all 16 layers (112 Linears).

The prepared ProMix model (see `promix.eval.ptq.prepare_ptq_model`) wraps every
Linear in an `ActQuantWrapper` that, per forward, does:

    x -> [online Hadamard (down_proj)] -> [reorder (o_proj, reorder_before_quant)]
      -> quantizer (NVFP4 main + MXFP8 high fake-quant) -> module (FP16 matmul)
      -> [out_quantizer (v_proj; FP16 here)]

This module keeps everything structural (Hadamard, o_proj reorder, column order,
out_quantizer, RMSNorm/attention/RoPE/embed/lm_head) UNCHANGED and only swaps the
`quantizer -> module` pair for the real kernel (or a bit-faithful fake twin):

  * `wire_real_kernel`  -> `wrapper.module = MixedQuantLinear` (real packed GEMM),
                           and `wrapper.quantizer.bits = 16` (disable fake-quant,
                           since MixedQuantLinear quantizes the activation itself).
  * `wire_fake_mixed`   -> `wrapper.module = FakeMixedLinear` (fake_quantize the
                           activation + FP16 matmul against the SAME dequantized
                           weight `W_deq` the real kernel uses). The numeric twin
                           of the real path — real-vs-twin isolates the GEMM /
                           packing / scale / accumulation path across all 112
                           Linears (the PPL gate).

Both wirings share one offline pack (`pack_all_targets`) so the twin's weight is
byte-identical to what the real kernel dequantizes: real-vs-twin differ ONLY in
the matmul (real fused kernel vs torch FP32 matmul of the dequantized operands).
"""

import re

import torch
import torch.nn as nn

from promix.quantize.quant_utils import (
    ActQuantWrapper, fake_quantize_nvfp4, fake_quantize_mxfp8,
)
from promix.quantize.weight_pack import pack_weight_mixed, split_k
from promix.quantize.mixed_linear import MixedQuantLinear

TARGET_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def _is_target(name):
    if ".layers." not in name:
        return None
    short = name.split(".")[-1]
    return short if short in TARGET_PROJS else None


def iter_target_wrappers(model):
    """Yield (name, short, wrapper) for every ActQuantWrapper that wraps one of
    the 7 target Linears inside a decoder layer (excludes lm_head)."""
    for name, mod in model.named_modules():
        if isinstance(mod, ActQuantWrapper):
            short = _is_target(name)
            if short is not None:
                yield name, short, mod


def _khigh_for(wrapper, K, high_fraction):
    """K_high for this Linear: prefer the value the quantizer was configured
    with (so the split matches the fake-quant baseline exactly); fall back to
    the global high_fraction split."""
    hbl = int(getattr(wrapper.quantizer, "high_bits_length", 0) or 0)
    if hbl > 0:
        K_low = K - hbl
        assert K_low % 16 == 0 and hbl % 32 == 0, (
            f"configured K_high={hbl} not block-aligned for K={K}")
        return K_low, hbl
    return split_k(K, high_fraction)


class FakeMixedLinear(nn.Module):
    """Fake-quant twin of MixedQuantLinear: fake_quantize the activation
    ([:K_low]=NVFP4, [K_low:]=MXFP8) and FP32-matmul it against the dequantized
    weight `W_deq` (the exact values the real kernel accumulates). Output +bias
    in `out_dtype`. Numerically identical to MixedQuantLinear up to GEMM
    accumulation order."""

    def __init__(self, W_deq, K_low, bias=None, out_dtype=torch.float16):
        super().__init__()
        self.K = W_deq.shape[1]
        self.K_low = int(K_low)
        self.N = W_deq.shape[0]
        self.out_dtype = out_dtype
        self.register_buffer("W_deq", W_deq.to(torch.float32).contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    def forward(self, x):
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1]).float()
        xq_lo = fake_quantize_nvfp4(x2d[:, :self.K_low])
        xq_hi = fake_quantize_mxfp8(x2d[:, self.K_low:])
        xq = torch.cat([xq_lo, xq_hi], dim=-1)
        D = (xq @ self.W_deq.t()).to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        return D.reshape(*in_shape[:-1], self.N)


def _frozen_scales_for(name, quantizers):
    """Look up GPTQ's frozen per-block weight scales for the target Linear at
    ActQuantWrapper module `name`.

    `gptq_fwrd` keys its returned `quantizers` dict by
    ``model.layers.{i}.{sub}.module`` (main/NVFP4 quantizer) and
    ``model.layers.{i}.{sub}.module,high_quantizer`` (high/MXFP8). The
    wrapper names yielded by `iter_target_wrappers` are exactly
    ``model.layers.{i}.{sub}`` (HF LlamaForCausalLM.named_modules), so the
    main key is simply ``name + ".module"``.

    Returns (nvfp4_scale, mxfp8_scale): each a (N, n_blocks) tensor frozen on
    the PRE-correction weight by `WeightQuantizer.find_params`, or None when
    the quantizers dict is absent or the entry/attribute is missing (RTN
    fallback).
    """
    if not quantizers:
        return None, None
    main_key = name + ".module"
    high_key = main_key + ",high_quantizer"
    nv_scale = mx_scale = None
    qm = quantizers.get(main_key)
    if qm is not None:
        nv_scale = getattr(qm, "frozen_scales", None)
    qh = quantizers.get(high_key)
    if qh is not None:
        mx_scale = getattr(qh, "frozen_scales", None)
    return nv_scale, mx_scale


def pack_all_targets(model, nv, mx, cd, high_fraction=0.125, verbose=True,
                     quantizers=None):
    """Pack every target Linear's (rotated+reordered) weight ONCE. Returns
    {name: PackedWeight}. Call before wiring; both wire_* consume this so the
    real kernel and the fake twin share byte-identical weight numerics.

    When GPTQ ran, its frozen per-block weight scales are used so the packed
    NVFP4/MXFP8 codes + SFB are built against GPTQ's FROZEN scale (the one its
    inner loop rounded the Hessian-corrected columns against) instead of a
    re-derived absmax (RTN) — this recovers GPTQ's full accuracy benefit.
    `quantizers` defaults to the dict stashed on the model by
    `run_gptq_if_enabled` (`model._gptq_quantizers`); pass an explicit dict to
    override, or leave None (and no stash) to use the RTN path.
    """
    if quantizers is None:
        quantizers = getattr(model, "_gptq_quantizers", None)
    nfrozen = 0
    packed = {}
    for name, short, wrapper in iter_target_wrappers(model):
        W = wrapper.module.weight.data
        K = W.shape[1]
        K_low, K_high = _khigh_for(wrapper, K, high_fraction)
        bias = getattr(wrapper.module, "bias", None)
        bias = None if bias is None else bias.data
        nv_scale, mx_scale = _frozen_scales_for(name, quantizers)
        if nv_scale is not None or mx_scale is not None:
            nfrozen += 1
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias,
                               frozen_nvfp4_scale=nv_scale,
                               frozen_mxfp8_scale=mx_scale)
        packed[name] = pk
        if verbose:
            print(f"  packed {name:48s} N={pk.N:5d} K={pk.K:5d} "
                  f"Kl={pk.K_low:5d} Kh={pk.K_high:4d}", flush=True)
    src = f"frozen-GPTQ-scale ({nfrozen}/{len(packed)})" if nfrozen else "RTN absmax"
    print(f"pack_all_targets: packed {len(packed)} Linears [{src}]", flush=True)
    return packed


def wire_real_kernel(model, packed, nv, mx, cd, qpf=None, dg=None, out_dtype=torch.float16):
    """Replace each target Linear's inner module with MixedQuantLinear (real
    packed fused kernel) and disable the wrapper's fake-quant input quantizer
    (MixedQuantLinear quantizes the activation itself). Keeps Hadamard / o_proj
    reorder / out_quantizer intact. When `dg` (decode_gemv_ext) is given, each
    MixedQuantLinear uses the memory-bound W4A4 GEMV at M==1 (decode). Returns
    the number of Linears wired."""
    n = 0
    for name, short, wrapper in iter_target_wrappers(model):
        pk = packed[name]
        mql = MixedQuantLinear(pk, nv, mx, cd, out_dtype=out_dtype, qpf=qpf, dg=dg)
        wrapper.module = mql
        wrapper.quantizer.bits = 16  # disable fake-quant; real quant is inside mql
        wrapper._inner_kwargs = set()  # MixedQuantLinear.forward(x) only
        # Prefill-fusion lever (down_proj online Hadamard only): fold 1/sqrt(n)
        # into the transform kernel (kills the separate /sqrt(n) div kernel).
        # VALIDATED numerics-safe: it is a constant fold (reciprocal-multiply vs
        # divide, <=1 ULP) -> whole-model prefill logit cosine 1.000000 vs the
        # pre-fusion path at M=2048/4096/8192 and wikitext-2 PPL 14.849 == the
        # pre-fusion 14.849 baseline (docs/prefill_fusion_validated_report.md).
        #
        # The other two prefill-fusion candidates were REVERTED here after the
        # numerics gate failed them (kept ONLY as reproducible negatives in the
        # prefill_fusion.py harness, never in this production path):
        #   - had_out_fp32 (feed FP32 Hadamard output straight to the qpf pack,
        #     skipping the .to(fp16) cast): changes the down_proj quant input
        #     precision -> logit cos 0.96-0.98, PPL 14.874 (+0.025). FAIL.
        #   - fp16-output GEMM epilogue (codesign_fused_fp16_ext.cu, cd16): the
        #     bf16->fp16 accumulator-store change -> logit cos 0.983, PPL 14.914
        #     (+0.064). FAIL. Production keeps the bf16-output `cd`.
        if getattr(wrapper, "online_full_had", False):
            wrapper.fold_had_scale = True
        n += 1
    print(f"wire_real_kernel: wired {n} Linears (qpf={'on' if qpf else 'off'}, "
          f"decode_gemv={'on' if dg else 'off'})", flush=True)
    return n


def wire_fake_mixed(model, packed, out_dtype=torch.float16):
    """Replace each target Linear's inner module with FakeMixedLinear (fake-quant
    twin, same W_deq as the real kernel) and disable the wrapper's input
    quantizer. Returns the number of Linears wired."""
    n = 0
    for name, short, wrapper in iter_target_wrappers(model):
        pk = packed[name]
        fml = FakeMixedLinear(pk.W_deq, pk.K_low, bias=pk.bias, out_dtype=out_dtype)
        wrapper.module = fml
        wrapper.quantizer.bits = 16  # disable fake-quant; FakeMixedLinear does its own
        wrapper._inner_kwargs = set()
        n += 1
    print(f"wire_fake_mixed: wired {n} Linears", flush=True)
    return n
