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
    the global high_fraction split only when it was never configured.

    A CONFIGURED value of 0 means a pure-NVFP4 Linear (0 MXFP8 rings, e.g. from
    rate-distortion allocation) and must be honored as 0 — not collapsed to the
    global 1/8 fallback, which would serve a 1/8 MXFP8 segment the evaluated
    model never had (silent served != evaluated mismatch)."""
    hbl_raw = getattr(wrapper.quantizer, "high_bits_length", None)
    if hbl_raw is not None:
        hbl = int(hbl_raw)
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

    # round-12: class-level toggle for the W4A16 PPL A/B. When ACT_EXACT is True
    # the activation is NOT fake-quantized (kept exact fp32) — the algorithm-level
    # equivalent of the W4A16 decode kernel (exact activation, quantized weight).
    # Default False reproduces the shipped W4A4 fake twin.
    ACT_EXACT = False

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
        if FakeMixedLinear.ACT_EXACT:
            xq = x2d                                  # W4A16: exact activation
        else:
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

    Returns (nvfp4_scale, mxfp8_scale, global_scale): the per-block frozen scales
    (each a (N, n_blocks) tensor, or None), plus the per-tensor two-level global
    `g` (round-6) stashed on the NVFP4 quantizer as `frozen_global` (None when
    two-level is off). `g` rides the fused-GEMM epilogue alpha so the SFB can store
    E4M3/E8M0 micro = eff/g.
    """
    if not quantizers:
        return None, None, None
    main_key = name + ".module"
    high_key = main_key + ",high_quantizer"
    nv_scale = mx_scale = g_scale = None
    qm = quantizers.get(main_key)
    if qm is not None:
        nv_scale = getattr(qm, "frozen_scales", None)
        g_scale = getattr(qm, "frozen_global", None)
    qh = quantizers.get(high_key)
    if qh is not None:
        mx_scale = getattr(qh, "frozen_scales", None)
    return nv_scale, mx_scale, g_scale


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
        nv_scale, mx_scale, g_scale = _frozen_scales_for(name, quantizers)
        if nv_scale is not None or mx_scale is not None:
            nfrozen += 1
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias,
                               frozen_nvfp4_scale=nv_scale,
                               frozen_mxfp8_scale=mx_scale,
                               global_scale=g_scale)
        packed[name] = pk
        if verbose:
            print(f"  packed {name:48s} N={pk.N:5d} K={pk.K:5d} "
                  f"Kl={pk.K_low:5d} Kh={pk.K_high:4d}", flush=True)
    src = f"frozen-GPTQ-scale ({nfrozen}/{len(packed)})" if nfrozen else "RTN absmax"
    print(f"pack_all_targets: packed {len(packed)} Linears [{src}]", flush=True)
    return packed


def _fused_mlp_forward(mlp, x):
    """LlamaMLP.forward replacement that hands the un-materialized gate/up GEMM
    outputs to the down_proj wrapper (fused silu*up -> Hadamard -> pack).  Falls
    back to the identical reference computation when fusion is disabled, so the
    fake-quant / pre-fusion paths are bit-identical to transformers' own."""
    g = mlp.gate_proj(x)
    u = mlp.up_proj(x)
    down = mlp.down_proj
    if getattr(down, "fused_swiglu_had", False) and \
            hasattr(down.module, "forward_swiglu_pack"):
        return down(x, mlp_gate=g, mlp_up=u)
    return down(mlp.act_fn(g) * u)


def _norm_pack_input(layer, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kwargs):
    """LlamaDecoderLayer.forward, with the fused norm->pack producer hook.

    This replacement changes NO arithmetic in the fallback: the two RMSNorms are
    the original modules, the residual adds and the attention/MLP calls are the
    reference's. It exists so the MLP's gate/up wrappers can receive the pre-norm
    tensor through `norm_residual` (round-26 lever B): with `norm_pack` enabled
    those wrappers replace add+RMSNorm+quant+pack with one fused kernel per group."""
    residual = hidden_states

    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (hidden_states,)

def wire_norm_pack_layers(model):
    """Install `_norm_pack_input` on every decoder layer whose Linears own a
    norm->pack group, and replace that layer's two RMSNorms with `_NormProducer`
    adapters. The adapters return the SAME tensor the reference RMSNorm returns
    (the layer forward's arithmetic is unchanged); they additionally run the
    fused RMSNorm + quantize + pack (round-26 lever B) once per norm and stash
    the packed activation in the group's shared context for the consumer Linears.

    NOTE on the residual add: Llama is a *pre-norm* decoder — the tensor fed to
    each RMSNorm IS the residual stream (`hidden_states = residual + attn_out`
    happened before the call), so the fused producer's "residual add" is the
    identity and the layer keeps the reference `residual + attn_out` itself.
    """
    import types
    n = 0
    for layer in model.model.layers:
        targets = [getattr(layer.self_attn, "q_proj", None),
                   getattr(layer.mlp, "gate_proj", None)]
        if not any(getattr(t, "norm_pack", False) for t in targets if t is not None):
            continue
        layer.forward = types.MethodType(_norm_pack_input, layer)
        for nm in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, nm)
            if not isinstance(mod, _NormProducer):
                setattr(layer, nm, _NormProducer(mod))
        n += 1
    print(f"wire_norm_pack_layers: installed on {n} layers", flush=True)
    return n


class _NormProducer(nn.Module):
    """RMSNorm adapter that ALSO runs the fused RMSNorm + quantize + pack producer
    (round-26 lever B) and publishes the packed activation to the consumer Linears
    of its norm->pack group.

    `forward(x)` returns exactly what the reference RMSNorm returns
    (`norm(x)`), so the decoder layer's arithmetic is unchanged. When the group's
    Linears are norm_pack-enabled and `arn` is present, it additionally packs the
    (identical) fp16 norm output once for the group: the packed CODES are shared
    and each consumer's SFA atom layout is derived from them
    (`norm_pack_for_n`), which removes the per-Linear quantize entirely and the
    duplicate packs for q/k/v and gate/up.
    """

    def __init__(self, norm):
        super().__init__()
        self.norm = norm
        self.producer = None     # _NormPackGroup or None

    def forward(self, hidden_states, **kwargs):
        x = self.norm(hidden_states)
        if self.producer is not None:
            self.producer.publish(hidden_states, x)
        return x

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.norm, name)


class _NormPackGroup:
    """Shared state for one (layer, norm) norm->pack group: the fused producer's
    result for the current forward (packed codes + per-N SFA layouts)."""

    def __init__(self, arn, N_gemm, K_low, K_high, eps):
        self.arn = arn
        self.N_gemm = int(N_gemm)
        self.K_low = int(K_low)
        self.K_high = int(K_high)
        self.eps = float(eps)
        self.res5 = None            # raw add_rmsnorm_pack_mixed result
        self.by_n = {}              # consumer N -> (code_lo, sfa_lo, code_hi, sfa_hi)

    def publish(self, x, normed):
        if self.arn is None:
            return
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        # to_add=None: Llama's norm input IS the residual stream (pre-norm), so
        # the fused producer's "residual add" is the identity.
        res = self.arn.add_rmsnorm_pack_mixed(
            x2d, None, self.N_gemm, self.K_low, self.K_high, self.eps)
        self.res5 = res[1:]
        self.by_n = {self.N_gemm: (res[1], res[2], res[3], res[4])}

    def get(self, N_gemm):
        if self.res5 is None:
            return None
        packed = self.by_n.get(int(N_gemm))
        if packed is None:
            packed = self.arn.norm_pack_for_n(self.res5, int(N_gemm),
                                               self.K_low, self.K_high)
            self.by_n[int(N_gemm)] = packed
        return packed

    def clear(self):
        self.res5 = None
        self.by_n = {}


def wire_fused_mlp(model):
    """Install `_fused_mlp_forward` on every decoder-layer MLP (no-op unless the
    down_proj wrapper has `fused_swiglu_had` set).  Returns the count."""
    import types
    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "act_fn"):
            continue
        mlp.forward = types.MethodType(_fused_mlp_forward, mlp)
        n += 1
    print(f"wire_fused_mlp: installed on {n} MLPs", flush=True)
    return n


def wire_real_kernel(model, packed, nv, mx, cd, qpf=None, dg=None, out_dtype=torch.float16, qpfm=None, hpk=None,
                     cd_bias=None, arn=None, norm_pack=False, hpk28=None, had28=None):
    """Replace each target Linear's inner module with MixedQuantLinear (real
    packed fused kernel) and disable the wrapper's fake-quant input quantizer
    (MixedQuantLinear quantizes the activation itself). Keeps Hadamard / o_proj
    reorder / out_quantizer intact. When `dg` (decode_gemv_ext) is given, each
    MixedQuantLinear uses the memory-bound W4A4 GEMV at M==1 (decode). When
    `qpfm` (quant_pack_fused_ext) is given, the per-forward activation pack uses
    the single-launch fused fp4+fp8 pack (lever 1; bit-exact). `cd_bias`
    (codesign_fused_bias_ext, round-26 lever A) folds the output cast + bias into
    the GEMM epilogue; `arn` (add_rmsnorm_pack_ext, round-26 lever B) is handed to
    each Linear for the fused norm->pack producer (enabled per wrapper by
    `wire_norm_pack`). Returns the number of Linears wired."""
    n = 0
    for name, short, wrapper in iter_target_wrappers(model):
        pk = packed[name]
        mql = MixedQuantLinear(pk, nv, mx, cd, out_dtype=out_dtype, qpf=qpf, dg=dg,
                               qpfm=qpfm, hpk=hpk, cd_bias=cd_bias, arn=arn,
                               hpk28=hpk28, had28=had28)
        # Lever A is only legal where the prefill GEMM runs (decode M=1 GEMV has
        # its own bias handling) — the decision is per forward, see MixedQuantLinear.
        mql.bias_in_epilogue = bool(cd_bias is not None)
        wrapper.module = mql
        wrapper.quantizer.bits = 16  # disable fake-quant; real quant is inside mql
        wrapper._inner_kwargs = set()  # MixedQuantLinear.forward(x) only
        # Fused down_proj producer path (SwiGLU + Hadamard + pack, one kernel):
        # the MLP hands the un-materialized gate/up GEMM outputs to this wrapper.
        # hpk handles the power-of-two intermediate (I=8192, 1B/3B); hpk28 handles
        # the non-power-of-two I=14336 (8B) via the 28x512 Hadamard decomposition.
        # (prefill only: with `dg` wired, M==1 takes the decode GEMV, whose
        # input is the materialized h — do not reroute it.)
        if ((hpk is not None or hpk28 is not None) and short == "down_proj" and dg is None
                and hasattr(mql, "forward_swiglu_pack")):
            wrapper.fused_swiglu_had = True
        # round-26 lever B: fused residual-add + RMSNorm + quant/pack producer.
        # Requires the prefill GEMM path (not decode: M==1 uses the GEMV whose
        # activation is packed inside the GEMV kernel) and no Hadamard (the
        # down_proj has its own fused producer above).
        # round-26 lever B is DEFAULT-OFF: the fused producer does not reproduce
        # the reference RMSNorm bit-exactly (reduction order) -> whole-model logit
        # cos 0.9889 < 0.99999 gate. Kept available behind `norm_pack=True` for
        # further work (see docs/edge_fusion_report.md).
        if (norm_pack and arn is not None and dg is None and not wrapper.online_full_had
                and short in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            wrapper.norm_pack = True
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
        #     (round-26 lever A revisits this WITH the bf16 round reproduced
        #     inside the epilogue + bias folded; see codesign_bias_epilogue.hpp.)
        if getattr(wrapper, "online_full_had", False):
            wrapper.fold_had_scale = True
        n += 1
    print(f"wire_real_kernel: wired {n} Linears (qpf={'on' if qpf else 'off'}, "
          f"qpfm={'on' if qpfm else 'off'}, decode_gemv={'on' if dg else 'off'}, "
          f"bias_epilogue={'on' if cd_bias else 'off'}, "
          f"norm_pack={'on' if arn else 'off'})", flush=True)
    if norm_pack and arn is not None and model is not None:
        wire_norm_pack(model)
    return n


NORM_PACK_GROUPS = (
    # group id,              member projections (all read the SAME norm output), norm module
    ("input",   ("q_proj", "k_proj", "v_proj"),   "input_layernorm"),
    ("postattn", ("gate_proj", "up_proj"),        "post_attention_layernorm"),
)


def wire_norm_pack(model):
    """Assign each norm->Linear group (q/k/v, gate/up) a shared producer context
    and attach it to the layer's norm adapter.

    One fused `add_rmsnorm_pack_mixed` call per (layer, norm) computes the
    RMSNorm and packs the activation ONCE for every member: the packed CODES are
    shared and each consumer's SFA atom layout is derived from them, replacing
    3 packs (q/k/v) + 2 packs (gate/up) with 2 packs per layer, and folding the
    ~6 elementwise RMSNorm kernels into the producer.
    """
    by_layer = {}
    for name, short, wrapper in iter_target_wrappers(model):
        # names look like "model.layers.7.self_attn.q_proj" / "...mlp.gate_proj"
        layer_idx = int(re.search(r"layers\.(\d+)\.", name).group(1))
        by_layer.setdefault(layer_idx, {})[short] = wrapper
    n_groups = 0
    for layer_idx, members in sorted(by_layer.items()):
        layer = model.model.layers[layer_idx]
        for gid, shorts, norm_name in NORM_PACK_GROUPS:
            present = [s for s in shorts if s in members
                       and getattr(members[s], "norm_pack", False)]
            if not present:
                continue
            wrappers = [members[s] for s in present]
            # K_low/K_high come from the Quantizer config (identical for the
            # group); N_gemm is only the SFA atom layout seed, any member's N
            # works because per-N layouts are derived on demand.
            w0 = wrappers[0]
            m0 = w0.module
            arn = getattr(m0, "_arn", None)
            if arn is None:
                continue
            grp = _NormPackGroup(arn, m0.N, m0.K_low, m0.K_high,
                                 float(getattr(model.config, "rms_norm_eps", 1e-5)))
            norm_mod = getattr(layer, norm_name)
            if not isinstance(norm_mod, _NormProducer):
                norm_mod = _NormProducer(norm_mod)
                setattr(layer, norm_name, norm_mod)
            norm_mod.producer = grp
            for w in wrappers:
                w._norm_pack_group_obj = grp
            n_groups += 1
    print(f"wire_norm_pack: {n_groups} shared norm->pack groups", flush=True)
    if n_groups:
        wire_norm_pack_layers(model)
    return n_groups


def wire_lmhead_mxfp8(model, lmh, qpf, out_dtype=torch.float16):
    """Replace the FP16 lm_head with a pure MXFP8 (W8A8) MxFp8Linear (lever 2).

    lm_head is the shared ~10 ms FP16 vocab GEMM. W8 MXFP8 holds wikitext-2 PPL
    at +0.048 (14.898 <= 14.95 gate); W4A4-mixed fails (+0.295). Returns the
    MxFp8Linear so the caller can restore the FP16 head if needed."""
    from promix.quantize.lmhead_linear import MxFp8Linear
    orig = model.lm_head
    model.lm_head = MxFp8Linear.from_linear(orig, lmh, qpf, out_dtype=out_dtype)
    print("wire_lmhead_mxfp8: lm_head -> MXFP8 W8A8", flush=True)
    return orig


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
