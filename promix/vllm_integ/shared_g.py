"""Shared-g re-quantization for vLLM's merged GEMMs (Phase 1, feasibility hard part #2).

vLLM's Llama hardcodes QKVParallelLinear / MergedColumnParallelLinear: q+k+v and
gate+up each run as ONE GEMM. Our fused kernel carries the round-6 two-level
per-tensor weight global ``g`` on the single epilogue alpha of that GEMM, so all
members of a merged group must share one ``g``. The E4M3/E8M0 SF bytes cannot be
rescaled post-hoc (the micro scales are on-grid by construction for the member's
OWN g), so the members are RE-QUANTIZED offline against a shared global:

    g_shared = max(g_member)               (keeps every micro <= its old value ->
                                            no E4M3 overflow; only bottom-end
                                            denormal clamping can occur)
    NVFP4 (per-16 E4M3 micro):  micro' = nearest-E4M3(eff / g_shared), clamped to
        the smallest positive E4M3; eff' = micro' * g_shared. eff is GPTQ's
        frozen effective block scale, so eff' deviates from the scale GPTQ
        rounded against only by the E4M3 re-rounding of the g-ratio.
    MXFP8 (per-32 E8M0 micro):  micro' = 2^ceil(log2(eff / g_shared) - tol)
        (round UP with a 2^-12 tolerance so an exact power of two is NOT bumped
        a bin by fp32 round-trip noise). Round-up preserves coverage (no new
        clipping); E4M3 elements rescale by powers of two exactly, so the high
        segment only loses sub-ulp at the denormal bottom.

This changes the quantized model's numerics slightly vs the per-projection-g
ProMix reference -> the shared-g model is RE-REFERENCED (export.py measures its
real-kernel PPL) and that number is the T2 gate target. down_proj / o_proj are
not merged and keep their own per-tensor g unchanged.
"""

import torch

from promix.quantize import quant_utils as qu

# vLLM merged-GEMM groups (checkpoint short-name -> members, in merge order).
MERGED_GROUPS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _e4m3_grid(device):
    return torch.tensor(qu._FP8_E4M3_POS, device=device, dtype=torch.float32)


def _e8m0_ceil_tol(x, tol_bits=12):
    """Smallest power of two >= x, with a 2^-tol_bits relative tolerance so a
    value that IS a power of two up to fp32 round-trip noise maps to itself."""
    e = torch.ceil(torch.log2(x.clamp_min(2.0 ** -127)) - 2.0 ** -tol_bits)
    return torch.exp2(e)


def rederive_frozen_scales_shared_g(eff_nv, eff_mx, g_shared):
    """Re-derive a member's frozen effective block scales against ``g_shared``.

    Args:
        eff_nv: (N, K_low//16) float32 frozen NVFP4 effective scales (g*micro),
                or None (K_low == 0 never happens for merged members).
        eff_mx: (N, K_high//32) float32 frozen MXFP8 effective scales, or None
                (K_high == 0).
        g_shared: float, the merged group's shared per-tensor global.

    Returns (eff_nv', eff_mx'): exactly decomposable as g_shared * on-grid micro
    (E4M3 / E8M0 respectively), so `pack_weight_mixed(frozen_scale=eff',
    global_scale=g_shared)` reproduces the micro bytes bit-exactly.
    """
    out_nv = out_mx = None
    if eff_nv is not None:
        e4 = _e4m3_grid(eff_nv.device)
        micro = qu._round_to_nearest_value(eff_nv.to(torch.float32) / g_shared, e4)
        micro = torch.clamp(micro, min=float(e4[e4 > 0].min()))
        out_nv = micro * g_shared
    if eff_mx is not None:
        micro = _e8m0_ceil_tol(eff_mx.to(torch.float32) / g_shared)
        out_mx = micro * g_shared
    return out_nv, out_mx


def shared_g_for_group(member_scales):
    """member_scales: list of (eff_nv, eff_mx, g) per member (g may be None when
    two-level is off). Returns (g_shared or None). Asserts consistency."""
    gs = [m[2] for m in member_scales]
    if all(g is None for g in gs):
        return None
    assert all(g is not None for g in gs), (
        "shared-g group has a mix of two-level and single-level members: "
        f"{gs}")
    return max(float(g) for g in gs)
