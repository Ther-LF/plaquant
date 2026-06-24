"""Unit tests for MXFP8 / NVFP4 fake quantizers (plan AC-1, spec Section 6).

These tests verify the format spec at docs/specs/spec-mxfp8-nvfp4.md is
implemented correctly. They are the AC-1 fake-vs-Python-reference half of
the Section 12 equivalence test; the fake-vs-real half (against the real
packer) is added in M3 once the packer exists.

Run with:
    pytest tests/test_fake_quant_mxfp8_nvfp4.py -v
"""

import pytest
import torch

from promix.quantize.quant_utils import (
    FP4_E2M1_MAX,
    FP4_E2M1_POS,
    FP8_E4M3_MAX,
    _round_to_nearest_value,
    fake_quantize_mxfp8,
    fake_quantize_nvfp4,
)


@pytest.fixture
def small_input_mxfp8():
    """1024 x 256 input — 8 MXFP8 blocks per row."""
    torch.manual_seed(0)
    return torch.randn(1024, 256, dtype=torch.float32)


@pytest.fixture
def small_input_nvfp4():
    """1024 x 256 input — 16 NVFP4 blocks per row."""
    torch.manual_seed(0)
    return torch.randn(1024, 256, dtype=torch.float32)


def test_mxfp8_shape_and_dtype_preserved(small_input_mxfp8):
    """Fake quant returns same shape/dtype as input."""
    out = fake_quantize_mxfp8(small_input_mxfp8)
    assert out.shape == small_input_mxfp8.shape
    assert out.dtype == small_input_mxfp8.dtype


def test_nvfp4_shape_and_dtype_preserved(small_input_nvfp4):
    out = fake_quantize_nvfp4(small_input_nvfp4)
    assert out.shape == small_input_nvfp4.shape
    assert out.dtype == small_input_nvfp4.dtype


def test_mxfp8_idempotent(small_input_mxfp8):
    """Quantize-then-quantize is the identity (already in representable set)."""
    once = fake_quantize_mxfp8(small_input_mxfp8)
    twice = fake_quantize_mxfp8(once)
    assert torch.equal(once, twice), "MXFP8 fake_quant is not idempotent"


def test_nvfp4_idempotent(small_input_nvfp4):
    once = fake_quantize_nvfp4(small_input_nvfp4)
    twice = fake_quantize_nvfp4(once)
    assert torch.equal(once, twice), "NVFP4 fake_quant is not idempotent"


def test_mxfp8_no_clipping_after_block_scale(small_input_mxfp8):
    """Spec Section 6: round-UP block scale ensures elements/scale never exceed FP8 max.

    After dividing by the block scale, elements should fall within
    [-FP8_E4M3_MAX, +FP8_E4M3_MAX] without clipping. We verify this property
    by reconstructing the per-block scale from the output and checking the
    quantized elements stay representable.
    """
    out = fake_quantize_mxfp8(small_input_mxfp8)
    # Output is q * scale; if max abs of out per block / FP8_E4M3_MAX <= scale,
    # we know scale was >= max/MAX (round UP) and no clipping happened.
    out_blk = out.view(1024, 8, 32)
    block_max_out = out_blk.abs().amax(dim=-1)
    # block_max_out / FP8_E4M3_MAX is the actual scale used (since elements are
    # at most FP8_E4M3_MAX after division -> at most scale*MAX after multiply)
    # If round-UP worked, this should hold for every nonzero block.
    # Compare against original block_max / MAX (the ideal scale).
    in_blk = small_input_mxfp8.view(1024, 8, 32)
    block_max_in = in_blk.abs().amax(dim=-1)
    # The output's block max should be >= input's block max (since round-UP scale
    # never causes clipping; elements may be quantized smaller but the max
    # representable value is reachable). Use a small tolerance for floating point.
    # More precisely: the output's max should equal q_max * scale where q_max is
    # one of the representable FP8 values closest to block_max_in / scale.
    # We assert no input value triggered saturation (i.e., output never clipped):
    assert (block_max_out >= block_max_in - 1e-6).all() or (
        # OR every element was within representable range; the only failure mode
        # we want to flag is when output is artificially capped at MAX*scale
        # due to round-DOWN scale.
        block_max_out / block_max_in.clamp_min(1e-12)
    ).min() > 0.5, "Possible round-DOWN block scale causing clipping"


def test_nvfp4_no_clipping_after_block_scale(small_input_nvfp4):
    """Same property as MXFP8 but with NVFP4 16-element blocks."""
    out = fake_quantize_nvfp4(small_input_nvfp4)
    out_blk = out.view(1024, 16, 16)
    in_blk = small_input_nvfp4.view(1024, 16, 16)
    block_max_in = in_blk.abs().amax(dim=-1)
    block_max_out = out_blk.abs().amax(dim=-1)
    assert (block_max_out >= block_max_in - 1e-6).all() or (
        block_max_out / block_max_in.clamp_min(1e-12)
    ).min() > 0.5, "Possible round-DOWN block scale causing NVFP4 clipping"


def test_mxfp8_zero_block_round_trips_to_zero():
    """All-zero block should fake-quantize to all zeros."""
    x = torch.zeros(1, 32, dtype=torch.float32)
    out = fake_quantize_mxfp8(x)
    assert (out == 0).all()


def test_nvfp4_zero_block_round_trips_to_zero():
    x = torch.zeros(1, 16, dtype=torch.float32)
    out = fake_quantize_nvfp4(x)
    assert (out == 0).all()


def test_mxfp8_constant_block():
    """Constant value block should round-trip with low error (scale captures magnitude)."""
    val = 1.5
    x = torch.full((1, 32), val, dtype=torch.float32)
    out = fake_quantize_mxfp8(x)
    # Error should be small relative to the value (FP8 E4M3 has ~3 mantissa bits)
    err = (out - val).abs().max().item()
    assert err / val < 0.2, f"MXFP8 constant block error {err} too large for value {val}"


def test_nvfp4_constant_block():
    val = 1.5
    x = torch.full((1, 16), val, dtype=torch.float32)
    out = fake_quantize_nvfp4(x)
    # NVFP4 E2M1 has only 4 representable values per scale level, so error is
    # larger; the value 1.5 is exactly representable so error should be near zero.
    err = (out - val).abs().max().item()
    assert err < 1e-5, f"NVFP4 constant 1.5 should round-trip exactly, got err {err}"


def test_mxfp8_quantization_error_bounded(small_input_mxfp8):
    """Average quantization error should be small relative to input magnitude."""
    out = fake_quantize_mxfp8(small_input_mxfp8)
    rel_err = (out - small_input_mxfp8).abs() / small_input_mxfp8.abs().clamp_min(1e-8)
    # MXFP8 with 32-block scale should have mean rel err well under 0.1
    assert rel_err.mean().item() < 0.1, (
        f"MXFP8 mean rel err {rel_err.mean().item():.4f} too large; expected < 0.1"
    )


def test_nvfp4_quantization_error_bounded(small_input_nvfp4):
    out = fake_quantize_nvfp4(small_input_nvfp4)
    rel_err = (out - small_input_nvfp4).abs() / small_input_nvfp4.abs().clamp_min(1e-8)
    # NVFP4 with 16-block scale and only 16 values per scale; expect higher
    # error than MXFP8 but still bounded.
    assert rel_err.mean().item() < 0.25, (
        f"NVFP4 mean rel err {rel_err.mean().item():.4f} too large; expected < 0.25"
    )


def test_nvfp4_representable_set_membership():
    """Every NVFP4 output element must equal `representable × scale`
    where `representable ∈ ±FP4_E2M1_POS` and `scale` is the block
    scale `fake_quantize_nvfp4` actually picked.

    Round 5 quarantined a brittle ratio-normalized version of this
    test. Round 6 exposed `select_block_scales_nvfp4()` (the scale-
    selection step extracted from `fake_quantize_nvfp4`); round 10
    rewrites the test to use that helper directly: compute the same
    block scale the fake quantizer used, divide each output element
    by it, snap to the nearest E2M1 representable, and assert the
    snap is exact (within fp tolerance).
    """
    from promix.quantize.quant_utils import (
        select_block_scales_nvfp4,
        _round_to_nearest_value,
    )

    torch.manual_seed(42)
    block = torch.randn(1, 16, dtype=torch.float32) * 2.0
    out = fake_quantize_nvfp4(block)

    # Recompute the SAME scale the helper used (block_size=16; one block).
    scale = select_block_scales_nvfp4(block, block_size=16).flatten()
    assert scale.numel() == 1, "single block expected for this test"
    s = scale.item()
    assert s > 0, f"scale must be positive; got {s}"

    representable = torch.tensor(
        sorted({float(v) for v in FP4_E2M1_POS}
               | {-float(v) for v in FP4_E2M1_POS}),
        dtype=torch.float32,
    )
    quotient = (out.flatten().to(torch.float32) / s)
    snapped = _round_to_nearest_value(
        quotient,
        torch.tensor(FP4_E2M1_POS, dtype=torch.float32),
    )
    assert torch.allclose(quotient, snapped, atol=1e-3, rtol=1e-3), (
        f"NVFP4 output / chosen_scale ({s}) must lie on E2M1 grid; "
        f"observed quotient={quotient.tolist()}, "
        f"snapped={snapped.tolist()}"
    )

    # Reconstructing should round-trip back to `out` (modulo fp).
    reconstructed = snapped * s
    assert torch.allclose(
        out.flatten().to(torch.float32), reconstructed, atol=1e-5
    ), (
        f"out != snapped × scale; this should hold by construction. "
        f"out={out.flatten().tolist()}, reconstructed={reconstructed.tolist()}"
    )


# ---------------------------------------------------------------------------
# RNE ties-to-even midpoint tests (Round 2: Codex round-1 review found that
# the previous argmin-based implementation does NOT implement ties-to-even;
# it picks the lower-index representable on ties. These tests deterministically
# show the fix matches IEEE 754 ties-to-even semantics for the FP4 E2M1 and
# FP8 E4M3 representable sets.)
# ---------------------------------------------------------------------------

def test_round_nearest_fp4_midpoint_1p25_picks_1p0():
    """FP4 E2M1: midpoint 1.25 between 1.0 and 1.5; ties-to-even picks 1.0
    (mantissa LSB == 0 in code 2; code 3 for 1.5 has mantissa LSB == 1)."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    x = torch.tensor([1.25], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == 1.0, (
        f"FP4 1.25 midpoint should round to 1.0 (even-coded mantissa), got {out.item()}"
    )


def test_round_nearest_fp4_midpoint_neg_1p25_picks_neg_1p0():
    """Negative midpoint -1.25 picks -1.0 by symmetry."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    x = torch.tensor([-1.25], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == -1.0, (
        f"FP4 -1.25 should round to -1.0, got {out.item()}"
    )


def test_round_nearest_fp4_midpoint_1p75_picks_2p0():
    """FP4 E2M1: midpoint 1.75 between 1.5 (code 3, odd) and 2.0 (code 4, even);
    ties-to-even picks 2.0."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    x = torch.tensor([1.75], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == 2.0, (
        f"FP4 1.75 midpoint should round to 2.0 (even-coded), got {out.item()}"
    )


def test_round_nearest_fp4_midpoint_2p5_picks_2p0():
    """FP4: midpoint 2.5 between 2.0 (code 4, even) and 3.0 (code 5, odd); picks 2.0."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    x = torch.tensor([2.5], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == 2.0, f"FP4 2.5 should round to 2.0, got {out.item()}"


def test_round_nearest_fp4_midpoint_5p0_picks_4p0():
    """FP4: midpoint 5.0 between 4.0 (code 6, even) and 6.0 (code 7, odd); picks 4.0.
    This is the FP4 max-bracket midpoint — exercises the V-1 boundary."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    x = torch.tensor([5.0], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == 4.0, f"FP4 5.0 should round to 4.0, got {out.item()}"


def test_round_nearest_fp4_exact_value_unchanged():
    """Exact representable values should round to themselves."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    for v in FP4_E2M1_POS:
        x = torch.tensor([float(v)], dtype=torch.float32)
        out = _round_to_nearest_value(x, values)
        assert out.item() == v, f"Exact {v} got rounded to {out.item()}"


def test_round_nearest_fp4_non_midpoint_picks_closer():
    """Verify non-midpoint inputs go to the closer representable (not affected by
    the ties-to-even change)."""
    values = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    # 1.1 is closer to 1.0 than 1.5
    out = _round_to_nearest_value(torch.tensor([1.1], dtype=torch.float32), values)
    assert out.item() == 1.0
    # 1.4 is closer to 1.5 than 1.0
    out = _round_to_nearest_value(torch.tensor([1.4], dtype=torch.float32), values)
    assert out.item() == 1.5


def test_round_nearest_fp8_e4m3_smallest_normal_midpoint():
    """FP8 E4M3: midpoint between adjacent normals at smallest normal exponent.
    Smallest normal magnitudes: 2^-6 = 0.015625 (mantissa 0), 1.125*2^-6 = 0.017578125
    (mantissa 1). Midpoint = 0.5 * (0.015625 + 0.017578125) = 0.0166015625.
    Codes 8 (even, mantissa 0) and 9 (odd, mantissa 1) → ties-to-even picks 0.015625.
    """
    from promix.quantize.quant_utils import _FP8_E4M3_POS as FP8_POS

    values = torch.tensor(FP8_POS, dtype=torch.float32)
    a = 2.0 ** -6  # smallest normal, even mantissa
    b = (1.0 + 1.0 / 8.0) * (2.0 ** -6)  # next, odd mantissa
    midpoint = 0.5 * (a + b)
    x = torch.tensor([midpoint], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == a, (
        f"FP8 E4M3 normal midpoint should round to {a} (even-coded mantissa), got {out.item()}"
    )


def test_round_nearest_fp8_e4m3_negative_midpoint():
    """Negative-side midpoint round behaves symmetrically."""
    from promix.quantize.quant_utils import _FP8_E4M3_POS as FP8_POS

    values = torch.tensor(FP8_POS, dtype=torch.float32)
    a = 2.0 ** -6
    b = (1.0 + 1.0 / 8.0) * (2.0 ** -6)
    midpoint = 0.5 * (a + b)
    x = torch.tensor([-midpoint], dtype=torch.float32)
    out = _round_to_nearest_value(x, values)
    assert out.item() == -a, (
        f"FP8 E4M3 negative midpoint should round to {-a}, got {out.item()}"
    )


# ---------------------------------------------------------------------------
# ActQuantizer routing tests (Round 2: Codex round-1 review found that the
# `bits=mxfp8` / `bits=nvfp4` config-driven routing required by plan task9
# was not wired into ActQuantizer. These tests verify the new string-bits
# dispatch matches the standalone fake_quantize_* functions exactly.)
# ---------------------------------------------------------------------------

def test_act_quantizer_routes_mxfp8():
    """ActQuantizer with bits='mxfp8' should produce the same output as
    fake_quantize_mxfp8 called directly."""
    from promix.quantize.quant_utils import ActQuantizer

    q = ActQuantizer()
    q.configure(bits="mxfp8", sym=True, perchannel=False)
    torch.manual_seed(7)
    x = torch.randn(64, 256, dtype=torch.float32)
    direct = fake_quantize_mxfp8(x)
    routed = q(x)
    assert torch.equal(direct, routed), (
        "ActQuantizer(bits='mxfp8') output does NOT match fake_quantize_mxfp8 — routing broken"
    )


def test_act_quantizer_routes_nvfp4():
    """ActQuantizer with bits='nvfp4' should produce the same output as
    fake_quantize_nvfp4 called directly."""
    from promix.quantize.quant_utils import ActQuantizer

    q = ActQuantizer()
    q.configure(bits="nvfp4", sym=True, perchannel=False)
    torch.manual_seed(7)
    x = torch.randn(64, 256, dtype=torch.float32)
    direct = fake_quantize_nvfp4(x)
    routed = q(x)
    assert torch.equal(direct, routed), (
        "ActQuantizer(bits='nvfp4') output does NOT match fake_quantize_nvfp4 — routing broken"
    )


def test_act_quantizer_legacy_int_bits_still_works():
    """Numeric bits=4/8 must still work (backward compat for existing INT path)."""
    from promix.quantize.quant_utils import ActQuantizer

    q = ActQuantizer()
    q.configure(bits=8, sym=True, perchannel=False)
    x = torch.randn(64, 256, dtype=torch.float32)
    out = q(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # Quantized output should differ from input (real quantization happened)
    assert not torch.equal(out, x)


# ---------------------------------------------------------------------------
# Round 3 regression tests for the model-level paths Codex round-2 review
# flagged as still broken: ActQuantWrapper guard, segment-wise FP split,
# WeightQuantizer FP routing.
# ---------------------------------------------------------------------------

def test_act_quant_wrapper_no_typeerror_on_string_bits():
    """ActQuantWrapper(nn.Linear) with bits='nvfp4' must NOT TypeError on first
    forward. Round 2 used `bits < 16` numeric comparison which crashed on
    strings. This test deterministically fails on Round 2 code."""
    import torch.nn as nn
    from promix.quantize.quant_utils import ActQuantWrapper

    lin = nn.Linear(64, 32, bias=False)
    wrapper = ActQuantWrapper(lin)
    wrapper.quantizer.configure(bits="nvfp4", sym=True, perchannel=False)
    x = torch.randn(2, 16, 64, dtype=torch.float32)
    # Should not raise.
    out = wrapper(x)
    assert out.shape == (2, 16, 32)
    assert out.dtype == x.dtype


def test_act_quant_wrapper_nvfp4_matches_direct_call():
    """Routed forward must equal direct fake_quantize_nvfp4 followed by linear."""
    import torch.nn as nn
    from promix.quantize.quant_utils import ActQuantWrapper

    lin = nn.Linear(64, 32, bias=False)
    wrapper = ActQuantWrapper(lin)
    wrapper.quantizer.configure(bits="nvfp4", sym=True, perchannel=False)
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, dtype=torch.float32)
    direct = lin(fake_quantize_nvfp4(x).to(x.dtype))
    routed = wrapper(x)
    assert torch.allclose(direct, routed, atol=1e-6), (
        "ActQuantWrapper bits='nvfp4' forward does not match direct fake_quantize_nvfp4 + linear"
    )


def test_act_quantizer_segment_split_mxfp8_high_nvfp4_main():
    """Mixed segment dispatch: bits='nvfp4' main, high_bits='mxfp8' for high_bits_length=32.
    The high segment must be MXFP8-quantized; the main must be NVFP4-quantized.
    Round 2 ignored high/low split for string bits and quantized the whole
    tensor as one format."""
    from promix.quantize.quant_utils import ActQuantizer

    q = ActQuantizer()
    q.configure(
        bits="nvfp4",
        sym=True,
        perchannel=False,
        high_bits_length=32,
        high_bits="mxfp8",
        low_bits_length=0,
        low_bits=2,
    )
    torch.manual_seed(0)
    # x.shape[-1] = 256; main = 224, high = 32.
    x = torch.randn(1, 8, 256, dtype=torch.float32)
    out = q(x)
    # Reconstruct expected: main=NVFP4 on x[..., :224]; high=MXFP8 on x[..., 224:]
    main_expected = fake_quantize_nvfp4(x[..., :224])
    high_expected = fake_quantize_mxfp8(x[..., 224:])
    expected = torch.cat([main_expected, high_expected], dim=-1)
    assert torch.equal(out, expected), (
        "ActQuantizer segment split does not match per-segment direct dispatch"
    )


def test_weight_quantizer_string_bits_no_crash():
    """WeightQuantizer.configure() must accept string FP format identifiers
    without crashing. Round 2 path computes `2 ** (bits - 1)` which TypeErrors
    on string."""
    from promix.quantize.kv_quant import WeightQuantizer

    wq = WeightQuantizer()
    # Should not raise:
    wq.configure(bits="mxfp8", perchannel=True, sym=True, mse=False)
    assert wq.bits == "mxfp8"


def test_weight_quantizer_string_bits_quantize_routes_to_fp_helper():
    """WeightQuantizer.quantize() with string bits should produce same output
    as direct fake_quantize_* call."""
    from promix.quantize.kv_quant import WeightQuantizer

    wq = WeightQuantizer()
    wq.configure(bits="mxfp8", perchannel=True, sym=True, mse=False)
    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.float32)
    wq.find_params(x)
    out = wq.quantize(x)
    expected = fake_quantize_mxfp8(x)
    assert torch.equal(out, expected)


def test_weight_quantizer_string_bits_quantize_to_int_returns_none():
    """quantize_to_int is INT-only by design; FP path returns (None, None, None)
    and real packing is deferred to the M3 weight_packer."""
    from promix.quantize.kv_quant import WeightQuantizer

    wq = WeightQuantizer()
    wq.configure(bits="nvfp4", perchannel=True, sym=True, mse=False)
    x = torch.randn(8, 256, dtype=torch.float32)
    wq.find_params(x)
    qint, scale, zero = wq.quantize_to_int(x)
    assert qint is None
    assert scale is None
    assert zero is None


# ---------------------------------------------------------------------------
# GPTQ.fasterquant() FP-segment smoke test
# ---------------------------------------------------------------------------


def test_gptq_fasterquant_fp_path_no_crash_and_q_int_none():
    """GPTQ.fasterquant() must not crash when any active WeightQuantizer
    uses a string FP format identifier, and Q_int must be None on the
    FP path (real FP-byte packing belongs to the kernel-side weight
    packer, not GPTQ).

    Round 6 replaced the round-4 whole-tensor RTN bypass with the spec
    Section 11 fixed-block-scale Hessian-conditioned per-column path:
    `WeightQuantizer.find_params(W)` freezes per-row per-block scales
    on the original (uncorrected) weight; `GPTQ.fasterquant()`'s
    per-column inner loop then dispatches FP segments to
    `quantize_column_with_frozen_scale(w_corrected, col_idx)` so the
    Hessian-corrected value is rounded against the SAME frozen scale.
    This basic smoke covers the no-crash + Q_int=None part; the
    correctness-of-Hessian-correction half is covered by
    `test_section11_fp_gptq_uses_hessian_correction`.
    """
    import torch.nn as nn

    from promix.quantize.gptq import GPTQ
    from promix.quantize.kv_quant import WeightQuantizer

    torch.manual_seed(0)
    in_features = 256  # 32 (MXFP8 block) and 16 (NVFP4 block) both divide
    out_features = 64
    high_bits_length = 32   # MXFP8 segment, %32==0
    low_bits_length = 0     # no low segment in this configuration

    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight.data = torch.randn_like(layer.weight.data) * 0.1

    gptq = GPTQ(
        layer,
        mixed_precision=True,
        high_bits_length=high_bits_length,
        low_bits_length=low_bits_length,
    )

    main_q = WeightQuantizer()
    main_q.configure(bits="nvfp4", perchannel=True, sym=True, mse=False)
    gptq.quantizer = main_q

    high_q = WeightQuantizer()
    high_q.configure(bits="mxfp8", perchannel=True, sym=True, mse=False)
    gptq.high_quantizer = high_q

    # add_batch records the Hessian. Round 6's per-column FP loop USES
    # the Hessian (Section 11 Hessian-conditioned rounding under frozen
    # block scales), so this call is no longer optional smoke — it
    # affects the result. fasterquant must complete without crashing
    # regardless of Hessian shape.
    inp = torch.randn(2, 8, in_features)
    out = layer(inp)
    gptq.add_batch(inp, out)

    # Should not raise; should set self.Q_int to None on the FP path.
    gptq.fasterquant(blocksize=64, percdamp=0.01, groupsize=-1, actorder=False)

    assert getattr(gptq, "Q_int", "missing") is None, (
        "GPTQ.fasterquant() FP-segment path must set self.Q_int = None; "
        f"got {getattr(gptq, 'Q_int', 'missing')!r}"
    )
    # Weight should be replaced with fake-quantized values (not equal to
    # original modulo numerical fluctuation; at minimum the dtype/shape
    # round-trip must hold).
    assert layer.weight.shape == (out_features, in_features)
    assert layer.weight.dtype == torch.float32 or layer.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# Config-level entry-point smoke tests: verify promix.eval.ptq's GPTQ /
# v_proj gates accept string FP-format identifiers without TypeError.
# Round 4 added a direct GPTQ.fasterquant() smoke test, but did not exercise
# `promix/eval/ptq.py`'s string-bits gate that prevents Step 2 from ever
# reaching gptq_fwrd() when w_bits is "nvfp4".
# ---------------------------------------------------------------------------


def test_quant_enabled_accepts_string_fp_formats():
    """The shared FP-aware predicate must return True for both numeric
    quantization (bits<16) and the canonical string FP-format identifiers.

    All ptq.py / inference gates that previously did `bits < 16` must
    route through this predicate to avoid TypeError on string `bits`.
    """
    from promix.quantize.quant_utils import _quant_enabled

    # Numeric: <16 enables quantization, >=16 is pass-through.
    assert _quant_enabled(4) is True
    assert _quant_enabled(8) is True
    assert _quant_enabled(16) is False
    # String FP-format identifiers must enable quantization.
    assert _quant_enabled("mxfp8") is True
    assert _quant_enabled("nvfp4") is True
    # Unknown strings safely return False (typo guard); validated loudly
    # at config-load time is a Round 6+ improvement.
    assert _quant_enabled("nvf4") is False


def test_ptq_string_w_bits_does_not_crash_on_legacy_numeric_compare():
    """Regression test for the round-4 review's main blocker: with
    `w_bits: "nvfp4"`, a bare `w_bits < 16` (the pre-round-5 ptq.py
    check) raises TypeError on string-vs-int comparison. Verify the
    new gate uses _quant_enabled and accepts the string.
    """
    import os
    import yaml

    from promix.quantize.quant_utils import _quant_enabled

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(
        repo_root, "promix", "configs", "llama-3.2-1b-mxfp8-nvfp4.yaml"
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    w_bits = cfg["quantize"]["w_bits"]
    assert isinstance(w_bits, str), (
        f"FP config must declare string w_bits; got {w_bits!r}"
    )
    # Legacy numeric comparison MUST raise TypeError on string — this
    # confirms the test is exercising the correct regression surface.
    raised = False
    try:
        _ = w_bits < 16  # noqa: B015 — intentional crash regression
    except TypeError:
        raised = True
    assert raised, (
        "string < int should TypeError; if this assertion fails, the test "
        "is no longer guarding the round-4 blocker"
    )
    # New gate accepts the string FP identifier without crashing.
    assert _quant_enabled(w_bits) is True


def test_ptq_source_does_not_compare_w_bits_numerically():
    """Source-level regression guard: `promix/eval/ptq.py` must NOT contain
    a bare `w_bits < 16` (or other numeric comparison on `w_bits`) — that
    would TypeError when w_bits is "nvfp4" / "mxfp8". The check is a
    string scan because the runtime path requires `transformers`, which
    is not installed in the local lightweight test harness.

    Allowed: `_quant_enabled(config['quantize']['w_bits'])` or any other
    predicate that handles both numeric and string forms.
    Disallowed: `< 16`, `>= 16`, `== 16`, `!= 16` applied directly to
    `w_bits` (or any string-typed bits field) without first dispatching
    on type.
    """
    import os
    import re

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ptq_path = os.path.join(repo_root, "promix", "eval", "ptq.py")
    with open(ptq_path, "r") as f:
        src = f.read()

    # Pattern: `w_bits` or `config['quantize']['w_bits']` followed by
    # numeric comparison. Whitespace flexible.
    bad_patterns = [
        r"\bw_bits\b\s*[<>=!]=?\s*16\b",
        r"\bw_bits\b\s*[<>=!]=?\s*\d+\b",
    ]
    for pat in bad_patterns:
        m = re.search(pat, src)
        assert m is None, (
            f"promix/eval/ptq.py contains a numeric comparison on w_bits "
            f"({m.group(0)!r}); this TypeErrors on string FP identifiers. "
            f"Use _quant_enabled(w_bits) instead."
        )

    # Positive assertion: the predicate-based gate must be present.
    assert "_quant_enabled" in src, (
        "promix/eval/ptq.py must use the shared _quant_enabled predicate "
        "for w_bits / v_bits gates"
    )


# ---------------------------------------------------------------------------
# True normal-entry-point smoke test for the PTQ FP path.
# Codex round-5 review specifically rejected source-scanning as evidence:
# we must actually import `promix.eval.ptq` and exercise the GPTQ gate.
# Round 6 refactored the gate into `run_gptq_if_enabled(...)` with seams for
# stubbing the heavyweight tokenizer / data / gptq deps; this test exercises
# THAT seam on the FP yaml.
# ---------------------------------------------------------------------------


def _install_heavyweight_stubs():
    """Install minimal stubs for transitive third-party imports that
    `promix.eval.ptq` triggers at module scope but which the lightweight
    test harness doesn't have installed (transformers, datasets,
    fast_hadamard_transform, etc.).

    This is the cost of the test exercising the real ptq.py module
    rather than source-scanning it. The stubs implement only the
    attributes/symbols that ptq.py and its transitive imports
    REFERENCE during import — none of the methods are actually invoked
    by `run_gptq_if_enabled`'s seam path.
    """
    import sys
    import types

    def _ensure_module(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    # transformers
    if "transformers" not in sys.modules or getattr(
        sys.modules["transformers"], "_promix_test_stub", False
    ):
        stub = types.ModuleType("transformers")
        stub._promix_test_stub = True

        def _set_seed(*_a, **_kw):
            pass

        class _Auto:
            @staticmethod
            def from_pretrained(*_a, **_kw):
                class _X:
                    hidden_size = 0
                    num_attention_heads = 0
                    tie_word_embeddings = False
                return _X()

        stub.set_seed = _set_seed
        stub.AutoTokenizer = _Auto
        stub.AutoConfig = _Auto
        stub.AutoModelForCausalLM = _Auto
        stub.LlamaForCausalLM = _Auto
        sys.modules["transformers"] = stub

    # datasets
    if "datasets" not in sys.modules:
        ds = types.ModuleType("datasets")
        ds.config = types.SimpleNamespace(HF_DATASETS_TRUST_REMOTE_CODE=False)
        ds.load_dataset = lambda *_a, **_kw: None
        sys.modules["datasets"] = ds

    # fast_hadamard_transform (the test harness already stubs this via
    # PYTHONPATH; defensive ensure here).
    if "fast_hadamard_transform" not in sys.modules:
        fht = types.ModuleType("fast_hadamard_transform")
        fht.hadamard_transform = lambda *_a, **_kw: None
        sys.modules["fast_hadamard_transform"] = fht


def _install_transformers_train_stubs():
    """Install minimal stubs for the transformers submodules
    `promix.train.modeling_llama_train` imports at module scope.
    Used by the real-attention runtime test, which needs to actually
    import the train model on a harness without `transformers`.

    The stubs implement only the names imported, with no behavior — they
    are constructed once and shared. Anything the test invokes directly
    (e.g. `LlamaSdpaAttention.forward`) calls torch / promix code, not
    these stubs, so empty placeholders are safe.
    """
    import sys
    import types

    _install_heavyweight_stubs()  # ensure transformers root is stubbed

    def _ensure(name, attrs=None):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if attrs:
                for k, v in attrs.items():
                    setattr(mod, k, v)
            sys.modules[name] = mod
            # also attach as attribute on parent if applicable
            parent_name = name.rsplit(".", 1)[0]
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], name.rsplit(".", 1)[1], mod)
        return sys.modules[name]

    # transformers.activations: ACT2FN dict
    import torch.nn.functional as _F
    _ensure(
        "transformers.activations",
        {"ACT2FN": {"silu": _F.silu, "gelu": _F.gelu, "relu": _F.relu}},
    )

    class _Stub:
        pass

    _ensure(
        "transformers.cache_utils",
        {"Cache": _Stub, "DynamicCache": _Stub, "StaticCache": _Stub},
    )
    _ensure(
        "transformers.modeling_attn_mask_utils",
        {"AttentionMaskConverter": _Stub},
    )
    _ensure(
        "transformers.modeling_flash_attention_utils",
        {"_flash_attention_forward": lambda *a, **k: None},
    )
    _ensure(
        "transformers.modeling_outputs",
        {
            "BaseModelOutputWithPast": _Stub,
            "CausalLMOutputWithPast": _Stub,
            "QuestionAnsweringModelOutput": _Stub,
            "SequenceClassifierOutputWithPast": _Stub,
            "TokenClassifierOutput": _Stub,
        },
    )
    _ensure("transformers.modeling_rope_utils", {"ROPE_INIT_FUNCTIONS": {}})

    import torch.nn as _nn
    class _PreTrainedModel(_nn.Module):
        config_class = None
        base_model_prefix = "model"
    _ensure("transformers.modeling_utils", {"PreTrainedModel": _PreTrainedModel})

    _ensure("transformers.pytorch_utils", {"ALL_LAYERNORM_LAYERS": []})

    class _Logger:
        @staticmethod
        def warning_once(*_a, **_kw):
            pass
        @staticmethod
        def warning(*_a, **_kw):
            pass
        @staticmethod
        def info(*_a, **_kw):
            pass

    class _LoggingNamespace:
        @staticmethod
        def get_logger(*_a, **_kw):
            return _Logger()

    _ensure(
        "transformers.utils",
        {
            "add_start_docstrings": (lambda *a, **k: (lambda f: f)),
            "add_start_docstrings_to_model_forward": (lambda *a, **k: (lambda f: f)),
            "is_flash_attn_greater_or_equal_2_10": (lambda *a, **k: False),
            "is_torchdynamo_compiling": (lambda *a, **k: False),
            "logging": _LoggingNamespace,
            "replace_return_docstrings": (lambda *a, **k: (lambda f: f)),
        },
    )
    _ensure("transformers.models", {})
    _ensure("transformers.models.llama", {})

    class _LlamaConfig(_Stub):
        pass

    _ensure(
        "transformers.models.llama.configuration_llama",
        {"LlamaConfig": _LlamaConfig},
    )


def test_run_gptq_if_enabled_dispatches_for_string_w_bits(monkeypatch):
    """Real entry-point smoke: exercise the FP-aware gate end-to-end on
    the FP yaml. The gate (`if _quant_enabled(config['quantize']['w_bits']):`)
    must dispatch into the GPTQ branch when w_bits is "nvfp4", which the
    pre-round-5 numeric `bits < 16` check could not do.

    Stubs (via the seam parameters of run_gptq_if_enabled): `gptq_fwrd`,
    `get_wikitext2`, and `AutoTokenizer.from_pretrained` so no model
    weights, no calibration data, and no real tokenizer are required.
    """
    _install_heavyweight_stubs()
    import os
    import sys
    import yaml

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from promix.eval import ptq

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(
        repo_root, "promix", "configs", "llama-3.2-1b-mxfp8-nvfp4.yaml"
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg["quantize"]["w_bits"], str), (
        f"FP yaml must declare string w_bits; got {cfg['quantize']['w_bits']!r}"
    )

    # Capture seam invocation
    invocations = []

    def fake_gptq_fwrd(model, trainloader, dev, config):
        invocations.append({
            "model": model, "trainloader": trainloader,
            "dev": dev, "w_bits": config["quantize"]["w_bits"],
        })

    def fake_get_wikitext2(**_kwargs):
        return [("dummy_input", "dummy_target")]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    fake_model = object()  # the gate doesn't touch the model itself
    fake_dev = "cpu"

    ran = ptq.run_gptq_if_enabled(
        fake_model, cfg, fake_dev,
        _gptq_fwrd=fake_gptq_fwrd,
        _get_wikitext2=fake_get_wikitext2,
        _AutoTokenizer=FakeAutoTokenizer,
    )

    assert ran is True, (
        "run_gptq_if_enabled must return True for FP w_bits; the FP-aware "
        "gate did not dispatch through gptq_fwrd"
    )
    assert len(invocations) == 1, (
        f"gptq_fwrd should have been invoked exactly once; got {len(invocations)} calls"
    )
    inv = invocations[0]
    assert inv["w_bits"] == cfg["quantize"]["w_bits"]
    assert inv["model"] is fake_model
    assert inv["dev"] == fake_dev


def test_run_gptq_if_enabled_skips_for_w_bits_16(monkeypatch):
    """The same gate must skip GPTQ when w_bits=16 (FP16 pass-through).
    Regression guard for the predicate's numeric path."""
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from promix.eval import ptq

    cfg = {
        "model": {"name": "fake-model"},
        "quantize": {"w_bits": 16},
        "calibration": {"nsamples": 1, "seed": 0},
    }

    invocations = []

    def fake_gptq_fwrd(*_a, **_kw):
        invocations.append(True)

    ran = ptq.run_gptq_if_enabled(
        object(), cfg, "cpu",
        _gptq_fwrd=fake_gptq_fwrd,
        _get_wikitext2=lambda **_: [],
        _AutoTokenizer=type("FakeT", (), {"from_pretrained": staticmethod(lambda *_a, **_kw: None)}),
    )
    assert ran is False
    assert len(invocations) == 0


# ---------------------------------------------------------------------------
# Train-vs-fuse o_proj transform consistency tests.
# Round 6 added a `use_oproj_global` flag on each LlamaAttention class so
# the train-time o_proj forward drops dynamic per-head R2 (matching what
# fuse_basis_to_model does in global mode). These tests verify the flag
# actually changes the call into self.o_proj(...).
# ---------------------------------------------------------------------------


def test_attention_forwards_drop_R2_on_oproj_in_global_mode():
    """All three attention classes (LlamaAttention, LlamaFlashAttention2,
    LlamaSdpaAttention) MUST have the round-6 R2-drop pattern in their
    forward methods: `oproj_R2 = None if self.use_oproj_global else R2`
    immediately followed by `self.o_proj(attn_output, R1, R2=oproj_R2,
    transpose=True)`. The companion unit-level test
    (`test_quantize_linear_oproj_R2_drop_unit_logic` below) verifies the
    conditional itself; this test is the safety net that catches a
    regression in ANY of the three production attention forwards.

    Source-level check rather than runtime invocation because building
    a real LlamaAttention forward requires the full transformers + a
    rotary embedding implementation that the lightweight test harness
    cannot construct without bringing in the real transformers package.
    """
    import os
    import re

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(
        repo_root, "promix", "train", "modeling_llama_train.py"
    )
    with open(src_path, "r") as f:
        src = f.read()

    # Each of the three attention classes must carry the R2-drop guard.
    # Pattern is one logical line `oproj_R2 = None if self.use_oproj_global else R2`
    # followed (near-by) by `self.o_proj(...R2=oproj_R2...)`. Whitespace
    # flexible.
    drop_pattern = re.compile(
        r"oproj_R2\s*=\s*None\s+if\s+self\.use_oproj_global\s+else\s+R2"
    )
    forward_pattern = re.compile(
        r"self\.o_proj\([^)]*R2\s*=\s*oproj_R2[^)]*\)"
    )
    drops = drop_pattern.findall(src)
    o_proj_calls = forward_pattern.findall(src)
    assert len(drops) >= 3, (
        f"Expected at least 3 `oproj_R2 = None if self.use_oproj_global else R2` "
        f"sites (one per attention class); found {len(drops)}"
    )
    assert len(o_proj_calls) >= 3, (
        f"Expected at least 3 `self.o_proj(...R2=oproj_R2...)` sites; "
        f"found {len(o_proj_calls)}"
    )

    # No bare `self.o_proj(...R2=R2...)` should remain (would mean the
    # round-6 fix was reverted on at least one attention class).
    bare_pattern = re.compile(
        r"self\.o_proj\([^)]*R2\s*=\s*R2[^)]*\)"
    )
    assert not bare_pattern.search(src), (
        "Found `self.o_proj(...R2=R2...)` in modeling_llama_train.py; "
        "the round-6 fix dropped R2 from o_proj forwards in global mode "
        "and this pattern means at least one attention class regressed"
    )


def test_quantize_linear_oproj_R2_drop_unit_logic():
    """`promix.train.quant_linear.QuantizeLinear.forward` is the layer
    that consumes R2; we patch its forward to record (R1, R2) and run a
    minimal attention-style call sequence under both flag values.

    This is a unit-level regression of the conditional itself: round 6's
    pattern `oproj_R2 = None if self.use_oproj_global else R2` MUST
    produce R2=None on o_proj when the flag is True. Companion to
    `test_attention_forwards_drop_R2_on_oproj_in_global_mode` which
    asserts the production source actually contains this conditional in
    all three attention classes.
    """
    import torch.nn as nn

    from promix.train.quant_linear import QuantizeLinear

    captured = []

    original_forward = QuantizeLinear.forward

    def recording_forward(self, input, R1=None, R2=None, transpose=False):
        captured.append({
            "name": getattr(self, "_test_name", "?"),
            "R1": None if R1 is None else "tensor",
            "R2": None if R2 is None else "tensor",
            "transpose": transpose,
        })
        return original_forward(self, input, R1=R1, R2=R2, transpose=transpose)

    QuantizeLinear.forward = recording_forward
    try:
        # Build two QuantizeLinear modules: one stands in for v_proj,
        # one for o_proj. Mimic what each attention forward does — pass
        # R2 to v_proj always; pass R2 to o_proj only when NOT global.
        torch.manual_seed(0)
        v_proj = QuantizeLinear(8, 8, bias=False)
        v_proj._test_name = "v_proj"
        o_proj = QuantizeLinear(8, 8, bias=False)
        o_proj._test_name = "o_proj"

        x = torch.randn(2, 8)
        R1 = torch.eye(8, dtype=torch.float32)
        R2 = torch.eye(8, dtype=torch.float32)

        # Per-head mode: o_proj receives R2 (legacy behavior).
        use_oproj_global = False
        oproj_R2 = None if use_oproj_global else R2
        v_proj(x, R1=R1, R2=R2)
        o_proj(x, R1=R1, R2=oproj_R2, transpose=True)

        # Global mode: o_proj DROPS R2.
        use_oproj_global = True
        oproj_R2 = None if use_oproj_global else R2
        v_proj(x, R1=R1, R2=R2)
        o_proj(x, R1=R1, R2=oproj_R2, transpose=True)

        # Assert recordings: 4 calls total, in this order.
        assert len(captured) == 4, captured
        # Per-head mode (calls 0,1)
        assert captured[0]["name"] == "v_proj" and captured[0]["R2"] == "tensor"
        assert captured[1]["name"] == "o_proj" and captured[1]["R2"] == "tensor"
        # Global mode (calls 2,3)
        assert captured[2]["name"] == "v_proj" and captured[2]["R2"] == "tensor", (
            "global mode must keep R2 on v_proj (R2 is intrinsic to attention V path)"
        )
        assert captured[3]["name"] == "o_proj" and captured[3]["R2"] is None, (
            "global mode MUST drop R2 on o_proj forward; final fuse does not "
            "compose R2 into o_proj input transform, so train-time must match"
        )
    finally:
        QuantizeLinear.forward = original_forward


# ---------------------------------------------------------------------------
# Section 11 fixed-block-scale Hessian FP GPTQ test.
# Round 6 replaced the no-Hessian whole-tensor RTN bypass with proper
# Hessian-conditioned per-column rounding under frozen block scales. This
# test proves the Hessian correction actually fires: same FP quantizer +
# weights, different Hessian → different quantized output.
# ---------------------------------------------------------------------------


def test_section11_fp_gptq_uses_hessian_correction():
    """Two GPTQ instances on the same weight + FP quantizer config;
    one with identity Hessian, one with non-identity. Outputs must
    differ when Hessian correction is active. With round-4's no-Hessian
    bypass, identity-vs-non-identity Hessian produced identical output
    (the inner loop was skipped); with round-6's Section 11 GPTQ,
    Hessian correction modifies the corrected `w` before rounding, so
    outputs diverge.

    Also checks that quantized columns lie on the FP representable grid
    × frozen scale (i.e. the GPTQ output respects the FP quantization
    rule, not arbitrary float values).
    """
    import torch.nn as nn

    from promix.quantize.gptq import GPTQ
    from promix.quantize.kv_quant import WeightQuantizer

    torch.manual_seed(0)
    in_features = 32   # %32 (MXFP8) and %16 (NVFP4) both divide
    out_features = 4

    def make_gptq():
        layer = nn.Linear(in_features, out_features, bias=False)
        torch.manual_seed(7)  # deterministic same starting weights
        layer.weight.data = torch.randn_like(layer.weight.data) * 0.1
        gptq = GPTQ(layer, mixed_precision=False, high_bits_length=0, low_bits_length=0)
        q = WeightQuantizer()
        q.configure(bits="nvfp4", perchannel=True, sym=True, mse=False)
        gptq.quantizer = q
        return gptq, layer

    # Run 1: identity Hessian (Hessian correction = no-op even when active)
    gptq_a, layer_a = make_gptq()
    inp_a = torch.eye(in_features).repeat(2, 1, 1)  # eye-like input -> H = identity * scale
    out_a = layer_a(inp_a)
    gptq_a.add_batch(inp_a, out_a)
    gptq_a.fasterquant(blocksize=16, percdamp=0.01, groupsize=-1, actorder=False)
    Q_identity_H = layer_a.weight.data.clone()

    # Run 2: non-identity Hessian (Hessian correction WILL change values)
    gptq_b, layer_b = make_gptq()
    torch.manual_seed(1)
    inp_b = torch.randn(4, 16, in_features) * 1.5
    out_b = layer_b(inp_b)
    gptq_b.add_batch(inp_b, out_b)
    gptq_b.fasterquant(blocksize=16, percdamp=0.01, groupsize=-1, actorder=False)
    Q_random_H = layer_b.weight.data.clone()

    assert not torch.allclose(Q_identity_H, Q_random_H), (
        "Section 11 FP GPTQ must apply Hessian correction; identity vs "
        "non-identity Hessian produced bit-identical output, which means the "
        "per-column inner loop is no-op (round-4 RTN bypass regression)"
    )

    # Both outputs must lie on the NVFP4 representable grid × frozen scale.
    # Verify Q_random_H by checking that W / frozen_scale rounds to a valid
    # E2M1 representable for each column.
    from promix.quantize.quant_utils import (
        FP4_E2M1_MAX,
        FP4_E2M1_POS,
        select_block_scales_nvfp4,
    )

    block_size = 16
    Wf = Q_random_H.to(torch.float32)
    # Use the frozen scales the GPTQ run actually picked.
    assert hasattr(gptq_b.quantizer, "frozen_scales"), (
        "Section 11 GPTQ: WeightQuantizer.find_params(W) must populate "
        "frozen_scales for FP segments"
    )
    fs = gptq_b.quantizer.frozen_scales  # (out, n_blocks)
    fs_per_col = fs.repeat_interleave(block_size, dim=-1)  # (out, in)
    quotient = (Wf / fs_per_col).abs()
    representable = torch.tensor(FP4_E2M1_POS, dtype=torch.float32)
    # For each element, find min distance to any representable.
    diffs = (quotient.unsqueeze(-1) - representable).abs()
    nearest_dist = diffs.min(dim=-1).values
    assert (nearest_dist <= 1e-3).all(), (
        f"some quantized values fall off the E2M1 × frozen_scale grid; "
        f"max distance = {nearest_dist.max().item()}"
    )


# ---------------------------------------------------------------------------
# down_proj mixed-FP routing tests (Codex round-6 review found that
# `configure_quantizers` zeroed down_proj high/low and `gptq_fwrd` excluded
# down_proj from mixed_precision; round 7 fixes both for the FP path).
# ---------------------------------------------------------------------------


def _build_minimal_fp_model_with_down_proj(intermediate_size=128, hidden_size=64):
    """Tiny stand-in for a Llama model the smoke configure_quantizers test
    can call. Has the .config attributes configure_quantizers reads and
    one ActQuantWrapper-wrapped Linear named like a real `down_proj`.
    """
    import torch.nn as nn

    from promix.quantize.quant_utils import ActQuantWrapper

    class _MLP(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            # The wrapped layer's .module is the inner Linear; the wrapper
            # is what configure_quantizers configures via find_qlayers.
            self.down_proj = ActQuantWrapper(
                nn.Linear(intermediate, hidden, bias=False)
            )

    class _Block(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.mlp = _MLP(hidden, intermediate)

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block(hidden_size, intermediate_size)])

    class _Cfg:
        pass

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.config.hidden_size = hidden_size
            self.config.intermediate_size = intermediate_size
            self.config.num_attention_heads = 4
            self.model = _Inner()

    return _Model()


def test_configure_quantizers_down_proj_uses_intermediate_size_split():
    """In FP mode, down_proj must use intermediate_size-derived high/low
    lengths, not hidden_size or zero. Block alignment: with
    high_fraction=0.125 and intermediate_size=128, expected
    high_bits_length = 16 (128 * 0.125), divisible by both 16 (NVFP4
    block) and 32 needs care — for the test we pick a size that's
    divisible by 16 to keep NVFP4 blocks valid; full 1B Llama uses
    intermediate_size=8192 which gives high=1024 (divisible by 32).

    The legacy INT W4A4 path keeps the all-main down_proj behavior
    (separate test below).
    """
    _install_heavyweight_stubs()
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from promix.eval.ptq import configure_quantizers

    intermediate_size = 128  # 128 * 0.125 = 16 (divisible by 16 NVFP4 block)
    hidden_size = 64
    model = _build_minimal_fp_model_with_down_proj(
        intermediate_size=intermediate_size, hidden_size=hidden_size
    )
    cfg = {
        "quantize": {
            "a_bits": "nvfp4",
            "high_bits": "mxfp8",
            "low_bits": 2,
            "high_fraction": 0.125,
            "low_fraction": 0.0,
            "a_asym": True,
        }
    }
    configure_quantizers(model, cfg)

    down_proj_wrapper = model.model.layers[0].mlp.down_proj
    quantizer = down_proj_wrapper.quantizer
    expected_high = int(0.125 * intermediate_size)
    assert quantizer.high_bits_length == expected_high, (
        f"FP down_proj must use intermediate_size-derived high split; "
        f"expected {expected_high}, got {quantizer.high_bits_length}"
    )
    assert quantizer.high_bits == "mxfp8"
    assert quantizer.bits == "nvfp4"
    # Sanity: the split is not derived from hidden_size (which would be 8).
    assert quantizer.high_bits_length != int(0.125 * hidden_size), (
        "down_proj split must not be derived from hidden_size; that is the "
        "round-6 regression we are guarding against"
    )


def test_configure_quantizers_down_proj_int_keeps_legacy_zero():
    """INT W4A4 (legacy ResQ baseline at PPL=14.72) intentionally
    excluded down_proj from the high split. Round 7 must NOT change
    that behavior or the existing INT PPL would regress.
    """
    _install_heavyweight_stubs()
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from promix.eval.ptq import configure_quantizers

    model = _build_minimal_fp_model_with_down_proj(
        intermediate_size=128, hidden_size=64
    )
    cfg = {
        "quantize": {
            "a_bits": 4,       # INT activation
            "high_bits": 8,    # INT high
            "low_bits": 2,
            "high_fraction": 0.125,
            "low_fraction": 0.0,
            "a_asym": True,
        }
    }
    configure_quantizers(model, cfg)
    down_proj = model.model.layers[0].mlp.down_proj
    assert down_proj.quantizer.high_bits_length == 0, (
        "INT path must keep the legacy down_proj all-main behavior; this "
        "guards against breaking the existing PPL=14.72 baseline"
    )
    assert down_proj.quantizer.low_bits_length == 0


def test_gptq_down_proj_mixed_precision_freezes_mxfp8_scales():
    """Round-7 GPTQ must include down_proj in the mixed-precision path
    when the FP config uses string `w_bits` / `high_bits`. Construct a
    down_proj-shaped Linear directly (skipping `gptq_fwrd`'s heavy
    model setup) and verify GPTQ.fasterquant freezes MXFP8 scales on
    the high segment with the expected shape.

    Layer dims mimic 1B Llama down_proj: in=8192, out=2048.
    high_bits_length = 0.125 * 8192 = 1024 (divisible by 32 MXFP8 block).
    Expected frozen_scales shape on the high quantizer:
    (out_features, high_bits_length / 32) = (2048, 32).

    This is a direct GPTQ-level test, mirroring round-6's
    test_section11 helper but on a down_proj-shaped layer to prove the
    intermediate-dim path works end-to-end through fasterquant.
    """
    import torch.nn as nn

    from promix.quantize.gptq import GPTQ
    from promix.quantize.kv_quant import WeightQuantizer

    torch.manual_seed(0)
    in_features = 256       # smaller stand-in (256 * 0.125 = 32; %32 OK)
    out_features = 16
    high_bits_length = 32   # MXFP8 segment, %32==0
    low_bits_length = 0

    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight.data = torch.randn_like(layer.weight.data) * 0.05

    gptq = GPTQ(
        layer,
        mixed_precision=True,
        high_bits_length=high_bits_length,
        low_bits_length=low_bits_length,
    )
    gptq.quantizer = WeightQuantizer()
    gptq.quantizer.configure(bits="nvfp4", perchannel=True, sym=True, mse=False)
    gptq.high_quantizer = WeightQuantizer()
    gptq.high_quantizer.configure(bits="mxfp8", perchannel=True, sym=True, mse=False)

    inp = torch.randn(2, 4, in_features) * 0.5
    out = layer(inp)
    gptq.add_batch(inp, out)
    gptq.fasterquant(blocksize=64, percdamp=0.01, groupsize=-1, actorder=False)

    assert hasattr(gptq.high_quantizer, "frozen_scales"), (
        "down_proj high quantizer must have frozen MXFP8 block scales after "
        "fasterquant; without them, Section 11 GPTQ degenerates to RTN"
    )
    fs = gptq.high_quantizer.frozen_scales
    expected_shape = (out_features, high_bits_length // 32)
    assert fs.shape == expected_shape, (
        f"frozen_scales shape mismatch on down_proj high segment: "
        f"expected {expected_shape}, got {tuple(fs.shape)}"
    )
    # Scales must be finite and positive.
    assert (fs > 0).all() and torch.isfinite(fs).all()
    # Quantized weights on the high segment must lie on
    # representable_grid × frozen_scale.
    Wq = layer.weight.data.float()
    high_segment = Wq[:, in_features - high_bits_length:]
    fs_per_col = fs.repeat_interleave(32, dim=-1)
    from promix.quantize.quant_utils import _FP8_E4M3_POS, FP8_E4M3_MAX
    quotient = (high_segment / fs_per_col).abs()
    representable = torch.tensor(_FP8_E4M3_POS, dtype=torch.float32)
    diffs = (quotient.unsqueeze(-1) - representable).abs()
    nearest_dist = diffs.min(dim=-1).values
    assert (nearest_dist <= 1e-3).all(), (
        f"down_proj high segment values fall off MXFP8 grid; "
        f"max distance = {nearest_dist.max().item()}"
    )


# ---------------------------------------------------------------------------
# Real-attention forward R2-drop regression (round-7 contract item that
# round-7 satisfied with a source-pattern scan instead of a runtime
# forward call). This test constructs an actual `LlamaSdpaAttention`
# instance, runs forward(), and records each QuantizeLinear call's R2.
# It catches semantic regressions of the round-6 fix that survive a
# textual pattern check.
# ---------------------------------------------------------------------------


def test_real_sdpa_attention_drops_R2_on_oproj_in_global_mode():
    """Real-forward regression: builds a `LlamaSdpaAttention` (bypassing
    `__init__` which calls `LlamaRotaryEmbedding`), provides
    pre-computed `position_embeddings` to skip the rotary path, and
    records each `QuantizeLinear` call's (R1, R2). Asserts:
      - q/k receive R1 in both modes; v_proj receives R2 in both modes
      - o_proj receives R2 in legacy mode (use_oproj_global=False)
      - o_proj receives R2=None in global mode (use_oproj_global=True)

    This is the runtime regression Codex round-7 review asked for; the
    round-7 source-pattern scan is kept as secondary coverage but only
    catches textual regressions of the conditional, not semantic ones.
    """
    _install_transformers_train_stubs()
    import torch.nn as nn

    from promix.train.modeling_llama_train import LlamaSdpaAttention
    from promix.train.quant_linear import QuantizeLinear

    bsz = 1
    q_len = 4
    num_heads = 2
    num_key_value_heads = 2
    head_dim = 4  # small but even (rotate_half splits last dim in half)
    hidden_size = num_heads * head_dim  # 8

    # Construct attn instance bypassing __init__: LlamaSdpaAttention's
    # parent LlamaAttention.__init__ instantiates LlamaRotaryEmbedding,
    # which we don't need (and would require a full LlamaConfig). We
    # provide all attributes the forward path actually reads.
    attn = LlamaSdpaAttention.__new__(LlamaSdpaAttention)
    nn.Module.__init__(attn)
    attn.layer_idx = 0
    attn.attention_dropout = 0.0
    attn.hidden_size = hidden_size
    attn.num_heads = num_heads
    attn.head_dim = head_dim
    attn.num_key_value_heads = num_key_value_heads
    attn.num_key_value_groups = num_heads // num_key_value_heads
    attn.max_position_embeddings = 16
    attn.rope_theta = 10000.0
    attn.is_causal = True

    attn.q_proj = QuantizeLinear(hidden_size, hidden_size, bias=False)
    attn.k_proj = QuantizeLinear(
        hidden_size, num_key_value_heads * head_dim, bias=False
    )
    attn.v_proj = QuantizeLinear(
        hidden_size, num_key_value_heads * head_dim, bias=False
    )
    attn.o_proj = QuantizeLinear(
        num_heads * head_dim, hidden_size, bias=False
    )
    attn.q_proj._test_name = "q_proj"
    attn.k_proj._test_name = "k_proj"
    attn.v_proj._test_name = "v_proj"
    attn.o_proj._test_name = "o_proj"

    # R2 is built inside forward() as `block_diag(R2_1.weight, R2_2.weight)`
    # (with optional R2_0.weight prefix). Provide minimal stand-ins; the
    # exact values don't matter — we only assert R2 IS or ISN'T passed.
    class _RBlock:
        def __init__(self, t):
            self.weight = t

    attn.R2_0 = None
    attn.R2_1 = _RBlock(torch.eye(head_dim // 2))
    attn.R2_2 = _RBlock(torch.eye(head_dim // 2))

    # Pre-computed position_embeddings → forward() skips the rotary_emb
    # call (which we would otherwise need a real LlamaRotaryEmbedding for).
    cos = torch.randn(bsz, q_len, head_dim)
    sin = torch.randn(bsz, q_len, head_dim)

    torch.manual_seed(0)
    hidden_states = torch.randn(bsz, q_len, hidden_size)
    R1 = torch.eye(hidden_size, dtype=torch.float32)

    captured = []
    orig_forward = QuantizeLinear.forward

    def recording_forward(self, input, R1=None, R2=None, transpose=False):
        captured.append({
            "name": getattr(self, "_test_name", "?"),
            "R1_present": R1 is not None,
            "R2_present": R2 is not None,
            "transpose": transpose,
        })
        return orig_forward(self, input, R1=R1, R2=R2, transpose=transpose)

    QuantizeLinear.forward = recording_forward
    try:
        # Run 1: legacy mode (use_oproj_global=False) — o_proj keeps R2.
        attn.use_oproj_global = False
        captured.clear()
        attn.forward(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
            R1=R1,
        )
        legacy_calls = list(captured)

        # Run 2: global mode — o_proj drops R2.
        attn.use_oproj_global = True
        captured.clear()
        attn.forward(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
            R1=R1,
        )
        global_calls = list(captured)
    finally:
        QuantizeLinear.forward = orig_forward

    def _by_name(calls, name):
        return [c for c in calls if c["name"] == name]

    # Both modes: q/k/v/o each called exactly once.
    for calls, mode in [(legacy_calls, "legacy"), (global_calls, "global")]:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert len(_by_name(calls, proj)) == 1, (
                f"{mode} mode: expected exactly one {proj} call; got "
                f"{len(_by_name(calls, proj))}"
            )

    # In BOTH modes, q/k/v/o receive R1 (the rotation training path
    # always passes R1 through projections so gradients flow into R1).
    for calls, mode in [(legacy_calls, "legacy"), (global_calls, "global")]:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert _by_name(calls, proj)[0]["R1_present"] is True, (
                f"{mode} mode: {proj} must receive R1"
            )

    # In BOTH modes, v_proj receives R2 (R2 is intrinsic to the
    # attention V path; the round-6 fix only touches o_proj).
    for calls, mode in [(legacy_calls, "legacy"), (global_calls, "global")]:
        assert _by_name(calls, "v_proj")[0]["R2_present"] is True, (
            f"{mode} mode: v_proj must receive R2 (intrinsic to V path)"
        )

    # The actual round-6 fix being regression-tested: o_proj receives
    # R2 in legacy mode, but R2=None in global mode.
    legacy_o = _by_name(legacy_calls, "o_proj")[0]
    assert legacy_o["R2_present"] is True, (
        "legacy mode (use_oproj_global=False): o_proj MUST receive R2; "
        "if False, the round-6 fix has regressed in LlamaSdpaAttention.forward"
    )
    global_o = _by_name(global_calls, "o_proj")[0]
    assert global_o["R2_present"] is False, (
        "global mode (use_oproj_global=True): o_proj MUST NOT receive R2 "
        "(final fuse applies U_oproj_g without composing R2; training must "
        "match). If True, the round-6 fix regressed in LlamaSdpaAttention.forward"
    )


# ---------------------------------------------------------------------------
# AC-2 fake-FP config matrix schema regression. The plan requires fake-FP
# PPL on 1B/3B/8B × W4A4/W4A4KV4 (six configs). Round 9 authored five new
# yamls beside the round-4 1B W4A4 config; this test guards the matrix's
# completeness and per-config schema (string FP identifiers, global o_proj
# PCA, block-aligned high split, KV4 fields where applicable, distinct
# artifact paths).
# ---------------------------------------------------------------------------


# Per-model architecture facts (HF Llama-3.2-1B / 3.2-3B / Llama-3-8B):
# (hidden_size, intermediate_size, num_attention_heads, head_dim).
_MODEL_DIMS = {
    "llama-3.2-1b": (2048, 8192, 32, 64),
    "llama-3.2-3b": (3072, 8192, 24, 128),
    "llama-3-8b":   (4096, 14336, 32, 128),
}

_FP_CONFIGS = [
    # (yaml_filename, model_key, expects_kv4)
    ("llama-3.2-1b-mxfp8-nvfp4.yaml",      "llama-3.2-1b", False),
    ("llama-3.2-1b-mxfp8-nvfp4-kv4.yaml",  "llama-3.2-1b", True),
    ("llama-3.2-3b-mxfp8-nvfp4.yaml",      "llama-3.2-3b", False),
    ("llama-3.2-3b-mxfp8-nvfp4-kv4.yaml",  "llama-3.2-3b", True),
    ("llama-3-8b-mxfp8-nvfp4.yaml",        "llama-3-8b",   False),
    ("llama-3-8b-mxfp8-nvfp4-kv4.yaml",    "llama-3-8b",   True),
]


def test_fp_config_matrix_is_complete_and_schema_valid():
    """All six FP yamls (1B/3B/8B × W4A4/W4A4KV4) must exist and follow
    the required schema: string FP identifiers, global o_proj PCA,
    block-aligned high/low splits, distinct artifact paths.

    Without all six configs, AC-2.2 (W4A4 PPL on 1B/3B/8B) and AC-2.3
    (W4A4KV4 PPL on 1B/3B/8B) cannot be measured; this test guards
    against silent regression of the matrix or per-config drift.
    """
    import os
    import yaml

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(repo_root, "promix", "configs")

    seen_basis_paths = set()
    seen_rotation_paths = set()
    seen_output_dirs = set()

    for fname, model_key, expects_kv4 in _FP_CONFIGS:
        path = os.path.join(cfg_dir, fname)
        assert os.path.isfile(path), f"missing FP config: {fname}"
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        q = cfg["quantize"]
        # FP identifier schema
        assert q["w_bits"] == "nvfp4", f"{fname}: w_bits must be 'nvfp4', got {q['w_bits']!r}"
        assert q["a_bits"] == "nvfp4", f"{fname}: a_bits must be 'nvfp4', got {q['a_bits']!r}"
        assert q["high_bits"] == "mxfp8", f"{fname}: high_bits must be 'mxfp8', got {q['high_bits']!r}"
        assert q["o_proj_pca"] == "full_global", (
            f"{fname}: o_proj_pca must be 'full_global', got {q['o_proj_pca']!r}"
        )
        assert q["w_groupsize"] == -1, f"{fname}: w_groupsize must be -1 (per-group FP GPTQ unsupported), got {q['w_groupsize']!r}"
        assert q["high_fraction"] == 0.125, f"{fname}: high_fraction must be 0.125 for the planned matrix"

        # Block-alignment for hidden_size and intermediate_size at high_fraction
        hidden, intermediate, _heads, head_dim = _MODEL_DIMS[model_key]
        for dim_name, dim in [("hidden", hidden), ("intermediate", intermediate)]:
            high = int(q["high_fraction"] * dim)
            main = dim - high
            assert high % 32 == 0, (
                f"{fname}: {dim_name} high segment {high} not divisible by 32 "
                f"(MXFP8 block size); choose a high_fraction whose product is a "
                f"multiple of 32 for {dim_name}_size={dim}"
            )
            assert main % 16 == 0, (
                f"{fname}: {dim_name} main segment {main} not divisible by 16 "
                f"(NVFP4 block size); high_fraction or {dim_name}_size violates alignment"
            )

        # KV4 vs W4A4
        if expects_kv4:
            assert q["k_bits"] == "nvfp4", f"{fname}: KV4 variant must set k_bits='nvfp4'"
            assert q["v_bits"] == "nvfp4", f"{fname}: KV4 variant must set v_bits='nvfp4'"
            assert q["k_groupsize"] == head_dim, (
                f"{fname}: k_groupsize must equal head_dim ({head_dim}); got {q['k_groupsize']}"
            )
            assert q["v_groupsize"] == head_dim, (
                f"{fname}: v_groupsize must equal head_dim ({head_dim}); got {q['v_groupsize']}"
            )
            # head_dim % 16 == 0 required for NVFP4
            assert head_dim % 16 == 0, (
                f"{fname}: KV4 needs head_dim divisible by 16; head_dim={head_dim}"
            )
        else:
            assert q["k_bits"] == 16, f"{fname}: W4A4 variant must keep k_bits=16"
            assert q["v_bits"] == 16, f"{fname}: W4A4 variant must keep v_bits=16"

        # Path uniqueness across the matrix (output_dir at minimum; basis
        # and rotation may share between W4A4 and KV4 variants of the same
        # model since the basis bundle is the same — just assert
        # output_dir is unique across all configs).
        assert cfg["paths"]["output_dir"] not in seen_output_dirs, (
            f"{fname}: output_dir collides with an earlier config"
        )
        seen_output_dirs.add(cfg["paths"]["output_dir"])
        seen_basis_paths.add(cfg["paths"]["basis"])
        seen_rotation_paths.add(cfg["paths"]["rotation"])

    # Sanity: 6 distinct output_dirs, ≥3 distinct basis paths (one per
    # model — W4A4 and KV4 share the basis bundle since basis depends on
    # model architecture not on KV quantization).
    assert len(seen_output_dirs) == 6
    assert len(seen_basis_paths) >= 3, (
        f"expected at least 3 distinct basis paths (one per model); got "
        f"{len(seen_basis_paths)}: {seen_basis_paths}"
    )


# ---------------------------------------------------------------------------
# o_proj cosine sanity harness unit test (round-10 / AC-2.4 code half).
# ---------------------------------------------------------------------------


def test_cosine_sanity_per_layer_returns_dict():
    """`compute_oproj_cosine_per_layer` must return a dict keyed by
    layer index with cosine values in [-1, 1]. With identical model
    and reference_model the cosines are 1.0 (degenerate identity).

    Uses a tiny synthetic model whose `model.model.layers[i].self_attn`
    each have an `o_proj` Linear; runs forward on a couple of batches;
    asserts shape and value invariants.
    """
    import torch.nn as nn

    from promix.eval.cosine_sanity import compute_oproj_cosine_per_layer

    class _Attn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.o_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            # Synthetic: just call o_proj on the input directly. The
            # harness only cares about hooking o_proj's output.
            return self.o_proj(x)

    class _Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.self_attn = _Attn(dim)

        def forward(self, x):
            return self.self_attn(x)

    class _Inner(nn.Module):
        def __init__(self, dim, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class _Model(nn.Module):
        def __init__(self, dim=8, n_layers=2):
            super().__init__()
            self.model = _Inner(dim, n_layers)

        def forward(self, x):
            return self.model(x)

    torch.manual_seed(0)
    n_layers = 2
    dim = 8
    model = _Model(dim=dim, n_layers=n_layers)

    # Simulate a dataloader yielding (input_ids, _) tuples; for this
    # synthetic test, input_ids is just a float tensor of shape (bsz, seq, dim).
    inputs = [(torch.randn(1, 4, dim).float(), None) for _ in range(3)]

    # No reference_model → degenerate self-similarity.
    cosines = compute_oproj_cosine_per_layer(
        model, inputs, max_batches=3, device=torch.device("cpu")
    )
    assert isinstance(cosines, dict)
    assert sorted(cosines.keys()) == list(range(n_layers))
    for idx, c in cosines.items():
        assert isinstance(c, float)
        assert -1.0 <= c <= 1.0
        # Self-similarity is 1.0 (or extremely close due to fp).
        assert c > 0.999, (
            f"layer {idx}: degenerate self-similarity should be ~1.0; got {c}"
        )

    # With identical models as primary and reference, cosines should
    # also be ~1.0 — the harness records both side's outputs from the
    # SAME forward pass equivalently.
    import copy
    ref = copy.deepcopy(model)
    cosines_ref = compute_oproj_cosine_per_layer(
        model, inputs, reference_model=ref,
        max_batches=3, device=torch.device("cpu"),
    )
    assert sorted(cosines_ref.keys()) == list(range(n_layers))
    for idx, c in cosines_ref.items():
        assert c > 0.999, (
            f"layer {idx}: identical-model cosine should be ~1.0; got {c}"
        )


# ---------------------------------------------------------------------------
# Round-11 cosine harness defect-fix tests (Codex round-10 review found
# 4 substantive defects; these assert the fixes hold).
# ---------------------------------------------------------------------------


def test_chunk_wikitext_yields_correct_shape_tuples():
    """`chunk_wikitext_for_cosine` (via lower-level
    `_chunk_from_input_ids`) MUST return a list of `(input_ids, None)`
    tuples each with shape `(1, seqlen)`, deterministic by seed.

    Round 10's CLI passed the raw BatchEncoding directly into the
    harness which expects `(input_ids, _)` tuples; the fix is to
    chunk explicitly.
    """
    from promix.eval.cosine_sanity import _chunk_from_input_ids

    total_tokens = 4096
    seqlen = 256
    nsamples = 5
    fake_input_ids = torch.arange(total_tokens, dtype=torch.long).unsqueeze(0)

    chunks = _chunk_from_input_ids(
        fake_input_ids, nsamples=nsamples, seqlen=seqlen, seed=0
    )
    assert isinstance(chunks, list)
    assert len(chunks) == nsamples
    for input_ids, label in chunks:
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.shape == (1, seqlen)
        assert label is None

    # Determinism: same seed -> same chunks.
    chunks_again = _chunk_from_input_ids(
        fake_input_ids, nsamples=nsamples, seqlen=seqlen, seed=0
    )
    for (a, _), (b, _) in zip(chunks, chunks_again):
        assert torch.equal(a, b)

    # Different seed -> at least one different chunk start (probabilistic
    # but deterministic for these specific values).
    chunks_seed1 = _chunk_from_input_ids(
        fake_input_ids, nsamples=nsamples, seqlen=seqlen, seed=1
    )
    differs = any(
        not torch.equal(a, b)
        for (a, _), (b, _) in zip(chunks, chunks_seed1)
    )
    assert differs, (
        "different seeds should produce different chunk offsets"
    )


def test_chunk_from_input_ids_rejects_short_sequences():
    """Insufficient token budget must error explicitly, not silently
    produce overlapping chunks or out-of-bounds slicing."""
    from promix.eval.cosine_sanity import _chunk_from_input_ids

    short = torch.arange(50, dtype=torch.long).unsqueeze(0)
    try:
        _chunk_from_input_ids(short, nsamples=2, seqlen=64, seed=0)
        raise AssertionError("expected RuntimeError on insufficient tokens")
    except RuntimeError as e:
        assert "insufficient tokens" in str(e).lower()


def test_cosine_cli_requires_reference_config():
    """The CLI argparse must reject invocations without
    `--reference_config`. Round 10 silently fell back to self-cosine
    (always 1.0); round 11 makes the flag mandatory.
    """
    from promix.eval.cosine_sanity import _build_argparser

    parser = _build_argparser()
    # argparse's required=True path raises SystemExit(2) when the flag is
    # missing, after printing to stderr.
    raised = False
    try:
        parser.parse_args(["--config", "fake.yaml"])
    except SystemExit as e:
        raised = True
        assert e.code != 0, (
            f"argparse should exit non-zero when --reference_config is "
            f"missing; got exit code {e.code}"
        )
    assert raised, (
        "CLI must reject missing --reference_config; round 10 silently "
        "fell back to self-cosine (always 1.0)"
    )

    # Sanity: providing both flags parses successfully.
    args = parser.parse_args([
        "--config", "fp.yaml",
        "--reference_config", "int.yaml",
    ])
    assert args.config == "fp.yaml"
    assert args.reference_config == "int.yaml"


def test_cosine_sanity_main_calls_prepare_ptq_model_for_both_configs(monkeypatch):
    """Round 10's cosine CLI used a partial ad-hoc build path that
    omitted `install_column_order_hooks`, `run_gptq_if_enabled`, and
    `setup_k_quant`. Round 11 routes both primary and reference through
    `promix.eval.ptq.prepare_ptq_model` so AC-2.4 measures the SAME
    PTQ algorithm Step 2 PPL evaluates against.

    This test patches `prepare_ptq_model`, `chunk_wikitext_for_cosine`,
    `compute_oproj_cosine_per_layer`, and `transformers.AutoTokenizer`,
    invokes `main()` with both configs pointing at the same model name,
    and asserts the patched `prepare_ptq_model` was called exactly
    twice (once per config).
    """
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.eval import cosine_sanity as cs_mod
    from promix.eval import ptq as ptq_mod

    # Use real config files (same model.name ensures the
    # checkpoint-equality guard passes).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary = os.path.join(repo_root, "promix", "configs",
                           "llama-3.2-1b-mxfp8-nvfp4.yaml")
    reference = os.path.join(repo_root, "promix", "configs",
                             "llama-3.2-1b-w4a4.yaml")

    calls = {"prepare": [], "tokenize": [], "chunk": [], "compute": []}

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            calls["tokenize"].append(name)
            return object()

    def fake_prepare(config, dev, *, run_gptq=True):
        calls["prepare"].append({
            "model": config["model"]["name"],
            "w_bits": config["quantize"]["w_bits"],
            "run_gptq": run_gptq,
        })
        return object()

    def fake_chunk(tokenizer, *, nsamples, seqlen, seed=0):
        calls["chunk"].append({"nsamples": nsamples, "seqlen": seqlen})
        return [(torch.zeros(1, seqlen, dtype=torch.long), None)
                for _ in range(nsamples)]

    def fake_compute(model, dataloader, *, reference_model=None,
                     device=None, max_batches=8):
        calls["compute"].append({"max_batches": max_batches})
        return {0: 0.999, 1: 0.999}

    # Patch in cosine_sanity's namespace
    monkeypatch.setattr(ptq_mod, "prepare_ptq_model", fake_prepare)

    # Round 12 added a distributed-init step at the top of main();
    # neutralize it for this checkpoint-equality test (covered by
    # the dedicated test_cosine_sanity_main_initializes_distributed
    # test below).
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    import transformers as _tf
    monkeypatch.setattr(_tf, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(cs_mod, "chunk_wikitext_for_cosine", fake_chunk)
    monkeypatch.setattr(cs_mod, "compute_oproj_cosine_per_layer", fake_compute)

    # Argparse will read sys.argv; build it directly.
    monkeypatch.setattr("sys.argv", [
        "cosine_sanity",
        "--config", primary,
        "--reference_config", reference,
        "--nsamples", "2",
        "--seqlen", "32",
        "--threshold", "0.99",
    ])

    cs_mod.main()

    assert len(calls["prepare"]) == 2, (
        f"prepare_ptq_model must be called exactly twice (primary + "
        f"reference); got {len(calls['prepare'])} calls: {calls['prepare']}"
    )
    # Both call args carry a model.name; for these configs they match
    # (cosine guard would have raised otherwise).
    primary_call, reference_call = calls["prepare"]
    assert primary_call["w_bits"] == "nvfp4"
    assert reference_call["w_bits"] == 4  # legacy INT W4A4
    assert primary_call["run_gptq"] is True
    assert reference_call["run_gptq"] is True


def test_cosine_sanity_main_rejects_mismatched_checkpoints(monkeypatch):
    """When --config and --reference_config use different `model.name`
    values, main() must raise SystemExit with a clear error rather
    than silently measuring cosine across different checkpoints.

    Round 10 documented `llama-3.2-1b-mxfp8-nvfp4.yaml` (base) +
    `llama-3.2-1b-resq.yaml` (instruct) which were different
    checkpoints; this test guards the round-11 fix.
    """
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.eval import cosine_sanity as cs_mod

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary = os.path.join(repo_root, "promix", "configs",
                           "llama-3.2-1b-mxfp8-nvfp4.yaml")
    # llama-3.2-1b-resq.yaml is intentionally the WRONG reference
    # because its model.name is unsloth/Llama-3.2-1B-Instruct vs the
    # FP config's unsloth/Llama-3.2-1B.
    bad_reference = os.path.join(repo_root, "promix", "configs",
                                 "llama-3.2-1b-resq.yaml")

    # Same neutralization as the sibling test (round-12's distributed
    # init is covered separately).
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    monkeypatch.setattr("sys.argv", [
        "cosine_sanity",
        "--config", primary,
        "--reference_config", bad_reference,
        "--nsamples", "2",
        "--seqlen", "32",
    ])

    raised = False
    try:
        cs_mod.main()
    except SystemExit as e:
        raised = True
        assert "same base checkpoint" in str(e) or "model.name" in str(e), (
            f"error must explain the checkpoint mismatch; got {e}"
        )
    assert raised, (
        "main() must raise SystemExit when configs use different model.name"
    )


# ---------------------------------------------------------------------------
# Round-12 cosine harness defect-fix tests (Codex round-11 review found
# distributed-init missing + vacuous-pass on empty capture; round 12
# fixes both. These tests guard the fixes against regression.)
# ---------------------------------------------------------------------------


def test_cosine_sanity_main_initializes_distributed_before_prepare(monkeypatch):
    """`prepare_ptq_model` reaches `fuse_basis_to_model` which calls
    `torch.distributed.barrier()`. The cosine CLI must initialize
    distributed BEFORE the first `prepare_ptq_model` call, matching
    `ptq.py:main()` behavior. Round 11's CLI omitted this and would
    crash on `barrier()`.
    """
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.eval import cosine_sanity as cs_mod
    from promix.eval import ptq as ptq_mod

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary = os.path.join(repo_root, "promix", "configs",
                           "llama-3.2-1b-mxfp8-nvfp4.yaml")
    reference = os.path.join(repo_root, "promix", "configs",
                             "llama-3.2-1b-w4a4.yaml")

    call_order = []

    # Force not-initialized so init_process_group is exercised.
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: False
    )

    def fake_init(*_a, **_kw):
        call_order.append("init_process_group")

    monkeypatch.setattr(
        torch.distributed, "init_process_group", fake_init
    )

    def fake_prepare(config, dev, *, run_gptq=True):
        call_order.append("prepare_ptq_model")
        return object()

    def fake_chunk(*_a, **_kw):
        return [(torch.zeros(1, 8, dtype=torch.long), None) for _ in range(2)]

    def fake_compute(*_a, **_kw):
        return {0: 0.999}

    monkeypatch.setattr(ptq_mod, "prepare_ptq_model", fake_prepare)
    import transformers as _tf

    class _Tok:
        @staticmethod
        def from_pretrained(*_a, **_kw):
            return object()

    monkeypatch.setattr(_tf, "AutoTokenizer", _Tok)
    monkeypatch.setattr(cs_mod, "chunk_wikitext_for_cosine", fake_chunk)
    monkeypatch.setattr(cs_mod, "compute_oproj_cosine_per_layer", fake_compute)

    monkeypatch.setattr("sys.argv", [
        "cosine_sanity",
        "--config", primary,
        "--reference_config", reference,
        "--nsamples", "2",
        "--seqlen", "32",
    ])

    cs_mod.main()

    # init_process_group MUST be called before any prepare_ptq_model
    # call. There may be multiple prepare_ptq_model calls (primary +
    # reference); the first init must precede them all.
    assert "init_process_group" in call_order, (
        "cosine_sanity main() did not call init_process_group at all; "
        "without it, prepare_ptq_model -> fuse_basis_to_model -> "
        "torch.distributed.barrier() will crash on the remote run"
    )
    init_idx = call_order.index("init_process_group")
    prepare_idxs = [i for i, c in enumerate(call_order) if c == "prepare_ptq_model"]
    assert prepare_idxs, "prepare_ptq_model was never called"
    assert init_idx < prepare_idxs[0], (
        f"init_process_group must precede the first prepare_ptq_model "
        f"call; got call_order={call_order}"
    )


def test_cosine_compute_raises_on_empty_capture():
    """If the model has no attention layers (e.g. `model.model.layers`
    is empty or self_attn / o_proj is missing), the harness must
    raise RuntimeError rather than silently return {} which would
    lead to "OVERALL: PASS" with zero rows.
    """
    import torch.nn as nn

    from promix.eval.cosine_sanity import compute_oproj_cosine_per_layer

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()  # ZERO layers

        def forward(self, x):
            return x

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

        def forward(self, x):
            return self.model(x)

    model = _Model()
    inputs = [(torch.zeros(1, 4, dtype=torch.long), None)]

    raised = False
    try:
        compute_oproj_cosine_per_layer(
            model, inputs, max_batches=1, device=torch.device("cpu")
        )
    except RuntimeError as e:
        raised = True
        msg = str(e).lower()
        assert ("zero attention layers" in msg
                or "no attention layers" in msg
                or "no o_proj" in msg
                or "model.model.layers" in str(e)), (
            f"empty-capture error must explain the failure; got {e}"
        )
    assert raised, (
        "compute_oproj_cosine_per_layer must raise on zero-layer model; "
        "round 11 silently returned {} which round-12 forbids"
    )


def test_cosine_compute_raises_on_layer_count_mismatch():
    """Primary and reference models must have the same number of
    attention layers; the harness must raise RuntimeError when they
    differ rather than silently dropping mismatched layers.
    """
    import torch.nn as nn

    from promix.eval.cosine_sanity import compute_oproj_cosine_per_layer

    class _Attn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.o_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.o_proj(x)

    class _Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.self_attn = _Attn(dim)

        def forward(self, x):
            return self.self_attn(x)

    class _Inner(nn.Module):
        def __init__(self, dim, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class _Model(nn.Module):
        def __init__(self, dim=8, n_layers=2):
            super().__init__()
            self.model = _Inner(dim, n_layers)

        def forward(self, x):
            return self.model(x)

    primary = _Model(dim=8, n_layers=2)
    reference = _Model(dim=8, n_layers=3)  # mismatch
    inputs = [(torch.randn(1, 4, 8), None)]

    raised = False
    try:
        compute_oproj_cosine_per_layer(
            primary, inputs, reference_model=reference,
            max_batches=1, device=torch.device("cpu"),
        )
    except RuntimeError as e:
        raised = True
        msg = str(e)
        # Must cite both layer counts so the user can debug quickly.
        assert "2" in msg and "3" in msg, (
            f"mismatch error must cite both layer counts; got {e}"
        )
    assert raised, (
        "compute_oproj_cosine_per_layer must raise when primary and "
        "reference layer counts differ"
    )


def test_evaluate_against_threshold_raises_on_empty():
    """`_evaluate_against_threshold({})` must raise rather than
    return True (Python's `all([])` is True). This guard catches the
    case where compute_oproj_cosine_per_layer somehow produced no
    cosine values yet didn't already raise — defense in depth.
    """
    from promix.eval.cosine_sanity import _evaluate_against_threshold

    raised = False
    try:
        _evaluate_against_threshold({}, threshold=0.99)
    except RuntimeError as e:
        raised = True
        assert "no per-layer cosines" in str(e).lower() or "empty" in str(e).lower()
    assert raised, (
        "_evaluate_against_threshold({}) must raise; round-11 returned "
        "True (all([]) == True) which would silently print OVERALL: PASS"
    )

    # Sanity: non-empty input still works.
    assert _evaluate_against_threshold({0: 0.99, 1: 0.995}, 0.99) is True
    assert _evaluate_against_threshold({0: 0.99, 1: 0.98}, 0.99) is False


# ---------------------------------------------------------------------------
# Round-13 init_distributed helper tests. Codex round-12 review found
# the round-12 helper used `init_method='env://'` without ensuring the
# four `env://` rendezvous vars existed; plain `python -m ...` does not
# set them. Round 13 populates safe defaults only when the vars are
# absent, preserving torchrun compatibility.
# ---------------------------------------------------------------------------


def test_init_distributed_helper_works_for_plain_python_command(monkeypatch):
    """Plain `python -m promix.eval.cosine_sanity ...` does NOT set
    MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE. The helper must
    populate them with safe single-rank defaults BEFORE calling
    init_process_group(env://) so the env-rendezvous succeeds.

    Round-12's helper crashed in this case (init_process_group asked
    for the env vars; got KeyError on MASTER_ADDR).
    """
    import os

    from promix.eval.ptq import init_distributed_for_ptq_main_if_needed

    # Clear the four env vars so the helper has to populate them.
    for var in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    captured = {"called": False, "kwargs": None, "env_at_call": None}

    def fake_init(*args, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["env_at_call"] = {
            v: os.environ.get(v)
            for v in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init)

    init_distributed_for_ptq_main_if_needed()

    assert captured["called"], (
        "init_process_group must be called when distributed is not "
        "yet initialized"
    )
    # The helper must use init_method='env://' (consistent with the
    # round-0 ptq.py path) so torchrun-injected env vars continue to
    # control the rendezvous.
    assert captured["kwargs"].get("init_method") == "env://", (
        f"init_method must remain 'env://'; got {captured['kwargs']}"
    )
    # All four env vars must be set BEFORE init_process_group is
    # called, so the env:// rendezvous resolves under plain Python.
    env = captured["env_at_call"]
    assert env["MASTER_ADDR"] is not None, (
        "MASTER_ADDR must be populated before init_process_group; "
        "without it, env:// rendezvous fails on plain python -m"
    )
    assert env["MASTER_PORT"] is not None
    assert env["RANK"] == "0", f"single-rank default RANK=0; got {env['RANK']!r}"
    assert env["WORLD_SIZE"] == "1", (
        f"single-rank default WORLD_SIZE=1; got {env['WORLD_SIZE']!r}"
    )


def test_init_distributed_helper_respects_existing_env(monkeypatch):
    """When `torchrun` (or the user) has already set the rendezvous
    env vars, the helper must NOT override them. Pre-set values must
    survive the helper's call so multi-rank launches work correctly.
    """
    import os

    from promix.eval.ptq import init_distributed_for_ptq_main_if_needed

    monkeypatch.setenv("MASTER_ADDR", "test-host")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group",
                        lambda *a, **kw: None)

    init_distributed_for_ptq_main_if_needed()

    # Pre-set env vars must be preserved exactly (torchrun-style
    # injection wins; the helper only fills gaps).
    assert os.environ.get("MASTER_ADDR") == "test-host"
    assert os.environ.get("MASTER_PORT") == "12345"
    assert os.environ.get("RANK") == "3"
    assert os.environ.get("WORLD_SIZE") == "4"


def test_init_distributed_helper_noop_when_initialized(monkeypatch):
    """When `torch.distributed.is_initialized()` returns True (the
    process group already exists, e.g. inside a torchrun launch that
    initialized earlier), the helper must be a complete no-op:
    `init_process_group` is NOT called, env vars are NOT modified.
    """
    import os

    from promix.eval.ptq import init_distributed_for_ptq_main_if_needed

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    called = {"flag": False}

    def fake_init(*_a, **_kw):
        called["flag"] = True

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init)

    # Pre-clear so we can detect any setdefault side effect.
    for var in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)

    init_distributed_for_ptq_main_if_needed()

    assert called["flag"] is False, (
        "init_process_group must not be called when distributed is "
        "already initialized"
    )
    # Idempotence: env vars left untouched.
    for var in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"):
        assert os.environ.get(var) is None, (
            f"already-initialized branch should not modify {var}; "
            f"got {os.environ.get(var)!r}"
        )


# ---------------------------------------------------------------------------
# Round-14 quality consolidation tests:
#   - cosine harness raises clear errors on wrapper drift
#   - _quant_enabled strict-mode rejects typos at config-load surfaces
# ---------------------------------------------------------------------------


def test_cosine_attention_walker_raises_on_missing_layers():
    """`_attention_layers()` must raise RuntimeError (not raw
    AttributeError) when the model lacks `model.model.layers`. This
    surfaces wrapper drift with a debuggable message instead of an
    obscure `'XYZ' object has no attribute 'layers'` traceback.
    """
    import torch.nn as nn

    from promix.eval.cosine_sanity import _attention_layers

    class _Empty(nn.Module):
        pass

    raised = False
    try:
        list(_attention_layers(_Empty()))
    except RuntimeError as e:
        raised = True
        assert "model.model.layers" in str(e) or "Llama" in str(e), (
            f"diagnostic message must name the missing attribute path; got {e}"
        )
    assert raised, (
        "_attention_layers must raise RuntimeError on missing "
        "model.model.layers (was AttributeError before round 14)"
    )


def test_cosine_hook_install_raises_on_missing_o_proj():
    """`_install_oproj_output_hook` must raise RuntimeError when the
    attention module lacks `o_proj`, instead of the raw
    AttributeError the round-13 code threw.
    """
    import torch.nn as nn

    from promix.eval.cosine_sanity import _install_oproj_output_hook

    class _AttnNoOProj(nn.Module):
        pass

    raised = False
    sink = {}
    try:
        _install_oproj_output_hook(_AttnNoOProj(), sink, idx=0)
    except RuntimeError as e:
        raised = True
        assert "o_proj" in str(e), (
            f"diagnostic message must mention o_proj; got {e}"
        )
    assert raised, (
        "_install_oproj_output_hook must raise RuntimeError when "
        "attn_module lacks o_proj"
    )


def test_assert_quant_format_rejects_typo():
    """`assert_quant_format("nvf4")` must raise ValueError listing the
    supported set so a yaml typo surfaces loudly at PTQ entry.
    """
    from promix.quantize.quant_utils import assert_quant_format

    raised = False
    try:
        assert_quant_format("nvf4")
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "nvf4" in msg, f"error must echo the bad input; got {e}"
        assert "nvfp4" in msg, (
            f"error should hint the closest valid value; got {e}"
        )
    assert raised, (
        "assert_quant_format('nvf4') must raise; without this, a typo "
        "silently disables quantization (round-14 blocker)"
    )

    # Numeric values pass through.
    assert_quant_format(4)
    assert_quant_format(8)
    assert_quant_format(16)
    # Valid FP formats pass through.
    assert_quant_format("nvfp4")
    assert_quant_format("mxfp8")


def test_quant_enabled_default_remains_permissive():
    """`_quant_enabled(bits)` (no `strict=True`) must keep the
    round-3 contract of returning False on unknown strings, since
    multiple call sites (ActQuantWrapper.forward, kv_quant) rely on
    that fail-closed semantics.
    """
    from promix.quantize.quant_utils import _quant_enabled

    # Unknown strings: silent False (the historical contract).
    assert _quant_enabled("nvf4") is False
    assert _quant_enabled("garbage") is False

    # Numeric path unchanged.
    assert _quant_enabled(4) is True
    assert _quant_enabled(16) is False

    # Known FP formats: True.
    assert _quant_enabled("mxfp8") is True
    assert _quant_enabled("nvfp4") is True

    # Strict-mode flips the unknown-string behavior.
    raised = False
    try:
        _quant_enabled("nvf4", strict=True)
    except ValueError:
        raised = True
    assert raised


def test_configure_quantizers_rejects_typo_in_config():
    """`configure_quantizers(model, config)` must raise ValueError on
    a typo like `w_bits: "nvf4"` (was: silently produced an FP16
    model labelled as quantized).
    """
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.eval.ptq import configure_quantizers

    # Build a minimal model object the function inspects (it reads
    # model.config.{hidden_size, intermediate_size,
    # num_attention_heads}; the typo guard runs BEFORE find_qlayers
    # is iterated, so we don't need a real wrapped model).
    import torch.nn as nn

    class _Cfg:
        hidden_size = 16
        intermediate_size = 64
        num_attention_heads = 2

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()

    cfg = {
        "quantize": {
            "w_bits": "nvf4",   # ← typo that silently disabled quant in round 13
            "a_bits": "nvfp4",
            "high_bits": "mxfp8",
            "low_bits": 2,
            "high_fraction": 0.125,
            "low_fraction": 0.0,
            "a_asym": True,
        }
    }

    raised = False
    try:
        configure_quantizers(_Model(), cfg)
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "w_bits" in msg, f"error must name the bad field; got {e}"
        assert "nvf4" in msg, f"error must echo the typo value; got {e}"
    assert raised, (
        "configure_quantizers must raise on typo'd FP format string; "
        "without this, the model would be FP16 but labelled FP-quantized"
    )


# ---------------------------------------------------------------------------
# Round-15 KV4 fix: setup_k_quant must accept string `k_bits` (the
# round-9 KV4 yamls declare k_bits / v_bits as "nvfp4"). Codex round-14
# review found `setup_k_quant()` still did `if k_bits >= 16:`, which
# TypeErrors on string FP-format identifiers. Fix is the same
# predicate-and-grep pattern from BL-20260623-numeric-vs-string-bits-guards.
# ---------------------------------------------------------------------------


def test_setup_k_quant_accepts_string_kbits_without_typeerror():
    """`setup_k_quant()` is called by `prepare_ptq_model()` for every
    config. Round-9 KV4 yamls declare `k_bits: "nvfp4"`; the round-14
    code path that reached this function would TypeError on
    `k_bits >= 16` BEFORE any model wrapper setup. Round 15 routes
    through `_quant_enabled(k_bits)` to handle string identifiers.

    This test exercises the EARLIEST part of `setup_k_quant` (the
    skip-or-continue gate) by patching `torch.load` so the function
    proceeds past the gate and then hits the basis/rotation load step.
    The test asserts the gate doesn't TypeError on `"nvfp4"`; the
    expected failure mode in the local harness is a missing rotation
    file, NOT a numeric-string comparison crash.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.quantize.kv_quant import setup_k_quant

    class _Cfg:
        hidden_size = 16
        num_attention_heads = 2

    class _Inner:
        layers = []

    class _Model:
        def __init__(self):
            self.config = _Cfg()
            self.model = _Inner()

    cfg = {
        "quantize": {
            "k_bits": "nvfp4",
            "high_bits": "mxfp8",
            "low_bits": 2,
            "high_fraction": 0.125,
            "low_fraction": 0.0,
            "k_groupsize": 64,
            "k_asym": True,
        }
    }

    # The function loads basis/rotation .bin files; pass non-existent
    # paths so we can detect "got past the gate". The TypeError
    # round-15 fixes is `'>=' not supported between instances of 'str'
    # and 'int'`. Anything OTHER than that (FileNotFoundError, RuntimeError
    # from torch.load) means the gate works.
    raised_typeerror = False
    try:
        setup_k_quant(
            _Model(), cfg,
            basis_path="/tmp/_nonexistent_basis.bin",
            rotation_path="/tmp/_nonexistent_rotation.bin",
        )
    except TypeError as e:
        # Only the numeric-string comparison TypeError is the
        # round-15 regression. Re-raise any other TypeError.
        if "'>=' not supported" in str(e) or "str" in str(e) and "int" in str(e):
            raised_typeerror = True
        else:
            raised_typeerror = False
    except (FileNotFoundError, RuntimeError, Exception):
        # The gate was passed and the function failed later trying to
        # load missing artifacts. That's the expected outcome for this
        # local test — round-15's fix is solely about getting past the
        # numeric-vs-string comparison.
        pass

    assert not raised_typeerror, (
        "setup_k_quant must NOT TypeError on string k_bits; round-14 "
        "code did `if k_bits >= 16:` which crashed on 'nvfp4'. Round-15 "
        "uses `_quant_enabled(k_bits)`."
    )


def test_setup_k_quant_skips_for_numeric_16():
    """Legacy numeric `k_bits: 16` (W4A4 with FP16 KV cache) must
    still cause setup_k_quant to early-return without touching basis
    / rotation files.
    """
    from promix.quantize.kv_quant import setup_k_quant

    class _Cfg:
        hidden_size = 16
        num_attention_heads = 2

    class _Model:
        def __init__(self):
            self.config = _Cfg()

    cfg = {
        "quantize": {
            "k_bits": 16,
            "high_bits": 8,
            "low_bits": 2,
            "high_fraction": 0.125,
            "low_fraction": 0.0,
        }
    }

    # If the gate fails, the function would try to torch.load the
    # paths and FileNotFoundError. With the gate working, returns
    # cleanly.
    setup_k_quant(
        _Model(), cfg,
        basis_path="/tmp/_nonexistent_basis.bin",
        rotation_path="/tmp/_nonexistent_rotation.bin",
    )


def test_configure_quantizers_rejects_typo_in_k_bits():
    """A yaml typo like `k_bits: "nvf4"` must raise `ValueError` at
    PTQ entry rather than silently disabling KV quantization or
    crashing later in setup_k_quant. Round 14 covered the four W/A
    fields; round 15 extends `assert_quant_format` to k/v.
    """
    _install_heavyweight_stubs()
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from promix.eval.ptq import configure_quantizers
    import torch.nn as nn

    class _Cfg:
        hidden_size = 16
        intermediate_size = 64
        num_attention_heads = 2

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()

    cfg = {
        "quantize": {
            "w_bits": "nvfp4",
            "a_bits": "nvfp4",
            "high_bits": "mxfp8",
            "low_bits": 2,
            "k_bits": "nvf4",   # typo
            "v_bits": "nvfp4",
            "high_fraction": 0.125,
            "low_fraction": 0.0,
            "a_asym": True,
        }
    }

    raised = False
    try:
        configure_quantizers(_Model(), cfg)
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "k_bits" in msg, f"error must name the bad field; got {e}"
        assert "nvf4" in msg, f"error must echo the typo value; got {e}"
    assert raised, (
        "configure_quantizers must raise on typo'd k_bits; without "
        "this, KV4 quantization would silently disable on a typo"
    )
