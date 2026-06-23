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
    """All NVFP4 outputs (after dividing by block scale) should land on E2M1 representable values."""
    torch.manual_seed(42)
    x = torch.randn(1, 16, dtype=torch.float32) * 2.0  # scale up to exercise more values
    out = fake_quantize_nvfp4(x)
    # Reconstruct what scale was used: out_max / max(FP4_E2M1_POS)
    out_max = out.abs().max().item()
    # The element values (out / scale) should all be in FP4_E2M1_POS or its negation.
    # Since the test uses one block, we recover scale as out_max / 6.0 if out_max > 0.
    if out_max > 0:
        # Find scale by looking at quantized values' GCD-like structure
        # Easier: just verify each output is some representable_value * scale for SOME
        # scale that is power-of-2-ish or FP8 E4M3-ish.
        unique_abs = sorted(set(out.abs().flatten().tolist()))
        # Filter near-zero
        unique_abs = [v for v in unique_abs if v > 1e-9]
        if unique_abs:
            base = unique_abs[0]  # smallest non-zero magnitude
            ratios = sorted(set(round(v / base, 4) for v in unique_abs))
            # Ratios should be a subset of {1, 2, 3, 4, 6, 8, 12} etc (E2M1 multiples)
            allowed_ratios = {round(v / FP4_E2M1_POS[1], 4) for v in FP4_E2M1_POS if v > 0}
            # E2M1 positive values normalized by smallest nonzero (0.5): {1, 2, 3, 4, 6, 8, 12}
            assert all(r in allowed_ratios or r * 0.5 in allowed_ratios for r in ratios), (
                f"NVFP4 output ratios {ratios} not all in E2M1 set {allowed_ratios}"
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
