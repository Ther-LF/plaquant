import pytest
from promix.quantize.cost_model import ring_cost


def test_fp8_costs_more_than_fp4():
    cc = {'nvfp4': 1.0 / 1469, 'mxfp8': 1.0 / 807}  # ~inverse TFLOPS
    c4 = ring_cost(64, 'nvfp4', w_prefill=1.0, w_decode=0.0, compute_coeff=cc, n_total_rings=48)
    c8 = ring_cost(64, 'mxfp8', w_prefill=1.0, w_decode=0.0, compute_coeff=cc, n_total_rings=48)
    assert c8 > c4


def test_decode_cost_is_bytes():
    cc = {'nvfp4': 0.0, 'mxfp8': 0.0}
    c4 = ring_cost(64, 'nvfp4', 0.0, 1.0, cc, 48)
    c8 = ring_cost(64, 'mxfp8', 0.0, 1.0, cc, 48)
    assert abs(c4 - 4.5 * 64 / 8) < 1e-6 and abs(c8 - 8.5 * 64 / 8) < 1e-6


def test_compute_coeff_defaults_from_peak_table():
    # When compute_coeff is omitted it initializes from the peak TFLOPS table
    # (NVFP4 1469 / FP8 807): coefficients are inverse-TFLOPS, so FP8 (slower
    # peak) carries a larger compute coefficient -> higher prefill cost.
    c4 = ring_cost(64, 'nvfp4', w_prefill=1.0, w_decode=0.0, n_total_rings=48)
    c8 = ring_cost(64, 'mxfp8', w_prefill=1.0, w_decode=0.0, n_total_rings=48)
    assert c8 > c4


def test_unknown_fmt_raises():
    with pytest.raises(ValueError):
        ring_cost(64, 'fp16', w_prefill=1.0, w_decode=1.0, n_total_rings=48)
