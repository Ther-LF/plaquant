"""End-to-end marginal cost of upgrading one 64-channel ring FP4 -> FP8.

See docs/superpowers/specs/2026-10-28-rate-distortion-mixed-precision-allocation-design.md
section 6. The cost of serving a ring in a given format has two parts, weighted
by the deployment scenario:

  - Prefill (TTFT, compute-bound): the fused-GEMM compute increment, modelled as
    ``compute_coeff[fmt]`` (inverse-TFLOPS by default, so the slower-peak format
    carries the larger coefficient).
  - Decode (TPOT, bandwidth-bound): the weight-byte increment. NVFP4 stores
    4.5 bits/elem, MXFP8 8.5 bits/elem, so ``bytes = bits/8 * ring_size``.

``C_f(j) = w_prefill * compute_coeff[fmt] + w_decode * byte_cost[fmt]``.
``deltaC(j) = ring_cost(..., 'mxfp8') - ring_cost(..., 'nvfp4')``.

``(w_prefill, w_decode)`` are deployment inputs (prompt/generation length), not
hyperparameters. ``compute_coeff`` is initialized from the peak TFLOPS table
(docs/hardware_format_peak.md: NVFP4 1469 / FP8 807) but is overridable with
whole-net measured r-dependent marginal latency.
"""

# Bits per stored element for each format (NVFP4 4.5b, MXFP8 8.5b).
_BITS = {'nvfp4': 4.5, 'mxfp8': 8.5}

# Peak dense TFLOPS per format (docs/hardware_format_peak.md).
_PEAK_TFLOPS = {'nvfp4': 1469.0, 'mxfp8': 807.0}

# Default compute coefficients = inverse peak TFLOPS (relative compute cost).
_DEFAULT_COMPUTE_COEFF = {k: 1.0 / v for k, v in _PEAK_TFLOPS.items()}


def ring_cost(ring_size, fmt, w_prefill, w_decode,
              compute_coeff=None, n_total_rings=None):
    """e2e marginal cost of serving one ring in format ``fmt``.

    ring_size:      channels per ring (64).
    fmt:            'nvfp4' or 'mxfp8'.
    w_prefill:      prefill (compute-bound) scenario weight.
    w_decode:       decode (bandwidth-bound) scenario weight.
    compute_coeff:  {fmt: coeff}; defaults to inverse peak-TFLOPS table.
    n_total_rings:  total rings in the net (accepted for interface parity /
                    whole-net share normalization; the per-ring marginal cost
                    is already expressed by ``compute_coeff``).

    Returns ``w_prefill*compute_coeff[fmt] + w_decode*byte_cost[fmt]``.
    """
    if fmt not in _BITS:
        raise ValueError(f"fmt must be one of {list(_BITS)}, got {fmt}")
    if compute_coeff is None:
        compute_coeff = _DEFAULT_COMPUTE_COEFF
    byte_cost = _BITS[fmt] * ring_size / 8.0
    return w_prefill * compute_coeff[fmt] + w_decode * byte_cost
