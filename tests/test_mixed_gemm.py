"""
ResQ Mixed-Precision GEMM Test Suite
=====================================
Tests mixed-precision INT4+INT8 GEMM correctness against ground truth
collected from the real ResQ quantization pipeline.

Data source: gemm_data/ on ceph (collected by project-resq/fake_quant/collect_gemm_data.py)
Target: kernels/mixed_gemm/

Usage:
    pytest tests/test_mixed_gemm.py -v
    pytest tests/test_mixed_gemm.py -v -k "q_proj"
    pytest tests/test_mixed_gemm.py -v -k "bs1"
"""

import os
import json
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_operator_data(data_dir: str, batch_size: int = 1):
    """Load all tensors for one operator at a given batch size.

    Returns a dict with keys:
        x_main_qint, x_main_scale, x_main_zero,
        x_high_qint, x_high_scale, x_high_zero,
        w_main_qint, w_main_scale,
        w_high_qint, w_high_scale,
        output_real_quant, output_fp16_baseline, input_fp16
    """
    d = Path(data_dir)
    bs = batch_size

    act_main = torch.load(d / f"act_quant_main_bs{bs}.pt", map_location="cpu", weights_only=False)
    act_high = torch.load(d / f"act_quant_high_bs{bs}.pt", map_location="cpu", weights_only=False)
    w_main = torch.load(d / "weight_int_main.pt", map_location="cpu", weights_only=False)
    w_high = torch.load(d / "weight_int_high.pt", map_location="cpu", weights_only=False)

    output_rq = torch.load(d / f"output_real_quant_bs{bs}.pt", map_location="cpu", weights_only=False)
    output_fp = torch.load(d / f"output_fp16_baseline_bs{bs}.pt", map_location="cpu", weights_only=False)
    input_fp = torch.load(d / f"input_fp16_bs{bs}.pt", map_location="cpu", weights_only=False)
    w_fp16 = torch.load(d / "weight_fp16.pt", map_location="cpu", weights_only=False)

    # Dtype validation: assert expected types, fail loudly if data collection was wrong
    def _assert_int16(t, name):
        assert t.dtype == torch.int16, f"{name}: expected int16, got {t.dtype}"
        return t

    def _assert_fp16(t, name):
        assert t.dtype == torch.float16, f"{name}: expected fp16, got {t.dtype}"
        return t

    return {
        # Activation main (4-bit unsigned [0,15], asymmetric)
        "x_main_qint": _assert_int16(act_main["q_int"], "act_main.q_int"),
        "x_main_scale": _assert_fp16(act_main["scale"], "act_main.scale"),
        "x_main_zero": _assert_fp16(act_main["zero"], "act_main.zero"),
        # Activation high (8-bit unsigned [0,255], asymmetric)
        "x_high_qint": _assert_int16(act_high["q_int"], "act_high.q_int"),
        "x_high_scale": _assert_fp16(act_high["scale"], "act_high.scale"),
        "x_high_zero": _assert_fp16(act_high["zero"], "act_high.zero"),
        # Weight main (4-bit signed [-8,7], symmetric, per-channel)
        "w_main_qint": _assert_fp16(w_main["q_int"], "w_main.q_int"),
        "w_main_scale": _assert_fp16(w_main["scale"], "w_main.scale"),
        # Weight high (8-bit signed [-128,127], symmetric, per-channel)
        "w_high_qint": _assert_fp16(w_high["q_int"], "w_high.q_int"),
        "w_high_scale": _assert_fp16(w_high["scale"], "w_high.scale"),
        # FP16 weight (original, for baseline verification)
        "weight_fp16": _assert_fp16(w_fp16, "weight_fp16"),
        # Ground truth
        "output_real_quant": _assert_fp16(output_rq, "output_real_quant"),
        "output_fp16_baseline": _assert_fp16(output_fp, "output_fp16_baseline"),
        "input_fp16": _assert_fp16(input_fp, "input_fp16"),
    }


def load_metadata(data_dir: str):
    """Load per-operator metadata (quantization config)."""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def reference_mixed_gemm_dequant(data: dict) -> torch.Tensor:
    """Reference: dequantize both sides to FP32, then matmul.

    This is the simplest reference — dequant everything then do FP32 GEMM.
    Should match output_real_quant exactly (same formula).
    """
    # Dequant activation: x_fp = scale * (q_int - zero)
    x_main = data["x_main_scale"].float() * (data["x_main_qint"].float() - data["x_main_zero"].float())
    x_high = data["x_high_scale"].float() * (data["x_high_qint"].float() - data["x_high_zero"].float())

    # Dequant weight: w_fp = scale * q_int (symmetric)
    w_main = data["w_main_scale"].float() * data["w_main_qint"].float()
    w_high = data["w_high_scale"].float() * data["w_high_qint"].float()

    # x: (batch, seq, K_main/K_high), w: (N, K_main/K_high)
    # output = x @ w^T
    out_main = torch.matmul(x_main, w_main.t())
    out_high = torch.matmul(x_high, w_high.t())

    return (out_main + out_high).half()


def reference_mixed_gemm_integer(data: dict) -> torch.Tensor:
    """Reference: integer matmul with shift+bias trick (simulates Tensor Core).

    This matches what our CUDA kernel should compute:
    Y = s_x_m * s_w_m * (q_x_shifted_m @ q_w_m^T + bias_m)
      + s_x_h * s_w_h * (q_x_shifted_h @ q_w_h^T + bias_h)

    Where:
        q_x_shifted = q_x - shift (shift=8 for 4-bit, shift=128 for 8-bit)
        bias = (shift - zero) * colsum(q_w)
    """
    # Extract per-token scales (take first column since broadcast)
    s_x_m = data["x_main_scale"][:, :, :1].float()   # (batch, seq, 1)
    z_x_m = data["x_main_zero"][:, :, :1].float()    # (batch, seq, 1)
    s_x_h = data["x_high_scale"][:, :, :1].float()
    z_x_h = data["x_high_zero"][:, :, :1].float()

    s_w_m = data["w_main_scale"].float()              # (N, 1)
    s_w_h = data["w_high_scale"].float()

    q_x_m = data["x_main_qint"].float()              # (batch, seq, K_main) [0, 15]
    q_x_h = data["x_high_qint"].float()              # (batch, seq, K_high) [0, 255]
    q_w_m = data["w_main_qint"].float()              # (N, K_main) [-8, 7]
    q_w_h = data["w_high_qint"].float()              # (N, K_high) [-128, 127]

    # Shift activations to signed: 4-bit shift=8, 8-bit shift=128
    shift_m = 8.0
    shift_h = 128.0

    q_x_m_shifted = q_x_m - shift_m                  # [-8, 7]
    q_x_h_shifted = q_x_h - shift_h                  # [-128, 127]

    # Integer matmul (simulates INT32 accumulator)
    int_out_m = torch.matmul(q_x_m_shifted, q_w_m.t())  # (batch, seq, N)
    int_out_h = torch.matmul(q_x_h_shifted, q_w_h.t())

    # Bias correction: (shift - zero) * colsum(q_w)
    colsum_w_m = q_w_m.sum(dim=1, keepdim=True).t()     # (1, N)
    colsum_w_h = q_w_h.sum(dim=1, keepdim=True).t()

    bias_m = (shift_m - z_x_m) * colsum_w_m             # (batch, seq, N)
    bias_h = (shift_h - z_x_h) * colsum_w_h

    # Final: scale * (int_matmul + bias)
    out_m = s_x_m * s_w_m.t() * (int_out_m + bias_m)
    out_h = s_x_h * s_w_h.t() * (int_out_h + bias_h)

    return (out_m + out_h).half()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(output: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """Compute comprehensive error metrics between output and ground truth.

    All computations done in FP32 for precision.

    Returns dict with:
        max_abs_err:    max |output - gt|
        mean_abs_err:   mean |output - gt|
        rel_err:        max |output - gt| / max |gt|
        rmse:           root mean square error
        cosine_sim:     cosine similarity (1.0 = identical direction)
        snr_db:         signal-to-noise ratio in dB (higher = better)
                        SNR = 10 * log10(signal_power / noise_power)
        max_ulp_err:    max error in FP16 ULP (units of least precision)
    """
    out = output.float().flatten()
    gt = ground_truth.float().flatten()
    diff = out - gt

    max_abs_err = diff.abs().max().item()
    mean_abs_err = diff.abs().mean().item()
    rel_err = max_abs_err / gt.abs().max().item() if gt.abs().max() > 0 else float('inf')
    rmse = diff.pow(2).mean().sqrt().item()

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(out.unsqueeze(0), gt.unsqueeze(0)).item()

    # SNR: signal power / noise power (in dB)
    signal_power = gt.pow(2).mean().item()
    noise_power = diff.pow(2).mean().item()
    if noise_power > 0:
        snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
    else:
        snr_db = float('inf')

    # ULP error for FP16: |diff| / eps_at_that_magnitude
    # FP16 epsilon at magnitude x is approximately x * 2^-10 (for normal numbers)
    # Filter out near-zero elements where ULP is meaninglessly small
    ulp_mask = gt.abs() > 0.01  # ignore elements with |gt| < 0.01
    if ulp_mask.any():
        gt_masked = gt[ulp_mask].abs()
        diff_masked = diff[ulp_mask].abs()
        ulp_size = gt_masked * (2.0 ** -10)  # FP16 has 10-bit mantissa
        ulp_err = diff_masked / ulp_size
        max_ulp_err = ulp_err.max().item()
        mean_ulp_err = ulp_err.mean().item()
    else:
        max_ulp_err = 0.0
        mean_ulp_err = 0.0

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rel_err": rel_err,
        "rmse": rmse,
        "cosine_sim": cosine_sim,
        "snr_db": snr_db,
        "max_ulp_err": max_ulp_err,
        "mean_ulp_err": mean_ulp_err,
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print metrics."""
    print(f"\n  {prefix}")
    print(f"    max_abs_err:  {metrics['max_abs_err']:.6f}")
    print(f"    mean_abs_err: {metrics['mean_abs_err']:.6f}")
    print(f"    rel_err:      {metrics['rel_err']:.6f}")
    print(f"    rmse:         {metrics['rmse']:.6f}")
    print(f"    cosine_sim:   {metrics['cosine_sim']:.8f}")
    print(f"    snr_db:       {metrics['snr_db']:.2f} dB")
    print(f"    max_ulp_err:  {metrics['max_ulp_err']:.1f} ULP")
    print(f"    mean_ulp_err: {metrics['mean_ulp_err']:.1f} ULP")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

GEMM_DATA_DIR = "/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_libra/user_spanaluo/plaquant/gemm_data"

OPERATORS = {
    "q_proj": "model_layers_0_self_attn_q_proj",
    "k_proj": "model_layers_0_self_attn_k_proj",
    "v_proj": "model_layers_0_self_attn_v_proj",
    "o_proj": "model_layers_0_self_attn_o_proj",
    "gate_proj": "model_layers_0_mlp_gate_proj",
    "up_proj": "model_layers_0_mlp_up_proj",
    "down_proj": "model_layers_0_mlp_down_proj",
}


@pytest.fixture(params=[1, 2, 4], ids=["bs1", "bs2", "bs4"])
def batch_size(request):
    return request.param


@pytest.fixture
def q_proj_dir():
    path = os.path.join(GEMM_DATA_DIR, OPERATORS["q_proj"])
    if not os.path.exists(path):
        pytest.skip(f"Data not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Tests: q_proj (Operator 1: per-token asymmetric activation, per-channel weight)
# ---------------------------------------------------------------------------

class TestQProjReference:
    """Verify our reference implementations match the ground truth."""

    def test_dequant_reference_matches_ground_truth(self, q_proj_dir, batch_size):
        """Dequant-then-matmul should match output_real_quant."""
        data = load_operator_data(q_proj_dir, batch_size)
        ref_output = reference_mixed_gemm_dequant(data)
        gt_output = data["output_real_quant"]

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[dequant ref, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 40, f"SNR too low: {metrics['snr_db']} dB"

    def test_integer_reference_matches_ground_truth(self, q_proj_dir, batch_size):
        """Integer matmul + shift/bias should match output_real_quant."""
        data = load_operator_data(q_proj_dir, batch_size)
        ref_output = reference_mixed_gemm_integer(data)
        gt_output = data["output_real_quant"]

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[integer ref, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 40, f"SNR too low: {metrics['snr_db']} dB"

    def test_dequant_vs_integer_consistency(self, q_proj_dir, batch_size):
        """Both reference implementations should produce identical results."""
        data = load_operator_data(q_proj_dir, batch_size)
        out_dequant = reference_mixed_gemm_dequant(data)
        out_integer = reference_mixed_gemm_integer(data)

        metrics = compute_metrics(out_dequant, out_integer)
        print_metrics(metrics, prefix=f"[dequant vs integer, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-5, f"Two refs disagree: rel_err={metrics['rel_err']}"
        assert metrics["max_ulp_err"] < 1.0, f"ULP error > 1: {metrics['max_ulp_err']}"


class TestQProjDataIntegrity:
    """Sanity checks on the loaded data."""

    def test_fp16_baseline_consistency(self, q_proj_dir, batch_size):
        """Verify input_fp16 @ weight_fp16.T == output_fp16_baseline."""
        data = load_operator_data(q_proj_dir, batch_size)
        input_fp = data["input_fp16"].float()
        w_fp = data["weight_fp16"].float()
        expected = data["output_fp16_baseline"]

        recomputed = torch.matmul(input_fp, w_fp.t()).half()
        metrics = compute_metrics(recomputed, expected)
        print_metrics(metrics, prefix=f"[fp16 baseline verify, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"FP16 baseline mismatch: rel_err={metrics['rel_err']}"

    def test_activation_value_ranges(self, q_proj_dir, batch_size):
        """Activation integers should be in expected ranges."""
        data = load_operator_data(q_proj_dir, batch_size)

        # 4-bit unsigned: [0, 15]
        assert data["x_main_qint"].min() >= 0
        assert data["x_main_qint"].max() <= 15

        # 8-bit unsigned: [0, 255]
        assert data["x_high_qint"].min() >= 0
        assert data["x_high_qint"].max() <= 255

    def test_weight_value_ranges(self, q_proj_dir, batch_size):
        """Weight integers should be in expected ranges."""
        data = load_operator_data(q_proj_dir, batch_size)

        # 4-bit signed: [-8, 7]
        assert data["w_main_qint"].min() >= -8
        assert data["w_main_qint"].max() <= 7

        # 8-bit signed: [-128, 127]
        assert data["w_high_qint"].min() >= -128
        assert data["w_high_qint"].max() <= 127

    def test_shapes(self, q_proj_dir, batch_size):
        """Verify tensor shapes match q_proj spec."""
        data = load_operator_data(q_proj_dir, batch_size)
        meta = load_metadata(q_proj_dir)

        # q_proj: (M, 2048) → (M, 2048), K_main=1792, K_high=256
        M = batch_size * 2048  # batch * seqlen
        N = 2048
        K_main = 1792
        K_high = 256

        assert data["x_main_qint"].shape == (batch_size, 2048, K_main)
        assert data["x_high_qint"].shape == (batch_size, 2048, K_high)
        assert data["w_main_qint"].shape == (N, K_main)
        assert data["w_high_qint"].shape == (N, K_high)
        assert data["output_real_quant"].shape == (batch_size, 2048, N)


# ---------------------------------------------------------------------------
# Tests: CUDA kernel (placeholder — will be filled when kernel is ready)
# ---------------------------------------------------------------------------

class TestQProjKernel:
    """Test our CUDA mixed_gemm kernel against ground truth."""

    @pytest.mark.skip(reason="Kernel not implemented yet")
    def test_kernel_matches_ground_truth(self, q_proj_dir, batch_size):
        """Our kernel output should match output_real_quant."""
        # import mixed_gemm
        # data = load_operator_data(q_proj_dir, batch_size)
        # kernel_output = mixed_gemm.forward(...)
        # gt_output = data["output_real_quant"]
        # assert close(kernel_output, gt_output)
        pass

    @pytest.mark.skip(reason="Kernel not implemented yet")
    def test_kernel_performance(self, q_proj_dir):
        """Benchmark: kernel should be faster than two separate GEMMs."""
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick standalone test without pytest
    print("Loading q_proj data...")
    data_dir = os.path.join(GEMM_DATA_DIR, OPERATORS["q_proj"])
    if not os.path.exists(data_dir):
        print(f"ERROR: Data not found at {data_dir}")
        exit(1)

    for bs in [1, 2, 4]:
        print(f"\n{'='*60}")
        print(f" batch_size={bs}")
        print(f"{'='*60}")
        data = load_operator_data(data_dir, bs)
        meta = load_metadata(data_dir)
        print(f"  Config: a_bits={meta.get('a_bits')}, a_sym={meta.get('a_sym')}")
        print(f"  x_main: {data['x_main_qint'].shape}, range [{data['x_main_qint'].min()}, {data['x_main_qint'].max()}]")
        print(f"  x_high: {data['x_high_qint'].shape}, range [{data['x_high_qint'].min()}, {data['x_high_qint'].max()}]")
        print(f"  w_main: {data['w_main_qint'].shape}, range [{data['w_main_qint'].min()}, {data['w_main_qint'].max()}]")
        print(f"  w_high: {data['w_high_qint'].shape}, range [{data['w_high_qint'].min()}, {data['w_high_qint'].max()}]")

        gt = data["output_real_quant"]

        # Dequant reference
        ref_dq = reference_mixed_gemm_dequant(data)
        m = compute_metrics(ref_dq, gt)
        print_metrics(m, prefix="[dequant ref vs GT]")

        # Integer reference
        ref_int = reference_mixed_gemm_integer(data)
        m = compute_metrics(ref_int, gt)
        print_metrics(m, prefix="[integer ref vs GT]")

        # Consistency
        m = compute_metrics(ref_dq, ref_int)
        print_metrics(m, prefix="[dequant vs integer]")
