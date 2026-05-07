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

    return {
        # Activation main (4-bit unsigned [0,15], asymmetric)
        "x_main_qint": act_main["q_int"],       # (batch, seq, K_main) int16
        "x_main_scale": act_main["scale"],       # (batch, seq, K_main) fp16 (broadcast)
        "x_main_zero": act_main["zero"],         # (batch, seq, K_main) fp16 (broadcast)
        # Activation high (8-bit unsigned [0,255], asymmetric)
        "x_high_qint": act_high["q_int"],        # (batch, seq, K_high) int16
        "x_high_scale": act_high["scale"],       # (batch, seq, K_high) fp16 (broadcast)
        "x_high_zero": act_high["zero"],         # (batch, seq, K_high) fp16 (broadcast)
        # Weight main (4-bit signed [-8,7], symmetric, per-channel)
        "w_main_qint": w_main["q_int"],          # (N, K_main) fp16
        "w_main_scale": w_main["scale"],         # (N, 1) fp16
        # Weight high (8-bit signed [-128,127], symmetric, per-channel)
        "w_high_qint": w_high["q_int"],          # (N, K_high) fp16
        "w_high_scale": w_high["scale"],         # (N, 1) fp16
        # Ground truth
        "output_real_quant": output_rq,          # (batch, seq, N) fp16
        "output_fp16_baseline": output_fp,       # (batch, seq, N) fp16
        "input_fp16": input_fp,                  # (batch, seq, K) fp16
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

        # Compute error metrics
        abs_diff = (ref_output.float() - gt_output.float()).abs()
        max_err = abs_diff.max().item()
        mean_err = abs_diff.mean().item()
        rel_err = max_err / gt_output.float().abs().max().item()

        print(f"\n  [dequant ref] max_err={max_err:.6f}, mean_err={mean_err:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 1e-2, f"Dequant reference too far from ground truth: rel_err={rel_err}"

    def test_integer_reference_matches_ground_truth(self, q_proj_dir, batch_size):
        """Integer matmul + shift/bias should match output_real_quant."""
        data = load_operator_data(q_proj_dir, batch_size)
        ref_output = reference_mixed_gemm_integer(data)
        gt_output = data["output_real_quant"]

        abs_diff = (ref_output.float() - gt_output.float()).abs()
        max_err = abs_diff.max().item()
        mean_err = abs_diff.mean().item()
        rel_err = max_err / gt_output.float().abs().max().item()

        print(f"\n  [integer ref] max_err={max_err:.6f}, mean_err={mean_err:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 1e-2, f"Integer reference too far from ground truth: rel_err={rel_err}"

    def test_dequant_vs_integer_consistency(self, q_proj_dir, batch_size):
        """Both reference implementations should produce identical results."""
        data = load_operator_data(q_proj_dir, batch_size)
        out_dequant = reference_mixed_gemm_dequant(data)
        out_integer = reference_mixed_gemm_integer(data)

        abs_diff = (out_dequant.float() - out_integer.float()).abs()
        max_err = abs_diff.max().item()
        rel_err = max_err / out_dequant.float().abs().max().item()

        print(f"\n  [consistency] max_err={max_err:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 1e-5, f"Dequant and integer refs disagree: rel_err={rel_err}"


class TestQProjDataIntegrity:
    """Sanity checks on the loaded data."""

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
        print(f"\n--- batch_size={bs} ---")
        data = load_operator_data(data_dir, bs)
        meta = load_metadata(data_dir)
        print(f"  Metadata: a_bits={meta.get('a_bits')}, a_sym={meta.get('a_sym')}")
        print(f"  x_main: {data['x_main_qint'].shape}, range [{data['x_main_qint'].min()}, {data['x_main_qint'].max()}]")
        print(f"  x_high: {data['x_high_qint'].shape}, range [{data['x_high_qint'].min()}, {data['x_high_qint'].max()}]")
        print(f"  w_main: {data['w_main_qint'].shape}, range [{data['w_main_qint'].min()}, {data['w_main_qint'].max()}]")
        print(f"  w_high: {data['w_high_qint'].shape}, range [{data['w_high_qint'].min()}, {data['w_high_qint'].max()}]")

        # Test dequant reference
        ref = reference_mixed_gemm_dequant(data)
        gt = data["output_real_quant"]
        rel_err = (ref.float() - gt.float()).abs().max().item() / gt.float().abs().max().item()
        print(f"  dequant ref vs GT: rel_err={rel_err:.6f} {'✓' if rel_err < 1e-2 else '✗'}")

        # Test integer reference
        ref_int = reference_mixed_gemm_integer(data)
        rel_err_int = (ref_int.float() - gt.float()).abs().max().item() / gt.float().abs().max().item()
        print(f"  integer ref vs GT: rel_err={rel_err_int:.6f} {'✓' if rel_err_int < 1e-2 else '✗'}")
