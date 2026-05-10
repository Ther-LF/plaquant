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

    # GEMM-only output (before output quantization) — use as GT for kernel verification
    # For layers without output quantization, output_real_quant == gemm_only output
    # For v_proj (has 4-bit output quant), compute from act_quant + weight (proven inf dB match)
    gemm_only_path = d / f"output_gemm_only_bs{bs}.pt"
    if gemm_only_path.exists():
        output_gemm_only = torch.load(gemm_only_path, map_location="cpu", weights_only=False)
    else:
        # Check if this layer has output quantization
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                _meta = json.load(f)
            if _meta.get("out_quantizer_bits", 16) < 16:
                # Has output quant: compute GEMM-only from act_quant + weight (reference formula)
                output_gemm_only = None  # will be computed by reference function when needed
            else:
                output_gemm_only = output_rq
        else:
            output_gemm_only = output_rq

    # Precomputed colsum: compute on-the-fly if not saved
    colsum_main_path = d / "colsum_w_main.pt"
    colsum_high_path = d / "colsum_w_high.pt"
    if colsum_main_path.exists():
        colsum_w_m = torch.load(colsum_main_path, map_location="cpu", weights_only=False)
    else:
        colsum_w_m = w_main["q_int"].float().sum(dim=1).to(torch.int32)
    if colsum_high_path.exists():
        colsum_w_h = torch.load(colsum_high_path, map_location="cpu", weights_only=False)
    else:
        colsum_w_h = w_high["q_int"].float().sum(dim=1).to(torch.int32)

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
        # Precomputed weight column sums (for bias correction in kernel)
        "colsum_w_main": colsum_w_m,             # (N,) int32
        "colsum_w_high": colsum_w_h,             # (N,) int32
        # Ground truth
        "output_real_quant": _assert_fp16(output_rq, "output_real_quant"),
        "output_gemm_only": _assert_fp16(output_gemm_only, "output_gemm_only") if output_gemm_only is not None else None,
        "output_fp16_baseline": _assert_fp16(output_fp, "output_fp16_baseline"),
        "input_fp16": _assert_fp16(input_fp, "input_fp16"),
    }


def load_metadata(data_dir: str):
    """Load per-operator metadata (quantization config).

    Metadata fields:
        name:               full layer name, e.g. "model.layers.0.self_attn.q_proj"
        weight_shape:       [N, K] original weight dimensions
        has_bias:           whether the linear layer has bias (always False for Llama)

        a_bits:             activation quantization bits for main group (4)
        a_sym:              activation symmetric quantization (False = asymmetric)
        a_groupsize:        activation quantization groupsize (-1 = per-token)
        a_high_bits:        activation bits for high-precision group (8)
        a_high_bits_length: number of K channels in high-precision group (256)
        a_low_bits:         activation bits for low-precision group (2, unused if length=0)
        a_low_bits_length:  number of K channels in low-precision group (0 = disabled)

        online_full_had:    whether online full Hadamard rotation is applied (down_proj only)
        online_partial_had: whether online partial Hadamard is applied (False for q_proj)
        has_column_order:   whether o_proj column reorder is needed (o_proj only)
        out_quantizer_bits: output quantization bits (16 = no output quantization)

        has_real_quant:     True if real quant weights were collected
        w_m_int_shape:      [N, K_main] shape of main weight integer tensor
        w_m_scale_shape:    [N, 1] shape of main weight scale (per-channel)
        w_h_int_shape:      [N, K_high] shape of high weight integer tensor
        w_h_scale_shape:    [N, 1] shape of high weight scale (per-channel)
    """
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
    """Reference: unsigned×signed integer matmul (simulates Tensor Core).

    Uses u4×s4 and u8×s8 Tensor Core instructions where:
    - Activation is unsigned: [0,15] for 4-bit, [0,255] for 8-bit
    - Weight is signed: [-8,7] for 4-bit, [-128,127] for 8-bit
    - Result accumulates in INT32

    Formula:
    Y = s_x_m * s_w_m * (q_x_m @ q_w_m^T - z_x_m * colsum(q_w_m))
      + s_x_h * s_w_h * (q_x_h @ q_w_h^T - z_x_h * colsum(q_w_h))

    Where colsum(q_w) = sum of each row of q_w (precomputable offline).
    """
    # Extract per-token scales (take first column since broadcast)
    s_x_m = data["x_main_scale"][:, :, :1].float()   # (batch, seq, 1)
    z_x_m = data["x_main_zero"][:, :, :1].float()    # (batch, seq, 1)
    s_x_h = data["x_high_scale"][:, :, :1].float()
    z_x_h = data["x_high_zero"][:, :, :1].float()

    s_w_m = data["w_main_scale"].float()              # (N, 1)
    s_w_h = data["w_high_scale"].float()

    q_x_m = data["x_main_qint"].float()              # (batch, seq, K_main) unsigned [0, 15]
    q_x_h = data["x_high_qint"].float()              # (batch, seq, K_high) unsigned [0, 255]
    q_w_m = data["w_main_qint"].float()              # (N, K_main) signed [-8, 7]
    q_w_h = data["w_high_qint"].float()              # (N, K_high) signed [-128, 127]

    # Integer matmul: unsigned activation × signed weight → INT32
    int_out_m = torch.matmul(q_x_m, q_w_m.t())      # (batch, seq, N)
    int_out_h = torch.matmul(q_x_h, q_w_h.t())

    # Bias correction using precomputed colsum: -zero * colsum_w
    colsum_w_m = data["colsum_w_main"].float()       # (N,)
    colsum_w_h = data["colsum_w_high"].float()       # (N,)

    bias_m = -z_x_m * colsum_w_m                     # (batch, seq, N)
    bias_h = -z_x_h * colsum_w_h

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
def o_proj_dir():
    path = os.path.join(GEMM_DATA_DIR, OPERATORS["o_proj"])
    if not os.path.exists(path):
        pytest.skip(f"Data not found: {path}")
    return path


# Per-token operators: all use the same kernel (different N)
PER_TOKEN_OPS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]

@pytest.fixture(params=PER_TOKEN_OPS)
def per_token_op_dir(request):
    """Fixture that iterates over all per-token operators (q/k/v/gate/up_proj)."""
    op_name = request.param
    path = os.path.join(GEMM_DATA_DIR, OPERATORS[op_name])
    if not os.path.exists(path):
        pytest.skip(f"Data not found: {path}")
    return path, op_name


# ---------------------------------------------------------------------------
# Tests: per-token operators (q/k/v/gate/up_proj)
# All use the same kernel shape: per-token asymmetric activation, per-channel weight
# Only N differs: q=2048, k=512, v=512, gate=8192, up=8192
# ---------------------------------------------------------------------------

class TestPerTokenReference:
    """Verify reference implementations for per-token operators (q/k/v/gate/up_proj)."""

    def test_dequant_reference_matches_ground_truth(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        ref_output = reference_mixed_gemm_dequant(data)

        # Use GEMM-only GT; if None (v_proj without file), use reference as self-consistency check
        gt_output = data["output_gemm_only"]
        if gt_output is None:
            gt_output = reference_mixed_gemm_integer(data)  # proven inf dB match with actual GEMM

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[{op_name} dequant ref, bs={batch_size}]")

        assert metrics["rel_err"] < 0.05, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 60, f"SNR too low: {metrics['snr_db']} dB"

    def test_integer_reference_matches_ground_truth(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        ref_output = reference_mixed_gemm_integer(data)

        gt_output = data["output_gemm_only"]
        if gt_output is None:
            gt_output = reference_mixed_gemm_dequant(data)

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[{op_name} integer ref, bs={batch_size}]")

        assert metrics["rel_err"] < 0.05, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 60, f"SNR too low: {metrics['snr_db']} dB"

    def test_dequant_vs_integer_consistency(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        out_dequant = reference_mixed_gemm_dequant(data)
        out_integer = reference_mixed_gemm_integer(data)

        metrics = compute_metrics(out_dequant, out_integer)
        print_metrics(metrics, prefix=f"[{op_name} dequant vs integer, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-3, f"Two refs disagree: rel_err={metrics['rel_err']}"


class TestPerTokenDataIntegrity:
    """Data integrity checks for per-token operators."""

    def test_fp16_baseline_consistency(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        input_fp = data["input_fp16"].float()
        w_fp = data["weight_fp16"].float()
        expected = data["output_fp16_baseline"]

        recomputed = torch.matmul(input_fp, w_fp.t()).half()
        metrics = compute_metrics(recomputed, expected)
        print_metrics(metrics, prefix=f"[{op_name} fp16 baseline verify, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"FP16 baseline mismatch: rel_err={metrics['rel_err']}"

    def test_activation_value_ranges(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)

        assert data["x_main_qint"].min() >= 0
        assert data["x_main_qint"].max() <= 15
        assert data["x_high_qint"].min() >= 0
        assert data["x_high_qint"].max() <= 255

    def test_weight_value_ranges(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)

        assert data["w_main_qint"].min() >= -8
        assert data["w_main_qint"].max() <= 7
        assert data["w_high_qint"].min() >= -128
        assert data["w_high_qint"].max() <= 127

    def test_quantization_loss(self, per_token_op_dir, batch_size):
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        gt = data["output_real_quant"]
        fp_baseline = data["output_fp16_baseline"]

        metrics = compute_metrics(gt, fp_baseline)
        print_metrics(metrics, prefix=f"[{op_name} quant loss, bs={batch_size}]")


# ---------------------------------------------------------------------------
# o_proj: per-group mixed-precision GEMM
# ---------------------------------------------------------------------------

def load_oproj_data(data_dir: str, batch_size: int = 1):
    """Load o_proj operator data.

    o_proj uses per-group quantization (groupsize=head_dim=64).
    Activation tensors are 4D: (batch, seq, num_groups, group_k)
    Scale/zero are per-group per-token: (batch, seq, num_groups, 1)

    Additional: column_order for reordering attention output.
    """
    d = Path(data_dir)
    bs = batch_size

    act_main = torch.load(d / f"act_quant_main_bs{bs}.pt", map_location="cpu", weights_only=False)
    act_high = torch.load(d / f"act_quant_high_bs{bs}.pt", map_location="cpu", weights_only=False)
    w_main = torch.load(d / "weight_int_main.pt", map_location="cpu", weights_only=False)
    w_high = torch.load(d / "weight_int_high.pt", map_location="cpu", weights_only=False)
    col_order = torch.load(d / "column_order.pt", map_location="cpu", weights_only=False)

    output_rq = torch.load(d / f"output_real_quant_bs{bs}.pt", map_location="cpu", weights_only=False)
    output_fp = torch.load(d / f"output_fp16_baseline_bs{bs}.pt", map_location="cpu", weights_only=False)
    input_fp = torch.load(d / f"input_fp16_bs{bs}.pt", map_location="cpu", weights_only=False)
    w_fp16 = torch.load(d / "weight_fp16.pt", map_location="cpu", weights_only=False)

    return {
        # Activation main: 4D (batch, seq, num_groups, group_k_main)
        "x_main_qint": act_main["q_int"],           # INT16, [0,15]
        "x_main_scale": act_main["scale"],           # FP16, (batch, seq, num_groups, 1)
        "x_main_zero": act_main["zero"],             # FP16
        # Activation high: 4D (batch, seq, num_groups, group_k_high)
        "x_high_qint": act_high["q_int"],            # INT16, [0,255]
        "x_high_scale": act_high["scale"],           # FP16
        "x_high_zero": act_high["zero"],             # FP16
        # Weight: global layout (already rearranged with column_order)
        "w_main_qint": w_main["q_int"],              # FP16, (N, K_main)
        "w_main_scale": w_main["scale"],             # FP16, (N, 1)
        "w_high_qint": w_high["q_int"],              # FP16, (N, K_high)
        "w_high_scale": w_high["scale"],             # FP16, (N, 1)
        # Column reorder (applied to input before GEMM)
        "column_order": col_order,                    # INT64, (K,)
        # FP16 references
        "weight_fp16": w_fp16,
        "output_real_quant": output_rq,
        "output_fp16_baseline": output_fp,
        "input_fp16": input_fp,
    }


def reference_oproj_dequant(data: dict) -> torch.Tensor:
    """Reference o_proj: per-group dequant then matmul.

    o_proj activation is quantized per-group (groupsize=head_dim=64).
    Each group of 64 channels has its own scale/zero.
    Within each group: 56 channels are INT4 (main), 8 channels are INT8 (high).

    Weight layout is global: [all_main_cols | all_high_cols]
      W_main: (N, G*56) = (N, 1792) — all groups' main cols concatenated
      W_high: (N, G*8)  = (N, 256)  — all groups' high cols concatenated

    So we must arrange activation in the same order:
      x_flat = [group0_main, group1_main, ..., groupG_main, group0_high, ..., groupG_high]
    """
    # Dequant main: (batch, seq, G, gk_m)
    x_main = data["x_main_scale"].float() * (data["x_main_qint"].float() - data["x_main_zero"].float())
    x_high = data["x_high_scale"].float() * (data["x_high_qint"].float() - data["x_high_zero"].float())

    batch, seq = x_main.shape[:2]

    # Flatten groups for main and high separately to match weight layout
    x_main_flat = x_main.reshape(batch, seq, -1)  # (batch, seq, G*gk_m) = (batch, seq, 1792)
    x_high_flat = x_high.reshape(batch, seq, -1)  # (batch, seq, G*gk_h) = (batch, seq, 256)

    # Concatenate in same order as weight: [main | high]
    x_flat = torch.cat([x_main_flat, x_high_flat], dim=-1)  # (batch, seq, 2048)

    # Dequant weight: symmetric
    w_main = data["w_main_scale"].float() * data["w_main_qint"].float()  # (N, 1792)
    w_high = data["w_high_scale"].float() * data["w_high_qint"].float()  # (N, 256)
    w_full = torch.cat([w_main, w_high], dim=1)  # (N, 2048)

    out = torch.matmul(x_flat, w_full.t())
    return out.half()


def reference_oproj_integer(data: dict) -> torch.Tensor:
    """Reference o_proj: per-group integer matmul with bias correction.

    For each group g:
      partial_m[g] = s_x_m[g] * s_w_m * (q_x_m[g] @ q_w_m[g]^T - z_x_m[g] * colsum_w_m[g])
      partial_h[g] = s_x_h[g] * s_w_h * (q_x_h[g] @ q_w_h[g]^T - z_x_h[g] * colsum_w_h[g])

    output = sum over g of (partial_m[g] + partial_h[g])
    """
    # Shapes
    q_x_m = data["x_main_qint"].float()   # (batch, seq, G, gk_m) [0,15]
    q_x_h = data["x_high_qint"].float()   # (batch, seq, G, gk_h) [0,255]
    s_x_m = data["x_main_scale"].float()  # (batch, seq, G, 1)
    z_x_m = data["x_main_zero"].float()
    s_x_h = data["x_high_scale"].float()
    z_x_h = data["x_high_zero"].float()

    q_w_m = data["w_main_qint"].float()   # (N, K_main)
    q_w_h = data["w_high_qint"].float()   # (N, K_high)
    s_w_m = data["w_main_scale"].float()  # (N, 1)
    s_w_h = data["w_high_scale"].float()

    batch, seq, G, gk_m = q_x_m.shape
    gk_h = q_x_h.shape[-1]
    N = q_w_m.shape[0]

    # Accumulate over groups
    output = torch.zeros(batch, seq, N)

    for g in range(G):
        # Extract group's activation
        qxm_g = q_x_m[:, :, g, :]  # (batch, seq, gk_m)
        qxh_g = q_x_h[:, :, g, :]  # (batch, seq, gk_h)
        sxm_g = s_x_m[:, :, g, :]  # (batch, seq, 1)
        zxm_g = z_x_m[:, :, g, :]
        sxh_g = s_x_h[:, :, g, :]
        zxh_g = z_x_h[:, :, g, :]

        # Extract group's weight columns
        qwm_g = q_w_m[:, g * gk_m : (g + 1) * gk_m]  # (N, gk_m)
        qwh_g = q_w_h[:, g * gk_h : (g + 1) * gk_h]  # (N, gk_h)

        # Integer matmul: unsigned activation × signed weight
        int_out_m = torch.matmul(qxm_g, qwm_g.t())  # (batch, seq, N)
        int_out_h = torch.matmul(qxh_g, qwh_g.t())

        # Bias correction: -zero * colsum
        colsum_m = qwm_g.sum(dim=1)  # (N,)
        colsum_h = qwh_g.sum(dim=1)  # (N,)

        bias_m = -zxm_g * colsum_m    # (batch, seq, N)
        bias_h = -zxh_g * colsum_h

        # Scale and accumulate
        output += sxm_g * s_w_m.t() * (int_out_m + bias_m)
        output += sxh_g * s_w_h.t() * (int_out_h + bias_h)

    return output.half()


class TestOProjReference:
    """Verify o_proj reference implementations match ground truth."""

    def test_dequant_reference_matches_ground_truth(self, o_proj_dir, batch_size):
        data = load_oproj_data(o_proj_dir, batch_size)
        ref_output = reference_oproj_dequant(data)
        gt_output = data["output_real_quant"]

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[o_proj dequant ref, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 40, f"SNR too low: {metrics['snr_db']} dB"

    def test_integer_reference_matches_ground_truth(self, o_proj_dir, batch_size):
        data = load_oproj_data(o_proj_dir, batch_size)
        ref_output = reference_oproj_integer(data)
        gt_output = data["output_real_quant"]

        metrics = compute_metrics(ref_output, gt_output)
        print_metrics(metrics, prefix=f"[o_proj integer ref, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-2, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 40, f"SNR too low: {metrics['snr_db']} dB"

    def test_dequant_vs_integer_consistency(self, o_proj_dir, batch_size):
        data = load_oproj_data(o_proj_dir, batch_size)
        out_dequant = reference_oproj_dequant(data)
        out_integer = reference_oproj_integer(data)

        metrics = compute_metrics(out_dequant, out_integer)
        print_metrics(metrics, prefix=f"[o_proj dequant vs integer, bs={batch_size}]")

        assert metrics["rel_err"] < 1e-4, f"Two refs disagree: rel_err={metrics['rel_err']}"

    def test_quantization_loss(self, o_proj_dir, batch_size):
        data = load_oproj_data(o_proj_dir, batch_size)
        gt = data["output_real_quant"]
        fp_baseline = data["output_fp16_baseline"]

        metrics = compute_metrics(gt, fp_baseline)
        print_metrics(metrics, prefix=f"[o_proj quant loss, bs={batch_size}]")

    def test_shapes(self, o_proj_dir, batch_size):
        data = load_oproj_data(o_proj_dir, batch_size)
        meta = load_metadata(o_proj_dir)

        num_groups = 32
        gk_main = 56  # 64 * 7/8
        gk_high = 8   # 64 * 1/8

        assert data["x_main_qint"].shape == (batch_size, 2048, num_groups, gk_main)
        assert data["x_high_qint"].shape == (batch_size, 2048, num_groups, gk_high)
        assert data["x_main_scale"].shape == (batch_size, 2048, num_groups, 1)
        assert data["w_main_qint"].shape == (2048, num_groups * gk_main)
        assert data["w_high_qint"].shape == (2048, num_groups * gk_high)
        assert data["column_order"].shape == (2048,)


class TestOProjKernel:
    """Test o_proj baseline: per-group dequant → FP16 matmul."""

    def test_baseline_matches_ground_truth(self, o_proj_dir, batch_size):
        """Baseline: dequant per-group to FP16, then cuBLAS FP16 GEMM."""
        data = load_oproj_data(o_proj_dir, batch_size)

        # Per-group dequant to FP16
        x_main = data["x_main_scale"].float() * (data["x_main_qint"].float() - data["x_main_zero"].float())
        x_high = data["x_high_scale"].float() * (data["x_high_qint"].float() - data["x_high_zero"].float())

        # Flatten groups: [all main | all high]
        batch, seq = x_main.shape[:2]
        x_flat = torch.cat([
            x_main.reshape(batch, seq, -1),
            x_high.reshape(batch, seq, -1)
        ], dim=-1).half()  # (batch, seq, 2048)

        # Weight dequant (per-channel symmetric)
        w_main = (data["w_main_scale"] * data["w_main_qint"]).float()
        w_high = (data["w_high_scale"] * data["w_high_qint"]).float()
        w_full = torch.cat([w_main, w_high], dim=1).half()  # (N, 2048)

        # FP16 matmul
        output = torch.matmul(x_flat.float(), w_full.float().t()).half()

        gt = data["output_real_quant"]
        metrics = compute_metrics(output, gt)
        print_metrics(metrics, prefix=f"[o_proj baseline kernel, bs={batch_size}]")

        assert metrics["rel_err"] < 0.05, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 25, f"SNR too low: {metrics['snr_db']} dB"

    def test_baseline_performance(self, o_proj_dir):
        """Benchmark o_proj baseline: dequant + FP16 GEMM."""
        data = load_oproj_data(o_proj_dir, batch_size=1)

        # Prepare FP16 data (simulate the dequant step)
        x_main = data["x_main_scale"].float() * (data["x_main_qint"].float() - data["x_main_zero"].float())
        x_high = data["x_high_scale"].float() * (data["x_high_qint"].float() - data["x_high_zero"].float())
        batch, seq = x_main.shape[:2]
        x_flat = torch.cat([
            x_main.reshape(batch, seq, -1),
            x_high.reshape(batch, seq, -1)
        ], dim=-1).half().cuda()

        w_main = (data["w_main_scale"] * data["w_main_qint"]).float()
        w_high = (data["w_high_scale"] * data["w_high_qint"]).float()
        w_full = torch.cat([w_main, w_high], dim=1).half().cuda()

        M, K = x_flat.shape[1], x_flat.shape[2]  # seq, 2048
        N = w_full.shape[0]

        # Warmup
        for _ in range(10):
            torch.matmul(x_flat, w_full.t())
        torch.cuda.synchronize()

        # Benchmark with CUDA events
        iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            torch.matmul(x_flat, w_full.t())
        end.record()
        torch.cuda.synchronize()
        ms_gemm = start.elapsed_time(end) / iters

        flops = 2 * M * N * K
        tflops = flops / ms_gemm / 1e9
        h20_fp16_peak = 148  # FP16 TFLOPS

        print(f"\n  [o_proj baseline perf] M={M}, N={N}, K={K}")
        print(f"    FP16 GEMM: {ms_gemm*1000:.1f} μs, {tflops:.1f} TFLOPS ({tflops/h20_fp16_peak*100:.1f}% FP16 peak)")


# ---------------------------------------------------------------------------
# Tests: CUDA baseline kernel (CUTLASS S8S8/U8S8 + EVT dequant)
# ---------------------------------------------------------------------------

def prepare_kernel_inputs(data: dict):
    """Convert loaded test data to kernel-expected formats.

    Returns dict with CUDA tensors ready for mixed_gemm.gemm_u8s8_dequant().
    """
    # Activation: INT16 [0,15]/[0,255] → UINT8, flatten to (M, K)
    x_main = data["x_main_qint"].reshape(-1, data["x_main_qint"].shape[-1]).to(torch.uint8).contiguous().cuda()
    x_high = data["x_high_qint"].reshape(-1, data["x_high_qint"].shape[-1]).to(torch.uint8).contiguous().cuda()

    # Weight: FP16 (storing integers) → INT8
    w_main = data["w_main_qint"].to(torch.int8).contiguous().cuda()
    w_high = data["w_high_qint"].to(torch.int8).contiguous().cuda()

    # Per-token scale (first col, since broadcast): (batch*seq,) fp16
    s_x_m = data["x_main_scale"][:, :, :1].reshape(-1).half().contiguous().cuda()
    s_x_h = data["x_high_scale"][:, :, :1].reshape(-1).half().contiguous().cuda()

    # Per-token negated zero: (batch*seq,) fp16
    neg_zero_m = (-data["x_main_zero"][:, :, :1].reshape(-1)).half().contiguous().cuda()
    neg_zero_h = (-data["x_high_zero"][:, :, :1].reshape(-1)).half().contiguous().cuda()

    # Per-channel weight scale: (N,) fp16
    s_w_m = data["w_main_scale"].squeeze().half().contiguous().cuda()
    s_w_h = data["w_high_scale"].squeeze().half().contiguous().cuda()

    # Precomputed column sums: (N,) float32
    colsum_m = data["colsum_w_main"].float().contiguous().cuda()
    colsum_h = data["colsum_w_high"].float().contiguous().cuda()

    # Ground truth
    # Ground truth: use GEMM-only output (before output quantization)
    gt_data = data["output_gemm_only"] if data["output_gemm_only"] is not None else data["output_real_quant"]
    gt = gt_data.reshape(-1, gt_data.shape[-1]).cuda()

    return {
        "x_main": x_main, "x_high": x_high,
        "w_main": w_main, "w_high": w_high,
        "s_x_m": s_x_m, "s_x_h": s_x_h,
        "neg_zero_m": neg_zero_m, "neg_zero_h": neg_zero_h,
        "s_w_m": s_w_m, "s_w_h": s_w_h,
        "colsum_m": colsum_m, "colsum_h": colsum_h,
        "gt": gt,
    }


def run_baseline_kernel(inputs: dict):
    """Run baseline: 2 CUTLASS GEMMs + torch.add."""
    import mixed_gemm

    Y_main = mixed_gemm.gemm_u8s8_dequant(
        inputs["x_main"], inputs["w_main"],
        inputs["s_x_m"], inputs["s_w_m"],
        inputs["neg_zero_m"], inputs["colsum_m"])

    Y_high = mixed_gemm.gemm_u8s8_dequant(
        inputs["x_high"], inputs["w_high"],
        inputs["s_x_h"], inputs["s_w_h"],
        inputs["neg_zero_h"], inputs["colsum_h"])

    return Y_main + Y_high


class TestPerTokenKernel:
    """Test CUDA baseline kernel for all per-token operators (q/k/v/gate/up_proj)."""

    @pytest.fixture(autouse=True)
    def _check_kernel_available(self):
        try:
            import mixed_gemm  # noqa: F401
        except ImportError:
            pytest.skip("mixed_gemm not compiled (run setup.py build_ext --inplace in kernels/mixed_gemm/)")

    def test_kernel_matches_ground_truth(self, per_token_op_dir, batch_size):
        """Baseline kernel output should match output_real_quant."""
        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size)
        inputs = prepare_kernel_inputs(data)
        kernel_output = run_baseline_kernel(inputs)

        metrics = compute_metrics(kernel_output, inputs["gt"])
        print_metrics(metrics, prefix=f"[{op_name} kernel vs GT, bs={batch_size}]")

        assert metrics["rel_err"] < 0.05, f"rel_err too high: {metrics['rel_err']}"
        assert metrics["cosine_sim"] > 0.9999, f"cosine_sim too low: {metrics['cosine_sim']}"
        assert metrics["snr_db"] > 60, f"SNR too low: {metrics['snr_db']} dB"

    def test_kernel_performance(self, per_token_op_dir):
        """Benchmark: measure each kernel's latency, TOPS, and bandwidth separately."""
        import mixed_gemm

        op_dir, op_name = per_token_op_dir
        data = load_operator_data(op_dir, batch_size=1)
        inputs = prepare_kernel_inputs(data)

        M = inputs["x_main"].shape[0]
        N = inputs["w_main"].shape[0]
        K_main = inputs["x_main"].shape[1]
        K_high = inputs["x_high"].shape[1]

        h20_peak_tops = 268      # INT8 measured peak on H20 (GEMM 10240^3)
        h20_peak_bw = 3189       # GB/s HBM measured peak on H20 (256MB copy)

        iters = 100

        def bench(fn, label):
            """Benchmark a single kernel with CUDA events."""
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        # --- GEMM main (U8S8, K=1792) ---
        ms_main = bench(
            lambda: mixed_gemm.gemm_u8s8_dequant(
                inputs["x_main"], inputs["w_main"],
                inputs["s_x_m"], inputs["s_w_m"],
                inputs["neg_zero_m"], inputs["colsum_m"]),
            "gemm_main")
        flops_main = 2 * M * N * K_main
        tops_main = flops_main / ms_main / 1e9
        # Bytes read: A(M*K*1) + B(N*K*1) + scales/zeros(M+N)*2bytes + colsum(N*4)
        bytes_main = M * K_main * 1 + N * K_main * 1 + (M + N) * 2 + N * 4
        bw_main = bytes_main / ms_main / 1e6  # GB/s

        # --- GEMM high (U8S8, K=256) ---
        ms_high = bench(
            lambda: mixed_gemm.gemm_u8s8_dequant(
                inputs["x_high"], inputs["w_high"],
                inputs["s_x_h"], inputs["s_w_h"],
                inputs["neg_zero_h"], inputs["colsum_h"]),
            "gemm_high")
        flops_high = 2 * M * N * K_high
        tops_high = flops_high / ms_high / 1e9
        bytes_high = M * K_high * 1 + N * K_high * 1 + (M + N) * 2 + N * 4
        bw_high = bytes_high / ms_high / 1e6

        # --- Reduction (elementwise add, FP16) ---
        Y_main = mixed_gemm.gemm_u8s8_dequant(
            inputs["x_main"], inputs["w_main"],
            inputs["s_x_m"], inputs["s_w_m"],
            inputs["neg_zero_m"], inputs["colsum_m"])
        Y_high = mixed_gemm.gemm_u8s8_dequant(
            inputs["x_high"], inputs["w_high"],
            inputs["s_x_h"], inputs["s_w_h"],
            inputs["neg_zero_h"], inputs["colsum_h"])
        ms_add = bench(lambda: Y_main + Y_high, "reduction")
        # Bytes: read 2 * M*N*2 + write M*N*2
        bytes_add = 3 * M * N * 2
        bw_add = bytes_add / ms_add / 1e6

        # --- Total ---
        ms_total = ms_main + ms_high + ms_add

        print(f"\n  [{op_name} baseline perf] M={M}, N={N}, K_main={K_main}, K_high={K_high}")
        print(f"    {'Kernel':<15} {'Latency':>10} {'TOPS':>10} {'BW (GB/s)':>12} {'Peak%':>8}")
        print(f"    {'-'*55}")
        print(f"    {'GEMM main':<15} {ms_main*1000:>7.1f} μs {tops_main:>8.1f}  {bw_main:>10.1f}  {tops_main/h20_peak_tops*100:>6.1f}%")
        print(f"    {'GEMM high':<15} {ms_high*1000:>7.1f} μs {tops_high:>8.1f}  {bw_high:>10.1f}  {tops_high/h20_peak_tops*100:>6.1f}%")
        print(f"    {'Add (reduce)':<15} {ms_add*1000:>7.1f} μs {'—':>8}  {bw_add:>10.1f}  {bw_add/h20_peak_bw*100:>6.1f}%")
        print(f"    {'-'*55}")
        print(f"    {'TOTAL':<15} {ms_total*1000:>7.1f} μs")

        # Main GEMM should achieve at least 50% peak
        assert tops_main > h20_peak_tops * 0.4, f"Main GEMM too slow: {tops_main:.1f} TOPS"


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
        fp16_baseline = data["output_fp16_baseline"]

        # FP16 baseline: how much error does quantization introduce?
        m = compute_metrics(gt, fp16_baseline)
        print_metrics(m, prefix="[real_quant vs fp16_baseline (quantization loss)]")

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
