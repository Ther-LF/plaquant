"""
PLAQuant custom attention with KV cache quantization.

Implements the QKRotationWrapper logic from ResQ:
- After RoPE, apply rotation (U_C) to Q and K
- Quantize K per-head (fake quant: quantize → dequant)
- V quantization is done via out_quantizer on v_proj output (handled in model.py)

Reference: project-resq/fake_quant/eval_utils/rotation_utils.py:545-634
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class KVQuantConfig:
    """Configuration for KV cache quantization."""

    def __init__(
        self,
        k_bits: int = 4,
        k_bits_high: int = 8,
        k_groupsize: int = 64,  # head_dim for per-head quant
        k_sym: bool = False,
        k_clip_ratio: float = 1.0,
        k_high_bits_length: int = 8,  # high_fraction * head_dim
        k_low_bits_length: int = 0,
        k_rotation: Optional[torch.Tensor] = None,  # (head_dim, head_dim) per-layer
        use_had: bool = False,  # True = use Hadamard on Q/K, False = use rotation matrix
    ):
        self.k_bits = k_bits
        self.k_bits_high = k_bits_high
        self.k_groupsize = k_groupsize
        self.k_sym = k_sym
        self.k_clip_ratio = k_clip_ratio
        self.k_high_bits_length = k_high_bits_length
        self.k_low_bits_length = k_low_bits_length
        self.k_rotation = k_rotation
        self.use_had = use_had


def quantize_k_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    kv_config: KVQuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotation and quantization to Q and K after RoPE.

    This implements QKRotationWrapper.forward() from ResQ.

    Args:
        q: (batch, num_heads, seq_len, head_dim) after RoPE
        k: (batch, num_kv_heads, seq_len, head_dim) after RoPE
        kv_config: KV quantization configuration

    Returns:
        q: rotated Q (quantized if use_had=False)
        k: rotated and quantized K
    """
    dtype = q.dtype

    if kv_config.k_bits >= 16:
        # No K quantization needed
        return q, k

    if kv_config.use_had:
        # Hadamard rotation on Q and K
        q = _hadamard_transform(q.float(), q.shape[-1]).to(dtype)
        k = _hadamard_transform(k.float(), k.shape[-1]).to(dtype)
    else:
        # Rotation matrix: quantize Q/K at 8-bit, then rotate
        R = kv_config.k_rotation.to(q.device, q.dtype)

        # Pre-rotation 8-bit quantization for Q
        q = _fake_quant_per_token(q, bits=8, sym=kv_config.k_sym)
        q = torch.matmul(q, R)

        # Pre-rotation 8-bit quantization for K
        k = _fake_quant_per_token(k, bits=8, sym=kv_config.k_sym)
        k = torch.matmul(k, R)

    # Quantize K per-head (fake quant: quantize → dequant)
    bsz, num_heads, seq_len, head_dim = k.shape

    if kv_config.k_groupsize == head_dim:
        # Per-head quantization: treat each head independently
        per_head_k = k.contiguous().view(-1, head_dim)
        per_head_k = _fake_quant_mixed_precision(
            per_head_k,
            bits=kv_config.k_bits,
            high_bits=kv_config.k_bits_high,
            high_bits_length=kv_config.k_high_bits_length,
            low_bits_length=kv_config.k_low_bits_length,
            sym=kv_config.k_sym,
            clip_ratio=kv_config.k_clip_ratio,
        )
        k = per_head_k.reshape(bsz, num_heads, seq_len, head_dim).to(dtype)
    else:
        raise NotImplementedError(f"k_groupsize={kv_config.k_groupsize} not supported")

    return q, k


def quantize_v_output(
    v: torch.Tensor,
    v_bits: int = 4,
    v_high_bits: int = 8,
    v_groupsize: int = 64,
    v_high_bits_length: int = 8,
    v_low_bits_length: int = 0,
    v_sym: bool = False,
    v_clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize V proj output (fake quant for V cache).

    This is applied to the output of v_proj before it enters attention.
    ResQ applies this as `out_quantizer` on the v_proj ActQuantWrapper.

    Args:
        v: (batch, seq_len, num_kv_heads * head_dim) — v_proj output
        Other args: quantization config

    Returns:
        Quantized (fake quant) V tensor
    """
    if v_bits >= 16:
        return v

    dtype = v.dtype
    # Reshape to (batch * seq, num_groups, groupsize) for per-group quant
    init_shape = v.shape
    # v_groupsize should be head_dim
    v_flat = v.reshape(-1, v.shape[-1])
    # Reshape to (M, num_groups, groupsize)
    num_groups = v_flat.shape[-1] // v_groupsize
    v_grouped = v_flat.reshape(v_flat.shape[0], num_groups, v_groupsize)

    v_grouped = _fake_quant_mixed_precision_grouped(
        v_grouped,
        bits=v_bits,
        high_bits=v_high_bits,
        high_bits_length=v_high_bits_length,
        low_bits_length=v_low_bits_length,
        sym=v_sym,
        clip_ratio=v_clip_ratio,
    )

    return v_grouped.reshape(init_shape).to(dtype)


# =============================================================================
# Internal helpers
# =============================================================================

def _hadamard_transform(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Fast Hadamard Transform on last dimension, normalized by 1/sqrt(dim).

    Uses recursive butterfly implementation for efficiency.
    For correctness validation, equivalent to: x @ scipy_hadamard(dim) / sqrt(dim)
    """
    # Iterative butterfly implementation
    h = 1
    while h < dim:
        # Split into pairs and compute butterfly
        x_even = x[..., 0::2]  # This doesn't work for in-place butterfly
        # Fall back to matrix multiplication for now
        break

    # Fallback: use matrix multiply with Hadamard matrix
    from inference.rotation import get_hadamard_matrix
    H = get_hadamard_matrix(dim, device=x.device).to(x.dtype)
    return x @ H


def _fake_quant_per_token(x: torch.Tensor, bits: int, sym: bool) -> torch.Tensor:
    """Fake quantize per-token: quantize then dequantize. Scale in FP32."""
    dtype = x.dtype
    maxq = 2 ** (bits - 1) - 1 if sym else 2**bits - 1

    orig_shape = x.shape
    x_f = x.reshape(-1, x.shape[-1]).float()
    tmp = torch.zeros(x_f.shape[0], device=x_f.device)

    if sym:
        xmin = torch.minimum(x_f.min(dim=-1)[0], tmp)
        xmax = torch.maximum(x_f.max(dim=-1)[0], tmp)
        xmax = torch.maximum(torch.abs(xmin), xmax)
        t = xmax == 0
        scale = (xmax / maxq).unsqueeze(-1)
        scale[t.unsqueeze(-1).expand_as(scale)] = 1
        q = torch.clamp(torch.round(x_f / scale), -(maxq + 1), maxq)
        return (q * scale).reshape(orig_shape).to(dtype)
    else:
        xmin = torch.minimum(x_f.min(dim=-1)[0], tmp)
        xmax = torch.maximum(x_f.max(dim=-1)[0], tmp)
        t = (xmin == 0) & (xmax == 0)
        xmin[t] = -1; xmax[t] = 1
        scale = ((xmax - xmin) / maxq).unsqueeze(-1)
        zero = torch.round((-xmin / scale.squeeze(-1))).unsqueeze(-1)
        q = torch.clamp(torch.round(x_f / scale) + zero, 0, maxq)
        return ((q - zero) * scale).reshape(orig_shape).to(dtype)


def _fake_quant_mixed_precision(
    x: torch.Tensor,
    bits: int,
    high_bits: int,
    high_bits_length: int,
    low_bits_length: int,
    sym: bool,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Fake quant with mixed precision (main + high groups)."""
    dtype = x.dtype
    K = x.shape[-1]
    main_end = K - high_bits_length
    low_end = low_bits_length

    x_main = x[..., low_end:main_end]
    x_high = x[..., main_end:] if high_bits_length > 0 else None

    # Main group
    maxq_m = 2 ** (bits - 1) - 1 if sym else 2**bits - 1
    x_main = _fake_quant_1d(x_main, maxq_m, sym, clip_ratio)

    # High group
    if x_high is not None:
        maxq_h = 2 ** (high_bits - 1) - 1 if sym else 2**high_bits - 1
        x_high = _fake_quant_1d(x_high, maxq_h, sym, clip_ratio)

    # Reassemble
    parts = []
    if low_bits_length > 0:
        parts.append(x[..., :low_end])  # low group unchanged for now
    parts.append(x_main)
    if x_high is not None:
        parts.append(x_high)
    return torch.cat(parts, dim=-1).to(dtype)


def _fake_quant_mixed_precision_grouped(
    x: torch.Tensor,
    bits: int,
    high_bits: int,
    high_bits_length: int,
    low_bits_length: int,
    sym: bool,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Fake quant with mixed precision, per-group (for V cache).

    x: (M, num_groups, groupsize)
    """
    dtype = x.dtype
    groupsize = x.shape[-1]
    main_end = groupsize - high_bits_length
    low_end = low_bits_length

    x_main = x[..., low_end:main_end]
    x_high = x[..., main_end:] if high_bits_length > 0 else None

    maxq_m = 2 ** (bits - 1) - 1 if sym else 2**bits - 1
    # Per-group: compute scale across last dim
    x_main = _fake_quant_per_group(x_main, maxq_m, sym, clip_ratio)

    if x_high is not None:
        maxq_h = 2 ** (high_bits - 1) - 1 if sym else 2**high_bits - 1
        x_high = _fake_quant_per_group(x_high, maxq_h, sym, clip_ratio)

    parts = []
    if low_bits_length > 0:
        parts.append(x[..., :low_end])
    parts.append(x_main)
    if x_high is not None:
        parts.append(x_high)
    return torch.cat(parts, dim=-1).to(dtype)


def _fake_quant_1d(x: torch.Tensor, maxq: int, sym: bool, clip_ratio: float) -> torch.Tensor:
    """Fake quant: per-row (last dim is feature). Scale computed in FP32."""
    dtype = x.dtype
    x_flat = x.reshape(-1, x.shape[-1])
    # Compute scale/zero in FP32 (matching ResQ's _find_params which uses torch.zeros in float32)
    x_f = x_flat.float()
    tmp = torch.zeros(x_f.shape[0], device=x_f.device)
    if sym:
        xmin = torch.minimum(x_f.min(dim=-1)[0], tmp) * clip_ratio
        xmax = torch.maximum(x_f.max(dim=-1)[0], tmp) * clip_ratio
        xmax = torch.maximum(torch.abs(xmin), xmax)
        t = xmax == 0
        scale = xmax / maxq
        scale[t] = 1
        scale = scale.unsqueeze(-1)
        q = torch.clamp(torch.round(x_f / scale), -(maxq + 1), maxq)
        result = q * scale
    else:
        xmin = torch.minimum(x_f.min(dim=-1)[0], tmp) * clip_ratio
        xmax = torch.maximum(x_f.max(dim=-1)[0], tmp) * clip_ratio
        t = (xmin == 0) & (xmax == 0)
        xmin[t] = -1; xmax[t] = 1
        scale = ((xmax - xmin) / maxq).unsqueeze(-1)
        zero = torch.round((-xmin / scale.squeeze(-1))).unsqueeze(-1)
        q = torch.clamp(torch.round(x_f / scale) + zero, 0, maxq)
        result = (q - zero) * scale
    return result.reshape(x.shape).to(dtype)


def _fake_quant_per_group(x: torch.Tensor, maxq: int, sym: bool, clip_ratio: float) -> torch.Tensor:
    """Fake quant per-group: x is (..., group_dim), scale computed per last dim in FP32."""
    dtype = x.dtype
    x_f = x.float()
    if sym:
        xmax = x_f.abs().amax(dim=-1, keepdim=True) * clip_ratio
        t = xmax == 0
        scale = xmax / maxq
        scale[t] = 1
        q = torch.clamp(torch.round(x_f / scale), -(maxq + 1), maxq)
        return (q * scale).to(dtype)
    else:
        xmin = x_f.amin(dim=-1, keepdim=True) * clip_ratio
        xmax = x_f.amax(dim=-1, keepdim=True) * clip_ratio
        xmin = torch.minimum(xmin, torch.zeros_like(xmin))
        xmax = torch.maximum(xmax, torch.zeros_like(xmax))
        t = (xmin == 0) & (xmax == 0)
        xmin[t] = -1; xmax[t] = 1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(x_f / scale) + zero, 0, maxq)
        return ((q - zero) * scale).to(dtype)
