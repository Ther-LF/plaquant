"""Quantization operations for real inference — activation quantize + INT4 packing."""

import torch


def quantize_activation_per_token(x, bits, asym=True):
    """Per-token activation quantization.

    Args:
        x: (M, K) FP16 tensor
        bits: quantization bits (4 or 8)
        asym: if True, asymmetric [0, maxq]; if False, symmetric [-maxq, maxq]

    Returns:
        q: (M, K) integer tensor (uint8 for 4-bit [0,15], uint8 for 8-bit [0,255])
        scale: (M, 1) per-token scale
        zero: (M, 1) per-token zero point
    """
    maxq = 2**bits - 1
    x_flat = x.reshape(-1, x.shape[-1])
    M = x_flat.shape[0]

    if asym:
        xmin = x_flat.min(dim=1, keepdim=True)[0].clamp(max=0)
        xmax = x_flat.max(dim=1, keepdim=True)[0].clamp(min=0)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = 1
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(x_flat / scale) + zero, 0, maxq)
    else:
        xmax = x_flat.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = xmax / (maxq // 2)
        zero = torch.zeros_like(scale)
        q = torch.clamp(torch.round(x_flat / scale) + (maxq // 2), 0, maxq)

    return q.to(torch.uint8), scale, zero


def shift_to_signed(q, bits):
    """Shift unsigned quantized values to signed for tensor core.

    4-bit: [0,15] → [-8, 7] (shift by 8)
    8-bit: [0,255] → [-128, 127] (shift by 128)
    """
    shift = 2 ** (bits - 1)
    return (q.to(torch.int16) - shift).to(torch.int8)


def pack_int4(q_int8):
    """Pack two INT4 values into one INT8 byte.

    Input: (M, K) int8 tensor with values in [-8, 7]
    Output: (M, K//2) int8 tensor (packed, low nibble first)

    Layout: byte = (high_element << 4) | (low_element & 0xF)
    """
    assert q_int8.shape[-1] % 2 == 0, "K must be even for INT4 packing"
    q = q_int8.view(*q_int8.shape[:-1], -1, 2)
    low = q[..., 0].to(torch.uint8) & 0x0F
    high = (q[..., 1].to(torch.uint8) & 0x0F) << 4
    packed = (low | high).to(torch.int8)
    return packed


def quantize_weight_per_channel_symmetric(W, bits):
    """Per-channel symmetric weight quantization.

    Args:
        W: (N, K) FP16 weight tensor
        bits: quantization bits

    Returns:
        q_int: (N, K) int8 tensor (centered at 0)
        scale: (N, 1) per-channel scale
    """
    maxq = 2 ** (bits - 1) - 1
    wmax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = wmax / maxq
    q = torch.clamp(torch.round(W / scale), -maxq - 1, maxq)
    return q.to(torch.int8), scale.half()


def compute_colsum(W_int):
    """Compute column sum of integer weight matrix (for bias correction).

    Args:
        W_int: (N, K) int8/int16 weight tensor

    Returns:
        colsum: (1, N) = sum over K dimension
    """
    return W_int.float().sum(dim=1, keepdim=True).T  # (1, N)
