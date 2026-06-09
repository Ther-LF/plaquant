"""Weight packing for real INT inference.

Pre-processes model weights after PTQ rotation into INT4/INT8 packed format
ready for tensor core GEMM.
"""

import torch

from promix.quantize.quant_utils import ActQuantWrapper, find_qlayers
from promix.inference.quant_ops import (
    quantize_weight_per_channel_symmetric,
    pack_int4,
    compute_colsum,
)


def pack_model_weights(model, w_bits=4):
    """Pack all linear layer weights for real INT inference.

    For each ActQuantWrapper with bits < 16:
    - Split weight into main (K_main) and high (K_high) groups
    - Quantize main to w_bits (default 4) per-channel symmetric → pack INT4
    - Quantize high to 8-bit per-channel symmetric
    - Pre-compute column sums for bias correction

    Stores packed data as buffers on the wrapper.
    """
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])

    for name, wrapper in qlayers.items():
        if wrapper.quantizer.bits >= 16:
            continue

        W = wrapper.module.weight.data  # (N, K)
        K = W.shape[1]
        high_len = wrapper.quantizer.high_bits_length
        K_main = K - high_len

        # Split into main and high groups
        W_main = W[:, :K_main]  # (N, K_main) — 4-bit group
        W_high = W[:, K_main:] if high_len > 0 else None  # (N, K_high) — 8-bit group

        # Quantize main group to INT4
        q_main, s_w_main = quantize_weight_per_channel_symmetric(W_main.float(), w_bits)
        q_main_packed = pack_int4(q_main)  # (N, K_main//2)
        colsum_main = compute_colsum(q_main)  # (1, N)

        # Store on wrapper
        wrapper.register_buffer('W_main_packed', q_main_packed)
        wrapper.register_buffer('s_w_main', s_w_main)
        wrapper.register_buffer('colsum_main', colsum_main)
        wrapper._K_main = K_main

        # Quantize high group to INT8
        if W_high is not None and high_len > 0:
            q_high, s_w_high = quantize_weight_per_channel_symmetric(W_high.float(), 8)
            colsum_high = compute_colsum(q_high)
            wrapper.register_buffer('W_high_int8', q_high)
            wrapper.register_buffer('s_w_high', s_w_high)
            wrapper.register_buffer('colsum_high', colsum_high)
            wrapper._K_high = high_len
        else:
            wrapper._K_high = 0

        wrapper._real_inference_ready = True

    print(f"Packed {sum(1 for w in qlayers.values() if getattr(w, '_real_inference_ready', False))} layers for real inference")
