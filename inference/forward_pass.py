"""
PLAQuant independent forward pass.

Implements ResQ's fake-quant forward without importing any ResQ code.
Verified to produce PPL = 34.8456 (matching ResQ's own eval pipeline).

Architecture:
- Loads pre-rotated weights from ResQ checkpoint
- Uses fake quantization (quantize → dequant → FP16 matmul)
- KV cache: K quantized after RoPE + rotation; V quantized after v_proj

Usage:
    python -m inference.forward_pass --model <path> --checkpoint <path> ...
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from inference.attention import (
    quantize_k_cache,
    quantize_v_output,
    KVQuantConfig,
    _hadamard_transform,
    _fake_quant_per_token,
    _fake_quant_mixed_precision,
    _fake_quant_mixed_precision_grouped,
)
from inference.rotation import get_hadamard_matrix


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: x * weight / rms(x)."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(dtype)


def fake_quant_activation(
    x: torch.Tensor,
    bits: int,
    high_bits_length: int,
    high_bits: int = 8,
    sym: bool = False,
    groupsize: int = -1,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Fake quantize activation: quantize then immediately dequant.

    This is what ActQuantWrapper.forward() does when bits < 16.
    """
    if bits >= 16:
        return x

    dtype = x.dtype

    if groupsize > 0:
        # Per-group quantization (for v_proj output and o_proj input)
        init_shape = x.shape
        num_groups = x.shape[-1] // groupsize
        x_grouped = x.reshape(x.shape[0], x.shape[1], num_groups, groupsize)
        x_grouped = _fake_quant_mixed_precision_grouped(
            x_grouped, bits=bits, high_bits=high_bits,
            high_bits_length=high_bits_length, low_bits_length=0,
            sym=sym, clip_ratio=clip_ratio,
        )
        return x_grouped.reshape(init_shape).to(dtype)
    else:
        # Per-token quantization
        x_flat = x.reshape(-1, x.shape[-1])
        x_flat = _fake_quant_mixed_precision(
            x_flat, bits=bits, high_bits=high_bits,
            high_bits_length=high_bits_length, low_bits_length=0,
            sym=sym, clip_ratio=clip_ratio,
        )
        return x_flat.reshape(x.shape).to(dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply RoPE to Q and K.

    q, k: (batch, seq_len, num_heads * head_dim) or (batch, seq_len, num_heads, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # Reshape if needed
    if q.dim() == 3:
        # Need to know head_dim to reshape - infer from cos
        head_dim = cos.shape[-1]
        num_heads_q = q.shape[-1] // head_dim
        num_heads_k = k.shape[-1] // head_dim
        q = q.view(q.shape[0], q.shape[1], num_heads_q, head_dim)
        k = k.view(k.shape[0], k.shape[1], num_heads_k, head_dim)
        was_3d = True
    else:
        was_3d = False

    # Apply rotary: (cos + i*sin) * (x1 + i*x2) → (x1*cos - x2*sin, x1*sin + x2*cos)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin

    if was_3d:
        q_out = q_out.reshape(q_out.shape[0], q_out.shape[1], -1)
        k_out = k_out.reshape(k_out.shape[0], k_out.shape[1], -1)

    return q_out.to(q.dtype), k_out.to(k.dtype)


class IndependentResQForward:
    """Complete independent forward pass for ResQ quantized Llama.

    Loads weights from /tmp/resq_weights_clean.pt (dumped from ResQ model)
    and runs the equivalent forward pass.
    """

    def __init__(self, weights: Dict[str, torch.Tensor], config):
        self.weights = weights
        self.config = config
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config['intermediate_size']
        self.rope_theta = config.get('rope_theta', 500000.0)
        self.device = 'cpu'

        # Precompute Hadamard matrix for down_proj
        self.had_matrix = get_hadamard_matrix(self.intermediate_size)

        # Precompute RoPE
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim
        ))
        self._inv_freq = inv_freq

    def get_rope(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    def to(self, device: str):
        self.device = device
        self.weights = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in self.weights.items()}
        self.had_matrix = self.had_matrix.to(device)
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass. Returns logits."""
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        x = F.embedding(input_ids, self.weights['embed'])

        # RoPE
        cos, sin = self.get_rope(seq_len, device)

        # Decoder layers
        for i in range(self.num_layers):
            x = self._forward_layer(x, i, cos, sin)

        # Final norm
        x = rms_norm(x, self.weights['final_norm'])

        # LM head (FP16 matmul, no quant)
        logits = F.linear(x.float(), self.weights['lm_head'].float())

        return logits

    def _forward_layer(self, x, layer_idx, cos, sin):
        p = f'l{layer_idx}'
        w = self.weights

        # === Self Attention ===
        residual = x
        x = rms_norm(x, w[f'{p}.ln1'])

        # Q/K/V projections (fake quant + FP16 matmul)
        q = self._quantized_linear(x, p + '.q_proj')
        k = self._quantized_linear(x, p + '.k_proj')
        v = self._quantized_linear(x, p + '.v_proj')

        # V output quantization (for V cache)
        v_out_bits = w.get(f'{p}.v_out_bits', 16)
        if v_out_bits < 16:
            v = quantize_v_output(
                v, v_bits=v_out_bits,
                v_high_bits=8,
                v_groupsize=w.get(f'{p}.v_out_groupsize', 64),
                v_high_bits_length=w.get(f'{p}.v_out_high_len', 8),
                v_sym=w.get(f'{p}.v_out_sym', False),
            )

        # Reshape for attention
        batch, seq_len = q.shape[0], q.shape[1]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])

        # K cache quantization (after RoPE)
        k_bits = w.get(f'{p}.k_bits', 16)
        if k_bits < 16:
            # Reshape to (batch, heads, seq, dim) for QKRotation
            q_4d = q.transpose(1, 2)
            k_4d = k.transpose(1, 2)

            kv_config = KVQuantConfig(
                k_bits=k_bits,
                k_bits_high=8,
                k_groupsize=w.get(f'{p}.k_groupsize', self.head_dim),
                k_sym=w.get(f'{p}.k_sym', False),
                k_high_bits_length=w.get(f'{p}.k_high_len', int(0.125 * self.head_dim)),
                k_rotation=w.get(f'{p}.k_rotation'),
                use_had=w.get(f'{p}.k_had', False),
            )
            q_4d, k_4d = quantize_k_cache(q_4d, k_4d, kv_config)

            q = q_4d.transpose(1, 2)
            k = k_4d.transpose(1, 2)

        # GQA expansion
        num_rep = self.num_heads // self.num_kv_heads
        if num_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, num_rep, -1).reshape(batch, seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, num_rep, -1).reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Attention
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # O projection (fake quant + FP16 matmul)
        # Apply column reorder for o_proj (rearranged for mixed-precision grouping)
        column_order = self.weights.get(f'{p}.o_proj_column_order')
        if column_order is not None:
            attn_out = attn_out[..., column_order.to(attn_out.device)]
        attn_out = self._quantized_linear(attn_out, p + '.o_proj')

        # Residual
        x = residual + attn_out

        # === MLP ===
        residual = x
        x = rms_norm(x, w[f'{p}.ln2'])

        gate = self._quantized_linear(x, p + '.mlp.gate_proj')
        up = self._quantized_linear(x, p + '.mlp.up_proj')
        x = F.silu(gate) * up

        # Down proj: Hadamard + fake quant + FP16 matmul
        x = self._quantized_linear(x, p + '.mlp.down_proj', online_had=True)

        # Residual
        x = residual + x

        return x

    def _quantized_linear(self, x, prefix, online_had=False):
        """Fake quant linear: optional Hadamard → quantize → dequant → matmul."""
        w = self.weights

        # Online Hadamard (only for down_proj)
        if online_had:
            x = (x.float() @ self.had_matrix.to(x.device)).to(x.dtype)

        # Activation fake quant
        # Determine quant config based on layer type
        if 'down_proj' in prefix:
            # down_proj: uniform 4-bit, no high group
            bits = 4
            high_bits_length = 0
            groupsize = -1
        elif 'o_proj' in prefix:
            # o_proj: per-group (groupsize = head_dim)
            bits = 4
            high_bits_length = int(0.125 * self.head_dim)
            groupsize = self.head_dim
        elif 'v_proj' in prefix:
            bits = 4
            high_bits_length = int(0.125 * self.hidden_size)
            groupsize = -1
        else:
            # q_proj, k_proj, gate_proj, up_proj: per-token mixed precision
            bits = 4
            high_bits_length = int(0.125 * self.hidden_size)
            groupsize = -1

        x = fake_quant_activation(
            x, bits=bits, high_bits_length=high_bits_length,
            high_bits=8, sym=False, groupsize=groupsize,
        )

        # FP16 matmul
        weight = w.get(f'{prefix}.weight')
        if weight is None:
            # This shouldn't happen if weights are correctly loaded
            raise KeyError(f'Weight not found: {prefix}.weight')

        return F.linear(x, weight)
