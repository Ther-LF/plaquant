"""
PLAQuant rotation module.

Implements PCA basis fusion and Hadamard rotation for ResQ-style quantization.
Reference: project-resq/fake_quant/eval_utils/rotation_utils.py

Key operations:
  - fuse_basis_shared: fuse U_attn_mlp rotation into all layer weights
  - fuse_layer_norms: absorb RMSNorm into adjacent linear weights
  - rearrange_columns: reorder o_proj columns for mixed-precision grouping
  - get_hadamard_matrix: generate normalized Hadamard matrix for online rotation
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from scipy.linalg import hadamard as scipy_hadamard


def get_hadamard_matrix(dim: int, device="cpu") -> torch.Tensor:
    """Generate normalized Hadamard matrix of given dimension.

    Returns H / sqrt(dim) so that H @ H.T = I (orthonormal).
    Requires dim to be a power of 2.
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"
    H = torch.tensor(scipy_hadamard(dim), dtype=torch.float32, device=device)
    return H / math.sqrt(dim)


def fuse_layer_norms(model):
    """Fuse RMSNorm weights into adjacent linear layers.

    After fusion, RMSNorm becomes a simple x / rms(x) without learnable weight.
    The weight is absorbed into the next linear layer's weight matrix.

    Reference: project-resq/fake_quant/utils/fuse_norm_utils.py
    """
    for layer in model.model.layers:
        # input_layernorm weight → fuse into qkv_proj and gate_up_proj
        _fuse_norm_into_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        # post_attention_layernorm weight → fuse into gate_proj and up_proj
        _fuse_norm_into_linear(
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )

    # Final norm → fuse into lm_head
    if hasattr(model.model, "norm") and hasattr(model, "lm_head"):
        _fuse_norm_into_linear(model.model.norm, [model.lm_head])


def _fuse_norm_into_linear(norm: nn.Module, linears: list):
    """Absorb norm.weight into linear.weight for each linear in list."""
    w = norm.weight.data.float()
    for linear in linears:
        linear.weight.data = (linear.weight.data.float() * w.unsqueeze(0)).to(
            linear.weight.dtype
        )
    # Set norm weight to ones (effectively identity after fusion)
    norm.weight.data.fill_(1.0)


def fuse_basis_shared(model, U_basis_path: str, R_rotation_path: str, high_fraction: float = 0.125):
    """Fuse shared PCA basis and random rotation into model weights.

    This implements the 'full_shared' rotation granularity from ResQ.
    After calling this, the model weights include the rotation — no online
    rotation needed except Hadamard for down_proj.

    Reference: project-resq/fake_quant/eval_utils/rotation_utils.py:378-446

    Args:
        model: HuggingFace model (already with fused norms)
        U_basis_path: path to U basis checkpoint (e.g., U-wikitext-512-*.bin)
        R_rotation_path: path to R rotation checkpoint (e.g., R-high-*.bin)
        high_fraction: fraction of dimensions for high-precision group
    """
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    high_bits_length = int(high_fraction * model_dim)

    # Load rotation matrices
    U_cpk = torch.load(U_basis_path, map_location="cpu", weights_only=False)
    R_dict = torch.load(R_rotation_path, map_location="cpu", weights_only=False)

    # Build block-diagonal R1 = diag(R1_0, R1_1, R1_2) where:
    # R1_1: (K_main, K_main) rotation for main group
    # R1_2: (K_high, K_high) rotation for high group
    R1_1 = R_dict["R1_1"].to(torch.float64)
    R1_2 = R_dict["R1_2"].to(torch.float64)
    assert R1_2.shape[0] == high_bits_length

    R1 = torch.block_diag(R1_1, R1_2)
    R1_0 = R_dict.get("R1_0")
    if R1_0 is not None:
        R1 = torch.block_diag(R1_0.to(torch.float64), R1)

    # Build R2 for V/O projection (per-head rotation)
    R2_1 = R_dict["R2_1"].to(torch.float64)
    R2_2 = R_dict["R2_2"].to(torch.float64)
    R2 = torch.block_diag(R2_1, R2_2)
    R2_0 = R_dict.get("R2_0")
    if R2_0 is not None:
        R2 = torch.block_diag(R2_0.to(torch.float64), R2)

    # U_attn = U_basis @ R1 (combined PCA + random rotation)
    U_attn = torch.matmul(U_cpk["attn_mlp"].to(torch.float64), R1)

    # Rotate embeddings: embed_tokens.weight = embed_tokens.weight @ U_attn
    _rotate_embeddings(model, U_attn)

    # Rotate lm_head: lm_head.weight = lm_head.weight @ U_attn
    _rotate_head(model, U_attn)

    # Rotate each layer
    layers = model.model.layers
    for idx, layer in enumerate(layers):
        # U_value for V/O projection (per-head, per-layer)
        key = f"layer.{idx}.self_attn.value"
        U_value = U_cpk[key].to(torch.float64)
        U_value = torch.matmul(U_value, R2)  # (num_kv_heads, head_dim, head_dim)

        # Rotate attention: Q/K/V input side by U_attn
        _rotate_attention_inputs(layer, U_attn)

        # Rotate V/O with per-head U_value
        _rotate_ov_proj(layer, num_heads, head_dim, U_value)

        # Rotate attention output (o_proj output side)
        _rotate_attention_output(layer, U_attn)

        # Rotate MLP input (gate/up input side)
        _rotate_mlp_input(layer, U_attn)

        # Rotate MLP output (down_proj output side)
        _rotate_mlp_output(layer, U_attn)


def rearrange_columns(model, high_fraction: float = 0.125):
    """Rearrange o_proj weight columns for mixed-precision per-head grouping.

    In ResQ, o_proj input is organized per-head. Within each head, the last
    `high_length_per_head` channels are high-precision. This function reorders
    columns so that all high-precision channels are contiguous at the end.

    Reference: project-resq/fake_quant/eval_utils/rotation_utils.py:239-300
    """
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    high_bits_length = int(high_fraction * model_dim)
    high_length_per_head = high_bits_length // num_heads

    for layer in model.model.layers:
        o_proj = layer.self_attn.o_proj
        in_dim = o_proj.weight.shape[1]
        num_replicated_heads = in_dim // head_dim

        # Build column order: [main_cols..., high_cols...]
        chunk_starts = torch.arange(0, in_dim, head_dim)
        # High precision columns: last `high_length_per_head` of each head
        high_cols = (
            chunk_starts.unsqueeze(1)
            + torch.arange(head_dim - high_length_per_head, head_dim)
        ).flatten()
        # Main columns: the rest
        all_cols = torch.arange(in_dim)
        main_cols = all_cols[~torch.isin(all_cols, high_cols)]
        new_order = torch.cat([main_cols, high_cols])

        # Rearrange weight columns
        o_proj.weight.data = o_proj.weight.data[:, new_order]


# =============================================================================
# Helper functions for rotation
# =============================================================================

def _rotate_embeddings(model, R: torch.Tensor):
    """embed_tokens.weight = embed_tokens.weight @ R"""
    W = model.model.embed_tokens.weight.data
    dtype = W.dtype
    model.model.embed_tokens.weight.data = (
        W.to(torch.float64) @ R
    ).to(dtype)


def _rotate_head(model, R: torch.Tensor):
    """lm_head.weight = lm_head.weight @ R"""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight.data
        dtype = W.dtype
        model.lm_head.weight.data = (W.to(torch.float64) @ R).to(dtype)


def _rotate_attention_inputs(layer, R: torch.Tensor):
    """Rotate Q/K/V projection weights: W_new = W @ R.T (input side)"""
    for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        W = proj.weight.data
        dtype = W.dtype
        proj.weight.data = (W.to(torch.float64) @ R.T).to(dtype)


def _rotate_attention_output(layer, R: torch.Tensor):
    """Rotate o_proj output: W_new = R @ W (output side)"""
    W = layer.self_attn.o_proj.weight.data
    dtype = W.dtype
    layer.self_attn.o_proj.weight.data = (R @ W.to(torch.float64)).to(dtype)


def _rotate_mlp_input(layer, R: torch.Tensor):
    """Rotate gate/up projection input: W_new = W @ R.T"""
    for proj in [layer.mlp.gate_proj, layer.mlp.up_proj]:
        W = proj.weight.data
        dtype = W.dtype
        proj.weight.data = (W.to(torch.float64) @ R.T).to(dtype)


def _rotate_mlp_output(layer, R: torch.Tensor):
    """Rotate down_proj output: W_new = R @ W"""
    W = layer.mlp.down_proj.weight.data
    dtype = W.dtype
    layer.mlp.down_proj.weight.data = (R @ W.to(torch.float64)).to(dtype)


def _rotate_ov_proj(layer, num_heads: int, head_dim: int, U_value: torch.Tensor):
    """Rotate V and O projections with per-head rotation.

    U_value: (num_kv_heads, head_dim, head_dim) per-head rotation matrices.

    V_proj output: each head's columns rotated by U_value[head]
    O_proj input: each head's rows rotated by U_value[head].T
    """
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    num_kv_heads = U_value.shape[0]
    dtype = v_proj.weight.dtype

    # Rotate V output (columns grouped by head)
    W_v = v_proj.weight.data.to(torch.float64)
    for h in range(num_kv_heads):
        start = h * head_dim
        end = start + head_dim
        W_v[start:end, :] = U_value[h] @ W_v[start:end, :]
    v_proj.weight.data = W_v.to(dtype)

    # Rotate O input (columns grouped by head, with GQA replication)
    W_o = o_proj.weight.data.to(torch.float64)
    num_heads_total = W_o.shape[1] // head_dim
    heads_per_kv = num_heads_total // num_kv_heads
    for h in range(num_heads_total):
        kv_h = h // heads_per_kv
        start = h * head_dim
        end = start + head_dim
        W_o[:, start:end] = W_o[:, start:end] @ U_value[kv_h].T
    o_proj.weight.data = W_o.to(dtype)
