"""Rotation utilities — fuse PCA basis + rotation into model weights.

Handles the full_shared rotation granularity:
- Single U_attn matrix shared across all layers for attention/MLP inputs
- Per-layer U_value for value projection
- Rearranges o_proj columns for mixed-precision grouping
"""

import torch
from tqdm import tqdm

from promix.utils import cleanup_memory
from promix.quantize.hadamard import matmul_hadU_cuda, get_hadK


def rotate_embeddings(model, R1):
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_head(model, R1):
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1):
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1):
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    for W in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)
    # Apply Hadamard to input side of down_proj weight
    # This pairs with the online_full_had applied to activation at runtime
    had_K, K = get_hadK(W.weight.data.shape[-1])
    dev = W.weight.data.device
    W_ = W.weight.data.float().cuda()
    W_ = matmul_hadU_cuda(W_, had_K, K)
    W.weight.data = W_.to(device=dev, dtype=dtype)


def rotate_ov_proj(layer, num_heads, head_dim, U_value, per_head=True):
    """Rotate v_proj output and o_proj input by per-head U_value."""
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    # v_proj: output rotation (right multiply on transposed weight)
    _apply_had_to_linear(v_proj, head_dim, output=True, R2=U_value, per_head=per_head)
    # o_proj: input rotation (right multiply on weight with inverse)
    _apply_had_to_linear(o_proj, head_dim, output=False, R2=U_value, per_head=per_head)


def _apply_had_to_linear(module, had_dim, output, R2, per_head):
    """Apply rotation R2 to a linear layer along head_dim blocks."""
    W_ = module.weight.data.float().cuda()
    dtype = module.weight.data.dtype
    dev = module.weight.data.device
    hadK = R2.to(torch.float64)

    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
        if per_head:
            num_heads = transposed_shape[-1] // had_dim
            for i in range(num_heads):
                temp[:, i, :] = temp[:, i, :].to(torch.float64) @ hadK[i]
        else:
            temp = temp.to(torch.float64) @ hadK
        W_ = temp.reshape(transposed_shape).t()
    else:
        init_shape = W_.shape
        temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
        if per_head:
            num_kv_heads = hadK.shape[0]
            num_kv_groups = init_shape[0] // (had_dim * num_kv_heads)
            for i in range(num_kv_heads):
                for j in range(num_kv_groups):
                    idx = j + num_kv_groups * i
                    try:
                        inverse = torch.inverse(hadK[i]).t()
                    except torch._C._LinAlgError:
                        inverse = hadK[i]
                    temp[:, idx, :] = temp[:, idx, :].to(torch.float64) @ inverse
        else:
            temp = temp.to(torch.float64) @ torch.inverse(hadK).t()
        W_ = temp.reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)


def rearrange_columns(model, high_fraction, low_fraction=0.0, o_proj_pca="per_head"):
    """Rearrange o_proj columns so [low|main|high] are physically contiguous.

    Per-head mode: each head's columns end with the highest-variance
    coordinates after the per-head value PCA. Without rearrangement the
    quantizer sees high channels scattered across heads (h0_high, h1_high,
    ...); the helper below moves all per-head high tails to the end of
    the full tensor producing [low_all | main_all | high_all].

    Global mode (o_proj_pca == "full_global"): the o_proj input has been
    rotated by a single hidden_dim PCA basis whose columns are sorted
    ascending by eigenvalue (see `eigen_decompose`), so the tensor is
    ALREADY in [low_all | main_all | high_all] order — the highest-variance
    coordinates sit at the end of the full hidden dim, not at the end of
    each head. Rearranging again would scramble the variance order.
    Skip the helper and leave `new_column_order` at its default `None`.
    """
    if o_proj_pca == "full_global":
        return
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    high_bits_length = int(high_fraction * model.config.hidden_size)
    low_bits_length = int(low_fraction * model.config.hidden_size)

    for layer in tqdm(model.model.layers, desc="Rearranging o_proj"):
        _rearrange_o_proj(layer, high_bits_length, low_bits_length, head_dim)


def _rearrange_o_proj(layer, high_bits_length, low_bits_length, head_dim):
    """Rearrange o_proj weight columns: [low|main|high] contiguous layout."""
    o_proj = layer.self_attn.o_proj
    in_dim = o_proj.weight.shape[-1]
    num_replicated_heads = in_dim // head_dim
    high_per_head = high_bits_length // num_replicated_heads
    low_per_head = low_bits_length // num_replicated_heads

    chunk_starts = torch.arange(0, in_dim, head_dim)

    # High-precision columns (end of each head) → move to end
    high_cols = (chunk_starts.unsqueeze(1) + torch.arange(head_dim - high_per_head, head_dim)).flatten()

    # Low-precision columns (start of each head) → move to beginning
    low_cols = (chunk_starts.unsqueeze(1) + torch.arange(0, low_per_head)).flatten() if low_per_head > 0 else torch.tensor([], dtype=torch.long)

    all_columns = torch.arange(in_dim)
    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[high_cols] = False
    if len(low_cols) > 0:
        mask[low_cols] = False
    remaining = all_columns[mask]

    if len(low_cols) > 0:
        new_column_order = torch.cat([low_cols, remaining, high_cols])
    else:
        new_column_order = torch.cat([remaining, high_cols])

    # Rearrange weight columns
    o_proj.weight.data = o_proj.weight.data[:, new_column_order]
    # Store order for runtime rearrangement of attention output
    layer.self_attn.new_column_order = new_column_order


def fuse_basis_to_model(model, basis_path, rotation_path, high_fraction, low_fraction=0.0):
    """Fuse PCA basis + rotation into model weights (full_shared mode).

    After this call:
    - embed_tokens output is in rotated coordinate system
    - All linear weights absorb the rotation
    - Residual stream stays in rotated coordinates throughout
    """
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    high_bits_length = int(high_fraction * model_dim)

    # Load basis and rotation
    U_cpk = torch.load(basis_path, weights_only=False)
    U_attn = U_cpk["attn_mlp"].cuda()

    R_dict = torch.load(rotation_path, weights_only=False)
    R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
    R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
    assert R1_2.shape[0] == high_bits_length

    R1 = torch.block_diag(R1_1, R1_2)
    R1_0 = R_dict["R1_0"]
    if R1_0 is not None:
        R1 = torch.block_diag(R1_0.cuda().to(torch.float64), R1)

    # Combine: T = U × R
    U_attn = torch.matmul(U_attn, R1)

    torch.distributed.barrier()

    # Rotate all model weights
    rotate_embeddings(model, U_attn)
    rotate_head(model, U_attn)
    cleanup_memory()

    # Per-layer rotations
    layers = model.model.layers
    trained_format = "R2_1" not in R_dict  # trained format uses per-layer keys

    if not trained_format:
        R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
        R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
        R2 = torch.block_diag(R2_1, R2_2)
        R2_0 = R_dict.get("R2_0")
        if R2_0 is not None:
            R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)

    # Detect whether the basis bundle was built with `o_proj_pca: full_global`
    # mode. Presence of this key on at least one layer opts the rotation
    # pipeline into the global-hidden-dim path for o_proj.
    use_oproj_global = any(
        f"layer.{i}.self_attn.o_proj_global" in U_cpk for i in range(len(layers))
    )
    if use_oproj_global:
        print(
            "[rotation] o_proj_global key detected in basis; applying full hidden_dim "
            "PCA to o_proj input (replaces per-head value rotation for o_proj only). "
            "v_proj output rotation remains per-head."
        )

    for idx, layer in enumerate(tqdm(layers, desc="Rotating layers")):
        rotate_attention_inputs(layer, U_attn)

        # Per-layer value rotation
        U_value = U_cpk[f"layer.{idx}.self_attn.value"].cuda()

        if trained_format:
            R2_1 = R_dict[f"model.layers.{idx}.self_attn.R2_1"].cuda().to(torch.float64)
            R2_2 = R_dict[f"model.layers.{idx}.self_attn.R2_2"].cuda().to(torch.float64)
            R2 = torch.block_diag(R2_1, R2_2)
            r2_0_key = f"model.layers.{idx}.self_attn.R2_0"
            if r2_0_key in R_dict:
                R2 = torch.block_diag(R_dict[r2_0_key].cuda().to(torch.float64), R2)

        U_value = torch.matmul(U_value, R2)
        if use_oproj_global:
            # v_proj output: keep per-head R2 (R2 is intrinsic to attention).
            _apply_had_to_linear(
                layer.self_attn.v_proj, head_dim, output=True,
                R2=U_value, per_head=True,
            )
            # o_proj input: apply the GLOBAL hidden_dim PCA (`o_proj_global`)
            # instead of per-head U_value. R2 composition is not threaded
            # through the o_proj input here; the global basis was fitted on
            # the un-rotated o_proj input and is treated as the canonical
            # o_proj-input rotation.
            U_oproj_g = U_cpk[f"layer.{idx}.self_attn.o_proj_global"].cuda().to(torch.float64)
            hidden_dim = U_oproj_g.shape[0]
            assert U_oproj_g.shape == (hidden_dim, hidden_dim), (
                f"o_proj_global must be hidden_dim x hidden_dim; got {U_oproj_g.shape}"
            )
            _apply_had_to_linear(
                layer.self_attn.o_proj, hidden_dim, output=False,
                R2=U_oproj_g, per_head=False,
            )
        else:
            rotate_ov_proj(layer, num_heads, head_dim, U_value, per_head=True)

        rotate_attention_output(layer, U_attn)
        rotate_mlp_input(layer, U_attn)
        rotate_mlp_output(layer, U_attn)
