"""RMSNorm fusion — absorb norm weights into adjacent linear layers.

After fusion, RMSNorm layers become identity (weight=ones),
allowing orthogonal rotations to pass through freely.
"""

import typing
import torch


def fuse_ln_linear(layernorm, linear_layers: typing.Iterable[torch.nn.Linear]):
    """Fuse LayerNorm/RMSNorm scale into adjacent linear layers."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = (
                linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            ).to(linear_dtype)


def fuse_layer_norms(model):
    """Fuse all RMSNorm weights into adjacent linears for a Llama model."""
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model.model.layers
    for layer in layers:
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )
        layer.input_layernorm.weight.data = torch.ones_like(
            layer.input_layernorm.weight.data
        )

    fuse_ln_linear(model.model.norm, [model.lm_head])
    model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)
