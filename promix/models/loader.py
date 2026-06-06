"""Llama model loading and patching for ProMix quantization.

Uses standard HuggingFace LlamaForCausalLM and patches it for:
- tie_word_embeddings handling
- new_column_order support via pre-forward hooks on o_proj
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_model(model_name, dtype=torch.float16):
    """Load Llama model with untied word embeddings (required for rotation)."""
    config = AutoConfig.from_pretrained(model_name)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, config=config
    ).cuda()

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Initialize new_column_order slots (set by rearrange_columns later)
    for layer in model.model.layers:
        layer.self_attn.new_column_order = None

    return model


def install_column_order_hooks(model):
    """Install pre-forward hooks on o_proj wrappers to apply column reordering.

    After rearrange_columns() sets layer.self_attn.new_column_order,
    this hook ensures the activation is reordered before entering o_proj.
    Must be called AFTER add_actquant() wraps the layers.
    """
    from promix.quantize.quant_utils import ActQuantWrapper

    for layer in model.model.layers:
        o_proj = layer.self_attn.o_proj
        if isinstance(o_proj, ActQuantWrapper):
            attn = layer.self_attn

            def make_hook(attn_ref):
                def hook(module, args):
                    order = attn_ref.new_column_order
                    if order is not None:
                        x = args[0]
                        return (x[..., order],) + args[1:]
                    return args
                return hook

            o_proj.register_forward_pre_hook(make_hook(attn))
