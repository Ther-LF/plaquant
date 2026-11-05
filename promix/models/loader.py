"""Llama model loading and patching for ProMix quantization.

Uses standard HuggingFace LlamaForCausalLM and patches it for:
- tie_word_embeddings handling
- new_column_order support via pre-forward hooks on o_proj
"""

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_model(model_name, dtype=torch.float16):
    """Load Llama model with untied word embeddings (required for rotation).

    PROMIX_LOAD_DEVICE=cpu (rd-scale, big models): keep the model CPU-resident
    instead of the default whole-model .cuda(). Every downstream build step is
    already device-safe for a CPU-resident model — the rotation fusers move one
    weight at a time to CUDA (rotation.py), and GPTQ/PPL eval are
    layer-offloading (gptq_fwrd / evaluate_ppl) — so this is purely a residency
    choice; 70B (140 GB fp16) cannot fit the 96 GB card. Default unchanged.
    """
    config = AutoConfig.from_pretrained(model_name)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, config=config
    )
    if os.environ.get("PROMIX_LOAD_DEVICE", "cuda").lower() != "cpu":
        model = model.cuda()

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Initialize new_column_order slots (set by rearrange_columns later)
    for layer in model.model.layers:
        layer.self_attn.new_column_order = None

    return model


def install_column_order_hooks(model):
    """Store column_order directly on o_proj wrappers.

    After rearrange_columns() sets layer.self_attn.new_column_order,
    this copies it to the wrapper so it's applied during forward.
    Must be called AFTER add_actquant() and rearrange_columns().
    """
    from promix.quantize.quant_utils import ActQuantWrapper

    for layer in model.model.layers:
        o_proj = layer.self_attn.o_proj
        if isinstance(o_proj, ActQuantWrapper):
            order = getattr(layer.self_attn, 'new_column_order', None)
            o_proj._column_order = order
