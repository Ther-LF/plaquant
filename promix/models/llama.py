"""Llama model plugin — supports Llama-2, Llama-3, Llama-3.2 family."""

from typing import Dict, List
import torch.nn as nn

from .base import ModelPlugin


class LlamaPlugin(ModelPlugin):
    model_type = "llama"

    def get_layers(self, model) -> List[nn.Module]:
        return list(model.model.layers)

    def get_projections(self, layer) -> Dict[str, nn.Module]:
        return {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }

    def get_embedding(self, model) -> nn.Module:
        return model.model.embed_tokens

    def get_lm_head(self, model) -> nn.Module:
        return model.lm_head

    def get_norm(self, model) -> nn.Module:
        return model.model.norm

    def get_hidden_size(self, model) -> int:
        return model.config.hidden_size

    def get_intermediate_size(self, model) -> int:
        return model.config.intermediate_size

    def get_num_heads(self, model) -> int:
        return model.config.num_attention_heads

    def get_head_dim(self, model) -> int:
        return model.config.hidden_size // model.config.num_attention_heads
