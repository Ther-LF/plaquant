"""Model plugin base class — defines the interface for model-specific logic."""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch.nn as nn


class ModelPlugin(ABC):
    """Every model family implements this interface.

    Quantize/eval code calls these methods instead of hardcoding
    layer names like 'model.layers' or 'self_attn.q_proj'.
    """

    model_type: str = ""

    @abstractmethod
    def get_layers(self, model) -> List[nn.Module]:
        """Return the list of transformer decoder layers."""
        ...

    @abstractmethod
    def get_projections(self, layer) -> Dict[str, nn.Module]:
        """Return name→module mapping for all linear projections in a layer.

        Expected keys: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        """
        ...

    @abstractmethod
    def get_embedding(self, model) -> nn.Module:
        """Return the word embedding layer."""
        ...

    @abstractmethod
    def get_lm_head(self, model) -> nn.Module:
        """Return the output language model head."""
        ...

    @abstractmethod
    def get_norm(self, model) -> nn.Module:
        """Return the final layer norm (before lm_head)."""
        ...

    @abstractmethod
    def get_hidden_size(self, model) -> int:
        """Return hidden dimension."""
        ...

    @abstractmethod
    def get_intermediate_size(self, model) -> int:
        """Return MLP intermediate dimension."""
        ...

    @abstractmethod
    def get_num_heads(self, model) -> int:
        """Return number of attention heads."""
        ...

    @abstractmethod
    def get_head_dim(self, model) -> int:
        """Return per-head dimension."""
        ...

    def get_attn(self, layer) -> nn.Module:
        """Return the self-attention module from a layer."""
        return layer.self_attn

    def get_mlp(self, layer) -> nn.Module:
        """Return the MLP module from a layer."""
        return layer.mlp

    def get_input_layernorm(self, layer) -> nn.Module:
        """Return input layernorm of a layer."""
        return layer.input_layernorm

    def get_post_attn_layernorm(self, layer) -> nn.Module:
        """Return post-attention layernorm of a layer."""
        return layer.post_attention_layernorm
