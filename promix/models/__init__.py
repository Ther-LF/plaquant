"""Model plugin registry."""

from .base import ModelPlugin
from .llama import LlamaPlugin

_REGISTRY = {
    "llama": LlamaPlugin,
}


def get_model_plugin(model_type: str) -> ModelPlugin:
    """Get model plugin by type name."""
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[model_type]()


def register_model_plugin(model_type: str, plugin_cls):
    """Register a new model plugin."""
    _REGISTRY[model_type] = plugin_cls
