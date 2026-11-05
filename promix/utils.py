"""ProMix foundation utilities."""

import gc
import logging

import torch
from fast_hadamard_transform import hadamard_transform

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def config_head_dim(config):
    """True per-head dim of the attention projections.

    Prefers an explicit ``config.head_dim`` (Qwen3 defines it independently of
    the residual width — e.g. Qwen3-32B: head_dim=128 while hidden//heads =
    5120//64 = 80) and falls back to ``hidden_size // num_attention_heads``
    (every Llama, where the two always coincide). All rotation/basis/ptq sites
    that reshape q/k/v/o per head MUST use this, never hidden//heads: the
    per-head weight blocks are head_dim-sized (q/k/v out_features =
    heads*head_dim, o_proj in_features = heads*head_dim), not
    (hidden//heads)-sized."""
    hd = getattr(config, "head_dim", None)
    return hd if hd else config.hidden_size // config.num_attention_heads


class HadamardTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, scale=1.0):
        # `scale` is applied INSIDE the fast_hadamard_transform CUDA kernel
        # (free fused multiply), so a caller that wants H(x)/sqrt(n) can pass
        # scale=1/sqrt(n) and avoid a separate elementwise div kernel. Default
        # 1.0 keeps every existing caller byte-identical.
        ctx.scale = scale
        return hadamard_transform(u, scale)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad, ctx.scale), None


def cleanup_memory(verbose=True):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            mem = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"GPU memory reserved: {mem:.2f} GB")
