"""ProMix foundation utilities."""

import gc
import logging

import torch
from fast_hadamard_transform import hadamard_transform

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
