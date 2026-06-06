"""ProMix foundation utilities."""

import gc
import logging

import torch
from fast_hadamard_transform import hadamard_transform

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class HadamardTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad)


def cleanup_memory(verbose=True):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            mem = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"GPU memory reserved: {mem:.2f} GB")
