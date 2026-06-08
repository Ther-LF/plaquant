"""Hadamard transform utilities for ProMix.

Uses fast_hadamard_transform library for power-of-2 dimensions.
For non-power-of-2 intermediate sizes, loads pre-computed Hadamard matrices.
"""

import json
import os

import torch

from promix.utils import HadamardTransform

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def get_hadK(n, transpose=False):
    """Get Hadamard-like matrix for dimension n.

    For power-of-2 dimensions (most modern LLMs), returns (None, 1)
    meaning fast_hadamard_transform handles everything.
    For non-power-of-2, returns a pre-computed Hadamard matrix.
    """
    if is_pow2(n):
        return None, 1

    for K in [172, 156, 140, 108, 60, 52, 44, 40, 36, 28, 20, 12]:
        if n % K == 0 and is_pow2(n // K):
            hadK = _get_hadamard_matrix(K)
            return (hadK.T if transpose else hadK), K

    raise ValueError(f"Cannot find Hadamard decomposition for n={n}")


def matmul_hadU_cuda(X, had_K, K):
    """Apply Hadamard rotation: X @ kron(had_K, H) / sqrt(n)."""
    n = X.shape[-1]
    if K == 1:
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(n).sqrt()
    input = X.view(-1, K, n // K)
    input = HadamardTransform.apply(input.contiguous()) / torch.tensor(n).sqrt()
    input = had_K.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def random_orthogonal_matrix(size, device):
    """Generate random orthogonal matrix via QR decomposition."""
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


_hadamard_cache = {}


def _get_hadamard_matrix(K):
    """Get pre-computed Hadamard matrix of order K."""
    if K in _hadamard_cache:
        return _hadamard_cache[K]

    data_path = os.path.join(_DATA_DIR, "hadamard_data.json")
    key = f"had{K}"

    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        if key in data:
            had = torch.tensor(data[key], dtype=torch.float32)
            _hadamard_cache[K] = had
            return had

    raise ValueError(
        f"No pre-computed Hadamard matrix for K={K}. "
        f"Please add '{key}' to {data_path}."
    )
