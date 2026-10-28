"""Uniform ring block-diagonal R1 rotation substrate.

Implements the "统一 64-环块对角 R1" design (spec
``docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md`` §3).

The full rotation is ``T = U @ R1`` where ``U`` is the dense PCA basis (global,
variance-sorted, unchanged) and ``R1`` is block-diagonal with one fixed
``ring_size × ring_size`` orthogonal Hadamard block per ring, covering all of
``hidden_dim``. Each ring only rotates within itself, so a per-layer precision
allocation that marks some rings MXFP8 (high) and the rest NVFP4 (low) never
needs to change the matrix: R1 is built once, shared across layers, and baked
into the embedding (zero online cost).

This first version uses a FIXED (not trained) Hadamard per ring, per spec §3.3
(fixed-Hadamard rings first, no Step-1 retrain). The ``ring_size`` must divide
``hidden_dim`` and be a power of two (64 = 2**6 is the default; it is the
smallest granularity satisfying both K_high % 32 == 0 and K_low % 64 == 0).

Pure math, torch only — no GPU, no fast_hadamard_transform dependency.
"""

import torch


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sylvester_hadamard(ring_size: int, dtype=torch.float64) -> torch.Tensor:
    """Return a normalized (orthogonal) Sylvester-Hadamard matrix.

    Built recursively as ``H_{2n} = [[H_n, H_n], [H_n, -H_n]]`` starting from
    ``H_1 = [[1]]``, then scaled by ``1/sqrt(ring_size)`` so the result is
    orthogonal (``Q @ Q.T == I``). ``ring_size`` must be a power of two.
    """
    if not _is_pow2(ring_size):
        raise ValueError(
            f"ring_size must be a power of two for a Sylvester Hadamard; got {ring_size}"
        )
    H = torch.ones((1, 1), dtype=dtype)
    while H.shape[0] < ring_size:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    # Normalize to an orthogonal matrix.
    return H / torch.sqrt(torch.tensor(float(ring_size), dtype=dtype))


def build_uniform_ring_R1(hidden_dim: int, ring_size: int = 64, dtype=torch.float64) -> torch.Tensor:
    """Build the uniform ring block-diagonal R1 rotation.

    Args:
        hidden_dim: Full hidden dimension (e.g. 2048/3072/4096). Must be
            divisible by ``ring_size``.
        ring_size: Size of each diagonal Hadamard block (default 64, a power of
            two). Must divide ``hidden_dim`` and be a power of two.
        dtype: Output dtype (default float64 for numerically clean rotations).

    Returns:
        A ``(hidden_dim, hidden_dim)`` orthogonal, block-diagonal tensor with
        ``hidden_dim // ring_size`` identical normalized Hadamard blocks on the
        diagonal and exact zeros elsewhere.
    """
    if ring_size <= 0:
        raise ValueError(f"ring_size must be positive; got {ring_size}")
    if hidden_dim % ring_size != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by ring_size ({ring_size})"
        )
    if not _is_pow2(ring_size):
        raise ValueError(
            f"ring_size must be a power of two for a Sylvester Hadamard; got {ring_size}"
        )

    block = sylvester_hadamard(ring_size, dtype=dtype)
    n_rings = hidden_dim // ring_size
    # block_diag of n_rings identical Hadamard blocks; off-block entries are
    # filled with exact zeros by torch.block_diag.
    return torch.block_diag(*([block] * n_rings))


def compose_rotation(U: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Compose the dense PCA basis U with the ring rotation R1: ``T = U @ R1``.

    If ``U`` is orthogonal and ``R1`` is orthogonal (as produced by
    :func:`build_uniform_ring_R1`), the product is orthogonal. ``U`` and ``R1``
    are promoted to a common dtype for the matmul.
    """
    if U.shape[-1] != R1.shape[0]:
        raise ValueError(
            f"U last dim ({U.shape[-1]}) must match R1 first dim ({R1.shape[0]})"
        )
    dtype = torch.promote_types(U.dtype, R1.dtype)
    return U.to(dtype) @ R1.to(dtype)
