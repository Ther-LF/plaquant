"""Unit tests for the uniform 64-ring block-diagonal R1 rotation substrate.

See docs spec "统一 64-环块对角 R1" §3. The full rotation is T = U @ R1 where U
is the dense PCA basis (unchanged) and R1 is block-diagonal with one fixed
Hadamard block of size `ring_size` (default 64) per ring across all of
hidden_dim. The rotation is shared across layers and applied once; per-layer
precision allocation only toggles which rings are high/low precision, never the
matrix itself.

Pure math, no GPU. Run:
    PYTHONPATH=/tmp/fhstub python -m pytest tests/test_ring_rotation.py -v
"""

import pytest
import torch

from promix.quantize.ring_rotation import build_uniform_ring_R1, compose_rotation


# Target hidden dims (all divisible by 64): 1B=2048, 3B=3072, 8B=4096.
HIDDEN_DIMS = [2048, 3072, 4096]
RING_SIZE = 64


def _random_orthogonal(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(size, size, dtype=torch.float64, generator=g)
    q, r = torch.linalg.qr(m)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_R1_is_orthogonal(hidden_dim):
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE)
    assert R1.shape == (hidden_dim, hidden_dim)
    assert R1.dtype == torch.float64
    ident = torch.eye(hidden_dim, dtype=torch.float64)
    assert torch.allclose(R1 @ R1.T, ident, atol=1e-8)
    assert torch.allclose(R1.T @ R1, ident, atol=1e-8)


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_R1_is_block_diagonal_at_ring_granularity(hidden_dim):
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE)
    n_rings = hidden_dim // RING_SIZE
    # Build a boolean mask for the block-diagonal region.
    block_mask = torch.zeros(hidden_dim, hidden_dim, dtype=torch.bool)
    for b in range(n_rings):
        s = b * RING_SIZE
        e = s + RING_SIZE
        block_mask[s:e, s:e] = True
    # Everything outside the diagonal blocks must be exactly zero.
    off_block = R1[~block_mask]
    assert torch.count_nonzero(off_block).item() == 0


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_each_diagonal_block_is_orthogonal(hidden_dim):
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE)
    n_rings = hidden_dim // RING_SIZE
    ident = torch.eye(RING_SIZE, dtype=torch.float64)
    for b in range(n_rings):
        s = b * RING_SIZE
        e = s + RING_SIZE
        block = R1[s:e, s:e]
        assert torch.allclose(block @ block.T, ident, atol=1e-8)


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_R1_det_is_plus_or_minus_one(hidden_dim):
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE)
    det = torch.linalg.det(R1)
    assert torch.isclose(det.abs(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_compose_with_random_orthogonal_stays_orthogonal(hidden_dim):
    U = _random_orthogonal(hidden_dim, seed=hidden_dim)
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE)
    T = compose_rotation(U, R1)
    assert T.shape == (hidden_dim, hidden_dim)
    ident = torch.eye(hidden_dim, dtype=torch.float64)
    assert torch.allclose(T @ T.T, ident, atol=1e-8)
    assert torch.allclose(T, U @ R1, atol=1e-12)


def test_hidden_dim_not_divisible_by_ring_raises():
    with pytest.raises(ValueError):
        build_uniform_ring_R1(2000, ring_size=RING_SIZE)  # 2000 % 64 != 0


def test_smaller_ring_size_also_works():
    # ring_size must still divide hidden_dim and be a power of two.
    R1 = build_uniform_ring_R1(256, ring_size=32)
    ident = torch.eye(256, dtype=torch.float64)
    assert torch.allclose(R1 @ R1.T, ident, atol=1e-8)


def test_non_power_of_two_ring_size_raises():
    # 96 % 48 == 0 (divides cleanly) but 48 is not a power of two, so the
    # Sylvester-Hadamard construction is undefined and must raise.
    with pytest.raises(ValueError):
        build_uniform_ring_R1(96, ring_size=48)


# ---------------------------------------------------------------------------
# build_main_transform wiring (opt-in ring-R1 regime, spec §3). rotation.py
# pulls in the Hadamard/CUDA chain at import; skip these if it cannot import
# in the pure-math test environment (no fast_hadamard_transform stub).
# ---------------------------------------------------------------------------
try:
    from promix.quantize.rotation import build_main_transform
    _HAVE_ROTATION = True
except Exception:  # pragma: no cover - import-env dependent
    _HAVE_ROTATION = False

rotation_required = pytest.mark.skipif(
    not _HAVE_ROTATION, reason="rotation.py (Hadamard chain) unimportable here")


@rotation_required
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
def test_build_main_transform_ring_r1_equals_U_times_ring(hidden_dim):
    U = _random_orthogonal(hidden_dim, seed=hidden_dim)
    T = build_main_transform(U, {}, identity_main_rotation=False,
                             ring_r1=True, ring_size=RING_SIZE)
    R1 = build_uniform_ring_R1(hidden_dim, ring_size=RING_SIZE).to(U.device)
    assert torch.allclose(T, U @ R1, atol=1e-10)
    ident = torch.eye(hidden_dim, dtype=torch.float64)
    assert torch.allclose(T @ T.T, ident, atol=1e-8)


@rotation_required
def test_build_main_transform_ring_r1_takes_priority_over_identity():
    # ring_r1 must win even when identity_main_rotation is also set.
    U = _random_orthogonal(2048, seed=7)
    T = build_main_transform(U, {}, identity_main_rotation=True,
                             ring_r1=True, ring_size=RING_SIZE)
    R1 = build_uniform_ring_R1(2048, ring_size=RING_SIZE).to(U.device)
    assert torch.allclose(T, U @ R1, atol=1e-10)


@rotation_required
def test_build_main_transform_default_off_is_identity_path():
    # ring_r1=False + identity_main_rotation=True => T == U (byte-identical).
    U = _random_orthogonal(2048, seed=11)
    T = build_main_transform(U, {}, identity_main_rotation=True)
    assert T is U


@rotation_required
def test_build_main_transform_default_off_dense_path_unchanged():
    # ring_r1=False + dense R_dict => T == U @ block_diag(R1_1, R1_2).
    d = 2048
    U = _random_orthogonal(d, seed=3)
    high = 256
    R1_1 = _random_orthogonal(d - high, seed=1)
    R1_2 = _random_orthogonal(high, seed=2)
    R_dict = {"R1_1": R1_1, "R1_2": R1_2, "R1_0": None}
    T = build_main_transform(U, R_dict, identity_main_rotation=False)
    expected = U @ torch.block_diag(R1_1, R1_2)
    assert torch.allclose(T, expected, atol=1e-10)
