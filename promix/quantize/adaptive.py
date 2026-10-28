"""Energy-threshold-driven per-layer high-precision cut (main path).

See docs/superpowers/specs/2026-10-09-adaptive-high-fraction-design.md.
All curves are in shared-U column order (ascending variance), so the
high-variance channels are the TAIL and the quantizer's last-K slice
matches `derive_khigh`'s tail-energy accounting.
"""
import torch


def per_channel_variance_in_basis(U, H):
    """diag(Uᵀ H U): per-channel variance after projecting covariance H
    onto basis U (columns = basis vectors). Returns a (d,) tensor whose
    c-th entry is the variance along U[:, c]."""
    U = torch.as_tensor(U, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    M = H @ U                      # (d, d)
    return (U * M).sum(dim=0)      # diag(Uᵀ H U)


def tail_energy_fraction(var_curve):
    """Cumulative fraction of total variance captured by the LAST k channels,
    for k = 1..n. Returns a (n,) tensor; entry [k-1] = frac for last k."""
    v = torch.as_tensor(var_curve, dtype=torch.float64).flatten().clamp_min(0)
    total = v.sum()
    if total <= 0:
        raise ValueError("variance curve sums to <= 0")
    cum = torch.cumsum(torch.flip(v, dims=[0]), dim=0) / total
    return cum


def _round_up(x, m):
    return ((int(x) + m - 1) // m) * m


def derive_khigh(var_curve, tau, align=64, hidden_dim=None):
    """Smallest multiple of `align` such that the last K_high channels capture
    >= tau of the variance. Clamped to [align, hidden_dim]."""
    v = torch.as_tensor(var_curve, dtype=torch.float64).flatten().clamp_min(0)
    n = hidden_dim if hidden_dim is not None else v.numel()
    cum = tail_energy_fraction(v)
    meets = (cum >= tau).nonzero()
    k = int(meets[0].item()) + 1 if len(meets) else n
    return max(align, min(_round_up(k, align), n))


def derive_per_layer_khigh(var_curves, tau, align=64, hidden_dim=None):
    """Per-layer K_high list from a list of variance curves."""
    return [derive_khigh(v, tau, align=align, hidden_dim=hidden_dim)
            for v in var_curves]


def main_path_avg_bits(khigh_list, hidden_dim, low_bits=4, high_bits=8):
    """Mean over layers of (K_low*low_bits + K_high*high_bits)/hidden_dim."""
    per = [((hidden_dim - k) * low_bits + k * high_bits) / hidden_dim
           for k in khigh_list]
    return sum(per) / len(per)


# ---------------------------------------------------------------------------
# Rate-distortion mixed-precision allocation
# (docs/superpowers/plans/2026-10-28-rate-distortion-mixed-precision-allocation.md)
# ---------------------------------------------------------------------------
from .quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8

_FMT = {'nvfp4': fake_quantize_nvfp4, 'mxfp8': fake_quantize_mxfp8}


def ring_mse(x_rot, ring_size, fmt):
    """Per-ring measured quantization MSE on rotated activations/weights.

    Uses the deployment pipeline's real fake-quant (the same MSE-searched
    block scales the kernel sees), so the measured MSE matches what the
    model is actually served with. `fake_quantize_nvfp4`/`_mxfp8` quantize
    along the last dim with their own block sizes (16 / 32) and derive
    block scales internally; `ring_size` (64) is a whole multiple of both,
    so per-ring MSE is simply the mean squared dequant error over each
    64-channel group.

    x_rot: (M, d). Returns (d//ring_size,) float tensor.
    """
    if fmt not in _FMT:
        raise ValueError(f"fmt must be one of {list(_FMT)}, got {fmt}")
    x = torch.as_tensor(x_rot, dtype=torch.float32)
    M, d = x.shape
    if d % ring_size != 0:
        raise ValueError(f"d={d} not a multiple of ring_size={ring_size}")
    q = _FMT[fmt](x)                       # dequantized, same shape
    err2 = (x - q).pow(2)                  # (M, d)
    per_ring = err2.reshape(M, d // ring_size, ring_size).mean(dim=(0, 2))
    return per_ring                        # (d//ring_size,)


def ring_deltaD(x_rot, ring_size):
    """D_fp4 - D_fp8 per ring (error reduced by upgrading FP4 -> FP8)."""
    return ring_mse(x_rot, ring_size, 'nvfp4') - ring_mse(x_rot, ring_size, 'mxfp8')
