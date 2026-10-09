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
