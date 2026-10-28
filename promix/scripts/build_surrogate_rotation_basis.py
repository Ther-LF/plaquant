"""Build a surrogate PCA-basis file where ONLY the shared residual-stream
rotation ("attn_mlp", i.e. the matrix fused as `T` in
promix.quantize.rotation.build_main_transform) is replaced by a foreign
rotation (QuaRot-style randomized Hadamard, or SpinQuant's learned R1),
while every other key (per-layer value/key_pos/down_proj/o_proj_global PCA)
is copied byte-for-byte from the real ProMix basis file.

This isolates "choice of residual-stream rotation" (R1 in QuaRot/SpinQuant/
ResQ terminology) as the ONLY axis that differs between the pure-NVFP4 r=0
variants, holding ProMix's architecture-specific per-head V/K rotation and
down_proj online-Hadamard plumbing fixed across all three rotation choices.

Usage:
  python3 build_surrogate_rotation_basis.py quarot  <real_basis.bin> <out.bin> <hidden_dim> [seed]
  python3 build_surrogate_rotation_basis.py spinquant <real_basis.bin> <out.bin> <spinquant_R.bin>
"""
import math
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from promix.quantize.hadamard import get_hadK, is_pow2


def hadamard_pow2(m, device="cuda", dtype=torch.float64):
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    assert H.shape[0] == m
    return H


def randomized_hadamard(n, seed, device="cuda"):
    """Explicit n x n orthonormal rotation representing QuaRot's data-free R1.

    For power-of-2 n, this IS a literal randomized-Hadamard transform: the
    same Sylvester construction promix.quantize.hadamard's fast-WHT path
    uses, right-multiplied by a seeded random Rademacher sign diagonal
    (QuaRot's "randomized Hadamard transform").

    For non-power-of-2 n (e.g. Llama-3.2-3B hidden_size=3072=2^10*3), a
    literal structured Hadamard core requires a precomputed order-K Paley/
    Sylvester matrix (K=12 here); this repo only ships the order-28 core
    (promix/quantize/hadamard_data.json, sized for 8B's down_proj online
    Hadamard) and building a fresh order-12 core is out of scope for this
    ablation. We fall back to a seeded Haar-random orthogonal matrix (QR of
    a Gaussian, via basis.py/hadamard.py's own `random_orthogonal_matrix`)
    as the data-free, non-PCA, non-learned rotation QuaRot's R1 represents.
    QuaRot's own ablations find structured-Hadamard and Haar-random
    rotations statistically equivalent for incoherence processing; both
    are "PCA off, data-independent" rotations, which is the property this
    variant isolates against variant #3's PCA+R1.
    """
    if is_pow2(n):
        Hn = hadamard_pow2(n, device=device) / math.sqrt(n)
        g = torch.Generator(device="cpu").manual_seed(seed)
        signs = (torch.randint(0, 2, (n,), generator=g).double() * 2 - 1).to(device)
        Hn_rand = Hn * signs.unsqueeze(0)
        err = (Hn_rand @ Hn_rand.T - torch.eye(n, device=device, dtype=torch.float64)).abs().max().item()
        assert err < 1e-6, f"randomized_hadamard not orthonormal, max err={err}"
        return Hn_rand.cpu(), "literal-randomized-hadamard"
    else:
        from promix.quantize.hadamard import random_orthogonal_matrix as _roq
        Hn_rand = _roq(n, device)
        err = (Hn_rand @ Hn_rand.T - torch.eye(n, device=device, dtype=torch.float64)).abs().max().item()
        assert err < 1e-6, f"random orthogonal proxy not orthonormal, max err={err}"
        return Hn_rand.cpu(), "haar-random-proxy(non-pow2-n)"


def main():
    mode = sys.argv[1]
    real_basis_path = sys.argv[2]
    out_path = sys.argv[3]

    U_cpk = torch.load(real_basis_path, map_location="cpu", weights_only=False)
    assert "attn_mlp" in U_cpk, "expected 'attn_mlp' key in real basis file"
    hidden_dim = U_cpk["attn_mlp"].shape[0]
    print(f"real attn_mlp shape: {U_cpk['attn_mlp'].shape}")

    if mode == "quarot":
        hidden_dim_arg = int(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        assert hidden_dim_arg == hidden_dim
        T, kind = randomized_hadamard(hidden_dim, seed, device="cuda")
        print(f"[quarot] built {kind} {T.shape}, seed={seed}")
    elif mode == "spinquant":
        spinquant_R_path = sys.argv[4]
        d = torch.load(spinquant_R_path, map_location="cpu", weights_only=False)
        R1 = d["R1"].double()
        assert R1.shape == (hidden_dim, hidden_dim), f"SpinQuant R1 shape {R1.shape} != {hidden_dim}"
        err = (R1 @ R1.T - torch.eye(hidden_dim)).abs().max().item()
        print(f"[spinquant] R1 shape {R1.shape}, orthogonality max err={err}")
        assert err < 1e-4
        T = R1
    else:
        raise ValueError(mode)

    U_cpk["attn_mlp"] = T.to(torch.float64)
    torch.save(U_cpk, out_path)
    print(f"Wrote surrogate basis: {out_path}")


if __name__ == "__main__":
    main()
