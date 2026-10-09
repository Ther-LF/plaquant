import torch
from promix.quantize.adaptive import (
    per_channel_variance_in_basis, tail_energy_fraction,
    derive_khigh, derive_per_layer_khigh, main_path_avg_bits,
)
from promix.quantize.rotation import build_main_transform
from promix.eval.ptq import layer_index_of

def test_variance_in_basis_identity():
    H = torch.diag(torch.tensor([1., 2., 3., 4.]))
    U = torch.eye(4)
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(v, torch.tensor([1., 2., 3., 4.], dtype=v.dtype))

def test_variance_in_basis_rotates():
    # U^T H U diagonal equals eigenvalues when U are eigenvectors
    H = torch.tensor([[2., 1.], [1., 2.]])
    evals, U = torch.linalg.eigh(H)
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(torch.sort(v).values,
                          torch.sort(evals.double()).values, atol=1e-10)

def test_tail_energy_endpoints():
    v = torch.tensor([0., 0., 1., 3.])      # total 4
    cum = tail_energy_fraction(v)           # last1=3/4, last2=4/4, ...
    assert torch.isclose(cum[0], torch.tensor(0.75, dtype=cum.dtype))
    assert torch.isclose(cum[-1], torch.tensor(1.0, dtype=cum.dtype))

def test_derive_khigh_alignment_and_threshold():
    # 16 channels, all energy in last 4 → need k>=4, round up to 64 then
    # clamp to hidden_dim (16) → but with align=4 for this small case:
    v = torch.zeros(16); v[-4:] = 1.0
    assert derive_khigh(v, tau=0.9, align=4, hidden_dim=16) == 4
    # ascending 1..16, tau=0.5: smallest tail k with sum>=68 is k=5 → align4 → 8
    v2 = torch.arange(1, 17, dtype=torch.float64)
    assert derive_khigh(v2, tau=0.5, align=4, hidden_dim=16) == 8

def test_derive_khigh_clamped_to_align_min():
    v = torch.zeros(16); v[-1] = 1.0
    # need k=1 → round up to align 4
    assert derive_khigh(v, tau=0.9, align=4, hidden_dim=16) == 4

def test_derive_per_layer_list():
    curves = [torch.zeros(16), torch.zeros(16)]
    curves[0][-4:] = 1.0      # needs 4
    curves[1][-8:] = 1.0      # needs 8
    ks = derive_per_layer_khigh(curves, tau=0.9, align=4, hidden_dim=16)
    assert ks == [4, 8]

def test_avg_bits():
    # 2 layers, hidden 16: K_high 4 and 8
    # layer0: (12*4 + 4*8)/16 = (48+32)/16 = 5.0
    # layer1: (8*4  + 8*8)/16 = (32+64)/16 = 6.0 → mean 5.5
    assert abs(main_path_avg_bits([4, 8], hidden_dim=16) - 5.5) < 1e-9

def test_variance_matches_eigvals_when_U_is_eigenbasis():
    # When U diagonalizes H, diag(U^T H U) == eigenvalues. This is the
    # property basis.py relies on to store per-layer curves consistently
    # with the shared eigenvalue ordering.
    torch.manual_seed(0)
    A = torch.randn(8, 8, dtype=torch.float64)
    H = A @ A.T                          # SPD
    evals, U = torch.linalg.eigh(H)      # ascending
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(v, evals, atol=1e-8)

def test_build_main_transform_identity_returns_U():
    U = torch.randn(8, 8, dtype=torch.float64)
    T = build_main_transform(U, R_dict=None, identity_main_rotation=True)
    assert torch.allclose(T, U)

def test_build_main_transform_applies_R1_when_not_identity():
    U = torch.eye(4, dtype=torch.float64)
    R_dict = {"R1_1": torch.eye(2, dtype=torch.float64) * 2,
              "R1_2": torch.eye(2, dtype=torch.float64) * 3,
              "R1_0": None}
    T = build_main_transform(U, R_dict=R_dict, identity_main_rotation=False)
    expected = torch.block_diag(torch.eye(2) * 2, torch.eye(2) * 3).double()
    assert torch.allclose(T, expected)

def test_layer_index_of():
    assert layer_index_of("model.layers.3.self_attn.q_proj.module") == 3
    assert layer_index_of("model.layers.11.mlp.gate_proj") == 11
    assert layer_index_of("lm_head") is None
