import torch
from promix.quantize.adaptive import (
    per_channel_variance_in_basis, tail_energy_fraction,
    derive_khigh, derive_per_layer_khigh, main_path_avg_bits,
)

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
