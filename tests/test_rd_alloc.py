import torch
from promix.quantize.adaptive import ring_mse, ring_deltaD


def test_ring_mse_shape_and_monotonic():
    torch.manual_seed(0)
    d, M, ring = 256, 512, 64
    x = torch.randn(M, d)
    mse_fp4 = ring_mse(x, ring, 'nvfp4')
    mse_fp8 = ring_mse(x, ring, 'mxfp8')
    assert mse_fp4.shape == (d // ring,)
    assert mse_fp8.shape == (d // ring,)
    # FP8 (8-bit) must be at least as accurate as FP4 (4-bit) per ring
    assert torch.all(mse_fp8 <= mse_fp4 + 1e-8)


def test_ring_deltaD_matches_difference():
    torch.manual_seed(1)
    d, M, ring = 128, 256, 64
    x = torch.randn(M, d)
    dd = ring_deltaD(x, ring)
    expected = ring_mse(x, ring, 'nvfp4') - ring_mse(x, ring, 'mxfp8')
    assert torch.allclose(dd, expected, atol=1e-10)
    assert torch.all(dd >= -1e-8)  # upgrading never hurts


def test_ring_mse_catches_outlier():
    # A heavy-tailed ring must show larger FP4 MSE than a Gaussian ring.
    # Fixture note (deviation from plan sketch): the plan's version concentrated
    # one 50x channel then renormalized each ring to equal TOTAL variance. That
    # crushes every non-outlier channel to near-zero magnitude, so the heavy
    # ring's *absolute* MSE actually drops below the Gaussian ring's — it tests
    # the opposite of what it claims. NVFP4 scales per 16-channel block (per
    # row/token, along the last dim), so the real outlier effect is: an outlier
    # inflates its block's scale and coarsens the quantization of its neighbors.
    # We isolate that by using an IDENTICAL unit-variance bulk in both rings and
    # adding scattered outliers only to the heavy ring (no renormalization), so
    # the only difference is the outliers themselves. Deterministic: ~0.31 vs
    # ~0.0074 FP4 MSE. Keeps the assertion's intent (heavy ring >= Gaussian).
    torch.manual_seed(2)
    ring = 64
    bulk = torch.randn(1024, ring)            # shared unit-variance bulk
    gauss = bulk.clone()
    heavy = bulk.clone()
    heavy = heavy + (torch.rand(1024, ring) < 0.02).float() * 50.0  # scattered outliers
    x = torch.cat([gauss, heavy], dim=1)      # ring0=gauss, ring1=heavy
    mse = ring_mse(x, ring, 'nvfp4')
    assert mse[1] > mse[0]  # outlier ring hurts more under FP4
