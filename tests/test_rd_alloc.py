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


from promix.quantize.adaptive import allocate_by_budget, masks_to_khigh


def test_allocate_hits_budget_and_picks_highest_deltaD():
    # layer0: deltaD [0.08,0.02,0.005,0.001]; layer1: [0.015,0.009,0.004,0.002]
    dd = [torch.tensor([0.08, 0.02, 0.005, 0.001]),
          torch.tensor([0.015, 0.009, 0.004, 0.002])]
    # 8 rings total, target mean fp8 = 0.25 -> 2 FP8 rings total
    masks = allocate_by_budget(dd, target_mean_fp8=0.25)
    total_fp8 = sum(int(m.sum()) for m in masks)
    assert total_fp8 == 2
    # the 2 globally-highest deltaD rings are layer0-ring0 (0.08) and layer0-ring1 (0.02)
    assert masks[0].tolist() == [True, True, False, False]
    assert masks[1].tolist() == [False, False, False, False]


def test_masks_to_khigh_alignment():
    masks = [torch.tensor([True, True, False, False]),
             torch.tensor([False, False, False, False])]
    kh = masks_to_khigh(masks, ring_size=64)
    assert kh == [128, 0]
    assert all(k % 64 == 0 for k in kh)



from promix.quantize.adaptive import rd_alloc_config


def test_rd_alloc_config_disabled_by_default():
    # Absent block -> disabled (zero-regression: no override installed).
    c = rd_alloc_config({})
    assert c['enabled'] is False
    assert c['ring_size'] == 64 and c['mode'] == 'perf' and c['target'] == 0.125
    # Explicit enabled:false (the shipped 3B config) is also disabled.
    c2 = rd_alloc_config({'rd_alloc': {'enabled': False, 'ring_size': 64,
                                       'mode': 'perf', 'target': 0.125}})
    assert c2['enabled'] is False


def test_rd_alloc_config_enabled_reads_fields():
    c = rd_alloc_config({'rd_alloc': {'enabled': True, 'ring_size': 64,
                                      'mode': 'accuracy', 'target': 8.30}})
    assert c['enabled'] is True
    assert c['mode'] == 'accuracy' and c['target'] == 8.30 and c['ring_size'] == 64
