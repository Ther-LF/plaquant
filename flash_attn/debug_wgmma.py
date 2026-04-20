"""Debug WGMMA: test with different scales."""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mixed_flash_attn
from ref_attention import fp32_ref_attention, compute_metrics

D = 256
scale_s = 1.0 / D ** 0.5
V = torch.randn(1, 1, 64, D, dtype=torch.float16, device='cuda')

# Test 1: small INT8 values, scale=0.1
print("Test 1: small INT8 + scale=0.1")
torch.manual_seed(42)
Q1 = torch.randint(-8, 7, (1, 1, 64, D), dtype=torch.int8, device='cuda')
K1 = torch.randint(-8, 7, (1, 1, 64, D), dtype=torch.int8, device='cuda')
O1 = mixed_flash_attn.int8_flash_attn(Q1, K1, V, 0.1, 0.1, scale_s, False)
torch.cuda.synchronize()
ref1 = fp32_ref_attention(Q1.float() * 0.1, K1.float() * 0.1, V.float(), scale=scale_s)
m1 = compute_metrics(O1.cpu(), ref1.cpu())
print(f"  CosSim={m1['cosine_sim']:.4f} MaxErr={m1['max_abs_err']:.4f}")

# Test 2: full range INT8, proper scale
print("Test 2: full range INT8 + scale=0.031")
torch.manual_seed(42)
sq = 4.0 / 127.0
Q2 = torch.randint(-128, 127, (1, 1, 64, D), dtype=torch.int8, device='cuda')
K2 = torch.randint(-128, 127, (1, 1, 64, D), dtype=torch.int8, device='cuda')
O2 = mixed_flash_attn.int8_flash_attn(Q2, K2, V, sq, sq, scale_s, False)
torch.cuda.synchronize()
ref2 = fp32_ref_attention(Q2.float() * sq, K2.float() * sq, V.float(), scale=scale_s)
m2 = compute_metrics(O2.cpu(), ref2.cpu())
print(f"  CosSim={m2['cosine_sim']:.4f} MaxErr={m2['max_abs_err']:.4f}")

# Test 3: all ones (should be perfect)
print("Test 3: all ones + scale=1.0")
Q3 = torch.ones(1, 1, 64, D, dtype=torch.int8, device='cuda')
K3 = torch.ones(1, 1, 64, D, dtype=torch.int8, device='cuda')
V3 = torch.ones(1, 1, 64, D, dtype=torch.float16, device='cuda')
O3 = mixed_flash_attn.int8_flash_attn(Q3, K3, V3, 1.0, 1.0, scale_s, False)
torch.cuda.synchronize()
ref3 = fp32_ref_attention(Q3.float(), K3.float(), V3.float(), scale=scale_s)
m3 = compute_metrics(O3.cpu(), ref3.cpu())
print(f"  CosSim={m3['cosine_sim']:.4f} MaxErr={m3['max_abs_err']:.4f}")

# Test 4: original test_kernel.py scales
print("Test 4: test_kernel.py scales (sq=0.036, sk=0.034)")
torch.manual_seed(42)
sq4, sk4 = 0.036146, 0.034058
Q4 = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
K4 = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
V4 = torch.randn(1, 2, 64, D, dtype=torch.float16, device='cuda')
O4 = mixed_flash_attn.int8_flash_attn(Q4, K4, V4, sq4, sk4, scale_s, False)
torch.cuda.synchronize()
ref4 = fp32_ref_attention(Q4.float() * sq4, K4.float() * sk4, V4.float(), scale=scale_s)
m4 = compute_metrics(O4.cpu(), ref4.cpu())
print(f"  CosSim={m4['cosine_sim']:.4f} MaxErr={m4['max_abs_err']:.4f}")
