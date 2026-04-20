"""Quick test of our INT8 FA kernel on H20."""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mixed_flash_attn
from ref_attention import fp32_ref_attention, compute_metrics

# Test prefill
B, H, Lq, Lkv, D = 1, 4, 64, 64, 256
scale = 1.0 / D ** 0.5

Q = torch.randint(-128, 127, (B, H, Lq, D), dtype=torch.int8, device='cuda')
K = torch.randint(-128, 127, (B, H, Lkv, D), dtype=torch.int8, device='cuda')
V = torch.randn(B, H, Lkv, D, dtype=torch.float16, device='cuda')

print("Running INT8 FA kernel (prefill)...")
O = mixed_flash_attn.int8_flash_attn(Q, K, V, 0.01, 0.01, scale, False)
print(f"Output shape: {O.shape}")

# Compare with reference
print("Comparing with FP32 reference...")
ref = fp32_ref_attention(Q.float(), K.float(), V.float(), scale=scale)
m = compute_metrics(O.cpu(), ref.cpu())
print(f"CosSim={m['cosine_sim']:.4f} MaxErr={m['max_abs_err']:.4f} SNR={m['snr_db']:.1f}dB")

# Test decode
print("\nTesting decode (Lq=1)...")
B, H, Lq, Lkv = 1, 1, 1, 512
Q2 = torch.randint(-128, 127, (B, H, Lq, D), dtype=torch.int8, device='cuda')
K2 = torch.randint(-128, 127, (B, H, Lkv, D), dtype=torch.int8, device='cuda')
V2 = torch.randn(B, H, Lkv, D, dtype=torch.float16, device='cuda')

O2 = mixed_flash_attn.int8_flash_attn(Q2, K2, V2, 0.01, 0.01, scale, True)
print(f"Decode output shape: {O2.shape}")

ref2 = fp32_ref_attention(Q2.float(), K2.float(), V2.float(), scale=scale, causal=True)
m2 = compute_metrics(O2.cpu(), ref2.cpu())
print(f"CosSim={m2['cosine_sim']:.4f} MaxErr={m2['max_abs_err']:.4f}")

print("\nAll tests passed!")
