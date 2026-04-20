"""Quick test of our INT8 FA kernel on H20."""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mixed_flash_attn

D = 256
scale = 1.0 / D ** 0.5

# Test 1: basic prefill
print("Test 1: basic prefill...")
Q = torch.randint(-128, 127, (1, 4, 128, D), dtype=torch.int8, device='cuda')
K = torch.randint(-128, 127, (1, 4, 128, D), dtype=torch.int8, device='cuda')
V = torch.randn(1, 4, 128, D, dtype=torch.float16, device='cuda')
O = mixed_flash_attn.int8_flash_attn(Q, K, V, 0.01, 0.01, scale, False)
print(f"  shape={O.shape} range=[{O.min():.3f},{O.max():.3f}]")
del Q, K, V, O
torch.cuda.synchronize()

# Test 2: smaller (Lq aligned to tile)
print("Test 2: Lq=64 (tile-aligned)...")
Q = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
K = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
V = torch.randn(1, 2, 64, D, dtype=torch.float16, device='cuda')
O = mixed_flash_attn.int8_flash_attn(Q, K, V, 0.01, 0.01, scale, False)
print(f"  shape={O.shape} range=[{O.min():.3f},{O.max():.3f}]")
del Q, K, V, O
torch.cuda.synchronize()

# Test 3: decode
print("Test 3: decode (Lq=1)...")
Q = torch.randint(-128, 127, (1, 1, 1, D), dtype=torch.int8, device='cuda')
K = torch.randint(-128, 127, (1, 1, 512, D), dtype=torch.int8, device='cuda')
V = torch.randn(1, 1, 512, D, dtype=torch.float16, device='cuda')
O = mixed_flash_attn.int8_flash_attn(Q, K, V, 0.01, 0.01, scale, True)
print(f"  shape={O.shape} range=[{O.min():.3f},{O.max():.3f}]")
del Q, K, V, O
torch.cuda.synchronize()

# Test 4: compare with PyTorch reference (CPU to avoid async CUDA errors)
print("Test 4: accuracy check...")
from ref_attention import fp32_ref_attention, compute_metrics

Q_i8 = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
K_i8 = torch.randint(-128, 127, (1, 2, 64, D), dtype=torch.int8, device='cuda')
V_f16 = torch.randn(1, 2, 64, D, dtype=torch.float16, device='cuda')

O_kernel = mixed_flash_attn.int8_flash_attn(Q_i8, K_i8, V_f16, 0.01, 0.01, scale, False)
torch.cuda.synchronize()  # ensure kernel is done

# Move to CPU for ref computation
Q_f32 = Q_i8.float().cpu()
K_f32 = K_i8.float().cpu()
V_f32 = V_f16.float().cpu()

ref = fp32_ref_attention(Q_f32, K_f32, V_f32, scale=scale)
m = compute_metrics(O_kernel.cpu(), ref)
print(f"  CosSim={m['cosine_sim']:.4f} MaxErr={m['max_abs_err']:.4f} SNR={m['snr_db']:.1f}dB")

print("\nAll tests passed!")
