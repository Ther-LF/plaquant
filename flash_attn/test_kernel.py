"""Quick test of our INT8 FA kernel on H20."""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mixed_flash_attn
from ref_attention import fp32_ref_attention, compute_metrics


def quantize_int8(x):
    amax = x.abs().max()
    scale = amax / 127.0
    q = torch.round(x / (scale + 1e-8)).clamp(-128, 127).to(torch.int8)
    return q, scale.item()


D = 256
softmax_scale = 1.0 / D ** 0.5

# Generate FP32 data and quantize
torch.manual_seed(42)
q_f32 = torch.randn(1, 2, 64, D)
k_f32 = torch.randn(1, 2, 64, D)
v_f32 = torch.randn(1, 2, 64, D)

Q_i8, sq = quantize_int8(q_f32)
K_i8, sk = quantize_int8(k_f32)
V_f16 = v_f32.half().cuda()

print(f"Scales: sq={sq:.6f}, sk={sk:.6f}, ss={softmax_scale:.6f}")

# Reference (FP32)
ref = fp32_ref_attention(q_f32, k_f32, v_f32, scale=softmax_scale)

# Our kernel
O = mixed_flash_attn.int8_flash_attn(
    Q_i8.cuda(), K_i8.cuda(), V_f16,
    sq, sk, softmax_scale, False)
torch.cuda.synchronize()

m = compute_metrics(O.cpu(), ref)
print(f"CosSim={m['cosine_sim']:.4f} MaxErr={m['max_abs_err']:.4f} "
      f"RMSE={m['rmse']:.4f} SNR={m['snr_db']:.1f}dB")

if m['cosine_sim'] > 0.99:
    print("ACCURACY CHECK PASSED!")
else:
    print("ACCURACY CHECK FAILED (expected CosSim > 0.99)")
