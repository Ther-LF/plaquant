"""Minimal debug script for fused mixed GEMM kernel."""
import torch
import mixed_gemm_l20

M, N, K_high, K_low = 128, 2048, 256, 1792
device = 'cuda'

print(f"M={M}, N={N}, K_high={K_high}, K_low={K_low}")

# Create inputs
A_low = torch.randint(-128, 127, (M, K_low), dtype=torch.int8, device=device)
B_low = torch.randint(-128, 127, (N, K_low), dtype=torch.int8, device=device)
A_high = torch.randint(-128, 127, (M, K_high), dtype=torch.int8, device=device)
B_high = torch.randint(-128, 127, (N, K_high), dtype=torch.int8, device=device)

print(f"A_low: {A_low.shape}, stride={A_low.stride()}, contiguous={A_low.is_contiguous()}")
print(f"B_low: {B_low.shape}, stride={B_low.stride()}, contiguous={B_low.is_contiguous()}")
print(f"A_high: {A_high.shape}, stride={A_high.stride()}, contiguous={A_high.is_contiguous()}")
print(f"B_high: {B_high.shape}, stride={B_high.stride()}, contiguous={B_high.is_contiguous()}")

torch.cuda.synchronize()
print("\nCalling fused_mixed_gemm...")
try:
    out = mixed_gemm_l20.fused_mixed_gemm(A_low, B_low, A_high, B_high)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}")
    print(f"Output sample: {out[0,:5]}")
    print(f"Has NaN: {out.isnan().any().item()}")
    print(f"Has Inf: {out.isinf().any().item()}")
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

# Test with smaller dimensions to narrow down
print("\n\nTrying smaller dimensions...")
for test_M, test_N, test_Kl, test_Kh in [(128, 128, 64, 64), (128, 128, 128, 64)]:
    A_l = torch.randint(-128, 127, (test_M, test_Kl), dtype=torch.int8, device=device)
    B_l = torch.randint(-128, 127, (test_N, test_Kl), dtype=torch.int8, device=device)
    A_h = torch.randint(-128, 127, (test_M, test_Kh), dtype=torch.int8, device=device)
    B_h = torch.randint(-128, 127, (test_N, test_Kh), dtype=torch.int8, device=device)
    try:
        torch.cuda.synchronize()
        o = mixed_gemm_l20.fused_mixed_gemm(A_l, B_l, A_h, B_h)
        torch.cuda.synchronize()
        print(f"  M={test_M} N={test_N} Kl={test_Kl} Kh={test_Kh} → OK, out[0,0]={o[0,0].item()}")
    except Exception as e:
        print(f"  M={test_M} N={test_N} Kl={test_Kl} Kh={test_Kh} → FAIL: {e}")
