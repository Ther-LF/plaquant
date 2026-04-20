"""Debug WGMMA: test with known input values."""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mixed_flash_attn

D = 256
B, H, Lq, Lkv = 1, 1, 64, 64  # Single tile

# Test 1: all ones → each S[i,j] should be 256 * 1 * 1 = 256
print("Test 1: Q=all 1s, K=all 1s")
Q = torch.ones(B, H, Lq, D, dtype=torch.int8, device='cuda')
K = torch.ones(B, H, Lkv, D, dtype=torch.int8, device='cuda')
V = torch.ones(B, H, Lkv, D, dtype=torch.float16, device='cuda')

scale = 1.0 / D ** 0.5
O = mixed_flash_attn.int8_flash_attn(Q, K, V, 1.0, 1.0, scale, False)
torch.cuda.synchronize()

# Softmax of uniform values → each row should sum to 1
row_sums = O[0, 0].float().sum(dim=1)
print(f"  Row sums (should be ~1): min={row_sums.min():.4f} max={row_sums.max():.4f}")
print(f"  O values range: [{O.min():.4f}, {O.max():.4f}]")

# Test 2: Q=identity-like, K=identity-like
print("\nTest 2: Q and K with different values")
torch.manual_seed(42)
Q2 = torch.randint(-8, 7, (B, H, Lq, D), dtype=torch.int8, device='cuda')
K2 = torch.randint(-8, 7, (B, H, Lkv, D), dtype=torch.int8, device='cuda')
V2 = torch.ones(B, H, Lkv, D, dtype=torch.float16, device='cuda')

O2 = mixed_flash_attn.int8_flash_attn(Q2, K2, V2, 0.1, 0.1, scale, False)
torch.cuda.synchronize()

print(f"  O shape: {O2.shape}")
print(f"  O values range: [{O2.min():.4f}, {O2.max():.4f}]")

# Reference: manual scalar computation for comparison
Q2f = Q2.float() * 0.1
K2f = K2.float() * 0.1
S_ref = torch.matmul(Q2f[0,0], K2f[0,0].T) * scale
P_ref = torch.softmax(S_ref, dim=-1)
O_ref = torch.matmul(P_ref, V2[0,0].float())
print(f"  Reference O range: [{O_ref.min():.4f}, {O_ref.max():.4f}]")
print(f"  Match: CosSim={torch.nn.functional.cosine_similarity(O2[0,0].float().flatten(), O_ref.flatten(), dim=0):.4f}")
