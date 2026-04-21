"""Debug WGMMA fragment extraction: identify wrong (row,col) positions.

Strategy: Q[i,k] = i+1 (row index), K[j,k] = 1 (all ones)
→ S[i,j] = D * (i+1) = 256*(i+1) uniform per row

If extraction is correct, each row in S_out should be uniform.
If any row has non-uniform values, those columns are at wrong positions.
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mixed_flash_attn

D = 256
B, H = 1, 1

# Q: each row has a distinct value (i+1 for row i)
Q = torch.zeros(B, H, 64, D, dtype=torch.int8, device='cuda')
for i in range(64):
    Q[0, 0, i, :] = i + 1  # row i has value i+1

# K: all ones
K = torch.ones(B, H, 64, D, dtype=torch.int8, device='cuda')

# Run debug extract
S_out = mixed_flash_attn.debug_wgmma_extract(Q, K)
torch.cuda.synchronize()
S = S_out[0, 0].cpu().int()  # (64, 64)

# Expected: S[i,j] = D * (i+1) = 256*(i+1)
expected = torch.zeros(64, 64, dtype=torch.int32)
for i in range(64):
    expected[i, :] = 256 * (i + 1)

print("=== S matrix analysis ===")
print(f"S shape: {S.shape}")
print(f"S min={S.min().item()}, max={S.max().item()}")

# Check which rows are non-uniform
print("\nRows with non-uniform values:")
bad_rows = 0
for i in range(64):
    row = S[i, :]
    if row.min() != row.max():
        bad_rows += 1
        unique_vals = row.unique()
        print(f"  Row {i}: {len(unique_vals)} unique values: {unique_vals.tolist()[:10]}...")
        # Show which columns are wrong
        expected_val = expected[i, 0].item()
        wrong_cols = (row != expected_val).nonzero(as_tuple=True)[0]
        print(f"    Expected={expected_val}, wrong cols ({len(wrong_cols)}): {wrong_cols.tolist()[:20]}...")

print(f"\nTotal bad rows: {bad_rows}/64")

# Check which columns are non-uniform (shouldn't happen if only row-swaps)
print("\nColumns with non-uniform values:")
bad_cols = 0
for j in range(64):
    col = S[:, j]
    if col.min() != col.max():
        bad_cols += 1
print(f"Total bad cols: {bad_cols}/64")

# Check if values are just permuted (all expected values exist somewhere)
expected_flat = set(expected.flatten().tolist())
actual_flat = set(S.flatten().tolist())
print(f"\nExpected unique values: {len(expected_flat)}")
print(f"Actual unique values: {len(actual_flat)}")
print(f"Missing expected values: {len(expected_flat - actual_flat)}")
print(f"Unexpected values: {len(actual_flat - expected_flat)}")

# Detailed: compare S with expected element-by-element
errors = (S != expected)
num_errors = errors.sum().item()
print(f"\nTotal element errors: {num_errors}/{64*64} ({100*num_errors/4096:.1f}%)")

# Show error positions
if num_errors > 0 and num_errors < 50:
    print("\nError positions (row, col): expected → actual:")
    err_pos = errors.nonzero(as_tuple=False)
    for pos in err_pos[:30]:
        r, c = pos[0].item(), pos[1].item()
        print(f"  ({r},{c}): expected={expected[r,c].item()}, actual={S[r,c].item()}")

# Try to figure out the permutation pattern
# For each row, check which row's expected value appears where
print("\n=== Row permutation analysis ===")
for i in range(min(8, 64)):
    row = S[i, :]
    # For each column, which row's expected value is there?
    col_sources = {}
    for j in range(64):
        val = row[j].item()
        # val / 256 - 1 = source row index
        src_row = val // 256 - 1
        if src_row not in col_sources:
            col_sources[src_row] = []
        col_sources[src_row].append(j)
    if len(col_sources) > 1:
        print(f"  Row {i} (expected all={256*(i+1)}):")
        for src, cols in sorted(col_sources.items()):
            print(f"    Source row {src} (val={256*(src+1)}): cols {cols[:10]}...")
