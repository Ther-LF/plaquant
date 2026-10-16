"""TMA plumbing probe runner. Validates the SM120 TMA + swizzle round-trip.
Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python tma_probe.py
"""
import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

ext = load(
    name="tma_probe_ext",
    sources=[os.path.join(HERE, "tma_probe_ext.cu")],
    extra_include_paths=[
        os.path.join(CUTLASS, "include"),
        os.path.join(CUTLASS, "tools/util/include"),
        os.path.join(CUTLASS, "examples/common"),
    ],
    extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                       "--expt-extended-lambda", "-O3", "-Xptxas", "-v"],
    verbose=True,
)

BN, KCH = 128, 64
N, K = 2048, 8192
n0, k0 = 256, 128   # arbitrary aligned tile
torch.manual_seed(0)
B = torch.randint(0, 256, (N, K), dtype=torch.uint8, device="cuda")
D = ext.tma_probe(B, N, K, n0, k0)
ref = B[n0:n0+BN, k0:k0+KCH].contiguous()
match = torch.equal(D.cpu(), ref.cpu())
nmismatch = (D.cpu() != ref.cpu()).sum().item()
print(f"TMA probe: match={match} mismatch={nmismatch}/{BN*KCH}")
print("TMA PROBE PASS" if match else "TMA PROBE FAIL")
