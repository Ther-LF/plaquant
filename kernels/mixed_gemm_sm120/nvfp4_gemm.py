"""JIT loader + oracle validation for the SM120 NVFP4xNVFP4 block-scaled GEMM.

Builds nvfp4_gemm_ext.cu with torch.utils.cpp_extension.load (verified sm_120a
flags), generates random-but-valid NVFP4 inputs (FP4 E2M1 codes + ue4m3 block
scales), and compares kernel output against a PyTorch dequant-matmul oracle.

Run inside the luofan-plaquant container on a free GPU.
"""
import os
import torch
from torch.utils.cpp_extension import load

CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="nvfp4_gemm_ext",
            sources=[os.path.join(HERE, "nvfp4_gemm_ext.cu")],
            extra_include_paths=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
                os.path.join(CUTLASS, "examples/common"),
            ],
            extra_cuda_cflags=[
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-O3",
            ],
            verbose=True,
        )
    return _ext


# ---- FP4 E2M1 grid (abs) indexed by code&7 ----
_ABS_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX = 6.0
BLOCK = 16


def _build_signed_table(device):
    vals, codes = [], []
    for i, g in enumerate(_ABS_GRID):
        vals.append(g)
        codes.append(i)
    for i, g in enumerate(_ABS_GRID):
        if i == 0:
            continue  # skip -0.0 (duplicate of +0.0)
        vals.append(-g)
        codes.append(i | 0x8)
    return (torch.tensor(vals, device=device, dtype=torch.float32),
            torch.tensor(codes, device=device, dtype=torch.uint8))


def quantize_nvfp4(x):
    """x: (R, K) float. Returns (codes uint8 (R,K), sf_bytes uint8 (R,K/16),
    deq float32 (R,K)). deq uses the ue4m3-decoded scale, matching the kernel."""
    R, K = x.shape
    assert K % BLOCK == 0
    dev = x.device
    xb = x.float().view(R, K // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    ideal = (amax / FP4_MAX).clamp_min(2.0 ** -9)
    # encode scale to ue4m3 (== positive float8_e4m3fn byte); decode back for use
    sf_f8 = ideal.to(torch.float8_e4m3fn)
    scale = sf_f8.to(torch.float32)                       # (R, K/16, 1) decoded
    scale = scale.clamp_min(2.0 ** -9)
    xs = (xb / scale).clamp(-FP4_MAX, FP4_MAX)            # (R, K/16, 16)

    vals, codes_tab = _build_signed_table(dev)
    # nearest grid value
    d = (xs.unsqueeze(-1) - vals.view(1, 1, 1, -1)).abs()
    nn = d.argmin(dim=-1)                                 # (R, K/16, 16) idx into table
    qval = vals[nn]                                       # signed grid value
    codes = codes_tab[nn]                                 # (R, K/16, 16) uint8

    deq = (qval * scale).view(R, K)
    codes = codes.view(R, K).contiguous()
    sf_bytes = sf_f8.view(R, K // BLOCK).view(torch.uint8).contiguous()
    return codes, sf_bytes, deq


def pack_fp4(codes):
    """codes: (R,K) uint8 in 0..15 -> packed (R,K/2) uint8, even k low nibble."""
    R, K = codes.shape
    c = codes.view(R, K // 2, 2)
    packed = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    return packed.contiguous()


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run_shape(M, N, K, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    A = torch.randn(M, K, device=dev)
    B = torch.randn(N, K, device=dev)

    A_codes, SFA, A_deq = quantize_nvfp4(A)
    B_codes, SFB, B_deq = quantize_nvfp4(B)

    A_packed = pack_fp4(A_codes)
    B_packed = pack_fp4(B_codes)

    ext = get_ext()
    D = ext.nvfp4_gemm(A_packed, SFA, B_packed, SFB, M, N, K)
    D = D.float()

    D_ref = A_deq @ B_deq.t()
    cos = cosine(D, D_ref)
    rel = (D - D_ref).norm() / (D_ref.norm() + 1e-12)
    print(f"  shape M={M} N={N} K={K}: cosine={cos:.6f}  rel_err={rel.item():.4e}  "
          f"Dnorm={D.norm().item():.3e} refnorm={D_ref.norm().item():.3e}")
    return cos


if __name__ == "__main__":
    print("Loading extension (first build may take a few minutes)...")
    get_ext()
    print("Extension loaded.\n")
    results = {}
    for (M, N, K) in [(1024, 1024, 1024), (2048, 2048, 1792), (512, 2048, 2048)]:
        results[(M, N, K)] = run_shape(M, N, K)
    print("\nSummary:")
    ok = True
    for k, v in results.items():
        status = "PASS" if v >= 0.999 else "FAIL"
        if v < 0.999:
            ok = False
        print(f"  {k}: cosine={v:.6f} [{status}]")
    print("\nALL PASS" if ok else "\nSOME FAILED")
