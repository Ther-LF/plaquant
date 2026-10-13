"""JIT loader + oracle validation for the SM120 MXFP8xMXFP8 block-scaled GEMM.

Builds mxfp8_gemm_ext.cu with torch.utils.cpp_extension.load (sm_120a flags),
generates random-but-valid MXFP8 inputs (FP8 E4M3 elements + ue8m0 power-of-2
block scales), and compares the kernel output against a PyTorch dequant-matmul
oracle that mirrors promix.quantize.quant_utils.fake_quantize_mxfp8.

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
            name="mxfp8_gemm_ext",
            sources=[os.path.join(HERE, "mxfp8_gemm_ext.cu")],
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


FP8_E4M3_MAX = 448.0
BLOCK = 32


def _e8m0_round_up(scale):
    """Round positive scale UP to nearest power-of-2 (E8M0). Matches quant_utils."""
    log2 = torch.log2(scale.clamp_min(torch.finfo(scale.dtype).tiny))
    exp_up = torch.ceil(log2)
    chosen = torch.pow(torch.tensor(2.0, device=scale.device, dtype=scale.dtype), exp_up)
    chosen = torch.where(scale > 0, chosen,
                         torch.tensor(2.0 ** -127, device=scale.device, dtype=scale.dtype))
    return chosen


def quantize_mxfp8(x):
    """x: (R, K) float. Returns (e4m3_bytes uint8 (R,K), sf_bytes uint8 (R,K/32),
    deq float32 (R,K)). The dequant mirrors fake_quantize_mxfp8 exactly and matches
    what the kernel decodes: e4m3_to_float(byte) * 2**(sf_byte - 127)."""
    R, K = x.shape
    assert K % BLOCK == 0
    xb = x.float().view(R, K // BLOCK, BLOCK)
    block_max = xb.abs().amax(dim=-1, keepdim=True).clamp_min(0.0)
    ideal = block_max / FP8_E4M3_MAX
    block_scale = _e8m0_round_up(ideal.clamp_min(2.0 ** -127))     # (R, K/32, 1) power-of-2

    # Element quant: divide by scale, clamp, round to nearest e4m3 via torch dtype.
    xs = (xb / block_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q_f8 = xs.to(torch.float8_e4m3fn)
    q_val = q_f8.to(torch.float32)                                # decoded element
    deq = (q_val * block_scale).view(R, K)

    e4m3_bytes = q_f8.view(R, K).view(torch.uint8).contiguous()

    # ue8m0 exponent byte: scale == 2**exp -> byte = exp + 127
    exp = torch.log2(block_scale).round().to(torch.int32)
    sf_bytes = (exp + 127).clamp(0, 254).to(torch.uint8).view(R, K // BLOCK).contiguous()
    return e4m3_bytes, sf_bytes, deq


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run_shape(M, N, K, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    A = torch.randn(M, K, device=dev)
    B = torch.randn(N, K, device=dev)

    A_bytes, SFA, A_deq = quantize_mxfp8(A)
    B_bytes, SFB, B_deq = quantize_mxfp8(B)

    ext = get_ext()
    D = ext.mxfp8_gemm(A_bytes, SFA, B_bytes, SFB, M, N, K).float()

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
    for (M, N, K) in [(1024, 1024, 1024), (2048, 2048, 256), (512, 2048, 384)]:
        results[(M, N, K)] = run_shape(M, N, K)
    print("\nSummary:")
    ok = True
    for k, v in results.items():
        status = "PASS" if v >= 0.999 else "FAIL"
        if v < 0.999:
            ok = False
        print(f"  {k}: cosine={v:.6f} [{status}]")
    print("\nALL PASS" if ok else "\nSOME FAILED")
