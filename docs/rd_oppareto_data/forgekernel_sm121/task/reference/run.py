"""Reference: the shipping promix dual-phase (NVFP4+MXFP8) fused GEMM on the GPU host.

Calls the installed promix tree's compiled extension (single kernel launch,
bias+cast folded into the epilogue). The row-major scale-factor inputs are
rearranged into the kernel's physical SF layout with pure torch view/permute
ops (verified bit-identical to the extension's own prepack kernels), so every
call is sync-free and CUDA-graph capturable; the steady state is the one GEMM
launch plus four small async rearranges.
"""
import os
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
_REPO = "/home/luofan/plaquant"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
    sys.path.insert(0, _REPO + "/kernels/mixed_gemm_sm120")

import torch  # noqa: E402

_mods = {}


def _load():
    if not _mods:
        from promix.eval.full_network_realkernel import load_kernels
        from promix.vllm_integ.kernels import load_cd_bias
        nv, mx, cd, qpf = load_kernels()
        bias_ext = load_cd_bias()
        assert bias_ext is not None, "codesign_fused_bias_ext failed to load"
        _mods.update(bias=bias_ext)
    return _mods


def _u8(t):
    return t.view(torch.uint8) if t.dtype == torch.int8 else t


def _blocked(sf):
    """Row-major [R, nsf] uint8 -> the kernel's physical SF layout (128x4 atom,
    32_4_4 swizzle), zero-padded to 128-row / 4-col multiples. Pure async torch
    ops; bit-identical to the extension's prepack_sf_* kernels."""
    rows, cols = sf.shape
    nrb, ncb = (rows + 127) // 128, (cols + 3) // 4
    pr, pc = nrb * 128, ncb * 4
    if (rows, cols) != (pr, pc):
        p = torch.zeros(pr, pc, dtype=sf.dtype, device=sf.device)
        p[:rows, :cols] = sf
        sf = p
    return (sf.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
              .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous())


def run(a_lo, sfa_lo, a_hi, sfa_hi, b_lo, sfb_lo, b_hi, sfb_hi, bias):
    m = _load()
    a_lo, sfa_lo, a_hi, sfa_hi = map(_u8, (a_lo, sfa_lo, a_hi, sfa_hi))
    b_lo, sfb_lo, b_hi, sfb_hi = map(_u8, (b_lo, sfb_lo, b_hi, sfb_hi))
    M = a_lo.shape[0]
    K_lo = a_lo.shape[1] * 2
    N = b_lo.shape[0]
    K_hi = a_hi.shape[1]
    return m["bias"].fused_gemm_prepacked_bias(
        a_lo.contiguous(), _blocked(sfa_lo), b_lo.contiguous(), _blocked(sfb_lo),
        a_hi.contiguous(), _blocked(sfa_hi), b_hi.contiguous(), _blocked(sfb_hi),
        bias.to(torch.float16), M, N, K_lo, K_hi, 1.0)
