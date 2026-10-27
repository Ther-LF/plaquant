"""Round-10 Task-2 lever: Triton fused RMSNorm (+ fused residual-add RMSNorm).

The round-10 per-op profile (r10_perop_vs_vllm) shows our HF `LlamaRMSNorm` is the
#1 non-GEMM prefill gap vs vLLM's fused `rms_norm` CUDA kernel: HF upcasts to fp32
and dispatches ~6 memory-bound kernels (pow, mean, add, rsqrt, mul, cast, mul),
while vLLM does it in a single fp16-I/O fp32-accumulate pass. This module is a
drop-in replacement computing the IDENTICAL math (fp32 mean-of-squares, rsqrt,
weight multiply, fp16 output) in one Triton kernel.

`TritonRMSNorm` matches `transformers...LlamaRMSNorm` (attrs `weight`,
`variance_epsilon`; `forward(x)->y`). `swap_rmsnorm(model)` walks the model and
replaces every LlamaRMSNorm in place, preserving weight tensor + eps.

CUDA-graph-capturable (Triton kernels replay fine). No quantization math touched.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _add_rmsnorm_kernel(X, RES, W, Y, HOUT, stride, N, eps, BLOCK: tl.constexpr):
    """h = x + res; y = rmsnorm(h) * w; hout = h (the new residual stream)."""
    row = tl.program_id(0)
    X += row * stride
    RES += row * stride
    Y += row * stride
    HOUT += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(HOUT + cols, h.to(HOUT.dtype.element_ty), mask=mask)
    var = tl.sum(h * h, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * rstd * w
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _block_for(n):
    return triton.next_power_of_2(n)


def triton_rmsnorm(x, weight, eps):
    in_shape = x.shape
    N = in_shape[-1]
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    x2d = x2d.contiguous()
    y = torch.empty_like(x2d)
    BLOCK = _block_for(N)
    _rmsnorm_kernel[(M,)](x2d, weight, y, x2d.stride(0), N, eps, BLOCK=BLOCK,
                          num_warps=8 if BLOCK >= 4096 else 4)
    return y.reshape(in_shape)


def triton_add_rmsnorm(x, residual, weight, eps):
    """Returns (normed, new_residual=x+residual)."""
    in_shape = x.shape
    N = in_shape[-1]
    x2d = x.reshape(-1, N).contiguous()
    r2d = residual.reshape(-1, N).contiguous()
    M = x2d.shape[0]
    y = torch.empty_like(x2d)
    h = torch.empty_like(x2d)
    BLOCK = _block_for(N)
    _add_rmsnorm_kernel[(M,)](x2d, r2d, weight, y, h, x2d.stride(0), N, eps,
                              BLOCK=BLOCK, num_warps=8 if BLOCK >= 4096 else 4)
    return y.reshape(in_shape), h.reshape(in_shape)


class TritonRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight          # share the SAME Parameter (no copy)
        self.variance_epsilon = eps

    def forward(self, x):
        return triton_rmsnorm(x, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def swap_rmsnorm(model):
    """Replace every LlamaRMSNorm with TritonRMSNorm (shared weight). Returns the
    count swapped. Idempotent-safe (skips already-swapped)."""
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
    except Exception:
        LlamaRMSNorm = None
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            is_rms = (LlamaRMSNorm is not None and isinstance(child, LlamaRMSNorm)) or (
                child.__class__.__name__ == "LlamaRMSNorm")
            if is_rms:
                eps = getattr(child, "variance_epsilon", 1e-5)
                setattr(parent, name, TritonRMSNorm(child.weight, eps))
                n += 1
    return n
