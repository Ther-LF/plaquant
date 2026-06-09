"""Real quantized forward — replaces fake quant with INT GEMM.

Uses INT4/INT8 tensor core GEMM (via CUTLASS) for actual acceleration.
Dequant formula: output = s_x * s_w * (q_x_shifted @ q_w^T + bias)
"""

import torch

from promix.quantize.hadamard import matmul_hadU_cuda
from promix.inference.quant_ops import (
    quantize_activation_per_token,
    shift_to_signed,
    pack_int4,
)


# Global reference to compiled kernel module (set by init_kernel())
_kernel_module = None


def init_kernel(kernel_path=None):
    """Load the compiled CUTLASS mixed GEMM kernel.

    Args:
        kernel_path: Path to compiled .so file. If None, tries to import.
    """
    global _kernel_module
    if kernel_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mixed_gemm_l20", kernel_path)
        _kernel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_kernel_module)
    else:
        try:
            import mixed_gemm_l20 as m
            _kernel_module = m
        except ImportError:
            raise RuntimeError(
                "mixed_gemm_l20 kernel not found. Compile it first:\n"
                "  cd kernels/mixed_gemm_l20 && python setup.py build_ext --inplace"
            )


def real_forward(wrapper, x):
    """Real quantized forward using INT tensor core GEMM.

    Replaces the fake quant path (quantize→dequant→FP16 GEMM) with:
    quantize→INT GEMM→dequant

    Args:
        wrapper: ActQuantWrapper with packed weights
        x: (batch, seq, K) FP16 input

    Returns:
        output: (batch, seq, N) FP16
    """
    assert getattr(wrapper, '_real_inference_ready', False), \
        "Call pack_model_weights() first"

    x_dtype = x.dtype
    init_shape = x.shape
    N = wrapper.module.weight.shape[0]

    # 1. Online Hadamard (down_proj only)
    if wrapper.online_full_had:
        if wrapper.fp32_had:
            x = matmul_hadU_cuda(x.float(), wrapper.had_K, wrapper.K).to(x_dtype)
        else:
            x = matmul_hadU_cuda(x, wrapper.had_K, wrapper.K)

    # Column order (o_proj)
    order = getattr(wrapper, '_column_order', None)
    if order is not None:
        x = x[..., order]

    # Flatten to 2D for GEMM
    x_2d = x.reshape(-1, x.shape[-1])  # (M, K)
    M = x_2d.shape[0]
    K_main = wrapper._K_main
    K_high = wrapper._K_high

    # 2. Split and quantize activation
    x_main = x_2d[..., :K_main]
    q_main, s_x_main, z_x_main = quantize_activation_per_token(x_main, bits=4, asym=True)

    if K_high > 0:
        x_high = x_2d[..., K_main:]
        q_high, s_x_high, z_x_high = quantize_activation_per_token(x_high, bits=8, asym=True)

    # 3. Shift to signed
    q_main_signed = shift_to_signed(q_main, 4)  # [-8, 7]
    q_main_packed = pack_int4(q_main_signed)     # (M, K_main//2)

    if K_high > 0:
        q_high_signed = shift_to_signed(q_high, 8)  # [-128, 127]

    # 4. INT GEMM
    # Use the fused kernel (it sums main+high internally)
    # But we need separate dequant... so use Solution A: two calls
    if _kernel_module is not None:
        raw_main = _kernel_module.fused_mixed_gemm(
            q_main_packed,           # (M, K_main//2) int8 packed INT4
            wrapper.W_main_packed,   # (N, K_main//2) int8 packed INT4
            torch.zeros(M, 0, dtype=torch.int8, device=x.device),  # empty high A
            torch.zeros(N, 0, dtype=torch.int8, device=x.device),  # empty high B
        )  # Only runs INT4 path, K_high=0

        if K_high > 0:
            raw_high = _kernel_module.fused_mixed_gemm(
                torch.zeros(M, 0, dtype=torch.int8, device=x.device),
                torch.zeros(N, 0, dtype=torch.int8, device=x.device),
                q_high_signed,           # (M, K_high) int8
                wrapper.W_high_int8,     # (N, K_high) int8
            )  # Only runs INT8 path, K_low=0
    else:
        # Fallback: PyTorch emulation (for testing without kernel)
        raw_main = (q_main_signed.float() @ wrapper.W_main_packed_float.T).half()
        if K_high > 0:
            raw_high = (q_high_signed.float() @ wrapper.W_high_int8.float().T).half()

    # 5. Dequant
    # Formula: output = s_x * s_w * (raw + bias)
    # bias = (shift - zero) * colsum(W)
    shift_main = 8.0
    s_w_main = wrapper.s_w_main.flatten().unsqueeze(0)  # (1, N)
    bias_main = (shift_main - z_x_main) * wrapper.colsum_main  # (M, 1) * (1, N) → (M, N)
    output = s_x_main * s_w_main * (raw_main.float() + bias_main)

    if K_high > 0:
        shift_high = 128.0
        s_w_high = wrapper.s_w_high.flatten().unsqueeze(0)
        bias_high = (shift_high - z_x_high) * wrapper.colsum_high
        output = output + s_x_high * s_w_high * (raw_high.float() + bias_high)

    return output.half().reshape(*init_shape[:-1], N)


def install_real_forward(model):
    """Replace all ActQuantWrapper forward methods with real_forward.

    Call after pack_model_weights().
    """
    from promix.quantize.quant_utils import ActQuantWrapper

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, ActQuantWrapper) and getattr(module, '_real_inference_ready', False):
            # Store original forward for fallback
            module._fake_forward = module.forward
            module.forward = lambda x, _m=module, **kwargs: real_forward(_m, x)
            count += 1

    print(f"Installed real forward on {count} layers")
