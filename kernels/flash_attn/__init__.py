"""Mixed-Precision FlashAttention for Hopper (SM90).

Fused attention kernel with INT8/INT4 Tensor Cores for Q*K^T,
FP16 for P*V, and optional grouped o_proj GEMM.
"""

__all__ = ["bench_flash_attn", "ref_attention"]
