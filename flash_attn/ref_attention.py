"""Reference implementations and baselines for mixed-precision attention.

Provides:
  1. fp32_ref_attention        — PyTorch FP32 ground truth (accuracy)
  2. fa_fp16_attention          — torch SDPA (FA2/FA3 backend, perf upper bound)
  3. int8_only_attention        — INT8 Q·K^T + FP16 P·V (quantization benefit)
  4. resq_baseline_attention    — INT8+INT4 mixed precision (ResQ approach)
  5. compute_metrics            — accuracy metrics (cosine sim, RMSE, etc.)
"""

import torch
import torch.nn.functional as F


def fp32_ref_attention(q, k, v, scale=None, causal=False):
    """FP32 reference attention.

    Args:
        q: (B, H, Lq, d_head) FP32
        k: (B, H, Lkv, d_head) FP32
        v: (B, H, Lkv, d_head) FP32
        scale: softmax scale, defaults to 1/sqrt(d_head)
        causal: apply causal mask

    Returns:
        out: (B, H, Lq, d_head) FP32
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    B, H, Lq, D = q.shape
    Lkv = k.shape[2]

    q = q.reshape(B * H, Lq, D)
    k = k.reshape(B * H, Lkv, D)
    v = v.reshape(B * H, Lkv, D)

    s = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*H, Lq, Lkv)

    if causal:
        mask = torch.triu(
            torch.ones(Lq, Lkv, device=s.device, dtype=torch.bool),
            diagonal=Lkv - Lq + 1,
        )
        s.masked_fill_(mask, float('-inf'))

    p = F.softmax(s, dim=-1, dtype=torch.float32)
    out = torch.matmul(p, v)  # (B*H, Lq, D)

    return out.reshape(B, H, Lq, D)


def resq_baseline_attention(
    q_int8, q_int4, k_int8, k_int4, v_fp16,
    scale_q8, scale_k8, scale_q4, scale_k4,
    scale=None, causal=False,
):
    """ResQ baseline: separate GEMMs for each precision component.

    Simulates the current ResQ approach:
      GEMM 1: INT8  Q_int8 @ K_int8^T
      GEMM 2: INT4  Q_int4 @ K_int4^T
      dequant + add + softmax
      GEMM 3: FP16  P @ V

    Args:
        q_int8:  (B, H, Lq, k_high)  INT8
        q_int4:  (B, H, Lq, k_low)   INT8 (values 0-15, stored as INT8)
        k_int8:  (B, H, Lkv, k_high) INT8
        k_int4:  (B, H, Lkv, k_low)  INT8 (values 0-15)
        v_fp16:  (B, H, Lkv, d_head) FP16
        scale_*: FP32 scalars or broadcastable tensors
        scale:   softmax scale (optional)
        causal:  apply causal mask

    Returns:
        out: (B, H, Lq, d_head) FP16
    """
    B, H, Lq, k_high = q_int8.shape
    Lkv = k_int8.shape[2]
    k_low = q_int4.shape[-1]
    d_head = k_high + k_low

    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    # ---- Q*K^T: INT8 part ----
    q8 = q_int8.float().reshape(B * H, Lq, k_high)
    k8 = k_int8.float().reshape(B * H, Lkv, k_high)
    s_int8 = torch.bmm(q8, k8.transpose(1, 2))  # (B*H, Lq, Lkv)
    s_int8 = s_int8 * _to_tensor(scale_q8, q8) * _to_tensor(scale_k8, k8)

    # ---- Q*K^T: INT4 part ----
    q4 = q_int4.float().reshape(B * H, Lq, k_low)
    k4 = k_int4.float().reshape(B * H, Lkv, k_low)
    s_int4 = torch.bmm(q4, k4.transpose(1, 2))
    s_int4 = s_int4 * _to_tensor(scale_q4, q4) * _to_tensor(scale_k4, k4)

    # ---- Combine + scale + softmax ----
    s = (s_int8 + s_int4) * scale  # (B*H, Lq, Lkv)

    if causal:
        mask = torch.triu(
            torch.ones(Lq, Lkv, device=s.device, dtype=torch.bool),
            diagonal=Lkv - Lq + 1,
        )
        s.masked_fill_(mask, float('-inf'))

    p = F.softmax(s, dim=-1, dtype=torch.float32).half()

    # ---- P*V (FP16) ----
    v = v_fp16.reshape(B * H, Lkv, d_head).half()
    o = torch.bmm(p, v)  # (B*H, Lq, d_head) FP16

    return o.reshape(B, H, Lq, d_head)


def fa_fp16_attention(q_fp16, k_fp16, v_fp16, scale=None, causal=False):
    """FA FP16 baseline using torch SDPA (calls FA2/FA3 backend on H20).

    This is the performance UPPER BOUND for any quantized attention.

    Args:
        q_fp16: (B, H, Lq, d_head) FP16
        k_fp16: (B, H, Lkv, d_head) FP16
        v_fp16: (B, H, Lkv, d_head) FP16
        scale: softmax scale (optional)
        causal: apply causal mask

    Returns:
        out: (B, H, Lq, d_head) FP16
    """
    if scale is None:
        scale = 1.0 / (q_fp16.shape[-1] ** 0.5)

    return F.scaled_dot_product_attention(
        q_fp16.half(), k_fp16.half(), v_fp16.half(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def int8_only_attention(
    q_int8, k_int8, v_fp16,
    scale_q8, scale_k8,
    scale=None, causal=False,
):
    """INT8-only FlashAttention: Q·K^T in INT8, P·V in FP16.

    Simulates what an INT8-native FlashAttention kernel would compute.
    Used to isolate the overhead of INT4+INT8 mixed precision vs pure INT8.

    Args:
        q_int8:  (B, H, Lq, d_head) INT8 (full d_head, not split)
        k_int8:  (B, H, Lkv, d_head) INT8
        v_fp16:  (B, H, Lkv, d_head) FP16
        scale_q8: scalar FP32
        scale_k8: scalar FP32
        scale:   softmax scale (optional)
        causal:  apply causal mask

    Returns:
        out: (B, H, Lq, d_head) FP16
    """
    B, H, Lq, D = q_int8.shape
    Lkv = k_int8.shape[2]

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # ---- Q·K^T: INT8 ----
    q8 = q_int8.float().reshape(B * H, Lq, D)
    k8 = k_int8.float().reshape(B * H, Lkv, D)
    s = torch.bmm(q8, k8.transpose(1, 2))  # (B*H, Lq, Lkv)
    s = s * _to_tensor(scale_q8, q8) * _to_tensor(scale_k8, k8) * scale

    if causal:
        mask = torch.triu(
            torch.ones(Lq, Lkv, device=s.device, dtype=torch.bool),
            diagonal=Lkv - Lq + 1,
        )
        s.masked_fill_(mask, float('-inf'))

    p = F.softmax(s, dim=-1, dtype=torch.float32).half()

    # ---- P·V (FP16) ----
    v = v_fp16.reshape(B * H, Lkv, D).half()
    o = torch.bmm(p, v)

    return o.reshape(B, H, Lq, D)


def _to_tensor(s, ref):
    """Convert scale to a broadcastable tensor on the same device as ref."""
    if torch.is_tensor(s):
        return s.to(ref.device).float()
    return torch.tensor(s, device=ref.device, dtype=torch.float32)


def compute_metrics(actual, expected):
    """Compute accuracy metrics between actual and expected tensors.

    Args:
        actual: (..., D) tensor (any dtype)
        expected: (..., D) tensor (any dtype)

    Returns:
        dict: max_abs_err, mae, rmse, cosine_sim, snr_db
    """
    a = actual.float().flatten()
    e = expected.float().flatten()
    diff = a - e

    max_abs_err = diff.abs().max().item()
    mae = diff.abs().mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    eps = 1e-8
    mape = (diff.abs() / (e.abs() + eps)).mean().item()

    cos_sim = F.cosine_similarity(a.unsqueeze(0), e.unsqueeze(0)).item()

    ref_power = e.pow(2).sum().item()
    err_power = diff.pow(2).sum().item()
    if err_power > 0:
        snr_db = 10 * torch.log10(torch.tensor(ref_power / err_power)).item()
    else:
        snr_db = float('inf')

    return {
        'max_abs_err': max_abs_err,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'cosine_sim': cos_sim,
        'snr_db': snr_db,
    }
