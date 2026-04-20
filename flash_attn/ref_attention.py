"""Reference implementations and ResQ baseline for mixed-precision attention.

Provides:
  1. fp32_ref_attention  — PyTorch FP32 ground truth
  2. resq_baseline_attention — ResQ's separate-kernel approach
  3. compute_metrics — accuracy metrics (cosine sim, RMSE, etc.)
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
        'cosine_sim': cos_sim,
        'snr_db': snr_db,
    }
