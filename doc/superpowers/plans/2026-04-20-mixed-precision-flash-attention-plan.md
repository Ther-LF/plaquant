# Mixed-Precision FlashAttention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mixed-precision FlashAttention kernel (INT8+INT4 TC for Q·K^T, FP16 for P·V, fused grouped o_proj) on Hopper SM90 with FA3-style warp specialization and pipeline.

**Architecture:** CUTLASS 3.x Hopper kernel with TMA loads, INT4+INT8 WGMMA for Q·K^T (in-register accumulation), FP16 WGMMA for P·V, online softmax, circular SMEM buffer (s=2), two-warpgroup pingpong, intra-warpgroup 2-stage pipeline, and optional fused o_proj grouped GEMM.

**Tech Stack:** CUTLASS 3.x, PyTorch, CUDA C++17, Python 3.10, H20/H100 (SM90)

**Spec:** `doc/superpowers/specs/2026-04-20-mixed-precision-flash-attention-design.md`

---

## File Structure

```
flash_attn/
  __init__.py                    # Package init, exports
  bench_flash_attn.py            # Benchmark script (Phase 1)
  ref_attention.py               # FP32 reference + ResQ baseline
  mixed_flash_attn.cu            # Main kernel (Phase 2-3)
  mixed_flash_attn_binding.cpp   # PyTorch binding
  setup.py                       # Build script
```

---

## Phase 1: Benchmark Infrastructure + Reference

### Task 1: Create directory and package init

**Files:**
- Create: `flash_attn/__init__.py`

- [ ] **Step 1: Create init file**

```python
"""Mixed-Precision FlashAttention for Hopper (SM90).

Fused attention kernel with INT8/INT4 Tensor Cores for Q·K^T,
FP16 for P·V, and optional grouped o_proj GEMM.
"""

__all__ = ["bench_flash_attn", "ref_attention"]
```

- [ ] **Step 2: Commit**

```bash
git add flash_attn/__init__.py
git commit -m "feat: add flash_attn package init"
```

### Task 2: Write FP32 reference + ResQ baseline

**Files:**
- Create: `flash_attn/ref_attention.py`

This module provides:
1. `fp32_ref_attention(q, k, v, scale)` — PyTorch FP32 reference
2. `resq_baseline_attention(q_int8, q_int4, k_int8, k_int4, v_fp16, scales)` — ResQ's separate-kernel approach using existing `resq_gemm_v2`
3. `compute_metrics(actual, expected)` — accuracy metrics

- [ ] **Step 1: Write ref_attention.py**

```python
"""Reference implementations and ResQ baseline for mixed-precision attention."""

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
            diagonal=Lkv - Lq + 1
        )
        s.masked_fill_(mask, float('-inf'))
    
    p = F.softmax(s, dim=-1, dtype=torch.float32)
    out = torch.matmul(p, v)  # (B*H, Lq, D)
    
    return out.reshape(B, H, Lq, D)


def resq_baseline_attention(
    q_int8, q_int4, k_int8, k_int4, v_fp16,
    scale_q8, scale_k8, scale_q4, scale_k4,
    scale=None, causal=False,
    o_proj_weight_int8=None, o_proj_weight_int4=None,
    o_proj_scales=None,
):
    """ResQ baseline: separate GEMM kernels for each precision.
    
    This simulates the current ResQ approach where Q·K^T is done via
    two separate GEMM kernel launches (INT8 + INT4), results dequantized
    and added, then softmax, then FP16 P·V, then separate grouped o_proj.
    
    Args:
        q_int8:  (B, H, Lq, k_high) INT8
        q_int4:  (B, H, Lq, k_low)  INT8 (values 0-15, stored as INT8 for convenience)
        k_int8:  (B, H, Lkv, k_high) INT8
        k_int4:  (B, H, Lkv, k_low)  INT8 (values 0-15)
        v_fp16:  (B, H, Lkv, d_head) FP16
        scale_*: FP32 scalars or (B,H,1,1) tensors
    
    Returns:
        out: (B, H, Lq, d_head) FP16
    """
    if scale is None:
        d_head = q_int8.shape[-1] + q_int4.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
    
    B, H, Lq, k_high = q_int8.shape
    _, _, Lkv, k_low = q_int4.shape
    d_head = k_high + k_low
    
    # Reshape for GEMM: (B*H*Lq, K)
    q8 = q_int8.reshape(-1, k_high).cuda()
    q4 = q_int4.reshape(-1, k_low).cuda()
    k8 = k_int8.reshape(-1, k_high).cuda()
    k4 = k_int4.reshape(-1, k_low).cuda()
    
    M = B * H * Lq
    N = B * H * Lkv
    
    # Q·K^T INT8 part
    s_int8 = torch.matmul(q8.float(), k8.float().t())  # (M, N) FP32
    s_int8 = s_int8 * scale_q8 * scale_k8
    
    # Q·K^T INT4 part
    s_int4 = torch.matmul(q4.float(), k4.float().t())  # (M, N) FP32
    s_int4 = s_int4 * scale_q4 * scale_k4
    
    s = (s_int8 + s_int4) * scale  # (M, N) FP32
    s = s.reshape(B * H, Lq, Lkv)
    
    if causal:
        mask = torch.triu(
            torch.ones(Lq, Lkv, device=s.device, dtype=torch.bool),
            diagonal=Lkv - Lq + 1
        )
        s.masked_fill_(mask, float('-inf'))
    
    p = F.softmax(s, dim=-1, dtype=torch.float32).half()  # (B*H, Lq, Lkv) FP16
    
    # P·V (FP16)
    v_flat = v_fp16.reshape(-1, d_head)  # (B*H*Lkv, d_head)
    p_flat = p.reshape(M, N)  # (B*H*Lq, B*H*Lkv)
    
    # Need to handle multi-head: P is (B*H, Lq, Lkv), V is (B*H, Lkv, d_head)
    o = torch.zeros(B * H, Lq, d_head, device=v_fp16.device, dtype=torch.float32)
    for h in range(B * H):
        o[h] = p[h].half() @ v_fp16.reshape(B, H, Lkv, d_head)[0, h].half().float()
    
    out = o.half()  # (B*H, Lq, d_head) FP16
    
    # o_proj (grouped per head, optional)
    if o_proj_weight_int8 is not None and o_proj_weight_int4 is not None:
        k_oh = o_proj_weight_int8.shape[-1]
        k_ol = o_proj_weight_int4.shape[-1]
        
        o_int8 = out[..., :k_oh].reshape(-1, k_oh)
        o_int4 = out[..., k_oh:].reshape(-1, k_ol)
        
        # Grouped GEMM per head
        # For simplicity, treat as separate matmuls per head
        wo8 = o_proj_weight_int8.float()
        wo4 = o_proj_weight_int4.float()
        
        result8 = o_int8.float() @ wo8.t() * o_proj_scales[0]
        result4 = o_int4.float() @ wo4.t() * o_proj_scales[1]
        
        out = (result8 + result4).half()
    
    return out.reshape(B, H, Lq, -1)


def compute_metrics(actual, expected):
    """Compute accuracy metrics between actual and expected tensors.
    
    Args:
        actual: (..., D) tensor
        expected: (..., D) tensor
    
    Returns:
        dict with keys: max_abs_err, mae, rmse, cosine_sim, snr_db
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
    snr_db = 10 * torch.log10(torch.tensor(ref_power / max(err_power, 1e-12))).item()
    
    return {
        'max_abs_err': max_abs_err,
        'mae': mae,
        'rmse': rmse,
        'cosine_sim': cos_sim,
        'snr_db': snr_db,
    }
```

- [ ] **Step 2: Commit**

```bash
git add flash_attn/ref_attention.py
git commit -m "feat: add FP32 reference and ResQ baseline attention"
```

### Task 3: Write benchmark script

**Files:**
- Create: `flash_attn/bench_flash_attn.py`

Patterns from FA3 benchmark:
- Config sweep over (batch, seqlen, headdim, k_ratio)
- CUDA event timing with warmup
- TFLOPs computation
- Accuracy vs FP32 reference
- Formatted output table

- [ ] **Step 1: Write bench_flash_attn.py**

```python
#!/usr/bin/env python3
"""Benchmark for Mixed-Precision FlashAttention.

Compares:
  1. FP32 reference (accuracy baseline)
  2. ResQ baseline (separate GEMM kernels, current approach)
  3. Mixed FlashAttention (our fused kernel, when available)

Usage:
    python bench_flash_attn.py
    python bench_flash_attn.py --seq-lens 1024,2048,4096
    python bench_flash_attn.py --k-high 128 --k-low 128
"""

import argparse
import itertools
import json
import math
import os
import time

import torch

from ref_attention import fp32_ref_attention, resq_baseline_attention, compute_metrics


# ============================================================
# Test Configuration
# ============================================================

def get_default_configs():
    """Return list of (name, dict) test configs."""
    configs = []
    
    # Prefill configs
    for seq_len in [1024, 2048, 4096, 8192]:
        configs.append((
            f"prefill_l{seq_len}",
            {'B': 1, 'H': 32, 'Lq': seq_len, 'Lkv': seq_len, 'd_head': 128, 'causal': False}
        ))
    
    # Decode configs
    for kv_len in [1024, 4096, 8192]:
        configs.append((
            f"decode_kv{kv_len}",
            {'B': 1, 'H': 1, 'Lq': 1, 'Lkv': kv_len, 'd_head': 128, 'causal': True}
        ))
    
    return configs


# ============================================================
# Data Generation
# ============================================================

def generate_test_data(B, H, Lq, Lkv, d_head, k_high, k_low, device='cuda', dtype=torch.float16):
    """Generate mixed-precision Q, K, V tensors for attention.
    
    Returns dict with all tensors on GPU.
    """
    # FP32 reference data
    q_fp32 = torch.randn(B, H, Lq, d_head, device='cpu')
    k_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')
    v_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')
    
    # Quantize Q: split along last dim into INT8 (first k_high) and INT4 (remaining)
    def quantize_to_int8(x, axis=-1):
        amax = x.abs().amax(dim=axis, keepdim=True)
        scale = amax / 127.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-128, 127).to(torch.int8)
        return q, scale.squeeze(-1)
    
    def quantize_to_int4(x, axis=-1):
        amax = x.abs().amax(dim=axis, keepdim=True)
        scale = amax / 7.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-8, 7).to(torch.int8)
        return q, scale.squeeze(-1)
    
    q_int8,  scale_q8  = quantize_to_int8(q_fp32[..., :k_high])
    q_int4,  scale_q4  = quantize_to_int4(q_fp32[..., k_high:])
    k_int8,  scale_k8  = quantize_to_int8(k_fp32[..., :k_high])
    k_int4,  scale_k4  = quantize_to_int4(k_fp32[..., k_high:])
    
    return {
        # FP32 reference
        'q_fp32': q_fp32.to(device),
        'k_fp32': k_fp32.to(device),
        'v_fp32': v_fp32.to(device),
        # Mixed-precision
        'q_int8': q_int8.to(device),
        'q_int4': q_int4.to(device),
        'k_int8': k_int8.to(device),
        'k_int4': k_int4.to(device),
        'v_fp16': v_fp32.half().to(device),
        # Scales (scalar or per-head)
        'scale_q8': scale_q8.float().to(device),
        'scale_k8': scale_k8.float().to(device),
        'scale_q4': scale_q4.float().to(device),
        'scale_k4': scale_k4.float().to(device),
    }


# ============================================================
# Timing Utilities (FA3-style: CUDA events)
# ============================================================

def bench_function(fn, warmup=5, repeat=100):
    """Time a no-arg function using CUDA events.
    
    Returns: avg latency in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    
    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def compute_tflops(B, H, Lq, Lkv, d_head, causal, latency_ms):
    """Compute TFLOPs for attention forward pass.
    
    FLOPs = 2 * B * H * Lq * Lkv * d_head  (for Q·K^T and P·V)
    For causal: effective Lkv is halved on average.
    """
    if causal:
        # Average KV length under causal mask
        avg_kv = Lkv / 2
    else:
        avg_kv = Lkv
    
    flops = 2 * B * H * Lq * avg_kv * d_head
    tflops = flops / (latency_ms * 1e-3) / 1e12
    return tflops


# ============================================================
# Benchmarks
# ============================================================

def bench_fp32_ref(data, causal, scale):
    """Benchmark FP32 reference attention."""
    q = data['q_fp32']
    k = data['k_fp32']
    v = data['v_fp32']
    
    def run():
        fp32_ref_attention(q, k, v, scale=scale, causal=causal)
    
    return run


def bench_resq_baseline(data, causal, scale):
    """Benchmark ResQ baseline (separate GEMMs)."""
    def run():
        resq_baseline_attention(
            data['q_int8'], data['q_int4'],
            data['k_int8'], data['k_int4'],
            data['v_fp16'],
            data['scale_q8'], data['scale_k8'],
            data['scale_q4'], data['scale_k4'],
            scale=scale, causal=causal,
        )
    
    return run


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Mixed-Precision FlashAttention')
    parser.add_argument('--seq-lens', type=str, default='1024,2048,4096,8192',
                        help='Comma-separated sequence lengths')
    parser.add_argument('--k-high', type=int, default=128,
                        help='INT8 part K dimension')
    parser.add_argument('--k-low', type=int, default=128,
                        help='INT4 part K dimension')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--accuracy-only', action='store_true',
                        help='Only check accuracy, skip perf')
    args = parser.parse_args()
    
    k_high = args.k_high
    k_low = args.k_low
    d_head = k_high + k_low
    
    configs = []
    for seq_len in [int(x) for x in args.seq_lens.split(',')]:
        configs.append((f"prefill_l{seq_len}", 1, 32, seq_len, seq_len, False))
        configs.append((f"decode_kv{seq_len}", 1, 1, 1, seq_len, True))
    
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return
    
    device = torch.device('cuda')
    scale = 1.0 / (d_head ** 0.5)
    
    # Header
    print(f"\n{'='*120}")
    print(f"Mixed-Precision FlashAttention Benchmark")
    print(f"k_high={k_high}, k_low={k_low}, d_head={d_head}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*120}")
    print(f"{'Config':<22} {'M':>6} {'N':>6} {'FLOPs':>10}", end='')
    print(f" {'Ref(ms)':>10} {'ResQ(ms)':>10} {'CosSim':>8} {'MaxErr':>8}")
    print(f"{'-'*120}")
    
    all_results = {}
    
    for name, B, H, Lq, Lkv, causal in configs:
        # Generate data
        data = generate_test_data(B, H, Lq, Lkv, d_head, k_high, k_low, device=device)
        
        M = B * H * Lq
        N = B * H * Lkv
        
        # Reference output (accuracy baseline)
        ref_out = fp32_ref_attention(
            data['q_fp32'], data['k_fp32'], data['v_fp32'],
            scale=scale, causal=causal
        )
        
        # ResQ baseline output
        resq_out = resq_baseline_attention(
            data['q_int8'], data['q_int4'],
            data['k_int8'], data['k_int4'],
            data['v_fp16'],
            data['scale_q8'], data['scale_k8'],
            data['scale_q4'], data['scale_k4'],
            scale=scale, causal=causal,
        )
        
        # Accuracy
        resq_metrics = compute_metrics(resq_out.float(), ref_out.float())
        
        # Performance
        if not args.accuracy_only:
            ref_fn = bench_fp32_ref(data, causal, scale)
            ref_ms = bench_function(ref_fn, warmup=args.warmup, repeat=args.repeat)
            
            resq_fn = bench_resq_baseline(data, causal, scale)
            resq_ms = bench_function(resq_fn, warmup=args.warmup, repeat=args.repeat)
        else:
            ref_ms = 0
            resq_ms = 0
        
        total_flops = 2 * B * H * Lq * (Lkv if not causal else Lkv/2) * d_head
        gflops = total_flops / 1e9
        
        print(f"{name:<22} {M:>6} {N:>6} {gflops:>8.1f}G", end='')
        print(f" {ref_ms:>10.3f} {resq_ms:>10.3f}", end='')
        print(f" {resq_metrics['cosine_sim']:>8.4f} {resq_metrics['max_abs_err']:>8.4f}")
        
        all_results[name] = {
            'config': {'B': B, 'H': H, 'Lq': Lq, 'Lkv': Lkv, 'd_head': d_head,
                       'k_high': k_high, 'k_low': k_low, 'causal': causal},
            'ref_latency_ms': ref_ms,
            'resq_latency_ms': resq_ms,
            'resq_accuracy': resq_metrics,
            'gflops': gflops,
        }
    
    print(f"\nAccuracy check: ResQ baseline vs FP32 reference")
    print(f"  Cosine sim > 0.99:  acceptable for mixed precision")
    print(f"  Max abs err < 0.5:  acceptable (quantization noise)")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run benchmark to verify it works**

```bash
cd /data/home/spanaluo/plaquant
python flash_attn/bench_flash_attn.py --seq-lens 512 --accuracy-only
```

Expected: prints accuracy metrics for ResQ baseline vs FP32 reference. Cosine sim should be > 0.99.

- [ ] **Step 3: Run with perf measurement**

```bash
python flash_attn/bench_flash_attn.py --seq-lens 1024,2048 --warmup 3 --repeat 20
```

Expected: prints timing table with ms and GFLOPs.

- [ ] **Step 4: Commit**

```bash
git add flash_attn/bench_flash_attn.py
git commit -m "feat: add mixed-precision FlashAttention benchmark script"
```

---

## Phase 2: Core Kernel (Q·K^T Mixed Precision + FA Tiling)

### Task 4: Write CUTLASS 3.x kernel skeleton (TMA + WGMMA for Q·K^T)

**Files:**
- Create: `flash_attn/mixed_flash_attn.cu`

This phase implements the core computation without warp specialization or pipeline — just correct TMA loads, INT4+INT8 WGMMA for Q·K^T, online softmax, and FP16 P·V.

(Detailed implementation to be filled in Phase 2 after Phase 1 benchmark is verified)

### Task 5: Write PyTorch binding

**Files:**
- Create: `flash_attn/mixed_flash_attn_binding.cpp`
- Create: `flash_attn/setup.py`

(Detailed implementation to be filled in Phase 2)

---

## Phase 3: FA3 Pipeline (Warp Specialization + Pingpong + 2-Stage)

(Tasks to be detailed after Phase 2 is working)

## Phase 4: Fused o_proj

(Tasks to be detailed after Phase 3)

---

## Execution

**Phase 1 is ready for immediate implementation.** Tasks 1-3 produce:
- A working benchmark script
- FP32 reference for accuracy
- ResQ baseline for performance comparison
- All testable independently before any CUDA kernel work
