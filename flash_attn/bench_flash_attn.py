#!/usr/bin/env python3
"""Benchmark for Mixed-Precision FlashAttention.

Compares:
  1. FP32 reference          — accuracy ground truth
  2. ResQ baseline           — separate GEMM kernels (current approach)
  3. Mixed FlashAttention    — our fused kernel (when available)

Usage:
    python flash_attn/bench_flash_attn.py
    python flash_attn/bench_flash_attn.py --seq-lens 1024,2048,4096
    python flash_attn/bench_flash_attn.py --k-high 128 --k-low 128
    python flash_attn/bench_flash_attn.py --accuracy-only
    python flash_attn/bench_flash_attn.py --output results.json
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

# Allow running from plaquant root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.ref_attention import (
    fp32_ref_attention,
    resq_baseline_attention,
    compute_metrics,
)


# ============================================================
# Data Generation
# ============================================================

def generate_test_data(B, H, Lq, Lkv, d_head, k_high, k_low,
                       device='cuda', seed=42):
    """Generate mixed-precision Q, K, V tensors for attention benchmarking.

    Returns dict with all tensors on GPU.
    """
    torch.manual_seed(seed)

    # FP32 reference data (on CPU first, then move)
    q_fp32 = torch.randn(B, H, Lq, d_head, device='cpu')
    k_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')
    v_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')

    def quantize_to_int8(x):
        """Per-tensor symmetric INT8 quantization."""
        amax = x.abs().max()
        scale = amax / 127.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-128, 127).to(torch.int8)
        return q, scale

    def quantize_to_int4_as_int8(x):
        """Per-tensor symmetric INT4 quantization, stored as INT8 values."""
        amax = x.abs().max()
        scale = amax / 7.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-8, 7).to(torch.int8)
        return q, scale

    q_int8, scale_q8 = quantize_to_int8(q_fp32[..., :k_high])
    q_int4, scale_q4 = quantize_to_int4_as_int8(q_fp32[..., k_high:])
    k_int8, scale_k8 = quantize_to_int8(k_fp32[..., :k_high])
    k_int4, scale_k4 = quantize_to_int4_as_int8(k_fp32[..., k_high:])

    return {
        'q_fp32': q_fp32.to(device),
        'k_fp32': k_fp32.to(device),
        'v_fp32': v_fp32.to(device),
        'q_int8':  q_int8.to(device),
        'q_int4':  q_int4.to(device),
        'k_int8':  k_int8.to(device),
        'k_int4':  k_int4.to(device),
        'v_fp16':  v_fp32.half().to(device),
        'scale_q8': scale_q8,
        'scale_k8': scale_k8,
        'scale_q4': scale_q4,
        'scale_k4': scale_k4,
    }


# ============================================================
# Timing (FA3-style: CUDA events)
# ============================================================

def bench_function(fn, warmup=5, repeat=100):
    """Time a no-arg function using CUDA events.

    Args:
        fn: callable with no arguments
        warmup: number of warmup iterations
        repeat: number of timed iterations

    Returns:
        avg_latency_ms: float
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def compute_flops(B, H, Lq, Lkv, d_head, causal):
    """Compute theoretical FLOPs for attention forward.

    FLOPs = 2 * B * H * Lq * avg_kv_len * d_head
    For causal: avg_kv_len ≈ Lkv / 2
    """
    avg_kv = Lkv / 2 if causal else Lkv
    return 2 * B * H * Lq * avg_kv * d_head


# ============================================================
# Benchmark Runners
# ============================================================

def make_fp32_ref_fn(data, causal, scale):
    q, k, v = data['q_fp32'], data['k_fp32'], data['v_fp32']
    def run():
        fp32_ref_attention(q, k, v, scale=scale, causal=causal)
    return run


def make_resq_fn(data, causal, scale):
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
# Config Sweep
# ============================================================

def build_configs(seq_lens):
    """Build list of (name, cfg_dict) for each test scenario."""
    configs = []
    for L in seq_lens:
        configs.append((f"prefill_L{L}", {
            'B': 1, 'H': 32, 'Lq': L, 'Lkv': L, 'causal': False,
        }))
    for L in seq_lens:
        configs.append((f"decode_KV{L}", {
            'B': 1, 'H': 1, 'Lq': 1, 'Lkv': L, 'causal': True,
        }))
    return configs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Mixed-Precision FlashAttention')
    parser.add_argument('--seq-lens', type=str, default='1024,2048,4096',
                        help='Comma-separated sequence lengths')
    parser.add_argument('--k-high', type=int, default=128)
    parser.add_argument('--k-low', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON')
    parser.add_argument('--accuracy-only', action='store_true',
                        help='Only check accuracy, skip perf')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    k_high = args.k_high
    k_low  = args.k_low
    d_head = k_high + k_low
    scale  = 1.0 / (d_head ** 0.5)
    device = torch.device('cuda')
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    configs  = build_configs(seq_lens)

    gpu_name = torch.cuda.get_device_name()
    print(f"\n{'='*130}")
    print(f"Mixed-Precision FlashAttention Benchmark")
    print(f"Config: k_high={k_high}, k_low={k_low}, d_head={d_head}")
    print(f"GPU:    {gpu_name}")
    print(f"{'='*130}")

    header = (f"{'Config':<18} {'Lq':>5} {'Lkv':>5} {'GFLOPs':>10} "
              f"{'Ref(ms)':>10} {'ResQ(ms)':>10} {'Speedup':>8} "
              f"{'CosSim':>8} {'MaxErr':>8} {'SNRdB':>7}")
    print(header)
    print('-' * 130)

    all_results = {}

    for name, cfg in configs:
        B, H, Lq, Lkv, causal = (
            cfg['B'], cfg['H'], cfg['Lq'], cfg['Lkv'], cfg['causal'])

        data = generate_test_data(
            B, H, Lq, Lkv, d_head, k_high, k_low,
            device=device, seed=args.seed)

        # ---- Accuracy ----
        ref_out = fp32_ref_attention(
            data['q_fp32'], data['k_fp32'], data['v_fp32'],
            scale=scale, causal=causal)

        resq_out = resq_baseline_attention(
            data['q_int8'], data['q_int4'],
            data['k_int8'], data['k_int4'],
            data['v_fp16'],
            data['scale_q8'], data['scale_k8'],
            data['scale_q4'], data['scale_k4'],
            scale=scale, causal=causal)

        metrics = compute_metrics(resq_out.float().cpu(),
                                  ref_out.float().cpu())

        # ---- Performance ----
        if args.accuracy_only:
            ref_ms, resq_ms, speedup = 0, 0, 0
        else:
            ref_fn  = make_fp32_ref_fn(data, causal, scale)
            ref_ms  = bench_function(ref_fn, args.warmup, args.repeat)

            resq_fn = make_resq_fn(data, causal, scale)
            resq_ms = bench_function(resq_fn, args.warmup, args.repeat)

            speedup = ref_ms / resq_ms if resq_ms > 0 else 0

        gflops = compute_flops(B, H, Lq, Lkv, d_head, causal) / 1e9

        print(f"{name:<18} {Lq:>5} {Lkv:>5} {gflops:>10.2f} "
              f"{ref_ms:>10.3f} {resq_ms:>10.3f} {speedup:>7.2f}x "
              f"{metrics['cosine_sim']:>8.4f} {metrics['max_abs_err']:>8.4f} "
              f"{metrics['snr_db']:>7.1f}")

        all_results[name] = {
            'config': {**cfg, 'd_head': d_head,
                       'k_high': k_high, 'k_low': k_low},
            'ref_latency_ms':   round(ref_ms, 4),
            'resq_latency_ms':  round(resq_ms, 4),
            'speedup_vs_ref':   round(speedup, 3),
            'accuracy': {k: round(v, 6) for k, v in metrics.items()},
            'gflops':           round(gflops, 2),
        }

    # Summary
    print(f"\n{'='*130}")
    print("Accuracy target: CosSim > 0.99, MaxErr < 1.0 (mixed-precision quantization noise)")
    avg_cos = sum(r['accuracy']['cosine_sim'] for r in all_results.values()) / max(len(all_results), 1)
    print(f"Average CosSim: {avg_cos:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
