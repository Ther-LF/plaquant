#!/usr/bin/env python3
"""Benchmark for Mixed-Precision FlashAttention.

Compares:
  1. FP32 reference          — accuracy ground truth
  2. FA FP16 (torch SDPA)    — performance upper bound
  3. INT8-only attention     — quantization benefit (no mixed precision)
  4. ResQ mixed precision    — INT8+INT4 (current approach)
  5. Mixed FlashAttention    — our fused kernel (when available)

Usage:
    python flash_attn/bench_flash_attn.py
    python flash_attn/bench_flash_attn.py --seq-lens 1024,2048,4096
    python flash_attn/bench_flash_attn.py --k-high 128 --k-low 128
    python flash_attn/bench_flash_attn.py --accuracy-only
    python flash_attn/bench_flash_attn.py --output results.json
"""

import argparse
import json
import os
import sys
import time

import torch

# Allow running from plaquant root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.ref_attention import (
    fp32_ref_attention,
    fa_fp16_attention,
    int8_only_attention,
    resq_baseline_attention,
    compute_metrics,
)


# ============================================================
# Data Generation
# ============================================================

def generate_test_data(B, H, Lq, Lkv, d_head, k_high, k_low,
                       device='cuda', seed=42):
    """Generate mixed-precision Q, K, V tensors for attention benchmarking."""
    torch.manual_seed(seed)

    # FP32 reference data
    q_fp32 = torch.randn(B, H, Lq, d_head, device='cpu')
    k_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')
    v_fp32 = torch.randn(B, H, Lkv, d_head, device='cpu')

    def quantize_to_int8(x):
        amax = x.abs().max()
        scale = amax / 127.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-128, 127).to(torch.int8)
        return q, scale

    def quantize_to_int4_as_int8(x):
        amax = x.abs().max()
        scale = amax / 7.0
        q = torch.round(x / scale.clamp(min=1e-8)).clamp(-8, 7).to(torch.int8)
        return q, scale

    # Mixed-precision: split Q/K along last dim
    q_int8_high, scale_q8 = quantize_to_int8(q_fp32[..., :k_high])
    q_int4,      scale_q4 = quantize_to_int4_as_int8(q_fp32[..., k_high:])
    k_int8_high, scale_k8 = quantize_to_int8(k_fp32[..., :k_high])
    k_int4,      scale_k4 = quantize_to_int4_as_int8(k_fp32[..., k_high:])

    # INT8-only: full d_head quantized
    q_int8_full, scale_q8_full = quantize_to_int8(q_fp32)
    k_int8_full, scale_k8_full = quantize_to_int8(k_fp32)

    return {
        # FP32 reference
        'q_fp32': q_fp32.to(device),
        'k_fp32': k_fp32.to(device),
        'v_fp32': v_fp32.to(device),
        # FP16 (for FA baseline)
        'q_fp16': q_fp32.half().to(device),
        'k_fp16': k_fp32.half().to(device),
        'v_fp16': v_fp32.half().to(device),
        # Mixed-precision (INT8 + INT4)
        'q_int8':  q_int8_high.to(device),
        'q_int4':  q_int4.to(device),
        'k_int8':  k_int8_high.to(device),
        'k_int4':  k_int4.to(device),
        'scale_q8': scale_q8,
        'scale_k8': scale_k8,
        'scale_q4': scale_q4,
        'scale_k4': scale_k4,
        # INT8-only (full d_head)
        'q_int8_full': q_int8_full.to(device),
        'k_int8_full': k_int8_full.to(device),
        'scale_q8_full': scale_q8_full,
        'scale_k8_full': scale_k8_full,
    }


# ============================================================
# Timing (FA3-style: CUDA events)
# ============================================================

def bench_function(fn, warmup=5, repeat=100):
    """Time a no-arg function using CUDA events.

    Returns: avg latency in milliseconds.
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

    FA3 formula: FLOPs = B * H * 2 * Lq * avg_kv_len * (headdim + headdim_v)
    = 4 * B * H * Lq * avg_kv_len * d_head  (when headdim_v = d_head)

    This counts both Q·K^T (2*Lq*Lkv*d) and P·V (2*Lq*Lkv*d).
    """
    avg_kv = Lkv / 2 if causal else Lkv
    return 4 * B * H * Lq * avg_kv * d_head


# ============================================================
# H20 Peak Throughput (Tensor Core)
# ============================================================

# NVIDIA H20 (Hopper SM90, 78 SMs, 400W TDP)
H20_PEAK = {
    'fp16':  148.0,   # TFLOPS (FP16 TC)
    'int8':  296.0,   # TOPS   (INT8 TC, 2x FP16)
    'int4':  592.0,   # TOPS   (INT4 TC, 2x INT8, 4x FP16) — estimated
}

# Effective peak for mixed precision (weighted by K ratio)
def mixed_peak(k_high, k_low):
    """Effective peak TOPS for mixed INT8/INT4."""
    total_k = k_high + k_low
    return (k_high * H20_PEAK['int8'] + k_low * H20_PEAK['int4']) / total_k

# Peak for each backend
def backend_peak(name, k_high=128, k_low=128):
    if name == 'fa_fp16':
        return H20_PEAK['fp16']
    elif name == 'int8_only':
        return H20_PEAK['int8']
    elif name == 'resq_mixed':
        return mixed_peak(k_high, k_low)
    elif name == 'our_int8':
        return H20_PEAK['int8']
    return 0


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
# Backend Definitions
# ============================================================

BACKENDS = ['fa_fp16', 'int8_only', 'resq_mixed', 'our_int8']


def run_backend(name, data, ref_out, scale, causal, accuracy_only,
                warmup, repeat):
    """Run accuracy + perf for one backend.

    Returns dict with keys: latency_ms, accuracy, error (if any).
    """
    result = {'latency_ms': 0, 'accuracy': None, 'error': None}

    try:
        if name == 'fa_fp16':
            out = fa_fp16_attention(
                data['q_fp16'], data['k_fp16'], data['v_fp16'],
                scale=scale, causal=causal)

            if not accuracy_only:
                def fn():
                    fa_fp16_attention(
                        data['q_fp16'], data['k_fp16'], data['v_fp16'],
                        scale=scale, causal=causal)
                result['latency_ms'] = bench_function(fn, warmup, repeat)

        elif name == 'int8_only':
            out = int8_only_attention(
                data['q_int8_full'], data['k_int8_full'], data['v_fp16'],
                data['scale_q8_full'], data['scale_k8_full'],
                scale=scale, causal=causal)

            if not accuracy_only:
                def fn():
                    int8_only_attention(
                        data['q_int8_full'], data['k_int8_full'],
                        data['v_fp16'],
                        data['scale_q8_full'], data['scale_k8_full'],
                        scale=scale, causal=causal)
                result['latency_ms'] = bench_function(fn, warmup, repeat)

        elif name == 'resq_mixed':
            out = resq_baseline_attention(
                data['q_int8'], data['q_int4'],
                data['k_int8'], data['k_int4'],
                data['v_fp16'],
                data['scale_q8'], data['scale_k8'],
                data['scale_q4'], data['scale_k4'],
                scale=scale, causal=causal)

            if not accuracy_only:
                def fn():
                    resq_baseline_attention(
                        data['q_int8'], data['q_int4'],
                        data['k_int8'], data['k_int4'],
                        data['v_fp16'],
                        data['scale_q8'], data['scale_k8'],
                        data['scale_q4'], data['scale_k4'],
                        scale=scale, causal=causal)
                result['latency_ms'] = bench_function(fn, warmup, repeat)

        elif name == 'our_int8':
            try:
                import mixed_flash_attn
            except ImportError:
                result['error'] = 'mixed_flash_attn not built'
                return result

            sq = data['scale_q8_full']
            sk = data['scale_k8_full']
            causal_flag = causal

            out = mixed_flash_attn.int8_flash_attn(
                data['q_int8_full'], data['k_int8_full'], data['v_fp16'],
                float(sq), float(sk), scale, causal_flag)

            if not accuracy_only:
                def fn():
                    mixed_flash_attn.int8_flash_attn(
                        data['q_int8_full'], data['k_int8_full'], data['v_fp16'],
                        float(sq), float(sk), scale, causal_flag)
                result['latency_ms'] = bench_function(fn, warmup, repeat)

        result['accuracy'] = compute_metrics(
            out.float().cpu(), ref_out.float().cpu())

    except Exception as e:
        import traceback
        result['error'] = str(e)
        traceback.print_exc()

    return result


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
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--accuracy-only', action='store_true')
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
    print(f"\n{'='*160}")
    print(f"Mixed-Precision FlashAttention Benchmark")
    print(f"Config: k_high={k_high}, k_low={k_low}, d_head={d_head}")
    print(f"GPU:    {gpu_name}")
    print(f"H20 Peak: FP16={H20_PEAK['fp16']} TF, INT8={H20_PEAK['int8']} TOPS, "
          f"INT4={H20_PEAK['int4']} TOPS, Mixed={mixed_peak(k_high, k_low):.0f} TOPS")
    print(f"{'='*160}")

    # Header — Latency
    print(f"\n{'Config':<18} {'Lq':>5} {'Lkv':>5} {'GFLOPs':>9} ", end='')
    for b in BACKENDS:
        print(f"{b+'(ms)':>10} ", end='')
    print(f"{'CosSim':>8} {'MaxErr':>8} {'MAE':>8} {'MAPE':>8}")
    print('-' * 160)

    # Header — Achieved TFLOPS/TOPS + Utilization
    print(f"{'Achieved (TF/TOPS)':<18} {'':>5} {'':>5} {'':>9} ", end='')
    for b in BACKENDS:
        print(f"{b+'(ops)':>10} ", end='')
    print()
    print(f"{'Utilization (%)':<18} {'':>5} {'':>5} {'':>9} ", end='')
    for b in BACKENDS:
        print(f"{b+'%peak':>10} ", end='')
    print()
    print('-' * 160)

    all_results = {}

    for name, cfg in configs:
        B, H, Lq, Lkv, causal = (
            cfg['B'], cfg['H'], cfg['Lq'], cfg['Lkv'], cfg['causal'])

        data = generate_test_data(
            B, H, Lq, Lkv, d_head, k_high, k_low,
            device=device, seed=args.seed)

        # Reference output
        ref_out = fp32_ref_attention(
            data['q_fp32'], data['k_fp32'], data['v_fp32'],
            scale=scale, causal=causal)

        gflops = compute_flops(B, H, Lq, Lkv, d_head, causal) / 1e9
        entry = {'config': {**cfg, 'd_head': d_head,
                            'k_high': k_high, 'k_low': k_low},
                 'gflops': round(gflops, 2)}

        # Run each backend
        latencies = {}
        accuracies = {}

        for backend in BACKENDS:
            r = run_backend(backend, data, ref_out, scale, causal,
                            args.accuracy_only, args.warmup, args.repeat)
            entry[backend] = {
                'latency_ms': round(r['latency_ms'], 4),
                'error': r['error'],
                'accuracy': r['accuracy'],
            }
            latencies[backend] = r['latency_ms']
            if r['accuracy']:
                accuracies[backend] = r['accuracy']

        # Latency row
        print(f"{name:<18} {Lq:>5} {Lkv:>5} {gflops:>9.2f} ", end='')
        for b in BACKENDS:
            l = entry[b]['latency_ms']
            print(f"{l:>10.3f} ", end='')
        resq_acc = entry['resq_mixed'].get('accuracy', {})
        cos_sim = resq_acc.get('cosine_sim', 0) if resq_acc else 0
        max_err = resq_acc.get('max_abs_err', 0) if resq_acc else 0
        mae = resq_acc.get('mae', 0) if resq_acc else 0
        mape = resq_acc.get('mape', 0) if resq_acc else 0
        print(f"{cos_sim:>8.4f} {max_err:>8.4f} {mae:>8.4f} {mape:>8.4f}")

        # Achieved TFLOPS/TOPS row
        total_ops = gflops * 1e9  # convert back to FLOPs
        print(f"{'  achieved ops':<18} {'':>5} {'':>5} {'':>9} ", end='')
        for b in BACKENDS:
            l = entry[b]['latency_ms']
            if l > 0:
                ops = total_ops / (l * 1e-3) / 1e12  # TFLOPS/TOPS
            else:
                ops = 0
            entry[b]['achieved_ops'] = round(ops, 2)
            print(f"{ops:>10.2f} ", end='')
        print()

        # Utilization % row
        print(f"{'  utilization %':<18} {'':>5} {'':>5} {'':>9} ", end='')
        for b in BACKENDS:
            peak = backend_peak(b, k_high, k_low)
            ops = entry[b].get('achieved_ops', 0)
            util = (ops / peak * 100) if peak > 0 else 0
            entry[b]['utilization_pct'] = round(util, 1)
            marker = '<<<' if util > 50 else ''
            print(f"{util:>9.1f}% ", end='')
        print()

        all_results[name] = entry

    # Summary
    print(f"\n{'='*150}")
    avg_cos = 0
    count = 0
    for r in all_results.values():
        resq = r.get('resq_mixed', {})
        acc = resq.get('accuracy', {}) if isinstance(resq, dict) else {}
        if acc:
            avg_cos += acc.get('cosine_sim', 0)
            count += 1
    if count > 0:
        print(f"Average CosSim (ResQ vs FP32): {avg_cos/count:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
