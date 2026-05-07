"""
ResQ Mixed-Precision GEMM Test Suite
=====================================
Phase 1: Collect all mixed-precision GEMM shapes from ResQ and build
correctness + performance benchmarks.

Target model: Llama-3.2-1B-Instruct
  hidden_size = 2048
  intermediate_size = 8192
  num_attention_heads = 32
  num_key_value_heads = 8
  head_dim = 64
  num_hidden_layers = 16

ResQ config: high_fraction=0.125, low_fraction=0.0
  high_bits=8, low_bits=2 (low not used since low_fraction=0)
  a_bits=4, k_bits=4, v_bits=4
"""

import torch
import torch.nn.functional as F
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import argparse


@dataclass
class MixedGemmSpec:
    """Specification of a single mixed-precision GEMM call in ResQ."""
    name: str           # e.g., "layers.0.self_attn.q_proj"
    M: int              # batch * seqlen (rows of activation)
    N: int              # output features
    K_total: int        # total input features
    K_high: int         # INT8 portion of K
    K_low: int          # INT4 portion of K (= K_total - K_high)
    high_bits: int      # 8
    low_bits: int       # 4
    groupsize: int      # -1 for per-token, >0 for per-group
    asym: bool          # True for asymmetric quantization
    layer_type: str     # "attn_input", "attn_output", "ffn_input", "kv_cache"


def get_resq_gemm_specs(
    batch_size: int = 1,
    seqlen: int = 1,
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    head_dim: int = 64,
    high_fraction: float = 0.125,
    num_layers: int = 16,
) -> List[MixedGemmSpec]:
    """
    Generate all mixed-precision GEMM specifications for a single forward pass.
    Only includes layers where high_bits_length > 0 (actual mixed precision).
    """
    M = batch_size * seqlen
    high_bits_length = int(high_fraction * hidden_size)  # 256

    # Per-group high bits for attention output (o_proj uses groupsize=head_dim)
    v_high_bits_length = int(high_fraction * head_dim)  # 8

    specs = []

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"

        # --- Attention input projections (q_proj, k_proj, v_proj) ---
        # Input quantizer: 4-bit asym, groupsize=-1, high_bits_length=256
        # GEMM: activation (M, 2048) × weight (2048, N)
        # The K dimension is split: 1792 INT4 + 256 INT8

        for proj_name, out_features in [
            ("q_proj", hidden_size),        # (M, 2048) × (2048, 2048)
            ("k_proj", num_key_value_heads * head_dim),  # (M, 2048) × (2048, 512)
            ("v_proj", num_key_value_heads * head_dim),  # (M, 2048) × (2048, 512)
        ]:
            specs.append(MixedGemmSpec(
                name=f"{prefix}.self_attn.{proj_name}",
                M=M,
                N=out_features,
                K_total=hidden_size,
                K_high=high_bits_length,
                K_low=hidden_size - high_bits_length,
                high_bits=8,
                low_bits=4,
                groupsize=-1,
                asym=True,
                layer_type="attn_input",
            ))

        # --- Attention output projection (o_proj) ---
        # Input quantizer: 4-bit asym, groupsize=head_dim=64, high_bits_length=8 (per group)
        # GEMM: activation (M, 2048) × weight (2048, 2048)
        # Per-group mixed precision: within each group of 64, 8 channels are INT8
        specs.append(MixedGemmSpec(
            name=f"{prefix}.self_attn.o_proj",
            M=M,
            N=hidden_size,
            K_total=hidden_size,
            K_high=v_high_bits_length,  # 8 per group of 64
            K_low=0,
            high_bits=8,
            low_bits=4,
            groupsize=head_dim,  # 64
            asym=True,
            layer_type="attn_output",
        ))

        # --- FFN input projections (gate_proj, up_proj) ---
        # Input quantizer: 4-bit asym, groupsize=-1, high_bits_length=256
        # GEMM: activation (M, 2048) × weight (2048, 8192)
        for proj_name in ["gate_proj", "up_proj"]:
            specs.append(MixedGemmSpec(
                name=f"{prefix}.mlp.{proj_name}",
                M=M,
                N=intermediate_size,
                K_total=hidden_size,
                K_high=high_bits_length,
                K_low=hidden_size - high_bits_length,
                high_bits=8,
                low_bits=4,
                groupsize=-1,
                asym=True,
                layer_type="ffn_input",
            ))

        # Note: down_proj has high_bits_length=0, so it's NOT mixed precision.
        # It uses uniform 4-bit quantization with Hadamard rotation.
        # We skip it here.

        # --- KV cache key quantization ---
        # k_bits=4, k_groupsize=64, high_bits_length=8 (=0.125*64)
        # This is not a GEMM but a quantize-dequantize operation on key states.
        # Included for completeness but may not need fused kernel.

    return specs


def get_unique_shapes(specs: List[MixedGemmSpec]) -> List[MixedGemmSpec]:
    """Deduplicate specs by (M, N, K_total, K_high, K_low, groupsize) to get unique GEMM shapes."""
    seen = set()
    unique = []
    for s in specs:
        key = (s.M, s.N, s.K_total, s.K_high, s.K_low, s.groupsize)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def reference_mixed_gemm(
    x: torch.Tensor,           # (M, K_total) quantized activation
    w: torch.Tensor,           # (N, K_total) quantized weight (transposed)
    K_high: int,               # number of INT8 channels
    scale_factor: float = 1.0, # optional scale for low-precision result
) -> torch.Tensor:
    """
    Reference implementation: two separate GEMMs + add.

    Simulates ResQ's computation:
      Output = x_low @ w_low.T + x_high @ w_high.T

    where x_low = x[:, :K_low], x_high = x[:, K_low:]
    (assuming PCA ordering: low-variance first, high-variance last)
    """
    K_total = x.shape[-1]
    K_low = K_total - K_high

    # Split along K dimension
    x_low = x[..., :K_low]    # INT4 portion
    x_high = x[..., K_low:]   # INT8 portion
    w_low = w[:, :K_low]
    w_high = w[:, K_low:]

    # Two separate GEMMs
    out_low = F.linear(x_low.float(), w_low.float())
    out_high = F.linear(x_high.float(), w_high.float())

    return (scale_factor * out_low + out_high).to(x.dtype)


def benchmark_two_gemms(
    x_low: torch.Tensor,
    w_low: torch.Tensor,
    x_high: torch.Tensor,
    w_high: torch.Tensor,
    warmup: int = 10,
    repeat: int = 100,
) -> float:
    """Benchmark the baseline: two separate GEMM calls + add."""
    # Warmup
    for _ in range(warmup):
        out = F.linear(x_low, w_low) + F.linear(x_high, w_high)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        out = F.linear(x_low, w_low) + F.linear(x_high, w_high)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat  # ms per iteration


def benchmark_mixed_gemm(
    mixed_gemm_fn,  # callable: (x, w, K_high) -> output
    x: torch.Tensor,
    w: torch.Tensor,
    K_high: int,
    warmup: int = 10,
    repeat: int = 100,
) -> float:
    """Benchmark the fused mixed-precision GEMM kernel."""
    # Warmup
    for _ in range(warmup):
        out = mixed_gemm_fn(x, w, K_high)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        out = mixed_gemm_fn(x, w, K_high)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat  # ms per iteration


def run_correctness_test(
    specs: List[MixedGemmSpec],
    mixed_gemm_fn=None,  # None = only test baseline
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Run correctness tests for all GEMM specs."""
    print(f"\n{'='*80}")
    print(f"Correctness Test (dtype={dtype})")
    print(f"{'='*80}")

    for spec in specs:
        # Create random quantized-like tensors
        x = torch.randint(-8, 8, (spec.M, spec.K_total), device=device).to(dtype)
        w = torch.randn(spec.N, spec.K_total, device=device, dtype=dtype)

        # Reference: two GEMMs
        ref_output = reference_mixed_gemm(x, w, spec.K_high)

        if mixed_gemm_fn is not None:
            # Fused kernel
            test_output = mixed_gemm_fn(x, w, spec.K_high)

            # Compare
            max_diff = (ref_output - test_output).abs().max().item()
            rel_diff = max_diff / ref_output.abs().max().item() if ref_output.abs().max() > 0 else 0
            passed = rel_diff < 1e-3
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {spec.name}: M={spec.M}, N={spec.N}, K={spec.K_total} "
                  f"(high={spec.K_high}, low={spec.K_low}) "
                  f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f}")
        else:
            print(f"  📋 {spec.name}: M={spec.M}, N={spec.N}, K={spec.K_total} "
                  f"(K_high={spec.K_high}, K_low={spec.K_low}, gs={spec.groupsize}) "
                  f"— baseline only")


def run_performance_test(
    specs: List[MixedGemmSpec],
    mixed_gemm_fn=None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    warmup: int = 20,
    repeat: int = 200,
):
    """Run performance benchmarks for all GEMM specs."""
    print(f"\n{'='*80}")
    print(f"Performance Test (dtype={dtype}, warmup={warmup}, repeat={repeat})")
    print(f"{'='*80}")
    print(f"{'Name':<45} {'M':>6} {'N':>6} {'K':>6} {'K_h':>5} {'K_l':>5} "
          f"{'Baseline(ms)':>12} {'Fused(ms)':>10} {'Speedup':>8}")
    print("-" * 130)

    results = []

    for spec in specs:
        x = torch.randint(-8, 8, (spec.M, spec.K_total), device=device).to(dtype)
        w = torch.randn(spec.N, spec.K_total, device=device, dtype=dtype)

        K_low = spec.K_total - spec.K_high
        x_low = x[:, :K_low]
        x_high = x[:, K_low:]
        w_low = w[:, :K_low]
        w_high = w[:, K_low:]

        # Baseline: two GEMMs
        baseline_ms = benchmark_two_gemms(x_low, w_low, x_high, w_high, warmup, repeat)

        fused_ms = -1.0
        speedup = -1.0
        if mixed_gemm_fn is not None:
            fused_ms = benchmark_mixed_gemm(mixed_gemm_fn, x, w, spec.K_high, warmup, repeat)
            speedup = baseline_ms / fused_ms

        result = {
            "name": spec.name,
            "M": spec.M, "N": spec.N, "K": spec.K_total,
            "K_high": spec.K_high, "K_low": K_low,
            "baseline_ms": baseline_ms,
            "fused_ms": fused_ms,
            "speedup": speedup,
        }
        results.append(result)

        fused_str = f"{fused_ms:.4f}" if fused_ms >= 0 else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup >= 0 else "N/A"
        print(f"  {spec.name:<43} {spec.M:>6} {spec.N:>6} {spec.K_total:>6} "
              f"{spec.K_high:>5} {K_low:>5} {baseline_ms:>12.4f} {fused_str:>10} {speedup_str:>8}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ResQ Mixed-Precision GEMM Test Suite")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,64,256,1024",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--seqlen", type=int, default=1,
                        help="Sequence length (1 for decode, >1 for prefill)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--mode", type=str, default="all", choices=["shapes", "correctness", "perf", "all"])
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    all_results = {}

    for bs in batch_sizes:
        M = bs * args.seqlen
        print(f"\n{'#'*80}")
        print(f"# Batch Size = {bs}, SeqLen = {args.seqlen}, M = {M}")
        print(f"{'#'*80}")

        specs = get_resq_gemm_specs(batch_size=bs, seqlen=args.seqlen)
        unique_specs = get_unique_shapes(specs)

        if args.mode in ("shapes", "all"):
            print(f"\n  Total GEMM calls per forward pass: {len(specs)}")
            print(f"  Unique shapes: {len(unique_specs)}")
            print(f"\n  Unique mixed-precision GEMM shapes:")
            for s in unique_specs:
                print(f"    {s.layer_type:<15} M={s.M:>6}, N={s.N:>6}, K={s.K_total:>6} "
                      f"(K_high={s.K_high}, K_low={s.K_low}, gs={s.groupsize})")

        if args.mode in ("correctness", "all"):
            run_correctness_test(unique_specs, mixed_gemm_fn=None, dtype=dtype)

        if args.mode in ("perf", "all"):
            results = run_performance_test(unique_specs, mixed_gemm_fn=None, dtype=dtype)
            all_results[f"bs{bs}_seq{args.seqlen}"] = results

    if args.output and all_results:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
