"""End-to-end inference benchmark — compare FP16 vs Real Quant.

Usage:
    python -m promix.inference.benchmark --config promix/configs/llama-3.2-1b.yaml
"""

import argparse
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from promix.utils import DEV


def measure_latency(fn, warmup=5, repeat=20):
    """Measure GPU latency using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times = sorted(times)
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return {"median_ms": median, "mean_ms": mean, "min_ms": times[0], "max_ms": times[-1]}


def benchmark_fp16(model, tokenizer, batch_sizes, seq_lengths):
    """Benchmark FP16 baseline (standard cuBLAS GEMM)."""
    results = {}
    model.eval()
    model.cuda()

    for bs in batch_sizes:
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 1000, (bs, seq_len), device=DEV)

            def run():
                with torch.no_grad():
                    model(input_ids)

            lat = measure_latency(run)
            key = f"bs{bs}_seq{seq_len}"
            results[key] = lat
            tokens = bs * seq_len
            print(f"  FP16 {key}: {lat['median_ms']:.2f} ms ({tokens / lat['median_ms'] * 1000:.0f} tok/s)")

    return results


def benchmark_real_quant(model, tokenizer, batch_sizes, seq_lengths):
    """Benchmark real quantized inference (INT4/INT8 GEMM)."""
    results = {}
    model.eval()
    model.cuda()

    for bs in batch_sizes:
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 1000, (bs, seq_len), device=DEV)

            def run():
                with torch.no_grad():
                    model(input_ids)

            lat = measure_latency(run)
            key = f"bs{bs}_seq{seq_len}"
            results[key] = lat
            tokens = bs * seq_len
            print(f"  Real {key}: {lat['median_ms']:.2f} ms ({tokens / lat['median_ms'] * 1000:.0f} tok/s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="ProMix E2E Benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--kernel_path", type=str, default=None,
                        help="Path to compiled mixed_gemm_l20.so")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    batch_sizes = [1, 4, 16, 64]
    seq_lengths = [128, 512, 2048]

    print(f"=" * 60)
    print(f"ProMix End-to-End Benchmark")
    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === FP16 Baseline ===
    print(f"\n{'='*40}")
    print("Config A: FP16 Baseline (cuBLAS)")
    print(f"{'='*40}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda().eval()
    model_fp16.config.use_cache = False
    fp16_results = benchmark_fp16(model_fp16, tokenizer, batch_sizes, seq_lengths)
    del model_fp16
    torch.cuda.empty_cache()

    # === Real Quant ===
    print(f"\n{'='*40}")
    print("Config B: Real Quant (INT4/INT8 GEMM)")
    print(f"{'='*40}")

    import transformers
    from promix.models.loader import load_model, install_column_order_hooks
    from promix.quantize.fuse_norm import fuse_layer_norms
    from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
    from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
    from promix.quantize.hadamard import get_hadK
    from promix.utils import cleanup_memory
    from promix.inference.weight_packer import pack_model_weights
    from promix.inference.real_forward import install_real_forward, init_kernel

    # Initialize kernel
    if args.kernel_path:
        init_kernel(args.kernel_path)
    else:
        try:
            init_kernel()
        except RuntimeError:
            print("WARNING: Kernel not available, using PyTorch fallback (no speedup)")

    # Prepare model
    transformers.set_seed(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=1, rank=0)

    model = load_model(model_name, dtype=torch.float16)
    fuse_layer_norms(model)
    fuse_basis_to_model(
        model,
        basis_path=config['paths']['basis'],
        rotation_path=config['paths']['rotation'],
        high_fraction=config['quantize']['high_fraction'],
    )
    rearrange_columns(model, high_fraction=config['quantize']['high_fraction'])
    cleanup_memory()
    add_actquant(model)

    # Setup down_proj Hadamard
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = True

    install_column_order_hooks(model)

    # Configure quantizers (needed for quantizer.bits check)
    from promix.eval.ptq import configure_quantizers
    configure_quantizers(model, config)

    # Pack weights and install real forward
    pack_model_weights(model, w_bits=config['quantize'].get('w_bits', 4))
    install_real_forward(model)
    model.config.use_cache = False

    real_results = benchmark_real_quant(model, tokenizer, batch_sizes, seq_lengths)

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY: Speedup (Real Quant vs FP16)")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'FP16 (ms)':<15} {'Real (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for key in fp16_results:
        fp16_ms = fp16_results[key]['median_ms']
        real_ms = real_results[key]['median_ms']
        speedup = fp16_ms / real_ms
        print(f"{key:<20} {fp16_ms:<15.2f} {real_ms:<15.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    main()
