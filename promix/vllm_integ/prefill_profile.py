#!/usr/bin/env python3
"""vllm-perf r5 lever 2: per-forward PREFILL decomposition profile.

The prefill pm/nv ratio has sat at 0.54-0.58 (8B) since Phase 3 with only
structural explanations on file (+19% weight bytes, MXFP8 ring at half TC
rate, per-linear act-quant chain). This probe decomposes ONE L-token prefill
step (graphs ON, the serving config) into kernel-time buckets so the gap is
attributed honestly: fused-GEMM vs pack vs Hadamard vs gather vs attention vs
norm/elementwise vs launch gaps — side by side for the promix engine and the
NVFP4/BF16 baselines.

Method: LLM(generate max_tokens=1) on a batch of identical ISL prompts;
median wall from unprofiled reps; kernel attribution from torch.profiler
(CUPTI sees kernels inside CUDA-graph replays) over --prof-reps profiled
reps, normalized per rep. Dumps BOTH the bucketed summary and the full
(kernel name, us, calls) table so the bucketing can be audited post-hoc.

Run (container, idle GPU):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=7 \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.prefill_profile \
      --model <ckpt> --tag pm3b --out out.json [--isl 2048] [--batch 1]
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


def log(*a):
    print(*a, flush=True)


# bucket -> list of lowercase substrings matched against the kernel name
# (first match wins, in this order)
BUCKETS = [
    ("marlin", ["marlin"]),
    ("fused_gemm", ["codesign", "fused_gemm", "cd_fused", "dual", "sm120_bs"]),
    ("nv_gemm", ["fp4", "nvfp4", "cutlass3x", "cutlass_", "fp8_blockscale",
                 "trtllm_gen", "gemm"]),
    ("pack_quant", ["quant_pack", "qpf", "pack_fp", "scaled_fp4_quant",
                    "quantizetofp4", "mxfp8_quant", "quant"]),
    ("gemv", ["gemv"]),
    ("hadamard", ["hadamard", "fht", "fast_hadamard"]),
    ("gather", ["index_select", "gather", "index_elementwise"]),
    ("attention", ["fmha", "flashinfer", "attention", "xqa", "mha", "paged",
                   "batchprefill", "batchdecode", "cascade", "append_kv",
                   "rope", "rotary", "reshape_and_cache"]),
    ("norm", ["rms", "norm"]),
    ("elementwise_cast", ["elementwise", "vectorized", "copy_", "cat",
                          "fill", "triton_poi", "triton_red", "triton_per",
                          "silu", "act_and_mul", "to_copy"]),
    ("memops", ["memcpy", "memset"]),
    ("sampler_logits", ["logits", "softmax", "argmax", "topk", "sort",
                        "sampler", "embedding"]),
]


def bucket_of(name):
    low = name.lower()
    for bucket, subs in BUCKETS:
        for s in subs:
            if s in low:
                return bucket
    return "other"


def make_prompt_ids(tok, L):
    filler = tok("The quick brown fox jumps over the lazy dog. " * 400,
                 add_special_tokens=False)["input_ids"]
    base = (filler * (L // len(filler) + 1))[:L]
    assert len(base) == L
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--isl", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--reps", type=int, default=10, help="unprofiled wall reps")
    ap.add_argument("--prof-reps", type=int, default=5)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--extra-llm-kwargs", default="{}")
    args = ap.parse_args()

    meta = {"tag": args.tag, "model": args.model, "argv": sys.argv,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    import vllm
    meta["vllm"] = vllm.__version__

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, dtype=args.dtype,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util,
                  enable_prefix_caching=False,
                  enforce_eager=args.enforce_eager, disable_log_stats=True)
    if args.quantization:
        kwargs["quantization"] = args.quantization
    kwargs.update(json.loads(args.extra_llm_kwargs))
    log("LLM kwargs:", kwargs)
    llm = LLM(**kwargs)

    ids = make_prompt_ids(tok, args.isl)
    prompts = [{"prompt_token_ids": ids} for _ in range(args.batch)]
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def run():
        llm.generate(prompts, sp, use_tqdm=False)

    # warmup (captures/compiles settle)
    for _ in range(3):
        run()

    # unprofiled wall (the honest step time)
    walls = []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    wall_ms = statistics.median(walls) * 1e3
    log(f"[{args.tag}] isl={args.isl} b={args.batch} wall={wall_ms:.2f} ms "
        f"(all: {[f'{w*1e3:.2f}' for w in walls]})")

    # profiled reps
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.prof_reps):
            run()

    rows = []
    for ev in prof.key_averages():
        dt = getattr(ev, "self_device_time_total", 0) or 0
        if ev.device_type is not None and dt > 0 and ev.count > 0:
            # keep only device kernels (cuda_time_total>0 and not host-side ops)
            if "cudaLaunch" in ev.key or "cudaGraph" in ev.key:
                continue
            rows.append({"name": ev.key, "us": dt / args.prof_reps,
                         "calls": ev.count / args.prof_reps})
    rows.sort(key=lambda r: -r["us"])
    total_us = sum(r["us"] for r in rows)

    buckets = {}
    for r in rows:
        b = bucket_of(r["name"])
        buckets.setdefault(b, {"us": 0.0, "calls": 0.0})
        buckets[b]["us"] += r["us"]
        buckets[b]["calls"] += r["calls"]
    log(f"[{args.tag}] kernel total {total_us/1e3:.2f} ms / wall {wall_ms:.2f} ms "
        f"(gap {wall_ms - total_us/1e3:.2f} ms)")
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]["us"]):
        log(f"  {b:>16}: {v['us']/1e3:8.3f} ms  ({v['calls']:.0f} launches)")

    out = {"meta": meta, "isl": args.isl, "batch": args.batch,
           "wall_ms": wall_ms, "walls_ms": [w * 1e3 for w in walls],
           "kernel_total_ms": total_us / 1e3,
           "buckets": buckets, "kernels": rows[:120]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log("WROTE", args.out)


if __name__ == "__main__":
    main()
