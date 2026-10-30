#!/usr/bin/env python3
"""Three-way serving benchmark: ProMix-in-vLLM vs vLLM-NVFP4 vs vLLM-BF16.

Same-strong-stack protocol (extends docs/baselines_scripts/bench_vllm.py):
one vLLM 0.32 engine per config, graphs/compile ON by default (the baselines
ran FULL_AND_PIECEWISE + inductor), prefix caching DISABLED, identical-copy
prompts of exactly ISL tokens, greedy, ignore_eos.

Per (case ISL/OSL x batch B):
  TTFT   = median wall of generate(max_tokens=1)      [1 warmup, --reps]
  FULL   = median wall of generate(max_tokens=OSL)
  TPOT   = (FULL - TTFT) / (OSL - 1)
  tok/s  = B * OSL / FULL (generated); B*(ISL+OSL)/FULL (total)

Plus a TTFT-only prefill sweep at L=2048 over the same batches.

Run (venv, inside the container):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=7 \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.bench_serve \
      --model <ckpt-or-hf-id> --tag promix3b --out results/promix3b.json
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


def median_run(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


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
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--cases", default="1024:128,128:1024",
                    help="comma list of ISL:OSL")
    ap.add_argument("--prefill-len", type=int, default=2048,
                    help="TTFT-only prefill sweep length (0 = skip)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--extra-llm-kwargs", default="{}")
    args = ap.parse_args()

    meta = {"tag": args.tag, "model": args.model, "argv": sys.argv,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
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
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    meta["load_s"] = time.perf_counter() - t0
    try:
        mc = llm.llm_engine.model_config
        meta["quantization_resolved"] = mc.quantization
        meta["dtype_resolved"] = str(mc.dtype)
        meta["enforce_eager"] = args.enforce_eager
    except Exception as e:
        meta["introspect_err"] = repr(e)

    batches = [int(x) for x in args.batches.split(",") if x]
    res = {"meta": meta, "cases": {}}

    def gen(prompts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            ignore_eos=True, detokenize=False)
        llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)

    for case in [c for c in args.cases.split(",") if c]:
        isl, osl = (int(x) for x in case.split(":"))
        base = make_prompt_ids(tok, isl)
        for B in batches:
            prompts = [{"prompt_token_ids": list(base)} for _ in range(B)]
            key = f"isl{isl}_osl{osl}_b{B}"
            try:
                gen(prompts, osl)                      # warmup (full)
                ttft, ttft_runs = median_run(lambda: gen(prompts, 1), args.reps)
                full, full_runs = median_run(lambda: gen(prompts, osl), args.reps)
                tpot = (full - ttft) / max(osl - 1, 1)
                res["cases"][key] = {
                    "isl": isl, "osl": osl, "batch": B,
                    "ttft_s": ttft, "full_s": full, "tpot_ms": tpot * 1e3,
                    "gen_tok_per_s": B * osl / full,
                    "total_tok_per_s": B * (isl + osl) / full,
                    "ttft_runs_s": ttft_runs, "full_runs_s": full_runs,
                }
                log(f"{key}: TTFT {ttft*1e3:.1f} ms  TPOT {tpot*1e3:.2f} ms  "
                    f"gen {B*osl/full:.1f} tok/s")
            except Exception as e:
                res["cases"][key] = {"error": repr(e)}
                log(key, "ERROR", repr(e))

    if args.prefill_len:
        L = args.prefill_len
        base = make_prompt_ids(tok, L)
        res["prefill_ttft"] = {}
        for B in batches:
            prompts = [{"prompt_token_ids": list(base)} for _ in range(B)]
            key = f"L{L}_b{B}"
            try:
                gen(prompts, 1)                        # warmup
                med, runs = median_run(lambda: gen(prompts, 1), args.reps)
                res["prefill_ttft"][key] = {"median_s": med, "runs_s": runs}
                log(f"prefill TTFT {key}: {med*1e3:.1f} ms")
            except Exception as e:
                res["prefill_ttft"][key] = {"error": repr(e)}
                log(key, "ERROR", repr(e))

    res["memory"] = {
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    log("WROTE", args.out)


if __name__ == "__main__":
    main()
