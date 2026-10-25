#!/usr/bin/env python3
"""Run only TTFT points (optionally a subset) and merge them into an existing result JSON.

Used to fill in the 8192-token TTFT point for models whose config caps at 8192
(needs max_model_len=12288 so prompt+1 output fits).
"""
import argparse, json, os, statistics, sys, time
os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--merge-into", required=True)
ap.add_argument("--quantization", default=None)
ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--gpu-mem-util", type=float, default=0.2)
ap.add_argument("--prefill-lens", default="8192")
ap.add_argument("--prefill-batches", default="1,8")
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tok = AutoTokenizer.from_pretrained(a.model)
kw = dict(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
          gpu_memory_utilization=a.gpu_mem_util, enable_prefix_caching=False, disable_log_stats=True, enforce_eager=True)
if a.quantization:
    kw["quantization"] = a.quantization
llm = LLM(**kw)

d = json.load(open(a.merge_into))
d.setdefault("ttft", {})
filler = tok("The quick brown fox jumps over the lazy dog. " * 400, add_special_tokens=False)["input_ids"]
for L in [int(x) for x in a.prefill_lens.split(",") if x]:
    base = (filler * (L // len(filler) + 1))[:L]
    for B in [int(x) for x in a.prefill_batches.split(",") if x]:
        prompts = [{"prompt_token_ids": list(base)} for _ in range(B)]
        sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        d["ttft"][f"L{L}_b{B}"] = {"median_s": med, "runs_s": ts}
        print(f"TTFT L={L} b={B}: {med*1000:.1f} ms", flush=True)
json.dump(d, open(a.merge_into, "w"), indent=2, default=str)
print("MERGED", a.merge_into)
