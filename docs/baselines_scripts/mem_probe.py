#!/usr/bin/env python3
"""Measure peak GPU memory actually consumed by a vLLM model load + steady-state run.

Samples nvidia-smi (whole-GPU used memory) in a background thread while the model
is loaded and a prefill(8192)+decode(b8) workload runs. Reports baseline (before
process start), peak, and delta. Requires the GPU to be otherwise quiet.
"""
import argparse, json, os, subprocess, threading, time

os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
G = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--id=" + G, "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem-util", type=float, default=0.75)
    a = ap.parse_args()

    base = gpu_used()
    stop = threading.Event()
    samples = []

    def sampler():
        while not stop.is_set():
            try:
                samples.append(gpu_used())
            except Exception:
                pass
            time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem_util, enable_prefix_caching=False, disable_log_stats=True)
    if a.quantization:
        kw["quantization"] = a.quantization
    llm = LLM(**kw)
    filler = list(range(100, 8292))
    llm.generate(prompts=[{"prompt_token_ids": filler}], sampling_params=SamplingParams(max_tokens=1), use_tqdm=False)
    llm.generate(prompts=[{"prompt_token_ids": list(range(1, 33))} for _ in range(8)],
                 sampling_params=SamplingParams(max_tokens=32), use_tqdm=False)
    time.sleep(1.0)
    stop.set()
    th.join(timeout=5)

    # weight size on disk
    real = os.path.realpath(a.model)
    wb = 0
    for r, _, fs in os.walk(real):
        for f in fs:
            try:
                wb += os.path.getsize(os.path.realpath(os.path.join(r, f)))
            except OSError:
                pass
    res = {"tag": a.tag, "model": a.model, "quantization": a.quantization,
           "gpu": G, "baseline_mib": base, "peak_mib": max(samples) if samples else None,
           "steady_mib": samples[-1] if samples else None,
           "delta_mib": (max(samples) - base) if samples else None,
           "nsamples": len(samples), "weight_bytes": wb,
           "max_memory_reserved_gib": __import__("torch").cuda.max_memory_reserved() / 2**30}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
