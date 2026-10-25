#!/usr/bin/env python3
"""Unified vLLM baseline benchmark: PPL, TTFT, decode tok/s, peak memory.

Protocol (identical across baselines):
- PPL: wikitext-2-raw-v1 test split, tokenizer of the model, add_special_tokens=False,
  texts joined with blank lines, token stream cut into NON-OVERLAPPING 2048-token chunks,
  NLL of every token from the 2nd onward is scored with vLLM prompt_logprobs.
- TTFT: batch 1 and 8 distinct prompts of exactly L tokens (repeated filler), generate 1 token,
  median of --reps after 1 warmup, prefix caching DISABLED.
- Decode: prompt 32 tokens, generate 128 tokens, median of --reps after 1 warmup,
  tok/s = total generated tokens / wall time (prefill included, negligible).
- Peak mem: torch.cuda.max_memory_reserved + nvidia-smi process used.
"""
import argparse, json, os, statistics, subprocess, sys, time, traceback

os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer


def log(*a):
    print(*a, flush=True)


def disk_size(path):
    real = os.path.realpath(path)
    if os.path.isfile(real):
        return os.path.getsize(real)
    tot = 0
    for r, _, fs in os.walk(real):
        for f in fs:
            p = os.path.join(r, f)
            try:
                tot += os.path.getsize(os.path.realpath(p))
            except OSError:
                pass
    return tot


def build_chunks(tok, seq_len, limit):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    if limit:
        chunks = chunks[:limit]
    return chunks


def ppl_from_logprobs(chunks, llm, SamplingParams, bs=8):
    nll, ntok, per_chunk = 0.0, 0, []
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i + bs]
        out = llm.generate(
            prompts=[{"prompt_token_ids": c} for c in batch],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1,
                                            detokenize=False),
            use_tqdm=False,
        )
        for req, c in zip(out, batch):
            lp = req.prompt_logprobs
            s, n = 0.0, 0
            missing = 0
            for pos in range(1, len(c)):
                d = lp[pos]
                if d is None:
                    missing += 1
                    continue
                v = d.get(c[pos])
                if v is None:
                    missing += 1
                    continue
                s += float(v.logprob)
                n += 1
            per_chunk.append({"nll": s, "n": n, "missing": missing})
            nll += s
            ntok += n
    return {"ppl": pow(2.718281828459045, -nll / ntok), "nll": nll, "scored_tokens": ntok,
            "chunks": len(chunks), "missing_logprob_slots": sum(c["missing"] for c in per_chunk)}


def median_run(fn, reps):
    ts = []
    for r in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--ppl-limit", type=int, default=0, help="0 = all chunks")
    ap.add_argument("--prefill-lens", default="2048,4096,8192")
    ap.add_argument("--prefill-batches", default="1,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decode-lens", default="1,8")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--extra-llm-kwargs", default="{}")
    args = ap.parse_args()

    meta = {"tag": args.tag, "model": args.model, "argv": sys.argv, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import vllm
        meta["vllm"] = vllm.__version__
    except Exception:
        meta["vllm"] = "n/a"
    if args.quantization:
        meta["quantization_arg"] = args.quantization

    tok = AutoTokenizer.from_pretrained(args.model)
    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util, enable_prefix_caching=False,
                  enforce_eager=args.enforce_eager, disable_log_stats=True)
    if args.quantization:
        kwargs["quantization"] = args.quantization
    kwargs.update(json.loads(args.extra_llm_kwargs))
    log("LLM kwargs:", kwargs)
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    meta["load_s"] = time.perf_counter() - t0

    # introspect actual config
    try:
        mc = llm.llm_engine.model_config
        meta["quantization_resolved"] = mc.quantization
        meta["dtype_resolved"] = str(mc.dtype)
        meta["max_model_len"] = mc.max_model_len
        meta["hf_config_quant"] = getattr(mc.hf_config, "quantization_config", None)
    except Exception as e:
        meta["introspect_err"] = repr(e)

    res = {"meta": meta}

    if not args.skip_ppl:
        chunks = build_chunks(tok, 2048, args.ppl_limit)
        log(f"PPL chunks: {len(chunks)}")
        t0 = time.perf_counter()
        res["ppl"] = ppl_from_logprobs(chunks, llm, SamplingParams)
        res["ppl"]["seconds"] = time.perf_counter() - t0
        log("PPL:", res["ppl"])

    # TTFT
    lens = [int(x) for x in args.prefill_lens.split(",") if x]
    batches = [int(x) for x in args.prefill_batches.split(",") if x]
    filler = tok("The quick brown fox jumps over the lazy dog. " * 400, add_special_tokens=False)["input_ids"]
    res["ttft"] = {}
    for L in lens:
        base = (filler * (L // len(filler) + 1))[:L]
        assert len(base) == L
        for B in batches:
            prompts = [{"prompt_token_ids": list(base)} for _ in range(B)]
            sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            try:
                llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)  # warmup
                med, allt = median_run(lambda: llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False), args.reps)
                res["ttft"][f"L{L}_b{B}"] = {"median_s": med, "runs_s": allt}
                log(f"TTFT L={L} b={B}: {med*1000:.1f} ms")
            except Exception as e:
                res["ttft"][f"L{L}_b{B}"] = {"error": repr(e)}
                log("TTFT error", L, B, repr(e))

    # Decode
    p32 = tok("Write a short paragraph about the history of computing.", add_special_tokens=False)["input_ids"][:32]
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True)
    res["decode"] = {}
    for B in [int(x) for x in args.decode_lens.split(",") if x]:
        prompts = [{"prompt_token_ids": list(p32)} for _ in range(B)]
        try:
            llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)  # warmup
            torch.cuda.reset_peak_memory_stats()
            med, allt = median_run(lambda: llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False), args.reps)
            res["decode"][f"b{B}"] = {"median_s": med, "tok_per_s": B * args.gen / med, "runs_s": allt}
            log("Decode b=%d: %.1f tok/s (%.3fs)" % (B, res["decode"]["b%d" % B]["tok_per_s"], med))
        except Exception as e:
            res["decode"][f"b{B}"] = {"error": repr(e)}
            log("Decode error", B, repr(e))

    res["memory"] = {
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
        res["memory"]["nvidia_smi_compute_apps"] = out.strip().splitlines()
    except Exception:
        pass
    try:
        res["weight_bytes"] = disk_size(args.model)
    except Exception as e:
        res["weight_bytes"] = None
        res["weight_err"] = repr(e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=str)
    log("WROTE", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
