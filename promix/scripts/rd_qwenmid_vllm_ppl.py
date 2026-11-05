#!/usr/bin/env python3
"""rd-qwenmid: vLLM prompt_logprobs PPL (wikitext + C4) for NVFP4 baselines.

Scoring conventions identical to the committed ladder (rd_scale / baselines):
- wikitext: wikitext-2-raw-v1 TEST split, model tokenizer,
  add_special_tokens=False, non-overlapping 2048-token chunks
  (docs/baselines_scripts/bench_vllm.py convention).
- C4: 256 x 2048 validation windows replicated verbatim from
  promix/eval/data.py:get_c4 (same seed/window sampling as evaluate_ppl runs).
- PPL from vLLM prompt_logprobs=1, position 1.. (first token unscored), e-base.

Output JSON mirrors docs/rd_scale_data/70b_nvfp4_ppl.json.
"""
import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_wikitext_chunks(tok, seq_len):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]


def build_c4_chunks(tok, nsamples, seq_len, seed):
    """Verbatim window-sampling logic of promix/eval/data.py:get_c4
    (eval_mode=True): same dataset file, same random stream, same windows."""
    import datasets as hfds
    valdata = hfds.load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    chunks = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            enc = tok(valdata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seq_len:
                break
        s = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
        chunks.append(enc.input_ids[0, s:s + seq_len].tolist())
    return chunks


def ppl_from_logprobs(chunks, llm, SamplingParams, bs=8):
    nll, ntok, missing_total = 0.0, 0, 0
    t0 = time.time()
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i + bs]
        out = llm.generate(
            prompts=[{"prompt_token_ids": c} for c in batch],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1,
                                           prompt_logprobs=1, detokenize=False),
            use_tqdm=False,
        )
        for req, c in zip(out, batch):
            lp = req.prompt_logprobs
            for pos in range(1, len(c)):
                d = lp[pos]
                v = None if d is None else d.get(c[pos])
                if v is None:
                    missing_total += 1
                    continue
                nll += float(v.logprob)
                ntok += 1
    return {"ppl": math.exp(-nll / ntok), "nll": nll, "scored_tokens": ntok,
            "chunks": len(chunks), "missing_logprob_slots": missing_total,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--c4_nsamples", type=int, default=256)
    ap.add_argument("--c4_seed", type=int, default=0,
                    help="must match the promix config calibration seed")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    wiki = build_wikitext_chunks(tok, args.seqlen)
    c4 = build_c4_chunks(tok, args.c4_nsamples, args.seqlen, args.c4_seed)
    print(f"wikitext chunks={len(wiki)}  c4 chunks={len(c4)}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="auto", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=False)

    res = {"meta": {"model": args.model,
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seqlen": args.seqlen, "c4_seed": args.c4_seed}}
    res["wikitext"] = ppl_from_logprobs(wiki, llm, SamplingParams, bs=args.bs)
    print(f"RESULT wikitext ppl={res['wikitext']['ppl']:.4f}", flush=True)
    res["c4"] = ppl_from_logprobs(c4, llm, SamplingParams, bs=args.bs)
    print(f"RESULT c4 ppl={res['c4']['ppl']:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
