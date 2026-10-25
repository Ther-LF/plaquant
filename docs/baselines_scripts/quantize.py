#!/usr/bin/env python3
"""One-shot quantization via llm-compressor.

scheme:
  gptq_w4a16  : GPTQModifier W4A16, group_size 128, sym, actorder static
  awq_w4a16   : AWQModifier W4A16, group_size 128, sym
  nvfp4       : NVFP4 W4A4 (weights+activations), group_size 16, no lm_head
  fp8         : FP8 W8A8 dynamic per-token activations
Calibration: wikitext-2-raw-v1 train, 256 samples, seqlen 2048.
"""
import argparse, os, time
os.environ.setdefault("HF_HOME", "/home/luofan/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--scheme", required=True, choices=["gptq_w4a16", "awq_w4a16", "nvfp4", "fp8"])
ap.add_argument("--nsamples", type=int, default=256)
ap.add_argument("--seqlen", type=int, default=2048)
args = ap.parse_args()
t0 = time.time()
print("scheme", args.scheme, "model", args.model, flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(ds["text"])
ids = tok(text, add_special_tokens=False)["input_ids"]
samples = [ids[i:i + args.seqlen] for i in range(0, args.seqlen * args.nsamples, args.seqlen)]
from datasets import Dataset
calib = Dataset.from_list([{"input_ids": s} for s in samples])
print("calib samples", len(calib), flush=True)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
IGNORE = ["lm_head"]

if args.scheme == "gptq_w4a16":
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE, dampening_frac=0.01)
elif args.scheme == "awq_w4a16":
    from llmcompressor.modifiers.awq import AWQModifier
    recipe = [AWQModifier(ignore=IGNORE), QuantizationModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)]
elif args.scheme == "nvfp4":
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=IGNORE)
elif args.scheme == "fp8":
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)

oneshot(model=model, dataset=calib, recipe=recipe, max_seq_length=args.seqlen, num_calibration_samples=len(calib))
os.makedirs(args.output, exist_ok=True)
model.save_pretrained(args.output, save_compressed=True)
tok.save_pretrained(args.output)
print("DONE", args.output, "elapsed", time.time() - t0, flush=True)
