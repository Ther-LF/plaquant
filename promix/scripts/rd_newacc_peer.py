"""rd-newacc: run the NEW W4A4 competitors (ARCQuant / SharQ / MicroMix)
AS-SHIPPED on OUR models and score them in OUR committed accuracy protocol.

Protocol (must match the committed scoreboard exactly):
  * WikiText-2: wikitext-2-raw-v1 TEST split, "\n\n".join, single tokenizer
    pass, non-overlapping 2048-token chunks, token-pooled exp(mean NLL).
    bf16 bridge targets: 1B 13.16 / 3B 7.82 / 8B 6.14.
  * C4: QuaRot-standard 256x2048 stream from en/c4-validation.00000-of-00008,
    seed 0 random windows (verbatim re-implementation of
    promix/eval/data.py:get_c4 eval_mode so it runs inside each peer's own
    pinned env without importing promix).
    bf16 bridge targets: 1B 18.51 / 3B 10.43 / 8B 8.88.

The PPL loop itself is each peer's own eval_ppl (the GPTQ-standard loop:
non-overlapping 2048 chunks, exp(mean NLL)) — mathematically identical to
promix/eval/evaluator.py:evaluate_ppl for equal-length chunks. The FP16
bridge run (--quant FP16) through the same code path is the gate that proves
the protocols coincide before any quantized delta is trusted.

Quantization is applied exactly as each peer's own main.py does (official
W4A4 recipe, no tuning): ARCQuant reorder_model_llama + NVFP4 (real agemm
kernels), SharQ quantize_model_llama (SHARQ fused kernels or SHARQ_SIM),
MicroMix reorder_model_llama + mixedgemm kernels (p4/p6/p8 channel mix).

Run with CWD = the peer repo root (relative kernel/saved paths), inside the
peer's pinned venv:

  python rd_newacc_peer.py --peer arcquant --repo . \
      --model /home/luofan/models/Llama-3.2-3B --quant NVFP4 --out out.json
"""
import argparse
import gzip
import json
import os
import random
import sys
import time

import torch


# --------------------------------------------------------------------------
# OUR protocol test streams (self-contained; no promix import inside peer env)
# --------------------------------------------------------------------------
class _TestEnc:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def build_wikitext_testenc(tokenizer):
    """Verbatim protocol of promix/eval/data.py:get_wikitext2 eval_mode."""
    import datasets
    testdata = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
    enc = tokenizer(text="\n\n".join(testdata["text"]), return_tensors="pt")
    return _TestEnc(enc.input_ids)


def build_c4_testenc(tokenizer, nsamples=256, seed=0, seqlen=2048):
    """Verbatim protocol of promix/eval/data.py:get_c4 eval_mode."""
    import datasets
    valdata = datasets.load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    valenc = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            enc = tokenizer(valdata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        s = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        valenc.append(enc.input_ids[:, s:s + seqlen])
    return _TestEnc(torch.hstack(valenc))


# --------------------------------------------------------------------------
# Peer-specific AS-SHIPPED quantization (mirrors each peer's own main.py)
# --------------------------------------------------------------------------
def _model_name(path):
    return path.rstrip("/").split("/")[-1]


def _summ(obj):
    """Best-effort numeric summary of a peer's saved allocation structure."""
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    t = torch.as_tensor(v, dtype=torch.float64)
                    out[str(k)] = dict(mean=float(t.mean()), sum=float(t.sum()),
                                       numel=int(t.numel()))
                except Exception:
                    out[str(k)] = _summ(v)
            return out
        t = torch.as_tensor(obj, dtype=torch.float64)
        return dict(mean=float(t.mean()), sum=float(t.sum()), numel=int(t.numel()))
    except Exception:
        return str(obj)[:200]


def quantize_arcquant(model, args):
    from model_utils import reorder_model_llama
    from datautils import DEV
    name = _model_name(args.model).lower()
    base = f"./saved/{name}"
    reorder_index = torch.load(
        f"{base}_reorder_index_wikitext2_{args.act_sort_metric}.pt",
        weights_only=False)
    select_nums = torch.load(
        f"{base}_select_num_wikitext2_{args.act_sort_metric}.pt",
        weights_only=False)
    model = reorder_model_llama(
        model, device=DEV, kv_cache=args.kv_cache,
        reorder_index=reorder_index, select_nums=select_nums,
        quant_type=args.quant)
    return model, dict(select_nums=_summ(select_nums))


def quantize_sharq(model, args):
    from model_utils import quantize_model_llama
    from datautils import DEV
    model = quantize_model_llama(
        model, device=DEV, kv_cache=args.kv_cache,
        quant_type=args.quant, extra_fusion=False)
    return model, {}


def quantize_micromix(model, args):
    from model_utils import reorder_model_llama
    name = _model_name(args.model)
    reorder_index = torch.load(
        f"./saved/{name}_reorder_index_wikitext2.pt", weights_only=False)
    p6_nums = torch.load(
        f"./saved/{name}_p6_num_wikitext2.pt", weights_only=False)
    p8_nums = torch.load(
        f"./saved/{name}_p8_num_wikitext2.pt", weights_only=False)
    model = reorder_model_llama(
        model, device="cuda:0", kv_cache=args.kv_cache,
        reorder_index=reorder_index, p8_nums=p8_nums, p6_nums=p6_nums)
    return model, dict(p6_nums=_summ(p6_nums), p8_nums=_summ(p8_nums))


QUANTIZERS = dict(arcquant=quantize_arcquant, sharq=quantize_sharq,
                  micromix=quantize_micromix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", required=True, choices=sorted(QUANTIZERS))
    ap.add_argument("--repo", required=True, help="peer repo root")
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="NVFP4",
                    help="peer quant_type, or FP16 for the bridge run")
    ap.add_argument("--act_sort_metric", default="max")
    ap.add_argument("--kv_cache", action="store_true")
    ap.add_argument("--datasets", default="wikitext,c4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "model"))
    sys.path.insert(0, repo)

    import transformers
    transformers.set_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)

    print(f"[rd_newacc] peer={args.peer} quant={args.quant} "
          f"model={args.model}", flush=True)

    want = args.datasets.split(",")
    streams = {}
    if "wikitext" in want:
        t = time.time()
        streams["wikitext"] = build_wikitext_testenc(tok)
        print(f"[rd_newacc] wikitext testenc "
              f"{streams['wikitext'].input_ids.shape} ({time.time()-t:.0f}s)",
              flush=True)
    if "c4" in want:
        t = time.time()
        streams["c4"] = build_c4_testenc(tok)
        print(f"[rd_newacc] c4 testenc {streams['c4'].input_ids.shape} "
              f"({time.time()-t:.0f}s)", flush=True)

    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.eval()

    extra = {}
    if args.quant.upper() != "FP16":
        t = time.time()
        model, extra = QUANTIZERS[args.peer](model, args)
        print(f"[rd_newacc] quantized ({time.time()-t:.0f}s)", flush=True)

    from eval import eval_ppl   # the peer's own GPTQ-standard loop
    results = dict(peer=args.peer, model=args.model, quant=args.quant,
                   kv_cache=bool(args.kv_cache), extra=extra)
    for name, te in streams.items():
        t = time.time()
        ppl = eval_ppl(model, te, "cuda")
        results[name] = float(ppl)
        print(f"[rd_newacc] {name} PPL = {ppl:.4f} ({time.time()-t:.0f}s)",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[rd_newacc] WROTE {args.out}", flush=True)
    print("RD_NEWACC_DONE", flush=True)


if __name__ == "__main__":
    main()
