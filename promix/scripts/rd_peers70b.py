"""rd-peers70b: run ARCQuant / SharQ AS-SHIPPED on Meta-Llama-3-70B-Instruct
and score them in OUR committed accuracy protocol (the rd_scale 70B protocol).

This is `rd_newacc_peer.py` (the 1B/3B/8B peer harness) made 70B-safe:

  * Model load      : CPU-resident bf16 (low_cpu_mem_usage). 70B fp16 = 140 GB
                      and can never be GPU-resident on one 96 GB RTX Pro 6000.
  * Quantization    : UNCHANGED — each peer's own as-shipped W4A4 recipe,
                      exactly as rd_newacc_peer.py calls it. Both peers'
                      quantizers are already layer-offloading (they move one
                      decoder layer to GPU, wrap/quantize it, move it back),
                      so they run at 70B without modification. The packed
                      weights they produce live on GPU (~40-45 GB at 70B).
  * PPL eval        : the ONE substitution — each peer's shipped eval_ppl
                      contains a whole-model `model.to("cuda:0")` which OOMs
                      at 70B. `eval_ppl_offload` below is the same GPTQ-
                      standard loop (identical chunking, identical NLL math:
                      non-overlapping 2048-token chunks, exp(sum nll / total
                      tokens)) with the whole-model move removed — layers are
                      streamed through the GPU one at a time, exactly like
                      promix/eval/evaluator.py:evaluate_ppl which produced
                      every committed 70B row (bf16 / NVFP4 / ProMix).
                      Gate: the bf16 bridge through THIS loop must reproduce
                      the committed 70B bridge (wiki 5.3159 / C4 9.2036).

Protocol (identical to the committed scoreboard / rd_scale 70B rows):
  * WikiText-2: wikitext-2-raw-v1 TEST split, "\n\n".join, single tokenizer
    pass, non-overlapping 2048-token chunks, token-pooled exp(mean NLL).
  * C4: QuaRot-standard 256x2048 windows from
    en/c4-validation.00000-of-00008, seed 0.

Run with CWD = the peer repo root, inside the peer's pinned venv
(venv_acl44: torch 2.11.0+cu130, transformers 4.45.2), with
PYTHONPATH=kernels/build for ARCQuant's agemm:

  python rd_peers70b.py --peer arcquant --repo . \
      --model /home/luofan/latches/models/Meta-Llama-3-70B-Instruct \
      --quant NVFP4 --out out.json
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# OUR protocol test streams (verbatim from rd_newacc_peer.py; self-contained)
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
# Peer-specific AS-SHIPPED quantization (verbatim from rd_newacc_peer.py)
# --------------------------------------------------------------------------
def _model_name(path):
    return path.rstrip("/").split("/")[-1]


def _summ(obj):
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


QUANTIZERS = dict(arcquant=quantize_arcquant, sharq=quantize_sharq)


# --------------------------------------------------------------------------
# 70B-safe PPL eval: the peers' own GPTQ-standard eval_ppl loop with the
# whole-model `model.to("cuda:0")` removed (layer-offload instead). The
# chunking and NLL pooling are IDENTICAL to both peers' model/eval.py and to
# promix/eval/evaluator.py:evaluate_ppl (which produced all committed 70B
# rows): non-overlapping seqlen chunks, exp(sum_nll / (nsamples*seqlen)).
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl_offload(model, testenc, dev, seqlen=2048):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    from tqdm import tqdm
    for i in tqdm(range(len(layers)), desc="(eval) layers"):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # park head pieces back on CPU so the next eval starts clean
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return ppl.item()


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

    print(f"[rd_peers70b] peer={args.peer} quant={args.quant} "
          f"model={args.model}", flush=True)

    want = [d for d in args.datasets.split(",") if d and d != "none"]
    streams = {}
    if "wikitext" in want:
        t = time.time()
        streams["wikitext"] = build_wikitext_testenc(tok)
        print(f"[rd_peers70b] wikitext testenc "
              f"{streams['wikitext'].input_ids.shape} ({time.time()-t:.0f}s)",
              flush=True)
    if "c4" in want:
        t = time.time()
        streams["c4"] = build_c4_testenc(tok)
        print(f"[rd_peers70b] c4 testenc {streams['c4'].input_ids.shape} "
              f"({time.time()-t:.0f}s)", flush=True)

    from transformers import LlamaForCausalLM
    t = time.time()
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"[rd_peers70b] model loaded CPU-resident ({time.time()-t:.0f}s)",
          flush=True)

    extra = {}
    if args.quant.upper() != "FP16":
        t = time.time()
        model, extra = QUANTIZERS[args.peer](model, args)
        print(f"[rd_peers70b] quantized ({time.time()-t:.0f}s) "
              f"gpu_alloc={torch.cuda.memory_allocated()/2**30:.1f}GB",
              flush=True)

    results = dict(peer=args.peer, model=args.model, quant=args.quant,
                   kv_cache=bool(args.kv_cache), extra=extra,
                   eval_loop="rd_peers70b.eval_ppl_offload (layer-offload; "
                             "peer eval_ppl math, whole-model .to removed)")
    for name, te in streams.items():
        t = time.time()
        ppl = eval_ppl_offload(model, te, "cuda")
        results[name] = float(ppl)
        print(f"[rd_peers70b] {name} PPL = {ppl:.4f} ({time.time()-t:.0f}s) "
              f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[rd_peers70b] WROTE {args.out}", flush=True)
    print("RD_PEERS70B_DONE", flush=True)


if __name__ == "__main__":
    main()
