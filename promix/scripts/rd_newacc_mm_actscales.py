"""rd-newacc: generate SmoothQuant-style activation scales for MicroMix.

MicroMix's shipped pipeline (model/main.py --do_quant) requires
`./saved/{model_name}.pt` — a SmoothQuant act_scales dict
{module_name: per-channel abs-max} consumed by model/smooth.py:smooth_lm —
but the repo ships NO generator for it. This is the standard SmoothQuant
generate_act_scales logic (per-channel abs-max of every nn.Linear input over a
calibration stream), using the same calibration source as MicroMix's own
reorder_indices.py (wikitext-2 train, seqlen 2048).

Run with CWD = MicroMix repo root, inside its pinned venv:
  python rd_newacc_mm_actscales.py --model /home/luofan/models/Llama-3.2-3B
"""
import argparse
import functools
import os
import random

import torch
import torch.nn as nn


@torch.no_grad()
def get_act_scales(model, tokenizer, nsamples=128, seqlen=2048):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(
                functools.partial(stat_input_hook, name=name)))

    import datasets
    traindata = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(0)
    for i in range(nsamples):
        s = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        model(trainenc.input_ids[:, s:s + seqlen].to(device))
        if (i + 1) % 16 == 0:
            print(f"  calib {i+1}/{nsamples}", flush=True)

    for h in hooks:
        h.remove()
    return act_scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoTokenizer, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).cuda()

    scales = get_act_scales(model, tok, args.nsamples, args.seqlen)
    os.makedirs("./saved", exist_ok=True)
    name = args.model.rstrip("/").split("/")[-1]
    out = f"./saved/{name}.pt"
    torch.save(scales, out)
    print(f"WROTE {out} ({len(scales)} entries)", flush=True)


if __name__ == "__main__":
    main()
