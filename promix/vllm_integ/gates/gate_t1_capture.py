"""Gate T1 (part 1, venv) — capture per-layer linear I/O from the vLLM model.

Runs ONE eager prefill (the first wikitext eval chunk from t2_reference.pt)
through the vLLM-ProMix model with forward hooks on every promix-quantized
LinearBase, capturing (x, y) per merged layer. The container-side replay
(gate_t1_replay.py) recomputes y from the exported checkpoint with the promix
reference implementation (torch 2.11 exts) and asserts bit-exact.

venv run:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.gates.gate_t1_capture \
      --ckpt ... --out /home/luofan/vllm_integ/t1_capture.pt
"""

import argparse
import os

import torch

from promix.vllm_integ.gates.common import build_llm, promix_layers

_CAPS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = torch.load(os.path.join(args.ckpt, "t2_reference.pt"), weights_only=False)
    ids = ref["input_ids"][0].tolist()          # first chunk, 2048 tokens

    llm = build_llm(args.ckpt)

    def install(model):
        handles = []
        for prefix, mod in promix_layers(model):
            def mk(nm):
                def hook(module, inputs, output):
                    if nm in _CAPS:
                        return
                    x = inputs[0]
                    y = output[0] if isinstance(output, tuple) else output
                    _CAPS[nm] = (x.detach().to("cpu", torch.float16),
                                 y.detach().to("cpu", torch.float16))
                return hook
            handles.append(mod.register_forward_hook(mk(prefix)))
        return len(handles)

    n = llm.apply_model(install)
    print(f"installed hooks on {n[0] if isinstance(n, list) else n} layers", flush=True)

    from vllm import SamplingParams, TokensPrompt
    llm.generate([TokensPrompt(prompt_token_ids=ids)],
                 SamplingParams(max_tokens=1, temperature=0.0))

    assert _CAPS, "no captures — hooks never fired"
    meta = {"ntokens": len(ids), "ckpt": args.ckpt}
    torch.save({"captures": _CAPS, "meta": meta}, args.out)
    tot = sum(v[0].numel() + v[1].numel() for v in _CAPS.values()) * 2 / 2**30
    print(f"T1_CAPTURE_DONE layers={len(_CAPS)} ~{tot:.2f} GiB -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
