"""Gate T2 — end-to-end logits/PPL vs the shared-g promix reference.

Scores every wikitext eval chunk (identical token windows) through the
vLLM-ProMix engine with prompt_logprobs=1 and compares against the
t2_reference.pt snapshot (shared-g real-kernel promix model, fp32 log-softmax
of fp16 logits):

  * top-1 agreement over all predicted positions  (gate: >= 99.9 %)
  * wikitext PPL  = exp(-mean target logprob)      (gate: |dPPL| <= 0.1)

venv run:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=6 \
  /home/luofan/baselines/venv/bin/python -m promix.vllm_integ.gates.gate_t2 \
      --ckpt /home/luofan/vllm_integ/promix-llama-3.2-3b
"""

import argparse
import json
import os

import torch

from promix.vllm_integ.gates.common import build_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = torch.load(os.path.join(args.ckpt, "t2_reference.pt"), weights_only=False)
    ids = ref["input_ids"]                    # (nchunks, S) int32
    ref_lp = ref["target_logprob"]            # (nchunks, S-1) fp32
    ref_top1 = ref["top1"]                    # (nchunks, S-1) int32
    nchunks, S = ids.shape
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)

    llm = build_llm(args.ckpt, max_model_len=S + 128)
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    prompts = [TokensPrompt(prompt_token_ids=ids[i].tolist()) for i in range(nchunks)]
    outs = llm.generate(prompts, sp)

    v_lp = torch.zeros(nchunks, S - 1, dtype=torch.float64)
    agree = 0
    total = 0
    for i, o in enumerate(outs):
        plp = o.prompt_logprobs
        assert plp is not None and len(plp) == S, (i, len(plp) if plp else None)
        toks = ids[i].tolist()
        for t in range(1, S):
            d = plp[t]
            tgt = toks[t]
            v_lp[i, t - 1] = d[tgt].logprob
            top1 = min(d.items(), key=lambda kv: kv[1].rank)[0]
            agree += int(top1 == int(ref_top1[i, t - 1]))
            total += 1

    ppl_vllm = torch.exp(-v_lp[:nchunks].mean()).item()
    ppl_ref = torch.exp(-ref_lp[:nchunks].double().mean()).item()
    agree_pct = 100.0 * agree / total
    dppl = abs(ppl_vllm - ppl_ref)
    lp_diff = (v_lp - ref_lp[:nchunks].double()).abs()
    gate_ppl = dppl <= 0.1
    gate_top1 = agree_pct >= 99.9
    verdict = "PASS" if (gate_ppl and gate_top1) else "FAIL"
    print(f"T2 ppl_vllm={ppl_vllm:.4f} ppl_ref={ppl_ref:.4f} |d|={dppl:.4f} "
          f"(gate<=0.1 {'ok' if gate_ppl else 'FAIL'})", flush=True)
    print(f"T2 top1_agreement={agree_pct:.3f}% over {total} positions "
          f"(gate>=99.9 {'ok' if gate_top1 else 'FAIL'})", flush=True)
    print(f"T2 target-logprob |diff|: mean={lp_diff.mean():.5f} "
          f"p99={lp_diff.flatten().kthvalue(int(0.99 * lp_diff.numel())).values:.5f} "
          f"max={lp_diff.max():.5f}", flush=True)
    print(f"GATE_T2_{verdict}", flush=True)
    out = args.out or os.path.join(args.ckpt, "gate_t2.json")
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "ppl_vllm": ppl_vllm, "ppl_ref": ppl_ref,
                   "dppl": dppl, "top1_agreement_pct": agree_pct,
                   "positions": total, "chunks": nchunks,
                   "logprob_absdiff_mean": lp_diff.mean().item(),
                   "logprob_absdiff_max": lp_diff.max().item()}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
