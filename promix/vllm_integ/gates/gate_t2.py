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
    ap.add_argument("--graphs", action="store_true",
                    help="run with enforce_eager=False (torch.compile + CUDA "
                         "graphs) — the Phase-2 graph-equivalence re-gate")
    ap.add_argument("--decode-route-m", type=int, default=0,
                    help="vllm-perf r1: cap max_num_batched_tokens to this value "
                         "so EVERY forward step is a small-M chunk that routes "
                         "through the decode GEMV (M<=PROMIX_DECODE_M_MAX). This "
                         "scores the full wikitext PPL THROUGH the decode path "
                         "(W4A16 = activation-quant OFF, so PPL is expected to "
                         "IMPROVE vs the W4A4 shared-g reference; report the "
                         "delta either way). 0 = normal prefill-path gate.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = torch.load(os.path.join(args.ckpt, "t2_reference.pt"), weights_only=False)
    ids = ref["input_ids"]                    # (nchunks, S) int32
    ref_lp = ref["target_logprob"]            # (nchunks, S-1) fp32
    ref_top1 = ref["top1"]                    # (nchunks, S-1) int32
    nchunks, S = ids.shape
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)

    llm_kw = {}
    if args.decode_route_m > 0:
        # Chunked prefill with a tiny token budget: every engine step scores at
        # most decode_route_m prompt tokens -> flattened M <= decode_route_m ->
        # the promix decode GEMV route (M=1 single-row / batched W4A16 kernel).
        llm_kw = dict(max_num_batched_tokens=args.decode_route_m,
                      max_num_seqs=max(1, min(args.decode_route_m, 8)),
                      enable_chunked_prefill=True)
    llm = build_llm(args.ckpt, max_model_len=S + 128,
                    enforce_eager=not args.graphs, **llm_kw)
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
    if args.decode_route_m > 0:
        # Decode-route numerics gate: W4A16 decode intentionally differs from the
        # W4A4 shared-g reference (activation quant OFF -> MORE accurate), so the
        # gate is "no degradation beyond 0.1 PPL" (signed), and top-1-vs-W4A4-ref
        # is reported but not gated (the logits legitimately differ).
        gate_ppl = (ppl_vllm - ppl_ref) <= 0.1
        gate_top1 = True
        print(f"T2-DECODE-ROUTE (M<={args.decode_route_m}) signed dPPL = "
              f"{ppl_vllm - ppl_ref:+.4f} (improvement if negative)", flush=True)
    else:
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
    out = args.out or os.path.join(
        args.ckpt,
        f"gate_t2_decode_m{args.decode_route_m}.json" if args.decode_route_m > 0
        else ("gate_t2_graphs.json" if args.graphs else "gate_t2.json"))
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "graphs": args.graphs,
                   "decode_route_m": args.decode_route_m,
                   "ppl_vllm": ppl_vllm, "ppl_ref": ppl_ref,
                   "dppl": dppl, "dppl_signed": ppl_vllm - ppl_ref,
                   "top1_agreement_pct": agree_pct,
                   "positions": total, "chunks": nchunks,
                   "logprob_absdiff_mean": lp_diff.mean().item(),
                   "logprob_absdiff_max": lp_diff.max().item()}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
