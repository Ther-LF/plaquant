"""T2 glue control: vLLM-fp16 vs HF-fp16 top-1 on the UNQUANTIZED base model.

If the ProMix T2 top-1 miss is pure engine glue (vLLM paged/flash attention +
RoPE + RMSNorm vs HF eager), then an UNQUANTIZED base model scored both ways
shows the SAME ~top-1 gap. This isolates "glue" from "quantization".

venv run (base model, no promix):
  CUDA_VISIBLE_DEVICES=2 /home/luofan/baselines/venv/bin/python \
    -m promix.vllm_integ.gates.t2_glue_control --ref /home/luofan/vllm_integ/smoke3b/t2_reference.pt \
    --model unsloth/Llama-3.2-3B
"""
import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="t2_reference.pt (for the token windows only)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-chunks", type=int, default=2)
    args = ap.parse_args()

    ref = torch.load(args.ref, weights_only=False)
    ids = ref["input_ids"]
    nchunks = min(args.max_chunks, ids.shape[0])
    S = ids.shape[1]

    # HF fp16 eager top-1
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager").cuda().eval()
    hf.config.use_cache = False
    hf_top1 = []
    with torch.no_grad():
        for i in range(nchunks):
            lg = hf(ids[i:i + 1].cuda().long()).logits.float()
            hf_top1.append(lg[0, :-1].argmax(-1).cpu())
    hf_top1 = torch.stack(hf_top1)
    del hf
    torch.cuda.empty_cache()

    # vLLM fp16 top-1 via prompt_logprobs
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=args.model, dtype="float16", enforce_eager=True,
              max_model_len=S + 128, gpu_memory_utilization=0.4,
              disable_log_stats=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i].tolist())
                         for i in range(nchunks)], sp)
    agree = tot = 0
    for i, o in enumerate(outs):
        plp = o.prompt_logprobs
        toks = ids[i].tolist()
        for t in range(1, S):
            top1 = min(plp[t].items(), key=lambda kv: kv[1].rank)[0]
            agree += int(top1 == int(hf_top1[i, t - 1]))
            tot += 1
    print(f"GLUE_CONTROL vllm-fp16 vs hf-fp16 (UNQUANTIZED {args.model}): "
          f"top1_agreement={100.0 * agree / tot:.3f}% over {tot} positions", flush=True)


if __name__ == "__main__":
    main()
