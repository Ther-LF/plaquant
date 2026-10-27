"""Round-8 Part B: FP8 KV-cache screen (the accuracy-safe alternative to the
round-2 KV4 that failed the <0.05 gate by +0.76..+1.25 PPL).

Measures, per model (1B/3B/8B): (a) wikitext-2 PPL delta of FP8 K+V cache vs FP16
cache (gate <0.05), for e4m3 and e5m2 at per-vector (per-token-per-head) scaling;
(b) KV memory ratio; (c) per-step KV-read bandwidth savings. SDPA is patched so
attention math stays FP16 — only the stored K/V are FP8-quantized (the real
KV-cache contract). No forward wiring; this is the accuracy+perf screen.
"""
import os, sys, json, argparse, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
DEV = "cuda"


def q_fp8(x, dt, fmax):
    s = x.shape
    xf = x.reshape(-1, s[-1]).float()
    scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-6) / fmax
    q = (xf / scale).to(dt).float() * scale
    return q.reshape(s).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--nsamples", type=int, default=40)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    from promix.eval.data import get_wikitext2

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    hd = model.config.hidden_size // model.config.num_attention_heads
    nkv = model.config.num_key_value_heads

    schemes = {
        "fp8_e4m3_pervec": lambda x: q_fp8(x, torch.float8_e4m3fn, 448.0),
        "fp8_e5m2_pervec": lambda x: q_fp8(x, torch.float8_e5m2, 57344.0),
    }
    orig = torch.nn.functional.scaled_dot_product_attention
    state = {"fn": None}

    def patched(q, k, v, *a, **kw):
        if state["fn"] is not None:
            k = state["fn"](k); v = state["fn"](v)
        return orig(q, k, v, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = patched

    enc = get_wikitext2(seed=0, seqlen=args.seqlen, tokenizer=tok, eval_mode=True).input_ids
    n = min(args.nsamples, enc.shape[1] // args.seqlen)

    def ppl():
        nll, cnt = 0.0, 0
        for i in range(n):
            ids = enc[:, i * args.seqlen:(i + 1) * args.seqlen].to(DEV)
            o = model(ids, labels=ids)
            nll += o.loss.float().item() * (args.seqlen - 1); cnt += (args.seqlen - 1)
        return float(torch.exp(torch.tensor(nll / cnt)))

    state["fn"] = None; p_fp16 = ppl()
    print(f"PPL fp16-KV = {p_fp16:.4f}  (hd={hd} nkv={nkv} n={n})", flush=True)
    res = {"model": args.model, "ppl_fp16_kv": p_fp16, "head_dim": hd, "n": n,
           "schemes": {}}
    for nm, fn in schemes.items():
        state["fn"] = fn; p = ppl(); d = p - p_fp16
        res["schemes"][nm] = dict(ppl=p, delta=d, gate_pass=bool(d < 0.05))
        print(f"  {nm:18s} PPL={p:.4f} delta={d:+.4f} ({'PASS' if d<0.05 else 'FAIL'} <0.05)", flush=True)
    torch.nn.functional.scaled_dot_product_attention = orig
    # memory/bandwidth model: FP16 2 B/elem -> FP8 1 B/elem + per-vec scale (fp16, hd elems)
    bytes_fp16 = 2.0
    bytes_fp8 = 1.0 + 2.0 / hd   # 1 byte code + one fp16 scale per head_dim vector
    res["kv_mem_ratio_fp16_over_fp8"] = bytes_fp16 / bytes_fp8
    res["kv_bw_ratio"] = bytes_fp16 / bytes_fp8
    print(f"KV memory/bandwidth FP16/FP8 = {bytes_fp16/bytes_fp8:.3f}x "
          f"(fp8={bytes_fp8:.4f} B/elem)", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nFP8KV_DONE", flush=True)


if __name__ == "__main__":
    main()
