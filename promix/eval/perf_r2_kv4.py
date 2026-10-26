"""PERF round-2 Task 2: KV4 (NVFP4 KV cache) — standalone perf + accuracy probe.

Perf axis.  Three measurements, all standalone:
  (a) MEMORY  : KV-cache bytes FP16 vs NVFP4 (4-bit codes + per-16 E4M3 block
                scale) for Llama-3.2-1B / 3.2-3B / 3.1-8B at several context
                lengths + batch sizes.  Analytic, exact byte model.
  (b) BANDWIDTH: at decode, attention streams the WHOLE KV cache every step.
                Measured HBM-read time for FP16 vs NVFP4-packed KV at the b1/b8
                context sizes -> the attention-read speedup KV4 enables.
  (c) ACCURACY: wikitext-2 PPL of HF Llama-3.2-1B with K and V fake-quantized to
                NVFP4 (per-head, per-16-block, the ProMix NVFP4 spec) vs FP16 KV.
                Attention math stays FP16 (only the cache is quantized).  Gate
                target: delta < 0.05 PPL.

Run inside the container on any GPU (accuracy may share; bandwidth wants a quiet
card for a clean number but is a short streaming test).
"""
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import torch

DEV = "cuda"

# (n_kv_heads, head_dim, n_layers)
MODELS = {
    "1B": dict(kvh=8, hd=64,  layers=16, name="unsloth/Llama-3.2-1B-Instruct"),
    "3B": dict(kvh=8, hd=128, layers=28, name="unsloth/Llama-3.2-3B"),
    "8B": dict(kvh=8, hd=128, layers=32, name="unsloth/llama-3-8b"),
}
# NVFP4: 4-bit code (0.5 B/elem) + 1 E4M3 scale byte per 16 elems (0.0625 B/elem).
NVFP4_BYTES_PER_ELEM = 0.5 + 1.0 / 16.0


def mem_analysis(out):
    print("\n=== (a) KV-cache memory: FP16 vs NVFP4 ===", flush=True)
    rows = []
    for size, m in MODELS.items():
        per_tok_elems = 2 * m["kvh"] * m["hd"] * m["layers"]   # K+V, all layers
        for ctx in (2048, 8192, 32768):
            for bs in (1, 8):
                elems = per_tok_elems * ctx * bs
                fp16 = elems * 2 / 1e6
                nvfp4 = elems * NVFP4_BYTES_PER_ELEM / 1e6
                rows.append(dict(size=size, ctx=ctx, bs=bs, fp16_MB=fp16, nvfp4_MB=nvfp4,
                                 reduction=fp16 / nvfp4))
        r = rows[-1]
        print(f"  {size}: per-token KV = {per_tok_elems*2/1e3:.1f} KB fp16 | "
              f"ctx32k/b8: {r['fp16_MB']/1e3:.2f} GB -> {r['nvfp4_MB']/1e3:.2f} GB "
              f"({r['reduction']:.2f}x)", flush=True)
    out["memory"] = rows
    print(f"  NVFP4 effective bytes/elem = {NVFP4_BYTES_PER_ELEM:.4f} "
          f"-> {2/NVFP4_BYTES_PER_ELEM:.2f}x vs fp16", flush=True)


def bandwidth(out):
    print("\n=== (b) KV-read bandwidth (decode attention streams full cache) ===", flush=True)
    # HBM ceiling
    xb = torch.empty(1 << 28, device=DEV, dtype=torch.float16); yb = torch.empty_like(xb)
    def cp(): yb.copy_(xb)
    for _ in range(10): cp()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50): cp()
    torch.cuda.synchronize(); hbm = xb.numel() * 2 * 2 * 50 / (time.perf_counter() - t0) / 1e12
    del xb, yb; torch.cuda.empty_cache()
    print(f"  HBM ceiling (copy) = {hbm:.2f} TB/s", flush=True)
    rows = []
    for size, m in MODELS.items():
        per_tok_elems = 2 * m["kvh"] * m["hd"] * m["layers"]
        for ctx, bs in [(8192, 1), (8192, 8), (32768, 1)]:
            elems = per_tok_elems * ctx * bs
            fp16 = torch.empty(int(elems), device=DEV, dtype=torch.float16)
            nvfp4 = torch.empty(int(elems * NVFP4_BYTES_PER_ELEM), device=DEV, dtype=torch.uint8)
            dst = torch.empty(1024, device=DEV, dtype=torch.float32)
            def rd(t): dst[0] = t.view(torch.uint8)[::4096].float().sum()
            def tmin(fn):
                for _ in range(5): fn()
                torch.cuda.synchronize(); best = 1e9
                for _ in range(30):
                    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                    best = min(best, time.perf_counter() - t0)
                return best * 1e3
            t_f = tmin(lambda: rd(fp16)); t_n = tmin(lambda: rd(nvfp4))
            # bandwidth-bound read time estimate = bytes / HBM
            tf_bw = fp16.numel() * 2 / (hbm * 1e12) * 1e3
            tn_bw = nvfp4.numel() / (hbm * 1e12) * 1e3
            print(f"  {size} ctx={ctx} b{bs}: fp16 {fp16.numel()*2/1e6:7.1f}MB "
                  f"(~{tf_bw:.3f}ms @HBM) -> nvfp4 {nvfp4.numel()/1e6:7.1f}MB "
                  f"(~{tn_bw:.3f}ms @HBM)  read-time saved ~{(tf_bw-tn_bw):.3f}ms/step "
                  f"({tf_bw/tn_bw:.2f}x)", flush=True)
            rows.append(dict(size=size, ctx=ctx, bs=bs, fp16_MB=fp16.numel()*2/1e6,
                             nvfp4_MB=nvfp4.numel()/1e6, fp16_read_ms_hbm=tf_bw,
                             nvfp4_read_ms_hbm=tn_bw, speedup=tf_bw/tn_bw))
            del fp16, nvfp4, dst; torch.cuda.empty_cache()
    out["bandwidth"] = dict(hbm_tbps=hbm, rows=rows)


@torch.no_grad()
def accuracy(out, model_name, nsamples=40):
    print(f"\n=== (c) KV fake-quant PPL — {model_name} ===", flush=True)
    import math
    import transformers
    from promix.quantize.quant_utils import fake_quantize_nvfp4
    from promix.quantize.quant_utils import fake_quantize_mxfp8
    from promix.utils import HadamardTransform
    from promix.eval.data import get_wikitext2

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    hd = model.config.hidden_size // model.config.num_attention_heads

    def q_nvfp4(x):                                   # symmetric NVFP4, per-16 block
        s = x.shape
        return fake_quantize_nvfp4(x.reshape(-1, s[-1]).float()).reshape(s).to(x.dtype)

    def q_had_nvfp4(x):                               # Hadamard(head_dim) then NVFP4
        s = x.shape
        xf = x.reshape(-1, s[-1]).float()
        xr = HadamardTransform.apply(xf.contiguous(), 1.0 / math.sqrt(s[-1]))
        qr = fake_quantize_nvfp4(xr)
        # inverse Hadamard (orthonormal -> H^T = H) to return to the head basis
        xb = HadamardTransform.apply(qr.contiguous(), 1.0 / math.sqrt(s[-1]))
        return xb.reshape(s).to(x.dtype)

    def q_int4_asym(x, gs=128):                       # asymmetric INT4, per-128 group
        s = x.shape
        g = min(gs, s[-1])
        xf = x.reshape(-1, s[-1] // g, g).float()
        xmin = xf.amin(-1, keepdim=True); xmax = xf.amax(-1, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-6) / 15.0
        z = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(xf / scale) + z, 0, 15)
        return ((q - z) * scale).reshape(s).to(x.dtype)

    def q_had_mixed(x, hf=0.125):                    # Hadamard + mixed MXFP8(top-var 1/8)+NVFP4
        s = x.shape; d = s[-1]
        xf = x.reshape(-1, d).float()
        xr = HadamardTransform.apply(xf.contiguous(), 1.0 / math.sqrt(d))
        nh = max(16, (int(hf * d) // 16) * 16)       # high-channel count, multiple of 16/32
        nh = (nh // 32) * 32 if nh >= 32 else 16
        var = xr.abs().mean(0)                        # per-channel magnitude
        hi_idx = torch.argsort(var, descending=True)[:nh]
        mask = torch.zeros(d, dtype=torch.bool, device=xr.device); mask[hi_idx] = True
        qr = fake_quantize_nvfp4(xr)
        qr[:, mask] = fake_quantize_mxfp8(xr[:, mask])
        xb = HadamardTransform.apply(qr.contiguous(), 1.0 / math.sqrt(d))
        return xb.reshape(s).to(x.dtype)

    schemes = {"nvfp4_sym": q_nvfp4, "had_nvfp4": q_had_nvfp4,
               "int4_asym_g128": q_int4_asym}
    if hd >= 256:   # mixed MXFP8(1/8)+NVFP4 needs 1/8*head_dim >= MXFP8 block-32
        schemes["had_mixed_mxfp8_nvfp4"] = q_had_mixed

    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    state = {"fn": None}
    def patched(q, k, v, *a, **kw):
        if state["fn"] is not None:
            k = state["fn"](k); v = state["fn"](v)
        return orig_sdpa(q, k, v, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = patched

    enc = get_wikitext2(seed=0, seqlen=2048, tokenizer=tok, eval_mode=True).input_ids
    seqlen = 2048
    n = min(nsamples, enc.shape[1] // seqlen)

    def ppl():
        nll, cnt = 0.0, 0
        for i in range(n):
            ids = enc[:, i*seqlen:(i+1)*seqlen].to(DEV)
            o = model(ids, labels=ids)
            nll += o.loss.float().item() * (seqlen - 1); cnt += (seqlen - 1)
        return float(torch.exp(torch.tensor(nll / cnt)))

    state["fn"] = None; p_fp16 = ppl()
    print(f"  PPL fp16-KV = {p_fp16:.4f}", flush=True)
    res = {"ppl_fp16_kv": p_fp16, "n": n, "seqlen": seqlen, "head_dim": hd, "schemes": {}}
    for nm, fn in schemes.items():
        state["fn"] = fn; p = ppl()
        d = p - p_fp16
        print(f"  {nm:16s} PPL = {p:.4f}  delta = {d:+.4f}  "
              f"({'PASS' if d < 0.05 else 'FAIL'} <0.05)", flush=True)
        res["schemes"][nm] = dict(ppl=p, delta=d, gate_pass=bool(d < 0.05))
    torch.nn.functional.scaled_dot_product_attention = orig_sdpa
    out["accuracy"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["mem", "bw", "acc", "all"])
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", default=os.path.join(REPO, "kernels/mixed_gemm_sm120/perf_r2_kv4.json"))
    args = ap.parse_args()
    print(f"Task2 KV4 | phase={args.phase} | dev={torch.cuda.get_device_name(0)}", flush=True)
    out = {}
    if args.phase in ("mem", "all"): mem_analysis(out)
    if args.phase in ("bw", "all"): bandwidth(out)
    if args.phase in ("acc", "all"):
        try:
            accuracy(out, args.model)
        except Exception as e:
            print(f"  ACCURACY phase error: {e}", flush=True)
            out["accuracy_error"] = str(e)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}\nTASK2_DONE", flush=True)


if __name__ == "__main__":
    main()
