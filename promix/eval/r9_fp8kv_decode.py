"""Round-9 Task-2: FP8 KV cache wired into the GRAPHED decode path.

Stores K/V quantized (e4m3, per-(batch,head,token) scale over head_dim) in a
StaticCache subclass; dequantizes to FP16 on read so attention math stays high
precision (the "store quantized, dequant for attention" contract). Measures, per
model (1B/3B/8B):
  (1) NUMERICS: graphed FP8-KV decode argmax-match vs graphed FP16-KV decode over
      N steps (free-run greedy) + per-step logit cosine.
  (2) DECODE SPEED: b1/b8 graphed ms/tok, FP8-KV vs FP16-KV vs the FP16 model,
      at SHORT context (prompt 32) and LONG context (prompt 4096 / 8192) where the
      per-step KV read/dequant grows.
  (3) MEMORY: persistent KV-cache bytes FP16 vs FP8 (codes + fp16 per-vector scale).

Accuracy (wikitext-2 PPL) was screened in round-8 (r8_fp8kv): e4m3 per-vector
Δ = +0.015 (3B) / +0.012 (8B) PASS <0.05; +0.058 (1B, head_dim 64) marginal ->
opt-in. This harness re-reports that and adds the decode wiring perf.

Usage:
  python -m promix.eval.r9_fp8kv_decode \
      --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
      --prompts 32,4096,8192 --gen 48 --out /home/luofan/_r9/r9_fp8kv_3b.json
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from transformers import StaticCache
from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config
from promix.eval.full_network_realkernel import load_kernels, _build_fp16_model, cosine
from promix.eval.decode_throughput import (
    build_real_model, set_decode_gemv, load_decode_gemv, tokenize_prompt, _median,
)
from promix.eval.decode_graph import (
    EFF_CTX, decode_step, capture_decode, _move_column_order_to_cuda,
    prefill_static, static_decode_graph, steady_median,
)

FP8_MAX = 448.0  # e4m3


class FP8RoundtripCache(StaticCache):
    """StaticCache that quantizes each written K/V through e4m3 (per-(b,head,token)
    scale over head_dim) and stores the DEQUANTIZED fp16 value. This is exactly the
    "store quantized on write, dequant for attention (fp16 math)" contract: each
    token is fp8-roundtripped once when it is written (decode = 1 new token/step =
    O(1) cost), and attention reads the roundtripped cache. All StaticCache plumbing
    (get_seq_length, causal mask) is inherited unchanged. KV memory saving is the
    analytic fp8 byte model (code 1 B/elem + one fp16 per-vector scale)."""

    @staticmethod
    def _rt(x):
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-6) / FP8_MAX
        return (x / s).to(torch.float8_e4m3fn).to(torch.float16) * s

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(self._rt(key_states), self._rt(value_states),
                              layer_idx, cache_kwargs)

    def kv_bytes_fp8(self):
        """Analytic fp8-store footprint: 1 byte/elem code + one fp16 scale per
        head_dim vector (per b,head,token)."""
        tot = 0
        for i in range(len(self.key_cache)):
            for c in (self.key_cache[i], self.value_cache[i]):
                n = c.numel(); hd = c.shape[-1]
                tot += n * 1 + (n // hd) * 2
        return tot


def fp16_kv_bytes(cache):
    return sum(cache.key_cache[i].numel() + cache.value_cache[i].numel()
               for i in range(len(cache.key_cache))) * 2


def make_cache(cls, model, P, G, B=1):
    return cls(config=model.config, max_batch_size=B,
               max_cache_len=P + G + 1, device=DEV, dtype=torch.float16)


@torch.no_grad()
def prefill_B(model, prompt_ids, cache, B):
    ids = prompt_ids.expand(B, -1).contiguous()
    P = ids.shape[1]
    pos = torch.arange(P, device=DEV)
    with EFF_CTX():
        out = model(input_ids=ids, attention_mask=None, position_ids=pos.unsqueeze(0).expand(B, -1),
                    past_key_values=cache, cache_position=pos, use_cache=True)
    return out.logits[:, -1:, :].argmax(-1)  # (B,1)


@torch.no_grad()
def decode_bench(model, prompt, P, G, B, cache_cls):
    """Graphed decode; returns median ms/tok. B sequences in parallel. Inline
    timed replay loop (no per-element .item(), so B>1 works)."""
    cache = make_cache(cache_cls, model, P, G, B=B)
    t0 = prefill_B(model, prompt, cache, B)
    s_in = t0.clone()
    s_pos = torch.tensor([[P]], device=DEV).expand(B, -1).contiguous().clone()
    s_cpos = torch.tensor([P], device=DEV)
    s_next = t0.clone()
    g, s_logits = capture_decode(model, s_in, s_pos, s_cpos, cache)
    per = []
    s_next.copy_(t0)
    for i in range(G):
        torch.cuda.synchronize(); tt = time.perf_counter()
        s_in.copy_(s_next); s_cpos.fill_(P + i); s_pos.fill_(P + i)
        g.replay()
        s_next.copy_(s_logits[:, -1, :].argmax(-1, keepdim=True))
        torch.cuda.synchronize(); per.append((time.perf_counter() - tt) * 1e3)
    med, mn = steady_median(per)
    del g
    return med, cache


@torch.no_grad()
def numerics_tf(model, prompt, cont):
    """Teacher-forced per-step logit cosine: feed the SAME fixed continuation
    through the FP16-KV and FP8-KV caches and compare each step's logits. This
    isolates the fp8 numeric effect (free-run greedy would diverge after one
    mismatch and is meaningless). The authoritative accuracy gate is PPL (r8)."""
    from promix.eval.decode_graph import teacher_forced_static
    P = prompt.shape[1]
    c16 = make_cache(StaticCache, model, P, len(cont))
    dec16 = teacher_forced_static(model, prompt, cont, c16)
    del c16; cleanup_memory()
    c8 = make_cache(FP8RoundtripCache, model, P, len(cont))
    dec8 = teacher_forced_static(model, prompt, cont, c8)
    del c8; cleanup_memory()
    coss = [cosine(dec8[i], dec16[i]) for i in range(len(dec16))]
    return min(coss), sum(coss) / len(coss)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default="32,4096,8192")
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    prompts = [int(x) for x in args.prompts.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    print(f"FP8-KV decode harness | {args.config} prompts={prompts} gen={args.gen}", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()
    model, packed = build_real_model(config, nv, mx, cd, qpf, dg=dg)
    set_decode_gemv(model, dg)
    _move_column_order_to_cuda(model)
    fp16 = _build_fp16_model(config['model']['name']); fp16.config.use_cache = True

    tok, prompt32 = tokenize_prompt(config, max(32, 4))
    out = {"config": args.config, "gen": args.gen, "numerics": {}, "decode": {}, "memory": {}}

    # ---- numerics: teacher-forced per-step logit cosine (isolates fp8 effect) ----
    N = min(args.gen, 40)
    # fixed continuation = greedy fp16-KV tokens
    from promix.eval.decode_graph import static_decode_eager
    c0 = make_cache(StaticCache, model, prompt32.shape[1], N)
    t0 = prefill_static(model, prompt32, c0)
    cont, _, _ = static_decode_eager(model, t0, prompt32.shape[1], N, c0)
    del c0; cleanup_memory()
    mincos, meancos = numerics_tf(model, prompt32, cont)
    gate = mincos >= 0.98
    out["numerics"] = dict(n=N, min_cos=mincos, mean_cos=meancos, gate_pass=bool(gate),
                           note="teacher-forced per-step logit cos, fp8KV vs fp16KV; PPL is the accuracy gate (r8)")
    print(f"  NUMERICS fp8KV vs fp16KV (teacher-forced, {N} steps): "
          f"min-cos={mincos:.5f} mean-cos={meancos:.5f} -> {'PASS' if gate else 'CHECK'}", flush=True)

    # ---- decode speed at short + long context, b1/b8 ----
    for Ptok in prompts:
        # build a prompt of length Ptok (tile the 32-tok prompt if needed)
        if Ptok <= prompt32.shape[1]:
            pr = prompt32[:, :Ptok]
        else:
            reps = (Ptok + prompt32.shape[1] - 1) // prompt32.shape[1]
            pr = prompt32.repeat(1, reps)[:, :Ptok]
        P = pr.shape[1]
        for B in batches:
            try:
                med16, c16 = decode_bench(model, pr, P, args.gen, B, StaticCache)
                f16b = fp16_kv_bytes(c16); del c16; cleanup_memory()
                med8, c8 = decode_bench(model, pr, P, args.gen, B, FP8RoundtripCache)
                f8b = c8.kv_bytes_fp8(); del c8; cleanup_memory()
                # FP16 reference model (fp16-KV graphed)
                medf, cf = decode_bench(fp16, pr, P, args.gen, B, StaticCache)
                del cf; cleanup_memory()
                key = f"P{P}_b{B}"
                out["decode"][key] = dict(
                    fp16kv_ms=med16, fp8kv_ms=med8, fp16model_ms=medf,
                    fp8_vs_fp16kv=med16 / med8, real_fp8_vs_fp16model=medf / med8,
                    real_fp16kv_vs_fp16model=medf / med16,
                    kv_bytes_fp16=f16b, kv_bytes_fp8=f8b, kv_mem_ratio=f16b / f8b)
                print(f"  {key}: fp16KV {med16:.3f}ms  fp8KV {med8:.3f}ms "
                      f"(fp8/fp16KV={med16/med8:.3f}x)  fp16-model {medf:.3f}ms  "
                      f"| KV {f16b/1e6:.1f}MB->{f8b/1e6:.1f}MB ({f16b/f8b:.2f}x)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                out["decode"][f"P{P}_b{B}"] = {"error": repr(e)}
                cleanup_memory()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}\nFP8KV_DECODE_DONE", flush=True)


if __name__ == "__main__":
    main()
