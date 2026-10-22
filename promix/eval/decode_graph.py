"""CUDA-graph DECODE capture harness (SM120 NVFP4+MXFP8 real-kernel model).

The decode-throughput round (docs/decode_throughput_report.md) showed the
competitive W4A4 DECODE path (memory-bound W4A4 GEMV, decode_gemv_ext.cu) lifts
full-model Llama-3.2-1B decode to 0.67x FP16 at M=1 — but is HOST / LAUNCH-LATENCY
bound: the per-token GPU weight-read work is <5 us/Linear while every path pays a
fixed ~10-15 us/launch + Python dispatch across the 112 Linears. The fix (this
round): CUDA-graph capture the whole per-token decoder forward with a STATIC KV
cache, erasing the 112-Linear Python/launch dispatch. (Same lever prefill_graph.py
proved for prefill, overhead_graph ~ 0.)

Approach (vLLM / TensorRT-LLM / HF-StaticCache decode CUDA-graph pattern):
  * STATIC KV cache = transformers.StaticCache pre-allocated to max_cache_len=P+G.
    StaticCache.update() writes K/V at `cache_position` via index_copy_ (a TENSOR
    index) -> graph-safe. The causal mask is rebuilt inside the forward from the
    `cache_position` tensor (LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    with attention_mask=None: `arange(L) > cache_position`), so replaying with a
    new cache_position value advances BOTH the write slot AND the attend length.
  * Capture ONE decode step (input_ids (1,1), position_ids (1,1), cache_position
    (1,)); replay G times, copy_-ing the next token + advancing the position index
    between replays (all on-GPU, no .item() in the hot loop).
  * Capture-safety: warm up on a side stream (cuBLAS/CUTLASS lazy init + the
    decode-GEMV `init_fp4_once` cudaMemcpyToSymbol happen OUTSIDE capture); move the
    o_proj column-reorder index to CUDA (a CPU-index gather would trigger an illegal
    copy during capture — same fix as prefill_graph.py).

NUMERICS GATE FIRST: graphed decode must produce IDENTICAL argmax tokens to eager
decode over N steps, and per-step logit cosine ~1.0. The decode-GEMV itself is
bit-exact (validated in decode_gemv.py / decode_throughput gate (c)); graph replay
runs the identical kernels so it cannot change outputs — a mismatch is a bug.

Usage:
  python -m promix.eval.decode_graph \
      --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --prompt 32 --gen 128
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from transformers import DynamicCache, StaticCache
from torch.nn.attention import sdpa_kernel, SDPBackend

# Static-KV decode must use an EXPLICIT causal mask (padded future slots), which
# dispatches SDPA to a DIFFERENT backend than the no-cache / DynamicCache
# `is_causal` path (flash). Those two backends differ numerically (~0.995 cos/layer
# -> ~0.95 over 16 layers), which would make a graphed static-cache decode look like
# it "changed" the model. Forcing the memory-EFFICIENT backend for the static path
# AND the reference makes static-cache decode bit-faithful to the no-cache prefill
# (measured: cos 0.999993, argmax 100%) — proving the deviation was backend choice,
# not a mask/cache bug. EFFICIENT is capturable and M=1-cheap.
import contextlib
try:
    _ = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    def EFF_CTX():
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
except Exception:
    def EFF_CTX():
        return contextlib.nullcontext()

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, LAYER_SHAPES, HIGH_FRACTION,
)
from promix.eval.decode_throughput import (
    build_real_model, set_decode_gemv, load_decode_gemv, tokenize_prompt,
    decode_logits_loop, _decode_breakdown, _measure_hbm_ceiling, _median, time_min,
)


# ---------------------------------------------------------------------------
# static-KV decode step + CUDA-graph capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_step(model, input_ids, position_ids, cache_position, cache):
    """One autoregressive decode forward against a STATIC KV cache. Returns the
    (1,1,V) logits. attention_mask=None -> the model builds a pure causal mask
    from `cache_position` (tensor-driven, graph-replay-safe). EFFICIENT SDPA
    backend forced for static-vs-reference numeric consistency."""
    with EFF_CTX():
        out = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids,
                    past_key_values=cache, cache_position=cache_position,
                    use_cache=True, num_logits_to_keep=1)
    return out.logits


@torch.no_grad()
def capture_decode(model, static_in, static_pos, static_cpos, cache, warmup=3):
    """Warm up the decode step on a side stream (lazy inits + init_fp4_once happen
    OUTSIDE capture), then capture it. Returns (graph, static_logits)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            decode_step(model, static_in, static_pos, static_cpos, cache)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with EFF_CTX(), torch.cuda.graph(g):
        out = model(input_ids=static_in, attention_mask=None, position_ids=static_pos,
                    past_key_values=cache, cache_position=static_cpos,
                    use_cache=True, num_logits_to_keep=1)
        static_logits = out.logits
    return g, static_logits


def _move_column_order_to_cuda(model):
    """CUDA-graph safety (identical to prefill_graph.py): the o_proj column-reorder
    index is a plain CPU tensor that model.to(DEV) does not move; a CPU-index gather
    during capture triggers an illegal CPU<->CUDA copy. Move them up front (pure
    reindex, no numeric change)."""
    n = 0
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
                n += 1
    return n


@torch.no_grad()
def make_static_cache(model, P, G):
    """Pre-allocated StaticCache sized to the full prompt+gen window."""
    return StaticCache(config=model.config, max_batch_size=1,
                       max_cache_len=P + G + 1, device=DEV, dtype=torch.float16)


@torch.no_grad()
def prefill_static(model, prompt_ids, cache, return_logits=False):
    """Eager prefill into the static cache (positions 0..P-1). Returns the first
    decode token t0 (argmax of the last-position logit) as a (1,1) tensor, or the
    full prefill logits when return_logits=True."""
    P = prompt_ids.shape[1]
    pos = torch.arange(P, device=DEV)
    with EFF_CTX():
        out = model(input_ids=prompt_ids, attention_mask=None,
                    position_ids=pos.unsqueeze(0), past_key_values=cache,
                    cache_position=pos, use_cache=True)
    if return_logits:
        return out.logits
    return out.logits[:, -1:, :].argmax(-1)   # (1,1)


# ---------------------------------------------------------------------------
# static-KV decode loops (eager vs graphed) — identical step, isolates the graph
# ---------------------------------------------------------------------------
@torch.no_grad()
def _full_prefill_logits(model, seq):
    """Ground-truth per-position logits of the full sequence, NO cache (the
    anchor decode_throughput gate (a) uses: decode must match this). Same EFFICIENT
    SDPA backend as the static decode so the only difference is cache vs no-cache."""
    with EFF_CTX():
        return model(input_ids=seq, attention_mask=None, use_cache=False).logits[0]  # (L,V)


@torch.no_grad()
def teacher_forced_static(model, prompt, cont, cache, graph_fns=None):
    """Feed the FIXED tokens `cont` through the static-cache decode path (no greedy
    chaos) and return the per-step next-token logits: dec[i] predicts position
    P-1+i (dec[0] = prefill last-position logit). If graph_fns is given
    (g, static_in, static_pos, static_cpos, static_logits), uses graph replay;
    else eager. Isolates the KV-cache/mask/M=1 path correctness."""
    P = prompt.shape[1]
    t0_logits = prefill_static(model, prompt, cache, return_logits=True)
    dec = [t0_logits[0, -1].float().cpu()]
    if graph_fns is None:
        static_in = torch.empty((1, 1), dtype=torch.long, device=DEV)
        for i, t in enumerate(cont):
            static_in.fill_(t)
            cpos = torch.tensor([P + i], device=DEV)
            ppos = torch.tensor([[P + i]], device=DEV)
            lg = decode_step(model, static_in, ppos, cpos, cache)
            dec.append(lg[0, -1].float().cpu())
    else:
        g, s_in, s_pos, s_cpos, s_logits = graph_fns
        for i, t in enumerate(cont):
            s_in.fill_(t); s_cpos.fill_(P + i); s_pos.fill_(P + i)
            g.replay()
            dec.append(s_logits[0, -1].float().cpu())
    return dec


@torch.no_grad()
def static_decode_eager(model, t0, P, G, cache, collect=False):
    """G-step greedy decode against the static cache, run EAGERLY (no graph).
    Baseline for the numerics gate (same kernels/cache as the graph)."""
    static_in = t0.clone()
    toks, per_step, logits = [], [], []
    for i in range(G):
        cpos = torch.tensor([P + i], device=DEV)
        ppos = torch.tensor([[P + i]], device=DEV)
        torch.cuda.synchronize(); tt = time.perf_counter()
        lg = decode_step(model, static_in, ppos, cpos, cache)   # (1,1,V)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize(); per_step.append((time.perf_counter() - tt) * 1e3)
        if collect:
            logits.append(lg[0, -1].float().cpu())
        static_in = nxt
        toks.append(int(nxt.item()))
    return toks, per_step, logits


@torch.no_grad()
def static_decode_graph(g, static_in, static_pos, static_cpos, static_logits,
                        static_next, t0, P, G, collect=False):
    """G-step greedy decode by REPLAYING the captured graph. The next token and the
    position index advance ON-GPU (copy_/fill_) between replays — no .item() in the
    timed hot loop (a sync per step only for the honest per-token wall-clock)."""
    static_next.copy_(t0)
    toks, per_step, logits = [], [], []
    for i in range(G):
        torch.cuda.synchronize(); tt = time.perf_counter()
        static_in.copy_(static_next)
        static_cpos.fill_(P + i)
        static_pos.fill_(P + i)
        g.replay()
        static_next.copy_(static_logits[:, -1, :].argmax(-1, keepdim=True))
        torch.cuda.synchronize(); per_step.append((time.perf_counter() - tt) * 1e3)
        if collect:
            logits.append(static_logits[0, -1].float().cpu())
        toks.append(int(static_next.item()))
    return toks, per_step, logits


def _replay_only_time(g, iters=100, warmup=20):
    """Min/median pure graph.replay() per-step (no token feedback) — the ceiling the
    graph gives before the host argmax/copy glue."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), _median(ts)


# ---------------------------------------------------------------------------
def steady_median(per_step):
    s = per_step[2:] if len(per_step) > 4 else per_step
    return _median(s), min(s)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--gate_steps", type=int, default=48)
    ap.add_argument("--fp16_graph", action="store_true", default=True,
                    help="also graph the FP16 model (fair graphed-vs-graphed context)")
    ap.add_argument("--no_fp16_graph", dest="fp16_graph", action="store_false")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "decode_graph.json"))
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"Decode-CUDA-graph harness | config={args.config} prompt={args.prompt} "
          f"gen={args.gen}", flush=True)
    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()

    print("Building real-kernel model (decode-GEMV on)...", flush=True)
    model, packed = build_real_model(config, nv, mx, cd, qpf, dg=dg)
    set_decode_gemv(model, dg)
    nmoved = _move_column_order_to_cuda(model)
    print(f"moved {nmoved} column-order index tensors to {DEV}", flush=True)
    print(f"attn_implementation = {model.config._attn_implementation}", flush=True)
    fp16 = _build_fp16_model(config['model']['name'])
    fp16.config.use_cache = True

    tok, prompt = tokenize_prompt(config, args.prompt)
    P = prompt.shape[1]
    G = args.gen
    n_layers = model.config.num_hidden_layers
    out = {"config": args.config, "prompt": P, "gen": G,
           "attn_impl": model.config._attn_implementation}

    # ======================================================================
    # 1. NUMERICS GATE FIRST
    #    (i)  static-cache machinery is bit-faithful to full-prefill  [FUSED, == shipped gate (a)]
    #    (ii) the captured graph reproduces eager decode EXACTLY      [GEMV, argmax + cos]
    #    (report) the pre-existing decode-GEMV vs fused full-model drift (== shipped gate (c))
    # ======================================================================
    print("=== PHASE 1: NUMERICS GATE ===", flush=True)
    NG = min(args.gate_steps, G)

    # fixed teacher-forced continuation = shipped DynamicCache greedy tokens (GEMV path)
    dyn_toks, _, _, _ = decode_logits_loop(model, prompt, NG, collect=False)
    cont = dyn_toks
    seq = torch.cat([prompt, torch.tensor([cont], device=DEV)], dim=1)

    # --- (i) STATIC-CACHE MACHINERY correctness, FUSED path (dg off), teacher-forced ---
    #     The exact analog of decode_throughput gate (a) (decode == full-prefill) but for
    #     StaticCache + explicit mask + M=1 instead of DynamicCache. Both sides use the
    #     SAME fused kernel, so this isolates the static-cache/mask/graph machinery from
    #     the orthogonal GEMV-vs-fused numeric (below). Expect cos ~1.0.
    set_decode_gemv(model, None)
    full_all_fused = _full_prefill_logits(model, seq)
    cache_tf = make_static_cache(model, P, NG)
    dec_fused_static = teacher_forced_static(model, prompt, cont, cache_tf)
    cos_mach = [cosine(dec_fused_static[i], full_all_fused[P - 1 + i].float().cpu())
                for i in range(len(dec_fused_static))]
    match_mach = sum(int(int(dec_fused_static[i].argmax()) ==
                         int(full_all_fused[P - 1 + i].argmax()))
                     for i in range(len(dec_fused_static))) / len(dec_fused_static)
    min_mach = min(cos_mach)
    print(f"  (i) STATIC-cache decode vs full-prefill [fused, teacher-forced]: "
          f"min cos={min_mach:.6f} argmax-match={match_mach*100:.1f}% "
          f"({len(dec_fused_static)} steps) -> proves StaticCache+mask+M=1 machinery", flush=True)
    del cache_tf
    cleanup_memory()

    # --- (report) decode-GEMV vs fused, full-model teacher-forced (PRE-EXISTING) ---
    #     The GEMV accumulates FP32 while the shipped fused GEMM emits bf16, so the two
    #     diverge over 16 layers (shipped decode_throughput gate (c) = cos 0.896 / 87.9%).
    #     Unchanged by this round; per-Linear the GEMV is cos 1.000000 vs the fp32 twin
    #     (decode_gemv.py). Reported, not gated.
    set_decode_gemv(model, dg)
    cache_gv = make_static_cache(model, P, NG)
    dec_gemv_static = teacher_forced_static(model, prompt, cont, cache_gv)
    cos_gv_fused = min(cosine(dec_gemv_static[i], dec_fused_static[i])
                       for i in range(len(dec_gemv_static)))
    match_gv_fused = sum(int(int(dec_gemv_static[i].argmax()) ==
                             int(dec_fused_static[i].argmax()))
                         for i in range(len(dec_gemv_static))) / len(dec_gemv_static)
    print(f"  (report) decode-GEMV vs fused [full-model, teacher-forced]: "
          f"min cos={cos_gv_fused:.6f} argmax-match={match_gv_fused*100:.1f}% "
          f"(PRE-EXISTING fp32-GEMV vs bf16-fused drift; == shipped gate (c))", flush=True)
    del cache_gv
    cleanup_memory()

    # --- (ii) GRAPH FAITHFULNESS: GEMV graph decode == GEMV eager-static decode ---
    # eager STATIC-cache GEMV decode (same kernels+cache as the graph, no capture)
    cache_e = make_static_cache(model, P, NG)
    t0 = prefill_static(model, prompt, cache_e)
    es_toks, _, es_logits = static_decode_eager(model, t0, P, NG, cache_e, collect=True)

    # graphed STATIC-cache decode
    cache_g = make_static_cache(model, P, NG)
    t0g = prefill_static(model, prompt, cache_g)
    static_in = t0g.clone()
    static_pos = torch.tensor([[P]], device=DEV)
    static_cpos = torch.tensor([P], device=DEV)
    static_next = t0g.clone()
    g, static_logits = capture_decode(model, static_in, static_pos, static_cpos, cache_g)
    gr_toks, _, gr_logits = static_decode_graph(
        g, static_in, static_pos, static_cpos, static_logits, static_next,
        t0g, P, NG, collect=True)

    match_ge = sum(int(a == b) for a, b in zip(gr_toks, es_toks)) / len(gr_toks)
    match_gd = sum(int(a == b) for a, b in zip(gr_toks, dyn_toks)) / len(gr_toks)
    cos_ge = min(cosine(a, b) for a, b in zip(gr_logits, es_logits))
    first_div_ge = next((i for i, (a, b) in enumerate(zip(gr_toks, es_toks)) if a != b), None)
    gate_pass = (match_ge >= 0.999999) and (cos_ge >= 0.9999) and \
                (match_mach >= 0.999999) and (min_mach >= 0.9999)
    print(f"  (ii) GRAPH vs eager-static [GEMV, free-run greedy]: argmax-match={match_ge*100:.2f}% "
          f"min-cos={cos_ge:.7f} (first-divergence={first_div_ge}) [{NG} steps] "
          f"-> proves the captured graph changes nothing", flush=True)
    print(f"       graph vs shipped DynamicCache free-run: argmax-match={match_gd*100:.2f}% "
          f"(free-running greedy + DynamicCache's SDPA is_causal vs static's explicit mask "
          f"-> fp-noise greedy divergence, expected; machinery proven exact by (i))", flush=True)
    print(f"  NUMERICS_GATE {'PASS' if gate_pass else 'FAIL'} "
          f"((i) static==prefill cos~1.0 & (ii) graph==eager argmax 100% cos>=0.9999)", flush=True)
    out["numerics"] = dict(n_steps=NG,
                           machinery_static_vs_prefill_fused_min_cos=min_mach,
                           machinery_static_vs_prefill_fused_argmax_match=match_mach,
                           gemv_vs_fused_fullmodel_min_cos=cos_gv_fused,
                           gemv_vs_fused_fullmodel_argmax_match=match_gv_fused,
                           graph_vs_eager_gemv_argmax_match=match_ge,
                           graph_vs_eager_gemv_min_cos=cos_ge,
                           graph_vs_dyncache_freerun_argmax_match=match_gd,
                           first_divergence=first_div_ge, gate_pass=bool(gate_pass))
    del cache_e, cache_g, g
    cleanup_memory()

    # ======================================================================
    # 2. DECODE THROUGHPUT: eager(before) -> graphed(after), vs FP16
    # ======================================================================
    print("=== PHASE 2: DECODE THROUGHPUT (before eager -> after graph) ===", flush=True)

    # ---- BEFORE: shipped eager decode (DynamicCache), real-GEMV + FP16 ----
    decode_logits_loop(model, prompt, min(G, 16))               # warm
    _, ps_real_eager, pf_real, _ = decode_logits_loop(model, prompt, G)
    med_real_eager, min_real_eager = steady_median(ps_real_eager)

    decode_logits_loop(fp16, prompt, min(G, 16))                # warm
    _, ps_fp16_eager, pf_fp16, _ = decode_logits_loop(fp16, prompt, G)
    med_fp16_eager, min_fp16_eager = steady_median(ps_fp16_eager)

    # ---- AFTER: graphed decode (static cache) ----
    cache_t = make_static_cache(model, P, G)
    t0t = prefill_static(model, prompt, cache_t)
    s_in = t0t.clone(); s_pos = torch.tensor([[P]], device=DEV)
    s_cpos = torch.tensor([P], device=DEV); s_next = t0t.clone()
    gT, s_logits = capture_decode(model, s_in, s_pos, s_cpos, cache_t)
    # realistic greedy loop (token feedback) + pure-replay ceiling
    _, ps_real_graph, _ = static_decode_graph(gT, s_in, s_pos, s_cpos, s_logits,
                                              s_next, t0t, P, G)
    med_real_graph, min_real_graph = steady_median(ps_real_graph)
    replay_min, replay_med = _replay_only_time(gT)

    # ---- FP16 graphed (fair graphed-vs-graphed) ----
    med_fp16_graph = min_fp16_graph = float("nan")
    if args.fp16_graph:
        try:
            cache_f = make_static_cache(fp16, P, G)
            t0f = prefill_static(fp16, prompt, cache_f)
            f_in = t0f.clone(); f_pos = torch.tensor([[P]], device=DEV)
            f_cpos = torch.tensor([P], device=DEV); f_next = t0f.clone()
            gF, f_logits = capture_decode(fp16, f_in, f_pos, f_cpos, cache_f)
            _, ps_fp16_graph, _ = static_decode_graph(gF, f_in, f_pos, f_cpos, f_logits,
                                                      f_next, t0f, P, G)
            med_fp16_graph, min_fp16_graph = steady_median(ps_fp16_graph)
            del cache_f, gF
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  fp16 graph failed: {e!r}", flush=True)

    tps = lambda ms: 1e3 / ms
    su_before = med_fp16_eager / med_real_eager
    su_after = med_fp16_eager / med_real_graph
    su_after_vs_fp16graph = (med_fp16_graph / med_real_graph
                             if med_fp16_graph == med_fp16_graph else float("nan"))
    print(f"  BEFORE eager : real-GEMV {med_real_eager:.3f}ms ({tps(med_real_eager):.1f} tok/s)"
          f"  FP16 {med_fp16_eager:.3f}ms ({tps(med_fp16_eager):.1f} tok/s)  "
          f"-> {su_before:.3f}x FP16", flush=True)
    print(f"  AFTER  graph : real-GEMV {med_real_graph:.3f}ms ({tps(med_real_graph):.1f} tok/s) "
          f"[pure-replay {replay_med:.3f}ms]  FP16-graph {med_fp16_graph:.3f}ms "
          f"({tps(med_fp16_graph):.1f} tok/s)", flush=True)
    print(f"  DECODE SPEEDUP vs FP16-eager: BEFORE={su_before:.3f}x  AFTER(graph)={su_after:.3f}x"
          f"  | vs FP16-graph={su_after_vs_fp16graph:.3f}x", flush=True)
    reached = su_after >= 1.5
    print(f"  >>> 1.5x on 1B decode (vs FP16-eager): {'REACHED' if reached else 'SHORT'} "
          f"({su_after:.3f}x)", flush=True)

    # ---- per-token breakdown (M=1) + weight-bandwidth ceiling ----
    bd = _decode_breakdown(nv, mx, cd, qpf, dg, n_layers, DEV)
    gemv_ms = bd["gemv_ms"]
    attn_rest_graph = med_real_graph - gemv_ms           # on-GPU attention/KV/norm/rope/lm_head
    hbm = _measure_hbm_ceiling(DEV)
    packed_bytes = sum(pk.W_lo_packed.numel() + pk.W_hi_bytes.numel()
                       + pk.physSFB_lo.numel() + pk.physSFB_hi.numel()
                       for pk in packed.values())
    fp16_bytes = sum(pk.N * pk.K * 2 for pk in packed.values())
    weight_floor_ms = packed_bytes / (hbm["use_tbps"] * 1e12) * 1e3
    fp16_weight_floor_ms = fp16_bytes / (hbm["use_tbps"] * 1e12) * 1e3
    print(f"  per-token breakdown (graphed): total={med_real_graph:.3f}ms = GEMV {gemv_ms:.3f}ms "
          f"+ attn/KV/norm/rope/lm_head/host {attn_rest_graph:.3f}ms", flush=True)
    print(f"  weight-bandwidth ceiling: packed {packed_bytes/1e6:.1f}MB / {hbm['use_tbps']:.2f}TB/s "
          f"= {weight_floor_ms:.3f}ms/token (FP16 {fp16_bytes/1e6:.1f}MB = {fp16_weight_floor_ms:.3f}ms)",
          flush=True)
    print(f"  graphed per-token / weight-floor = {med_real_graph/weight_floor_ms:.1f}x "
          f"(1B decode is {med_real_graph/weight_floor_ms:.0f}x off its OWN 4-bit weight-BW ceiling)",
          flush=True)

    out["throughput"] = dict(
        prompt=P, gen=G,
        prefill_real_ms=pf_real, prefill_fp16_ms=pf_fp16,
        real_eager_ms=med_real_eager, real_eager_tps=tps(med_real_eager),
        fp16_eager_ms=med_fp16_eager, fp16_eager_tps=tps(med_fp16_eager),
        real_graph_ms=med_real_graph, real_graph_tps=tps(med_real_graph),
        real_graph_min_ms=min_real_graph,
        replay_only_min_ms=replay_min, replay_only_med_ms=replay_med,
        fp16_graph_ms=med_fp16_graph,
        fp16_graph_tps=(tps(med_fp16_graph) if med_fp16_graph == med_fp16_graph else None),
        speedup_before_vs_fp16eager=su_before,
        speedup_after_vs_fp16eager=su_after,
        speedup_after_vs_fp16graph=su_after_vs_fp16graph,
        reached_1p5x=bool(reached),
        breakdown=dict(gemv_ms=gemv_ms, attn_rest_graph_ms=attn_rest_graph,
                       fused_gemm_ms=bd["gemm_ms"], pack_ms=bd["pack_ms"],
                       fp16_linear_ms=bd["fp16_linear_ms"]),
        weight_bw=dict(packed_bytes=packed_bytes, fp16_bytes=fp16_bytes,
                       hbm_tbps=hbm["use_tbps"], weight_floor_ms=weight_floor_ms,
                       fp16_weight_floor_ms=fp16_weight_floor_ms,
                       graph_over_floor=med_real_graph / weight_floor_ms))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("DECODE_GRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
