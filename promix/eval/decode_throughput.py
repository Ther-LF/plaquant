"""Decode-throughput harness (SM120 NVFP4+MXFP8 real-kernel model).

Measures the COMPETITIVE end-to-end W4A4 number: autoregressive DECODE
throughput (tokens/s, batch 1, M=1 per step, KV cache) of the full real-kernel
Llama-3.2-1B vs a plain FP16 model, and DIAGNOSES the decode regime (is it
memory-bound? is the prefill-oriented 128x128 fused GEMM realizing the ~4x
weight-bandwidth reduction at M=1, or leaving it on the table?).

Decode is a MEMORY-bandwidth game: 4-bit weights => ~4x less weight HBM traffic
per token => decode (memory-bound GEMV) can be multi-x faster. The fused kernel
here is a 128x128 tensor-core GEMM tuned for large M (prefill); at M=1 the TC
units are ~1/128 utilized and the win must come from weight streaming, not TC
throughput.

Phases (comma-sep --phases):
  * correct  : correctness gates
      (a) real-kernel decode-loop (KV cache, M=1) logits vs full-prefill logits
          of the same tokens -> cosine + argmax match (M=1 + cache path correct)
      (b) real-kernel decode next-token vs fake-twin next-token (teacher-forced)
          -> cosine + token-match (quant faithful at M=1)
  * decode   : tokens/s real vs FP16 (prompt P, generate G, KV cache), median &
               min per-token, prefill time, + per-token component breakdown
               (pack / GEMM / FP16-linear / attention+rest) at M=1.
  * diagnose : memory-bound diagnosis
      - weight-memory reduction: FP16 weight bytes vs packed (4-bit/8-bit+SF)
        bytes, per-Linear and whole-model.
      - achieved weight-read BW at M=1: per-shape fused-GEMM-only time ->
        packed_bytes/time; FP16 cuBLAS GEMV time -> fp16_bytes/time; both vs an
        empirically measured HBM ceiling. => is the fused kernel streaming the
        4-bit weights at ~HBM BW (realizing the reduction) or wasting it?

Remote-only; run on a FREE GPU (idle-waiter). Line-buffered + JSON to disk.

Usage:
  python -m promix.eval.decode_throughput \
      --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --phases correct,decode,diagnose --prompt 32 --gen 128
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
from transformers import DynamicCache

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.quantize import full_network as fn
from promix.quantize.weight_pack import split_k
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, LAYER_SHAPES, HIGH_FRACTION,
)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_min(fn_, iters=50, warmup=10):
    """Min-of-N per-call wall-clock (ms). Min rejects co-tenant spikes."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn_()
    torch.cuda.synchronize()
    best = float("inf")
    tot = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        fn_()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        tot += dt
    return best * 1e3, (tot / iters) * 1e3


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------
def build_real_model(config, nv, mx, cd, qpf, dg=None):
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV)
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=dg)
    model.to(DEV).eval()
    model.config.use_cache = True
    return model, packed


def set_decode_gemv(model, dg):
    """Toggle the decode-GEMV M=1 path on/off across all MixedQuantLinear."""
    n = 0
    for m in model.modules():
        if isinstance(m, MixedQuantLinear):
            m._dg = dg
            n += 1
    return n


def load_decode_gemv():
    from torch.utils.cpp_extension import load
    return load(name="decode_gemv_ext",
                sources=[os.path.join(KERN, "decode_gemv_ext.cu")],
                extra_cuda_cflags=["-std=c++17", "-O3"], verbose=False)


def tokenize_prompt(config, P):
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    from promix.eval.data import get_wikitext2
    loader = get_wikitext2(seed=0, seqlen=max(P, 8), tokenizer=tok, eval_mode=True)
    ids = loader.input_ids[:, :P].to(DEV)
    return tok, ids


# ---------------------------------------------------------------------------
# decode loop (manual, timed per step)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_logits_loop(model, prompt_ids, n_steps, collect=False):
    """Greedy decode n_steps tokens with KV cache. Returns (tokens, per_step_ms,
    prefill_ms, logits_list). per-step wall-clock synced each step."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = DynamicCache()
    out = model(input_ids=prompt_ids, use_cache=True, past_key_values=cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    past = out.past_key_values
    logits = out.logits[:, -1:, :]
    cur = logits.argmax(-1)                      # (1,1)
    toks = []
    per_step = []
    lg = [logits[:, -1, :].float().cpu()] if collect else None
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=cur, use_cache=True, past_key_values=past)
        torch.cuda.synchronize()
        per_step.append((time.perf_counter() - t0) * 1e3)
        past = out.past_key_values
        step_logits = out.logits[:, -1, :]
        if collect:
            lg.append(step_logits.float().cpu())
        cur = step_logits.argmax(-1, keepdim=True)
        toks.append(int(cur.item()))
    return toks, per_step, prefill_ms, lg


# ---------------------------------------------------------------------------
# Phase: correctness gates
# ---------------------------------------------------------------------------
@torch.no_grad()
def _fixed_continuation(model, prompt, G):
    """Greedy teacher-forced continuation: G tokens, each the argmax of a full
    (use_cache=False) forward over the growing sequence. Returns cont (list of
    G ints) and full_all logits (P+G, V) of the final sequence."""
    seq = prompt
    for _ in range(G):
        lg = model(input_ids=seq, use_cache=False).logits[0, -1]
        seq = torch.cat([seq, lg.argmax().view(1, 1)], dim=1)
    full_all = model(input_ids=seq, use_cache=False).logits[0]   # (P+G, V)
    cont = seq[0, prompt.shape[1]:].tolist()
    return cont, full_all


@torch.no_grad()
def _decode_fixed(model, prompt, cont):
    """Incremental KV-cache decode feeding the FIXED tokens `cont`. Returns the
    per-step next-token logits: dec[i] predicts position P-1+i (dec[0] = the
    prefill last-position logit)."""
    cache = DynamicCache()
    out = model(input_ids=prompt, use_cache=True, past_key_values=cache)
    past = out.past_key_values
    dec = [out.logits[0, -1]]
    for t in cont:
        out = model(input_ids=torch.tensor([[t]], device=DEV),
                    use_cache=True, past_key_values=past)
        past = out.past_key_values
        dec.append(out.logits[0, -1])
    return dec


@torch.no_grad()
def run_correct(config, model, packed, nv, mx, cd, qpf, dg, args, out):
    print("=== PHASE: DECODE CORRECTNESS GATES ===", flush=True)
    tok, prompt = tokenize_prompt(config, args.prompt)
    G = min(args.gen, 32)
    P = prompt.shape[1]
    # gates (a)/(b) validate the fused decode path's self-consistency vs prefill
    # (exact) and quant faithfulness; the decode-GEMV's per-Linear bit-exactness
    # (cos 0.999998 vs fused) is validated separately in decode_gemv.py. Force
    # fused here so (a) is a clean kernel-vs-itself check.
    set_decode_gemv(model, None)

    # Fixed teacher-forced continuation + the real-kernel full logits.
    cont, full_all = _fixed_continuation(model, prompt, G)

    # (a) real-kernel incremental decode (KV cache, M=1) vs its own full-prefill
    #     of the SAME tokens. Tests the M=1 + KV-cache path is self-consistent.
    dec_real = _decode_fixed(model, prompt, cont)
    cos_a, match_a = [], 0
    for i, dl in enumerate(dec_real):
        fa = full_all[P - 1 + i]
        cos_a.append(cosine(dl, fa))
        match_a += int(int(dl.argmax()) == int(fa.argmax()))
    min_a = min(cos_a); match_a /= len(dec_real)
    print(f"  (a) decode(M=1,KVcache) vs full-prefill: min cos={min_a:.6f} "
          f"argmax-match={match_a*100:.1f}% ({len(dec_real)} steps)", flush=True)

    # (b) M=1 FAITHFULNESS with SHARED context, benchmarked against PREFILL.
    #     The full-model real-vs-twin logit cosine is < 1 purely from 16 layers of
    #     GEMM-accumulation-order difference (real fused vs twin fp32), the SAME at
    #     prefill and decode (per-Linear M=1 fused-vs-twin is 0.999998, see the GEMV
    #     probe). So the right gate is: a single M=1 decode step adds NO error beyond
    #     what prefill already has. We compare:
    #       cos_pref = real vs twin, full-prefill last-position logit (no cache)
    #       cos_m1   = real vs twin, single M=1 step on an IDENTICAL (cloned) cache
    #     PASS if cos_m1 >= cos_pref - 0.003 (decode no worse than prefill).
    def clone_cache(past):
        legacy = past.to_legacy_cache()
        return DynamicCache.from_legacy_cache(
            tuple((k.clone(), v.clone()) for (k, v) in legacy))

    seq = torch.cat([prompt, torch.tensor([cont], device=DEV)], dim=1)
    t0 = torch.tensor([[cont[0]]], device=DEV)

    fn.wire_fake_mixed(model, packed)
    model.to(DEV).eval(); model.config.use_cache = True
    twin_pref = model(input_ids=seq, use_cache=False).logits[0, P - 1]
    base = DynamicCache()
    model(input_ids=prompt, use_cache=True, past_key_values=base)
    twin_m1 = model(input_ids=t0, use_cache=True,
                    past_key_values=clone_cache(base)).logits[0, -1]

    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)
    model.to(DEV).eval(); model.config.use_cache = True
    real_pref = model(input_ids=seq, use_cache=False).logits[0, P - 1]
    real_m1 = model(input_ids=t0, use_cache=True,
                    past_key_values=clone_cache(base)).logits[0, -1]

    cos_pref = cosine(real_pref, twin_pref)
    cos_m1 = cosine(real_m1, twin_m1)
    match_b = float(int(real_m1.argmax()) == int(twin_m1.argmax()))
    print(f"  (b) real vs fake-twin full-model logit cosine: prefill={cos_pref:.6f} "
          f"M=1-decode={cos_m1:.6f} (argmax-match={bool(match_b)})", flush=True)

    # (c) integrated decode-GEMV vs fused decode, teacher-forced (same cont):
    #     confirms the wired decode-GEMV M=1 path matches the fused decode path
    #     end-to-end (argmax match; per-step logit cosine).
    cos_c = match_c = None
    if dg is not None:
        set_decode_gemv(model, None)
        dec_fused = _decode_fixed(model, prompt, cont)
        set_decode_gemv(model, dg)
        dec_gemv = _decode_fixed(model, prompt, cont)
        set_decode_gemv(model, None)
        cs = [cosine(a, b) for a, b in zip(dec_fused, dec_gemv)]
        match_c = sum(int(int(a.argmax()) == int(b.argmax()))
                      for a, b in zip(dec_fused, dec_gemv)) / len(dec_fused)
        cos_c = min(cs)
        print(f"  (c) decode-GEMV vs fused decode (teacher-forced): min cos={cos_c:.6f} "
              f"argmax-match={match_c*100:.1f}% ({len(dec_fused)} steps)", flush=True)

    gate = (min_a >= 0.9999) and (cos_m1 >= cos_pref - 0.003)
    print(f"  DECODE_CORRECTNESS {'PASS' if gate else 'FAIL'} "
          f"(a: decode==prefill cos>=0.9999; b: M=1 no worse than prefill)", flush=True)
    out["correct"] = dict(
        decode_vs_prefill_min_cos=min_a, decode_vs_prefill_argmax_match=match_a,
        real_vs_twin_prefill_cos=cos_pref, real_vs_twin_m1_cos=cos_m1,
        real_vs_twin_m1_argmax_match=match_b,
        gemv_vs_fused_decode_min_cos=cos_c, gemv_vs_fused_decode_argmax_match=match_c,
        n_steps=len(dec_real), gate_pass=bool(gate))
    return gate


# ---------------------------------------------------------------------------
# Phase: decode throughput + per-token breakdown
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_decode(config, model, fp16, nv, mx, cd, qpf, dg, args, out):
    print("=== PHASE: DECODE THROUGHPUT (tokens/s) ===", flush=True)
    tok, prompt = tokenize_prompt(config, args.prompt)
    G = args.gen

    def measure(m, label):
        # warm (JIT/cuBLAS lazy init, cache alloc), then median over G steps.
        _toks, _ps, _pf, _ = decode_logits_loop(m, prompt, min(G, 16))
        toks, per_step, prefill_ms, _ = decode_logits_loop(m, prompt, G)
        # drop first 2 steps (allocator growth) for steady-state
        steady = per_step[2:] if len(per_step) > 4 else per_step
        med = _median(steady)
        mn = min(steady)
        tps_med = 1e3 / med
        tps_min = 1e3 / mn
        print(f"  [{label}] prefill={prefill_ms:.2f}ms | per-token median={med:.3f}ms "
              f"min={mn:.3f}ms | tok/s median={tps_med:.1f} min-lat={tps_min:.1f}", flush=True)
        return dict(prefill_ms=prefill_ms, per_token_median_ms=med, per_token_min_ms=mn,
                    tokens_s_median=tps_med, tokens_s_minlat=tps_min, n_gen=G)

    # real-kernel, FUSED M=1 path (prefill-oriented GEMM at M=1)
    set_decode_gemv(model, None)
    real_fused = measure(model, "REAL fused-GEMM@M=1")
    # real-kernel, DECODE-GEMV M=1 path (the fix)
    real_gemv = None
    if dg is not None:
        set_decode_gemv(model, dg)
        real_gemv = measure(model, "REAL decode-GEMV@M=1")
        set_decode_gemv(model, None)
    fp = measure(fp16, "FP16")

    su_fused = fp["per_token_median_ms"] / real_fused["per_token_median_ms"]
    print(f"  DECODE SPEEDUP vs FP16: fused-GEMM={su_fused:.3f}x", flush=True)
    su_gemv = None
    if real_gemv is not None:
        su_gemv = fp["per_token_median_ms"] / real_gemv["per_token_median_ms"]
        print(f"                          decode-GEMV={su_gemv:.3f}x "
              f"(GEMV vs fused full-model: "
              f"{real_fused['per_token_median_ms']/real_gemv['per_token_median_ms']:.3f}x)", flush=True)

    # per-token component breakdown at M=1 (sum over 112 Linears).
    bd = _decode_breakdown(nv, mx, cd, qpf, dg, model.config.num_hidden_layers, DEV)
    real_lin = bd["pack_ms"] + bd["gemm_ms"]
    attn_rest_real = real_fused["per_token_median_ms"] - real_lin
    attn_rest_fp16 = fp["per_token_median_ms"] - bd["fp16_linear_ms"]
    print(f"  per-token breakdown (M=1): real pack={bd['pack_ms']:.4f} fused-gemm={bd['gemm_ms']:.4f} "
          f"gemv={bd['gemv_ms']:.4f} (fused-linear={real_lin:.4f}) attn+rest/host={attn_rest_real:.4f} ms", flush=True)
    print(f"                             fp16 linear={bd['fp16_linear_ms']:.4f} "
          f"attn+rest/host={attn_rest_fp16:.4f} ms", flush=True)

    out["decode"] = dict(real_fused=real_fused, real_gemv=real_gemv, fp16=fp,
                         speedup_fused=su_fused, speedup_gemv=su_gemv,
                         prompt=args.prompt, gen=G,
                         breakdown=dict(pack_ms=bd["pack_ms"], fused_gemm_ms=bd["gemm_ms"],
                                        gemv_ms=bd["gemv_ms"], real_fused_linear_ms=real_lin,
                                        fp16_linear_ms=bd["fp16_linear_ms"],
                                        attn_rest_real_ms=attn_rest_real,
                                        attn_rest_fp16_ms=attn_rest_fp16))


@torch.no_grad()
def _decode_breakdown(nv, mx, cd, qpf, dg, n_layers, dev):
    """Per-token (M=1): sum over the 7 shapes x n_layers of fused pack+GEMM,
    decode-GEMV, and FP16 cuBLAS GEMV."""
    M = 1
    pack_sum = gemm_sum = gemv_sum = fp16_lin = 0.0
    for name, N, K in LAYER_SHAPES:
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(hash(name) % 1000)
        W = torch.randn(N, K, device=dev) * 0.08
        x = torch.randn(M, K, device=dev)
        x[:, K_low:] *= 3.0
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf, dg=dg)
        t_pack, _ = time_min(lambda: mql.pack_only(x), iters=50, warmup=10)
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)
        t_gemm, _ = time_min(lambda: cd.fused_gemm_prepacked(
            A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
            A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi, M, N, K_low, K_high),
            iters=50, warmup=10)
        t_gemv = 0.0
        if dg is not None:
            t_gemv, _ = time_min(lambda: mql._decode_gemv(x), iters=50, warmup=10)
        Wh = W.half(); xh = x.half()
        t_fp16, _ = time_min(lambda: xh @ Wh.t(), iters=50, warmup=10)
        pack_sum += t_pack; gemm_sum += t_gemm; gemv_sum += t_gemv; fp16_lin += t_fp16
        del mql, W, x
    cleanup_memory()
    return dict(pack_ms=pack_sum * n_layers, gemm_ms=gemm_sum * n_layers,
                gemv_ms=gemv_sum * n_layers, fp16_linear_ms=fp16_lin * n_layers)


# ---------------------------------------------------------------------------
# Phase: memory-bound diagnosis
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_diagnose(config, packed, nv, mx, cd, qpf, args, out):
    print("=== PHASE: MEMORY-BOUND DIAGNOSIS ===", flush=True)

    # --- empirical HBM ceiling: big memory-bound ops ---
    hbm = _measure_hbm_ceiling(DEV)
    print(f"  empirical HBM ceiling: copy={hbm['copy_tbps']:.2f} TB/s  "
          f"fp16-gemv={hbm['gemv_tbps']:.2f} TB/s  (use={hbm['use_tbps']:.2f} TB/s)", flush=True)

    # --- weight-memory reduction (whole model, from the packed dict) ---
    fp16_bytes = packed_bytes = sf_bytes = 0
    per_linear = []
    for name, pk in packed.items():
        fb = pk.N * pk.K * 2                      # FP16 weight bytes
        wlo = pk.W_lo_packed.numel()             # uint8
        whi = pk.W_hi_bytes.numel()
        sflo = pk.physSFB_lo.numel()
        sfhi = pk.physSFB_hi.numel()
        pb = wlo + whi + sflo + sfhi
        fp16_bytes += fb; packed_bytes += pb; sf_bytes += sflo + sfhi
        per_linear.append(dict(name=name, N=pk.N, K=pk.K, K_low=pk.K_low, K_high=pk.K_high,
                               fp16_bytes=fb, packed_bytes=pb,
                               data_bytes=wlo + whi, sf_bytes=sflo + sfhi,
                               reduction=fb / pb))
    red = fp16_bytes / packed_bytes
    eff_bits = packed_bytes / (fp16_bytes / 2 / 8 * 8) * 8  # bytes/elem*8 = bits/elem
    eff_bits = packed_bytes * 8 / (fp16_bytes / 2)
    print(f"  weight-memory (112 quantized Linears): FP16={fp16_bytes/1e6:.1f} MB "
          f"-> packed={packed_bytes/1e6:.1f} MB (data={ (packed_bytes-sf_bytes)/1e6:.1f} + "
          f"SF={sf_bytes/1e6:.1f}) => {red:.2f}x reduction, {eff_bits:.2f} bits/weight", flush=True)

    # --- per-shape achieved weight-read BW at M=1 ---
    print("  per-shape M=1 weight-read bandwidth (fused vs fp16 cuBLAS):", flush=True)
    M = 1
    shape_rows = []
    for name, N, K in LAYER_SHAPES:
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(hash(name) % 1000)
        W = torch.randn(N, K, device=DEV) * 0.08
        x = torch.randn(M, K, device=DEV)
        x[:, K_low:] *= 3.0
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)
        packed_wb = (mql.W_lo_packed.numel() + mql.W_hi_bytes.numel()
                     + mql.physSFB_lo.numel() + mql.physSFB_hi.numel())
        fp16_wb = N * K * 2
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)
        t_gemm, _ = time_min(lambda: cd.fused_gemm_prepacked(
            A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
            A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi, M, N, K_low, K_high),
            iters=80, warmup=15)
        Wh = W.half(); xh = x.half()
        t_fp16, _ = time_min(lambda: xh @ Wh.t(), iters=80, warmup=15)
        bw_fused = packed_wb / (t_gemm / 1e3) / 1e12     # TB/s
        bw_fp16 = fp16_wb / (t_fp16 / 1e3) / 1e12
        floor_ms = packed_wb / (hbm["use_tbps"] * 1e12) * 1e3   # ideal time at HBM BW
        print(f"    {name:10s} N={N:5d} K={K:5d}: fused M=1 {t_gemm*1e3:7.1f}us "
              f"({bw_fused:.3f} TB/s, {bw_fused/hbm['use_tbps']*100:4.1f}% HBM)  | "
              f"fp16 {t_fp16*1e3:7.1f}us ({bw_fp16:.3f} TB/s) | "
              f"ideal(4bit@HBM)={floor_ms*1e3:.1f}us  fused/ideal={t_gemm/floor_ms:.1f}x", flush=True)
        shape_rows.append(dict(name=name, N=N, K=K, packed_wb=packed_wb, fp16_wb=fp16_wb,
                               fused_us=t_gemm * 1e3, fp16_us=t_fp16 * 1e3,
                               bw_fused_tbps=bw_fused, bw_fp16_tbps=bw_fp16,
                               hbm_frac=bw_fused / hbm["use_tbps"],
                               ideal_us=floor_ms * 1e3, fused_over_ideal=t_gemm / floor_ms))
        del mql, W, x
    cleanup_memory()

    tot_fused_us = sum(r["fused_us"] for r in shape_rows)
    tot_fp16_us = sum(r["fp16_us"] for r in shape_rows)
    tot_ideal_us = sum(r["ideal_us"] for r in shape_rows)
    print(f"  SUM over 7 shapes (1 layer): fused GEMM M=1 = {tot_fused_us:.1f}us  "
          f"fp16 = {tot_fp16_us:.1f}us  4bit@HBM ideal = {tot_ideal_us:.1f}us", flush=True)
    print(f"  => fused is {tot_fused_us/tot_ideal_us:.1f}x SLOWER than the 4-bit HBM "
          f"bandwidth floor; fp16/fused M=1 = {tot_fp16_us/tot_fused_us:.2f}x", flush=True)

    out["diagnose"] = dict(
        hbm=hbm,
        weight_memory=dict(fp16_bytes=fp16_bytes, packed_bytes=packed_bytes,
                           sf_bytes=sf_bytes, reduction=red, eff_bits=eff_bits,
                           per_linear=per_linear),
        m1_bandwidth=dict(shapes=shape_rows, sum_fused_us=tot_fused_us,
                          sum_fp16_us=tot_fp16_us, sum_ideal_us=tot_ideal_us,
                          fused_over_ideal=tot_fused_us / tot_ideal_us,
                          fp16_over_fused=tot_fp16_us / tot_fused_us))


@torch.no_grad()
def _measure_hbm_ceiling(dev):
    """Empirical achievable HBM BW via a large D2D copy and a big memory-bound
    FP16 GEMV (reads a big weight once)."""
    x = torch.empty(1 << 28, device=dev, dtype=torch.float16)   # 512 MB
    y = torch.empty_like(x)
    bytes_cp = x.numel() * 2 * 2                                 # read + write
    t_cp, _ = time_min(lambda: y.copy_(x), iters=40, warmup=10)
    copy_tbps = bytes_cp / (t_cp / 1e3) / 1e12
    # big fp16 GEMV: (1,Kb) @ (Kb,Nb) reads Nb*Kb*2 bytes
    Nb, Kb = 8192, 16384
    W = torch.randn(Nb, Kb, device=dev, dtype=torch.float16)
    xg = torch.randn(1, Kb, device=dev, dtype=torch.float16)
    t_g, _ = time_min(lambda: xg @ W.t(), iters=40, warmup=10)
    gemv_tbps = Nb * Kb * 2 / (t_g / 1e3) / 1e12
    use = max(copy_tbps, gemv_tbps)
    del x, y, W, xg
    cleanup_memory()
    return dict(copy_tbps=copy_tbps, gemv_tbps=gemv_tbps, use_tbps=use)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--phases", type=str, default="correct,decode,diagnose")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "decode_throughput.json"))
    args = ap.parse_args()

    phases = set(args.phases.split(","))
    config = load_config(args.config)
    print(f"Decode-throughput harness | config={args.config} phases={phases} "
          f"prompt={args.prompt} gen={args.gen}", flush=True)
    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()

    print("Building real-kernel model (no GPTQ; latency weight-value independent)...", flush=True)
    model, packed = build_real_model(config, nv, mx, cd, qpf, dg=dg)
    fp16 = _build_fp16_model(config['model']['name'])
    fp16.config.use_cache = True

    out = {"config": args.config, "prompt": args.prompt, "gen": args.gen}
    if "correct" in phases:
        run_correct(config, model, packed, nv, mx, cd, qpf, dg, args, out)
    if "decode" in phases:
        run_decode(config, model, fp16, nv, mx, cd, qpf, dg, args, out)
    if "diagnose" in phases:
        run_diagnose(config, packed, nv, mx, cd, qpf, args, out)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("DECODE_THROUGHPUT_DONE", flush=True)


if __name__ == "__main__":
    main()
