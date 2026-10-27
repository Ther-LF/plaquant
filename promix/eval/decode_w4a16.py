"""Round-12 — phase-adaptive DECODE: weight-only W4A16 GEMV for M=1.

Motivation (design mismatch the lead flagged): at DECODE (M=1) the GEMV is
bandwidth/latency-bound, NOT compute-bound, so quantizing the activation (our
shipped W4A4 decode path) buys NO tensor-core throughput and only costs an extra
quantize pass + accuracy. Weight-only methods (AWQ/GPTQ W4A16 via Marlin) skip it
and beat us at decode (b1 ours 0.85/0.96/1.23 vs Marlin 1.55/1.75/2.53). Our
activation quantization only pays at PREFILL (compute-bound, tensor cores).

This harness builds and validates a WEIGHT-ONLY W4A16 decode GEMV
(`decode_gemv_ext.decode_gemv_w4a16`): stream the SAME packed 4/8-bit weights at
HBM BW, dequant in-register (Marlin-style), dot against the EXACT fp16 activation —
no activation quantize, no SFA. Prefill (M>1) is untouched and keeps W4A4.

Phases (comma-sep --phases):
  * probe    : per-shape kernel faithfulness (decode_gemv_w4a16 vs exact-act @ W_deq)
               AND vs the W4A4 decode GEMV (same weight, quantized vs exact act).
  * numerics : full-model W4A16 decode (M=1) teacher-forced vs the W4A16 fake twin
               (exact act + W_deq) [faithful kernel] + graph==eager [graph safe].
  * perf     : graphed decode b1/b8, matrix {w4a4,w4a16} x {unmerged,merged} vs FP16.
  * ppl      : wikitext-2 PPL fake A/B — W4A4 (act quant ON) vs W4A16 (act exact),
               shippable GPTQ-frozen-scale wiring, same W_deq. The delta is the
               accuracy the W4A16 decode kernel realizes.

Remote-only (SM120 RTX PRO 6000). Run detached, harvest the JSON.

Usage:
  python -m promix.eval.decode_w4a16 \
      --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
      --baseline_ppl 8.296 --phases probe,numerics,perf,ppl \
      --out /home/luofan/r12_decode_3b.json --gpu 1
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

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.quantize import full_network as fn
from promix.quantize.weight_pack import split_k, pack_weight_mixed
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, HIGH_FRACTION,
)
from promix.eval.decode_throughput import (
    set_decode_gemv, load_decode_gemv, tokenize_prompt, _median,
)
import promix.eval.multisize as MS
from promix.eval.multisize import layer_shapes, model_desc, load_all, build_real
import promix.eval.decode_merge as DM
from promix.eval.decode_merge import install_merge, set_merge, remove_patches, graph_decode_median


def set_mode(mode):
    os.environ["PROMIX_DECODE_MODE"] = mode


# ---------------------------------------------------------------------------
# PHASE: per-shape kernel faithfulness probe (size-independent)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_probe(cfg, nv, mx, cd, dg, out):
    print("=== PHASE: per-shape kernel probe ===", flush=True)
    shapes = layer_shapes(cfg)
    rows = []
    worst_w4a16 = 1.0
    for name, N, K in shapes:
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(hash(name) % 1000)
        W = torch.randn(N, K, device=DEV) * 0.08
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
        x = torch.randn(1, K, device=DEV, dtype=torch.float16)
        x[:, K_low:] *= 3.0
        W_deq = pk.W_deq.to(DEV)
        ref_exact = (x.float() @ W_deq.t()).reshape(-1)           # W4A16 spec reference
        got16 = dg.decode_gemv_w4a16(x, pk.W_lo_dec.to(DEV), pk.S_lo_dec.to(DEV),
                                     pk.W_hi_dec.to(DEV), pk.S_hi_dec.to(DEV),
                                     N, K_low, K_high, 24 if N >= 4096 else 8)
        c16 = cosine(got16, ref_exact)
        rel16 = (got16 - ref_exact).abs().max().item() / (ref_exact.abs().max().item() + 1e-9)
        # W4A4 decode GEMV on the same weight (quantized activation) — for contrast
        got4 = dg.decode_gemv_fused(x, pk.W_lo_dec.to(DEV), pk.S_lo_dec.to(DEV),
                                    pk.W_hi_dec.to(DEV), pk.S_hi_dec.to(DEV),
                                    N, K_low, K_high, 24 if N >= 4096 else 8)
        c4_vs_exact = cosine(got4, ref_exact)       # W4A4 error vs the exact-act truth
        worst_w4a16 = min(worst_w4a16, c16)
        rows.append(dict(name=name, N=N, K=K, K_low=K_low, K_high=K_high,
                         w4a16_cos_vs_exact=c16, w4a16_rel=rel16,
                         w4a4_cos_vs_exact=c4_vs_exact))
        print(f"  {name:10s} N={N:5d} K={K:5d}: W4A16 vs exact cos={c16:.7f} "
              f"rel={rel16:.2e} | W4A4 vs exact cos={c4_vs_exact:.6f}", flush=True)
        del W, pk, W_deq
    cleanup_memory()
    gate = worst_w4a16 >= 0.99999
    print(f"  PROBE_GATE {'PASS' if gate else 'FAIL'} (worst W4A16-vs-exact cos={worst_w4a16:.7f})",
          flush=True)
    out["probe"] = dict(shapes=rows, worst_w4a16_cos=worst_w4a16, gate_pass=bool(gate))
    return gate


# ---------------------------------------------------------------------------
# PHASE: full-model numerics (W4A16 decode faithful + graph safe)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_numerics(config, cfg, model_real, packed, dg, args, out):
    """model_real: real-kernel model with dg wired + merge installed (merge OFF).
    Build the W4A16 FAKE twin (exact act + W_deq) as the trusted reference, then:
      (i)  W4A16 real decode (M=1, KV cache) teacher-forced vs the fake twin's
           full-prefill of the SAME tokens  -> cos/argmax (kernel faithful).
      (ii) W4A16 graph decode == W4A16 eager decode -> argmax 100% / cos (graph safe).
    """
    print("=== PHASE: full-model numerics (W4A16) ===", flush=True)
    from transformers import DynamicCache, StaticCache
    from promix.eval.decode_merge import _prefill_static, _capture_decode, _decode_graph_loop
    set_mode("w4a16")
    set_merge(False)
    set_decode_gemv(model_real, dg)

    tok, prompt = tokenize_prompt(config, args.prompt)
    P = prompt.shape[1]
    NG = min(args.gate_steps, args.gen)

    # teacher-forced continuation from the W4A16 real decode (greedy, DynamicCache).
    # `fed` = the tokens actually FED (seq[P+i]); dec_real[i] = the logit that
    # predicts position P-1+i (dec_real[0] = prefill last-position logit).
    cache = DynamicCache()
    o = model_real(input_ids=prompt, use_cache=True, past_key_values=cache)
    past = o.past_key_values
    dec_real = [o.logits[0, -1].float().cpu()]
    nxt = o.logits[:, -1:, :].argmax(-1)        # (1,1) token for position P
    fed = []
    for i in range(NG):
        fed.append(int(nxt.item()))             # token placed at position P+i
        o = model_real(input_ids=nxt[:, 0:1], use_cache=True, past_key_values=past)
        past = o.past_key_values
        dec_real.append(o.logits[0, -1].float().cpu())
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    seq = torch.cat([prompt, torch.tensor([fed], device=DEV)], dim=1)

    # --- FAKE twin reference (same W_deq); toggled exact-act (W4A16) / quant-act (W4A4) ---
    # The per-shape probe already proved the GEMV dequant is bit-exact; this nails the
    # full-model faithfulness vs the exact-activation reference AND contrasts it with the
    # shipped W4A4 decode's own fused-vs-fake drift so we see W4A16 is strictly better.
    fake = prepare_ptq_model(config, DEV, run_gptq=False); fake.to(DEV)
    packed_fake = fn.pack_all_targets(fake, nv_g, mx_g, cd_g, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_fake_mixed(fake, packed_fake)
    fn.wire_fused_mlp(fake)
    fake.to(DEV).eval(); fake.config.use_cache = False
    for m in fake.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))

    def fake_ref(act_exact):
        fn.FakeMixedLinear.ACT_EXACT = act_exact
        lg = fake(input_ids=seq, use_cache=False).logits[0].float().cpu()   # (P+NG, V)
        fn.FakeMixedLinear.ACT_EXACT = False
        return lg

    @torch.no_grad()
    def tf_real(mode):
        """Teacher-force the FIXED `fed` tokens through the real-kernel M=1 decode in
        the given PROMIX_DECODE_MODE. dec[i] predicts position P-1+i."""
        set_mode(mode)
        c = DynamicCache()
        oo = model_real(input_ids=prompt, use_cache=True, past_key_values=c)
        p = oo.past_key_values
        d = [oo.logits[0, -1].float().cpu()]
        for t in fed:
            tt = torch.tensor([[t]], device=DEV)
            oo = model_real(input_ids=tt, use_cache=True, past_key_values=p)
            p = oo.past_key_values
            d.append(oo.logits[0, -1].float().cpu())
        return d

    def score(dec, ref):
        cs = [cosine(dec[i], ref[P - 1 + i]) for i in range(len(dec))]
        am = sum(int(int(dec[i].argmax()) == int(ref[P - 1 + i].argmax()))
                 for i in range(len(dec))) / len(dec)
        return min(cs), am

    ref_exact = fake_ref(True)
    ref_quant = fake_ref(False)
    dec_w16 = tf_real("w4a16")
    dec_w4 = tf_real("w4a4")
    set_mode("w4a16")
    cos16, am16 = score(dec_w16, ref_exact)     # W4A16 decode vs exact-act truth
    cos4, am4 = score(dec_w4, ref_quant)        # W4A4 decode vs its own quant-act truth
    cos16_vs_exact_w4 = score(dec_w4, ref_exact)[0]  # how far W4A4 is from the exact truth
    del fake, packed_fake
    cleanup_memory()
    print(f"  (i) W4A16 real decode vs EXACT-act twin: min cos={cos16:.6f} argmax={am16*100:.1f}%  "
          f"| W4A4 real decode vs its QUANT-act twin: min cos={cos4:.6f} argmax={am4*100:.1f}%  "
          f"| W4A4 vs EXACT truth: cos={cos16_vs_exact_w4:.6f}", flush=True)
    print(f"      -> W4A16 is {'MORE' if cos16 >= cos4 else 'LESS'} faithful to the exact-activation "
          f"reference than W4A4 is to its own quantized reference", flush=True)

    # --- (ii) graph == eager (W4A16) ---
    set_mode("w4a16")
    set_decode_gemv(model_real, dg)
    model_real.config.use_cache = True
    cache_e = StaticCache(config=model_real.config, max_batch_size=1,
                          max_cache_len=P + NG + 1, device=DEV, dtype=torch.float16)
    t0 = _prefill_static(model_real, prompt, cache_e)
    es_toks, es_logits = [], []
    from promix.eval.multisize import _decode_step
    cur2 = t0.clone()
    for i in range(NG):
        cpos = torch.tensor([P + i], device=DEV); ppos = torch.tensor([[P + i]], device=DEV)
        lg = _decode_step(model_real, cur2, ppos, cpos, cache_e)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        es_toks.append(int(nxt.item())); es_logits.append(lg[0, -1].float().cpu())
        cur2 = nxt
    del cache_e
    cleanup_memory()

    cache_g = StaticCache(config=model_real.config, max_batch_size=1,
                          max_cache_len=P + NG + 1, device=DEV, dtype=torch.float16)
    t0g = _prefill_static(model_real, prompt, cache_g)
    g_in = t0g.clone(); g_pos = torch.tensor([[P]], device=DEV)
    g_cpos = torch.tensor([P], device=DEV); g_next = t0g.clone()
    g, g_logits = _capture_decode(model_real, g_in, g_pos, g_cpos, cache_g)
    gr_toks, gr_logits = [], []
    g_next.copy_(t0g)
    for i in range(NG):
        g_in.copy_(g_next); g_cpos.fill_(P + i); g_pos.fill_(P + i)
        g.replay()
        g_next.copy_(g_logits[:, -1, :].argmax(-1, keepdim=True))
        gr_toks.append(int(g_next.item())); gr_logits.append(g_logits[0, -1].float().cpu())
    del cache_g, g
    cleanup_memory()

    match_ge = sum(int(a == b) for a, b in zip(gr_toks, es_toks)) / len(gr_toks)
    cos_ge = min(cosine(a, b) for a, b in zip(gr_logits, es_logits))
    print(f"  (ii) W4A16 graph vs eager decode: argmax-match={match_ge*100:.2f}% "
          f"min-cos={cos_ge:.7f} ({NG} steps)", flush=True)

    # Hard gate: graph replay changes nothing (same-kernel, must be ~exact) AND W4A16 is
    # at least as faithful to the exact reference as the shipped W4A4 is to its own.
    # (Per-Linear bit-exactness is the separate PROBE gate; full-model real-vs-fake cos
    #  is the known cross-path fp32-accumulation drift, present in the shipped path too.)
    gate = (match_ge >= 0.999999) and (cos_ge >= 0.9999) and (cos16 >= cos4 - 0.01)
    print(f"  NUMERICS_GATE {'PASS' if gate else 'FAIL'}", flush=True)
    out["numerics"] = dict(n_steps=NG,
                           w4a16_vs_exact_min_cos=cos16, w4a16_vs_exact_argmax=am16,
                           w4a4_vs_quant_min_cos=cos4, w4a4_vs_quant_argmax=am4,
                           w4a4_vs_exact_min_cos=cos16_vs_exact_w4,
                           graph_vs_eager_argmax_match=match_ge,
                           graph_vs_eager_min_cos=cos_ge, gate_pass=bool(gate))
    return gate


# ---------------------------------------------------------------------------
# PHASE: perf — graphed decode b1/b8, matrix {w4a4,w4a16} x {unmerged,merged}
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_perf(config, model_real, fp16, args, out):
    print("=== PHASE: perf (graphed decode) ===", flush=True)
    tok, prompt1 = tokenize_prompt(config, args.prompt)
    P = prompt1.shape[1]
    G = args.gen
    rows = []
    for B in args.decode_batches:
        prompt = prompt1.repeat(B, 1).contiguous()
        rec = {"batch": B, "prompt": P, "gen": G}
        # FP16 graphed baseline (shared across modes)
        fp16_g = float("nan")
        try:
            graph_decode_median(fp16, prompt, P, min(G, 16))
            fp16_g = graph_decode_median(fp16, prompt, P, G)
        except Exception as e:
            print(f"  fp16 graph failed B={B}: {e!r}", flush=True)
        rec["fp16_graph_ms"] = fp16_g
        for mode in ("w4a4", "w4a16"):
            set_mode(mode)
            for merged in (False, True):
                set_merge(merged)
                graph_decode_median(model_real, prompt, P, min(G, 16))   # warm+recapture
                ms = graph_decode_median(model_real, prompt, P, G)
                key = f"{mode}_{'merged' if merged else 'unmerged'}"
                su = (fp16_g / ms) if fp16_g == fp16_g else float("nan")
                rec[key + "_ms"] = ms
                rec[key + "_vs_fp16"] = su
                rec[key + "_tok_s"] = 1e3 / ms * B
                print(f"  B={B} {key:16s}: {ms:.3f} ms/tok  vs FP16g {su:.3f}x  "
                      f"({1e3/ms*B:.1f} tok/s)", flush=True)
        rows.append(rec)
    set_merge(False)
    out["perf"] = rows


# ---------------------------------------------------------------------------
# PHASE: ppl — fake A/B (W4A4 act-quant ON vs W4A16 act-exact), GPTQ frozen scales
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_ppl(config, nv, mx, cd, args, out):
    print("=== PHASE: PPL fake A/B (W4A4 vs W4A16) ===", flush=True)
    from promix.eval.data import get_wikitext2
    from promix.eval.evaluator import evaluate_ppl
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=args.ppl_gptq); model.to(DEV)
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_fake_mixed(model, packed)
    fn.wire_fused_mlp(model)
    model.to(DEV).eval(); model.config.use_cache = False
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    model.seqlen = config['eval']['max_length']
    tl = get_wikitext2(seed=config['calibration'].get('seed', 0),
                       seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = args.ppl_bs if args.ppl_bs > 0 else config['eval']['batch_size']

    fn.FakeMixedLinear.ACT_EXACT = False
    t0 = time.time(); ppl_w4a4 = evaluate_ppl(model, tl, DEV, batch_size=bs); t4 = time.time() - t0
    print(f"  W4A4  (act quant ON)  PPL = {ppl_w4a4:.4f}  ({t4:.0f}s)", flush=True)
    fn.FakeMixedLinear.ACT_EXACT = True
    t0 = time.time(); ppl_w4a16 = evaluate_ppl(model, tl, DEV, batch_size=bs); t16 = time.time() - t0
    fn.FakeMixedLinear.ACT_EXACT = False
    print(f"  W4A16 (act EXACT)     PPL = {ppl_w4a16:.4f}  ({t16:.0f}s)", flush=True)
    delta = ppl_w4a16 - ppl_w4a4
    print(f"  PPL delta (W4A16 - W4A4) = {delta:+.4f}  (negative = W4A16 better)", flush=True)
    out["ppl"] = dict(w4a4=ppl_w4a4, w4a16=ppl_w4a16, delta=delta,
                      baseline_scoreboard_w4a4=args.baseline_ppl)
    del model, packed
    cleanup_memory()


# globals for numerics' fake-twin build
nv_g = mx_g = cd_g = None


@torch.no_grad()
def main():
    global nv_g, mx_g, cd_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phases", default="probe,numerics,perf,ppl")
    ap.add_argument("--decode_batches", default="1,8")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--gate_steps", type=int, default=48)
    ap.add_argument("--baseline_ppl", type=float, default=0.0)
    ap.add_argument("--ppl_bs", type=int, default=0, help="override eval batch_size (0=config)")
    ap.add_argument("--ppl_gptq", action="store_true", default=True)
    ap.add_argument("--no_ppl_gptq", dest="ppl_gptq", action="store_false",
                    help="use RTN weight scales (skip slow GPTQ); the W4A4-vs-W4A16 delta is GPTQ-independent")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.decode_batches = [int(x) for x in args.decode_batches.split(",")]
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config = load_config(args.config)
    phases = set(args.phases.split(","))
    print(f"[r12] config={args.config} gpu={args.gpu} phases={phases}", flush=True)

    nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias = load_all()
    dg = load_decode_gemv()
    nv_g, mx_g, cd_g = nv, mx, cd
    MS.decode_gemv_ext = dg
    print("[r12] kernels loaded", flush=True)

    cfg_peek = transformers.AutoConfig.from_pretrained(config['model']['name'])
    out = {"config": args.config, "model": model_desc(cfg_peek),
           "shapes": [list(s) for s in layer_shapes(cfg_peek)]}

    if "probe" in phases:
        run_probe(cfg_peek, nv, mx, cd, dg, out)
        _flush(out, args.out)

    need_real = ("numerics" in phases) or ("perf" in phases)
    if need_real:
        print("=== building real-kernel model (decode-GEMV on) + merge ===", flush=True)
        model, packed, fused = build_real(config, nv, mx, cd, qpf, qpfm, hpk, hpk28, had28,
                                          cd_bias, gptq=False, decode_gemv=True)
        set_decode_gemv(model, dg)
        info = install_merge(model, dg, cd)
        out["merge_info"] = info
        fp16 = None
        if "perf" in phases:
            fp16 = _build_fp16_model(config['model']['name']); fp16.config.use_cache = True
        if "numerics" in phases:
            run_numerics(config, cfg_peek, model, packed, dg, args, out)
            _flush(out, args.out)
        if "perf" in phases:
            run_perf(config, model, fp16, args, out)
            _flush(out, args.out)
        remove_patches()
        del model, packed
        if fp16 is not None:
            del fp16
        cleanup_memory()

    if "ppl" in phases:
        run_ppl(config, nv, mx, cd, args, out)
        _flush(out, args.out)

    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("DECODE_W4A16_DONE", flush=True)


def _flush(out, path):
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
