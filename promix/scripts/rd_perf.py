"""Task 6 (perf) — END-TO-END latency payoff of the rate-distortion allocation.

The "performance payoff" curve: push the average FP8 ring-budget DOWN (more of
the network served in NVFP4 instead of MXFP8) and measure how much REAL
end-to-end speedup it buys on the Pro 6000 (SM120), at iso-accuracy.

Two measurements per budget b in {1/8, 1/12, 1/16, 1/24}, R1-off (pure PCA,
identity main rotation — the cheapest regime, same protocol as rd_validity):

  (a) ACCURACY: wikitext-2 PPL of the measured-MSE allocation at budget b,
      via the EXACT rd_validity path (build_eval + GPTQ + eval_ppl). This is the
      accuracy lever — WHICH 64-rings route to FP8.

  (b) LATENCY: real packed NVFP4+MXFP8 fused-kernel model (full_network.
      wire_real_kernel) built with the SAME per-Linear K_high allocation, driven
      through configure_quantizers' opt-in `model._rd_khigh_override` seam so
      pack_all_targets packs each Linear at its allocated K_high. We measure the
      FULL model forward (not an isolated GEMM) with cudaEvent timing, warmup +
      median over iters, for BOTH:
        * prefill (TTFT): one long-prompt forward (M = --prefill_M), compute-bound.
        * decode  (TPOT): per-token KV-cache forward (M=1), bandwidth-bound,
          using the decode-GEMV weight-streaming path (the deployed decode path).

The headline number: at the SMALLEST budget whose measured-MSE PPL is still
<= uniform-1/8's PPL (iso-accuracy within R1-off), the e2e speedup vs the
uniform-1/8 real-kernel baseline. R1-off PPL is slightly worse than the shipped
R-on recipe, so THIS CURVE IS THE R1-OFF LOWER BOUND on the payoff; the R-on
version (built in parallel) will shift accuracy and move the iso-accuracy knee.

If the fused kernel rejects an edge allocation (a Linear with K_high==0 or
K_low==0), the LATENCY row for that budget falls back to the ring-aligned
UNIFORM-at-b allocation (same average budget, no empty segment) and is flagged
`latency_uniform_fallback=true`; the e2e cost is driven by the average budget,
so the uniform-at-b latency is a faithful stand-in for the budget→speed axis.
Latency is weight-value independent, so the latency build skips GPTQ.

Remote-only (Pro 6000, inside luofan-plaquant). Line-buffered + JSON to disk.
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
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from transformers import DynamicCache

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.quantize import full_network as fn
from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper
from promix.quantize.kv_quant import setup_k_quant

# Reuse rd_validity's proven R1-off build + allocation machinery verbatim.
import rd_validity as rdv
from rd_validity import RING, ALLOC, _is_alloc_target

# Reuse the validated real-kernel loaders.
from promix.eval.full_network_realkernel import load_kernels
from promix.eval.decode_throughput import load_decode_gemv, _build_fp16_model


# ---------------------------------------------------------------------------
# cudaEvent timing
# ---------------------------------------------------------------------------
def cuda_time_median_ms(fn_, iters=20, warmup=5):
    """Median per-call latency (ms) via cudaEvent, after warmup."""
    for _ in range(warmup):
        fn_()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn_()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    n = len(times)
    return times[n // 2] if n % 2 else 0.5 * (times[n // 2 - 1] + times[n // 2])


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------
# Real-kernel model build for a given per-Linear allocation (NO GPTQ).
# Mirrors rd_validity.build_eval's reorder+override block, then swaps the
# fake-quant module for the real packed fused kernel (pack_all_targets reads the
# configured per-Linear K_high, so the pack is per-layer-allocated).
# ---------------------------------------------------------------------------
def _apply_reorder_override(model, chosen):
    """chosen = {name: set(fp8 ring idx)} over the ALLOC Linears. Reorders each
    Linear's weight columns so the chosen rings are the LAST K_high, tags the
    wrapper for the matching activation reorder, and records the per-Linear
    K_high override read by configure_quantizers + pack_all_targets."""
    override = {}
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        if not _is_alloc_target(name):
            continue
        S = chosen.get(name, set())
        nr = w.module.weight.shape[1] // RING
        non_s = [r for r in range(nr) if r not in S]
        s_lst = [r for r in range(nr) if r in S]
        ring_order = non_s + s_lst
        order = torch.cat([torch.arange(r * RING, (r + 1) * RING)
                           for r in ring_order]).to(w.module.weight.device)
        W = w.module.weight.data
        w.module.weight.data = W[:, order].contiguous()
        w._column_order = order
        w.reorder_before_quant = True
        kh = len(s_lst) * RING
        w.module._edriven_khigh = int(kh)
        override[name] = int(kh)
    model._rd_khigh_override = override
    return override


def build_real(config, r1_off, chosen, nv, mx, cd, qpf, dg):
    """Real packed fused-kernel model for latency. chosen=None -> shipped uniform
    1/8 (no override, byte-identical). Else per-Linear measured allocation."""
    model = rdv.build_base(config, r1_off)
    model.to(DEV)
    if chosen is not None:
        _apply_reorder_override(model, chosen)
    ptqmod.configure_quantizers(model, config)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    # pack_all_targets reads each wrapper.quantizer.high_bits_length (the per-
    # Linear K_high just configured) via _khigh_for -> per-layer allocated pack.
    # Weights must be on CUDA here (the SF prepack kernels require CUDA uint8).
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=0.125,
                                 verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=dg)
    model.to(DEV).eval()
    model.config.use_cache = True
    return model, packed


def realized_khigh(model):
    """{name: [K_high, K]} over the 7 target Linears, read from the wired kernel."""
    out = {}
    for name, short, w in fn.iter_target_wrappers(model):
        out[name] = [int(getattr(w.module, "K_high", 0)),
                     int(getattr(w.module, "K", 0))]
    return out


def _ring_aligned_eighth(K, ring=64):
    """round(1/8 * K) snapped to the ring grid — the shipped uniform K_high."""
    return int(round(0.125 * K / ring)) * ring


# ---------------------------------------------------------------------------
# Latency measurement: full-model prefill (TTFT) + decode (TPOT).
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_prefill_ms(model, M, vocab, iters, warmup):
    ids = torch.randint(0, vocab, (1, M), device=DEV)
    model.config.use_cache = False

    def fwd():
        return model(input_ids=ids, use_cache=False).logits

    return cuda_time_median_ms(fwd, iters=iters, warmup=warmup)


@torch.no_grad()
def measure_decode_ms(model, prompt_ids, gen, warmup_steps=8):
    """Per-token (TPOT) median over steady-state KV-cache decode steps, cudaEvent."""
    model.config.use_cache = True

    def one_run(n):
        cache = DynamicCache()
        out = model(input_ids=prompt_ids, use_cache=True, past_key_values=cache)
        past = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(-1)
        step_ms = []
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            out = model(input_ids=cur, use_cache=True, past_key_values=past)
            e.record()
            torch.cuda.synchronize()
            step_ms.append(s.elapsed_time(e))
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return step_ms

    one_run(min(warmup_steps, max(gen // 2, 4)))   # warmup (alloc growth, JIT)
    step_ms = one_run(gen)
    steady = step_ms[4:] if len(step_ms) > 8 else step_ms
    return _median(steady), min(steady)


# ---------------------------------------------------------------------------
# Edge-aware latency for one allocation (measured; uniform-at-b fallback).
# ---------------------------------------------------------------------------
def _aligned_uniform_chosen(stats, b):
    """Uniform-at-b: every ALLOC Linear gets round(b*nr) FP8 rings (>=1), as the
    highest indices. Returns {name: set(ring idx)} with NO empty/full segment."""
    chosen = {}
    for name in stats:
        nr = stats[name]["nr"]
        n = max(1, min(nr - 1, int(round(b * nr))))
        chosen[name] = set(range(nr - n, nr))
    return chosen


def measure_latency_for(config, r1_off, chosen, stats, b, nv, mx, cd, qpf, dg,
                        args):
    """Build+measure the real-kernel latency for `chosen`; on a fused-kernel edge
    failure (K_high==0/K_low==0), retry with ring-aligned uniform-at-b."""
    vocab = None
    tok, prompt = rdv_tokenize(config, args.decode_prompt)
    for attempt, (ch, fb) in enumerate((
            (chosen, False),
            (_aligned_uniform_chosen(stats, b) if chosen is not None else None, True))):
        try:
            model, packed = build_real(config, r1_off, ch, nv, mx, cd, qpf, dg)
            vocab = model.config.vocab_size
            kh = realized_khigh(model)
            # smoke: one prefill + one decode step must not throw.
            _ = measure_prefill_ms(model, 64, vocab, iters=1, warmup=1)
            _ = measure_decode_ms(model, prompt, 2, warmup_steps=1)
            pre = measure_prefill_ms(model, args.prefill_M, vocab,
                                     iters=args.iters, warmup=args.warmup)
            dec_med, dec_min = measure_decode_ms(model, prompt, args.decode_gen)
            del model, packed
            cleanup_memory()
            return dict(prefill_ms=pre, decode_ms=dec_med, decode_min_ms=dec_min,
                        khigh=kh, uniform_fallback=fb)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"  [latency] attempt {attempt} failed ({type(ex).__name__}: {ex}); "
                  f"{'falling back to uniform-at-b' if attempt == 0 else 'giving up'}",
                  flush=True)
            cleanup_memory()
            if attempt == 1 or chosen is None:
                return dict(error=repr(ex))
    return dict(error="unreachable")


def rdv_tokenize(config, P):
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2
    loader = get_wikitext2(seed=0, seqlen=max(P, 8), tokenizer=tok, eval_mode=True)
    ids = loader.input_ids[:, :P].to(DEV)
    return tok, ids


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r1_off", action="store_true")
    ap.add_argument("--budgets", default="0.125,0.08333,0.0625,0.041667")
    ap.add_argument("--prefill_M", type=int, default=2048)
    ap.add_argument("--decode_prompt", type=int, default=256)
    ap.add_argument("--decode_gen", type=int, default=48)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--skip_ppl", action="store_true",
                    help="measure latency only (debug)")
    ap.add_argument("--out", default="/home/luofan/RD_PERF.json")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rdv.resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    print(f"START rd_perf r1_off={args.r1_off} budgets={budgets} "
          f"prefill_M={args.prefill_M} decode(prompt={args.decode_prompt},"
          f"gen={args.decode_gen}) model={config['model']['name']}", flush=True)

    print("Loading SM120 kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()
    print("kernels loaded", flush=True)

    results = {"meta": dict(r1_off=args.r1_off, budgets=budgets,
                            prefill_M=args.prefill_M,
                            decode_prompt=args.decode_prompt,
                            decode_gen=args.decode_gen, iters=args.iters,
                            model=config["model"]["name"],
                            basis=os.path.basename(config["paths"]["basis"]))}

    # --- collect R1-off rotated activation stats ONCE ------------------------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, args.r1_off)
    cmodel.to(DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()
    total_rings = sum(stats[n]["nr"] for n in stats)
    print(f"ALLOC units={len(stats)} total_rings={total_rings}", flush=True)
    results["meta"]["total_rings"] = total_rings

    # --- uniform-1/8 baseline (PPL + real-kernel latency) --------------------
    rows = {}
    print("=== BASELINE uniform-1/8 ===", flush=True)
    base = {}
    if not args.skip_ppl:
        t0 = time.time()
        m = rdv.build_eval(config, args.r1_off, None)
        frac, hi, totc = rdv.avg_fp8_fraction(m)
        base["ppl"] = rdv.eval_ppl(m, config)
        base["avg_fp8"] = frac
        del m
        cleanup_memory()
        print(f"  uniform-1/8 PPL={base['ppl']:.4f} avg_fp8={frac:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    lat = measure_latency_for(config, args.r1_off, None, stats, 0.125,
                              nv, mx, cd, qpf, dg, args)
    base["latency"] = lat
    # ZERO-REGRESSION check: with rd_alloc disabled (no override) the uniform
    # real-kernel K_high per Linear must equal the ring-aligned 1/8 split.
    if "khigh" in lat:
        mism = {n: kv for n, kv in lat["khigh"].items()
                if kv[0] != _ring_aligned_eighth(kv[1])}
        base["zero_regression_uniform_ok"] = (len(mism) == 0)
        base["zero_regression_mismatches"] = mism
        print(f"  ZERO-REGRESSION uniform K_high==1/8 split: "
              f"{'OK' if not mism else f'MISMATCH {mism}'}", flush=True)
    print(f"  uniform-1/8 latency: prefill={lat.get('prefill_ms')} ms "
          f"decode={lat.get('decode_ms')} ms", flush=True)
    results["baseline_uniform_1_8"] = base

    # --- per-budget measured-MSE allocation ----------------------------------
    for b in budgets:
        budget_rings = int(round(b * total_rings))
        chosen = rdv.measured_alloc(stats, budget_rings, total_rings)
        n_fp8 = sum(len(v) for v in chosen.values())
        tag = f"b={b:.5f}"
        print(f"=== BUDGET {tag}  target_rings={budget_rings} "
              f"(realized {n_fp8}) ===", flush=True)
        row = dict(budget=b, budget_rings=budget_rings, n_fp8_rings=n_fp8,
                   avg_fp8_rings=n_fp8 / total_rings)

        if not args.skip_ppl:
            t0 = time.time()
            m = rdv.build_eval(config, args.r1_off, chosen)
            frac, hi, totc = rdv.avg_fp8_fraction(m)
            row["ppl"] = rdv.eval_ppl(m, config)
            row["avg_fp8"] = frac
            del m
            cleanup_memory()
            print(f"  PPL={row['ppl']:.4f} avg_fp8={frac:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time()
        lat = measure_latency_for(config, args.r1_off, chosen, stats, b,
                                  nv, mx, cd, qpf, dg, args)
        row["latency"] = lat
        if "error" not in lat:
            if base.get("latency", {}).get("prefill_ms"):
                row["prefill_speedup"] = (base["latency"]["prefill_ms"]
                                          / lat["prefill_ms"])
            if base.get("latency", {}).get("decode_ms"):
                row["decode_speedup"] = (base["latency"]["decode_ms"]
                                         / lat["decode_ms"])
            print(f"  latency: prefill={lat['prefill_ms']:.3f} ms "
                  f"(x{row.get('prefill_speedup', float('nan')):.3f}) "
                  f"decode={lat['decode_ms']:.3f} ms "
                  f"(x{row.get('decode_speedup', float('nan')):.3f}) "
                  f"fallback={lat.get('uniform_fallback')} ({time.time()-t0:.0f}s)",
                  flush=True)
        else:
            print(f"  latency FAILED: {lat['error']}", flush=True)
        rows[f"{b:.5f}"] = row
        with open(args.out, "w") as f:
            json.dump({**results, "budgets": rows}, f, indent=2, default=str)

    results["budgets"] = rows

    # --- iso-accuracy payoff -------------------------------------------------
    if not args.skip_ppl and "ppl" in base:
        ref_ppl = base["ppl"]
        feasible = [(b, rows[f"{b:.5f}"]) for b in budgets
                    if "ppl" in rows[f"{b:.5f}"]
                    and rows[f"{b:.5f}"]["ppl"] <= ref_ppl + 1e-9]
        feasible.sort(key=lambda t: t[0])  # smallest budget first
        if feasible:
            b_star, r_star = feasible[0]
            payoff = dict(budget=b_star, ppl=r_star.get("ppl"),
                          ref_uniform_ppl=ref_ppl,
                          prefill_speedup=r_star.get("prefill_speedup"),
                          decode_speedup=r_star.get("decode_speedup"))
            results["iso_accuracy_payoff"] = payoff
            print(f"ISO-ACCURACY PAYOFF: smallest budget b={b_star:.5f} with "
                  f"PPL={r_star.get('ppl'):.4f} <= uniform {ref_ppl:.4f} -> "
                  f"prefill x{r_star.get('prefill_speedup')} "
                  f"decode x{r_star.get('decode_speedup')}", flush=True)
        else:
            results["iso_accuracy_payoff"] = dict(
                note="no budget met iso-accuracy", ref_uniform_ppl=ref_ppl)
            print("ISO-ACCURACY PAYOFF: no budget met uniform-1/8 PPL", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"WROTE {args.out}", flush=True)
    print("RD_PERF_DONE", flush=True)


if __name__ == "__main__":
    main()
