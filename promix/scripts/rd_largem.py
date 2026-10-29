"""rd-largem: how big do the FORMAT and ALLOCATION levers get at serving-scale
effective M (= batch x seq)?

All prior W4A4 perf numbers were at M=2048 (bs=1 single-seq prefill) — the
least compute-bound point, where the FP4/FP8 format lever is ~1.3x GEMM / small
e2e and the measured-MSE allocation is ~1% e2e. W4A4 peers measure at much
larger effective M (QuaRot bs<=64, Atom batch 2-256, QServe ISL1024 continuous
batching). This harness grows M via BATCH (seq fixed at 2048, batch 1..16 ->
M_eff 2048..32768), which makes the 7 quantized-Linear GEMMs compute-bound while
keeping attention's per-sequence share fixed, and measures at each M_eff:

  1. FORMAT lever  : pure-NVFP4 endpoint (K_high=0 all 7 targets, single-phase
     fused path) vs the MXFP8 endpoint (pure K_low=0 attempted; falls back to
     near-pure K_high=K-64 if the kernel can't build it) —
       (a) e2e prefill ratio t_FP8/t_FP4 (interleaved A/B, paired per-iter
           ratios, contention-robust), and
       (b) the isolated in-model Linear-GEMM ratio (cudaEvent brackets around
           every MixedQuantLinear._gemm, summed per forward).
  2. ALLOCATION lever: measured-MSE@1/8 vs uniform-1/8, same interleaved A/B
     e2e protocol (rd_relat).
  3. GEMM share of prefill: gemm_ms / whole_ms on the uniform-1/8 production
     config (and on both format endpoints) at each M_eff.

Reuses rd_perf (build_real / realized_khigh / load_kernels), rd_perfpuzzle
(MixedQuantLinear cudaEvent instrumentation), rd_sweep (MODELS / resolve_paths /
_clamp_chosen) and rd_validity (build_base / collect_stats / measured_alloc).
Models are kept pairwise-resident (never more than 2), W_deq freed (8B memory).
OOM at a batch point is recorded and the sweep stops there (the cap).

    CUDA_VISIBLE_DEVICES=2 python3 promix/scripts/rd_largem.py \
        --model 3b --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
        --batches 1,2,4,8,16 --out /home/luofan/RD_LARGEM_3b.json
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transformers

import rd_validity as rdv
import rd_perf
import rd_perfpuzzle as rpp          # import installs the _gemm/_pack cudaEvent hooks
from rd_relat import _free_wdeq, _med
from rd_sweep import MODELS, resolve_paths, bits_per_weight, _clamp_chosen
from promix.eval import ptq as ptqmod
from promix.quantize import full_network as fn
from promix.quantize.kv_quant import setup_k_quant
from promix.utils import cleanup_memory, DEV


def _ev():
    return torch.cuda.Event(enable_timing=True)


# ---------------------------------------------------------------------------
# Forced per-Linear K_high build over ALL 7 targets (incl. o_proj) — the format
# endpoints. khfn(K) -> K_high. Mirrors rd_perf.build_real otherwise.
# ---------------------------------------------------------------------------
def build_forced(config, khfn, nv, mx, cd, qpf, dg):
    model = rdv.build_base(config, True)      # r1_off (same protocol as rd_perf)
    model.to(DEV)
    override = {}
    for name, short, w in fn.iter_target_wrappers(model):
        K = int(w.module.weight.shape[1])
        kh = int(khfn(K))
        override[name] = kh
        w.module._edriven_khigh = kh
    model._rd_khigh_override = override
    ptqmod.configure_quantizers(model, config)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=0.125,
                                 verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=dg)
    model.to(DEV).eval()
    model.config.use_cache = False
    _free_wdeq(packed)
    cleanup_memory()
    return model, packed


# ---------------------------------------------------------------------------
# Batched interleaved A/B prefill (rd_relat protocol, ids (B, S)).
# ---------------------------------------------------------------------------
@torch.no_grad()
def interleaved_prefill_b(mA, mB, B, S, vocab, iters, warmup):
    ids = torch.randint(0, vocab, (B, S), device=DEV)
    mA.config.use_cache = False
    mB.config.use_cache = False

    def t(m):
        s, e = _ev(), _ev()
        torch.cuda.synchronize()
        s.record()
        m(input_ids=ids, use_cache=False)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    for _ in range(warmup):
        t(mA); t(mB)
    a, b = [], []
    for _ in range(iters):
        a.append(t(mA)); b.append(t(mB))
    del ids
    return a, b


def paired_row(a, b):
    """A/B paired-ratio summary: ratio[i] = a[i]/b[i] (same instantaneous box
    load). Returns medians/mins both directions."""
    pairs = sorted(x / y for x, y in zip(a, b))
    return dict(a_min=min(a), a_med=_med(a), b_min=min(b), b_med=_med(b),
                ratio_pair_med=_med(pairs), ratio_pair_min=pairs[0],
                ratio_pair_max=pairs[-1],
                ratio_min=min(a) / min(b), ratio_med=_med(a) / _med(b))


# ---------------------------------------------------------------------------
# Batched decomposed prefill (rd_perfpuzzle instrumentation, ids (B, S)):
# clean whole + instrumented whole/gemm/pack per forward.
# ---------------------------------------------------------------------------
@torch.no_grad()
def decomposed_prefill_b(model, B, S, vocab, iters, warmup):
    ids = torch.randint(0, vocab, (B, S), device=DEV)
    model.config.use_cache = False

    def fwd():
        return model(input_ids=ids, use_cache=False).logits

    rpp._EV["on"] = False
    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()

    clean = []
    for _ in range(iters):
        ws, we = _ev(), _ev()
        ws.record(); fwd(); we.record(); torch.cuda.synchronize()
        clean.append(ws.elapsed_time(we))

    whole, pack, gemm, nlin = [], [], [], []
    rpp._EV["on"] = True
    for _ in range(iters):
        rpp._reset_ev()
        ws, we = _ev(), _ev()
        ws.record(); fwd(); we.record(); torch.cuda.synchronize()
        whole.append(ws.elapsed_time(we))
        pack.append(rpp._sum_ev("pack"))
        gemm.append(rpp._sum_ev("gemm"))
        nlin.append(len(rpp._EV["gemm"]))
    rpp._EV["on"] = False
    del ids

    w, p, g, cw = _med(whole), _med(pack), _med(gemm), _med(clean)
    return dict(clean_whole_ms=cw, whole_ms=w, pack_ms=p, gemm_ms=g,
                rest_ms=w - p - g, n_linear_calls=int(_med(nlin)),
                gemm_share_instr=g / w, linear_share_instr=(p + g) / w,
                gemm_share_clean=g / cw)


def _iters_for(B, base):
    return max(5, int(round(base / (B ** 0.5))))


def khigh_stats(model):
    kh = rd_perf.realized_khigh(model)
    tot_hi = sum(h for h, _ in kh.values())
    tot_k = sum(K for _, K in kh.values())
    return dict(avg_fp8=tot_hi / max(tot_k, 1), khigh=kh)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=13)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--budget", type=float, default=0.125)
    ap.add_argument("--skip_format", action="store_true")
    ap.add_argument("--skip_alloc", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/luofan/RD_LARGEM_{args.model}.json"
    batches = [int(x) for x in args.batches.split(",")]

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_paths(config, args.model)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    print(f"START rd_largem model={args.model} name={config['model']['name']} "
          f"batches={batches} seq={args.seq} iters={args.iters}", flush=True)

    print("Loading SM120 kernels (JIT may take minutes)...", flush=True)
    nv, mx, cd, qpf = rd_perf.load_kernels()
    dg = rd_perf.load_decode_gemv()
    print("kernels loaded", flush=True)

    out = {"meta": dict(model=args.model, name=config["model"]["name"],
                        batches=batches, seq=args.seq, iters=args.iters,
                        budget=args.budget, protocol=(
                            "interleaved A/B paired ratios (format: FP8-endpoint vs "
                            "FP4-endpoint; alloc: uniform-1/8 vs measured); GEMM "
                            "split via cudaEvent hooks on MixedQuantLinear"))}

    def dump():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # ======================================================================
    # PHASE 1 — FORMAT lever: FP4 endpoint vs FP8 endpoint
    # ======================================================================
    if not args.skip_format:
        print("\n=== PHASE 1: FORMAT lever ===", flush=True)
        print("Building pure-NVFP4 endpoint (K_high=0 all 7 targets)...", flush=True)
        f4, f4p = build_forced(config, lambda K: 0, nv, mx, cd, qpf, dg)
        vocab = f4.config.vocab_size
        print(f"  F4: {khigh_stats(f4)['avg_fp8']=:}", flush=True)

        f8 = f8p = f8_mode = None
        for mode, khfn in (("pure_K_low=0", lambda K: K),
                           ("near_pure_K_low=64", lambda K: K - 64)):
            cand = candp = None
            try:
                cand, candp = build_forced(config, khfn, nv, mx, cd, qpf, dg)
                # smoke: one tiny forward must not throw
                interleaved_prefill_b(cand, f4, 1, 64, vocab, 1, 0)
                f8, f8p, f8_mode = cand, candp, mode
                break
            except Exception as ex:
                print(f"  FP8 endpoint build '{mode}' failed: "
                      f"{type(ex).__name__}: {ex}", flush=True)
                del cand, candp
                cleanup_memory()
        assert f8 is not None, "no FP8 endpoint buildable"
        cand = candp = None   # drop the loop aliases of f8/f8p (refcount leak)
        s8 = khigh_stats(f8)
        print(f"  F8 endpoint mode={f8_mode} avg_fp8={s8['avg_fp8']:.4f}", flush=True)
        fmt = {"f8_mode": f8_mode, "f8_avg_fp8": s8["avg_fp8"], "rows": {}}
        out["format"] = fmt
        dump()

        for B in batches:
            M = B * args.seq
            it = _iters_for(B, args.iters)
            print(f"--- format B={B} M_eff={M} iters={it} ---", flush=True)
            try:
                t0 = time.time()
                interleaved_prefill_b(f8, f4, B, args.seq, vocab, 1, 0)  # alloc warm
                a8, a4 = interleaved_prefill_b(f8, f4, B, args.seq, vocab, it,
                                               args.warmup)
                e2e = paired_row(a8, a4)   # ratio = t_FP8 / t_FP4 (>1 => FP4 wins)
                d4 = decomposed_prefill_b(f4, B, args.seq, vocab,
                                          max(5, it - 3), 1)
                d8 = decomposed_prefill_b(f8, B, args.seq, vocab,
                                          max(5, it - 3), 1)
                row = dict(batch=B, M_eff=M, e2e_fp8_over_fp4=e2e,
                           f4_decomp=d4, f8_decomp=d8,
                           gemm_ratio_fp8_over_fp4=d8["gemm_ms"] / d4["gemm_ms"],
                           linear_ratio_fp8_over_fp4=(
                               (d8["gemm_ms"] + d8["pack_ms"])
                               / (d4["gemm_ms"] + d4["pack_ms"])),
                           secs=round(time.time() - t0, 1))
                print(f"  e2e t_FP8/t_FP4 paired-med={e2e['ratio_pair_med']:.4f} "
                      f"(min-of-each {e2e['ratio_min']:.4f}) | GEMM ratio "
                      f"{row['gemm_ratio_fp8_over_fp4']:.4f} | F4 gemm_share "
                      f"{d4['gemm_share_instr']*100:.1f}% F8 "
                      f"{d8['gemm_share_instr']*100:.1f}% ({row['secs']}s)",
                      flush=True)
                fmt["rows"][str(B)] = row
            except torch.cuda.OutOfMemoryError as ex:
                print(f"  OOM at B={B} (M_eff={M}): {ex}", flush=True)
                fmt["rows"][str(B)] = dict(batch=B, M_eff=M, oom=True)
                cleanup_memory()
                dump()
                break
            dump()
        del f4, f4p, f8, f8p
        cleanup_memory()

    # ======================================================================
    # PHASE 2 — ALLOCATION lever + GEMM share on the production uniform-1/8
    # ======================================================================
    if not args.skip_alloc:
        print("\n=== PHASE 2: ALLOCATION lever ===", flush=True)
        tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
        from promix.eval.data import get_wikitext2
        calib = get_wikitext2(nsamples=args.calib_samples,
                              seed=config["calibration"].get("seed", 0),
                              seqlen=2048, tokenizer=tok, eval_mode=False)
        cmodel = rdv.build_base(config, True)
        cmodel.to(DEV)
        stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
        del cmodel
        cleanup_memory()
        total_rings = sum(stats[n]["nr"] for n in stats)
        br = int(round(args.budget * total_rings))
        chosen = _clamp_chosen(rdv.measured_alloc(stats, br, total_rings), stats)
        print(f"  alloc total_rings={total_rings} target={br}", flush=True)

        print("Building uniform-1/8 + measured models...", flush=True)
        base, basep = rd_perf.build_real(config, True, None, nv, mx, cd, qpf, dg)
        _free_wdeq(basep)
        meas, measp = rd_perf.build_real(config, True, chosen, nv, mx, cd, qpf, dg)
        _free_wdeq(measp)
        cleanup_memory()
        vocab = base.config.vocab_size
        km = khigh_stats(meas)
        bpw, _ = bits_per_weight(km["khigh"])
        alloc = {"measured_avg_fp8": km["avg_fp8"], "bits_per_weight": bpw,
                 "uniform_avg_fp8": khigh_stats(base)["avg_fp8"], "rows": {}}
        out["alloc"] = alloc
        dump()

        for B in batches:
            M = B * args.seq
            it = _iters_for(B, args.iters)
            print(f"--- alloc B={B} M_eff={M} iters={it} ---", flush=True)
            try:
                t0 = time.time()
                interleaved_prefill_b(base, meas, B, args.seq, vocab, 1, 0)
                au, am = interleaved_prefill_b(base, meas, B, args.seq, vocab, it,
                                               args.warmup)
                e2e = paired_row(au, am)   # ratio = t_uniform / t_measured
                du = decomposed_prefill_b(base, B, args.seq, vocab,
                                          max(5, it - 3), 1)
                row = dict(batch=B, M_eff=M, e2e_uniform_over_measured=e2e,
                           uniform_decomp=du, secs=round(time.time() - t0, 1))
                print(f"  e2e t_unif/t_meas paired-med={e2e['ratio_pair_med']:.4f} "
                      f"(min-of-each {e2e['ratio_min']:.4f}) | uniform GEMM share "
                      f"{du['gemm_share_instr']*100:.1f}% (clean "
                      f"{du['gemm_share_clean']*100:.1f}%) linear "
                      f"{du['linear_share_instr']*100:.1f}% whole "
                      f"{du['clean_whole_ms']:.1f}ms ({row['secs']}s)", flush=True)
                alloc["rows"][str(B)] = row
            except torch.cuda.OutOfMemoryError as ex:
                print(f"  OOM at B={B} (M_eff={M}): {ex}", flush=True)
                alloc["rows"][str(B)] = dict(batch=B, M_eff=M, oom=True)
                cleanup_memory()
                dump()
                break
            dump()
        del base, basep, meas, measp
        cleanup_memory()

    dump()
    print(f"WROTE {args.out}", flush=True)
    print("RD_LARGEM_DONE", flush=True)


if __name__ == "__main__":
    main()
