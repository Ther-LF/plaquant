"""rd-relat: contention-robust e2e latency of the measured-MSE allocation vs the
uniform-1/8 baseline, WITH the K_high==0 kernel fix (single-phase pure-NVFP4).

The shared Pro-6000 box has volatile CPU/GPU contention that corrupts ABSOLUTE
e2e latency (the prefill/decode host-dispatch path is sensitive to it). So we
time the uniform-1/8 baseline and each budget point INTERLEAVED in ONE process
(A,B,A,B,...), microseconds apart, and report the ratio from the per-model MIN
(cleanest sample) and MEDIAN. Interleaving makes the A/B ratio robust to box
load even when absolutes drift.

Reuses rd_perf (build_real / realized_khigh / rdv_tokenize / load_kernels) and
rd_sweep (MODELS / resolve_paths / measured_alloc / _clamp_chosen / bpw). The
baseline model is built once and kept; each budget's measured model is built,
A/B-timed against it, then freed.

    CUDA_VISIBLE_DEVICES=0 python3 promix/scripts/rd_relat.py \
        --model 3b --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
        --budgets 0,0.03125,0.0625,0.083333,0.125 --out /home/luofan/RD_RELAT2_3b.json
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
from transformers import DynamicCache

import rd_validity as rdv
import rd_perf
from rd_sweep import MODELS, resolve_paths, bits_per_weight, _clamp_chosen
from rd_validity import ALLOC
from promix.eval import ptq as ptqmod
from promix.utils import cleanup_memory, DEV


def _med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _ev():
    return torch.cuda.Event(enable_timing=True)


@torch.no_grad()
def interleaved_prefill(mA, mB, M, vocab, iters, warmup):
    ids = torch.randint(0, vocab, (1, M), device=DEV)
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
    return a, b


@torch.no_grad()
def interleaved_decode(mA, mB, prompt_ids, gen, warmup=6):
    def setup(m):
        m.config.use_cache = True
        cache = DynamicCache()
        out = m(input_ids=prompt_ids, use_cache=True, past_key_values=cache)
        return out.past_key_values, out.logits[:, -1:, :].argmax(-1)

    pA, cA = setup(mA)
    pB, cB = setup(mB)

    def step(m, past, cur):
        s, e = _ev(), _ev()
        torch.cuda.synchronize()
        s.record()
        out = m(input_ids=cur, use_cache=True, past_key_values=past)
        e.record()
        torch.cuda.synchronize()
        return (s.elapsed_time(e), out.past_key_values,
                out.logits[:, -1, :].argmax(-1, keepdim=True))

    for _ in range(warmup):
        _, pA, cA = step(mA, pA, cA)
        _, pB, cB = step(mB, pB, cB)
    a, b = [], []
    for _ in range(gen):
        ta, pA, cA = step(mA, pA, cA)
        tb, pB, cB = step(mB, pB, cB)
        a.append(ta); b.append(tb)
    return a, b


def khigh_summary(model):
    """(n_khigh0, n_targets, realized_avg_fp8) over the 7 target Linears,
    counting K_high==0 only among the 6 REALLOCATED Linears (o_proj pinned)."""
    kh = rd_perf.realized_khigh(model)   # {name: [K_high, K]}
    n0 = sum(1 for n, (h, K) in kh.items()
             if h == 0 and n.split(".")[-1] != "o_proj")
    nt = sum(1 for n in kh if n.split(".")[-1] != "o_proj")
    tot_hi = sum(h for h, _ in kh.values())
    tot_k = sum(K for _, K in kh.values())
    return n0, nt, (tot_hi / max(tot_k, 1)), kh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--budgets", default="0,0.03125,0.0625,0.083333,0.125")
    ap.add_argument("--prefill_M", type=int, default=2048)
    ap.add_argument("--decode_prompt", type=int, default=256)
    ap.add_argument("--decode_gen", type=int, default=40)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/luofan/RD_RELAT2_{args.model}.json"
    budgets = [float(x) for x in args.budgets.split(",")]

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_paths(config, args.model)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    print(f"START rd_relat model={args.model} name={config['model']['name']} "
          f"budgets={budgets} prefill_M={args.prefill_M}", flush=True)

    # --- collect R1-off rotated stats once -----------------------------------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, r1_off=True)
    cmodel.to(DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()
    total_rings = sum(stats[n]["nr"] for n in stats)
    print(f"ALLOC units={len(stats)} total_rings={total_rings}", flush=True)

    print("Loading SM120 kernels (JIT may take minutes)...", flush=True)
    nv, mx, cd, qpf = rd_perf.load_kernels()
    dg = rd_perf.load_decode_gemv()
    print("kernels loaded", flush=True)

    _, prompt = rd_perf.rdv_tokenize(config, args.decode_prompt)

    out = {"meta": dict(model=args.model, name=config["model"]["name"],
                        budgets=budgets, prefill_M=args.prefill_M,
                        decode_prompt=args.decode_prompt, decode_gen=args.decode_gen,
                        iters=args.iters, total_rings=total_rings,
                        protocol="interleaved A/B (min+median), fix active")}
    rows = {}
    out["rows"] = rows

    # --- uniform-1/8 baseline (kept resident) --------------------------------
    print("Building uniform-1/8 baseline (resident)...", flush=True)
    base, base_packed = rd_perf.build_real(config, True, None, nv, mx, cd, qpf, dg)
    vocab = base.config.vocab_size
    bn0, bnt, bfrac, bkh = khigh_summary(base)
    out["baseline"] = dict(n_khigh0=bn0, n_targets=bnt, avg_fp8_realized=bfrac)
    print(f"  baseline n_khigh0={bn0}/{bnt} avg_fp8={bfrac:.4f}", flush=True)

    for b in budgets:
        br = int(round(b * total_rings))
        chosen = _clamp_chosen(rdv.measured_alloc(stats, br, total_rings), stats)
        chosen_n0 = sum(1 for S in chosen.values() if len(S) == 0)
        tag = f"measured@{b:.5f}"
        print(f"=== {tag} target_rings={br} chosen_khigh0_linears={chosen_n0} ===",
              flush=True)
        t0 = time.time()
        try:
            meas, meas_packed = rd_perf.build_real(config, True, chosen,
                                                   nv, mx, cd, qpf, dg)
            n0, nt, frac, kh = khigh_summary(meas)
            bpw, realized = bits_per_weight(kh)
            # smoke
            interleaved_prefill(base, meas, 64, vocab, 1, 1)
            a_pre, m_pre = interleaved_prefill(base, meas, args.prefill_M, vocab,
                                               args.iters, args.warmup)
            a_dec, m_dec = interleaved_decode(base, meas, prompt, args.decode_gen)
            # PAIRED per-iteration ratios = gold-standard (base[i], meas[i] timed
            # microseconds apart -> same instantaneous box load -> ratio robust).
            pre_pair = sorted(a / m for a, m in zip(a_pre, m_pre))
            dec_pair = sorted(a / m for a, m in zip(a_dec, m_dec))
            row = dict(
                budget=b, target_rings=br, n_khigh0=n0, n_targets=nt,
                chosen_khigh0_linears=chosen_n0,
                avg_fp8_realized=frac, bits_per_weight=bpw,
                base_prefill_min=min(a_pre), meas_prefill_min=min(m_pre),
                base_prefill_med=_med(a_pre), meas_prefill_med=_med(m_pre),
                base_decode_min=min(a_dec), meas_decode_min=min(m_dec),
                base_decode_med=_med(a_dec), meas_decode_med=_med(m_dec),
                # paired ratios (primary)
                prefill_x_pair_med=_med(pre_pair),
                prefill_x_pair_max=pre_pair[-1],
                decode_x_pair_med=_med(dec_pair),
                decode_x_pair_max=dec_pair[-1],
                # min/median-of-each (secondary cross-check)
                prefill_x_min=min(a_pre) / min(m_pre),
                prefill_x_med=_med(a_pre) / _med(m_pre),
                decode_x_min=min(a_dec) / min(m_dec),
                decode_x_med=_med(a_dec) / _med(m_dec),
            )
            print(f"  n_khigh0={n0}/{nt} avg_fp8={frac:.4f} bpw={bpw:.3f} | "
                  f"PREFILL x(paired-med)={row['prefill_x_pair_med']:.3f} "
                  f"(min-of-each {row['prefill_x_min']:.3f}) "
                  f"base_min={min(a_pre):.2f} meas_min={min(m_pre):.2f}ms | "
                  f"DECODE x(paired-med)={row['decode_x_pair_med']:.3f} "
                  f"(min {row['decode_x_min']:.3f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del meas, meas_packed
            cleanup_memory()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            row = dict(budget=b, error=repr(ex), chosen_khigh0_linears=chosen_n0)
            print(f"  FAILED: {ex}", flush=True)
            cleanup_memory()
        rows[tag] = row
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"WROTE {args.out}", flush=True)
    print("RD_RELAT_DONE", flush=True)


if __name__ == "__main__":
    main()
