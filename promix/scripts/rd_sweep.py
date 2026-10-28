"""Full Pareto sweep of the rate-distortion mixed-precision allocation.

Measures the complete budget knob (avg FP8 ring-fraction) in BOTH directions:
below 1/8 (same-accuracy-faster candidates) and above 1/8 (same-speed-more-
accurate candidates), for THREE model sizes, reporting per budget:

  * wikitext-2 PPL   (accuracy axis, via rd_validity's proven R1-off build:
                      measured-MSE deltaD alloc -> reorder -> GPTQ -> eval)
  * e2e prefill (TTFT, M=2048) + decode (TPOT, M=1) latency with the REAL
    packed NVFP4+MXFP8 fused kernel (via rd_perf's machinery).

Regime: R1-off (pure PCA, identity main rotation) -- the cheapest, rotation-
consistent regime, same as the validity gate. R-on will shift absolute PPL.

This driver is MODEL-AGNOSTIC (1B/3B/8B) and writes results incrementally to a
JSON (one per model+phase) so partial progress survives. A separate local merge
assembles docs/rd_pareto_sweep.csv + report.

Phases (so PPL and latency can run on separate GPUs in parallel):
  --phase ppl   : PPL only (no SM120 kernel JIT needed)
  --phase lat   : latency only (loads the real fused kernel)
  --phase both  : both, serial per point

NOTHING here touches rotation wiring or the ptq rotation code: it only drives
the existing rd_validity / rd_perf build + measurement functions with a per-
model path resolution matching the on-disk artifacts.
"""
import argparse
import json
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, HERE)  # ablation_acc_per_bit, rd_validity, rd_perf

import transformers

import rd_validity as rdv
from rd_validity import RING, ALLOC
from promix.eval import ptq as ptqmod
from promix.utils import cleanup_memory

# NVFP4 = 4-bit mantissa + E4M3 scale per 16 -> 4 + 8/16 = 4.5 bits/weight.
# MXFP8 = 8-bit + E8M0 scale per 32          -> 8 + 8/32 = 8.25 bits/weight.
BPW_FP4 = 4.5
BPW_FP8 = 8.25

# ---------------------------------------------------------------------------
# Per-model path resolution matching the on-disk R1-off (per-head PCA) artifacts
# in ./rotation/. Mirrors rd_validity.resolve_paths for 3B and extends it to the
# 1B (-Instruct) and 8B (llama-3-8b) short names + local model dirs.
# ---------------------------------------------------------------------------
MODELS = {
    "1b": dict(short="Llama-3.2-1B-Instruct",
               model="unsloth/Llama-3.2-1B-Instruct",
               local="/home/luofan/models/Llama-3.2-1B-Instruct",
               bf16_target=13.16),
    "3b": dict(short="Llama-3.2-3B",
               model="unsloth/Llama-3.2-3B",
               local="/home/luofan/models/Llama-3.2-3B",
               bf16_target=7.82),
    "8b": dict(short="llama-3-8b",
               model="unsloth/llama-3-8b",
               local="/home/luofan/models/llama-3-8b",
               bf16_target=6.14),
}


def resolve_paths(config, model_key):
    m = MODELS[model_key]
    name = m["model"]
    if os.path.isdir(m["local"]):
        name = m["local"]
    config["model"]["name"] = name
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{m['short']}.bin")
    config["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{m['short']}.bin")
    config["quantize"]["o_proj_pca"] = "per_head"   # matches the per-head basis
    config["quantize"]["high_fraction"] = 0.125
    return config


def bits_per_weight(khigh_map):
    """Exact bits/weight from the realized per-Linear K_high map ({name:[kh,K]})
    over the 7 target Linears: K_low served NVFP4 (4.5b), K_high served MXFP8."""
    tot_k = sum(K for _, K in khigh_map.values())
    tot_hi = sum(kh for kh, _ in khigh_map.values())
    tot_lo = tot_k - tot_hi
    if tot_k == 0:
        return float("nan"), 0.0
    bpw = (tot_lo * BPW_FP4 + tot_hi * BPW_FP8) / tot_k
    return bpw, tot_hi / tot_k


def _aligned_uniform_chosen(stats, b):
    """Uniform-at-b: every ALLOC Linear gets round(b*nr) FP8 rings as the highest
    indices (>=1, <=nr-1 clamp only to avoid degenerate full/empty for latency;
    for PPL we keep round() without clamp when 0<b<1)."""
    chosen = {}
    for name in stats:
        nr = stats[name]["nr"]
        n = int(round(b * nr))
        n = max(0, min(nr, n))
        chosen[name] = set(range(nr - n, nr))
    return chosen


# ---------------------------------------------------------------------------
def build_points(stats, budgets, total_rings):
    """List of (alloc, budget, chosen). chosen=None -> shipped uniform-1/8."""
    pts = [("uniform", 0.125, None)]                       # anchor baseline
    pts.append(("uniform", 0.0625, _aligned_uniform_chosen(stats, 0.0625)))  # ref
    for b in budgets:
        budget_rings = int(round(b * total_rings))
        chosen = rdv.measured_alloc(stats, budget_rings, total_rings)
        pts.append(("measured", b, chosen))
    return pts


def point_key(alloc, budget):
    return f"{alloc}@{budget:.5f}"


# ---------------------------------------------------------------------------
def run(model_key, config_path, phase, budgets, args):
    config = ptqmod.load_config(os.path.join(REPO, config_path))
    config = resolve_paths(config, model_key)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    config["eval"]["batch_size"] = args.eval_bs
    m = MODELS[model_key]

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))
    print(f"START rd_sweep model={model_key} phase={phase} "
          f"name={config['model']['name']} basis={os.path.basename(config['paths']['basis'])} "
          f"budgets={budgets}", flush=True)

    out = {"meta": dict(model=model_key, name=config["model"]["name"], phase=phase,
                        basis=os.path.basename(config["paths"]["basis"]),
                        budgets=budgets, bf16_target=m["bf16_target"],
                        prefill_M=args.prefill_M, decode_prompt=args.decode_prompt,
                        decode_gen=args.decode_gen, iters=args.iters)}
    rows = {}
    out["rows"] = rows

    def flush():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # --- bf16 bridge gate ---------------------------------------------------
    if phase in ("ppl", "both") and not args.skip_bf16:
        t0 = time.time()
        bf16 = rdv.bf16_bridge(config)
        out["bf16_bridge"] = bf16
        ok = abs(bf16 - m["bf16_target"]) < 0.25
        print(f"BF16_BRIDGE ppl={bf16:.4f} target={m['bf16_target']} "
              f"{'OK' if ok else 'MISMATCH'} ({time.time()-t0:.0f}s)", flush=True)
        out["bf16_ok"] = ok
        flush()
        cleanup_memory()

    # --- collect R1-off rotated stats ONCE ----------------------------------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = rdv.build_base(config, r1_off=True)
    cmodel.to(rdv.DEV)
    stats = rdv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()
    total_rings = sum(stats[n]["nr"] for n in stats)
    out["meta"]["total_rings"] = total_rings
    print(f"ALLOC units={len(stats)} total_rings={total_rings}", flush=True)

    pts = build_points(stats, budgets, total_rings)

    # --- latency kernel load (lazy) -----------------------------------------
    nv = mx = cd = qpf = dg = None
    lat_ref = None
    if phase in ("lat", "both"):
        import rd_perf
        print("Loading SM120 kernels (JIT may take minutes)...", flush=True)
        nv, mx, cd, qpf = rd_perf.load_kernels()
        dg = rd_perf.load_decode_gemv()
        print("kernels loaded", flush=True)

    lat_args = types.SimpleNamespace(
        prefill_M=args.prefill_M, decode_prompt=args.decode_prompt,
        decode_gen=args.decode_gen, iters=args.iters, warmup=args.warmup)

    # --- per-point measurement ----------------------------------------------
    for alloc, budget, chosen in pts:
        key = point_key(alloc, budget)
        row = rows.setdefault(key, dict(model=model_key, alloc=alloc, budget=budget))
        print(f"=== POINT {key} ===", flush=True)

        if phase in ("ppl", "both"):
            t0 = time.time()
            model = rdv.build_eval(config, r1_off=True, chosen=chosen)
            frac, hi, totc = rdv.avg_fp8_fraction(model)
            # realized per-Linear K_high over the 7 targets for exact bpw
            kh_map = {}
            from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper
            for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
                short = name.split(".")[-1]
                if ".layers." not in name or short not in (ALLOC + ("o_proj",)):
                    continue
                q = w.quantizer
                if not isinstance(getattr(q, "bits", None), str):
                    continue
                kh_map[name] = [int(q.high_bits_length), int(w.module.weight.shape[1])]
            bpw, realized = bits_per_weight(kh_map)
            ppl = rdv.eval_ppl(model, config)
            row.update(ppl=ppl, avg_fp8_realized=frac, bits_per_weight=bpw)
            print(f"  PPL={ppl:.4f} avg_fp8={frac:.4f} bpw={bpw:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model
            cleanup_memory()
            flush()

        if phase in ("lat", "both"):
            t0 = time.time()
            import rd_perf
            lat = rd_perf.measure_latency_for(config, True, chosen, stats, budget,
                                              nv, mx, cd, qpf, dg, lat_args)
            if "error" not in lat:
                row["prefill_ms"] = lat["prefill_ms"]
                row["decode_ms"] = lat["decode_ms"]
                if "avg_fp8_realized" not in row and "khigh" in lat:
                    bpw, realized = bits_per_weight(lat["khigh"])
                    row["bits_per_weight"] = bpw
                    row["avg_fp8_realized"] = realized
                if alloc == "uniform" and abs(budget - 0.125) < 1e-9:
                    lat_ref = lat
                print(f"  prefill={lat['prefill_ms']:.3f}ms decode={lat['decode_ms']:.3f}ms "
                      f"fallback={lat.get('uniform_fallback')} ({time.time()-t0:.0f}s)",
                      flush=True)
            else:
                row["lat_error"] = lat["error"]
                print(f"  latency FAILED: {lat['error']}", flush=True)
            flush()

    # --- speedups vs uniform-1/8 (latency phase only) -----------------------
    if lat_ref is not None:
        pr0 = lat_ref["prefill_ms"]
        de0 = lat_ref["decode_ms"]
        for key, row in rows.items():
            if "prefill_ms" in row:
                row["prefill_x_vs_uniform18"] = pr0 / row["prefill_ms"]
            if "decode_ms" in row:
                row["decode_x_vs_uniform18"] = de0 / row["decode_ms"]
        flush()

    print(f"WROTE {args.out}", flush=True)
    print("RD_SWEEP_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", default="both", choices=["ppl", "lat", "both"])
    ap.add_argument(
        "--budgets",
        default="0,0.03125,0.041667,0.0625,0.083333,0.125,0.1875,0.25,0.375,0.5,0.75,1.0")
    ap.add_argument("--prefill_M", type=int, default=2048)
    ap.add_argument("--decode_prompt", type=int, default=256)
    ap.add_argument("--decode_gen", type=int, default=48)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--skip_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/luofan/RD_SWEEP_{args.model}_{args.phase}.json"
    budgets = [float(x) for x in args.budgets.split(",")]
    run(args.model, args.config, args.phase, budgets, args)


if __name__ == "__main__":
    main()
