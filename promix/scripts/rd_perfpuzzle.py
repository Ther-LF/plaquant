"""RD PERF-PUZZLE — why is PURE-NVFP4 (r=0) prefill latency ~= the 1/8 mixed
config end-to-end, when the GEMM microbench shows r=0 is 5-10% below 1/8?

Leading hypothesis: DILUTION. The FP4/FP8 difference only touches the 7
quantized Linears; attention(SDPA)+RoPE+RMSNorm+down_proj online Hadamard+
embeddings+lm_head are all FP16 and r-independent. If the quantized-Linear GEMM
is a modest fraction of full-model prefill wall-clock at M=2048, a ~10% GEMM win
dilutes to ~1% e2e.

This harness DECOMPOSES full-model prefill (M=2048) on the REAL packed
NVFP4+MXFP8 fused-kernel model into:
  (1) the 7 quantized-Linear GEMMs (fused kernel) + their activation PACK,
  (2) everything-else (attention + norm + rope + hadamard + lm_head + embed).

Method: monkey-patch MixedQuantLinear._pack_activation and ._gemm to bracket
every call with a cudaEvent pair (NO per-call sync). After each full forward we
sync ONCE and sum the per-segment elapsed times. The whole forward is bracketed
by an outer event pair. everything-else = whole - (pack + gemm). We do this for
BOTH configs:
  * uniform-1/8  (chosen=None   -> shipped K_high = round(1/8 K) per Linear)
  * pure-NVFP4   (chosen={}     -> K_high = 0 every Linear)
so we get:
  - Linear-GEMM share of prefill wall-clock,
  - the measured FP4-vs-1/8 latency of JUST the GEMMs (the ~10% should show
    HERE if it is dilution),
  - whether the pure-NVFP4 (K_high=0) PACK path has extra overhead (still
    launches the fp8 pack kernel? GEMM path less optimized?).

Plus one torch.profiler run per config to bucket "everything-else" kernel time
by name (SDPA / norm / rope / hadamard / lm_head / elementwise).

Remote-only (Pro 6000 SM120, inside luofan-plaquant). Line-buffered + JSON.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

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

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.eval.full_network_realkernel import load_kernels
from promix.eval.decode_throughput import load_decode_gemv

import rd_validity as rdv
import rd_perf
from promix.quantize import full_network as fn
from promix.quantize.kv_quant import setup_k_quant


# ---------------------------------------------------------------------------
# Build a real packed model with a FORCED per-Linear K_high (same for all 7
# targets). kh=None -> shipped uniform-1/8 (byte-identical). kh=64 -> the
# leanest MXFP8 segment the fused kernel supports (one 64-wide ring); the kernel
# cannot build K_high==0 (pure-NVFP4): both the fp8 activation pack and the
# Phase-2 MXFP8 TMA descriptor are unconditionally constructed and assert on
# zero-K, so kh=64 is the honest "near-pure-NVFP4" floor on this kernel.
# ---------------------------------------------------------------------------
def build_forced(config, kh, kernels):
    nv, mx, cd, qpf, dg = kernels
    model = rdv.build_base(config, True)
    model.to(DEV)
    if kh is not None:
        override = {}
        for name, short, w in fn.iter_target_wrappers(model):
            override[name] = int(kh)
            w.module._edriven_khigh = int(kh)
        model._rd_khigh_override = override
    ptqmod.configure_quantizers(model, config)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=0.125,
                                 verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=dg)
    model.to(DEV).eval()
    model.config.use_cache = True
    return model, packed


# ---------------------------------------------------------------------------
# cudaEvent instrumentation of MixedQuantLinear: accumulate pack + gemm GPU
# time per forward with NO per-call sync (events bracket each segment; we sum
# after a single sync).
# ---------------------------------------------------------------------------
_EV = {"pack": [], "gemm": [], "on": False, "shapes": defaultdict(lambda: [0, 0, 0])}

_orig_pack = MixedQuantLinear._pack_activation
_orig_gemm = MixedQuantLinear._gemm


def _timed_pack(self, x2d):
    if not _EV["on"]:
        return _orig_pack(self, x2d)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig_pack(self, x2d)
    e.record()
    _EV["pack"].append((s, e))
    # per-shape bookkeeping (count only; timing summed globally)
    _EV["shapes"][(self.N, self.K, self.K_high)][2] += 1
    return out


def _timed_gemm(self, A_lo, pSFA_lo, A_hi, pSFA_hi, M):
    if not _EV["on"]:
        return _orig_gemm(self, A_lo, pSFA_lo, A_hi, pSFA_hi, M)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = _orig_gemm(self, A_lo, pSFA_lo, A_hi, pSFA_hi, M)
    e.record()
    _EV["gemm"].append((s, e))
    return out


MixedQuantLinear._pack_activation = _timed_pack
MixedQuantLinear._gemm = _timed_gemm


def _reset_ev():
    _EV["pack"].clear()
    _EV["gemm"].clear()


def _sum_ev(key):
    return sum(s.elapsed_time(e) for (s, e) in _EV[key])


# ---------------------------------------------------------------------------
@torch.no_grad()
def decomposed_prefill(model, M, vocab, iters, warmup):
    """Return dict with whole/pack/gemm/rest median-ms over `iters`, each a
    per-forward total from the SAME forwards (so the split is self-consistent)."""
    ids = torch.randint(0, vocab, (1, M), device=DEV)
    model.config.use_cache = False

    def fwd():
        return model(input_ids=ids, use_cache=False).logits

    # warmup (JIT, autotune, alloc)
    _EV["on"] = False
    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()

    # CLEAN whole-model timing (outer event only, NO per-call instrumentation
    # -> matches rd_perf/sweep; the per-Linear events below add CPU launch
    # overhead that inflates the absolute whole, so the honest e2e number is
    # this clean one; the instrumented split is used only for RATIOS/shares).
    clean = []
    for _ in range(iters):
        ws = torch.cuda.Event(enable_timing=True)
        we = torch.cuda.Event(enable_timing=True)
        ws.record(); fwd(); we.record(); torch.cuda.synchronize()
        clean.append(ws.elapsed_time(we))

    whole, pack, gemm, nlin = [], [], [], []
    _EV["on"] = True
    for _ in range(iters):
        _reset_ev()
        ws = torch.cuda.Event(enable_timing=True)
        we = torch.cuda.Event(enable_timing=True)
        ws.record()
        fwd()
        we.record()
        torch.cuda.synchronize()
        whole.append(ws.elapsed_time(we))
        pack.append(_sum_ev("pack"))
        gemm.append(_sum_ev("gemm"))
        nlin.append(len(_EV["gemm"]))
    _EV["on"] = False

    def med(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    w, p, g = med(whole), med(pack), med(gemm)
    cw = med(clean)
    return dict(clean_whole_ms=cw, whole_ms=w, pack_ms=p, gemm_ms=g,
                linear_ms=p + g, rest_ms=w - p - g, n_linear_calls=int(med(nlin)),
                gemm_share=g / w, linear_share=(p + g) / w,
                clean_whole_all=sorted(clean),
                whole_all=sorted(whole), gemm_all=sorted(gemm),
                pack_all=sorted(pack))


@torch.no_grad()
def profile_buckets(model, M, vocab):
    """One torch.profiler pass; bucket CUDA self-time by kernel name into
    coarse categories to characterize 'everything-else'."""
    ids = torch.randint(0, vocab, (1, M), device=DEV)
    model.config.use_cache = False
    from torch.profiler import profile, ProfilerActivity
    # warmup
    model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(3):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        cuda_us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "self_cuda_time_total", 0)
        if cuda_us and cuda_us > 0:
            rows.append((ev.key, float(cuda_us)))
    rows.sort(key=lambda t: -t[1])
    total = sum(c for _, c in rows)

    def bucket(name):
        n = name.lower()
        if any(k in n for k in ("fused_gemm", "codesign", "gemm_prepacked",
                                 "blockscaled", "mma")):
            return "linear_gemm"
        if any(k in n for k in ("quant_pack", "pack_float", "quantize", "sfa",
                                 "interleave", "prepack")):
            return "linear_pack"
        if any(k in n for k in ("attention", "sdpa", "flash", "fmha", "softmax",
                                 "bmm", "scaled_dot")):
            return "attention"
        if any(k in n for k in ("rms", "norm", "layernorm")):
            return "rmsnorm"
        if any(k in n for k in ("rope", "rotary", "sin", "cos", "embedding")):
            return "rope_embed"
        if any(k in n for k in ("hadamard", "transform", "sqrt")):
            return "hadamard"
        if any(k in n for k in ("elementwise", "mul", "add", "silu", "gelu",
                                 "copy", "cat", "vectorized", "index")):
            return "elementwise"
        return "other"

    buckets = defaultdict(float)
    for name, c in rows:
        buckets[bucket(name)] += c
    return dict(total_cuda_us=total,
                buckets={k: v for k, v in sorted(buckets.items(),
                                                 key=lambda t: -t[1])},
                top20=[(n, round(c, 1)) for n, c in rows[:20]])


def realized_khigh(model):
    out = {}
    from promix.quantize import full_network as fn
    for name, short, w in fn.iter_target_wrappers(model):
        out[short] = [int(getattr(w.module, "K_high", -1)),
                      int(getattr(w.module, "K", -1))]
    return out


def run_config(tag, kh, config, kernels, args):
    print(f"\n=== BUILD {tag} (K_high={'1/8 shipped' if kh is None else kh}) ===",
          flush=True)
    model, packed = build_forced(config, kh, kernels)
    vocab = model.config.vocab_size
    khr = realized_khigh(model)
    print(f"  realized K_high (short->[K_high,K]): {khr}", flush=True)
    _ = decomposed_prefill(model, 64, vocab, iters=1, warmup=1)
    dec = decomposed_prefill(model, args.prefill_M, vocab, args.iters, args.warmup)
    print(f"  prefill M={args.prefill_M}: clean_whole={dec['clean_whole_ms']:.3f} "
          f"instr_whole={dec['whole_ms']:.3f} gemm={dec['gemm_ms']:.3f} "
          f"({dec['gemm_share']*100:.1f}%) pack={dec['pack_ms']:.3f} "
          f"linear={dec['linear_ms']:.3f} ({dec['linear_share']*100:.1f}%) "
          f"rest={dec['rest_ms']:.3f} calls={dec['n_linear_calls']}", flush=True)
    prof = profile_buckets(model, args.prefill_M, vocab)
    print(f"  profiler buckets(us): {prof['buckets']}", flush=True)
    print(f"  profiler top8: {prof['top20'][:8]}", flush=True)
    del model, packed
    cleanup_memory()
    return dict(tag=tag, realized_khigh=khr, prefill=dec, profiler=prof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefill_M", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--near_pure_khigh", type=int, default=64,
                    help="K_high for the near-pure-NVFP4 config (kernel cannot "
                         "do 0; 64 = one ring = leanest supported)")
    ap.add_argument("--out", default="/home/luofan/RD_PERFPUZZLE.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rdv.resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    print(f"START rd_perfpuzzle model={config['model']['name']} "
          f"prefill_M={args.prefill_M} iters={args.iters} "
          f"near_pure_khigh={args.near_pure_khigh}", flush=True)
    print("Loading SM120 kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()
    kernels = (nv, mx, cd, qpf, dg)
    print("kernels loaded", flush=True)

    results = {"meta": dict(model=config["model"]["name"],
                            prefill_M=args.prefill_M, iters=args.iters,
                            near_pure_khigh=args.near_pure_khigh,
                            basis=os.path.basename(config["paths"]["basis"]))}

    r_u = run_config("uniform_1_8", None, config, kernels, args)
    r_f = run_config("near_pure_nvfp4", args.near_pure_khigh, config, kernels, args)
    results["uniform_1_8"] = r_u
    results["near_pure_nvfp4"] = r_f

    # ---- headline deltas (clean whole for e2e; instrumented for GEMM ratio) ----
    cu, cf = r_u["prefill"]["clean_whole_ms"], r_f["prefill"]["clean_whole_ms"]
    gu, gf = r_u["prefill"]["gemm_ms"], r_f["prefill"]["gemm_ms"]
    pu, pf = r_u["prefill"]["pack_ms"], r_f["prefill"]["pack_ms"]
    lu, lf = r_u["prefill"]["linear_ms"], r_f["prefill"]["linear_ms"]
    summary = dict(
        e2e_clean_whole_uniform_ms=cu, e2e_clean_whole_nearpure_ms=cf,
        e2e_speedup_nearpure_vs_uniform=cu / cf,
        gemm_uniform_ms=gu, gemm_nearpure_ms=gf,
        gemm_speedup_nearpure_vs_uniform=gu / gf,
        pack_uniform_ms=pu, pack_nearpure_ms=pf,
        pack_speedup_nearpure_vs_uniform=(pu / pf) if pf else None,
        linear_speedup_nearpure_vs_uniform=lu / lf,
        gemm_share_instr_uniform=r_u["prefill"]["gemm_share"],
        linear_share_instr_uniform=r_u["prefill"]["linear_share"],
    )
    results["summary"] = summary
    print("\n===== SUMMARY =====", flush=True)
    for k, v in summary.items():
        print(f"  {k} = {v}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"WROTE {args.out}", flush=True)
    print("RD_PERFPUZZLE_DONE", flush=True)


if __name__ == "__main__":
    main()
