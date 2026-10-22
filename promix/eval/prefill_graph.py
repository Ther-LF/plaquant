"""Prefill-overhead reduction harness (SM120 NVFP4+MXFP8 real-kernel model).

Converts the measured *compute* win (fused pack+GEMM beats FP16 cuBLAS Linear at
every prefill M, per docs/full_network_realkernel_report.md) into an *end-to-end*
win by erasing the per-call Python / kernel-launch dispatch overhead that made the
full-model prefill 0.68-0.82x FP16.

Primary lever: **CUDA-graph capture** of the whole decoder prefill forward (all 112
real-kernel Linears + the FP16 rest). After capture the 112x~4 kernel launches +
reshape/.to(fp16)/bias/Hadamard Python glue replay with ~zero host overhead.

Enablers (NOT kernel math):
  * codesign_fused_ext.cu `run_with_cached_ws` skips its host `cudaStreamSynchronize`
    while the stream is capturing (guarded by cudaStreamIsCapturing) — the only
    capture-breaker in the GEMM path. Non-capture behavior unchanged.
  * quant_pack_float_ext.cu `init_grids_once` does a one-shot cudaMemcpyToSymbol; it
    is pre-warmed before capture (warmup forward) so no sync happens during capture.

Secondary lever (lever 3): online Hadamard on down_proj runs in FP32 on the real
path (real-model-only cost). We quantify it and optionally run it in FP16, gating on
output cosine.

Gate: graphed full-model prefill output cosine vs the eager real-kernel output
>= 0.9999 (replay runs the identical kernels, so expect ~1.0).

Usage:
  python -m promix.eval.prefill_graph \
      --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --ms 2048,4096,8192 --phases capture,hadamard
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
from promix.quantize.quant_utils import ActQuantWrapper
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, _component_breakdown, cosine, LAYER_SHAPES,
    HIGH_FRACTION,
)


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
def time_eager(fn_, iters=20, warmup=5):
    """Min-of-N wall-clock of an eager callable (each call its own launches).
    Min-of-N (matching the report methodology) rejects transient co-tenant
    contention spikes on the shared box."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn_()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn_()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def time_graph(graph, iters=30, warmup=5):
    """Min-of-N wall-clock of replaying a captured CUDA graph."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def capture_forward(model, static_ids):
    """Capture `model(static_ids).logits` into a CUDA graph.

    Standard torch CUDA-graph protocol: warm up on a side stream (so cuBLAS /
    cuDNN / CUTLASS one-time lazy init + our qpf grid upload happen OUTSIDE
    capture), then capture. Returns (graph, static_logits). Replay after
    `static_ids.copy_(new_ids)`; result lands in `static_logits`.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = model(static_ids)
            _ = out.logits
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(static_ids)
        static_logits = out.logits
    return g, static_logits


# ---------------------------------------------------------------------------
# Phase: CUDA-graph capture of the full prefill forward
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_capture(config, model, fp16, nv, mx, cd, qpf, args, out):
    print("=== PHASE: CUDA-GRAPH PREFILL CAPTURE ===", flush=True)
    n_layers = model.config.num_hidden_layers
    vocab = model.config.vocab_size
    Ms = [int(x) for x in args.ms.split(",")]
    rows = []
    for M in Ms:
        torch.manual_seed(M)
        static_ids = torch.randint(0, vocab, (1, M), device=DEV)

        # ---- BEFORE: eager real-kernel full forward (the 0.68-0.82x path) ----
        eager_out = model(static_ids).logits.clone()
        t_real_eager = time_eager(lambda: model(static_ids).logits,
                                  iters=(10 if M <= 4096 else 6), warmup=4)

        # ---- AFTER: capture + replay ----
        cap_err = None
        t_real_graph = float("nan")
        cos_gate = float("nan")
        try:
            g, static_logits = capture_forward(model, static_ids)
            g.replay(); torch.cuda.synchronize()
            graph_out = static_logits.clone()
            cos_gate = cosine(eager_out, graph_out)
            t_real_graph = time_graph(g, iters=(30 if M <= 4096 else 15), warmup=5)
        except Exception as e:
            import traceback; traceback.print_exc()
            cap_err = repr(e)

        # ---- FP16 reference: eager + (try) graph ----
        t_fp16_eager = time_eager(lambda: fp16(static_ids).logits,
                                  iters=(10 if M <= 4096 else 6), warmup=4)
        t_fp16_graph = float("nan")
        try:
            gf, _ = capture_forward(fp16, static_ids)
            t_fp16_graph = time_graph(gf, iters=(30 if M <= 4096 else 15), warmup=5)
            del gf
        except Exception as e:
            print(f"  fp16 capture failed: {e!r}", flush=True)

        # ---- COMBINED lever: graph + FP16 online Hadamard (lever 3) ----
        # The residual the graph can't remove at large M is REAL GPU compute from the
        # FP32 online Hadamard on down_proj (16x). Running it in FP16 is bit-stable here
        # (per-op cos 1.0) and is captured into the same graph. Cosine is gated vs the
        # ORIGINAL FP32-Hadamard real path (eager_out).
        t_real_graph_h16 = float("nan"); cos_h16 = float("nan"); su_h16 = float("nan")
        if args.had16:
            hw = _had_wrappers(model)
            saved = [w.fp32_had for w in hw]
            for w in hw:
                w.fp32_had = False
            try:
                gh, static_h = capture_forward(model, static_ids)
                gh.replay(); torch.cuda.synchronize()
                cos_h16 = cosine(eager_out, static_h.clone())
                t_real_graph_h16 = time_graph(gh, iters=(30 if M <= 4096 else 15), warmup=5)
                su_h16 = t_fp16_eager / t_real_graph_h16
                del gh
            except Exception as e:
                import traceback; traceback.print_exc()
            for w, s in zip(hw, saved):
                w.fp32_had = s

        bd = _component_breakdown(nv, mx, cd, qpf, M, n_layers, DEV)
        fp16_rest = t_fp16_eager - bd["fp16_linear_ms"]
        ideal = bd["pack_ms"] + bd["gemm_ms"] + fp16_rest  # zero-overhead target

        su_before = t_fp16_eager / t_real_eager
        su_after = (t_fp16_eager / t_real_graph) if t_real_graph == t_real_graph else float("nan")
        su_after_vs_fp16graph = ((t_fp16_graph / t_real_graph)
                                 if (t_real_graph == t_real_graph and t_fp16_graph == t_fp16_graph)
                                 else float("nan"))

        print(f"  M={M:5d}: real eager={t_real_eager:.3f}ms  real GRAPH={t_real_graph:.3f}ms "
              f"real GRAPH+had16={t_real_graph_h16:.3f}ms "
              f"| fp16 eager={t_fp16_eager:.3f}ms  fp16 graph={t_fp16_graph:.3f}ms", flush=True)
        print(f"         speedup vs FP16-eager:  BEFORE(eager)={su_before:.3f}x  "
              f"AFTER(graph)={su_after:.3f}x  GRAPH+had16={su_h16:.3f}x "
              f"| vs FP16-graph: {su_after_vs_fp16graph:.3f}x", flush=True)
        print(f"         breakdown: pack={bd['pack_ms']:.3f} gemm={bd['gemm_ms']:.3f} "
              f"fp16-rest={fp16_rest:.3f}  zero-overhead-ideal={ideal:.3f}ms  "
              f"(overhead eager={t_real_eager-ideal:.3f} graph={t_real_graph-ideal:.3f})", flush=True)
        print(f"         NUMERICS cos(graph vs eager)={cos_gate:.6f} "
              f"cos(graph+had16 vs eager)={cos_h16:.6f} "
              f"-> {'PASS' if (cos_gate==cos_gate and cos_gate>=0.9999) else 'FAIL'}"
              + (f"  [capture_err={cap_err}]" if cap_err else ""), flush=True)

        rows.append(dict(
            M=M, t_real_eager_ms=t_real_eager, t_real_graph_ms=t_real_graph,
            t_real_graph_had16_ms=t_real_graph_h16,
            t_fp16_eager_ms=t_fp16_eager, t_fp16_graph_ms=t_fp16_graph,
            speedup_before_eager=su_before, speedup_after_graph=su_after,
            speedup_graph_had16=su_h16,
            speedup_after_vs_fp16graph=su_after_vs_fp16graph,
            pack_ms=bd["pack_ms"], gemm_ms=bd["gemm_ms"],
            fp16_linear_ms=bd["fp16_linear_ms"], fp16_rest_ms=fp16_rest,
            zero_overhead_ideal_ms=ideal,
            overhead_eager_ms=t_real_eager - ideal,
            overhead_graph_ms=(t_real_graph - ideal) if t_real_graph == t_real_graph else None,
            cosine_graph_vs_eager=cos_gate,
            cosine_graph_had16_vs_eager=cos_h16,
            gate_pass=bool(cos_gate == cos_gate and cos_gate >= 0.9999),
            capture_error=cap_err))
        del static_ids, eager_out
        cleanup_memory()
    out["capture"] = rows


# ---------------------------------------------------------------------------
# Phase: online-Hadamard precision lever (fp32 -> fp16) quantification
# ---------------------------------------------------------------------------
def _had_wrappers(model):
    return [w for _, _, w in fn.iter_target_wrappers(model)
            if getattr(w, "online_full_had", False)]


@torch.no_grad()
def run_hadamard(config, model, nv, mx, cd, qpf, args, out):
    print("=== PHASE: ONLINE-HADAMARD FP32->FP16 LEVER ===", flush=True)
    from promix.quantize.hadamard import matmul_hadU_cuda
    hw = _had_wrappers(model)
    print(f"  online-Hadamard wrappers (down_proj): {len(hw)}", flush=True)
    if not hw:
        out["hadamard"] = {"note": "no online-Hadamard wrappers"}
        return

    w0 = hw[0]
    K = w0.module.K  # down_proj in-dim (8192)
    Ms = [int(x) for x in args.ms.split(",")]
    rows = []
    for M in Ms:
        x16 = torch.randn(1, M, K, device=DEV, dtype=torch.float16)

        def had_fp32():
            return matmul_hadU_cuda(x16.float(), w0.had_K, w0.K).to(torch.float16)

        def had_fp16():
            return matmul_hadU_cuda(x16, w0.had_K, w0.K)

        t32 = time_eager(had_fp32, iters=30, warmup=8)
        t16 = time_eager(had_fp16, iters=30, warmup=8)
        y32 = had_fp32(); y16 = had_fp16()
        cos = cosine(y32, y16)
        n_had = len(hw)
        print(f"  M={M:5d}: per-call Hadamard fp32={t32*1e3:.1f}us fp16={t16*1e3:.1f}us "
              f"| x{n_had} down_proj = fp32 {t32*n_had:.3f}ms vs fp16 {t16*n_had:.3f}ms "
              f"(save {(t32-t16)*n_had:.3f}ms) | cos(fp16 vs fp32)={cos:.6f}", flush=True)
        rows.append(dict(M=M, per_call_fp32_ms=t32, per_call_fp16_ms=t16,
                         n_down_proj=n_had,
                         total_fp32_ms=t32 * n_had, total_fp16_ms=t16 * n_had,
                         save_ms=(t32 - t16) * n_had, cosine_fp16_vs_fp32=cos))
        del x16
        cleanup_memory()
    out["hadamard"] = rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--phases", type=str, default="capture,hadamard")
    ap.add_argument("--had16", action="store_true", default=False,
                    help="also measure graph + FP16 online Hadamard (lever 3 combined)")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "prefill_graph.json"))
    args = ap.parse_args()

    phases = set(args.phases.split(","))
    config = load_config(args.config)
    print(f"Prefill-overhead harness | config={args.config} ms={args.ms} phases={phases}", flush=True)

    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()

    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    print("Building real-kernel model (no GPTQ; latency is weight-value independent)...", flush=True)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV)
    model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)
    model.to(DEV).eval()

    # CUDA-graph capture safety: the o_proj column-reorder index (`_column_order`,
    # mirror `new_column_order`) is a plain CPU tensor attribute that model.to(DEV)
    # does NOT move; gathering with a CPU index during capture triggers an illegal
    # CPU<->CUDA copy. Move every such index to the model device up front (pure
    # reindex, no numeric change).
    n_moved = 0
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
                n_moved += 1
    print(f"moved {n_moved} column-order index tensors to {DEV}", flush=True)

    fp16 = _build_fp16_model(config['model']['name'])

    out = {"config": args.config, "ms": args.ms}
    if "capture" in phases:
        run_capture(config, model, fp16, nv, mx, cd, qpf, args, out)
    if "hadamard" in phases:
        run_hadamard(config, model, nv, mx, cd, qpf, args, out)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("PREFILL_GRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
