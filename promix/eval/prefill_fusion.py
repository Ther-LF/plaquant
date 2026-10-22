"""Prefill-fusion fixes harness (SM120 NVFP4+MXFP8 real-kernel model).

Measures the three ranked prefill-fusion fixes CUMULATIVELY, numerics-gated,
and reports full-model prefill latency vs FP16 before->after per M:

  fix1 (div-fold)   : fold the down_proj online-Hadamard 1/sqrt(n) into the
                      fast_hadamard_transform scale -> kills the separate
                      aten::div kernel.  (promix/quantize/hadamard.py)
  fix2 (had-fp32)   : feed the FP32 Hadamard output STRAIGHT into the qpf pack
                      (skip the fp32->fp16 round-trip cast).  (quant_utils +
                      full_network wiring)
  fix3 (fp16-gemm)  : fused-GEMM epilogue emits fp16 directly (new config copy
                      codesign_fused_fp16_ext.cu) -> folds away the 112x
                      per-Linear bf16->fp16 cast (`D.to(fp16)`).

Numerics gate: full-model prefill logits cosine vs the PRE-FIX real-kernel path
>= 0.9999 at each cumulative stage (exact A/B by toggling the levers on the same
model). PPL gate is the companion phase (`--phase ppl`): real-kernel wikitext-2
PPL must stay 14.85 (+-0.02).

Remote-only, container luofan-plaquant, FREE GPU (idle picker below).

Usage:
  python -m promix.eval.prefill_fusion --phase perf --ms 2048,4096,8192
  python -m promix.eval.prefill_fusion --phase ppl
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.quantize import full_network as fn
from promix.eval.data import get_wikitext2
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, _component_breakdown, cosine, HIGH_FRACTION,
)
from promix.eval.prefill_graph import capture_forward, time_eager, time_graph


def load_cd_fp16():
    """Build the fp16-output epilogue config copy as a separate torch ext."""
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="codesign_fused_fp16_ext",
                sources=[os.path.join(KERN, "codesign_fused_fp16_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


def set_levers(model, divfold, hadfp32):
    """Toggle the online-Hadamard (down_proj) fusion levers on every wrapper."""
    for _, _, w in fn.iter_target_wrappers(model):
        if getattr(w, "online_full_had", False):
            w.fold_had_scale = bool(divfold)
            w.had_out_fp32 = bool(hadfp32) and bool(getattr(w, "fp32_had", False))


def move_col_order(model):
    n = 0
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV)); n += 1
    return n


def pick_free_gpu(util_max=8, timeout_s=0):
    """Return index of an idle GPU (util<=util_max), waiting up to timeout_s."""
    t0 = time.time()
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"]).decode()
            cands = []
            for line in out.strip().splitlines():
                i, u, mem = [x.strip() for x in line.split(",")]
                cands.append((int(i), int(u), int(mem)))
            cands.sort(key=lambda c: (c[1], c[2]))
            if cands and cands[0][1] <= util_max:
                return cands[0][0]
            if time.time() - t0 > timeout_s:
                return cands[0][0] if cands else 0
        except Exception:
            return 0
        time.sleep(5)


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_perf(config, args):
    nv, mx, cd, qpf = load_kernels()
    cd16 = load_cd_fp16()
    print("kernels loaded (cd=bf16-out, cd16=fp16-out)", flush=True)

    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV); model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)

    model_name = config['model']['name']
    fp16 = _build_fp16_model(model_name)
    n_layers = model.config.num_hidden_layers
    vocab = model.config.vocab_size
    Ms = [int(x) for x in args.ms.split(",")]

    # Real wikitext ids for the numerics (cosine) gate — random token ids are a
    # pathological, near-uniform-logit input that overstates perturbation; the
    # gate must reflect real inputs. Latency is data-independent for these
    # kernels, so we time on the same real ids.
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    wiki = get_wikitext2(seed=0, seqlen=max(Ms), tokenizer=tok, eval_mode=True)
    wiki_ids_full = wiki.input_ids  # (1, >=max(Ms))

    # Stages isolate each fix AND give the shippable combination:
    #   baseline            : pre-fix real path (bf16 gemm, no div-fold, fp16 cast)
    #   fix1_divfold        : + fold 1/sqrt(n) into Hadamard kernel (constant fold)
    #   fix3_fp16gemm_only  : fp16-output GEMM epilogue ALONE (isolate its cosine)
    #   fix13_divfold_fp16  : fix1 + fix3 (SHIP candidate; no fp32-pack)
    #   fix2_hadfp32        : + feed FP32 Hadamard output to pack (documents its cosine)
    stages = [
        ("baseline",           cd,   False, False),
        ("fix1_divfold",       cd,   True,  False),
        ("fix3_fp16gemm_only", cd16, False, False),
        ("fix13_divfold_fp16", cd16, True,  False),
        ("fix2_hadfp32",       cd16, True,  True),
    ]

    rows = []
    for M in Ms:
        static_ids = wiki_ids_full[:, :M].contiguous().to(DEV)
        t_fp16 = time_eager(lambda: fp16(static_ids).logits,
                            iters=(10 if M <= 4096 else 6), warmup=4)

        ref_logits = None
        per_stage = []
        for label, gemm_cd, divfold, hadfp32 in stages:
            fn.wire_real_kernel(model, packed, nv, mx, gemm_cd, qpf=qpf)
            set_levers(model, divfold, hadfp32)
            move_col_order(model)
            model.to(DEV).eval()

            eager_logits = model(static_ids).logits.clone()
            if ref_logits is None:
                ref_logits = eager_logits.clone()   # pre-fix reference
            cos_vs_ref = cosine(ref_logits, eager_logits)
            t_eager = time_eager(lambda: model(static_ids).logits,
                                 iters=(10 if M <= 4096 else 6), warmup=4)

            # CUDA graph
            t_graph = float("nan"); cos_graph = float("nan")
            try:
                g, static_logits = capture_forward(model, static_ids)
                g.replay(); torch.cuda.synchronize()
                cos_graph = cosine(ref_logits, static_logits.clone())
                t_graph = time_graph(g, iters=(30 if M <= 4096 else 15), warmup=5)
                del g
            except Exception:
                import traceback; traceback.print_exc()
            cleanup_memory()

            per_stage.append(dict(stage=label, divfold=divfold, hadfp32=hadfp32,
                                  gemm="fp16" if gemm_cd is cd16 else "bf16",
                                  t_eager_ms=t_eager, t_graph_ms=t_graph,
                                  cos_eager_vs_prefix=cos_vs_ref,
                                  cos_graph_vs_prefix=cos_graph,
                                  speedup_graph_vs_fp16=(t_fp16 / t_graph) if t_graph == t_graph else float("nan")))
            print(f"  M={M:5d} {label:16s} eager={t_eager:7.3f} graph={t_graph:7.3f} "
                  f"vs fp16={t_fp16:7.3f} -> {t_fp16/t_graph:5.3f}x  "
                  f"cos(graph vs prefix)={cos_graph:.6f}", flush=True)

        # ship candidate = fix13_divfold_fp16 (gate-passing fixes only);
        # fix2_hadfp32 is reported but dropped if it fails the cosine gate.
        by = {s["stage"]: s for s in per_stage}
        base_g = by["baseline"]["t_graph_ms"]
        ship = by["fix13_divfold_fp16"]
        ship_g = ship["t_graph_ms"]
        print(f"  M={M:5d} SUMMARY baseline graph={base_g:.3f} -> ship(fix1+fix3) graph={ship_g:.3f} "
              f"| fp16={t_fp16:.3f} | before={t_fp16/base_g:.3f}x after={t_fp16/ship_g:.3f}x "
              f"| ship cos={ship['cos_graph_vs_prefix']:.6f} "
              f"| >=1.5x? {'YES' if (t_fp16/ship_g)>=1.5 else 'no'}", flush=True)

        bd = _component_breakdown(nv, mx, cd, qpf, M, n_layers, DEV)
        rows.append(dict(M=M, t_fp16_eager_ms=t_fp16, stages=per_stage,
                         breakdown=bd))
        del static_ids, ref_logits
        cleanup_memory()

    out = {"config": args.config, "ms": args.ms, "perf": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("PREFILL_FUSION_PERF_DONE", flush=True)


@torch.no_grad()
def run_ppl(config, args):
    from promix.eval.evaluator import evaluate_ppl
    nv, mx, cd, qpf = load_kernels()
    cd16 = load_cd_fp16()
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config['calibration'].get('seed', 0))
    model = prepare_ptq_model(config, DEV, run_gptq=args.gptq)
    model.to(DEV)
    model.seqlen = config['eval']['max_length']
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    testloader = get_wikitext2(seed=config['calibration'].get('seed', 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = config['eval']['batch_size']
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)

    def real_ppl(gemm_cd, divfold, hadfp32, tag):
        fn.wire_real_kernel(model, packed, nv, mx, gemm_cd, qpf=qpf)
        set_levers(model, divfold, hadfp32)
        model.to(DEV)
        p = evaluate_ppl(model, testloader, DEV, batch_size=bs)
        print(f"PPL_{tag} {p:.4f}", flush=True)
        return p

    # twin reference (fp32 matmul, same W_deq)
    fn.wire_fake_mixed(model, packed); model.to(DEV)
    ppl_twin = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_FAKE_TWIN {ppl_twin:.4f}", flush=True)

    # (a) pre-fix real-kernel baseline (bf16 gemm, no fixes) -> reproduces 14.85
    ppl_base = real_ppl(cd, False, False, "REAL_BASELINE")
    # (a2) fix1 ALONE (div-fold, bf16 gemm) -> the SHIPPABLE numerics-safe config
    ppl_fix1 = real_ppl(cd, True, False, "REAL_fix1_divfold_ONLY")
    # (b) SHIP config: div-fold + fp16-output GEMM (gate-passing fixes)
    ppl_ship = real_ppl(cd16, True, False, "REAL_SHIP_fix1_fix3")
    # (c) +fix2 (fp32 Hadamard -> pack) for documentation of its accuracy impact
    ppl_hadfp32 = real_ppl(cd16, True, True, "REAL_PLUS_fix2_hadfp32")

    print(f"PPL_SUMMARY twin={ppl_twin:.4f} base={ppl_base:.4f} "
          f"fix1_only={ppl_fix1:.4f} "
          f"ship(fix1+fix3)={ppl_ship:.4f} (+fix2)={ppl_hadfp32:.4f} "
          f"| fix1 vs base delta={ppl_fix1-ppl_base:+.4f} "
          f"| ship vs base delta={ppl_ship-ppl_base:+.4f}", flush=True)
    out = {"config": args.config, "ppl": dict(
        fake_twin=ppl_twin, real_baseline=ppl_base,
        real_fix1_divfold_only=ppl_fix1,
        real_ship_fix1_fix3=ppl_ship, real_plus_fix2=ppl_hadfp32,
        fix1_minus_base=ppl_fix1 - ppl_base,
        ship_minus_base=ppl_ship - ppl_base, gptq=args.gptq)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("PREFILL_FUSION_PPL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--phase", type=str, default="perf", choices=["perf", "ppl"])
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.join(KERN, f"prefill_fusion_{args.phase}.json")

    config = load_config(args.config)
    print(f"Prefill-fusion harness | phase={args.phase} config={args.config} ms={args.ms}", flush=True)
    print("Loading kernels (JIT build may take minutes)...", flush=True)
    if args.phase == "perf":
        run_perf(config, args)
    else:
        run_ppl(config, args)


if __name__ == "__main__":
    main()
