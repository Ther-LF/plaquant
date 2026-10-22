"""Full-network real-kernel integration harness (SM120 NVFP4+MXFP8).

Scales the validated single-layer path (`single_layer_realkernel.py`) to the
WHOLE Llama-3.2-1B: all 16 decoder layers' 7 Linears (112 Linears) run the real
packed fused kernel. Produces the two end-to-end numbers the milestone wants:

  (A) PPL (accuracy gate) — wikitext-2 perplexity of
        * STOCK  : the fake-quant baseline (activation fake-quant + GPTQ FP
                   weights + FP16 matmul) == the ProMix FP NVFP4+MXFP8 recipe
                   (reference 14.92).
        * TWIN   : fake-quant twin (fake_quantize activation + FP32 matmul vs the
                   SAME dequantized weight the real kernel uses, no GPTQ-vs-RTN
                   ambiguity) — the tight numeric twin of the real kernel.
        * REAL   : the real packed fused-kernel model.
      GATE: |REAL - TWIN| <= 0.1 (isolates packing/scale/accumulation across the
      112 Linears). TWIN/REAL vs STOCK and FP16 give the absolute accuracy verdict.

  (B) PREFILL throughput — full-model forward latency (seq lengths giving
      M=2048/4096/8192) for the real-kernel model vs a plain FP16 model.
      tokens/s + speedup + an honest time breakdown (pack / GEMM / FP16-non-Linear
      / Python-dispatch overhead) across the 112 per-forward kernel launches.

Remote-only: run inside luofan-plaquant on a FREE GPU. Line-buffered stdout +
JSON to disk for harvest.

Usage:
  python -m promix.eval.full_network_realkernel \
      --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --phases ppl,prefill --ms 2048,4096,8192
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
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import yaml
import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import full_network as fn
from promix.quantize.mixed_linear import MixedQuantLinear

# Llama-3.2-1B decoder-layer Linear shapes (per layer): (name, N, K)
LAYER_SHAPES = [
    ("q_proj",   2048, 2048),
    ("k_proj",    512, 2048),
    ("v_proj",    512, 2048),
    ("o_proj",   2048, 2048),
    ("gate_proj", 8192, 2048),
    ("up_proj",   8192, 2048),
    ("down_proj", 2048, 8192),
]
HIGH_FRACTION = 0.125


def load_kernels():
    from torch.utils.cpp_extension import load
    import nvfp4_gemm as nv
    import mxfp8_gemm as mx
    nv.get_ext(); mx.get_ext()
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    cd = load(name="codesign_fused_ext",
              sources=[os.path.join(KERN, "codesign_fused_ext.cu")],
              extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    qpf = load(name="quant_pack_float_ext",
               sources=[os.path.join(KERN, "quant_pack_float_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    return nv, mx, cd, qpf


def time_ms(fn_, iters=30, warmup=8):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
# Phase A: PPL
# ---------------------------------------------------------------------------
def run_ppl(config, nv, mx, cd, qpf, args, out):
    print("=== PHASE A: FULL-NETWORK PPL ===", flush=True)
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config['calibration'].get('seed', 0))

    model = prepare_ptq_model(config, DEV, run_gptq=args.gptq)
    model.to(DEV)
    model.seqlen = config['eval']['max_length']
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    testloader = get_wikitext2(seed=config['calibration'].get('seed', 0),
                               seqlen=model.seqlen, tokenizer=tokenizer, eval_mode=True)
    bs = config['eval']['batch_size']

    ntargets = sum(1 for _ in fn.iter_target_wrappers(model))
    print(f"targets (7 x n_layers): {ntargets} Linears", flush=True)

    # Pack every target weight ONCE (shared by twin + real).
    print("Packing all target weights (offline)...", flush=True)
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)

    # (1) STOCK fake-quant baseline (the 14.92 recipe), model unchanged.
    ppl_stock = None
    if args.stock:
        model.to(DEV)
        ppl_stock = evaluate_ppl(model, testloader, DEV, batch_size=bs)
        print(f"PPL_STOCK_FAKEQUANT {ppl_stock:.4f}", flush=True)

    # (2) TWIN fake-quant (same W_deq as real, FP32 matmul).
    fn.wire_fake_mixed(model, packed)
    model.to(DEV)
    ppl_twin = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_FAKE_TWIN {ppl_twin:.4f}", flush=True)

    # (3) REAL packed fused kernel.
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)
    model.to(DEV)
    ppl_real = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_REAL_KERNEL {ppl_real:.4f}", flush=True)

    gate_delta = abs(ppl_real - ppl_twin)
    gate_pass = gate_delta <= 0.1
    print(f"PPL_GATE |real-twin|={gate_delta:.4f} <=0.1 -> "
          f"{'PASS' if gate_pass else 'FAIL'}", flush=True)
    out["ppl"] = dict(stock_fakequant=ppl_stock, fake_twin=ppl_twin, real_kernel=ppl_real,
                      gate_delta=gate_delta, gate_pass=gate_pass, reference_fp_ppl=14.92,
                      ntargets=ntargets, gptq=args.gptq)

    # Per-layer cosine spot-check (always a few; full sweep only if gate fails).
    try:
        spot = _per_layer_cosine(config, nv, mx, cd, qpf, packed,
                                 layers=None if not gate_pass else [0, 8, 15])
        out["per_layer_cosine"] = spot
    except Exception as e:
        import traceback; traceback.print_exc()
        out["per_layer_cosine"] = {"error": repr(e)}

    del model, packed
    cleanup_memory()
    return gate_pass


@torch.no_grad()
def _per_layer_cosine(config, nv, mx, cd, qpf, packed_unused, layers=None):
    """Capture real rotated activations at each target quantizer input for the
    chosen layers, then compare MixedQuantLinear vs FakeMixedLinear (twin) and vs
    fake_quantize reference, per Linear. Rebuilds a fresh capture model."""
    from promix.quantize.quant_utils import find_qlayers, ActQuantWrapper
    print("=== per-layer cosine spot-check ===", flush=True)
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    loader = get_wikitext2(seed=0, seqlen=2048, tokenizer=tokenizer, eval_mode=True)
    ids = loader.input_ids[:, :2048].to(DEV)

    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))
    want = set(layers)
    caps = {}
    hooks = []
    for name, short, wrapper in fn.iter_target_wrappers(model):
        li = int(name.split(".layers.")[1].split(".")[0])
        if li not in want:
            continue
        caps[name] = {"W": wrapper.module.weight.data.detach().clone(),
                      "wrapper": wrapper}

        def mk(nm):
            def pre_hook(mod, inp):
                if "x" not in caps[nm]:
                    caps[nm]["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1]).clone()
            return pre_hook
        hooks.append(wrapper.quantizer.register_forward_pre_hook(mk(name)))
    model.to(DEV).eval()
    model(ids)
    for h in hooks:
        h.remove()

    from promix.quantize.quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8
    results = []
    for name, cap in caps.items():
        if "x" not in cap:
            continue
        x = cap["x"].to(DEV); W = cap["W"].to(DEV)
        wrapper = cap["wrapper"]
        K = W.shape[1]
        K_low, K_high = fn._khigh_for(wrapper, K, HIGH_FRACTION)
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)
        # override split to the configured one
        y_real = mql(x).float()
        # twin
        from promix.quantize.weight_pack import pack_weight_mixed
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
        xq = torch.cat([fake_quantize_nvfp4(x[:, :K_low].float()),
                        fake_quantize_mxfp8(x[:, K_low:].float())], dim=-1)
        y_twin = (xq @ pk.W_deq.t()).float()
        c = cosine(y_real, y_twin)
        results.append(dict(name=name, K_low=K_low, K_high=K_high, cos_real_vs_twin=c))
        print(f"  {name:48s} cos(real vs twin)={c:.6f}", flush=True)
    del model
    cleanup_memory()
    worst = min((r["cos_real_vs_twin"] for r in results), default=1.0)
    print(f"per-layer cosine worst={worst:.6f}", flush=True)
    return dict(per_linear=results, worst=worst)


# ---------------------------------------------------------------------------
# Phase B: prefill throughput
# ---------------------------------------------------------------------------
def _build_fp16_model(model_name):
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    if cfg.tie_word_embeddings:
        cfg.tie_word_embeddings = False
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, config=cfg)
    m.config.use_cache = False
    return m.to(DEV).eval()


@torch.no_grad()
def _component_breakdown(nv, mx, cd, qpf, M, n_layers, dev):
    """Per-M: sum over the 7 shapes (x n_layers) of fused pack + GEMM (real-kernel
    ideal compute) and of FP16 cuBLAS Linear GEMM. Synthetic operands; latency is
    data-independent for these formats."""
    pack_sum = gemm_sum = fp16_lin_sum = 0.0
    for name, N, K in LAYER_SHAPES:
        from promix.quantize.weight_pack import split_k
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(hash(name) % 1000)
        W = (torch.randn(N, K, device=dev) * 0.08)
        x = torch.randn(M, K, device=dev)
        x[:, K_low:] *= 3.0
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)
        t_pack = time_ms(lambda: mql.pack_only(x))
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

        def gemm_only():
            return cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi, M, N, K_low, K_high)
        t_gemm = time_ms(gemm_only)
        Wh = W.half(); xh = x.half()
        t_fp16 = time_ms(lambda: xh @ Wh.t())
        pack_sum += t_pack; gemm_sum += t_gemm; fp16_lin_sum += t_fp16
        del mql, W, x
    cleanup_memory()
    return dict(pack_ms=pack_sum * n_layers, gemm_ms=gemm_sum * n_layers,
                fp16_linear_ms=fp16_lin_sum * n_layers)


@torch.no_grad()
def run_prefill(config, nv, mx, cd, qpf, args, out):
    print("=== PHASE B: FULL-MODEL PREFILL THROUGHPUT ===", flush=True)
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model_name = config['model']['name']

    # Real-kernel model fully on GPU. GEMM/pack latency is weight-value
    # independent, so skip GPTQ here (saves a full calibration pass).
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV)
    model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)
    model.to(DEV).eval()
    n_layers = model.config.num_hidden_layers

    fp16 = _build_fp16_model(model_name)

    rows = []
    Ms = [int(x) for x in args.ms.split(",")]
    vocab = model.config.vocab_size
    for M in Ms:
        ids = torch.randint(0, vocab, (1, M), device=DEV)

        def fwd_real():
            return model(ids).logits

        def fwd_fp16():
            return fp16(ids).logits

        it = 10 if M <= 4096 else 6
        t_real = time_ms(fwd_real, iters=it, warmup=4)
        t_fp16 = time_ms(fwd_fp16, iters=it, warmup=4)
        bd = _component_breakdown(nv, mx, cd, qpf, M, n_layers, DEV)
        fp16_rest = t_fp16 - bd["fp16_linear_ms"]
        overhead = t_real - bd["pack_ms"] - bd["gemm_ms"] - fp16_rest
        speedup = t_fp16 / t_real
        tps_real = M / (t_real / 1e3)
        tps_fp16 = M / (t_fp16 / 1e3)
        print(f"  M={M:5d}: real={t_real:.3f}ms ({tps_real:.0f} tok/s) | "
              f"fp16={t_fp16:.3f}ms ({tps_fp16:.0f} tok/s) | speedup={speedup:.3f}x",
              flush=True)
        print(f"         breakdown: pack={bd['pack_ms']:.3f} gemm={bd['gemm_ms']:.3f} "
              f"fp16-rest={fp16_rest:.3f} python/launch-overhead={overhead:.3f} ms "
              f"(fp16 linear={bd['fp16_linear_ms']:.3f})", flush=True)
        rows.append(dict(M=M, t_real_ms=t_real, t_fp16_ms=t_fp16, speedup=speedup,
                         tokens_s_real=tps_real, tokens_s_fp16=tps_fp16,
                         pack_ms=bd["pack_ms"], gemm_ms=bd["gemm_ms"],
                         fp16_linear_ms=bd["fp16_linear_ms"], fp16_rest_ms=fp16_rest,
                         python_overhead_ms=overhead))
    out["prefill"] = rows
    del model, fp16, packed
    cleanup_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--phases", type=str, default="ppl,prefill")
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--stock", action="store_true", default=True)
    ap.add_argument("--no-stock", dest="stock", action="store_false")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "full_network_realkernel.json"))
    args = ap.parse_args()

    phases = set(args.phases.split(","))
    config = load_config(args.config)
    print(f"Full-network real-kernel harness | config={args.config}", flush=True)
    print(f"  phases={phases} gptq={args.gptq} ms={args.ms}", flush=True)

    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    print(cd.compat_report() if hasattr(cd, "compat_report") else "cd loaded", flush=True)

    out = {"config": args.config, "ppl": None, "prefill": None, "per_layer_cosine": None}
    if "ppl" in phases:
        run_ppl(config, nv, mx, cd, qpf, args, out)
    if "prefill" in phases:
        run_prefill(config, nv, mx, cd, qpf, args, out)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("FULL_NETWORK_REALKERNEL_DONE", flush=True)


if __name__ == "__main__":
    main()
