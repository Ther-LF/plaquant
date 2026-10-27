"""Round-10 Task-2 — Triton fused RMSNorm lever: PPL gate + prefill A/B.

Measures the biggest non-GEMM prefill lever found in Task 1 (RMSNorm). For the
SHIPPED real-kernel model (PROMIX_HAD_MODE defaults to fp16 now):

  phase `ppl`      : wikitext-2 PPL with HF RMSNorm vs Triton RMSNorm (real kernel).
                     Gate: |Δ| <= 0.01 (no regression vs shipped).
  phase `prefill`  : CUDA-graphed full-forward prefill, HF-norm vs Triton-norm,
                     + the FP16 baseline; min-of-N, idle-gated. Reports e2e ms and
                     the saving. (×vLLM-BF16 computed in the writeup from the
                     round-9 baseline TTFT: 3B 183.9ms, 8B 388.8ms @M8192.)
  phase `cosine`   : per-op (Triton vs HF norm output) and full-model logits cosine.

Usage (inside our stack, ncu container torch 2.11):
  CUDA_VISIBLE_DEVICES=2 python -m promix.eval.r10_rmsnorm_lever \
      --config promix/configs/llama-3-8b-fp-perhead.yaml \
      --ms 2048,4096,8192 --phases cosine,prefill,ppl --gptq \
      --out /home/luofan/_r10/r10_rms_8b.json
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
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, _component_breakdown, cosine, HIGH_FRACTION,
)
from promix.eval.prefill_graph import capture_forward, time_graph, time_eager
from promix.quantize.triton_rmsnorm import swap_rmsnorm, triton_rmsnorm
from promix.quantize.triton_rope import swap_rope, restore_rope


def wait_idle(util_thresh=8, need=3, timeout_s=180):
    import subprocess
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    t0 = time.time(); ok = 0
    while time.time() - t0 < timeout_s:
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--id=" + dev, "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
            util = int(q)
        except Exception:
            return
        if util <= util_thresh:
            ok += 1
            if ok >= need:
                return
        else:
            ok = 0
        time.sleep(1.0)


def build_real(config):
    nv, mx, cd, qpf = load_kernels()
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV)
    model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf)
    model.to(DEV).eval()
    n_moved = 0
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV)); n_moved += 1
    return model, (nv, mx, cd, qpf)


@torch.no_grad()
def phase_cosine(model, vocab, out):
    print("=== PHASE cosine (Triton RMSNorm vs HF) ===", flush=True)
    # per-op: run HF norm output vs triton norm output on random activations
    import transformers.models.llama.modeling_llama as ml
    hidden = model.config.hidden_size
    eps = model.config.rms_norm_eps
    w = None
    for mod in model.modules():
        if mod.__class__.__name__ == "LlamaRMSNorm":
            w = mod.weight; break
    perop = None
    if w is not None:
        x = torch.randn(1, 4096, hidden, device=DEV, dtype=torch.float16) * 0.2
        ref = (w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype))
        tri = triton_rmsnorm(x, w, eps)
        perop = cosine(ref, tri)
        print(f"  per-op RMSNorm cosine(triton vs HF) = {perop:.8f} "
              f"max|Δ|={(ref-tri).abs().max().item():.3e}", flush=True)
    # full-model logits cosine
    ids = torch.randint(0, vocab, (1, 2048), device=DEV)
    ref_logits = model(ids).logits.clone()
    n = swap_rmsnorm(model)
    tri_logits = model(ids).logits.clone()
    full = cosine(ref_logits, tri_logits)
    print(f"  swapped {n} RMSNorm; full-model logits cosine = {full:.8f}", flush=True)
    out["cosine"] = dict(per_op_rmsnorm=perop, full_model_logits=full, n_swapped=n)
    # leave model in HF-norm state for later phases that toggle explicitly
    return


@torch.no_grad()
def phase_prefill(model, fp16, kerns, Ms, out, rope=False):
    print("=== PHASE prefill (HF-norm vs Triton-norm, graphed) ===", flush=True)
    nv, mx, cd, qpf = kerns
    n_layers = model.config.num_hidden_layers
    vocab = model.config.vocab_size
    rows = []
    # ensure HF norm present
    for M in Ms:
        torch.manual_seed(M)
        static_ids = torch.randint(0, vocab, (1, M), device=DEV)

        # fp16 baseline (graph)
        t_fp16_graph = float("nan")
        try:
            gf, _ = capture_forward(fp16, static_ids)
            wait_idle(); t_fp16_graph = time_graph(gf, iters=(30 if M <= 4096 else 15), warmup=5)
            del gf
        except Exception as e:
            print("  fp16 capture failed", repr(e), flush=True)

        # --- HF norm (shipped) ---
        eager_out = model(static_ids).logits.clone()
        g_hf, logits_hf = capture_forward(model, static_ids)
        g_hf.replay(); torch.cuda.synchronize()
        cos_hf = cosine(eager_out, logits_hf.clone())
        wait_idle(); t_hf = time_graph(g_hf, iters=(30 if M <= 4096 else 15), warmup=5)
        del g_hf

        # --- Triton norm ---
        n_sw = swap_rmsnorm(model)
        tri_eager = model(static_ids).logits.clone()
        cos_tri_vs_hf = cosine(eager_out, tri_eager)
        g_tri, logits_tri = capture_forward(model, static_ids)
        g_tri.replay(); torch.cuda.synchronize()
        wait_idle(); t_tri = time_graph(g_tri, iters=(30 if M <= 4096 else 15), warmup=5)
        del g_tri

        # --- Triton norm + Triton RoPE (stacked) ---
        t_tri_rope = float("nan"); cos_rope_vs_hf = float("nan")
        if rope:
            orig_rope = swap_rope()
            try:
                rope_eager = model(static_ids).logits.clone()
                cos_rope_vs_hf = cosine(eager_out, rope_eager)
                g_r, logits_r = capture_forward(model, static_ids)
                g_r.replay(); torch.cuda.synchronize()
                wait_idle(); t_tri_rope = time_graph(g_r, iters=(30 if M <= 4096 else 15), warmup=5)
                del g_r
            except Exception as e:
                import traceback; traceback.print_exc()
            restore_rope(orig_rope)
        # restore HF norm for next M (rebuild not needed; re-swap is cheap but we
        # need HF back — swap replaces LlamaRMSNorm, now they are TritonRMSNorm, so
        # restore by reloading norm modules is complex; instead keep triton for the
        # rest of the sweep — subsequent M only needs triton timing, and HF timing
        # we already have per-M only if measured before swap). To measure HF per M
        # we must swap back: reconstruct HF norm from the shared weight.
        _restore_hf_norm(model)

        su_hf_vs_fp16 = t_fp16_graph / t_hf if t_hf == t_hf else float("nan")
        su_tri_vs_fp16 = t_fp16_graph / t_tri if t_tri == t_tri else float("nan")
        su_tri_rope_vs_fp16 = t_fp16_graph / t_tri_rope if t_tri_rope == t_tri_rope else float("nan")
        save = t_hf - t_tri
        save_rope = t_tri - t_tri_rope if t_tri_rope == t_tri_rope else float("nan")
        print(f"  M={M:5d}: fp16graph={t_fp16_graph:.3f} HF-norm={t_hf:.3f} "
              f"Triton-norm={t_tri:.3f} +RoPE={t_tri_rope:.3f}ms  save_norm={save:.3f} "
              f"save_rope={save_rope:.3f}ms | ×our-fp16 HF={su_hf_vs_fp16:.3f} "
              f"norm={su_tri_vs_fp16:.3f} norm+rope={su_tri_rope_vs_fp16:.3f} "
              f"| cos(HF graph vs eager)={cos_hf:.6f} cos(tri vs HF)={cos_tri_vs_hf:.6f} "
              f"cos(norm+rope vs HF)={cos_rope_vs_hf:.6f}", flush=True)
        rows.append(dict(M=M, t_fp16_graph_ms=t_fp16_graph, t_hfnorm_ms=t_hf,
                         t_tritonnorm_ms=t_tri, t_tritonnorm_rope_ms=t_tri_rope,
                         save_ms=save, save_rope_ms=save_rope,
                         su_hfnorm_vs_fp16=su_hf_vs_fp16,
                         su_tritonnorm_vs_fp16=su_tri_vs_fp16,
                         su_tritonnorm_rope_vs_fp16=su_tri_rope_vs_fp16,
                         cos_hf_graph_vs_eager=cos_hf, cos_triton_vs_hf=cos_tri_vs_hf,
                         cos_norm_rope_vs_hf=cos_rope_vs_hf,
                         n_swapped=n_sw))
        del static_ids, eager_out
        cleanup_memory()
    out["prefill"] = rows


def _restore_hf_norm(model):
    """Rebuild HF LlamaRMSNorm from TritonRMSNorm (shared weight) so the next
    A/B iteration starts from the shipped baseline."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from promix.quantize.triton_rmsnorm import TritonRMSNorm
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, TritonRMSNorm):
                m = LlamaRMSNorm(child.weight.shape[0], eps=child.variance_epsilon)
                m.weight = child.weight
                m.to(child.weight.device)
                setattr(parent, name, m)


@torch.no_grad()
def phase_ppl(config, model, out, rope=False):
    print("=== PHASE ppl (HF-norm vs Triton-norm, real kernel) ===", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    seqlen = config['eval']['max_length']
    testloader = get_wikitext2(seed=config['calibration'].get('seed', 0),
                               seqlen=seqlen, tokenizer=tokenizer, eval_mode=True)
    bs = config['eval']['batch_size']
    model.seqlen = seqlen
    _restore_hf_norm(model)
    ppl_hf = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"  PPL_HF_NORM {ppl_hf:.4f}", flush=True)
    n = swap_rmsnorm(model)
    ppl_tri = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"  PPL_TRITON_NORM {ppl_tri:.4f} (swapped {n})", flush=True)
    delta = ppl_tri - ppl_hf
    gate = abs(delta) <= 0.01
    print(f"  PPL_GATE Δ={delta:+.4f} |Δ|<=0.01 -> {'PASS' if gate else 'FAIL'}", flush=True)
    res = dict(ppl_hf_norm=ppl_hf, ppl_triton_norm=ppl_tri, delta=delta, gate_pass=gate)
    if rope:
        orig_rope = swap_rope()
        ppl_rope = evaluate_ppl(model, testloader, DEV, batch_size=bs)
        restore_rope(orig_rope)
        d2 = ppl_rope - ppl_hf
        g2 = abs(d2) <= 0.01
        print(f"  PPL_TRITON_NORM+ROPE {ppl_rope:.4f}  ΔvsHF={d2:+.4f} -> "
              f"{'PASS' if g2 else 'FAIL'}", flush=True)
        res.update(ppl_triton_norm_rope=ppl_rope, delta_norm_rope=d2, gate_pass_norm_rope=g2)
    out["ppl"] = res
    _restore_hf_norm(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--phases", default="cosine,prefill,ppl")
    ap.add_argument("--gptq", action="store_true", default=False)
    ap.add_argument("--rope", action="store_true", default=False,
                    help="also stack the Triton RoPE lever (#2 gap)")
    ap.add_argument("--out", default="/home/luofan/_r10/r10_rms.json")
    args = ap.parse_args()
    phases = set(args.phases.split(","))
    Ms = [int(x) for x in args.ms.split(",")]
    config = load_config(args.config)
    print(f"r10 RMSNorm lever | config={args.config} phases={phases} "
          f"HAD_MODE={os.environ.get('PROMIX_HAD_MODE','fp16(default)')}", flush=True)

    model, kerns = build_real(config)
    vocab = model.config.vocab_size
    out = {"config": args.config, "ms": args.ms,
           "had_mode": os.environ.get("PROMIX_HAD_MODE", "fp16(default)")}

    if "cosine" in phases:
        phase_cosine(model, vocab, out)
        _restore_hf_norm(model)
    if "prefill" in phases:
        fp16 = _build_fp16_model(config['model']['name'])
        n_moved = 0
        phase_prefill(model, fp16, kerns, Ms, out, rope=args.rope)
        del fp16
        cleanup_memory()
    if "ppl" in phases:
        phase_ppl(config, model, out, rope=args.rope)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("R10_RMSNORM_DONE", flush=True)


if __name__ == "__main__":
    main()
