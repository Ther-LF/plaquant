"""Cross-size scaling harness: the SAME real-kernel path on 1B / 3B / 8B.

Every prior full-network number was Llama-3.2-1B-specific: the harness in
`full_network_realkernel.py` hardcodes the 1B decoder shapes for its component
breakdown, and `edge_fusion.py` hardcodes the 1B path shape for its probe. This
harness derives every shape from the model config and runs the full protocol for
an arbitrary Llama size:

  --phases ppl      wikitext-2 PPL gate (real kernel vs the size's fake-quant
                    baseline). The SHIPPABLE 1B wiring is used end to end:
                    qpfm (single-launch pack) + hpk (fused SwiGLU+Hadamard+pack
                    on down_proj) + cd_bias (epilogue bias/cast) + frozen GPTQ
                    scales, i.e. exactly the config that produced 1B REAL 14.8494
                    and prefill 1.29/1.35/1.30x.
  --phases prefill  full-model CUDA-graph prefill vs plain FP16 at M=2048/4096/
                    8192, median-of-N, with the same-run component breakdown
                    (pack / GEMM / FP16 rest / overhead), all 1B-shape constants
                    replaced by config-derived shapes.
  --phases decode   static-KV CUDA-graph decode, batch 1 and batch 8, prompt 32 /
                    gen 128, vs FP16 eager AND FP16 graphed, with the weight-
                    bandwidth ceiling for the model's actual packed bytes.

Fused down_proj producer note (HPK): the kernel in hadamard_pack_ext.cu is
templated but instantiated only at the power-of-2 row width I=8192 and its
register/chunk traits (kNChunks=8, kLogNChunks=3) assume a power-of-2 chunk
count. 1B and 3B have I=8192 -> fused producer ON. 8B has I=14336 = 7*2048 ->
the producer silently falls back to the reference chain
(silu*up -> fp32 Hadamard(had_K) -> fp16 -> pack), which is still real-kernel
for the GEMM and bit-identical arithmetically, just slower in the producer.
The report records this explicitly instead of pretending it is fused.

Remote-only (SM120 RTX PRO 6000). Kernels are JIT-loaded from the repo's
existing extensions; no PROTECTED file is touched.

Usage:
  python -m promix.eval.multisize --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
      --baseline_ppl 8.65 --phases ppl,prefill,decode --out /home/luofan/multisize_3b.json
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
from transformers import StaticCache
from torch.nn.attention import sdpa_kernel, SDPBackend

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.eval.data import get_wikitext2
from promix.eval.evaluator import evaluate_ppl
from promix.quantize import full_network as fn
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, HIGH_FRACTION,
)
from promix.eval.prefill_graph import capture_forward, time_eager, time_graph
from promix.eval.hadamard_swiglu_fusion import load_hpk, load_hpk28, load_qpfm, pick_free_gpu
from promix.eval.edge_fusion import load_cd_bias, set_levers
from promix.eval.decode_throughput import (
    set_decode_gemv, load_decode_gemv, tokenize_prompt,
    _decode_breakdown, _measure_hbm_ceiling, _median,
)

HPK_ROW = 8192  # the only instantiated fused-producer row width


def _wait_gpu(max_mem, max_util, timeout_s, min_free_mib=0, exclude=()):
    """Block until some GPU has <= max_mem MiB used, <= max_util % util and at
    least `min_free_mib` free. Serializes against the baseline agent."""
    import subprocess
    t0 = time.time()
    while True:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"]).decode()
        cands = []
        for line in out.strip().splitlines():
            i, u, m, tot = [x.strip() for x in line.split(",")]
            if i in exclude:
                continue
            if int(m) <= max_mem and int(u) <= max_util and (int(tot) - int(m)) >= min_free_mib:
                cands.append((int(m), int(i)))
        if cands:
            return min(cands)[1]
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no GPU <= {max_mem} MiB / {max_util}% util with "
                               f">= {min_free_mib} MiB free after {timeout_s}s")
        print(f"[multisize] waiting for idle GPU ({int(time.time()-t0)}s)", flush=True)
        time.sleep(30)


# ---------------------------------------------------------------------------
# config-derived decoder shapes
# ---------------------------------------------------------------------------
def layer_shapes(cfg):
    """[(name, N, K)] for the 7 target Linears of ONE layer, from the config.

    1B (hidden 2048 / inter 8192 / 32 heads / 8 kv heads) reproduces the
    hardcoded LAYER_SHAPES in full_network_realkernel.py exactly.
    """
    H = cfg.hidden_size
    I = cfg.intermediate_size
    hd = H // cfg.num_attention_heads
    q = cfg.num_attention_heads * hd
    kv = cfg.num_key_value_heads * hd
    return [("q_proj", q, H), ("k_proj", kv, H), ("v_proj", kv, H),
            ("o_proj", H, q), ("gate_proj", I, H), ("up_proj", I, H),
            ("down_proj", H, I)]


def model_desc(cfg):
    return dict(hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
                num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.hidden_size // cfg.num_attention_heads,
                vocab_size=cfg.vocab_size, fused_producer=(cfg.intermediate_size == HPK_ROW))


# ---------------------------------------------------------------------------
# kernels + model build (shippable wiring)
# ---------------------------------------------------------------------------
def load_all():
    nv, mx, cd, qpf = load_kernels()
    qpfm = load_qpfm()
    hpk = load_hpk()
    hpk28, had28 = load_hpk28()   # 8B (I=14336) fused producer + 28x28 Hadamard
    cd_bias = load_cd_bias()
    return nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias


def build_real(config, nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias, gptq,
               decode_gemv=False):
    """Same sequence as the 1B shipping path (edge_fusion.build): prepare the
    rotated/reordered/GPTQ model, pack all targets with GPTQ's frozen scales,
    then wire the real kernel with the full fusion set. The fused down_proj
    producer is `hpk` for I=8192 (1B/3B) and `hpk28` for I=14336 (8B)."""
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=gptq)
    model.to(DEV)
    model.config.use_cache = False
    I = model.config.intermediate_size
    hpk_use = hpk if I == HPK_ROW else None
    hpk28_use = hpk28 if I == 14336 else None
    had28_use = had28 if I == 14336 else None
    fused = (hpk_use is not None) or (hpk28_use is not None)
    print(f"[multisize] intermediate_size={I} -> fused down_proj producer "
          f"{'ON (hpk)' if hpk_use is not None else ('ON (hpk28)' if hpk28_use is not None else 'OFF (reference producer)')}",
          flush=True)
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, qpfm=qpfm, hpk=hpk_use,
                        cd_bias=cd_bias, hpk28=hpk28_use, had28=had28_use,
                        dg=(decode_gemv_ext if (decode_gemv and decode_gemv_ext) else None))
    fn.wire_fused_mlp(model)
    model.to(DEV).eval()
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
    return model, packed, fused


decode_gemv_ext = None


# ---------------------------------------------------------------------------
# PPL
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_ppl(config, model, packed, args, out, fused_producer):
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    model.seqlen = config['eval']['max_length']
    tl = get_wikitext2(seed=config['calibration'].get('seed', 0),
                       seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = config['eval']['batch_size']
    set_levers(model, a=True, b=False)
    t0 = time.time()
    ppl = evaluate_ppl(model, tl, DEV, batch_size=bs)
    dt = time.time() - t0
    base = args.baseline_ppl
    delta = ppl - base
    gate = delta <= 0.1
    print(f"PPL_REAL_KERNEL {ppl:.4f}  baseline(fake-quant) {base:.4f}  "
          f"delta {delta:+.4f}  gate(<=+0.1) {'PASS' if gate else 'FAIL'}  "
          f"({dt:.0f}s)", flush=True)
    out["ppl"] = dict(real_kernel=ppl, baseline_fakequant=base, delta=delta,
                      gate_pass=bool(gate), fused_producer=fused_producer,
                      seconds=dt)


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------
def _breakdown(nv, mx, cd, qpf, shapes, M, n_layers, dev):
    from promix.quantize.weight_pack import split_k
    from promix.quantize.mixed_linear import MixedQuantLinear
    pack_sum = gemm_sum = fp16_lin_sum = 0.0
    for name, N, K in shapes:
        K_low, K_high = split_k(K, HIGH_FRACTION)
        torch.manual_seed(hash(name) % 1000)
        W = (torch.randn(N, K, device=dev) * 0.08)
        x = torch.randn(M, K, device=dev)
        x[:, K_low:] *= 3.0
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)
        t_pack = time_eager(lambda: mql.pack_only(x), iters=20, warmup=5)
        A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

        def gemm_only():
            return cd.fused_gemm_prepacked(
                A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi, M, N, K_low, K_high)
        t_gemm = time_eager(gemm_only, iters=20, warmup=5)
        Wh = W.half(); xh = x.half()
        t_fp16 = time_eager(lambda: xh @ Wh.t(), iters=20, warmup=5)
        pack_sum += t_pack; gemm_sum += t_gemm; fp16_lin_sum += t_fp16
        del mql, W, x
    cleanup_memory()
    return dict(pack_ms=pack_sum * n_layers, gemm_ms=gemm_sum * n_layers,
                fp16_linear_ms=fp16_lin_sum * n_layers)


@torch.no_grad()
def run_prefill(model, fp16, config, nv, mx, cd, qpf, args, out):
    from promix.eval.data import get_wikitext2 as _wt
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    shapes = layer_shapes(model.config)
    n_layers = model.config.num_hidden_layers
    wik = _wt(seed=0, seqlen=max(args.ms), tokenizer=tok, eval_mode=True).input_ids
    rows = []
    for M in args.ms:
        ids = wik[:, :M].contiguous().to(DEV)
        rec = dict(M=M, shapes=[list(s) for s in shapes])
        ref = None
        try:
            t_fp16 = time_eager(lambda: fp16(ids).logits, iters=(10 if M <= 4096 else 6), warmup=4)
            ref = model(ids).logits.clone()
            g, sl = capture_forward(model, ids)
            g.replay(); torch.cuda.synchronize()
            cos_g = cosine(ref, sl.clone())
            t_graph = time_graph(g, iters=(30 if M <= 4096 else 15), warmup=5)
            del g
            # FP16 graph (fair graphed baseline)
            t_fp16_graph = float("nan")
            try:
                gf, _ = capture_forward(fp16, ids)
                t_fp16_graph = time_graph(gf, iters=(30 if M <= 4096 else 15), warmup=5)
                del gf
            except Exception as e:
                print(f"  fp16 graph failed: {e!r}", flush=True)
            bd = _breakdown(nv, mx, cd, qpf, shapes, M, n_layers, DEV)
            fp16_rest = t_fp16 - bd["fp16_linear_ms"]
            ideal = bd["pack_ms"] + bd["gemm_ms"] + fp16_rest
            ops = sum(N * K for _, N, K in shapes) * n_layers * 2 * M / 1e12
            rec.update(t_real_graph_ms=t_graph, t_fp16_eager_ms=t_fp16,
                       t_fp16_graph_ms=t_fp16_graph,
                       speedup_vs_fp16_eager=t_fp16 / t_graph,
                       speedup_vs_fp16_graph=(t_fp16_graph / t_graph
                                              if t_fp16_graph == t_fp16_graph else None),
                       tokens_s_real=M / (t_graph / 1e3),
                       tokens_s_fp16=M / (t_fp16 / 1e3),
                       pack_ms=bd["pack_ms"], gemm_ms=bd["gemm_ms"],
                       fp16_linear_ms=bd["fp16_linear_ms"], fp16_rest_ms=fp16_rest,
                       zero_overhead_ideal_ms=ideal,
                       overhead_graph_ms=t_graph - ideal,
                       linear_share_of_fp16_time=bd["fp16_linear_ms"] / t_fp16,
                       gemm_step_tflops=ops / (t_graph / 1e3),
                       cosine_graph_vs_eager=cos_g)
            print(f"  M={M:5d}: real(graph)={t_graph:8.3f} ms  fp16(eager)={t_fp16:8.3f} "
                  f"fp16(graph)={t_fp16_graph:.3f}  speedup={t_fp16/t_graph:.3f}x "
                  f"(vs fp16-graph {t_fp16_graph/t_graph if t_fp16_graph==t_fp16_graph else float('nan'):.3f}x)  "
                  f"cos={cos_g:.7f}", flush=True)
            print(f"         pack={bd['pack_ms']:.3f} gemm={bd['gemm_ms']:.3f} "
                  f"fp16-rest={fp16_rest:.3f} ideal={ideal:.3f} overhead={t_graph-ideal:.3f} ms "
                  f"| linear share of FP16 = {bd['fp16_linear_ms']/t_fp16*100:.1f}%", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  M={M}: OOM -> {e}", flush=True)
            rec["error"] = "OOM"
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  M={M}: ERROR -> {e!r}", flush=True)
            rec["error"] = repr(e)
            torch.cuda.empty_cache()
        rows.append(rec)
        del ids
        if ref is not None:
            del ref
        cleanup_memory()
    out["prefill"] = rows


# ---------------------------------------------------------------------------
# decode (static-KV graph, batch b)
# ---------------------------------------------------------------------------
def _eff_ctx():
    try:
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        import contextlib
        return contextlib.nullcontext()


@torch.no_grad()
def _decode_step(model, ids, pos, cpos, cache):
    with _eff_ctx():
        out = model(input_ids=ids, attention_mask=None, position_ids=pos,
                    past_key_values=cache, cache_position=cpos, use_cache=True,
                    num_logits_to_keep=1)
    return out.logits


@torch.no_grad()
def _capture_decode(model, ids, pos, cpos, cache, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            _decode_step(model, ids, pos, cpos, cache)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with _eff_ctx(), torch.cuda.graph(g):
        out = model(input_ids=ids, attention_mask=None, position_ids=pos,
                    past_key_values=cache, cache_position=cpos, use_cache=True,
                    num_logits_to_keep=1)
        logits = out.logits
    return g, logits


@torch.no_grad()
def _prefill_static(model, prompt, cache):
    P = prompt.shape[1]
    pos = torch.arange(P, device=DEV)
    with _eff_ctx():
        out = model(input_ids=prompt, attention_mask=None,
                    position_ids=pos.unsqueeze(0), past_key_values=cache,
                    cache_position=pos, use_cache=True)
    return out.logits[:, -1:, :].argmax(-1)


@torch.no_grad()
def _decode_graph_loop(g, s_in, s_pos, s_cpos, s_logits, s_next, t0, P, G):
    """G-step greedy decode replayed from one graph, 1 token/step, batch B.
    Returns per-step wall-clock (ms) with a sync each step (honest per-token)."""
    s_next.copy_(t0)
    per = []
    for i in range(G):
        torch.cuda.synchronize(); t = time.perf_counter()
        s_in.copy_(s_next)
        s_cpos.fill_(P + i)
        s_pos.fill_(P + i)
        g.replay()
        s_next.copy_(s_logits[:, -1, :].argmax(-1, keepdim=True))
        torch.cuda.synchronize(); per.append((time.perf_counter() - t) * 1e3)
    return per


@torch.no_grad()
def _decode_eager_loop(model, prompt, n_steps):
    """Batch-B greedy decode with a DynamicCache, timed per step (sync each
    step). decode_throughput.decode_logits_loop is batch-1 only (`.item()`);
    this is the same measurement generalized. Returns (per_step_ms,
    prefill_ms, tokens[B][n_steps])."""
    from transformers import DynamicCache
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cache = DynamicCache()
    out = model(input_ids=prompt, use_cache=True, past_key_values=cache)
    torch.cuda.synchronize(); prefill_ms = (time.perf_counter() - t0) * 1e3
    past = out.past_key_values
    cur = out.logits[:, -1:, :].argmax(-1)          # (B,1)
    per, toks = [], []
    for _ in range(n_steps):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(input_ids=cur, use_cache=True, past_key_values=past)
        torch.cuda.synchronize(); per.append((time.perf_counter() - t) * 1e3)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(cur[:, 0].tolist())
    return per, prefill_ms, toks


@torch.no_grad()
def run_decode(model, fp16, config, args, out, packed):
    tok, prompt1 = tokenize_prompt(config, args.prompt)
    P = prompt1.shape[1]
    G = args.gen
    hbm = _measure_hbm_ceiling(DEV)
    packed_bytes = sum(pk.W_lo_packed.numel() + pk.W_hi_bytes.numel()
                       + pk.physSFB_lo.numel() + pk.physSFB_hi.numel()
                       for pk in packed.values())
    fp16_bytes = sum(pk.N * pk.K * 2 for pk in packed.values())
    floor = packed_bytes / (hbm["use_tbps"] * 1e12) * 1e3
    fp16_floor = fp16_bytes / (hbm["use_tbps"] * 1e12) * 1e3
    rows = []
    for B in args.decode_batches:
        prompt = prompt1.repeat(B, 1).contiguous()
        print(f"  --- decode batch {B} (prompt {P} / gen {G}) ---", flush=True)
        # real decode-GEMV eager (shipped) + graphed
        set_decode_gemv(model, decode_gemv_ext)
        _decode_eager_loop(model, prompt, min(G, 16))            # warm
        ps_eager, pf_real, _ = _decode_eager_loop(model, prompt, G)
        med_eager = _median(ps_eager[2:])
        cache = StaticCache(config=model.config, max_batch_size=B,
                            max_cache_len=P + G + 1, device=DEV, dtype=torch.float16)
        t0 = _prefill_static(model, prompt, cache)
        s_in = t0.clone(); s_pos = torch.tensor([[P]], device=DEV)
        s_cpos = torch.tensor([P], device=DEV); s_next = t0.clone()
        g, s_logits = _capture_decode(model, s_in, s_pos, s_cpos, cache)
        per = _decode_graph_loop(g, s_in, s_pos, s_cpos, s_logits, s_next, t0, P, G)
        med_graph = _median(per[2:])
        del cache, g
        cleanup_memory()
        # FP16 eager + graphed
        _decode_eager_loop(fp16, prompt, min(G, 16))             # warm
        ps_fp16, pf_fp16, _ = _decode_eager_loop(fp16, prompt, G)
        med_fp16 = _median(ps_fp16[2:])
        med_fp16_g = float("nan")
        try:
            cf = StaticCache(config=fp16.config, max_batch_size=B,
                             max_cache_len=P + G + 1, device=DEV, dtype=torch.float16)
            t0f = _prefill_static(fp16, prompt, cf)
            f_in = t0f.clone(); f_pos = torch.tensor([[P]], device=DEV)
            f_cpos = torch.tensor([P], device=DEV); f_next = t0f.clone()
            gf, f_logits = _capture_decode(fp16, f_in, f_pos, f_cpos, cf)
            per_f = _decode_graph_loop(gf, f_in, f_pos, f_cpos, f_logits, f_next, t0f, P, G)
            med_fp16_g = _median(per_f[2:])
            del cf, gf
        except Exception as e:
            print(f"  fp16 decode graph failed: {e!r}", flush=True)
        cleanup_memory()
        rows.append(dict(batch=B, prompt=P, gen=G,
                         real_eager_ms=med_eager, real_graph_ms=med_graph,
                         fp16_eager_ms=med_fp16, fp16_graph_ms=med_fp16_g,
                         speedup_graph_vs_fp16_eager=med_fp16 / med_graph,
                         speedup_graph_vs_fp16_graph=(med_fp16_g / med_graph
                                                      if med_fp16_g == med_fp16_g else None),
                         tokens_s_real=1e3 / med_graph * B,
                         tokens_s_fp16=1e3 / med_fp16 * B,
                         prefill_real_ms=pf_real, prefill_fp16_ms=pf_fp16))
        su_e = med_fp16 / med_graph
        su_g = (med_fp16_g / med_graph) if med_fp16_g == med_fp16_g else float("nan")
        tps_r = 1e3 / med_graph * B
        print(f"  B={B}: REAL eager {med_eager:.3f} -> graph {med_graph:.3f} ms/tok "
              f"({tps_r:.1f} tok/s) | FP16 eager {med_fp16:.3f} -> graph "
              f"{med_fp16_g:.3f} ms/tok ({1e3/med_fp16_g*B if med_fp16_g==med_fp16_g else float('nan'):.1f} tok/s)",
              flush=True)
        print(f"       speedup vs FP16-eager {su_e:.3f}x | vs FP16-graph {su_g:.3f}x", flush=True)
    out["decode"] = rows
    out["weight_memory"] = dict(packed_bytes=packed_bytes, fp16_bytes=fp16_bytes,
                                sf_bytes=sum(pk.physSFB_lo.numel() + pk.physSFB_hi.numel()
                                             for pk in packed.values()),
                                compression=fp16_bytes / packed_bytes,
                                bits_per_weight=packed_bytes * 8 / (fp16_bytes / 2),
                                hbm_tbps=hbm["use_tbps"],
                                weight_floor_ms=floor, fp16_weight_floor_ms=fp16_floor)
    print(f"  weight memory: FP16 {fp16_bytes/1e6:.1f} MB -> packed {packed_bytes/1e6:.1f} MB "
          f"= {fp16_bytes/packed_bytes:.2f}x ({packed_bytes*8/(fp16_bytes/2):.2f} bits/w) | "
          f"BW floor @M=1 {floor:.3f} ms/tok", flush=True)


# ---------------------------------------------------------------------------
def main():
    global decode_gemv_ext
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phases", default="ppl,prefill,decode")
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--decode_batches", default="1,8")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--baseline_ppl", type=float, required=True)
    ap.add_argument("--gpu", type=int, default=None, help="pin a GPU; else idle-waiter")
    ap.add_argument("--idle_wait", action="store_true", default=False,
                    help="block until a GPU satisfies mem<=--idle_max_mem MiB and util<=--idle_max_util")
    ap.add_argument("--idle_max_mem", type=int, default=2000)
    ap.add_argument("--idle_max_util", type=int, default=3)
    ap.add_argument("--idle_timeout", type=int, default=3600)
    ap.add_argument("--min_free_mib", type=int, default=0,
                    help="require this much FREE memory on the chosen GPU (7B/8B: ~60000)")
    ap.add_argument("--exclude_gpus", type=str, default="",
                    help="comma-separated GPU indices the waiter must not pick")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.ms = [int(x) for x in args.ms.split(",")]
    args.decode_batches = [int(x) for x in args.decode_batches.split(",")]

    gpu = args.gpu if args.gpu is not None else pick_free_gpu()
    if args.idle_wait:
        exclude = tuple(int(x) for x in args.exclude_gpus.split(",") if x != "")
        gpu = _wait_gpu(args.idle_max_mem, args.idle_max_util, args.idle_timeout,
                        min_free_mib=args.min_free_mib, exclude=exclude)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[multisize] config={args.config} gpu={gpu} phases={args.phases}", flush=True)

    config = load_config(args.config)
    nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias = load_all()
    decode_gemv_ext = load_decode_gemv()
    print("[multisize] kernels loaded", flush=True)

    out = {"config": args.config, "gpu": gpu, "phases": args.phases,
           "model": None, "shapes": None}
    phases = set(args.phases.split(","))
    cfg_peek = transformers.AutoConfig.from_pretrained(config['model']['name'])
    out["model"] = model_desc(cfg_peek)
    out["shapes"] = [list(s) for s in layer_shapes(cfg_peek)]

    if "ppl" in phases:
        print("=== PPL (shippable wiring) ===", flush=True)
        model, packed, fused = build_real(config, nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias,
                                          gptq=True)
        run_ppl(config, model, packed, args, out, fused)
        del model, packed
        cleanup_memory()

    if "prefill" in phases:
        # Separate build: the fused down_proj producer (hpk) is only wired when
        # the decode-GEMV is absent (M==1 takes its own path), and the published
        # 1B prefill numbers were measured WITH it. Build a prefill-only model so
        # the cross-size table matches the 1B config exactly.
        print("=== building real model for PREFILL (no GPTQ) ===", flush=True)
        model, packed, fused = build_real(config, nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias,
                                          gptq=False, decode_gemv=False)
        fp16 = _build_fp16_model(config['model']['name'])
        fp16.config.use_cache = False
        print("=== PREFILL (CUDA graph) ===", flush=True)
        model.config.use_cache = False
        run_prefill(model, fp16, config, nv, mx, cd, qpf, args, out)
        del model, fp16, packed
        cleanup_memory()

    if "decode" in phases:
        print("=== building real model for DECODE (no GPTQ, decode-GEMV on) ===", flush=True)
        model, packed, fused = build_real(config, nv, mx, cd, qpf, qpfm, hpk, hpk28, had28, cd_bias,
                                          gptq=False, decode_gemv=True)
        fp16 = _build_fp16_model(config['model']['name'])
        fp16.config.use_cache = True
        print("=== DECODE (static-KV CUDA graph) ===", flush=True)
        model.config.use_cache = True
        run_decode(model, fp16, config, args, out, packed)
        del model, fp16, packed
        cleanup_memory()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("MULTISIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
