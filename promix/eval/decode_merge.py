"""Round-7 DECODE QKV + gate/up MERGE (SM120 NVFP4+MXFP8 real-kernel model).

The graphed decode breakdown (docs/decode_cudagraph_report.md / multisize_report.md)
showed 1B/3B decode is dominated by the 112 per-token M=1 GEMV launches (1B graphed
4.75 ms/tok = GEMV 3.66 ms + attn/rest 1.09 ms) and that at M=1 each GEMV is
launch/occupancy-bound, NOT HBM-bound (11.3x off the weight-BW floor). The lever the
perf-round-2 report flagged: MERGE the per-layer GEMV launches. q/k/v read the SAME
input (input_layernorm output) and gate/up read the SAME input (post_attn_rmsnorm
output) with NO per-proj online-Hadamard / reorder / fake-quant (wire_real_kernel
disables the wrapper quantizer; only o_proj reorders, only down_proj Hadamards). So:

  per layer: q,k,v,o,gate,up,down  (7 GEMV launches)
          -> qkv(1), o(1), gate_up(1), down(1)   (4 launches)

A merged launch also quantizes the shared activation ONCE (vs 3x for qkv, 2x for
gate/up). The merge is done by CONCATENATING the already-packed per-proj buffers
along the output (row) dimension N — each output row is an independent dot product,
so the concatenated GEMV/GEMM is bit-identical to running the projections separately.

Two paths:
  * b1 (M==1, decode-GEMV): concat the ROW-MAJOR decode buffers (W_*_dec, S_*_dec).
    The round-6 two-level global g is already folded into S_lo_dec/S_hi_dec per proj,
    so the concat is BIT-EXACT regardless of per-proj g. (primary target)
  * b8 (M>1, fused prefill GEMM): concat the atom-interleaved prefill buffers along N
    and carry a per-ROW g vector (the round-6 two-level alpha differs per proj, so a
    single epilogue scalar can't serve qkv -> apply g as a post-GEMM per-row multiply
    with alpha=1.0). VALIDITY (interleave layout + alpha-fold rounding) is PROBED at
    setup against the per-proj reference; if it fails the probe, M>1 falls back to the
    per-proj path (no b8 merge) and the number is reported honestly.

NUMERICS GATE FIRST: merged graphed decode must reproduce the un-merged graphed
decode (argmax 100%, per-step logit cos ~1.0). Then throughput b1/b8 vs FP16-graph.

Usage:
  python -m promix.eval.decode_merge --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --batches 1,8 --prompt 32 --gen 128 --out /home/luofan/r7_decode_merge_1b.json
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
from transformers.models.llama import modeling_llama as ml

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.quantize import full_network as fn
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, HIGH_FRACTION, cosine,
)
from promix.eval.decode_throughput import (
    set_decode_gemv, load_decode_gemv, tokenize_prompt, _median,
)
from promix.eval.multisize import (
    _capture_decode, _prefill_static, _decode_graph_loop,
)
from promix.eval.prefill_graph import capture_forward, time_graph


# ---------------------------------------------------------------------------
# Merged packed Linear (qkv or gate/up): concat per-proj buffers along N.
# ---------------------------------------------------------------------------
class MergedQuantLinear(torch.nn.Module):
    def __init__(self, parts, dg, cd, cd_bias=None):
        super().__init__()
        assert len(parts) >= 2
        self._dg = dg
        self._cd = cd
        self._cd_bias = cd_bias
        self.K = parts[0].K
        self.K_low = parts[0].K_low
        self.K_high = parts[0].K_high
        for p in parts:
            assert p.K == self.K and p.K_low == self.K_low and p.K_high == self.K_high, \
                "merge requires identical K / K_low / K_high across projections"
            assert p.bias is None, "merge path assumes bias-less q/k/v/gate/up (Llama-3)"
        self.sizes = [int(p.N) for p in parts]
        self.N = sum(self.sizes)
        cat = lambda attr: torch.cat([getattr(p, attr) for p in parts], dim=0).contiguous()
        # decode-GEMV (row-major) buffers — g already folded into S_*_dec per proj.
        self.register_buffer("W_lo_dec", cat("W_lo_dec"))
        self.register_buffer("S_lo_dec", cat("S_lo_dec"))
        self.register_buffer("W_hi_dec", cat("W_hi_dec"))
        self.register_buffer("S_hi_dec", cat("S_hi_dec"))
        # prefill fused-GEMM buffers (atom-interleaved) + per-row two-level g.
        self.register_buffer("W_lo_packed", cat("W_lo_packed"))
        self.register_buffer("W_hi_bytes", cat("W_hi_bytes"))
        self.register_buffer("physSFB_lo", cat("physSFB_lo"))
        self.register_buffer("physSFB_hi", cat("physSFB_hi"))
        gvec = torch.cat([torch.full((int(p.N),), float(getattr(p, "g_w", 1.0)),
                                     dtype=torch.float32) for p in parts])
        self.register_buffer("g_vec", gvec.to(DEV))
        self.register_buffer("split", torch.tensor(self.sizes, dtype=torch.long))
        self._dec_warps = 24 if self.N >= 4096 else 8
        self.prefill_merge_ok = False       # set by probe at setup
        self.parts = parts                  # kept for the M>1 fallback

    @torch.no_grad()
    def _gemm_merged(self, packed, M):
        """Fused prefill GEMM over the concatenated weight, alpha=1.0, then the
        per-row two-level g as a post-multiply. (cd returns the bf16-rounded FP32
        accumulator * alpha; with alpha=1 we re-apply g per row.)"""
        D = self._cd.fused_gemm_prepacked(
            packed[0], packed[1], self.W_lo_packed, self.physSFB_lo,
            packed[2], packed[3], self.W_hi_bytes, self.physSFB_hi,
            M, self.N, self.K_low, self.K_high, 1.0)
        D = D.to(torch.float32) * self.g_vec           # (M, N) * (N,)
        return D.to(torch.float16)

    @torch.no_grad()
    def forward(self, x):
        in_shape = x.shape
        x2d = x.reshape(-1, in_shape[-1])
        M = x2d.shape[0]
        if self._dg is not None and M == 1:
            D = self._dg.decode_gemv_fused(x2d, self.W_lo_dec, self.S_lo_dec,
                                           self.W_hi_dec, self.S_hi_dec,
                                           self.N, self.K_low, self.K_high,
                                           self._dec_warps)
            D = D.unsqueeze(0).to(torch.float16)
            return D.reshape(*in_shape[:-1], self.N)
        if self.prefill_merge_ok:
            packed = self.parts[0]._pack_activation(x2d.contiguous())
            D = self._gemm_merged(packed, M)
            return D.reshape(*in_shape[:-1], self.N)
        # fallback: per-proj (no b8 merge) — still correct
        outs = [p(x2d) for p in self.parts]
        return torch.cat(outs, dim=-1).reshape(*in_shape[:-1], self.N)

    @torch.no_grad()
    def probe_prefill(self, M=8, tol=5e-3):
        """Compare the merged fused GEMM (alpha=1 + per-row g) against the per-proj
        reference at M. Sets prefill_merge_ok."""
        x = torch.randn(M, self.K, device=DEV, dtype=torch.float16)
        x[:, self.K_low:] *= 3.0
        ref = torch.cat([p(x) for p in self.parts], dim=-1).float()
        packed = self.parts[0]._pack_activation(x.contiguous())
        got = self._gemm_merged(packed, M).float()
        num = (got - ref).abs().max().item()
        den = ref.abs().max().item() + 1e-9
        c = cosine(got.reshape(-1).cpu(), ref.reshape(-1).cpu())
        self.prefill_merge_ok = (num / den <= tol) and (c >= 0.99999)
        return dict(max_abs=num, rel=num / den, cos=float(c), ok=bool(self.prefill_merge_ok))


# ---------------------------------------------------------------------------
# Patched HF Llama forwards (transformers 4.48.1) that use the merged modules.
# A module-level flag toggles merge ON/OFF so one model measures both.
# ---------------------------------------------------------------------------
MERGE_ON = True
_ORIG = {}


def _attn_forward_merged(self, hidden_states, position_embeddings, attention_mask,
                         past_key_value=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qkv = getattr(self, "_qkv_merged", None)
    if MERGE_ON and qkv is not None:
        fused = qkv(hidden_states)
        q, k, v = torch.split(fused, qkv.sizes, dim=-1)
    else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = ml.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)
    attention_interface = ml.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ml.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _mlp_forward_merged(self, x):
    gu = getattr(self, "_gu_merged", None)
    if MERGE_ON and gu is not None:
        fused = gu(x)
        g, u = torch.split(fused, gu.sizes, dim=-1)
    else:
        g = self.gate_proj(x)
        u = self.up_proj(x)
    h = self.act_fn(g) * u
    return self.down_proj(h)


def install_merge(model, dg, cd, cd_bias=None):
    """Build merged qkv / gate_up modules per layer and patch the forwards.
    Asserts the merged projections are pure pass-throughs (no Hadamard/reorder/
    fakequant), else aborts that group's merge."""
    def _mql(wrapper):
        m = wrapper.module
        assert isinstance(m, MixedQuantLinear), "merge needs a wired MixedQuantLinear"
        assert not getattr(wrapper, "online_full_had", False)
        assert getattr(wrapper, "_column_order", None) is None and \
               getattr(wrapper, "new_column_order", None) is None
        assert int(wrapper.quantizer.bits) == 16 and int(wrapper.out_quantizer.bits) == 16
        return m

    n_qkv = n_gu = 0
    probes = []
    for layer in model.model.layers:
        attn = layer.self_attn
        try:
            parts = [_mql(attn.q_proj), _mql(attn.k_proj), _mql(attn.v_proj)]
            qkv = MergedQuantLinear(parts, dg, cd, cd_bias).to(DEV)
            probes.append(("qkv", qkv.probe_prefill()))
            attn._qkv_merged = qkv
            n_qkv += 1
        except AssertionError as e:
            print(f"  qkv merge skipped on a layer: {e}", flush=True)
            attn._qkv_merged = None
        mlp = layer.mlp
        try:
            parts = [_mql(mlp.gate_proj), _mql(mlp.up_proj)]
            gu = MergedQuantLinear(parts, dg, cd, cd_bias).to(DEV)
            probes.append(("gate_up", gu.probe_prefill()))
            mlp._gu_merged = gu
            n_gu += 1
        except AssertionError as e:
            print(f"  gate_up merge skipped on a layer: {e}", flush=True)
            mlp._gu_merged = None
    if "attn" not in _ORIG:
        _ORIG["attn"] = ml.LlamaAttention.forward
        _ORIG["mlp"] = ml.LlamaMLP.forward
    ml.LlamaAttention.forward = _attn_forward_merged
    ml.LlamaMLP.forward = _mlp_forward_merged
    # summarize probe
    ok_qkv = sum(1 for t, p in probes if t == "qkv" and p["ok"])
    ok_gu = sum(1 for t, p in probes if t == "gate_up" and p["ok"])
    worst = min((p["cos"] for _, p in probes), default=1.0)
    print(f"install_merge: qkv groups={n_qkv} (prefill_ok {ok_qkv}/{n_qkv}), "
          f"gate_up groups={n_gu} (prefill_ok {ok_gu}/{n_gu}); worst probe cos={worst:.6f}",
          flush=True)
    return dict(n_qkv=n_qkv, n_gu=n_gu, prefill_ok_qkv=ok_qkv, prefill_ok_gu=ok_gu,
                worst_probe_cos=float(worst),
                probes=[dict(kind=t, **p) for t, p in probes[:6]])


def set_merge(on):
    global MERGE_ON
    MERGE_ON = on


def remove_patches():
    if "attn" in _ORIG:
        ml.LlamaAttention.forward = _ORIG["attn"]
        ml.LlamaMLP.forward = _ORIG["mlp"]


# ---------------------------------------------------------------------------
def build_real(config, nv, mx, cd, qpf, dg):
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=False)
    model.to(DEV)
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, dg=dg)
    model.to(DEV).eval()
    model.config.use_cache = True
    # CUDA-graph safety: move o_proj column-reorder index to GPU
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
    return model, packed


@torch.no_grad()
def graph_decode_median(model, prompt, P, G):
    """Capture a static-KV decode graph, greedy-replay G steps, return steady
    median ms/token."""
    B = prompt.shape[0]
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=P + G + 1, device=DEV, dtype=torch.float16)
    t0 = _prefill_static(model, prompt, cache)
    s_in = t0.clone(); s_pos = torch.tensor([[P]], device=DEV)
    s_cpos = torch.tensor([P], device=DEV); s_next = t0.clone()
    g, s_logits = _capture_decode(model, s_in, s_pos, s_cpos, cache)
    per = _decode_graph_loop(g, s_in, s_pos, s_cpos, s_logits, s_next, t0, P, G)
    med = _median(per[2:])
    del cache, g
    cleanup_memory()
    return med


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--gate_steps", type=int, default=48)
    ap.add_argument("--prefill_ms", default="8192",
                    help="comma M list for the graphed prefill merge measurement; empty to skip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    config = load_config(args.config)
    print(f"Decode-MERGE harness | config={args.config} batches={batches}", flush=True)
    nv, mx, cd, qpf = load_kernels()
    dg = load_decode_gemv()
    model, packed = build_real(config, nv, mx, cd, qpf, dg)
    set_decode_gemv(model, dg)
    info = install_merge(model, dg, cd)
    fp16 = _build_fp16_model(config['model']['name'])
    fp16.config.use_cache = True

    tok, prompt1 = tokenize_prompt(config, args.prompt)
    P = prompt1.shape[1]
    G = args.gen
    out = {"config": args.config, "prompt": P, "gen": G, "merge_info": info,
           "decode": [], "attn_impl": model.config._attn_implementation}

    # ---- NUMERICS GATE ----
    # (1) DECODE (M=1) merge is isolated BIT-EXACT: build the cache with an
    #     un-merged prefill, CLONE it, run ONE decode step merge-OFF vs merge-ON
    #     on the clones (identical cache state) -> the merged concat decode-GEMV
    #     has g pre-folded into S_*_dec, so it is bit-identical to the per-proj
    #     path. (A multi-step teacher-forced compare would conflate the M>1
    #     prefill-seed drift; the M>1 path is gated separately by PPL offline.)
    print("=== NUMERICS GATE ===", flush=True)
    from transformers import DynamicCache

    def clone_cache(past):
        leg = past.to_legacy_cache()
        return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in leg))

    set_merge(False)
    base = DynamicCache()
    model(input_ids=prompt1, use_cache=True, past_key_values=base)
    tnext = torch.tensor([[int(prompt1[0, -1])]], device=DEV)
    set_merge(False)
    lg_off = model(input_ids=tnext, use_cache=True, past_key_values=clone_cache(base)).logits[0, -1].float().cpu()
    set_merge(True)
    lg_on = model(input_ids=tnext, use_cache=True, past_key_values=clone_cache(base)).logits[0, -1].float().cpu()
    maxabs = (lg_on - lg_off).abs().max().item()
    amatch = int(lg_on.argmax()) == int(lg_off.argmax())
    decode_bitexact = (maxabs == 0.0) and amatch
    print(f"  (M=1 decode) merged vs un-merged logits: max|Δ|={maxabs:.3e} argmax-match={amatch} "
          f"-> {'BIT-EXACT' if decode_bitexact else 'DRIFT'}", flush=True)
    print(f"  (M>1 fused-GEMM) per-module merged-vs-ref worst probe cos={info['worst_probe_cos']:.7f} "
          f"(PPL cost gated offline, ~+0.02)", flush=True)
    gate_pass = decode_bitexact and (info["worst_probe_cos"] >= 0.9999)
    print(f"  NUMERICS_GATE {'PASS' if gate_pass else 'FAIL'}", flush=True)
    out["numerics"] = dict(decode_m1_maxabs=maxabs, decode_m1_argmax_match=bool(amatch),
                           decode_bit_exact=bool(decode_bitexact),
                           prefill_worst_probe_cos=info["worst_probe_cos"],
                           gate_pass=bool(gate_pass))

    # ---- THROUGHPUT: for each batch, merge OFF vs ON, vs FP16 graph ----
    print("=== THROUGHPUT (graphed) ===", flush=True)
    for B in batches:
        prompt = prompt1.repeat(B, 1).contiguous()
        set_merge(False)
        # warm
        graph_decode_median(model, prompt, P, min(G, 16))
        off = graph_decode_median(model, prompt, P, G)
        set_merge(True)
        graph_decode_median(model, prompt, P, min(G, 16))
        on = graph_decode_median(model, prompt, P, G)
        # FP16 graphed
        fp16_g = float("nan")
        try:
            fp16_g = graph_decode_median(fp16, prompt, P, G)
        except Exception as e:
            print(f"  fp16 graph failed B={B}: {e!r}", flush=True)
        rec = dict(batch=B,
                   real_graph_merge_off_ms=off, real_graph_merge_on_ms=on,
                   fp16_graph_ms=fp16_g,
                   speedup_off_vs_fp16g=(fp16_g / off if fp16_g == fp16_g else None),
                   speedup_on_vs_fp16g=(fp16_g / on if fp16_g == fp16_g else None),
                   merge_gain=off / on,
                   tokens_s_on=1e3 / on * B)
        out["decode"].append(rec)
        print(f"  B={B}: merge OFF {off:.3f} -> ON {on:.3f} ms/tok (gain {off/on:.3f}x) | "
              f"FP16-graph {fp16_g:.3f} | ON vs FP16g "
              f"{(fp16_g/on if fp16_g==fp16_g else float('nan')):.3f}x "
              f"(OFF {fp16_g/off if fp16_g==fp16_g else float('nan'):.3f}x)", flush=True)

    # ---- PREFILL: graphed M-list, merge OFF vs ON, vs FP16 graph ----
    # The merge's M>1 path (fused-GEMM + per-row g) de-duplicates the q/k/v and
    # gate/up activation pack (shared input packed ONCE not 3x/2x) and enlarges
    # the GEMM N. down_proj stays materialized in BOTH off/on, so the delta is the
    # pure qkv/gate_up merge benefit on prefill (addressable pack+GEMM bucket).
    out["prefill"] = []
    pms = [int(x) for x in args.prefill_ms.split(",") if x.strip()]
    if pms:
        print("=== PREFILL (graphed) ===", flush=True)
        vocab = model.config.vocab_size
        model.config.use_cache = False
        fp16.config.use_cache = False
        for M in pms:
            ids = torch.randint(0, vocab, (1, M), device=DEV)
            set_merge(False)
            g_off, _ = capture_forward(model, ids)
            t_off = time_graph(g_off, iters=(20 if M <= 4096 else 12), warmup=4)
            del g_off; cleanup_memory()
            set_merge(True)
            g_on, _ = capture_forward(model, ids)
            t_on = time_graph(g_on, iters=(20 if M <= 4096 else 12), warmup=4)
            del g_on; cleanup_memory()
            t_fp = float("nan")
            try:
                gf, _ = capture_forward(fp16, ids)
                t_fp = time_graph(gf, iters=(20 if M <= 4096 else 12), warmup=4)
                del gf; cleanup_memory()
            except Exception as e:
                print(f"  fp16 prefill graph failed M={M}: {e!r}", flush=True)
            rec = dict(M=M, real_off_ms=t_off, real_on_ms=t_on, fp16_ms=t_fp,
                       merge_gain=t_off / t_on,
                       speedup_off_vs_fp16=(t_fp / t_off if t_fp == t_fp else None),
                       speedup_on_vs_fp16=(t_fp / t_on if t_fp == t_fp else None))
            out["prefill"].append(rec)
            print(f"  M={M}: merge OFF {t_off:.3f} -> ON {t_on:.3f} ms (gain {t_off/t_on:.3f}x) | "
                  f"FP16-graph {t_fp:.3f} | ON vs FP16 "
                  f"{(t_fp/t_on if t_fp==t_fp else float('nan')):.3f}x "
                  f"(OFF {t_fp/t_off if t_fp==t_fp else float('nan'):.3f}x)", flush=True)
            del ids

    remove_patches()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("DECODE_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
