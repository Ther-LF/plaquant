"""Fine-grained per-stage profile of the ProMix real-kernel prefill forward.

Breaks the Llama-3.2-1B prefill forward into per-stage CUDA-event timers so the
e2e real-vs-FP16 gap is attributed HONESTLY to every sub-stage (not lumped into
one opaque "overhead"/"Hadamard" bucket). For the real-kernel model AND the FP16
baseline, per M (=2048/4096/8192), we time:

  embed, rotary/mask prep (residual),
  per decoder layer:
    input_rmsnorm, q/k/v Linear (each: pack vs GEMM vs cast+bias vs wrapper-glue),
    qkv view/transpose, RoPE, sdpa (QK^T+softmax+PV fused), attn reshape/contig,
    o_proj (reorder + pack + GEMM + cast), residual add,
    post_attn_rmsnorm, gate/up Linear, SiLU*mul, down_proj (FP32 Hadamard + pack +
    GEMM + cast), residual add,
  final_norm, lm_head.

Two measurement passes:
  (1) CUDA-event stage timing (minimal host distortion) accumulated over N iters,
      summed across the 16 layers → the reconciling per-stage ms table. The sum is
      cross-checked against the clean (uninstrumented) whole-model latency.
  (2) torch.profiler op-level trace (record_function stage annotations + raw aten
      ops) to catch hidden dtype casts / reshapes / D2D copies / launch gaps.

Profiling/instrumentation ONLY — no kernel or algorithm change. PROTECTED kernel
files untouched.

Usage:
  python -m promix.eval.prefill_stage_profile \
      --config promix/configs/llama-3.2-1b-fp-perhead.yaml \
      --ms 2048,4096,8192 --iters 20 --models real,fp16 --profiler
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from transformers.models.llama import modeling_llama as ml

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import (
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.quantize import full_network as fn
from promix.quantize.quant_utils import ActQuantWrapper
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.quantize.hadamard import matmul_hadU_cuda
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, HIGH_FRACTION,
)


# ---------------------------------------------------------------------------
# Stage profiler: CUDA-event regions + optional record_function
# ---------------------------------------------------------------------------
class StageProf:
    def __init__(self):
        self.events = defaultdict(list)   # name -> [(iter_idx, start_evt, end_evt), ...]
        self.enabled = False              # record CUDA events
        self.record_fn = False            # also emit torch.profiler record_function
        self.cur_iter = 0

    @contextmanager
    def region(self, name):
        if not self.enabled and not self.record_fn:
            yield
            return
        if self.enabled:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
        if self.record_fn:
            with torch.profiler.record_function(name):
                yield
        else:
            yield
        if self.enabled:
            e.record()
            self.events[name].append((self.cur_iter, s, e))

    def per_iter(self, n_iters):
        """Return {name: [ms_iter0, ms_iter1, ...]} summing all occurrences of a
        stage WITHIN each iteration (per-layer stages occur 16x per forward)."""
        torch.cuda.synchronize()
        out = {name: [0.0] * n_iters for name in self.events}
        for name, triples in self.events.items():
            for it, s, e in triples:
                if 0 <= it < n_iters:
                    out[name][it] += s.elapsed_time(e)
        return out

    def clear(self):
        self.events.clear()
        self.cur_iter = 0


PROF = StageProf()


# A non-overlapping PARTITION of the forward (used for the reconciliation sum).
# The Linear *totals* (attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj) are the
# leaves at the "one Linear" granularity; their internal lin.* sub-regions
# (pack/gemm/cast/hadamard/reorder) are a DRILL-DOWN reported separately and are
# NOT added into the partition sum. `mlp.total` is a parent of its children and
# is excluded. This partition is valid for BOTH the real and fp16 models.
PARTITION = {
    "input_rmsnorm", "residual_add1", "post_attn_rmsnorm", "residual_add2",
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "attn.qkv_view_transpose", "attn.rope", "attn.reshape_contig",
    "attn.sdpa_qk_softmax_pv",
    "attn.eager_qkT", "attn.eager_softmax", "attn.eager_pv",
    "mlp.gate_proj", "mlp.up_proj", "mlp.silu_mul", "mlp.down_proj",
    "embed", "rotary_emb", "final_norm", "lm_head",
}
# Internal drill-down of the Linear totals (sum of these == Linear-total minus
# per-Linear python/reshape glue). Reported separately, excluded from partition.
LIN_INTERNAL = {
    "lin.pack", "lin.gemm", "lin.cast_bias", "lin.down_hadamard", "lin.down_fused",
    "lin.o_reorder", "lin.fakequant", "lin.out_quant",
}


# ---------------------------------------------------------------------------
# Instrumented forwards (HF transformers 4.48.1 Llama).  These REPLACE the
# class .forward methods while profiling; they call the exact same sub-ops,
# only wrapped in PROF.region(...).  Restored on teardown.
# ---------------------------------------------------------------------------
def _mlp_forward(self, x):
    with PROF.region("mlp.gate_proj"):
        g = self.gate_proj(x)
    with PROF.region("mlp.up_proj"):
        u = self.up_proj(x)
    with PROF.region("mlp.silu_mul"):
        h = self.act_fn(g) * u
    with PROF.region("mlp.down_proj"):
        d = self.down_proj(h)
    return d


def _attn_forward(self, hidden_states, position_embeddings, attention_mask,
                  past_key_value=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    with PROF.region("attn.q_proj"):
        q = self.q_proj(hidden_states)
    with PROF.region("attn.k_proj"):
        k = self.k_proj(hidden_states)
    with PROF.region("attn.v_proj"):
        v = self.v_proj(hidden_states)
    with PROF.region("attn.qkv_view_transpose"):
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    with PROF.region("attn.rope"):
        query_states, key_states = ml.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = ml.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ml.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    with PROF.region("attn.sdpa_qk_softmax_pv"):
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs)

    with PROF.region("attn.reshape_contig"):
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    with PROF.region("attn.o_proj"):
        attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _attn_forward_eager(self, hidden_states, position_embeddings, attention_mask,
                        past_key_value=None, cache_position=None, **kwargs):
    """Variant that forces the eager QK^T/softmax/PV split so those three sub-ops
    can be timed individually (the model normally runs sdpa as one fused kernel)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    with PROF.region("attn.q_proj"):
        q = self.q_proj(hidden_states)
    with PROF.region("attn.k_proj"):
        k = self.k_proj(hidden_states)
    with PROF.region("attn.v_proj"):
        v = self.v_proj(hidden_states)
    with PROF.region("attn.qkv_view_transpose"):
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    with PROF.region("attn.rope"):
        query_states, key_states = ml.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_rep = ml.repeat_kv(key_states, self.num_key_value_groups)
    value_rep = ml.repeat_kv(value_states, self.num_key_value_groups)
    with PROF.region("attn.eager_qkT"):
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_rep.shape[-2]]
    with PROF.region("attn.eager_softmax"):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    with PROF.region("attn.eager_pv"):
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous()
    with PROF.region("attn.reshape_contig"):
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    with PROF.region("attn.o_proj"):
        attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _decoder_forward(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kwargs):
    residual = hidden_states
    with PROF.region("input_rmsnorm"):
        hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_value=past_key_value,
        output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
    with PROF.region("residual_add1"):
        hidden_states = residual + hidden_states
    residual = hidden_states
    with PROF.region("post_attn_rmsnorm"):
        hidden_states = self.post_attention_layernorm(hidden_states)
    with PROF.region("mlp.total"):
        hidden_states = self.mlp(hidden_states)
    with PROF.region("residual_add2"):
        hidden_states = residual + hidden_states
    return (hidden_states,)


# ActQuantWrapper.forward — time Hadamard + reorder (the real-model-only glue).
# The inner module (MixedQuantLinear / nn.Linear) is timed inside its own forward.
def _wrapper_forward(self, x, column_order=None, **kwargs):
    from promix.quantize.quant_utils import _quant_enabled
    x_dtype = x.dtype
    col_order = None
    if column_order is not None:
        if column_order.dtype in (torch.long, torch.int, torch.int32, torch.int64,
                                   torch.uint8, torch.bool):
            col_order = column_order
        else:
            kwargs.setdefault("R1", column_order)
    if self.online_full_had:
        with PROF.region("lin.down_hadamard"):
            if self.fp32_had:
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = matmul_hadU_cuda(x, self.had_K, self.K)
    order = col_order if col_order is not None else getattr(self, '_column_order', None)
    if order is not None and self.reorder_before_quant:
        with PROF.region("lin.o_reorder"):
            x = x[..., order]
        order = None
    if _quant_enabled(self.quantizer.bits):
        with PROF.region("lin.fakequant"):
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
    if order is not None:
        with PROF.region("lin.o_reorder"):
            x = x[..., order]
    inner = {k: v for k, v in kwargs.items() if k in self._inner_kwargs}
    x = self.module(x, **inner).to(x_dtype)
    if _quant_enabled(self.out_quantizer.bits):
        with PROF.region("lin.out_quant"):
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()
    return x


def _mixed_forward(self, x):
    in_shape = x.shape
    x2d = x.reshape(-1, in_shape[-1])
    M = x2d.shape[0]
    with PROF.region("lin.pack"):
        A_lo, pSFA_lo, A_hi, pSFA_hi = self._pack_activation(x2d)
    with PROF.region("lin.gemm"):
        D = self._cd.fused_gemm_prepacked(
            A_lo, pSFA_lo, self.W_lo_packed, self.physSFB_lo,
            A_hi, pSFA_hi, self.W_hi_bytes, self.physSFB_hi,
            M, self.N, self.K_low, self.K_high)
    with PROF.region("lin.cast_bias"):
        D = D.to(self.out_dtype)
        if self.bias is not None:
            D = D + self.bias.to(D.dtype)
        D = D.reshape(*in_shape[:-1], self.N)
    return D


_ORIG = {}


def _mlp_forward_fused(self, x):
    """Fused down_proj producer: hand gate/up to the wrapper's fused kernel."""
    with PROF.region("mlp.gate_proj"):
        g = self.gate_proj(x)
    with PROF.region("mlp.up_proj"):
        u = self.up_proj(x)
    d = self.down_proj
    if getattr(d, "fused_swiglu_had", False) and hasattr(d.module, "forward_swiglu_pack"):
        with PROF.region("mlp.down_proj"):
            return d(x, mlp_gate=g, mlp_up=u)
    with PROF.region("mlp.silu_mul"):
        h = self.act_fn(g) * u
    with PROF.region("mlp.down_proj"):
        return d(h)


def _wrapper_forward_fused(self, x, column_order=None, **kwargs):
    gate = kwargs.pop("mlp_gate", None); up = kwargs.pop("mlp_up", None)
    if self.fused_swiglu_had and gate is not None and up is not None \
            and hasattr(self.module, "forward_swiglu_pack"):
        with PROF.region("lin.down_fused"):
            return self.module.forward_swiglu_pack(gate, up).to(x.dtype)
    return _wrapper_forward_orig(self, x, column_order=column_order, **kwargs)


def install_patches(eager_attn=False, fused=False):
    global _wrapper_forward_orig
    _wrapper_forward_orig = _wrapper_forward
    _ORIG["mlp"] = ml.LlamaMLP.forward
    _ORIG["mlp"] = ml.LlamaMLP.forward
    _ORIG["attn"] = ml.LlamaAttention.forward
    _ORIG["dec"] = ml.LlamaDecoderLayer.forward
    _ORIG["wrap"] = ActQuantWrapper.forward
    _ORIG["mixed"] = MixedQuantLinear.forward
    ml.LlamaMLP.forward = _mlp_forward_fused if fused else _mlp_forward
    ml.LlamaAttention.forward = _attn_forward_eager if eager_attn else _attn_forward
    ml.LlamaDecoderLayer.forward = _decoder_forward
    ActQuantWrapper.forward = _wrapper_forward_fused if fused else _wrapper_forward
    MixedQuantLinear.forward = _mixed_forward


def remove_patches():
    ml.LlamaMLP.forward = _ORIG["mlp"]
    ml.LlamaAttention.forward = _ORIG["attn"]
    ml.LlamaDecoderLayer.forward = _ORIG["dec"]
    ActQuantWrapper.forward = _ORIG["wrap"]
    MixedQuantLinear.forward = _ORIG["mixed"]


# one-time module stages (embed / final norm / lm_head) via forward hooks
_ONCE_EVENTS = {}


def install_once_hooks(model):
    handles = []
    targets = {
        "embed": model.model.embed_tokens,
        "final_norm": model.model.norm,
        "lm_head": model.lm_head,
    }
    rot = getattr(model.model, "rotary_emb", None)
    if rot is not None:
        targets["rotary_emb"] = rot

    def mk(name):
        def pre(mod, inp):
            if not PROF.enabled:
                return
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            _ONCE_EVENTS[name] = s

        def post(mod, inp, out):
            if not PROF.enabled or name not in _ONCE_EVENTS:
                return
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            PROF.events[name].append((PROF.cur_iter, _ONCE_EVENTS.pop(name), e))
        return pre, post

    for nm, mod in targets.items():
        pre, post = mk(nm)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    return handles


# ---------------------------------------------------------------------------
def wait_for_idle(util_thresh=8, need=3, timeout_s=300):
    """Block until this CUDA device's utilization is low for `need` consecutive
    samples. Memory is NOT gated: our own loaded model holds several GB, so only
    a co-tenant's compute utilization is a meaningful contention signal."""
    import subprocess
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    t0 = time.time()
    ok = 0
    while time.time() - t0 < timeout_s:
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--id=" + dev,
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
            util = int(q)
        except Exception as e:
            print(f"  idle-waiter: nvidia-smi failed ({e!r}); proceeding", flush=True)
            return
        if util <= util_thresh:
            ok += 1
            if ok >= need:
                return
        else:
            ok = 0
        time.sleep(1.5)
    print(f"  idle-waiter: timeout after {timeout_s}s; proceeding anyway", flush=True)


def time_clean(fn_, iters, warmup):
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


@torch.no_grad()
def profile_model(model, label, M, vocab, iters, warmup, eager_attn=False, fused=False):
    ids = torch.randint(0, vocab, (1, M), device=DEV)

    def fwd():
        return model(ids).logits

    # (1) clean whole-model latency (no instrumentation), min-of-N
    PROF.enabled = False
    PROF.record_fn = False
    wait_for_idle()
    t_clean_min = time_clean(fwd, iters, warmup)

    # (2) instrumented stage timing; CUDA-event regions tagged per-iteration so
    # we can select the MIN-total iteration (dodges transient co-tenant spikes
    # the same way min-of-N does for the whole-model number).
    PROF.clear()
    _ONCE_EVENTS.clear()
    install_patches(eager_attn=eager_attn, fused=fused)
    hooks = install_once_hooks(model)
    PROF.enabled = True
    PROF.record_fn = False
    torch.cuda.synchronize()
    for _ in range(warmup):
        fwd()
    PROF.clear()
    _ONCE_EVENTS.clear()
    wait_for_idle()
    for i in range(iters):
        PROF.cur_iter = i
        fwd()
    per_iter = PROF.per_iter(iters)
    PROF.enabled = False
    for h in hooks:
        h.remove()
    remove_patches()

    # choose the iteration with the smallest PARTITION-sum (cleanest forward)
    part_sums = []
    for i in range(iters):
        part_sums.append(sum(v[i] for k, v in per_iter.items() if k in PARTITION))
    best_i = min(range(iters), key=lambda i: part_sums[i])
    stages = {k: v[best_i] for k, v in per_iter.items()}

    partition_sum = sum(v for k, v in stages.items() if k in PARTITION)
    lin_internal_sum = sum(v for k, v in stages.items() if k in LIN_INTERNAL)
    lin_totals = sum(v for k, v in stages.items()
                     if k in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    per_linear_glue = lin_totals - lin_internal_sum
    residual = t_clean_min - partition_sum
    # The real model is launch-bound: per-op CUDA-event instrumentation spaces out
    # the 112-Linear launch stream and clocks the GPU down, inflating each region
    # vs the dense clean run (partition_sum > clean). FP16 (few big kernels) is not
    # affected (scale~1). We keep RAW stage times and also report stages NORMALIZED
    # so the partition reconciles to the ground-truth clean_min (preserves the
    # measured per-stage proportions). Cross-checked against the artifact-free
    # isolated dense pack/GEMM/Hadamard timings (cross_check phase) + torch.profiler.
    scale = (t_clean_min / partition_sum) if partition_sum > 0 else 1.0
    stages_norm = {k: v * scale for k, v in stages.items()}
    print(f"  [{label}] M={M} eager_attn={eager_attn}: clean_min={t_clean_min:.3f}ms "
          f"best_iter={best_i} partition_sum_raw={partition_sum:.3f}ms (scale={scale:.3f}) "
          f"residual={residual:.3f}ms (lin_total={lin_totals:.3f} "
          f"lin_internal={lin_internal_sum:.3f} per_linear_glue={per_linear_glue:.3f})",
          flush=True)
    del ids
    return dict(label=label, M=M, eager_attn=eager_attn,
                t_clean_min_ms=t_clean_min, best_iter=best_i,
                partition_sum_ms=partition_sum, norm_scale=scale,
                lin_total_ms=lin_totals, lin_internal_ms=lin_internal_sum,
                per_linear_glue_ms=per_linear_glue,
                residual_ms=residual, stages=stages, stages_norm=stages_norm)


@torch.no_grad()
def run_torch_profiler(model, label, M, vocab, out_dir):
    ids = torch.randint(0, vocab, (1, M), device=DEV)

    def fwd():
        return model(ids).logits

    PROF.enabled = False
    PROF.record_fn = True
    install_patches(eager_attn=False)
    hooks = install_once_hooks(model)
    torch.cuda.synchronize()
    for _ in range(3):
        fwd()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(5):
            fwd()
        torch.cuda.synchronize()
    PROF.record_fn = False
    for h in hooks:
        h.remove()
    remove_patches()

    tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    path = os.path.join(out_dir, f"profiler_{label}_M{M}.txt")
    with open(path, "w") as f:
        f.write(tbl)
    # structured top ops
    ka = prof.key_averages()
    rows = []
    for e in ka:
        cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        rows.append(dict(name=e.key, cuda_total_us=float(cuda_us),
                         count=int(e.count)))
    rows.sort(key=lambda r: r["cuda_total_us"], reverse=True)
    print(f"  [profiler {label} M={M}] wrote {path}", flush=True)
    del ids
    return rows[:40]


@torch.no_grad()
def run_cross_check(nv, mx, cd, qpf, model, Ms, n_layers):
    """Artifact-free cross-check of the launch-bound stages, each measured in its
    OWN dense loop (no per-op event spacing): isolated pack+GEMM per shape (summed
    x16 layers), and the FP32 online Hadamard on down_proj (16x). These are the
    un-inflated ground truth for the dominant real-only stages."""
    from promix.eval.full_network_realkernel import _component_breakdown, time_ms
    rows = {}
    # down_proj online Hadamard (FP32), the real-model-only cost
    hw = [w for _, _, w in fn.iter_target_wrappers(model)
          if getattr(w, "online_full_had", False)]
    for M in Ms:
        wait_for_idle()
        bd = _component_breakdown(nv, mx, cd, qpf, M, n_layers, DEV)
        had = {"per_call_fp32_us": None, "total_fp32_ms": None, "per_call_fp16_us": None}
        if hw:
            w0 = hw[0]
            K = w0.module.K
            x16 = torch.randn(1, M, K, device=DEV, dtype=torch.float16)
            t32 = time_ms(lambda: matmul_hadU_cuda(x16.float(), w0.had_K, w0.K).to(torch.float16),
                          iters=30, warmup=8)
            t16 = time_ms(lambda: matmul_hadU_cuda(x16, w0.had_K, w0.K), iters=30, warmup=8)
            had = {"per_call_fp32_us": t32 * 1e3, "total_fp32_ms": t32 * len(hw),
                   "per_call_fp16_us": t16 * 1e3, "total_fp16_ms": t16 * len(hw),
                   "n_down_proj": len(hw)}
            del x16
        rows[M] = {"pack_ms": bd["pack_ms"], "gemm_ms": bd["gemm_ms"],
                   "fp16_linear_ms": bd["fp16_linear_ms"], "hadamard": had}
        print(f"  [cross-check M={M}] isolated pack={bd['pack_ms']:.3f} gemm={bd['gemm_ms']:.3f} "
              f"fp16_linear={bd['fp16_linear_ms']:.3f} | Hadamard fp32 total={had['total_fp32_ms']}",
              flush=True)
        cleanup_memory()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--models", type=str, default="real,fp16")
    ap.add_argument("--eager-attn", action="store_true", default=False,
                    help="also run a forced-eager-attention pass to split QK^T/softmax/PV")
    ap.add_argument("--profiler", action="store_true", default=False)
    ap.add_argument("--fused", action="store_true", default=False,
                    help="use the fused SwiGLU+Hadamard+pack down_proj producer")
    ap.add_argument("--cross-check", action="store_true", default=False,
                    help="also run isolated dense pack/GEMM/Hadamard cross-check")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "prefill_stage_profile.json"))
    ap.add_argument("--csv", type=str, default=os.path.join(KERN, "prefill_stage_profile.csv"))
    args = ap.parse_args()

    Ms = [int(x) for x in args.ms.split(",")]
    which = set(args.models.split(","))
    config = load_config(args.config)
    out_dir = os.path.dirname(args.out)
    print(f"Stage-profile harness | config={args.config} ms={args.ms} iters={args.iters} "
          f"models={which} eager_attn={args.eager_attn} profiler={args.profiler}", flush=True)

    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)

    model = fp16 = None
    if "real" in which:
        print("Building real-kernel model...", flush=True)
        model = prepare_ptq_model(config, DEV, run_gptq=False)
        model.to(DEV)
        model.config.use_cache = False
        packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
        hpk_arg = None
        if args.fused:
            from promix.eval.hadamard_swiglu_fusion import load_hpk, load_qpfm
            hpk_arg = load_hpk()
            fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf,
                                qpfm=load_qpfm(), hpk=hpk_arg)
        model.to(DEV).eval()
        n_moved = 0
        for m in model.modules():
            for attr in ("_column_order", "new_column_order"):
                t = getattr(m, attr, None)
                if torch.is_tensor(t) and t.device.type != "cuda":
                    setattr(m, attr, t.to(DEV))
                    n_moved += 1
        print(f"attn_impl={model.config._attn_implementation} moved {n_moved} order idx", flush=True)
    if "fp16" in which:
        print("Building FP16 baseline...", flush=True)
        fp16 = _build_fp16_model(config['model']['name'])
        print(f"fp16 attn_impl={fp16.config._attn_implementation}", flush=True)

    vocab = (model or fp16).config.vocab_size
    results = {"config": args.config, "ms": args.ms, "iters": args.iters,
               "runs": [], "profiler": {}}

    for M in Ms:
        if model is not None:
            results["runs"].append(
                profile_model(model, "real", M, vocab, args.iters, args.warmup, fused=args.fused))
            if args.eager_attn:
                results["runs"].append(
                    profile_model(model, "real_eagerattn", M, vocab,
                                  args.iters, args.warmup, eager_attn=True))
        if fp16 is not None:
            results["runs"].append(
                profile_model(fp16, "fp16", M, vocab, args.iters, args.warmup))
        cleanup_memory()

    if args.profiler:
        for M in Ms:
            if model is not None:
                results["profiler"][f"real_M{M}"] = run_torch_profiler(
                    model, "real", M, vocab, out_dir)
            if fp16 is not None:
                results["profiler"][f"fp16_M{M}"] = run_torch_profiler(
                    fp16, "fp16", M, vocab, out_dir)
            cleanup_memory()

    if args.cross_check and model is not None:
        n_layers = model.config.num_hidden_layers
        results["cross_check"] = run_cross_check(nv, mx, cd, qpf, model, Ms, n_layers)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # flat CSV: one row per (label, M, stage)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "M", "eager_attn", "stage", "ms_per_forward",
                    "in_partition", "clean_min_ms", "partition_sum_ms", "residual_ms"])
        for r in results["runs"]:
            for stage, ms in sorted(r["stages"].items(), key=lambda kv: -kv[1]):
                w.writerow([r["label"], r["M"], r["eager_attn"], stage,
                            f"{ms:.5f}", stage in PARTITION, f"{r['t_clean_min_ms']:.5f}",
                            f"{r['partition_sum_ms']:.5f}", f"{r['residual_ms']:.5f}"])
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print(f"CSV_WRITTEN {args.csv}", flush=True)
    print("PREFILL_STAGE_PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
