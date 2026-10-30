"""Graph-safe custom op for the promix linear path (Phase 2).

vLLM 0.32 compiles the model forward with Dynamo (fullgraph) and captures
piecewise/full CUDA graphs. The promix apply path calls raw pybind extensions
(quant_pack_float, fused_gemm_prepacked, fast_hadamard_transform) that Dynamo
cannot trace. We wrap the WHOLE per-layer apply (optional Hadamard / column
reorder -> activation quant+pack -> fused dual-format GEMM) in ONE opaque
custom op, `torch.ops.vllm.promix_linear(x, handle)`:

  * impl       runs the exact Phase-1 eager body (bit-identical kernels);
               branch-free w.r.t. shape at trace time (all branching happens
               inside the op at run time, legal per captured graph size).
  * fake_impl  shape propagation for Dynamo/inductor: (*x.shape[:-1], N) fp16.

The op is registered at plugin-import time, so any process that resolves the
"promix" quant method (entry point or plain import) also has the op. Layers
are addressed by an integer handle into a process-local registry, assigned in
process_weights_after_loading (which runs before vLLM compiles the model).

Stream-capture safety of the underlying exts was proven by the promix
prefill-graph work (run_with_cached_ws guards its sync with
cudaStreamIsCapturing; workspaces/ext caches are allocated during the warmup
passes vLLM runs before capture).
"""

import torch

# handle -> {"run": callable(x)->Tensor(fp16), "N": int,
#            "mql": MixedQuantLinear|None, "vrq": ext|None}
# mql/vrq are populated for norm-fusable layers (qkv/gate_up) so the r3
# producer-fusion ops (promix_rms_quant_pack / promix_gemm_packed) can run.
_REGISTRY: list = []


def register_layer(run, N: int, mql=None, vrq=None) -> int:
    _REGISTRY.append({"run": run, "N": int(N), "mql": mql, "vrq": vrq})
    return len(_REGISTRY) - 1


def handle_can_norm_fuse(handle: int) -> bool:
    """True iff the r3 norm->quant_pack fusion pass may rewrite this layer:
    the fused kernel needs a dual-format split (K_high>0) and the vrq ext."""
    e = _REGISTRY[handle]
    return (e.get("mql") is not None and e.get("vrq") is not None
            and e["mql"].K_high > 0)


def _promix_linear(x: torch.Tensor, handle: int) -> torch.Tensor:
    return _REGISTRY[handle]["run"](x)


def _promix_linear_fake(x: torch.Tensor, handle: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _REGISTRY[handle]["N"]),
                       dtype=torch.float16)


# ---------------------------------------------------------------------------
# vllm-perf r3 lever 2: producer-side quant fusion ops.
#
# promix_rms_quant_pack = vLLM's fused_add_rms_norm (bit-exact replica) + the
# qpf NVFP4/MXFP8 quant+pack in ONE kernel (vllm_rms_quant_pack_ext). It also
# emits the fp16 norm output xn so the decode GEMV routes (which consume fp16)
# keep working, and the functional residual_out.
#
# promix_gemm_packed consumes the packed activation (GEMM route) or xn (GEMV
# route, decided at run time per captured M — the r1 capture-safe pattern).
#
# The fusion pass (fusion_pass.py) rewrites
#   vllm_ir.fused_add_rms_norm -> promix_linear
# into
#   promix_rms_quant_pack -> promix_gemm_packed
# removing one kernel launch and one fp16 [M,K] HBM read per norm site.
# ---------------------------------------------------------------------------

def _sfa_numel(M: int, K: int, vec: int) -> int:
    """numel of size(filter_zeros(Sm1xxBlockScaledConfig<vec>::tile_atom_to_
    shape_SFA(M, N, K, 1))): 128-row atoms x 4-SF-column atoms. Verified
    against the ext by norm_pack_probe."""
    nblk = K // vec
    return ((M + 127) // 128) * 128 * ((nblk + 3) // 4) * 4


def _promix_rms_quant_pack(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    e = _REGISTRY[handle]
    mql, vrq = e["mql"], e["vrq"]
    out = vrq.vllm_add_rms_norm_quant_pack(
        x, residual, weight, e["N"], mql.K_low, mql.K_high, eps)
    return tuple(out)


def _promix_rms_quant_pack_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, handle: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    e = _REGISTRY[handle]
    mql = e["mql"]
    M = x.shape[0]
    u8 = dict(dtype=torch.uint8)
    return (
        x.new_empty((M, mql.K_low // 2), **u8),
        x.new_empty((_sfa_numel(M, mql.K_low, 16),), **u8),
        x.new_empty((M, mql.K_high), **u8),
        x.new_empty((_sfa_numel(M, mql.K_high, 32),), **u8),
        x.new_empty((M, x.shape[1])),
        x.new_empty((M, x.shape[1])),
    )


def _promix_gemm_packed(
    code_lo: torch.Tensor, sfa_lo: torch.Tensor,
    code_hi: torch.Tensor, sfa_hi: torch.Tensor,
    xn: torch.Tensor, handle: int,
) -> torch.Tensor:
    e = _REGISTRY[handle]
    mql = e["mql"]
    M = xn.shape[0]
    # M is static per captured vLLM CUDA graph -> capture-safe branch (r1).
    if mql.gemv_route(M):
        return mql(xn)                         # decode GEMV consumes fp16
    return mql.forward_norm_pack((code_lo, sfa_lo, code_hi, sfa_hi))


def _promix_gemm_packed_fake(
    code_lo: torch.Tensor, sfa_lo: torch.Tensor,
    code_hi: torch.Tensor, sfa_hi: torch.Tensor,
    xn: torch.Tensor, handle: int,
) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], _REGISTRY[handle]["N"]),
                        dtype=torch.float16)


_registered = False


def ensure_registered():
    global _registered
    if _registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="promix_linear",
        op_func=_promix_linear,
        mutates_args=[],
        fake_impl=_promix_linear_fake,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        op_name="promix_rms_quant_pack",
        op_func=_promix_rms_quant_pack,
        mutates_args=[],
        fake_impl=_promix_rms_quant_pack_fake,
        dispatch_key="CUDA",
    )
    direct_register_custom_op(
        op_name="promix_gemm_packed",
        op_func=_promix_gemm_packed,
        mutates_args=[],
        fake_impl=_promix_gemm_packed_fake,
        dispatch_key="CUDA",
    )
    _registered = True
