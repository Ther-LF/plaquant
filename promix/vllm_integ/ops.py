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

# handle -> {"run": callable(x)->Tensor(fp16), "N": int}
_REGISTRY: list = []


def register_layer(run, N: int) -> int:
    _REGISTRY.append({"run": run, "N": int(N)})
    return len(_REGISTRY) - 1


def _promix_linear(x: torch.Tensor, handle: int) -> torch.Tensor:
    return _REGISTRY[handle]["run"](x)


def _promix_linear_fake(x: torch.Tensor, handle: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _REGISTRY[handle]["N"]),
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
    _registered = True
