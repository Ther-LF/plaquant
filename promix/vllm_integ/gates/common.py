"""Shared helpers for the vLLM-side gate harnesses (run in the vLLM venv).

All gates run the engine IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0) with
enforce_eager so the model is hookable and the plugin loads via plain import.
"""

import os

# Must be set before vllm import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402

import promix.vllm_integ.plugin  # noqa: F401,E402  (registers "promix")

MERGE_MEMBERS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "o_proj": ("o_proj",),
    "down_proj": ("down_proj",),
}


def build_llm(ckpt, max_model_len=2176, gpu_mem=0.6, enforce_eager=True):
    from vllm import LLM
    return LLM(model=ckpt, dtype="float16", enforce_eager=enforce_eager,
               max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
               disable_log_stats=True)


def promix_layers(model):
    """Yield (prefix, module) for every promix-quantized LinearBase."""
    for name, mod in model.named_modules():
        if hasattr(mod, "promix_mql"):
            yield name, mod


def hf_member_names(prefix):
    """vLLM merged prefix -> list of HF per-projection module names (merge order).

    'model.layers.3.self_attn.qkv_proj' ->
        ['model.layers.3.self_attn.q_proj', ...k_proj, ...v_proj]
    """
    kind = prefix.split(".")[-1]
    base = prefix.rsplit(".", 1)[0]
    return [f"{base}.{m}" for m in MERGE_MEMBERS[kind]], kind


def expected_merged_buffers(state, prefix, cd, device="cuda"):
    """Compute the EXPECTED post-load packed buffers for one merged layer from
    the exported safetensors tensors (cat along N + the kernel prepack)."""
    members, kind = hf_member_names(prefix)

    def cat(suffix):
        ts = [state[m + "." + suffix].to(device) for m in members]
        return torch.cat(ts, dim=0).contiguous()

    W_lo = cat("weight_lo_packed")
    SF_lo = cat("weight_lo_scale")
    N = W_lo.shape[0]
    K_low = W_lo.shape[1] * 2
    has_hi = (members[0] + ".weight_hi_bytes") in state
    if has_hi:
        W_hi = cat("weight_hi_bytes")
        SF_hi = cat("weight_hi_scale")
        K_high = W_hi.shape[1]
    else:
        W_hi = torch.empty((N, 0), dtype=torch.uint8, device=device)
        SF_hi = None
        K_high = 0
    gs = torch.cat([state[m + ".weight_global_scale"].to(device) for m in members])
    assert torch.unique(gs).numel() == 1, (prefix, gs)
    phys_lo = cd.prepack_sf_fp4_b(SF_lo, N, K_low)
    phys_hi = (cd.prepack_sf_fp8_b(SF_hi, N, K_high) if K_high > 0
               else torch.empty((0,), dtype=torch.uint8, device=device))
    out = dict(W_lo_packed=W_lo, physSFB_lo=phys_lo, W_hi_bytes=W_hi,
               physSFB_hi=phys_hi, g=float(gs[0]), N=N, K_low=K_low, K_high=K_high)
    if kind == "o_proj":
        key = members[0] + ".column_order"
        if key in state:
            out["column_order"] = state[key].to(device, torch.long)
    return out
