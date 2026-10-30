"""ProMix -> vLLM exporter (Phase 1).

Builds the prepared (rotated + reordered + GPTQ) ProMix model, re-quantizes the
vLLM-merged groups (qkv, gate_up) with a SHARED per-tensor global g (see
shared_g.py), re-references the real-kernel wikitext PPL for the shared-g model
(the T2 gate target), and exports a vLLM-loadable checkpoint:

  out_dir/
    model.safetensors       packed NVFP4/MXFP8 weights (linear-layout SF) +
                            fp16 non-quantized weights, HF Llama names
    config.json             llama config + quantization_config{quant_method:
                            "promix", k_high map, hadamard/o_proj flags, ...}
    tokenizer*              tokenizer files
    promix_export_meta.json shared-g re-reference PPL + per-projection-g PPL +
                            per-linear K split / g table
    t2_reference.pt         wikitext eval chunks + the shared-g real-kernel
                            model's per-position target logprob & top-1 ids
                            (fp32 log_softmax of fp16 logits) for gate T2

Run inside the container base env (torch 2.11 + exts) on a free GPU:

  docker exec -d -w /home/luofan/plaquant-vllm-integ -e CUDA_VISIBLE_DEVICES=6 \
    -e HF_HOME=/home/luofan/hf_cache -e PYTHONPATH=/home/luofan/plaquant-vllm-integ \
    -e PYTHONUNBUFFERED=1 luofan-plaquant bash -c \
    'python -m promix.vllm_integ.export --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
       --out /home/luofan/vllm_integ/promix-llama-3.2-3b > /home/luofan/EXPORT_3B.log 2>&1'
"""

import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers  # noqa: E402

from promix.utils import DEV, cleanup_memory  # noqa: E402
from promix.eval.ptq import (  # noqa: E402
    load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed,
)
from promix.eval.evaluator import evaluate_ppl  # noqa: E402
from promix.eval.data import get_wikitext2  # noqa: E402
from promix.quantize import full_network as fn  # noqa: E402
from promix.quantize import quant_utils as qu  # noqa: E402
from promix.quantize.weight_pack import (  # noqa: E402
    pack_weight_mixed, quantize_nvfp4_spec, quantize_mxfp8_spec,
)
from promix.eval.full_network_realkernel import load_kernels  # noqa: E402
from promix.vllm_integ.shared_g import (  # noqa: E402
    MERGED_GROUPS, rederive_frozen_scales_shared_g, shared_g_for_group,
)

HIGH_FRACTION = 0.125


# ---------------------------------------------------------------------------
# member scale collection (GPTQ frozen or two-level auto — ONE code path after)
# ---------------------------------------------------------------------------
def member_scales(name, wrapper, K_low, K_high, quantizers):
    """(eff_nv (N,K_low/16) f32, eff_mx (N,K_high/32) f32 or None, g float or None)
    for the Linear at `name`, matching exactly what pack_all_targets would use."""
    W = wrapper.module.weight.data
    nv_s, mx_s, g = fn._frozen_scales_for(name, quantizers)
    if nv_s is not None:
        g = None if g is None else float(g)
        return (nv_s.to(torch.float32), None if mx_s is None else mx_s.to(torch.float32), g)
    # RTN path (no GPTQ): replicate pack_weight_mixed's two-level auto branch.
    if not getattr(qu, "WEIGHT_TWO_LEVEL", False):
        return (None, None, None)
    Wf = W.to(torch.float32)
    N = Wf.shape[0]
    xb_lo = Wf[:, :K_low].reshape(N, K_low // 16, 16)
    eff_nv, g_auto = qu._weight_two_level_nvfp4_eff(xb_lo, xb_lo.abs().amax(-1))
    eff_mx = None
    if K_high > 0:
        xb_hi = Wf[:, K_low:].reshape(N, K_high // 32, 32)
        eff_mx = qu._weight_two_level_mxfp8_eff(xb_hi, xb_hi.abs().amax(-1), g_auto)
    return (eff_nv, eff_mx, float(g_auto))


def pack_all_targets_shared_g(model, nv, mx, cd, verbose=True):
    """Like full_network.pack_all_targets, but members of each vLLM merged group
    (q/k/v, gate/up) are re-quantized against a SHARED per-tensor global g.
    Returns (packed {name: PackedWeight}, meta {name: dict}, linear_sf
    {name: (SF_lo (N,K_low/16) u8, SF_hi (N,K_high/32) u8 or None)})."""
    quantizers = getattr(model, "_gptq_quantizers", None)
    wrappers = {}   # name -> (short, wrapper)
    for name, short, wrapper in fn.iter_target_wrappers(model):
        wrappers[name] = (short, wrapper)

    # group names per decoder layer
    by_layer = {}
    for name, (short, _w) in wrappers.items():
        li = int(re.search(r"layers\.(\d+)\.", name).group(1))
        by_layer.setdefault(li, {})[short] = name

    plan = {}  # name -> (eff_nv, eff_mx, g) AFTER shared-g rederivation
    for li, shorts in sorted(by_layer.items()):
        grouped = set()
        for gshort, members in MERGED_GROUPS.items():
            names = [shorts[m] for m in members if m in shorts]
            assert len(names) == len(members), f"layer {li}: missing members for {gshort}"
            scales = []
            for nm in names:
                short, w = wrappers[nm]
                K = w.module.weight.shape[1]
                K_low, K_high = fn._khigh_for(w, K, HIGH_FRACTION)
                scales.append(member_scales(nm, w, K_low, K_high, quantizers) + (K_low, K_high))
            # merged GEMM: all members must share the SAME K split
            ksplits = {(s[3], s[4]) for s in scales}
            assert len(ksplits) == 1, f"layer {li} {gshort}: mixed K splits {ksplits}"
            g_shared = shared_g_for_group([s[:3] for s in scales])
            for nm, (eff_nv, eff_mx, g, K_low, K_high) in zip(names, scales):
                if g_shared is None:
                    plan[nm] = (eff_nv, eff_mx, None, K_low, K_high, None)
                else:
                    nv2, mx2 = rederive_frozen_scales_shared_g(eff_nv, eff_mx, g_shared)
                    plan[nm] = (nv2, mx2, g_shared, K_low, K_high, g)
                grouped.add(nm)
        for short, nm in shorts.items():
            if nm in grouped:
                continue
            w = wrappers[nm][1]
            K = w.module.weight.shape[1]
            K_low, K_high = fn._khigh_for(w, K, HIGH_FRACTION)
            eff_nv, eff_mx, g = member_scales(nm, w, K_low, K_high, quantizers)
            plan[nm] = (eff_nv, eff_mx, g, K_low, K_high, g)

    packed, meta, linear_sf = {}, {}, {}
    for nm, (eff_nv, eff_mx, g, K_low, K_high, g_orig) in plan.items():
        short, w = wrappers[nm]
        W = w.module.weight.data
        bias = getattr(w.module, "bias", None)
        bias = None if bias is None else bias.data
        pk = pack_weight_mixed(W, K_low, K_high, nv, mx, cd, bias=bias,
                               frozen_nvfp4_scale=eff_nv,
                               frozen_mxfp8_scale=eff_mx,
                               global_scale=g)
        # linear-layout SFs for export + self-check vs the pack (guards drift)
        Wf = W.to(torch.float32)
        codes_lo, sf_lo, _ = quantize_nvfp4_spec(
            Wf[:, :K_low].contiguous(), frozen_scale=eff_nv, global_scale=g)
        assert torch.equal(nv.pack_fp4(codes_lo), pk.W_lo_packed), nm
        assert torch.equal(cd.prepack_sf_fp4_b(sf_lo.contiguous(), pk.N, K_low),
                           pk.physSFB_lo), nm
        sf_hi = None
        if K_high > 0:
            bytes_hi, sf_hi, _ = quantize_mxfp8_spec(
                Wf[:, K_low:].contiguous(), frozen_scale=eff_mx, global_scale=g)
            assert torch.equal(bytes_hi, pk.W_hi_bytes), nm
            assert torch.equal(cd.prepack_sf_fp8_b(sf_hi.contiguous(), pk.N, K_high),
                               pk.physSFB_hi), nm
        packed[nm] = pk
        linear_sf[nm] = (sf_lo, sf_hi)
        meta[nm] = dict(N=pk.N, K=pk.K, K_low=K_low, K_high=K_high,
                        g=pk.g_w, g_orig=g_orig)
        if verbose:
            go = "None" if g_orig is None else f"{g_orig:.3e}"
            print(f"  packed {nm:48s} N={pk.N:5d} K={pk.K:5d} Kh={K_high:4d} "
                  f"g={pk.g_w:.3e} (orig {go})", flush=True)
    print(f"pack_all_targets_shared_g: packed {len(packed)} Linears", flush=True)
    return packed, meta, linear_sf


# ---------------------------------------------------------------------------
# checkpoint writer
# ---------------------------------------------------------------------------
def vllm_prefix_key(name, short):
    """promix module name -> the k_high-map key (vLLM merged-prefix suffix)."""
    li = re.search(r"layers\.(\d+)\.", name).group(1)
    if short in ("q_proj", "k_proj", "v_proj"):
        return f"layers.{li}.self_attn.qkv_proj"
    if short in ("gate_proj", "up_proj"):
        return f"layers.{li}.mlp.gate_up_proj"
    if short == "o_proj":
        return f"layers.{li}.self_attn.o_proj"
    return f"layers.{li}.mlp.down_proj"


def export_checkpoint(model, packed, meta, linear_sf, out_dir, config,
                      ref_ppl_sharedg, ref_ppl_perproj):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    k_high_map = {}
    for name, short, wrapper in fn.iter_target_wrappers(model):
        pk = packed[name]
        sf_lo, sf_hi = linear_sf[name]
        state[name + ".weight_lo_packed"] = pk.W_lo_packed.cpu()
        state[name + ".weight_lo_scale"] = sf_lo.cpu()
        if pk.K_high > 0:
            state[name + ".weight_hi_bytes"] = pk.W_hi_bytes.cpu()
            state[name + ".weight_hi_scale"] = sf_hi.cpu()
        state[name + ".weight_global_scale"] = torch.tensor([pk.g_w], dtype=torch.float32)
        assert pk.bias is None, f"{name}: llama linears are bias-free"
        if short == "o_proj":
            order = getattr(wrapper, "_column_order", None)
            if order is not None:
                state[name + ".column_order"] = order.to(torch.int32).cpu()
        key = vllm_prefix_key(name, short)
        prev = k_high_map.setdefault(key, pk.K_high)
        assert prev == pk.K_high, f"{key}: inconsistent K_high within merged group"

    # non-quantized weights (fp16, HF Llama names)
    state["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.to(torch.float16).cpu()
    state["lm_head.weight"] = _lm_head_weight(model).to(torch.float16).cpu()
    state["model.norm.weight"] = model.model.norm.weight.data.to(torch.float16).cpu()
    for i, layer in enumerate(model.model.layers):
        state[f"model.layers.{i}.input_layernorm.weight"] = \
            layer.input_layernorm.weight.data.to(torch.float16).cpu()
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = \
            layer.post_attention_layernorm.weight.data.to(torch.float16).cpu()
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    print(f"wrote model.safetensors ({len(state)} tensors)", flush=True)

    o_proj_reorder = any(k.endswith(".column_order") for k in state)
    d = model.config.to_dict()
    d["architectures"] = ["LlamaForCausalLM"]
    d["tie_word_embeddings"] = False
    d["torch_dtype"] = "float16"
    d["quantization_config"] = {
        "quant_method": "promix",
        "fmt": "nvfp4(low)+mxfp8(high)",
        "high_fraction": config["quantize"]["high_fraction"],
        "k_high": k_high_map,
        "hadamard_down_proj": True,
        "had_mode": os.environ.get("PROMIX_HAD_MODE", "fp16"),
        "fold_had_scale": True,
        "o_proj_reorder": o_proj_reorder,
        "act_mse": {
            "enabled": os.environ.get("PROMIX_ACT_MSE", "1") != "0",
            "fast": os.environ.get("PROMIX_ACT_MSE_FAST", "1") != "0",
            "steps": int(os.environ.get("PROMIX_ACT_MSE_STEPS", "3")),
        },
        "weight_two_level": bool(qu.WEIGHT_TWO_LEVEL),
        "out_dtype": "float16",
        "reference_ppl_sharedg": ref_ppl_sharedg,
        "reference_ppl_perproj_g": ref_ppl_perproj,
        "source_config": os.path.basename(config.get("_path", "unknown")),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "promix_export_meta.json"), "w") as f:
        json.dump({"reference_ppl_sharedg": ref_ppl_sharedg,
                   "reference_ppl_perproj_g": ref_ppl_perproj,
                   "per_linear": {k: v for k, v in meta.items()}}, f, indent=2)
    print(f"wrote config.json + tokenizer + meta -> {out_dir}", flush=True)


def _lm_head_weight(model):
    lm = model.lm_head
    mod = getattr(lm, "module", lm)   # ActQuantWrapper(lm_head) or plain Linear
    return mod.weight.data


# ---------------------------------------------------------------------------
# T2 reference snapshot (shared-g real-kernel model)
# ---------------------------------------------------------------------------
@torch.no_grad()
def t2_snapshot(model, testloader, seqlen, out_path, max_chunks=None):
    ids_all = testloader.input_ids
    nchunks = ids_all.numel() // seqlen
    if max_chunks:
        nchunks = min(nchunks, max_chunks)
    ids = ids_all[:, :nchunks * seqlen].view(nchunks, seqlen)
    model.to(DEV).eval()
    model.config.use_cache = False
    tgt_lp, top1 = [], []
    for i in range(nchunks):
        chunk = ids[i:i + 1].to(DEV)
        logits = model(chunk).logits.float()           # (1, S, V)
        lp = torch.log_softmax(logits, dim=-1)
        pred = lp[0, :-1]                              # predicts tokens 1..S-1
        tgt = chunk[0, 1:]
        tgt_lp.append(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu())
        top1.append(pred.argmax(-1).to(torch.int32).cpu())
        if (i + 1) % 20 == 0:
            print(f"  t2 snapshot {i + 1}/{nchunks}", flush=True)
    tgt_lp = torch.stack(tgt_lp)                       # (nchunks, S-1) fp32
    top1 = torch.stack(top1)                           # (nchunks, S-1) int32
    ppl_t2 = torch.exp(-tgt_lp.mean()).item()
    torch.save({"input_ids": ids.to(torch.int32), "target_logprob": tgt_lp,
                "top1": top1, "seqlen": seqlen, "ppl_fp32lsm": ppl_t2}, out_path)
    print(f"T2_REFERENCE ppl(fp32-lsm)={ppl_t2:.4f} chunks={nchunks} -> {out_path}",
          flush=True)
    return ppl_t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "promix/configs/llama-3.2-3b-fp-perhead.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--perproj-ppl", action="store_true", default=True,
                    help="also measure the per-projection-g real-kernel PPL "
                         "(the pre-shared-g reference) for the delta report")
    ap.add_argument("--no-perproj-ppl", dest="perproj_ppl", action="store_false")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="limit eval/snapshot chunks (smoke runs)")
    args = ap.parse_args()

    config = load_config(args.config)
    config["_path"] = args.config
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    print("Loading kernels (JIT may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_kernels()

    model = prepare_ptq_model(config, DEV, run_gptq=args.gptq)
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    testloader = get_wikitext2(seed=config["calibration"].get("seed", 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    if args.max_chunks:
        testloader["input_ids"] = testloader.input_ids[:, :args.max_chunks * model.seqlen]
    bs = config["eval"]["batch_size"]

    # Pack BOTH ways from the intact model (wiring destroys the fp weights).
    ppl_perproj = None
    packed_pp = None
    if args.perproj_ppl:
        print("Packing (per-projection g, the pre-shared-g reference)...", flush=True)
        packed_pp = fn.pack_all_targets(model, nv, mx, cd,
                                        high_fraction=HIGH_FRACTION, verbose=False)
    print("Packing (shared-g per merged group)...", flush=True)
    packed_sg, meta, linear_sf = pack_all_targets_shared_g(model, nv, mx, cd)

    if packed_pp is not None:
        fn.wire_real_kernel(model, packed_pp, nv, mx, cd, qpf=qpf)
        model.to(DEV)
        ppl_perproj = evaluate_ppl(model, testloader, DEV, batch_size=bs)
        print(f"PPL_REAL_PERPROJ_G {ppl_perproj:.4f}", flush=True)
        del packed_pp
        cleanup_memory()

    fn.wire_real_kernel(model, packed_sg, nv, mx, cd, qpf=qpf)
    model.to(DEV)
    ppl_sharedg = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    print(f"PPL_REAL_SHAREDG {ppl_sharedg:.4f}  <- T2 reference (pipeline protocol)",
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    t2_snapshot(model, testloader, model.seqlen,
                os.path.join(args.out, "t2_reference.pt"),
                max_chunks=args.max_chunks)

    export_checkpoint(model, packed_sg, meta, linear_sf, args.out, config,
                      ref_ppl_sharedg=ppl_sharedg, ref_ppl_perproj=ppl_perproj)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
