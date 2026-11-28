#!/usr/bin/env python3
"""Recreate the llm-compressor FP8_DYNAMIC checkpoint on GB10 (data-free).

Same recipe as Pro6000 /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-fp8:
  QuantizationModifier(targets=Linear, scheme=FP8_DYNAMIC, ignore=[lm_head])
Weights: per-channel symmetric fp8 e4m3 (scale = amax/448, bf16, shape [out,1]).
Activations: dynamic per-token (runtime, nothing stored).
Validated by sha256 against ground-truth tensors from the Pro6000 checkpoint.
"""
import argparse, hashlib, json, os, shutil, glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="BF16 snapshot dir")
ap.add_argument("--out", required=True)
args = ap.parse_args()

LINEAR_SUFFIX = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)

FP8_MAX = 448.0
out_tensors = {}
shards = sorted(glob.glob(os.path.join(args.base, "model*.safetensors")))
assert shards, "no safetensors shards found"
for shard in shards:
    with safe_open(shard, "pt") as f:
        for name in f.keys():
            w = f.get_tensor(name)
            if name.endswith(LINEAR_SUFFIX) and name != "lm_head.weight":
                assert w.dtype == torch.bfloat16, (name, w.dtype)
                wc = w.cuda()
                # memoryless minmax observer, symmetric channel strategy
                min_vals = torch.min(wc.min(dim=1, keepdim=True).values,
                                     torch.zeros(1, device=wc.device, dtype=wc.dtype))
                max_vals = torch.max(wc.max(dim=1, keepdim=True).values,
                                     torch.zeros(1, device=wc.device, dtype=wc.dtype))
                max_val_pos = torch.max(min_vals.abs(), max_vals.abs())
                scale = max_val_pos / FP8_MAX          # bf16 math, [out,1]
                eps = torch.finfo(torch.bfloat16).eps
                scale = torch.where(scale == 0,
                                    torch.tensor(eps, dtype=scale.dtype, device=scale.device),
                                    scale)
                # ct-0.19 compress path: fp32 divide, clamp, direct RNE cast to fp8
                q = torch.clamp(wc.float() / scale.float(), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                out_tensors[name] = q.cpu()
                out_tensors[name.replace(".weight", ".weight_scale")] = scale.cpu()
            else:
                out_tensors[name] = w

os.makedirs(args.out, exist_ok=True)

# ground truth from Pro6000 checkpoint (sha256[:16] of raw bytes)
GT = {
    "model.layers.0.self_attn.q_proj.weight": "644cbe666d8b7767",
    "model.layers.0.self_attn.q_proj.weight_scale": "78f6134c6b39d358",
    "model.layers.5.mlp.down_proj.weight": "4442eb0c54c9d080",
    "model.layers.5.mlp.down_proj.weight_scale": "4c6185e0931f1639",
}
ok = True
for n, ref in GT.items():
    h = hashlib.sha256(out_tensors[n].view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
    print(("MATCH " if h == ref else "MISMATCH ") + n, h, "ref", ref)
    ok &= (h == ref)
print("HASH_VALIDATION", "PASS" if ok else "FAIL")

save_file(out_tensors, os.path.join(args.out, "model.safetensors"),
          metadata={"format": "pt"})
for fn in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
           "special_tokens_map.json"):
    src = os.path.join(args.base, fn)
    if os.path.exists(src):
        shutil.copy(src, args.out)
print("WROTE", args.out, "tensors", len(out_tensors))
