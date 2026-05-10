"""
Convert ResQ checkpoint to SGLang-compatible format.

Input:  ResQ ptq.py checkpoint (qmodels/W4A4KV4-*.pt)
Output: Directory with safetensors + quantize_config.json

Usage:
    python scripts/convert_resq_to_sglang.py \
        --input qmodels/W4A4KV4-Llama-3.2-1B-v2.pt \
        --model-path unsloth/Llama-3.2-1B-Instruct \
        --output /path/to/output_dir \
        --high-fraction 0.125

The output directory can be loaded by SGLang with --quantization resq.
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def convert_resq_checkpoint(
    input_path: str,
    model_path: str,
    output_dir: str,
    high_fraction: float = 0.125,
    a_bits: int = 4,
    high_bits: int = 8,
    a_sym: bool = False,
    w_bits: int = 4,
    w_sym: bool = True,
):
    """Convert ResQ checkpoint to SGLang format."""
    print(f"Loading checkpoint: {input_path}")
    sys.path.insert(0, os.path.dirname(input_path))
    # Need ResQ's utils on path for unpickling
    resq_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), "..")
    if os.path.exists(os.path.join(resq_dir, "utils")):
        sys.path.insert(0, resq_dir)

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    w_int_weights = ckpt["w_int_weights"]
    model_state = ckpt["model"]
    w_quantizers = ckpt["w_quantizers"]

    # Collect all linear layer names
    layer_keys = list(w_int_weights.keys())
    print(f"Found {len(layer_keys)} quantized layers")

    # Build output state dict
    output_tensors = {}

    # Copy non-quantized weights (embeddings, norms, lm_head)
    for key, val in model_state.items():
        # Skip quantizer state and .module.weight (we use w_int_weights instead)
        if ".quantizer." in key or ".out_quantizer." in key:
            continue
        if ".module.weight" in key:
            continue
        # Keep: embed_tokens, norm, lm_head, layernorm weights
        if any(x in key for x in [
            "embed_tokens", "norm.weight", "lm_head",
            "input_layernorm", "post_attention_layernorm",
        ]):
            output_tensors[key] = val.half()
            continue
        # Also keep the original .weight for reference (FP16 weight before rotation)
        # Actually SGLang expects weight keys directly — we'll rename
        if key.endswith(".weight") and ".module" not in key:
            # These are the un-rotated original weights — skip them
            # (quantized layers will use w_main_int/w_high_int instead)
            parts = key.rsplit(".weight", 1)[0]
            # Check if this layer has a quantized version
            module_key = parts + ".module"
            if module_key in w_int_weights:
                continue
            # Non-quantized layer weight
            output_tensors[key] = val.half()

    # Process each quantized linear layer
    for module_key in layer_keys:
        w_int = w_int_weights[module_key]  # (N, K) int16
        N, K = w_int.shape

        # Compute split dimensions
        high_len = int(K * high_fraction)
        main_len = K - high_len

        # Split weights
        w_main = w_int[:, :main_len].to(torch.int8)  # (N, K_main) [-8, 7]
        w_high = w_int[:, main_len:].to(torch.int8) if high_len > 0 else None  # (N, K_high) [-128, 127]

        # Get scales from quantizer
        q_main = w_quantizers[module_key]
        q_high_key = module_key + ",high_quantizer"
        q_high = w_quantizers.get(q_high_key)

        # Extract per-channel scale
        # The quantizer stores scale as computed during GPTQ
        # We need to recompute from the FP16 weight and int weight
        # scale = max(|W_channel|) / maxq for symmetric
        fp16_weight_key = module_key + ".weight"
        if fp16_weight_key in model_state:
            w_fp16 = model_state[fp16_weight_key]
        else:
            # Fall back: compute scale from int weight range
            w_fp16 = None

        # For symmetric quantization: scale = max(|W|, dim=1) / maxq
        maxq_main = 2 ** (w_bits - 1) - 1  # 7 for 4-bit
        maxq_high = 2 ** (high_bits - 1) - 1  # 127 for 8-bit

        if w_fp16 is not None:
            # Compute scale from original FP16 weight
            w_fp16_main = w_fp16[:, :main_len]
            w_fp16_high = w_fp16[:, main_len:] if high_len > 0 else None

            s_w_main = w_fp16_main.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq_main
            s_w_high = w_fp16_high.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq_high if w_fp16_high is not None else None
        else:
            # Fallback: derive from int range (less accurate)
            s_w_main = w_main.float().abs().amax(dim=1, keepdim=True) / maxq_main
            s_w_high = w_high.float().abs().amax(dim=1, keepdim=True) / maxq_high if w_high is not None else None

        # Compute column sums for bias correction (asymmetric activation quant)
        colsum_main = w_main.float().sum(dim=1)  # (N,)
        colsum_high = w_high.float().sum(dim=1) if w_high is not None else None  # (N,)

        # Convert module_key to SGLang parameter naming
        # module_key: "model.layers.0.self_attn.q_proj.module"
        # SGLang key: "model.layers.0.self_attn.q_proj.w_main_int"
        layer_prefix = module_key.replace(".module", "")

        output_tensors[f"{layer_prefix}.w_main_int"] = w_main
        output_tensors[f"{layer_prefix}.w_main_scale"] = s_w_main.half()
        output_tensors[f"{layer_prefix}.w_main_colsum"] = colsum_main

        if w_high is not None:
            output_tensors[f"{layer_prefix}.w_high_int"] = w_high
            output_tensors[f"{layer_prefix}.w_high_scale"] = s_w_high.half()
            output_tensors[f"{layer_prefix}.w_high_colsum"] = colsum_high

    # Load rotation matrices
    rotation_dir = os.path.join(os.path.dirname(input_path), "rotation")
    if os.path.exists(rotation_dir):
        # Find U and R matrices
        u_files = [f for f in os.listdir(rotation_dir) if f.startswith("U-")]
        r_files = [f for f in os.listdir(rotation_dir) if f.startswith("R-")]
        if u_files:
            U = torch.load(
                os.path.join(rotation_dir, u_files[0]),
                map_location="cpu", weights_only=False
            )
            if isinstance(U, torch.Tensor):
                output_tensors["resq_U_basis"] = U.half()
                print(f"  Loaded U basis: {U.shape}")
        if r_files:
            R = torch.load(
                os.path.join(rotation_dir, r_files[0]),
                map_location="cpu", weights_only=False
            )
            if isinstance(R, torch.Tensor):
                output_tensors["resq_R_rotation"] = R.half()
                print(f"  Loaded R rotation: {R.shape}")

    # Save output
    os.makedirs(output_dir, exist_ok=True)

    # Save as safetensors (split if too large)
    print(f"Saving {len(output_tensors)} tensors to {output_dir}")
    save_file(output_tensors, os.path.join(output_dir, "model.safetensors"))

    # Save quantization config
    quant_config = {
        "quant_method": "resq",
        "a_bits": a_bits,
        "high_bits": high_bits,
        "high_fraction": high_fraction,
        "a_sym": a_sym,
        "w_bits": w_bits,
        "w_sym": w_sym,
        "clip_ratio": 1.0,
    }
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy tokenizer config from original model
    from transformers import AutoTokenizer
    print(f"Copying tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)

    # Save model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config.quantization_config = quant_config
    config.architectures = ["ResQLlamaForCausalLM"]
    config.save_pretrained(output_dir)

    print(f"Done! Output saved to {output_dir}")
    print(f"  Load with: --model-path {output_dir} --quantization resq")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ResQ checkpoint to SGLang format")
    parser.add_argument("--input", required=True, help="Path to ResQ qmodel .pt file")
    parser.add_argument("--model-path", required=True, help="HuggingFace model path for tokenizer/config")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--high-fraction", type=float, default=0.125)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--a-sym", action="store_true", default=False)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--w-sym", action="store_true", default=True)
    args = parser.parse_args()

    convert_resq_checkpoint(
        input_path=args.input,
        model_path=args.model_path,
        output_dir=args.output,
        high_fraction=args.high_fraction,
        a_bits=args.a_bits,
        high_bits=args.high_bits,
        a_sym=args.a_sym,
        w_bits=args.w_bits,
        w_sym=args.w_sym,
    )
