"""
PLAQuant WikiText-2 PPL evaluation.

Usage:
    cd /path/to/plaquant
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
    python -m inference.eval_ppl \
        --model unsloth/Llama-3.2-1B-Instruct \
        --checkpoint project-resq/fake_quant/qmodels/W4A4KV4-Llama-3.2-1B-v2.pt \
        --rotation project-resq/fake_quant/rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B-Instruct.bin \
        --basis project-resq/fake_quant/rotation/U-wikitext-512-Llama-3.2-1B-Instruct.bin
"""

import argparse
import os
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResQ model PPL")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--checkpoint", required=True, help="ResQ quantized checkpoint (.pt)")
    parser.add_argument("--rotation", required=True, help="R rotation file path")
    parser.add_argument("--basis", required=True, help="U basis file path")
    parser.add_argument("--resq-path", default="project-resq/fake_quant", help="Path to ResQ fake_quant dir")
    parser.add_argument("--high-fraction", type=float, default=0.125)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--k-groupsize", type=int, default=64)
    parser.add_argument("--v-groupsize", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-real-quant", action="store_true", help="Use fake quant instead of real quant")
    args = parser.parse_args()

    # Initialize distributed (required by ResQ's fuse_basis_to_model)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")

    from inference.model import ResQModel

    model = ResQModel.from_checkpoint(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        rotation_path=args.rotation,
        basis_path=args.basis,
        resq_path=args.resq_path,
        high_fraction=args.high_fraction,
        a_bits=args.a_bits,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        high_bits=args.high_bits,
        k_groupsize=args.k_groupsize,
        v_groupsize=args.v_groupsize,
        real_quant=not args.no_real_quant,
        seqlen=args.seqlen,
    )

    ppl = model.evaluate_ppl(device=args.device)
    print(f"\n*** WikiText-2 PPL: {ppl:.4f} ***")

    torch.distributed.destroy_process_group()
    return ppl


if __name__ == "__main__":
    main()
