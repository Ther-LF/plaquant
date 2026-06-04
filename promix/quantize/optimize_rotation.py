"""Rotation optimization for ResQ mixed-precision quantization (Step 1).

Optimizes orthogonal rotation matrices R1 (hidden dim) and R2 (head dim)
on the Stiefel manifold using Cayley-transform SGD. The rotation is trained
to minimize LM loss after basis reordering, making quantization more effective.

This uses project-resq's modified Llama model (with trainable RotateModules)
and SGDG optimizer for Stiefel manifold optimization.

Usage:
    python -m promix.quantize.optimize_rotation --config promix/configs/llama-3.2-1b-resq.yaml
"""

import os
import sys
import datetime
import random

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    default_data_collator,
)

# Add project-resq to path for model/optimizer imports
_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)

from train_utils.modeling_llama_train import LlamaForCausalLM
from train_utils.optimizer import SGDG
from train_utils.main import prepare_model
from train_utils.fsdp_trainer import FSDPTrainer
from utils.data_utils import CustomJsonDataset, get_wikitext2
from utils.hadamard_utils import random_orthogonal_matrix
from utils import utils, data_utils, eval_utils


class RotateModule(torch.nn.Module):
    def __init__(self, R_init):
        super().__init__()
        self.weight = torch.nn.Parameter(R_init.to(torch.float32).cuda())

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def optimize_rotation(
    model_name: str,
    basis_path: str,
    output_dir: str,
    seqlen: int = 2048,
    seed: int = 0,
    high_fraction: float = 0.125,
    low_fraction: float = 0.0,
    sparse_fraction: float = 0.0,
    rotation_granularity: str = "full_shared",
    learning_rate: float = 1.5,
    max_steps: int = 100,
    train_batch_size: int = 1,
    eval_batch_size: int = 8,
    k_bits: int = 16,
    v_bits: int = 16,
    k_groupsize: int = 64,
    v_groupsize: int = 64,
    high_bits: int = 8,
    low_bits: int = 2,
):
    """Optimize rotation matrices on the Stiefel manifold.

    Args:
        model_name: HuggingFace model name
        basis_path: Path to PCA basis file (U file from Step 0)
        output_dir: Directory to save optimized rotation
        seqlen: Sequence length
        seed: Random seed
        high_fraction: Fraction of channels for high-precision
        low_fraction: Fraction for low-precision
        sparse_fraction: Fraction for sparsity
        rotation_granularity: Only "full_shared" is supported
        learning_rate: Learning rate for SGDG
        max_steps: Number of training steps
        train_batch_size: Training batch size
        eval_batch_size: Eval batch size
        k_bits: Key quantization bits (16 = no quant during training)
        v_bits: Value quantization bits
        k_groupsize: Key group size
        v_groupsize: Value group size
        high_bits: High-precision bits
        low_bits: Low-precision bits

    Returns:
        Path to saved optimized rotation file
    """
    assert rotation_granularity == "full_shared", "Only full_shared is supported"

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Optimizing rotation for {model_name}")
    print(f"  basis: {basis_path}")
    print(f"  lr={learning_rate}, max_steps={max_steps}")
    print(f"  high_fraction={high_fraction}, low_fraction={low_fraction}")
    print()

    dtype = torch.bfloat16
    seed_everything(seed)

    # Load model with untied embeddings
    config = AutoConfig.from_pretrained(model_name)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, config=config
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Prepare model (fuse norms, add quant wrappers, fuse basis)
    # Build args namespace for prepare_model
    from argparse import Namespace
    ptq_args = Namespace(
        seed=seed,
        rotate_mode="resq",
        high_fraction=high_fraction,
        low_fraction=low_fraction,
        sparse_fraction=sparse_fraction,
        rotation_granularity=rotation_granularity,
        optimized_basis_path=basis_path,
        optimized_rotation_path=output_dir,
        train_rotations=True,
        w_bits=16,
        a_bits=16,
        k_bits=k_bits,
        v_bits=v_bits,
        a_groupsize=128,
        k_groupsize=k_groupsize,
        v_groupsize=v_groupsize,
        a_asym=True,
        k_asym=True,
        v_asym=True,
        a_clip_ratio=1.0,
        k_clip_ratio=1.0,
        v_clip_ratio=1.0,
        w_clip=True,
        high_bits=high_bits,
        low_bits=low_bits,
        int8_down_proj=False,
        k_pre_rope=False,
        fp32_had=True,
        down_proj_blocksize=256,
        bsz=eval_batch_size,
        capture_layer_io=False,
        layer_idx=0,
    )
    model, R_dict = prepare_model(ptq_args, model)

    # Model dimensions
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model_dim // num_heads
    high_length_hidden = int(high_fraction * model_dim)
    low_length_hidden = int(low_fraction * model_dim)
    high_length_head = int(high_fraction * head_dim)
    low_length_head = int(low_fraction * head_dim)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create trainable rotation modules
    R1_1 = random_orthogonal_matrix(model_dim - high_length_hidden - low_length_hidden, "cuda")
    R1_2 = random_orthogonal_matrix(high_length_hidden, "cuda")
    model.R1_1 = RotateModule(R1_1)
    model.R1_2 = RotateModule(R1_2)

    if low_length_hidden != 0:
        R1_0 = random_orthogonal_matrix(low_length_hidden, "cuda")
        model.R1_0 = RotateModule(R1_0)

    R2_1 = random_orthogonal_matrix(head_dim - high_length_head - low_length_head, "cuda")
    R2_2 = random_orthogonal_matrix(high_length_head, "cuda")

    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].self_attn.R2_1 = RotateModule(R2_1)
        model.model.layers[i].self_attn.R2_2 = RotateModule(R2_2)
        if low_length_head != 0:
            R2_0 = random_orthogonal_matrix(low_length_head, "cuda")
            model.model.layers[i].self_attn.R2_0 = RotateModule(R2_0)

    # Enable gradients for rotation parameters only
    for name, p in model.named_parameters():
        if any(rn in name for rn in ["R1_1.weight", "R1_2.weight", "R1_0.weight",
                                      "R2_0.weight", "R2_1.weight", "R2_2.weight"]):
            p.requires_grad = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=seqlen,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    model.config.use_cache = False
    model.seqlen = seqlen

    # Pre-training PPL evaluation
    testloader = get_wikitext2(seed=seed, seqlen=2048, tokenizer=tokenizer, eval_mode=True)
    with torch.no_grad():
        ppl_before = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    print(f"PPL before optimization: {ppl_before:.2f}")

    # Training dataset
    calibration_datasets = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = CustomJsonDataset(
        calibration_datasets["train"], tokenizer, block_size=min(seqlen, 2048)
    )
    test_data = CustomJsonDataset(
        calibration_datasets["test"], tokenizer, block_size=seqlen
    )

    # Set up SGDG optimizer (Stiefel manifold)
    trainable_parameters = [
        {
            "params": [model.R1_1.weight, model.R1_2.weight],
            "stiefel": True,
            "lr": learning_rate,
            "momentum": 0.9,
            "nesterov": True,
        }
    ]
    if low_length_hidden != 0:
        trainable_parameters.append({
            "params": [model.R1_0.weight],
            "stiefel": True,
            "lr": learning_rate,
            "momentum": 0.9,
            "nesterov": True,
        })

    for i in range(model.config.num_hidden_layers):
        params = [
            model.model.layers[i].self_attn.R2_1.weight,
            model.model.layers[i].self_attn.R2_2.weight,
        ]
        trainable_parameters.append({
            "params": params,
            "stiefel": True,
            "lr": learning_rate,
            "momentum": 0.9,
            "nesterov": True,
        })
        if low_length_head != 0:
            trainable_parameters.append({
                "params": [model.model.layers[i].self_attn.R2_0.weight],
                "stiefel": True,
                "lr": learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            })

    optimizer = SGDG(trainable_parameters, lr=learning_rate)

    # Training args
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoint"),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        save_safetensors=False,
        logging_steps=1,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
    )

    # Train
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )

    dist.barrier()
    torch.cuda.empty_cache()
    dist.barrier()

    print("Starting rotation optimization...")
    trainer.train()

    # Extract trained rotations
    cpu_state = trainer.model.state_dict()
    R_dict = R_dict | {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if any(rn in key for rn in ["R1_1.weight", "R1_2.weight", "R1_0.weight",
                                     "R2_0.weight", "R2_1.weight", "R2_2.weight"])
    }
    if "R1_0" not in R_dict:
        R_dict["R1_0"] = None

    # Save optimized rotation
    short_name = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    rotation_path = os.path.join(
        output_dir,
        f"R-high-{high_fraction}-low-{low_fraction}-sparse-{sparse_fraction}-{short_name}.bin",
    )
    if local_rank == 0:
        torch.save(R_dict, rotation_path)
        print(f"\nSaved optimized rotation: {rotation_path}")

    # Post-training PPL evaluation
    dist.barrier()
    with torch.no_grad():
        ppl_after = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    print(f"PPL after optimization: {ppl_after:.2f}")
    print(f"PPL improvement: {ppl_before - ppl_after:.4f}")

    return rotation_path


def main():
    """CLI entry point."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Optimize rotation matrices")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or os.path.dirname(config["paths"]["rotation"])
    basis_path = config["paths"]["basis"]

    rotation_path = optimize_rotation(
        model_name=config["model"]["name"],
        basis_path=basis_path,
        output_dir=output_dir,
        seqlen=config["calibration"].get("seqlen", 2048),
        seed=config["calibration"].get("seed", 0),
        high_fraction=config["quantize"]["high_fraction"],
        low_fraction=config["quantize"].get("low_fraction", 0.0),
        sparse_fraction=config["quantize"].get("sparse_fraction", 0.0),
        rotation_granularity=config["quantize"]["rotation_granularity"],
        learning_rate=args.learning_rate or 1.5,
        max_steps=args.max_steps or 100,
        k_bits=config["quantize"]["k_bits"],
        v_bits=config["quantize"]["v_bits"],
        k_groupsize=config["quantize"]["k_groupsize"],
        v_groupsize=config["quantize"]["v_groupsize"],
        high_bits=config["quantize"]["high_bits"],
        low_bits=config["quantize"]["low_bits"],
    )

    print(f"\nDone! Optimized rotation saved to: {rotation_path}")


if __name__ == "__main__":
    main()
