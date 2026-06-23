"""Rotation optimization for ResQ mixed-precision quantization (Step 1).

Optimizes orthogonal rotation matrices R1 (hidden dim) and R2 (head dim)
on the Stiefel manifold using Cayley-transform SGD. The rotation is trained
to minimize LM loss after basis reordering, making quantization more effective.

Usage:
    python -m promix.quantize.optimize_rotation --config promix/configs/llama-3.2-1b-resq.yaml
"""

import os
import sys
import datetime
import random
from typing import Any, Dict

import datasets
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from promix.train.modeling_llama_train import LlamaForCausalLM
from promix.train.optimizer import SGDG
from promix.quantize.hadamard import random_orthogonal_matrix, get_hadK, matmul_hadU_cuda
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.utils import DEV, cleanup_memory
from promix.eval.data import get_wikitext2


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


@torch.no_grad()
def _evaluate_ppl_cuda(model, testenc, seqlen, batch_size=8):
    """Simple on-GPU PPL evaluation (model stays on CUDA)."""
    from tqdm import tqdm
    model.eval()
    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[:, :nsamples * seqlen].view(nsamples, seqlen).cuda()
    batches = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss()
    for batch in tqdm(batches, desc="(Eval) Batches"):
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        nlls.append(loss.float())
    return torch.exp(torch.stack(nlls).mean()).item()


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
    a_bits=4,
    a_asym: bool = True,
    o_proj_pca: str = "per_head",
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

    # Prepare model for training: fuse norms, fuse basis (without R), add quant wrappers
    transformers.set_seed(seed)
    model.eval()
    fuse_layer_norms(model)

    # Fuse basis WITHOUT rotation (train_rotations=True mode)
    # Only apply U (PCA basis), not R — R will be applied dynamically in forward
    from promix.quantize.rotation import (
        rotate_embeddings, rotate_head, rotate_attention_inputs,
        rotate_attention_output, rotate_mlp_input, rotate_mlp_output,
        rotate_ov_proj, _apply_had_to_linear,
    )
    U_cpk = torch.load(basis_path, weights_only=False)
    U_attn = U_cpk["attn_mlp"].cuda()

    torch.distributed.barrier()
    rotate_embeddings(model, U_attn)
    rotate_head(model, U_attn)
    cleanup_memory()

    num_heads_model = model.config.num_attention_heads
    head_dim_model = model.config.hidden_size // num_heads_model

    # Detect the global o_proj PCA basis emitted by `o_proj_pca: full_global`
    # in basis.py. When present, replace the per-head `value` PCA on o_proj
    # input with a single hidden_dim transform.
    use_oproj_global = any(
        f"layer.{i}.self_attn.o_proj_global" in U_cpk for i in range(len(model.model.layers))
    )
    if use_oproj_global:
        print(
            "[optimize_rotation] o_proj_global detected; applying full hidden_dim PCA "
            "to o_proj input (per-head v_proj output rotation kept)."
        )

    for idx, layer in enumerate(model.model.layers):
        rotate_attention_inputs(layer, U_attn)
        U_value = U_cpk[f"layer.{idx}.self_attn.value"].cuda()
        if use_oproj_global:
            _apply_had_to_linear(
                layer.self_attn.v_proj, head_dim_model, output=True,
                R2=U_value, per_head=True,
            )
            U_oproj_g = U_cpk[f"layer.{idx}.self_attn.o_proj_global"].cuda().to(torch.float64)
            hidden_dim_model = U_oproj_g.shape[0]
            _apply_had_to_linear(
                layer.self_attn.o_proj, hidden_dim_model, output=False,
                R2=U_oproj_g, per_head=False,
            )
        else:
            rotate_ov_proj(layer, num_heads_model, head_dim_model, U_value, per_head=True)
        rotate_attention_output(layer, U_attn)
        rotate_mlp_input(layer, U_attn)
        rotate_mlp_output(layer, U_attn)

    cleanup_memory()
    add_actquant(model)

    # Rotation training MUST configure the activation quantizers with the
    # same FP/INT bits + segment split as PTQ eval. Without this call,
    # ActQuantWrapper.quantizer stays at default bits=16 and the wrappers
    # are no-ops, so trainer.train() optimizes R against zero quantization
    # noise. Lazy import avoids a circular dependency between
    # promix.quantize and promix.eval.
    from promix.eval.ptq import configure_quantizers as _configure_quantizers

    # `o_proj_pca` MUST match the value the basis bundle was built with.
    # If the bundle has `o_proj_global` keys but configure_quantizers is
    # called without `o_proj_pca: "full_global"`, Step 1 trains the
    # rotation under a per-head o_proj quantizer noise model while Step 2
    # later configures o_proj as global hidden-dim — the two steps then
    # disagree on the o_proj quantizer's groupsize and high/low split.
    _configure_quantizers(model, {"quantize": {
        "a_bits": a_bits,
        "high_bits": high_bits,
        "low_bits": low_bits,
        "high_fraction": high_fraction,
        "low_fraction": low_fraction,
        "a_asym": a_asym,
        "v_bits": 16,  # KV not quantized during R training
        "k_bits": 16,
        "o_proj_pca": o_proj_pca,
    }})
    print(
        f"[optimize_rotation] configured activation quantizers: "
        f"a_bits={a_bits} high_bits={high_bits} low_bits={low_bits}"
    )

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = True

    R_dict = {}

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

    # Move model to CUDA before any forward pass (rotation modules R1/R2 were
    # already created on cuda; this brings embed_tokens/lm_head/layers along).
    model = model.cuda()

    # Pre-training PPL evaluation
    testloader = get_wikitext2(seed=seed, seqlen=2048, tokenizer=tokenizer, eval_mode=True)
    with torch.no_grad():
        ppl_before = _evaluate_ppl_cuda(model, testloader, seqlen, eval_batch_size)
    print(f"PPL before optimization: {ppl_before:.2f}")

    # Training dataset
    from promix.eval.data import CustomJsonDataset
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
        ppl_after = _evaluate_ppl_cuda(model, testloader, seqlen, eval_batch_size)
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
        a_bits=config["quantize"].get("a_bits", 4),
        a_asym=config["quantize"].get("a_asym", True),
        o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"),
    )

    print(f"\nDone! Optimized rotation saved to: {rotation_path}")


if __name__ == "__main__":
    main()
