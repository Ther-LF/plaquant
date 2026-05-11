"""
PLAQuant model module.

Uses ResQ's modeling_llama_2.LlamaForCausalLM and eval_utils.main.ptq_model()
for correct model setup (rotation fusion, quant wrappers, KV quant attention).

Usage:
    model = ResQModel.from_checkpoint(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        checkpoint_path="project-resq/fake_quant/qmodels/W4A4KV4-Llama-3.2-1B-v2.pt",
        rotation_path="project-resq/fake_quant/rotation/R-*.bin",
        basis_path="project-resq/fake_quant/rotation/U-*.bin",
    )
    ppl = model.evaluate_ppl()
"""

import sys
import os
import torch
from typing import Optional


class ResQModel:
    """ResQ quantized model using ResQ's own model class and setup pipeline."""

    def __init__(self, model, tokenizer, ptq_args, seqlen=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.ptq_args = ptq_args
        self.seqlen = seqlen
        self.device = "cpu"

    @classmethod
    def from_checkpoint(
        cls,
        model_name: str,
        checkpoint_path: str,
        rotation_path: str,
        basis_path: str,
        resq_path: str = "project-resq/fake_quant",
        high_fraction: float = 0.125,
        a_bits: int = 4,
        k_bits: int = 4,
        v_bits: int = 4,
        high_bits: int = 8,
        k_groupsize: int = 64,
        v_groupsize: int = 64,
        real_quant: bool = True,
        seqlen: int = 2048,
    ) -> "ResQModel":
        """Load model using ResQ's exact pipeline."""
        # Add ResQ to path
        sys.path.insert(0, resq_path)

        # Import ResQ modules
        from eval_utils.modeling_llama_2 import LlamaForCausalLM
        from eval_utils.main import ptq_model
        from utils.process_args import process_args_ptq
        from transformers import AutoTokenizer

        # Build sys.argv for process_args_ptq
        sys.argv = [
            "ptq.py",
            "--input_model", model_name,
            "--w_bits", "4",
            "--a_bits", str(a_bits),
            "--k_bits", str(k_bits),
            "--v_bits", str(v_bits),
            "--high_bits", str(high_bits),
            "--high_fraction", str(high_fraction),
            "--low_fraction", "0.0",
            "--a_asym", "--k_asym", "--v_asym",
            "--k_groupsize", str(k_groupsize),
            "--v_groupsize", str(v_groupsize),
            "--rotate", "--rotate_mode", "resq",
            "--rotation_granularity", "full_shared",
            "--optimized_rotation_path", rotation_path,
            "--optimized_basis_path", basis_path,
            "--fp32_had",
        ]
        if real_quant:
            sys.argv += ["--real_quant", "--load_qmodel", checkpoint_path]

        model_args, training_args, ptq_args = process_args_ptq()

        # Load model using ResQ's LlamaForCausalLM
        print(f"Loading model: {model_name}")
        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu"
        )
        model.seqlen = seqlen
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Full ResQ setup: fuse norms, rotations, quant wrappers, load weights
        print("Setting up quantization (ptq_model)...")
        ptq_model(ptq_args, model, model_args=model_args)
        print("Model ready.")

        return cls(model, tokenizer, ptq_args, seqlen)

    def evaluate_ppl(self, device: str = "cuda") -> float:
        """Evaluate WikiText-2 PPL using ResQ's evaluator."""
        sys.path.insert(0, "project-resq/fake_quant")
        from utils import data_utils, eval_utils

        self.model.eval()
        testloader = data_utils.get_wikitext2(
            seed=0, seqlen=self.seqlen, tokenizer=self.tokenizer, eval_mode=True
        )
        ppl = eval_utils.evaluator(self.model, testloader, device, self.ptq_args)
        return ppl

    def to(self, device: str):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self

    def set_kernel(self, impl: str = "cutlass"):
        """Replace PyTorch GEMM with CUTLASS in all quantized layers.

        This monkey-patches forward_real_quant to use CUTLASS kernels.
        """
        raise NotImplementedError("Kernel swap not yet implemented")
