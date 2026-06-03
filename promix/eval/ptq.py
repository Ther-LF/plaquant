"""ProMix PTQ evaluation — main entry point.

Equivalent to project-resq/fake_quant/ptq.py but using our model plugin architecture.
For Phase 1, delegates most logic to project-resq's eval_utils/main.py (ptq_model).

Usage:
    python -m promix.eval.ptq --config promix/configs/llama-3.2-1b-resq.yaml
"""

import sys
import os
import argparse

import torch
import yaml

# Add project-resq to path
_resq_path = os.path.join(os.path.dirname(__file__), '../../project-resq/fake_quant')
if _resq_path not in sys.path:
    sys.path.insert(0, _resq_path)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ProMix PTQ Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"ProMix PTQ Evaluation")
    print(f"  Model: {config['model']['name']}")
    print(f"  Type: {config['model']['type']}")
    print(f"  Quantize: a_bits={config['quantize']['a_bits']}, high_bits={config['quantize']['high_bits']}")
    print(f"  High fraction: {config['quantize']['high_fraction']}")
    print()

    # For Phase 1: delegate to project-resq's ptq_model function
    # This ensures we get identical results to validate our pipeline
    from utils.process_args import process_args_ptq
    from eval_utils.main import ptq_model
    from eval_utils.modeling_llama_2 import LlamaForCausalLM
    from transformers import AutoTokenizer
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    # Build args namespace from config (mimicking ResQ's argparse)
    ptq_args = argparse.Namespace(
        input_model=config['model']['name'],
        per_device_eval_batch_size=config['eval']['batch_size'],
        model_max_length=config['eval']['max_length'],
        fp16=True,
        bf16=False,
        w_bits=config['quantize']['w_bits'],
        a_bits=config['quantize']['a_bits'],
        k_bits=config['quantize']['k_bits'],
        v_bits=config['quantize']['v_bits'],
        high_bits=config['quantize']['high_bits'],
        low_bits=config['quantize']['low_bits'],
        w_clip=config['quantize']['w_clip'],
        a_asym=config['quantize']['a_asym'],
        k_asym=config['quantize']['k_asym'],
        v_asym=config['quantize']['v_asym'],
        k_groupsize=config['quantize']['k_groupsize'],
        v_groupsize=config['quantize']['v_groupsize'],
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize']['low_fraction'],
        rotate_mode=config['quantize']['rotate_mode'],
        optimized_rotation_path=config['paths']['rotation'],
        optimized_basis_path=config['paths']['basis'],
        rotation_granularity=config['quantize']['rotation_granularity'],
        rotate=True,
        tasks=",".join(config['eval']['tasks']),
        output_dir=config['paths']['output_dir'],
        seed=42,
        w_groupsize=-1,
        a_groupsize=-1,
        a_clip_ratio=1.0,
        w_asym=False,
        lm_eval=True,
        lm_eval_batch_size='auto',
        distribute_model=False,
        train_rotations=False,
        sparse_fraction=0.0,
        nsamples=128,
        cal_dataset='wikitext2',
        cal_nsamples=128,
        cal_seqlen=512,
    )

    # Load model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        ptq_args.input_model,
        torch_dtype=torch.float16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(ptq_args.input_model)

    # Run PTQ
    print("Running PTQ pipeline...")
    ptq_model(ptq_args, model)

    # Evaluate
    print("Evaluating...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=ptq_args.per_device_eval_batch_size)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=config['eval']['tasks'],
        batch_size=ptq_args.per_device_eval_batch_size,
    )

    print("\n" + make_table(results))

    # Save results
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    import json
    with open(os.path.join(config['paths']['output_dir'], 'results.json'), 'w') as f:
        json.dump(results['results'], f, indent=2)
    print(f"\nResults saved to {config['paths']['output_dir']}/results.json")


if __name__ == "__main__":
    main()
