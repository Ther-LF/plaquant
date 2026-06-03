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

    # Build args namespace from config (mimicking ResQ's argparse)
    # Include ALL attributes that ptq_model() might access
    ptq_args = argparse.Namespace(
        input_model=config['model']['name'],
        per_device_eval_batch_size=config['eval']['batch_size'],
        model_max_length=config['eval']['max_length'],
        fp16=True,
        bf16=False,
        # Weight quantization
        w_bits=config['quantize']['w_bits'],
        w_clip=config['quantize']['w_clip'],
        w_asym=False,
        w_groupsize=-1,
        w_rtn=False,
        # Activation quantization
        a_bits=config['quantize']['a_bits'],
        a_asym=config['quantize']['a_asym'],
        a_groupsize=-1,
        a_clip_ratio=1.0,
        # KV cache quantization
        k_bits=config['quantize']['k_bits'],
        v_bits=config['quantize']['v_bits'],
        k_asym=config['quantize']['k_asym'],
        v_asym=config['quantize']['v_asym'],
        k_groupsize=config['quantize']['k_groupsize'],
        v_groupsize=config['quantize']['v_groupsize'],
        k_clip_ratio=1.0,
        v_clip_ratio=1.0,
        k_pre_rope=False,
        # Mixed precision
        high_bits=config['quantize']['high_bits'],
        low_bits=config['quantize']['low_bits'],
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize']['low_fraction'],
        # Rotation
        rotate_mode=config['quantize']['rotate_mode'],
        optimized_rotation_path=config['paths']['rotation'],
        optimized_basis_path=config['paths']['basis'],
        rotation_granularity=config['quantize']['rotation_granularity'],
        rotate=True,
        train_rotations=False,
        sparse_fraction=0.0,
        residual_fraction=0.0,
        # Hadamard
        fp32_had=False,
        # Misc
        int8_down_proj=False,
        load_qmodel_path=None,
        save_qmodel_path=None,
        tasks=",".join(config['eval']['tasks']),
        output_dir=config['paths']['output_dir'],
        seed=0,
        nsamples=128,
        lm_eval=True,
        lm_eval_batch_size='auto',
        distribute_model=False,
        # Eval utils
        bsz=config['eval']['batch_size'],
        capture_layer_io=False,
        eval_dataset='wikitext2',
        layer_idx=10,
        vision_lm=False,
        multigpu=False,
        model_name=config['model']['name'].split('/')[-1],
    )

    # Initialize distributed if not already done (torchrun handles this normally)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=1, rank=0)

    # Load model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        ptq_args.input_model,
        torch_dtype=torch.float16,
    ).cuda()
    # Note: model.seqlen is set AFTER ptq_model (matching ResQ ptq.py line 393)
    tokenizer = AutoTokenizer.from_pretrained(ptq_args.input_model)

    # Run PTQ
    print("Running PTQ pipeline...")
    ptq_model(ptq_args, model)
    # Note: after ptq_model, model layers are on CPU
    # evaluator() handles per-layer GPU placement internally

    # Set seqlen after ptq_model (matching ResQ's ptq.py line 393)
    model.seqlen = config['eval']['max_length']
    ptq_args.vision_lm = False
    ptq_args.model_name = config['model']['name'].split('/')[-1]

    # Use ResQ's evaluate() function directly to ensure identical behavior
    print("Evaluating (using ResQ evaluate function)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("resq_ptq",
        os.path.join(_resq_path, "ptq.py"))
    resq_ptq = importlib.util.module_from_spec(spec)

    # We only need the evaluate function, extract it manually
    tokenizer = AutoTokenizer.from_pretrained(ptq_args.input_model)
    from utils.data_utils import get_wikitext2
    from utils.eval_utils import evaluator as ppl_evaluator
    from utils import utils

    model.config.use_cache = False
    testloader = get_wikitext2(seed=ptq_args.seed, seqlen=model.seqlen, tokenizer=tokenizer, eval_mode=True, vision=False)
    ppl = ppl_evaluator(model, testloader, utils.DEV, ptq_args)
    model.config.use_cache = True
    print(f"\n{'='*50}")
    print(f"Wikitext-2 Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    # Save results
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    import json
    results = {"wikitext2_ppl": ppl}
    with open(os.path.join(config['paths']['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {config['paths']['output_dir']}/results.json")


if __name__ == "__main__":
    main()
