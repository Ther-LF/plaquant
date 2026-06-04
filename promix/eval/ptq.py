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

    # Change to fake_quant directory (ResQ's imports assume cwd = fake_quant)
    os.chdir(_resq_path)

    # Use ResQ's process_args_ptq to construct args (ensures 100% match)
    import sys
    saved_argv = sys.argv
    sys.argv = ['ptq.py',
        '--input_model', config['model']['name'],
        '--per_device_eval_batch_size', str(config['eval']['batch_size']),
        '--model_max_length', str(config['eval']['max_length']),
        '--fp16', 'True', '--bf16', 'False',
        '--w_bits', str(config['quantize']['w_bits']),
        '--a_bits', str(config['quantize']['a_bits']),
        '--high_bits', str(config['quantize']['high_bits']),
        '--low_bits', str(config['quantize']['low_bits']),
        '--w_clip', '--a_asym', '--k_asym', '--v_asym',
        '--k_groupsize', str(config['quantize']['k_groupsize']),
        '--v_groupsize', str(config['quantize']['v_groupsize']),
        '--high_fraction', str(config['quantize']['high_fraction']),
        '--low_fraction', str(config['quantize']['low_fraction']),
        '--rotate_mode', config['quantize']['rotate_mode'],
        '--optimized_rotation_path', config['paths']['rotation'],
        '--optimized_basis_path', config['paths']['basis'],
        '--rotation_granularity', config['quantize']['rotation_granularity'],
        '--rotate',
        '--output_dir', config['paths']['output_dir'],
    ]
    from utils.process_args import process_args_ptq
    model_args, training_args, ptq_args = process_args_ptq()
    sys.argv = saved_argv

    # Initialize distributed if not already done (torchrun handles this normally)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=1, rank=0)

    # Load model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_args.input_model,
        torch_dtype=torch.float16,
    ).cuda()

    # Run PTQ (pass model_args for calibration data loading)
    print("Running PTQ pipeline...")
    ptq_model(ptq_args, model, model_args)

    # Debug: print model/quantizer state after ptq_model
    print("\n=== DEBUG: Model state after ptq_model ===")
    print(f"embed_tokens device: {model.model.embed_tokens.weight.device}")
    print(f"lm_head device: {model.lm_head.weight.device}")
    print(f"norm device: {model.model.norm.weight.device}")
    for i, layer in enumerate(model.model.layers[:2]):
        print(f"\nLayer {i}:")
        qp = layer.self_attn.q_proj
        print(f"  q_proj type: {type(qp).__name__}")
        if hasattr(qp, 'quantizer'):
            q = qp.quantizer
            print(f"  q_proj.quantizer: bits={q.bits}, high_bits={q.high_bits}, high_bits_length={q.high_bits_length}, low_bits={q.low_bits}, low_bits_length={q.low_bits_length}")
            print(f"  q_proj.online_full_had={qp.online_full_had}")
        if hasattr(qp, 'module'):
            print(f"  q_proj.module.weight: device={qp.module.weight.device}, shape={qp.module.weight.shape}")
        dp = layer.mlp.down_proj
        if hasattr(dp, 'quantizer'):
            q = dp.quantizer
            print(f"  down_proj.quantizer: bits={q.bits}, high_bits={q.high_bits}, high_bits_length={q.high_bits_length}")
            print(f"  down_proj.online_full_had={dp.online_full_had}")
    print("=== END DEBUG ===\n")

    # Set seqlen after ptq_model (matching ResQ ptq.py line 393)
    model.seqlen = training_args.model_max_length
    ptq_args.vision_lm = False
    ptq_args.model_name = model_args.input_model.split('/')[-1]

    # Evaluate
    print("Evaluating wikitext perplexity...")
    from utils.data_utils import get_wikitext2
    from utils.eval_utils import evaluator as ppl_evaluator
    from utils import utils

    tokenizer = AutoTokenizer.from_pretrained(model_args.input_model)
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
