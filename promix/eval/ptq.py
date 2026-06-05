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

    yaml_config = load_config(args.config)
    print(f"ProMix PTQ Evaluation")
    print(f"  Model: {yaml_config['model']['name']}")
    print(f"  Type: {yaml_config['model']['type']}")
    print(f"  Quantize: a_bits={yaml_config['quantize']['a_bits']}, high_bits={yaml_config['quantize']['high_bits']}")
    print(f"  High fraction: {yaml_config['quantize']['high_fraction']}")
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
        '--input_model', yaml_config['model']['name'],
        '--per_device_eval_batch_size', str(yaml_config['eval']['batch_size']),
        '--model_max_length', str(yaml_config['eval']['max_length']),
        '--fp16', 'True', '--bf16', 'False',
        '--w_bits', str(yaml_config['quantize']['w_bits']),
        '--a_bits', str(yaml_config['quantize']['a_bits']),
        '--k_bits', str(yaml_config['quantize']['k_bits']),
        '--v_bits', str(yaml_config['quantize']['v_bits']),
        '--high_bits', str(yaml_config['quantize']['high_bits']),
        '--low_bits', str(yaml_config['quantize']['low_bits']),
        '--w_clip', '--a_asym', '--k_asym', '--v_asym',
        '--k_groupsize', str(yaml_config['quantize']['k_groupsize']),
        '--v_groupsize', str(yaml_config['quantize']['v_groupsize']),
        '--high_fraction', str(yaml_config['quantize']['high_fraction']),
        '--low_fraction', str(yaml_config['quantize']['low_fraction']),
        '--rotate_mode', yaml_config['quantize']['rotate_mode'],
        '--optimized_rotation_path', yaml_config['paths']['rotation'],
        '--optimized_basis_path', yaml_config['paths']['basis'],
        '--rotation_granularity', yaml_config['quantize']['rotation_granularity'],
        '--rotate',
        '--output_dir', yaml_config['paths']['output_dir'],
    ]
    from utils.process_args import process_args_ptq
    model_args, training_args, ptq_args = process_args_ptq()
    sys.argv = saved_argv

    # Initialize distributed if not already done (torchrun handles this normally)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=1, rank=0)

    # Load model (must untie word embeddings before rotation, matching ResQ ptq.py line 308-310)
    print("Loading model...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_args.input_model)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.float16
    model = LlamaForCausalLM.from_pretrained(
        model_args.input_model,
        torch_dtype=dtype,
        config=config,
    ).cuda()

    # Clone embed_tokens to lm_head (ResQ ptq.py line 334)
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # Reset basis_change to identity (ResQ ptq.py line 336-337)
    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    # Run PTQ (pass model_args for calibration data loading)
    print("Running PTQ pipeline...")
    model = ptq_model(ptq_args, model, model_args)

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
    print(f"\n{'='*50}")
    print(f"Wikitext-2 Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    # lm-eval benchmarks
    tasks_str = ",".join(yaml_config['eval'].get('tasks', []))
    if tasks_str:
        print(f"\nRunning lm-eval tasks: {tasks_str}")
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM
        import datasets as hf_datasets
        hf_datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        model.to(utils.DEV)
        lm = HFLM(pretrained=model, tokenizer=tokenizer)
        lm._device = utils.DEV

        t_results = lm_evaluator.simple_evaluate(
            lm,
            tasks=tasks_str.split(","),
            num_fewshot=yaml_config['eval'].get('num_fewshot', 0),
            batch_size=yaml_config['eval'].get('batch_size', 1),
        )
        print(make_table(t_results))
    else:
        t_results = None

    # Save results
    os.makedirs(yaml_config['paths']['output_dir'], exist_ok=True)
    import json
    results = {"wikitext2_ppl": ppl}
    if t_results and "results" in t_results:
        results["lm_eval"] = t_results["results"]
    with open(os.path.join(yaml_config['paths']['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {yaml_config['paths']['output_dir']}/results.json")


if __name__ == "__main__":
    main()
