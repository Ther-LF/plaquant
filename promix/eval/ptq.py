"""ProMix PTQ evaluation — fully independent of project-resq.

Usage:
    python -m promix.eval.ptq --config promix/configs/llama-3.2-1b-resq.yaml
"""

import argparse
import json
import os

import torch
import transformers
import yaml

from promix.utils import DEV, cleanup_memory
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.quantize.hadamard import get_hadK
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def configure_quantizers(model, config):
    """Configure activation quantizers on all wrapped layers."""
    qcfg = config['quantize']
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model_dim // num_heads
    high_fraction = qcfg['high_fraction']
    low_fraction = qcfg.get('low_fraction', 0.0)
    high_bits_length = int(high_fraction * model_dim)
    low_bits_length = int(low_fraction * model_dim)

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        layer_bits = qcfg['a_bits']
        layer_groupsize = -1
        layer_sym = not qcfg.get('a_asym', True)
        layer_high_length = high_bits_length
        layer_low_length = low_bits_length

        if "v_proj" in name and qcfg.get('v_bits', 16) < 16:
            v_groupsize = head_dim
            v_high = int(v_groupsize * high_fraction)
            v_low = int(v_groupsize * low_fraction)
            qlayers[name].out_quantizer.configure(
                bits=qcfg['v_bits'],
                groupsize=v_groupsize,
                sym=not qcfg.get('v_asym', True),
                high_bits_length=v_high,
                high_bits=qcfg['high_bits'],
                low_bits_length=v_low,
                low_bits=qcfg['low_bits'],
            )

        if "o_proj" in name:
            layer_groupsize = head_dim
            layer_high_length = int(head_dim * high_fraction)
            layer_low_length = int(head_dim * low_fraction)

        if "lm_head" in name:
            layer_bits = 16
            layer_high_length = 0
            layer_low_length = 0

        if "down_proj" in name:
            layer_high_length = 0
            layer_low_length = 0

        qlayers[name].quantizer.configure(
            bits=layer_bits,
            groupsize=layer_groupsize,
            sym=layer_sym,
            clip_ratio=1.0,
            high_bits_length=layer_high_length,
            high_bits=qcfg['high_bits'],
            low_bits_length=layer_low_length,
            low_bits=qcfg['low_bits'],
        )


def setup_down_proj_hadamard(model):
    """Set online Hadamard rotation for down_proj layers."""
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = True


def main():
    parser = argparse.ArgumentParser(description="ProMix PTQ Evaluation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"ProMix PTQ Evaluation (independent)")
    print(f"  Model: {config['model']['name']}")
    print(f"  Quantize: a_bits={config['quantize']['a_bits']}, high_bits={config['quantize']['high_bits']}")
    print(f"  High fraction: {config['quantize']['high_fraction']}")
    print()

    # Initialize distributed (needed for rotation barrier)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=1, rank=0)

    transformers.set_seed(config['calibration'].get('seed', 0))

    # 1. Load model
    print("Loading model...")
    model = load_model(config['model']['name'], dtype=torch.float16)

    # 2. Fuse layer norms
    print("Fusing layer norms...")
    fuse_layer_norms(model)

    # 3. Fuse basis + rotation into weights
    print("Fusing basis and rotation...")
    fuse_basis_to_model(
        model,
        basis_path=config['paths']['basis'],
        rotation_path=config['paths']['rotation'],
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize'].get('low_fraction', 0.0),
    )

    # 4. Rearrange columns
    rearrange_columns(
        model,
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize'].get('low_fraction', 0.0),
    )
    cleanup_memory()

    # 5. Add quantization wrappers
    add_actquant(model)
    setup_down_proj_hadamard(model)
    install_column_order_hooks(model)

    # 6. Configure quantizers
    configure_quantizers(model, config)

    # 7. Evaluate PPL
    print("\nEvaluating wikitext perplexity...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    model.seqlen = config['eval']['max_length']
    testloader = get_wikitext2(
        seed=config['calibration'].get('seed', 0),
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    ppl = evaluate_ppl(model, testloader, DEV, batch_size=config['eval']['batch_size'])
    print(f"\n{'='*50}")
    print(f"Wikitext-2 Perplexity: {ppl:.2f}")
    print(f"{'='*50}")

    # 8. lm-eval benchmarks
    tasks_str = ",".join(config['eval'].get('tasks', []))
    t_results = None
    if tasks_str:
        print(f"\nRunning lm-eval tasks: {tasks_str}")
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM
        import datasets as hf_datasets
        hf_datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        model.to(DEV)
        lm = HFLM(pretrained=model, tokenizer=tokenizer)
        lm._device = DEV
        t_results = lm_evaluator.simple_evaluate(
            lm,
            tasks=tasks_str.split(","),
            num_fewshot=config['eval'].get('num_fewshot', 0),
            batch_size=config['eval'].get('batch_size', 1),
        )
        print(make_table(t_results))

    # 9. Save results
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results = {"wikitext2_ppl": ppl}
    if t_results and "results" in t_results:
        results["lm_eval"] = t_results["results"]
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
