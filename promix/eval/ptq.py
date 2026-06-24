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
from promix.quantize.quant_utils import (
    ActQuantWrapper,
    _quant_enabled,
    add_actquant,
    find_qlayers,
)
from promix.quantize.hadamard import get_hadK
from promix.quantize.kv_quant import setup_k_quant
from promix.quantize.gptq import gptq_fwrd
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def configure_quantizers(model, config):
    """Configure activation quantizers on all wrapped layers."""
    qcfg = config['quantize']
    model_dim = model.config.hidden_size
    intermediate_size = getattr(model.config, 'intermediate_size', None)
    num_heads = model.config.num_attention_heads
    head_dim = model_dim // num_heads
    high_fraction = qcfg['high_fraction']
    low_fraction = qcfg.get('low_fraction', 0.0)
    high_bits_length = int(high_fraction * model_dim)
    low_bits_length = int(low_fraction * model_dim)
    # When the basis bundle was built with global hidden-dim PCA on o_proj
    # input, the per-head group concept is gone for o_proj. The o_proj
    # quantizer must use groupsize=-1 with hidden-dim-derived high/low
    # split, matching every other Linear (q/k/v/gate/up/down). Detect via
    # the same config knob basis.py / rotation.py read.
    o_proj_global = qcfg.get('o_proj_pca', 'per_head') == 'full_global'
    # FP block-scaled formats (mxfp8 / nvfp4) extend high/low routing to
    # `down_proj` per the plan: q/k/v/gate/up/down/o_proj all follow the
    # same MXFP8 high + NVFP4 main split, with `down_proj`'s split taken
    # on `intermediate_size` (the post-MLP-up dim) instead of
    # `hidden_size`. The legacy INT W4A4 path intentionally zeroed out
    # down_proj high channels (the ResQ baseline that produced
    # PPL=14.72), so we keep that behavior for INT and switch to the
    # plan-conformant split for FP.
    is_fp = isinstance(qcfg.get('high_bits'), str) or isinstance(
        qcfg.get('a_bits'), str
    )
    if is_fp and intermediate_size is None:
        raise RuntimeError(
            "FP mixed-precision config requires model.config.intermediate_size"
            "; got None — cannot derive down_proj high/low split."
        )

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name in qlayers:
        layer_bits = qcfg['a_bits']
        layer_groupsize = -1
        layer_sym = not qcfg.get('a_asym', True)
        layer_high_length = high_bits_length
        layer_low_length = low_bits_length

        if "v_proj" in name and _quant_enabled(qcfg.get('v_bits', 16)):
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
            if o_proj_global:
                # Global hidden-dim split (no per-head groupsize): o_proj
                # follows the same code path as q/k/v/gate/up/down.
                layer_groupsize = -1
                layer_high_length = high_bits_length
                layer_low_length = low_bits_length
            else:
                layer_groupsize = head_dim
                layer_high_length = int(head_dim * high_fraction)
                layer_low_length = int(head_dim * low_fraction)

        if "lm_head" in name:
            layer_bits = 16
            layer_high_length = 0
            layer_low_length = 0

        if "down_proj" in name:
            if is_fp:
                # FP path: down_proj joins the q/k/v/gate/up/o_proj
                # mixed-precision split. The split lengths come from
                # `intermediate_size`, not `hidden_size`, because
                # down_proj's input is the post-up-proj/gate-proj
                # intermediate-dim activation. Block-alignment holds
                # for all 1B/3B/8B intermediate dims at high_fraction
                # ∈ {1/8, 1/4, 1/2}: e.g. 8192 × 0.125 = 1024
                # (divisible by 32 for MXFP8, by 16 for NVFP4); 14336
                # × 0.125 = 1792 (1792/32 = 56). Online Hadamard
                # transform on down_proj input is unchanged; it
                # composes cleanly with per-block FP scales.
                layer_high_length = int(high_fraction * intermediate_size)
                layer_low_length = int(low_fraction * intermediate_size)
            else:
                # Legacy INT W4A4: down_proj is single-precision
                # (no high/low split). Preserves the ResQ baseline
                # PPL=14.72.
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


def run_gptq_if_enabled(model, config, dev, *, _gptq_fwrd=None,
                        _get_wikitext2=None, _AutoTokenizer=None):
    """Run GPTQ weight quantization iff `w_bits` is anything except FP16.

    Extracted from `main()` so the FP-aware gate can be exercised in tests
    without booting the full `main()` model-load + tokenizer + evaluator
    machinery. Keeps the same control flow `main()` uses; the underscore
    keyword arguments are seams for tests to stub the heavyweight
    dependencies (real callers leave them at None to use the imports).

    The gate uses `_quant_enabled(...)`, the shared FP-aware predicate,
    so string FP-format identifiers like "nvfp4" / "mxfp8" route through
    GPTQ instead of TypeErroring on a bare `bits < 16`.

    Returns True iff GPTQ ran, False if w_bits was 16 (no quant).
    """
    if not _quant_enabled(config['quantize']['w_bits']):
        return False
    if _gptq_fwrd is None:
        _gptq_fwrd = gptq_fwrd
    if _get_wikitext2 is None:
        from promix.eval.data import get_wikitext2 as _get_wikitext2
    if _AutoTokenizer is None:
        _AutoTokenizer = transformers.AutoTokenizer
    print("Running GPTQ weight quantization...")
    tokenizer_calib = _AutoTokenizer.from_pretrained(config['model']['name'])
    trainloader = _get_wikitext2(
        nsamples=config['calibration'].get('nsamples', 128),
        seed=config['calibration'].get('seed', 0),
        seqlen=2048,
        tokenizer=tokenizer_calib,
        eval_mode=False,
    )
    _gptq_fwrd(model, trainloader, dev, config)
    return True


def init_distributed_for_ptq_main_if_needed():
    """Initialize a single-process `nccl env://` distributed context.

    `prepare_ptq_model()` reaches `fuse_basis_to_model()` which calls
    `torch.distributed.barrier()` (the rotation pipeline assumes a
    valid process group even for single-GPU runs). Both
    `promix.eval.ptq.main()` and `promix.eval.cosine_sanity.main()`
    must call this helper BEFORE `prepare_ptq_model()` to avoid the
    barrier crashing.

    Idempotent: no-op when distributed is already initialized.
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=1, rank=0)


def prepare_ptq_model(config, dev, *, run_gptq=True):
    """Build a fully PTQ-prepared model from a config dict.

    Performs the canonical Step 2 sequence: load model, fuse layer
    norms, fuse basis + rotation, rearrange columns (with `o_proj_pca`
    awareness), add ActQuantWrapper, setup down_proj online Hadamard,
    install column-order hooks on attention output, run GPTQ if the
    weight bits are quantized, configure activation quantizers, and
    setup KV cache quantization. After this returns, the model is
    ready for forward passes that exercise the full PTQ algorithm
    Step 2 evaluates against.

    Used by both `ptq.main()` (Step 2 PPL) and
    `promix.eval.cosine_sanity` (AC-2.4 layer-wise cosine), so the two
    measurement paths cannot drift in their PTQ build sequence.

    Args:
        config: yaml config dict (already loaded).
        dev: torch device string / object passed to GPTQ + KV setup.
        run_gptq: keep True for production paths; tests can pass False
            to skip the GPTQ tokenizer + dataloader machinery and
            still get a fully wrapped model.

    Returns: the prepared `model`.
    """
    print("Loading model...")
    model = load_model(config['model']['name'], dtype=torch.float16)
    print("Fusing layer norms...")
    fuse_layer_norms(model)
    print("Fusing basis and rotation...")
    fuse_basis_to_model(
        model,
        basis_path=config['paths']['basis'],
        rotation_path=config['paths']['rotation'],
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize'].get('low_fraction', 0.0),
    )
    rearrange_columns(
        model,
        high_fraction=config['quantize']['high_fraction'],
        low_fraction=config['quantize'].get('low_fraction', 0.0),
        o_proj_pca=config['quantize'].get('o_proj_pca', 'per_head'),
    )
    cleanup_memory()
    add_actquant(model)
    setup_down_proj_hadamard(model)
    install_column_order_hooks(model)
    if run_gptq:
        run_gptq_if_enabled(model, config, dev)
    configure_quantizers(model, config)
    setup_k_quant(model, config,
                  basis_path=config['paths']['basis'],
                  rotation_path=config['paths']['rotation'])
    return model


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

    # Initialize distributed (needed by fuse_basis_to_model's barrier).
    init_distributed_for_ptq_main_if_needed()

    transformers.set_seed(config['calibration'].get('seed', 0))

    # 1-7. Build the canonical PTQ model. The same helper is used by
    # `promix.eval.cosine_sanity` so AC-2.4 measures the SAME path
    # Step 2 PPL evaluates against (no drift between measurement
    # surfaces).
    model = prepare_ptq_model(config, DEV, run_gptq=True)

    # 8. Evaluate PPL
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

    # 9. lm-eval benchmarks
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

    # 10. Save results
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
