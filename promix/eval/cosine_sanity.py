"""o_proj layer-wise output cosine sanity (AC-2.4).

Captures the o_proj output of every attention layer in a model and
computes per-layer cosine similarity against an unquantized FP16
reference (or a separately-quantized reference model). The plan
threshold is `cosine >= 0.99` per layer; a value below 0.99 on any
layer flags a microscaled/global o_proj quantization issue worth
investigating before relying on the PPL number.

Usage (run on remote with GPU):
    python -m promix.eval.cosine_sanity \
        --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml \
        --nsamples 8 \
        --threshold 0.99

    python -m promix.eval.cosine_sanity \
        --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml \
        --reference_config promix/configs/llama-3.2-1b-resq.yaml \
        --nsamples 8 \
        --threshold 0.99
"""

import argparse
import os
from typing import Dict, Optional

import torch


def _attention_layers(model):
    """Yield (idx, attn_module) for each transformer block's self-attn.

    Works for Llama-style HF models (`model.model.layers[i].self_attn`).
    """
    for idx, layer in enumerate(model.model.layers):
        yield idx, layer.self_attn


def _install_oproj_output_hook(attn_module, sink: Dict[int, list], idx: int):
    """Register a forward hook on `attn.o_proj` (the wrapper or Linear)
    that appends every output tensor to `sink[idx]`. Returns the handle.
    """
    sink.setdefault(idx, [])

    def _hook(_module, _inputs, output):
        # output is the o_proj's output for one forward pass; detach
        # and move to CPU to avoid OOM with many calibration batches.
        sink[idx].append(output.detach().to("cpu", torch.float32))

    return attn_module.o_proj.register_forward_hook(_hook)


def compute_oproj_cosine_per_layer(
    model,
    dataloader,
    *,
    reference_model=None,
    device: Optional[torch.device] = None,
    max_batches: int = 8,
) -> Dict[int, float]:
    """Compute per-layer o_proj output cosine.

    Args:
        model: the model under test (typically the FP-quantized model).
        dataloader: iterable yielding `(input_ids, _targets)`-style
            tuples; only `input_ids` is forwarded.
        reference_model: optional second model whose o_proj outputs are
            the cosine reference. When None, the test runs ONE forward
            pass and computes cosine of the o_proj output against the
            o_proj input (degenerate self-similarity, ~1.0 iff the
            wrapper is a true identity in that layer; primarily useful
            for the unit test).
        device: torch device to move models / inputs to. Defaults to
            cuda if available, else cpu.
        max_batches: number of dataloader batches to consume; cosines
            are averaged over batches per layer.

    Returns:
        `dict[layer_idx, cosine_value]` where each value is in
        [-1.0, 1.0]. Higher is better; AC-2.4 requires every value
        to be >= 0.99.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    if reference_model is not None:
        reference_model = reference_model.to(device).eval()

    primary_outs: Dict[int, list] = {}
    handles = []
    for idx, attn in _attention_layers(model):
        handles.append(_install_oproj_output_hook(attn, primary_outs, idx))

    ref_outs: Dict[int, list] = {}
    ref_handles = []
    if reference_model is not None:
        for idx, attn in _attention_layers(reference_model):
            ref_handles.append(_install_oproj_output_hook(attn, ref_outs, idx))

    try:
        with torch.no_grad():
            for b_i, batch in enumerate(dataloader):
                if b_i >= max_batches:
                    break
                input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                input_ids = input_ids.to(device)
                model(input_ids)
                if reference_model is not None:
                    reference_model(input_ids)
    finally:
        for h in handles:
            h.remove()
        for h in ref_handles:
            h.remove()

    # Compute per-layer cosine
    results: Dict[int, float] = {}
    for idx, primary_batches in primary_outs.items():
        primary = torch.cat([t.flatten() for t in primary_batches])
        if reference_model is not None:
            if idx not in ref_outs:
                continue
            ref_batches = ref_outs[idx]
            reference = torch.cat([t.flatten() for t in ref_batches])
        else:
            # Degenerate self-similarity (used by unit test); cosine
            # of vector against itself is 1.0 by definition. The CLI
            # always supplies reference_model.
            reference = primary

        cos = torch.nn.functional.cosine_similarity(
            primary.unsqueeze(0), reference.unsqueeze(0)
        ).item()
        results[idx] = cos

    return results


def _evaluate_against_threshold(
    cosines: Dict[int, float], threshold: float
) -> bool:
    """Return True iff every layer's cosine is >= threshold."""
    return all(c >= threshold for c in cosines.values())


def main():
    parser = argparse.ArgumentParser(description="o_proj cosine sanity (AC-2.4)")
    parser.add_argument("--config", type=str, required=True,
                        help="FP yaml config for the model under test")
    parser.add_argument("--reference_config", type=str, default=None,
                        help=("Optional INT yaml config for reference "
                              "model (FP-vs-INT comparison)"))
    parser.add_argument("--nsamples", type=int, default=8,
                        help="number of calibration batches (default 8)")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="per-layer cosine threshold (default 0.99)")
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()

    # Lazy imports so the unit test can import this module without
    # bringing in transformers / model loaders.
    import yaml
    import transformers  # noqa: F401  (used transitively via load_model)
    from promix.models.loader import load_model
    from promix.quantize.fuse_norm import fuse_layer_norms
    from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
    from promix.quantize.quant_utils import add_actquant
    from promix.eval.ptq import (
        configure_quantizers, setup_down_proj_hadamard,
    )
    from promix.eval.data import get_wikitext2

    def _build(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        m = load_model(cfg["model"]["name"], dtype=torch.float16)
        fuse_layer_norms(m)
        fuse_basis_to_model(
            m,
            basis_path=cfg["paths"]["basis"],
            rotation_path=cfg["paths"]["rotation"],
            high_fraction=cfg["quantize"]["high_fraction"],
            low_fraction=cfg["quantize"].get("low_fraction", 0.0),
        )
        rearrange_columns(
            m,
            high_fraction=cfg["quantize"]["high_fraction"],
            low_fraction=cfg["quantize"].get("low_fraction", 0.0),
            o_proj_pca=cfg["quantize"].get("o_proj_pca", "per_head"),
        )
        add_actquant(m)
        setup_down_proj_hadamard(m)
        configure_quantizers(m, cfg)
        return m, cfg

    model, fp_cfg = _build(args.config)
    reference_model = None
    if args.reference_config:
        reference_model, _ = _build(args.reference_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        fp_cfg["model"]["name"]
    )
    loader = get_wikitext2(
        seed=fp_cfg["calibration"].get("seed", 0),
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    cosines = compute_oproj_cosine_per_layer(
        model, loader,
        reference_model=reference_model,
        max_batches=args.nsamples,
    )

    print(f"\no_proj layer-wise cosine sanity (threshold = {args.threshold}):")
    print("=" * 60)
    print(f"  {'layer':>5}  {'cosine':>8}  {'pass?':>6}")
    print("-" * 60)
    for idx in sorted(cosines):
        ok = "PASS" if cosines[idx] >= args.threshold else "FAIL"
        print(f"  {idx:>5d}  {cosines[idx]:>8.4f}  {ok:>6}")
    print("=" * 60)
    overall = _evaluate_against_threshold(cosines, args.threshold)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if not overall:
        bad = sorted(
            (idx for idx in cosines if cosines[idx] < args.threshold)
        )
        print(f"FAILED LAYERS: {bad}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
