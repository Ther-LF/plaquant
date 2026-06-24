"""o_proj layer-wise output cosine sanity (AC-2.4).

Measures per-layer cosine similarity between an FP-quantized model's
o_proj output and a reference (typically INT per-group W4A4) model's
o_proj output, batch by batch on the Wikitext eval set. The plan
threshold is `cosine >= 0.99` per layer.

Both primary and reference models are built through
`promix.eval.ptq.prepare_ptq_model`, so the harness measures the SAME
PTQ algorithm Step 2 PPL evaluates against — no drift between the
two measurement paths.

Usage (run on remote with GPU access):
    python -m promix.eval.cosine_sanity \
        --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml \
        --reference_config promix/configs/llama-3.2-1b-w4a4.yaml \
        --nsamples 8 --threshold 0.99
"""

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _attention_layers(model):
    """Yield (idx, attn_module) for each transformer block's self-attn.

    Works for Llama-style HF models (`model.model.layers[i].self_attn`).
    """
    for idx, layer in enumerate(model.model.layers):
        yield idx, layer.self_attn


def _install_oproj_output_hook(attn_module, sink: Dict[int, list], idx: int):
    """Register a forward hook on `attn.o_proj` (the wrapper or Linear)
    that appends every output tensor to `sink[idx]`.
    """
    sink.setdefault(idx, [])

    def _hook(_module, _inputs, output):
        sink[idx].append(output.detach().to("cpu", torch.float32))

    return attn_module.o_proj.register_forward_hook(_hook)


def chunk_wikitext_for_cosine(
    tokenizer, nsamples: int, seqlen: int, *, seed: int = 0
) -> List[Tuple[torch.Tensor, None]]:
    """Tokenize WikiText-2 eval and produce `nsamples` deterministic
    `(input_ids[1, seqlen], None)` tuples for the cosine harness.

    The harness's API takes an iterable of `(input_ids, _)` tuples
    (matching the calibration loader contract used by GPTQ). Round-10's
    CLI passed the raw BatchEncoding object from
    `get_wikitext2(eval_mode=True)` directly; the BatchEncoding doesn't
    iterate as `(input_ids, _)` tuples. This helper builds the right
    contract from the tokenizer output explicitly.

    Args:
        tokenizer: HF tokenizer, ready to encode WikiText-2.
        nsamples: number of `[1, seqlen]` chunks to take.
        seqlen: each chunk's sequence length.
        seed: RNG seed for chunk-start positions; deterministic.

    Returns: list of length `nsamples`; each entry is
    `(input_ids, None)` with `input_ids.shape == (1, seqlen)`.
    """
    import random

    import datasets  # lazy: avoid bringing it in for unit tests

    testdata = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )["test"]
    testenc = tokenizer(
        text="\n\n".join(testdata["text"]), return_tensors="pt"
    )
    return _chunk_from_input_ids(
        testenc.input_ids, nsamples=nsamples, seqlen=seqlen, seed=seed
    )


def _chunk_from_input_ids(
    input_ids: torch.Tensor, *, nsamples: int, seqlen: int, seed: int = 0
) -> List[Tuple[torch.Tensor, None]]:
    """Lower-level chunker: takes a `(1, total_tokens)` tensor of token
    IDs and returns `nsamples` non-overlapping `(input_ids[1, seqlen],
    None)` chunks at deterministic random offsets.

    Factored out for unit testing without bringing `datasets` /
    network access into the test harness.
    """
    import random as _random

    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
        f"expected (1, T) tokens; got shape {tuple(input_ids.shape)}"
    )
    total = input_ids.shape[1]
    if total < seqlen + 1:
        raise RuntimeError(
            f"insufficient tokens for cosine sanity: have {total}, need >= {seqlen+1}"
        )
    rng = _random.Random(seed)
    out: List[Tuple[torch.Tensor, None]] = []
    for _ in range(nsamples):
        i = rng.randint(0, total - seqlen - 1)
        chunk = input_ids[:, i : i + seqlen]
        out.append((chunk, None))
    return out


def compute_oproj_cosine_per_layer(
    model,
    dataloader: Iterable,
    *,
    reference_model=None,
    device: Optional[torch.device] = None,
    max_batches: int = 8,
) -> Dict[int, float]:
    """Compute per-layer o_proj output cosine.

    Args:
        model: the model under test (typically the FP-quantized model).
        dataloader: iterable yielding `(input_ids, _ignored)` tuples;
            only `input_ids` is forwarded.
        reference_model: optional second model whose o_proj outputs are
            the cosine reference. When provided, returns `cosine(model
            o_proj output, reference o_proj output)` per layer — this is
            the AC-2.4 measurement.
            **Unit-test only**: when None, returns degenerate
            self-similarity (~1.0) per layer; the production CLI never
            invokes this path because `--reference_config` is required.
        device: torch device. Defaults to cuda if available, else cpu.
        max_batches: number of dataloader batches to consume.

    Returns: `dict[layer_idx, cosine_value]` with values in [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    if reference_model is not None:
        reference_model = reference_model.to(device).eval()

    primary_outs: Dict[int, list] = {}
    primary_layer_count = 0
    handles = []
    for idx, attn in _attention_layers(model):
        handles.append(_install_oproj_output_hook(attn, primary_outs, idx))
        primary_layer_count += 1

    # Fail closed: empty layer iteration means the model has no
    # accessible `model.model.layers[i].self_attn` and the harness
    # would silently return {} -> "OVERALL: PASS" with no rows.
    if primary_layer_count == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(
            "cosine sanity: primary model has zero attention layers "
            "(model.model.layers is empty or self_attn / o_proj missing); "
            "cannot measure AC-2.4 without layers"
        )

    ref_outs: Dict[int, list] = {}
    ref_handles = []
    reference_layer_count = 0
    if reference_model is not None:
        for idx, attn in _attention_layers(reference_model):
            ref_handles.append(_install_oproj_output_hook(attn, ref_outs, idx))
            reference_layer_count += 1
        if reference_layer_count != primary_layer_count:
            for h in handles:
                h.remove()
            for h in ref_handles:
                h.remove()
            raise RuntimeError(
                f"cosine sanity: primary has {primary_layer_count} "
                f"attention layers but reference has "
                f"{reference_layer_count}; the two models must have "
                f"matching architecture for AC-2.4"
            )

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

    # Post-forward validation: fail closed on any contract violation
    # so an empty / asymmetric capture cannot silently print PASS.
    if not primary_outs:
        raise RuntimeError(
            "cosine sanity: hooks captured zero o_proj outputs; "
            "verify the dataloader produced batches and the forward "
            "pass actually traversed the attention layers"
        )
    for idx, batches in primary_outs.items():
        if len(batches) == 0:
            raise RuntimeError(
                f"cosine sanity: layer {idx} captured zero primary "
                f"o_proj outputs; the forward pass did not reach this "
                f"layer"
            )
    if reference_model is not None:
        missing = sorted(set(primary_outs) - set(ref_outs))
        if missing:
            raise RuntimeError(
                f"cosine sanity: layers {missing} have primary captures "
                f"but no reference captures; the two models' forward "
                f"paths disagree on which layers are exercised"
            )
        for idx, batches in ref_outs.items():
            if len(batches) == 0:
                raise RuntimeError(
                    f"cosine sanity: layer {idx} captured zero "
                    f"reference o_proj outputs"
                )

    results: Dict[int, float] = {}
    for idx, primary_batches in primary_outs.items():
        primary = torch.cat([t.flatten() for t in primary_batches])
        if reference_model is not None:
            reference = torch.cat([t.flatten() for t in ref_outs[idx]])
        else:
            # Unit-test sentinel: cosine of the vector with itself.
            reference = primary

        cos = torch.nn.functional.cosine_similarity(
            primary.unsqueeze(0), reference.unsqueeze(0)
        ).item()
        results[idx] = cos

    return results


def _evaluate_against_threshold(
    cosines: Dict[int, float], threshold: float
) -> bool:
    """Return True iff every layer's cosine is >= threshold.

    Fails closed on empty input: `all([])` is True in Python, which
    would let an empty cosine dict silently print "OVERALL: PASS"
    with zero rows. AC-2.4 requires every attention layer to satisfy
    the threshold; "no layers measured" is a harness failure, not
    a pass.
    """
    if not cosines:
        raise RuntimeError(
            "cosine sanity: no per-layer cosines to evaluate; the harness "
            "produced an empty result. This is a measurement failure, "
            "not a pass — investigate the model build / forward pass."
        )
    return all(c >= threshold for c in cosines.values())


def _build_argparser():
    """Argparse layered separately so tests can exercise CLI argument
    validation without invoking the (heavy) main path."""
    parser = argparse.ArgumentParser(description="o_proj cosine sanity (AC-2.4)")
    parser.add_argument("--config", type=str, required=True,
                        help="FP yaml config for the model under test")
    parser.add_argument("--reference_config", type=str, required=True,
                        help=("INT yaml config for the reference model. "
                              "Layer-wise cosine is measured between the "
                              "two models' o_proj outputs. Both configs "
                              "MUST point at the same base checkpoint "
                              "(model.name) — comparing different "
                              "checkpoints is not an AC-2.4 measurement."))
    parser.add_argument("--nsamples", type=int, default=8,
                        help="number of WikiText-2 calibration chunks (default 8)")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="per-layer cosine threshold (default 0.99)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="WikiText-2 chunk seqlen (default 2048)")
    return parser


def main():
    args = _build_argparser().parse_args()

    # Lazy imports so unit tests can import this module without
    # bringing in transformers / model loaders.
    import yaml
    import transformers

    from promix.eval.ptq import (
        init_distributed_for_ptq_main_if_needed,
        prepare_ptq_model,
    )
    from promix.utils import DEV

    # `prepare_ptq_model()` reaches `fuse_basis_to_model()` which calls
    # `torch.distributed.barrier()`. Initialize the same single-process
    # nccl/env-init context PTQ uses before any model prep.
    init_distributed_for_ptq_main_if_needed()

    with open(args.config) as f:
        primary_cfg = yaml.safe_load(f)
    with open(args.reference_config) as f:
        ref_cfg = yaml.safe_load(f)

    if primary_cfg["model"]["name"] != ref_cfg["model"]["name"]:
        raise SystemExit(
            f"--config and --reference_config must use the same base "
            f"checkpoint (model.name); got "
            f"{primary_cfg['model']['name']!r} vs "
            f"{ref_cfg['model']['name']!r}. Comparing different checkpoints "
            f"is not an AC-2.4 measurement."
        )

    model = prepare_ptq_model(primary_cfg, DEV, run_gptq=True)
    reference_model = prepare_ptq_model(ref_cfg, DEV, run_gptq=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        primary_cfg["model"]["name"]
    )
    chunks = chunk_wikitext_for_cosine(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen,
        seed=primary_cfg.get("calibration", {}).get("seed", 0),
    )

    cosines = compute_oproj_cosine_per_layer(
        model, chunks,
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
        bad = sorted(idx for idx in cosines if cosines[idx] < args.threshold)
        print(f"FAILED LAYERS: {bad}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
