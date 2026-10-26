"""Round-2 ALGORITHM-axis ablation: improve the ACTIVATION quantizer at EQUAL
bits/weight, under the exact frozen-GPTQ fake-quant protocol of round 1.

Round 1 proved the entire W4A4-vs-GPTQ deficit on 3B is ACTIVATION quantization
(+0.56; weights-only 8.084 already beats GPTQ 8.20), and that the activation
error is BROAD (more MXFP8 channels recover only -0.04). So the lever is a better
activation NVFP4 quantizer at the SAME bits (activations are free on bits/weight).

This harness reuses round-1's `build_model` (fuse-norm -> PCA rotation -> rearrange
-> actquant -> online-Hadamard(down) -> frozen-GPTQ -> configure_quantizers -> KV)
and the SAME wikitext-2 `evaluate_ppl`. Anchor: 3B r=1/8 = 8.6470.

Variants (all activation-only, weight bits unchanged):
  baseline : stock single-level NVFP4 activation (sanity == anchor)
  2level   : per-TOKEN (per-row) FP32 global * per-16 E4M3 micro NVFP4 activation
             (vLLM/llm-compressor style). MXFP8 high segment unchanged.
  mse      : per-block activation NVFP4 scale MSE search (A2-on-activations):
             the block scale stays on the E4M3 grid but is chosen to minimise
             per-block reconstruction MSE (search base + 3 lower grid steps).
  smooth   : SmoothQuant. Per-input-channel scale s folded OFFLINE into the weight
             (W[:,k]*=s_k, re-GPTQ'd) and divided out of the activation
             (x[:,k]/=s_k). Narrows per-channel activation dynamic range.
             Applied to main path (q/k/v/gate/up) + down; o_proj skipped (per-head
             reorder). alpha tunable (--alpha, default 0.5).
  outlier  : per-token scale correction on high-kurtosis tokens only (a per-row
             FP32 global applied to the main NVFP4 segment for the top
             --outlier_frac tokens by kurtosis; rest stay single-level).

Each prints `ABLATION_RESULT <json>` and writes a JSON sidecar.
Deployability is NOT decided here (accuracy first); see docs/round2_activation_quant.md.
"""
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import quant_utils as qu
from promix.quantize import basis as basismod
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.quantize.kv_quant import setup_k_quant
from promix.scripts import ablation_acc_per_bit as a1mod

SIZES = a1mod.SIZES
bits_per_weight = a1mod.bits_per_weight
ensure_rotation_file = a1mod.ensure_rotation_file
MAIN_PATH = a1mod.MAIN_PATH
TARGETS = a1mod.TARGETS
SMOOTH_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")  # no o_proj


# ---------------------------------------------------------------------------
# Activation-only NVFP4 variants (patch qu._apply_fp_format -> affects ONLY the
# activation path; GPTQ weights use select_block_scales/frozen-scale, KV is off)
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0


def _nvfp4_2level(x, block=16):
    """Per-row(token) FP32 global * per-16 E4M3 micro NVFP4 fake-quant."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    *p, k = xf.shape
    amax = xf.abs().amax(-1, keepdim=True)                              # (...,1) per row
    gs = (amax / (_E4M3_MAX * qu.FP4_E2M1_MAX)).clamp_min(2.0 ** -40)   # FP32 global per row
    blk = xf.reshape(*p, k // block, block)
    e4 = torch.tensor(qu._FP8_E4M3_POS, device=xf.device, dtype=torch.float32)
    bmax = blk.abs().amax(-1, keepdim=True)                            # (...,nb,1)
    micro_ideal = (bmax / qu.FP4_E2M1_MAX / gs.unsqueeze(-2)).clamp_min(2.0 ** -9)
    micro = qu._ceil_to_value(micro_ideal, e4)                         # E4M3 grid
    eff = micro * gs.unsqueeze(-2)                                     # (...,nb,1) effective scale
    fp4 = torch.tensor(qu.FP4_E2M1_POS, device=xf.device, dtype=torch.float32)
    q = qu._round_to_nearest_value((blk / eff).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX), fp4)
    return (q * eff).reshape(*p, k).to(orig_dtype)


def _next_lower_grid(sc, grid):
    g = grid.view(*([1] * sc.dim()), -1)
    masked = torch.where(g < sc.unsqueeze(-1), g, torch.full_like(g, -1.0))
    best = masked.max(dim=-1).values
    return torch.where(best < 0, sc, best)


def _nvfp4_mse(x, block=16, steps=3):
    """Per-block NVFP4 with MSE-optimal block scale on the E4M3 grid (search base
    + `steps` lower grid values, pick min per-block MSE). Activation analogue of
    round-1's A2 weight scale search."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    *p, k = xf.shape
    blk = xf.reshape(*p, k // block, block)
    e4 = torch.tensor(qu._FP8_E4M3_POS, device=xf.device, dtype=torch.float32)
    fp4 = torch.tensor(qu.FP4_E2M1_POS, device=xf.device, dtype=torch.float32)
    ideal = (blk.abs().amax(-1, keepdim=True) / qu.FP4_E2M1_MAX).clamp_min(2.0 ** -9)
    base = qu._ceil_to_value(ideal, e4)
    cands = [base]
    cur = base
    for _ in range(steps):
        cur = _next_lower_grid(cur, e4)
        cands.append(cur)
    best_err = best_sc = None
    for sc in cands:
        q = qu._round_to_nearest_value((blk / sc).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX), fp4)
        err = ((q * sc - blk) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_sc = err, sc
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_sc = torch.where(take, sc, best_sc)
    q = qu._round_to_nearest_value((blk / best_sc).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX), fp4)
    return (q * best_sc).reshape(*p, k).to(orig_dtype)


def _nvfp4_outlier(x, block=16, frac=0.1):
    """Single-level NVFP4 for most rows; a per-row FP32 global * E4M3 micro
    (two-level) ONLY for the top-`frac` rows by excess kurtosis."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    flat = xf.reshape(-1, xf.shape[-1])
    # excess kurtosis per row
    mu = flat.mean(-1, keepdim=True)
    d = flat - mu
    var = (d ** 2).mean(-1).clamp_min(1e-12)
    kurt = (d ** 4).mean(-1) / (var ** 2)
    nrow = flat.shape[0]
    n_out = max(1, int(frac * nrow))
    out_idx = torch.topk(kurt, n_out).indices
    out_mask = torch.zeros(nrow, dtype=torch.bool, device=flat.device)
    out_mask[out_idx] = True
    base = qu.fake_quantize_nvfp4(xf, block)
    two = _nvfp4_2level(xf, block)
    mask = out_mask.view(*xf.shape[:-1], 1)
    out = torch.where(mask, two, base)
    return out.to(orig_dtype)


def make_apply_fp_format(variant, args):
    orig = qu._apply_fp_format

    def patched(x, fmt):
        if fmt == "nvfp4":
            if variant == "2level":
                return _nvfp4_2level(x)
            if variant == "mse":
                return _nvfp4_mse(x, steps=args.mse_steps)
            if variant == "outlier":
                return _nvfp4_outlier(x, frac=args.outlier_frac)
        if fmt == "mxfp8" and variant == "2level_both":
            return _nvfp4_2level(x) if False else orig(x, fmt)
        return orig(x, fmt)

    return patched


# ---------------------------------------------------------------------------
# SmoothQuant: offline per-channel scale folded into weight (re-GPTQ'd) and
# divided out of the activation. Alpha-weighted migration strength.
# ---------------------------------------------------------------------------
def _patch_actquant_smooth():
    """Divide the quantizer input by wrapper.quantizer._smooth_scale (per-channel,
    in the quantizer's channel space) and DO NOT remultiply — the scale is carried
    by the pre-scaled weight. No-op when _smooth_scale is unset."""
    orig_forward = qu.ActQuantizer.forward

    def forward(self, x):
        s = getattr(self, "_smooth_scale", None)
        if s is not None and not (isinstance(self.bits, int) and self.bits == 16):
            x = x / s.to(x.device, dtype=x.dtype)
        return orig_forward(self, x)

    qu.ActQuantizer.forward = forward
    return orig_forward


def collect_act_absmax(model, config, targets):
    """One calibration forward (activation quant OFF) capturing per-channel absmax
    at each targeted inner-module input. The inner nn.Linear input equals the
    quantizer's channel space (post online-Hadamard / post-reorder), which is also
    the weight-column space for non-reordered Linears (main path + down; o_proj is
    excluded). Hooking the quantizer would NOT fire here because activation quant
    is disabled pre-GPTQ, so we hook `module` instead. Returns {qname: absmax(K,)}."""
    stats = {}
    hooks = []
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, w in qlayers.items():
        if name.split(".")[-1] not in targets:
            continue

        def mk(nm):
            def pre_hook(module, inp):
                x = inp[0]
                a = x.detach().reshape(-1, x.shape[-1]).abs().amax(0).float()
                if nm in stats:
                    stats[nm] = torch.maximum(stats[nm], a)
                else:
                    stats[nm] = a
            return pre_hook
        hooks.append(w.module.register_forward_pre_hook(mk(name)))

    # calibration data (reuse wikitext train split, a few seqs)
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    from promix.eval.data import get_wikitext2 as _gw
    trainloader = _gw(seed=config["calibration"].get("seed", 0),
                      seqlen=config["eval"]["max_length"], tokenizer=tok, eval_mode=False,
                      nsamples=config["calibration"].get("nsamples", 128))
    model.eval()
    nb = min(16, len(trainloader))
    with torch.no_grad():
        for i in range(nb):
            batch = trainloader[i][0].to(DEV) if isinstance(trainloader[i], (list, tuple)) else trainloader[i].to(DEV)
            model(batch)
    for h in hooks:
        h.remove()
    return stats


def apply_smooth(model, absmax, alpha):
    """Fold s_k = clamp(actmax_k^alpha / wmax_k^(1-alpha)) into weight columns and
    stash the per-channel divisor on the quantizer. s is normalised so its geomean
    is ~1 (keeps magnitudes sane)."""
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    n = 0
    for name, w in qlayers.items():
        if name not in absmax:
            continue
        W = w.module.weight.data                      # (N, K)
        amax = absmax[name].to(W.device).clamp_min(1e-8)   # (K,)
        wmax = W.abs().amax(0).clamp_min(1e-8)         # (K,)
        s = (amax ** alpha) / (wmax ** (1.0 - alpha))
        s = s.clamp_min(1e-4)
        s = s / s.log().mean().exp()                   # geomean-normalise to ~1
        w.module.weight.data = (W * s.view(1, -1)).to(W.dtype)
        w.quantizer._smooth_scale = s.to(W.dtype)
        n += 1
    print(f"smooth: folded per-channel scale into {n} Linears (alpha={alpha})", flush=True)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_model(config, args):
    rw = args.r
    config["quantize"]["high_fraction"] = rw

    model = load_model(config["model"]["name"], dtype=torch.float16)
    fuse_layer_norms(model)
    fuse_basis_to_model(model, basis_path=config["paths"]["basis"],
                        rotation_path=config["paths"]["rotation"],
                        high_fraction=rw, low_fraction=0.0)
    rearrange_columns(model, high_fraction=rw, low_fraction=0.0,
                      o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"))
    cleanup_memory()
    add_actquant(model)
    ptqmod.setup_down_proj_hadamard(model)
    install_column_order_hooks(model)

    if args.variant == "smooth":
        model.to(DEV)
        absmax = collect_act_absmax(model, config, SMOOTH_TARGETS)
        apply_smooth(model, absmax, args.alpha)
        cleanup_memory()

    # optional round-1 weight-A2 MSE scale search (stacks with activation variants)
    wmse_restore = None
    if args.weight_mse:
        wmse_restore = (qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8)
        qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8 = a1mod.make_scale_search("both")
        print("[weight_mse] patched weight select_block_scales to MSE search (both)", flush=True)
    try:
        ptqmod.run_gptq_if_enabled(model, config, DEV)
    finally:
        if wmse_restore is not None:
            qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8 = wmse_restore
    ptqmod.configure_quantizers(model, config)

    if abs(args.ra - rw) > 1e-9:
        a1mod.override_act_high(model, args.ra, model.config.hidden_size,
                                getattr(model.config, "intermediate_size", None))

    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), required=True)
    ap.add_argument("--variant", choices=["baseline", "2level", "mse", "smooth", "outlier"], required=True)
    ap.add_argument("--r", type=float, default=0.125)
    ap.add_argument("--ra", type=float, default=None, help="activation high fraction (default=r)")
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant migration strength")
    ap.add_argument("--mse_steps", type=int, default=3)
    ap.add_argument("--outlier_frac", type=float, default=0.1)
    ap.add_argument("--weight_mse", action="store_true",
                    help="also apply round-1 weight-A2 MSE block-scale search during GPTQ")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/luofan/round2_results")
    args = ap.parse_args()
    args.ra = args.r if args.ra is None else args.ra

    os.makedirs(args.out, exist_ok=True)
    si = SIZES[args.size]
    config = ptqmod.load_config(os.path.join(REPO, si["config"]))
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["rotation"] = os.path.join(rotdir, f"R-high-{args.r}-low-0.0-sparse-0.0-{si['short']}.bin")
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{si['short']}.bin")
    ensure_rotation_file(config, si["short"])

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    # activation-only runtime patches
    if args.variant in ("2level", "mse", "outlier"):
        qu._apply_fp_format = make_apply_fp_format(args.variant, args)
        print(f"[{args.variant}] patched activation _apply_fp_format", flush=True)
    smooth_restore = None
    if args.variant == "smooth":
        smooth_restore = _patch_actquant_smooth()

    meta = dict(size=args.size, variant=args.variant, r=args.r, ra=args.ra,
                bits_per_weight=round(bits_per_weight(args.r), 4),
                alpha=args.alpha, mse_steps=args.mse_steps, outlier_frac=args.outlier_frac,
                weight_mse=args.weight_mse, tag=args.tag)
    print(f"START {json.dumps(meta)}", flush=True)

    model = build_model(config, args)
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    testloader = get_wikitext2(seed=config["calibration"].get("seed", 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = 1 if args.size == "8b" else config["eval"]["batch_size"]
    t0 = time.time()
    ppl = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    meta.update(ppl=float(ppl), eval_seconds=round(time.time() - t0, 1))
    print(f"ABLATION_RESULT {json.dumps(meta)}", flush=True)
    path = os.path.join(args.out, f"{args.size}_act_{args.variant}_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"WROTE {path}", flush=True)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
