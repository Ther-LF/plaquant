"""Round-11 ALGORITHM axis: MEASURED per-layer error sensitivity + global
bit-budget allocation for the mixed NVFP4(main)+MXFP8(high) partition.

Unlike round-1 `edriven` (heuristic weight-error / act-variance signals) this
measures the REAL ΔPPL of moving each layer's high-precision budget, under the
CURRENT SHIPPED config (act-MSE on, two-level weight scale, r=1/8 uniform;
3B anchor PPL = 8.296).

Modes
-----
anchor     : all layers r=`--base_r` (reproduce the shipped anchor; the gate).
leaveone   : all layers r=`--base_r` EXCEPT `--zero_layer N` -> r=`--zero_r`
             (default 0). ΔPPL vs anchor = sensitivity of that layer's budget.
addone     : all layers r=`--base_r` EXCEPT `--boost_layer N` -> r=`--boost_r`
             (default 0.25). marginal GAIN of adding precision there.
plan       : arbitrary per-layer r from `--plan_json` (a JSON dict
             {"<layer_idx>": r}, r in {0,0.0625,0.125,0.25}); unlisted layers
             use `--base_r`. The allocation-validation path.

The per-layer high length is applied to BOTH the GPTQ weight split (via the
`_edriven_khigh` tag + the ablation harness's GPTQ patch) AND the activation
quantizer high_bits_length, for ALL 7 target Linears in a layer — a clean
leave-one-out of the hardware-native [NVFP4 main | MXFP8 high] partition that
keeps the SHIPPED channel order / rotation / two-level scale unchanged.

bits/weight (incl FP8 scale factors) is computed EXACTLY from the realized
per-Linear K_low/K_high, weighted by element count.
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
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))

import transformers

import ablation_acc_per_bit as abl  # reuse helpers (two-level sharedg, GPTQ patch)
from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import quant_utils as qu
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.quantize.kv_quant import setup_k_quant

SIZES = abl.SIZES
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NVFP4_SCALE_BITS = 8.0 / 16.0
MXFP8_SCALE_BITS = 8.0 / 32.0


def _round32(x):
    return int(round(x / 32.0)) * 32


def khigh_for(r, in_features):
    """Block-aligned K_high for fraction r on a Linear of width in_features.
    K_high % 32 == 0 (MXFP8), and K_low = K - K_high % 16 == 0 (NVFP4)."""
    kh = _round32(r * in_features)
    assert kh % 32 == 0 and (in_features - kh) % 16 == 0, (
        f"r={r} in={in_features} -> K_high={kh} not block-aligned")
    return kh


def build_layer_r(model, base_r, plan):
    """Return {layer_idx: r} for all decoder layers."""
    n_layers = len(model.model.layers)
    lr = {i: base_r for i in range(n_layers)}
    for k, v in (plan or {}).items():
        lr[int(k)] = float(v)
    return lr


def apply_sens_khigh(model, layer_r):
    """Tag every target Linear's inner module with `_edriven_khigh` (read by the
    ablation GPTQ patch) and record per-Linear (K, K_low, K_high) for bits
    accounting. Returns the list of (name, K, K_low, K_high)."""
    accounting = []
    for name, mod in model.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        if ".layers." not in name:
            continue
        short = name.split(".")[-1]
        if short not in TARGETS:
            continue
        li = int(name.split(".layers.")[1].split(".")[0])
        K = mod.module.weight.shape[1]
        kh = khigh_for(layer_r[li], K)
        mod.module._edriven_khigh = int(kh)
        mod._sens_khigh = int(kh)  # stash for the activation override
        accounting.append((name, K, K - kh, kh))
    return accounting


def override_act_high_sens(model):
    """Set every target Linear's ACTIVATION quantizer high_bits_length to the
    value stashed in apply_sens_khigh (so acts match the weight split exactly)."""
    for name, mod in model.named_modules():
        if not isinstance(mod, ActQuantWrapper):
            continue
        kh = getattr(mod, "_sens_khigh", None)
        if kh is None:
            continue
        q = mod.quantizer
        if not isinstance(getattr(q, "bits", None), str):
            continue  # not an active FP quantizer (e.g. disabled)
        q.high_bits_length = int(kh)


def bits_per_weight_elems(model, accounting):
    """Exact element-count-weighted bits/weight incl FP8 scale factors."""
    # map name -> N (out_features)
    nmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, ActQuantWrapper) and name.split(".")[-1] in TARGETS \
                and ".layers." in name:
            nmap[name] = mod.module.weight.shape[0]
    tot_lo = tot_hi = 0
    for name, K, K_low, K_high in accounting:
        N = nmap[name]
        tot_lo += N * K_low
        tot_hi += N * K_high
    tot = tot_lo + tot_hi
    bits = (tot_lo * (4.0 + NVFP4_SCALE_BITS) + tot_hi * (8.0 + MXFP8_SCALE_BITS)) / tot
    return bits, tot_lo, tot_hi


def build_model(config, size, layer_r, r1_off=False):
    config["quantize"]["high_fraction"] = 0.125  # base; per-layer overrides win
    model = load_model(config["model"]["name"], dtype=torch.float16)
    fuse_layer_norms(model)
    fuse_basis_to_model(model, basis_path=config["paths"]["basis"],
                        rotation_path=config["paths"]["rotation"],
                        high_fraction=0.125, low_fraction=0.0,
                        identity_main_rotation=r1_off)
    rearrange_columns(model, high_fraction=0.125, low_fraction=0.0,
                      o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"))
    cleanup_memory()
    add_actquant(model)
    ptqmod.setup_down_proj_hadamard(model)
    install_column_order_hooks(model)

    accounting = apply_sens_khigh(model, layer_r)
    gptq_restore = abl.patch_gptq_edriven()

    # SHIPPED two-level weight scale (variant B: shared global across NVFP4 low +
    # MXFP8 high), nearest E4M3 micro + per-block micro-MSE.
    w2l_restore = (qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8)
    nvf, mxf = abl.make_weight_2level_sharedg("nearest", True, pow2=False)
    qu.select_block_scales_nvfp4 = nvf
    qu.select_block_scales_mxfp8 = mxf
    try:
        ptqmod.run_gptq_if_enabled(model, config, DEV)
    finally:
        abl.gptqmod.GPTQ.__init__ = gptq_restore
        qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8 = w2l_restore

    ptqmod.configure_quantizers(model, config)
    override_act_high_sens(model)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    return model, accounting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), default="3b")
    ap.add_argument("--mode", choices=["anchor", "leaveone", "addone", "plan"], required=True)
    ap.add_argument("--base_r", type=float, default=0.125)
    ap.add_argument("--zero_layer", type=int, default=None)
    ap.add_argument("--zero_r", type=float, default=0.0)
    ap.add_argument("--boost_layer", type=int, default=None)
    ap.add_argument("--boost_r", type=float, default=0.25)
    ap.add_argument("--plan_json", default=None)
    ap.add_argument("--r1_off", action="store_true",
                    help="identity main rotation R1 (makes intermediate-r valid: "
                         "channel order stays monotonic by variance from U).")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/luofan/round11_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    si = SIZES[args.size]
    config = ptqmod.load_config(os.path.join(REPO, si["config"]))
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{si['short']}.bin")
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{si['short']}.bin")

    plan = {}
    if args.mode == "leaveone":
        assert args.zero_layer is not None
        plan = {args.zero_layer: args.zero_r}
    elif args.mode == "addone":
        assert args.boost_layer is not None
        plan = {args.boost_layer: args.boost_r}
    elif args.mode == "plan":
        assert args.plan_json is not None
        with open(args.plan_json) as f:
            plan = json.load(f)

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    cfg_model = transformers.AutoConfig.from_pretrained(config["model"]["name"])
    n_layers = cfg_model.num_hidden_layers
    layer_r = {i: args.base_r for i in range(n_layers)}
    for k, v in plan.items():
        layer_r[int(k)] = float(v)

    meta = dict(size=args.size, mode=args.mode, base_r=args.base_r,
                plan={str(k): v for k, v in plan.items()}, r1_off=args.r1_off,
                tag=args.tag, act_mse=qu.ACT_MSE_SCALE_SEARCH,
                weight_mse=qu.WEIGHT_MSE_SCALE_SEARCH,
                weight_two_level=qu.WEIGHT_TWO_LEVEL)
    print(f"START {json.dumps(meta)}", flush=True)

    model, accounting = build_model(config, args.size, layer_r, r1_off=args.r1_off)
    bits, tot_lo, tot_hi = bits_per_weight_elems(model, accounting)
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    testloader = get_wikitext2(seed=config["calibration"].get("seed", 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = 1 if args.size == "8b" else config["eval"]["batch_size"]
    t0 = time.time()
    ppl = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    meta.update(ppl=float(ppl), bits_per_weight=round(bits, 4),
                layer_r={str(i): layer_r[i] for i in sorted(layer_r)},
                eval_seconds=round(time.time() - t0, 1))
    print(f"SENS_RESULT {json.dumps(meta)}", flush=True)
    path = os.path.join(args.out, f"{args.size}_{args.mode}_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"WROTE {path}", flush=True)
    print("SENS_DONE", flush=True)


if __name__ == "__main__":
    main()
