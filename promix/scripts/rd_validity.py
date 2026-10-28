"""Task 3 — THE VALIDITY GATE for rate-distortion mixed-precision allocation.

Compares three per-Linear FP8/FP4 allocations at a MATCHED average FP8
ring-fraction, in the cheapest regime (R1-off = pure PCA, identity main
rotation), and reports wikitext-2 PPL for each:

  (A) uniform       : every Linear K_high = 1/8 * d          (current shipped)
  (B) variance-proxy: adaptive.derive_khigh on per-ring activation variance,
                      tau tuned so mean FP8 == target                (energy proxy)
  (C) measured-MSE  : adaptive.allocate_by_budget on per-ring MEASURED ΔD
                      (D_fp4 - D_fp8) of the R1-off-rotated activations

GATE (spec §9, plan Task 3): PASS iff PPL(C) < PPL(B) AND PPL(C) <= PPL(A).
FAIL iff C ≈ B ≈ A (measured-MSE allocation buys nothing over the proxy).

This is a MEASUREMENT experiment (no pytest asserts) — the verdict is the PPL
comparison. It reuses the proven R1-off build (round11_sensitivity) and the
reorder + weight-permute + GPTQ-edriven machinery (ablation_acc_per_bit) so no
new quantization logic is introduced; the only thing that varies across A/B/C
is WHICH 64-rings each Linear routes to FP8.

o_proj is held at uniform 1/8 in ALL THREE configs (controlled constant): its
per-head PCA rotation uses the reorder path already, and re-selecting its rings
would entangle the comparison; keeping it fixed isolates the signal on the
q/k/v/gate/up/down_proj units that are reallocated.
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

import ablation_acc_per_bit as abl  # proven reorder + GPTQ-edriven helpers
from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2
from promix.quantize import quant_utils as qu
from promix.quantize import gptq as gptqmod
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.quantize.kv_quant import setup_k_quant
from promix.quantize.adaptive import (
    ring_mse, ring_deltaD, allocate_by_budget, masks_to_khigh, derive_khigh,
)

RING = 64
# Units that get REALLOCATED across A/B/C. o_proj is deliberately excluded and
# held at uniform 1/8 (see module docstring).
ALLOC = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")


def _is_alloc_target(name):
    short = name.split(".")[-1]
    return ".layers." in name and short in ALLOC


# ---------------------------------------------------------------------------
# Model build (mirrors round11_sensitivity.build_model up to GPTQ, R1-off).
# ---------------------------------------------------------------------------
def build_base(config, r1_off, ring_r1=False):
    model = load_model(config["model"]["name"], dtype=torch.float16)
    fuse_layer_norms(model)
    fuse_basis_to_model(model, basis_path=config["paths"]["basis"],
                        rotation_path=config["paths"]["rotation"],
                        high_fraction=0.125, low_fraction=0.0,
                        identity_main_rotation=r1_off, ring_r1=ring_r1)
    rearrange_columns(model, high_fraction=0.125, low_fraction=0.0,
                      o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"))
    cleanup_memory()
    add_actquant(model)
    ptqmod.setup_down_proj_hadamard(model)
    install_column_order_hooks(model)
    return model


# ---------------------------------------------------------------------------
# Collect R1-off-rotated activation statistics (per-ring ΔD + per-ring variance)
# for the reallocated Linears, via forward-pre hooks on the inner nn.Linear
# (= post-online-Hadamard for down_proj, pre-quant for all). No GPTQ: fp16
# weights give the clean rotated calibration activation the allocation reads.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_stats(model, calib, max_tokens=4096):
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    acc = {}
    handles = []

    def make_hook(name):
        def hook(_module, inp):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()
            st = acc.get(name)
            d = x.shape[1]
            if st is None:
                nr = d // RING
                st = dict(n=0, e4=torch.zeros(nr), e8=torch.zeros(nr),
                          csum=torch.zeros(d), csq=torch.zeros(d), d=d, nr=nr)
                acc[name] = st
            if st["n"] >= max_tokens:
                return
            take = min(x.shape[0], max_tokens - st["n"])
            x = x[:take]
            q4 = qu.fake_quantize_nvfp4(x)
            q8 = qu.fake_quantize_mxfp8(x)
            e4 = (x - q4).pow(2).reshape(take, st["nr"], RING).sum(dim=(0, 2))
            e8 = (x - q8).pow(2).reshape(take, st["nr"], RING).sum(dim=(0, 2))
            st["e4"] += e4.cpu()
            st["e8"] += e8.cpu()
            st["csum"] += x.sum(0).cpu()
            st["csq"] += x.pow(2).sum(0).cpu()
            st["n"] += take
        return hook

    for name, w in qlayers.items():
        if _is_alloc_target(name):
            handles.append(w.module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    for batch in calib:
        ids = batch[0].to(DEV)
        model(ids)
        if all(acc.get(n, {"n": 0})["n"] >= max_tokens
               for n, w in qlayers.items() if _is_alloc_target(n)):
            break
    for h in handles:
        h.remove()

    stats = {}
    for name, st in acc.items():
        n = max(st["n"], 1)
        mse4 = st["e4"] / (n * RING)          # per-ring MSE, NVFP4
        mse8 = st["e8"] / (n * RING)          # per-ring MSE, MXFP8
        deltaD = (mse4 - mse8)                # upgrade benefit per ring
        mean = st["csum"] / n
        var = (st["csq"] / n - mean.pow(2)).clamp_min(0)   # per-channel variance
        ring_var = var.reshape(st["nr"], RING).mean(dim=1)  # per-ring variance
        stats[name] = dict(deltaD=deltaD, ring_var=ring_var, nr=st["nr"],
                           d=st["d"], ntok=st["n"])
    return stats


# ---------------------------------------------------------------------------
# Allocations (all at matched total FP8-ring budget over the ALLOC units).
# ---------------------------------------------------------------------------
def uniform_rings(stats):
    """K_high = round(1/8 * d) in rings, per reallocated Linear."""
    return {n: int(round(0.125 * stats[n]["d"] / RING)) for n in stats}


def variance_alloc(stats, budget_rings):
    """Per-layer energy-threshold proxy: derive_khigh on each Linear's
    ascending-sorted per-ring variance, single global tau tuned to the budget.
    Returns {name: set(fp8 ring indices)} (highest-variance rings)."""
    names = list(stats)

    def count_for(name, tau):
        v = stats[name]["ring_var"]
        v_asc, _ = torch.sort(v)                 # ascending -> tail = high var
        return derive_khigh(v_asc, tau, align=1, hidden_dim=stats[name]["nr"])

    def total(tau):
        return sum(count_for(n, tau) for n in names)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total(mid) < budget_rings:
            lo = mid
        else:
            hi = mid
    tau = hi
    chosen = {}
    for n in names:
        c = count_for(n, tau)
        topk = torch.topk(stats[n]["ring_var"], c).indices.tolist() if c > 0 else []
        chosen[n] = set(topk)
    return chosen, tau


def measured_alloc(stats, budget_rings, total_rings):
    """Global ΔD single-threshold allocation via allocate_by_budget."""
    names = list(stats)
    dd = [stats[n]["deltaD"] for n in names]
    target = budget_rings / total_rings
    masks = allocate_by_budget(dd, target_mean_fp8=target)
    chosen = {}
    for n, m in zip(names, masks):
        chosen[n] = set(torch.nonzero(m).flatten().tolist())
    return chosen


# ---------------------------------------------------------------------------
# Apply an allocation to a freshly-built base model (reorder + weight-permute +
# GPTQ-edriven tag + activation override), run GPTQ, configure, KV, return model.
# ---------------------------------------------------------------------------
def build_eval(config, r1_off, chosen, ring_r1=False):
    """chosen = None -> uniform shipped path. Else {name: set(fp8 ring idx)}."""
    model = build_base(config, r1_off, ring_r1=ring_r1)

    override = {}
    if chosen is not None:
        for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
            if not _is_alloc_target(name):
                continue
            S = chosen.get(name, set())
            nr = w.module.weight.shape[1] // RING
            non_s = [r for r in range(nr) if r not in S]
            s_lst = [r for r in range(nr) if r in S]
            ring_order = non_s + s_lst
            order = torch.cat([torch.arange(r * RING, (r + 1) * RING)
                               for r in ring_order]).to(w.module.weight.device)
            W = w.module.weight.data
            w.module.weight.data = W[:, order].contiguous()
            w._column_order = order
            w.reorder_before_quant = True
            kh = len(s_lst) * RING
            w.module._edriven_khigh = int(kh)
            override[name] = int(kh)
        model._rd_khigh_override = override

    gptq_restore = abl.patch_gptq_edriven() if chosen is not None else None
    try:
        ptqmod.run_gptq_if_enabled(model, config, DEV)
    finally:
        if gptq_restore is not None:
            gptqmod.GPTQ.__init__ = gptq_restore

    ptqmod.configure_quantizers(model, config)
    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    return model


def avg_fp8_fraction(model):
    """Realized mean FP8 fraction = total FP8 activation channels / total
    channels over ALL 7 target Linears (reads the configured quantizers)."""
    tot_hi = tot = 0
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        short = name.split(".")[-1]
        if ".layers." not in name or short not in (ALLOC + ("o_proj",)):
            continue
        q = w.quantizer
        if not isinstance(getattr(q, "bits", None), str):
            continue
        K = w.module.weight.shape[1]
        tot_hi += int(q.high_bits_length)
        tot += K
    return tot_hi / max(tot, 1), tot_hi, tot


def eval_ppl(model, config):
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    testloader = get_wikitext2(seed=config["calibration"].get("seed", 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = config["eval"]["batch_size"]
    return float(evaluate_ppl(model, testloader, DEV, batch_size=bs))


def bf16_bridge(config):
    """Pure bf16 baseline (no rotation, no quant) — must reproduce 3B=7.82."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], torch_dtype=torch.bfloat16).to(DEV)
    return eval_ppl(model, config)


def resolve_paths(config):
    """Force the EXISTING, proven R1-off artifacts (per-head PCA basis) and the
    local model, regardless of which 3B config file was passed. The mxfp8-nvfp4
    config points at a basis that was never generated; the fp-perhead
    `-Llama-3.2-3B` basis/rotation + local model are what exist on disk and are
    what round11_sensitivity's R1-off path uses."""
    name = config["model"]["name"]
    short = "Llama-3.2-3B" if "3B" in name or "3b" in name else os.path.basename(name)
    local = "/home/luofan/models/Llama-3.2-3B"
    if "3B" in name and os.path.isdir(local):
        config["model"]["name"] = local
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{short}.bin")
    config["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{short}.bin")
    config["quantize"]["o_proj_pca"] = "per_head"   # matches the existing basis
    config["quantize"]["high_fraction"] = 0.125
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r1_off", action="store_true")
    ap.add_argument("--target_mean_fp8", type=float, default=0.125)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--out", default="/home/luofan/RD_VALIDITY.json")
    args = ap.parse_args()

    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = resolve_paths(config)
    for p in (config["paths"]["basis"], config["paths"]["rotation"]):
        assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    print(f"START rd_validity r1_off={args.r1_off} target={args.target_mean_fp8} "
          f"model={config['model']['name']} basis={os.path.basename(config['paths']['basis'])}",
          flush=True)

    results = {"meta": dict(r1_off=args.r1_off, target=args.target_mean_fp8,
                            model=config["model"]["name"],
                            basis=os.path.basename(config["paths"]["basis"]),
                            act_mse=qu.ACT_MSE_SCALE_SEARCH,
                            weight_mse=qu.WEIGHT_MSE_SCALE_SEARCH,
                            weight_two_level=qu.WEIGHT_TWO_LEVEL)}

    # --- bf16 bridge (sanity: must reproduce ~7.82) --------------------------
    t0 = time.time()
    ppl_bf16 = bf16_bridge(config)
    print(f"BF16_BRIDGE ppl={ppl_bf16:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["bf16_bridge"] = ppl_bf16
    cleanup_memory()

    # --- collect R1-off rotated activation stats -----------------------------
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    cmodel = build_base(config, args.r1_off)
    cmodel.to(DEV)
    stats = collect_stats(cmodel, calib, max_tokens=args.collect_tokens)
    del cmodel
    cleanup_memory()

    total_rings = sum(stats[n]["nr"] for n in stats)
    uni = uniform_rings(stats)
    budget_rings = sum(uni.values())            # exact matched budget
    print(f"ALLOC units={len(stats)} total_rings={total_rings} "
          f"budget_fp8_rings={budget_rings} "
          f"(target_frac={budget_rings/total_rings:.4f})", flush=True)

    chosen_var, tau = variance_alloc(stats, budget_rings)
    chosen_mse = measured_alloc(stats, budget_rings, total_rings)
    n_var = sum(len(v) for v in chosen_var.values())
    n_mse = sum(len(v) for v in chosen_mse.values())
    print(f"VARIANCE tau={tau:.4f} fp8_rings={n_var}", flush=True)
    print(f"MEASURED fp8_rings={n_mse}", flush=True)
    # overlap of the two allocations (how much they actually differ)
    overlap = sum(len(chosen_var[n] & chosen_mse[n]) for n in stats)
    print(f"VAR∩MSE ring overlap={overlap}/{budget_rings} "
          f"({overlap/max(budget_rings,1)*100:.1f}%)", flush=True)
    results["alloc"] = dict(total_rings=total_rings, budget_rings=budget_rings,
                            tau=tau, n_var=n_var, n_mse=n_mse, overlap=overlap)

    configs = [("A_uniform", None), ("B_variance", chosen_var),
               ("C_measured", chosen_mse)]
    for tag, chosen in configs:
        t0 = time.time()
        model = build_eval(config, args.r1_off, chosen)
        frac, hi, tot = avg_fp8_fraction(model)
        ppl = eval_ppl(model, config)
        dt = time.time() - t0
        print(f"RESULT {tag} ppl={ppl:.4f} avg_fp8={frac:.4f} "
              f"({hi}/{tot} ch) ({dt:.0f}s)", flush=True)
        results[tag] = dict(ppl=ppl, avg_fp8=frac, hi_ch=hi, tot_ch=tot)
        del model
        cleanup_memory()

    a = results["A_uniform"]["ppl"]
    b = results["B_variance"]["ppl"]
    c = results["C_measured"]["ppl"]
    gate_pass = (c < b) and (c <= a + 1e-6)
    results["gate"] = dict(A=a, B=b, C=c, bf16=ppl_bf16,
                           verdict="PASS" if gate_pass else "FAIL")
    print("=" * 64, flush=True)
    print(f"GATE  bf16={ppl_bf16:.3f} | A_uniform={a:.4f}  "
          f"B_variance={b:.4f}  C_measured={c:.4f}", flush=True)
    print(f"GATE  C<B: {c < b}  C<=A: {c <= a + 1e-6}  -> "
          f"{'PASS' if gate_pass else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {args.out}", flush=True)
    print("RD_VALIDITY_DONE", flush=True)


if __name__ == "__main__":
    main()
