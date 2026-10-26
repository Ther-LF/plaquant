"""Round-1 ALGORITHM-axis ablation harness: accuracy-per-bit for ProMix FP
(NVFP4 main + MXFP8 high) under the FROZEN-GPTQ fake-quant protocol.

RESEARCH-ONLY accuracy harness. Reuses the exact ProMix PTQ build
(`prepare_ptq_model`: fuse-norm -> PCA rotation -> rearrange -> actquant ->
online-Hadamard(down) -> GPTQ(frozen FP block scales + Hessian error feedback)
-> configure activation quantizers -> KV) and the SAME wikitext-2 evaluator the
pipeline PPL uses (`evaluate_ppl`). STOCK fake-quant == the shipping recipe's
accuracy (3B 8.647, 8B 6.821 per docs/multisize_report.md). No kernel build.

bits/weight = WEIGHT storage only = (1-r_w)*4.5 + r_w*8.25.  Activation precision
(r_a) is FREE on the bits/weight axis (activations are not stored); it trades
GEMM compute instead.  This is why decoupling r_w (weight) from r_a (activation)
can move the accuracy-per-weight-bit Pareto front.

Modes
-----
rsweep   : fixed r (r_w=r_a=r), A4 fixed-r curve. r=0 == pure NVFP4 head-to-head.
decouple : independent --rw / --ra (weight vs activation high fraction).
           bits/weight depends ONLY on rw; ra is free on the memory axis.
a2       : weight block-scale MSE search on the FP segments (deployable; scale
           stays on the E8M0/E4M3 grid). --scale_search high|main|both.
edriven  : ERROR-DRIVEN variable K_high PER LAYER (shared q/k/v/gate/up) under a
           global budget = n_layers*round32(r*hidden). Selection score per
           channel = act_var(c) * sum_L (err_nvfp4(c) - err_mxfp8(c)).  Greedy
           32-chunk knapsack allocates the budget across layers (variable r/layer)
           at EQUAL total weight bits vs fixed r.
a1       : fixed-count high SELECTION criterion (actvar|wenergy|combined).
a3       : rotation on|off (identity main rotation).

--weights_only : set a_bits/k_bits/v_bits=16 (activations FP16).  Isolates WEIGHT
                 quant error (answers "why worse than GPTQ W4A16?").

Each run prints `ABLATION_RESULT <json>` and writes a JSON sidecar.
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
from promix.quantize import gptq as gptqmod
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.quantize.kv_quant import setup_k_quant

SIZES = {
    "1b": dict(config="promix/configs/llama-3.2-1b-fp-perhead.yaml", short="Llama-3.2-1B-Instruct"),
    "3b": dict(config="promix/configs/llama-3.2-3b-fp-perhead.yaml", short="Llama-3.2-3B"),
    "8b": dict(config="promix/configs/llama-3-8b-fp-perhead.yaml", short="llama-3-8b"),
}
NVFP4_SCALE_BITS = 8.0 / 16.0
MXFP8_SCALE_BITS = 8.0 / 32.0
MAIN_PATH = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def bits_per_weight(r_w):
    return (1.0 - r_w) * (4.0 + NVFP4_SCALE_BITS) + r_w * (8.0 + MXFP8_SCALE_BITS)


def _round32(x):
    return int(round(x / 32.0)) * 32


# ---------------------------------------------------------------------------
# A2: per-block MSE scale search (grid-constrained, deployable)
# ---------------------------------------------------------------------------
def _next_lower_grid(sc, grid):
    g = grid.view(*([1] * sc.dim()), -1)
    masked = torch.where(g < sc.unsqueeze(-1), g, torch.full_like(g, -1.0))
    best = masked.max(dim=-1).values
    return torch.where(best < 0, sc, best)


def _mse_best(Wblk, cands, max_fmt, elem_grid):
    best_err = best_sc = None
    for sc in cands:
        xs = (Wblk / sc).clamp(-max_fmt, max_fmt)
        deq = qu._round_to_nearest_value(xs, elem_grid) * sc
        err = ((deq - Wblk) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_sc = err, sc
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_sc = torch.where(take, sc, best_sc)
    return best_sc


def make_scale_search(which):
    orig_nv, orig_mx = qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8

    def nv_search(W, block_size=16):
        if which not in ("main", "both"):
            return orig_nv(W, block_size)
        Wf = W.to(torch.float32); *p, k = Wf.shape; nb = k // block_size
        Wblk = Wf.reshape(*p, nb, block_size)
        e4 = torch.tensor(qu._FP8_E4M3_POS, device=Wf.device, dtype=torch.float32)
        ideal = (Wblk.abs().amax(-1, keepdim=True) / qu.FP4_E2M1_MAX).clamp_min(2.0 ** -9)
        base = qu._ceil_to_value(ideal, e4); cands = [base]; cur = base
        for _ in range(3):
            cur = _next_lower_grid(cur, e4); cands.append(cur)
        fp4 = torch.tensor(qu.FP4_E2M1_POS, device=Wf.device, dtype=torch.float32)
        return _mse_best(Wblk, cands, qu.FP4_E2M1_MAX, fp4).squeeze(-1)

    def mx_search(W, block_size=32):
        if which not in ("high", "both"):
            return orig_mx(W, block_size)
        Wf = W.to(torch.float32); *p, k = Wf.shape; nb = k // block_size
        Wblk = Wf.reshape(*p, nb, block_size)
        ideal = (Wblk.abs().amax(-1, keepdim=True).clamp_min(0.0)) / qu.FP8_E4M3_MAX
        base = qu._e8m0_round_up(ideal.clamp_min(2.0 ** -127))
        cands = [base, base / 2.0, base / 4.0]
        e4 = torch.tensor(qu._FP8_E4M3_POS, device=Wf.device, dtype=torch.float32)
        return _mse_best(Wblk, cands, qu.FP8_E4M3_MAX, e4).squeeze(-1)

    return nv_search, mx_search


# ---------------------------------------------------------------------------
# Two-level NVFP4 scale (vLLM-style): per-row FP32 global * per-16 E4M3 micro.
# Our shipping fake_quantize_nvfp4 is SINGLE-level (per-16 E4M3 only); vLLM
# NVFP4 uses a calibrated per-tensor global * per-16 micro, which gives the
# E4M3 micro-scales far better relative precision. This is the suspected #1
# accuracy-per-bit lever (our pure-NVFP4 r=0 is much worse than vLLM NVFP4).
# Deployable: micro stays on the E4M3 grid, global is one FP32 scalar per row
# (exactly vLLM's nvfp4-pack-quantized layout).
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0


def _nvfp4_2level_effscale(x, block=16):
    """Return effective per-block scale (..., K//block) = global * micro, where
    global is per-row (last-dim) FP32 and micro is E4M3 on the grid."""
    xf = x.to(torch.float32)
    *p, k = xf.shape
    amax = xf.abs().amax(-1, keepdim=True)                       # (...,1) per row
    gs = (amax / (_E4M3_MAX * qu.FP4_E2M1_MAX)).clamp_min(2.0 ** -40)
    blk = xf.reshape(*p, k // block, block)
    bmax = blk.abs().amax(-1, keepdim=True)                      # (...,nb,1)
    e4 = torch.tensor(qu._FP8_E4M3_POS, device=xf.device, dtype=torch.float32)
    micro_ideal = (bmax / qu.FP4_E2M1_MAX / gs.unsqueeze(-2)).clamp_min(2.0 ** -9)
    micro = qu._ceil_to_value(micro_ideal, e4)                   # E4M3 grid
    eff = (micro * gs.unsqueeze(-2))                             # (...,nb,1)
    return eff.squeeze(-1)


def fake_quantize_nvfp4_2level(x, block_size=16):
    orig_dtype = x.dtype
    eff = _nvfp4_2level_effscale(x, block_size)                  # (...,nb)
    *p, k = x.shape
    xb = x.to(torch.float32).reshape(*p, k // block_size, block_size)
    sc = eff.unsqueeze(-1)
    fp4 = torch.tensor(qu.FP4_E2M1_POS, device=x.device, dtype=torch.float32)
    q = qu._round_to_nearest_value((xb / sc).clamp(-qu.FP4_E2M1_MAX, qu.FP4_E2M1_MAX), fp4)
    return (q * sc).reshape(*p, k).to(orig_dtype)


def select_block_scales_nvfp4_2level(W, block_size=16):
    return _nvfp4_2level_effscale(W, block_size)


# ---------------------------------------------------------------------------
# per-channel error scores (on rotated weight)
# ---------------------------------------------------------------------------
def _col_err(W, fq):
    """sum over rows of (W - fq(W))^2, per input channel. fq is a full-tensor
    fake quantizer (block-scaled)."""
    with torch.no_grad():
        Wq = fq(W.float())
        return ((W.float() - Wq) ** 2).sum(dim=0)   # (K,)


def _actvar(evalbundle, li, K, device):
    key = f"layer.{li}.attn_mlp_var"
    if evalbundle is not None and key in evalbundle:
        v = torch.as_tensor(evalbundle[key], dtype=torch.float32, device=device).flatten()
        if v.numel() == K:
            return v.clamp_min(0)
    return torch.ones(K, device=device)


# ---------------------------------------------------------------------------
# A1: fixed-count high selection for main-path Linears (per-Linear reorder)
# ---------------------------------------------------------------------------
def apply_a1_selection(model, config, criterion, evalbundle):
    r = config["quantize"]["high_fraction"]
    n = 0
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        if name.split(".")[-1] not in MAIN_PATH:
            continue
        W = w.module.weight.data; K = W.shape[1]; Kh = int(r * K)
        if Kh == 0:
            continue
        we = (W.float() ** 2).sum(0)
        if criterion == "wenergy":
            score = we
        elif criterion == "combined":
            li = ptqmod.layer_index_of(name)
            score = _actvar(evalbundle, li, K, W.device) * we
        else:
            raise ValueError(criterion)
        order = torch.argsort(score)
        w.module.weight.data = W[:, order].contiguous()
        w._column_order = order.to(W.device); w.reorder_before_quant = True
        n += 1
    print(f"a1[{criterion}]: reordered {n} main-path Linears", flush=True)


# ---------------------------------------------------------------------------
# edriven: per-layer variable K_high (shared q/k/v/gate/up), global budget
# ---------------------------------------------------------------------------
def plan_edriven(model, config, evalbundle, signal="weight"):
    """Return {layer_idx: k_i (mult 32)} and {layer_idx: order tensor} for the
    main path, allocating budget T=n_layers*round32(r*hidden) by greedy 32-chunk
    knapsack on a per-channel gain.

    signal:
      weight   : gain = act_var(c) * sum_L (err_nvfp4(c)-err_mxfp8(c))  [weight err]
      act      : gain = act_var(c)                 [activation-variance driven]
      combined : act_var(c) * (1 + sum_L weighterr) [both]
    """
    hidden = model.config.hidden_size
    n_layers = len(model.model.layers)
    r = config["quantize"]["high_fraction"]
    T_chunks = (n_layers * _round32(r * hidden)) // 32

    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    gains = {}      # li -> (hidden,) gain
    orders = {}     # li -> argsort ascending (tail=high)
    for li in range(n_layers):
        av = _actvar(evalbundle, li, hidden, DEV)
        if signal == "act":
            g = av.clone()
        else:
            werr = torch.zeros(hidden, device=DEV)
            for sub in MAIN_PATH:
                name = f"model.layers.{li}.self_attn.{sub}" if sub in ("q_proj","k_proj","v_proj") \
                       else f"model.layers.{li}.mlp.{sub}"
                if name not in qlayers:
                    continue
                W = qlayers[name].module.weight.data.to(DEV)
                werr = werr + (_col_err(W, qu.fake_quantize_nvfp4)
                               - _col_err(W, qu.fake_quantize_mxfp8)).clamp_min(0)
            g = av * werr if signal == "weight" else av * (1.0 + werr)
        gains[li] = g
        orders[li] = torch.argsort(gains[li])     # ascending; tail = highest gain
    # chunk marginals: for layer li, chunk j (0-indexed) gain = sum of gains of
    # the (j+1)-th highest-gain block of 32 channels.
    marg = []  # (marginal_gain, li, j)
    for li in range(n_layers):
        gsorted = torch.sort(gains[li], descending=True).values
        nch = hidden // 32
        for j in range(nch):
            marg.append((float(gsorted[j*32:(j+1)*32].sum()), li, j))
    marg.sort(key=lambda t: -t[0])
    take = {li: 0 for li in range(n_layers)}
    for _, li, j in marg[:T_chunks]:
        take[li] += 1
    khigh = {li: take[li] * 32 for li in range(n_layers)}
    print(f"edriven budget T={T_chunks*32} ch over {n_layers} layers; "
          f"k_i range [{min(khigh.values())},{max(khigh.values())}]", flush=True)
    return khigh, orders


def apply_edriven(model, khigh, orders):
    """Reorder main-path weights (shared per-layer order), tag inner Linears with
    _edriven_khigh for the GPTQ patch, set activation reorder."""
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, w in qlayers.items():
        sub = name.split(".")[-1]
        if sub not in MAIN_PATH:
            continue
        li = ptqmod.layer_index_of(name)
        order = orders[li].to(w.module.weight.device)
        w.module.weight.data = w.module.weight.data[:, order].contiguous()
        w.module._edriven_khigh = int(khigh[li])
        w._column_order = order
        w.reorder_before_quant = True


def patch_gptq_edriven():
    orig = gptqmod.GPTQ.__init__
    def patched(self, layer, mixed_precision=False, high_bits_length=0, low_bits_length=0):
        k = getattr(layer, "_edriven_khigh", None)
        if k is not None:
            high_bits_length = int(k); mixed_precision = True
        orig(self, layer, mixed_precision, high_bits_length, low_bits_length)
    gptqmod.GPTQ.__init__ = patched
    return orig


# ---------------------------------------------------------------------------
# activation high-length override (for decouple / edriven / ra)
# ---------------------------------------------------------------------------
def override_act_high(model, ra, hidden, intermediate, khigh=None):
    """Set each target wrapper's ACTIVATION quantizer high_bits_length. If khigh
    (edriven per-layer map) given, main-path uses k_i; else int(ra*K) per K."""
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        sub = name.split(".")[-1]
        if sub not in TARGETS:
            continue
        q = w.quantizer
        if not isinstance(getattr(q, "bits", None), str):
            continue  # weights_only (bits=16) -> leave disabled
        if khigh is not None and sub in MAIN_PATH:
            q.high_bits_length = int(khigh[ptqmod.layer_index_of(name)])
            continue
        K = w.module.weight.shape[1]
        q.high_bits_length = _round32(ra * K)


# ---------------------------------------------------------------------------
# unified builder
# ---------------------------------------------------------------------------
def build_model(config, args):
    rw = args.rw
    config["quantize"]["high_fraction"] = rw
    if args.nvfp4_2level:
        qu.fake_quantize_nvfp4 = fake_quantize_nvfp4_2level
        qu.select_block_scales_nvfp4 = select_block_scales_nvfp4_2level
        print("[nvfp4_2level] patched NVFP4 to two-level (global*micro) scale", flush=True)
    if args.weights_only:
        config["quantize"]["a_bits"] = 16
        config["quantize"]["k_bits"] = 16
        config["quantize"]["v_bits"] = 16

    model = load_model(config["model"]["name"], dtype=torch.float16)
    fuse_layer_norms(model)
    identity_rot = (args.mode == "a3" and args.rotation == "off")
    fuse_basis_to_model(model, basis_path=config["paths"]["basis"],
                        rotation_path=config["paths"]["rotation"],
                        high_fraction=rw, low_fraction=0.0,
                        identity_main_rotation=identity_rot)
    rearrange_columns(model, high_fraction=rw, low_fraction=0.0,
                      o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"))
    cleanup_memory()
    add_actquant(model)
    ptqmod.setup_down_proj_hadamard(model)
    install_column_order_hooks(model)

    evalbundle = None
    if args.mode in ("a1", "edriven") or args.criterion == "combined":
        epath = config["paths"]["basis"].replace("U-", "E-")
        evalbundle = torch.load(epath, weights_only=False)

    khigh = None
    gptq_restore = None
    if args.mode == "a1":
        apply_a1_selection(model, config, args.criterion, evalbundle)
    elif args.mode == "edriven":
        khigh, orders = plan_edriven(model, config, evalbundle, signal=args.edriven_signal)
        apply_edriven(model, khigh, orders)
        gptq_restore = patch_gptq_edriven()

    # A2 scale search patch (active during GPTQ find_params)
    a2_restore = None
    if args.mode == "a2" and args.scale_search != "none":
        a2_restore = (qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8)
        qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8 = make_scale_search(args.scale_search)

    try:
        ptqmod.run_gptq_if_enabled(model, config, DEV)
    finally:
        if gptq_restore is not None:
            gptqmod.GPTQ.__init__ = gptq_restore
        if a2_restore is not None:
            qu.select_block_scales_nvfp4, qu.select_block_scales_mxfp8 = a2_restore

    ptqmod.configure_quantizers(model, config)

    # activation high-length override (decouple ra / edriven k_i). Skip when
    # weights_only (activations already disabled by a_bits=16).
    if not args.weights_only:
        hidden = model.config.hidden_size
        inter = getattr(model.config, "intermediate_size", None)
        if khigh is not None:
            override_act_high(model, rw, hidden, inter, khigh=khigh)
        elif abs(args.ra - rw) > 1e-9:
            override_act_high(model, args.ra, hidden, inter)

    setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                  rotation_path=config["paths"]["rotation"])
    return model


def ensure_rotation_file(config, size_short):
    rot = config["paths"]["rotation"]
    if os.path.exists(rot):
        return
    qcfg = config["quantize"]
    print(f"Generating rotation file {rot} ...", flush=True)
    basismod.compute_basis(
        model_name=config["model"]["name"], output_dir=os.path.dirname(rot),
        seqlen=config["calibration"]["seqlen"], nsamples=config["calibration"]["nsamples"],
        seed=config["calibration"].get("seed", 0), calib_dataset=config["calibration"]["dataset"],
        high_fraction=qcfg["high_fraction"], low_fraction=0.0, sparse_fraction=0.0,
        down_proj_blocksize=qcfg.get("down_proj_blocksize", 256),
        rotation_granularity=qcfg["rotation_granularity"],
        o_proj_pca=qcfg.get("o_proj_pca", "per_head"))
    assert os.path.exists(rot), f"compute_basis did not produce {rot}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), required=True)
    ap.add_argument("--mode", choices=["rsweep", "decouple", "a1", "a2", "a3", "edriven"], required=True)
    ap.add_argument("--r", type=float, default=0.125, help="fixed high_fraction (rsweep/a1/a2/a3/edriven budget)")
    ap.add_argument("--rw", type=float, default=None, help="weight high fraction (decouple)")
    ap.add_argument("--ra", type=float, default=None, help="activation high fraction (decouple)")
    ap.add_argument("--criterion", choices=["actvar", "wenergy", "combined"], default="actvar")
    ap.add_argument("--scale_search", choices=["none", "high", "main", "both"], default="none")
    ap.add_argument("--edriven_signal", choices=["weight", "act", "combined"], default="weight")
    ap.add_argument("--rotation", choices=["on", "off"], default="on")
    ap.add_argument("--weights_only", action="store_true")
    ap.add_argument("--nvfp4_2level", action="store_true",
                    help="use vLLM-style two-level NVFP4 scale (per-row FP32 global * per-16 E4M3 micro)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/luofan/ablation_results")
    ap.add_argument("--model_path", default=None,
                    help="local model dir override (e.g. when HF download is blocked)")
    args = ap.parse_args()

    # resolve rw/ra
    if args.mode == "decouple":
        args.rw = args.r if args.rw is None else args.rw
        args.ra = args.r if args.ra is None else args.ra
    else:
        args.rw = args.r
        args.ra = args.r

    os.makedirs(args.out, exist_ok=True)
    si = SIZES[args.size]
    config = ptqmod.load_config(os.path.join(REPO, si["config"]))
    if args.model_path:
        config["model"]["name"] = args.model_path
    rotdir = os.path.dirname(config["paths"]["rotation"])
    config["paths"]["rotation"] = os.path.join(rotdir, f"R-high-{args.rw}-low-0.0-sparse-0.0-{si['short']}.bin")
    config["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{si['short']}.bin")
    ensure_rotation_file(config, si["short"])

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    meta = dict(size=args.size, mode=args.mode, rw=args.rw, ra=args.ra,
                bits_per_weight=round(bits_per_weight(args.rw), 4),
                criterion=args.criterion, scale_search=args.scale_search,
                rotation=args.rotation, weights_only=args.weights_only,
                edriven_signal=args.edriven_signal, nvfp4_2level=args.nvfp4_2level, tag=args.tag)
    print(f"START {json.dumps(meta)}", flush=True)

    model = build_model(config, args)
    model.to(DEV); model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    testloader = get_wikitext2(seed=config["calibration"].get("seed", 0),
                               seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = 1 if args.size == "8b" else config["eval"]["batch_size"]
    t0 = time.time()
    ppl = evaluate_ppl(model, testloader, DEV, batch_size=bs)
    meta.update(ppl=float(ppl), eval_seconds=round(time.time() - t0, 1))
    print(f"ABLATION_RESULT {json.dumps(meta)}", flush=True)
    path = os.path.join(args.out, f"{args.size}_{args.mode}_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"WROTE {path}", flush=True)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
