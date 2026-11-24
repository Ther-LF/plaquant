#!/usr/bin/env python
"""rd_m31.py — Motivation §3.1 channel-level precision-allocation experiments (M31).

Implements the four channel-selection strategies and both allocation scopes of
docs/paper_motivation_3_1_experiments.md on the real NVFP4(blk16)/MXFP8(blk32)
fake-quant + frozen-GPTQ pipeline:

  RND  random priority           — raw representation (load + fuse_layer_norms only)
  SS   mean squared activation   — raw representation
  MA   mean absolute activation  — raw representation
  AV   centered activation variance measured on the ResQ-style PCA representation
       (identity main rotation = pure U; down_proj uses the online-Hadamard
       representation because the intermediate dim has no PCA basis in ResQ)

Scope: within (per-layer quota h_l(r) = feasible value nearest r*d_l) or across
(same total H = sum_l h_l(r), prefix-increment 32-chunk quota-constrained
max-score selection, spec section 6).

Precision unit = CHANNEL. Selected channels are packed by ascending stable
channel id into the [main(NVFP4) | high(MXFP8)] layout via a column permutation
applied identically to activations (_column_order + reorder_before_quant) and
weight columns; block scales are then recomputed on the final layout by the
stock fake-quant / frozen-GPTQ machinery. o_proj is excluded from the eligible
set and held FP16 in every config (see docs/motivation_3_1/inventory.md §5).

Subcommands:
  collect   — per-channel fit statistics for one model x representation
  selfcheck — no-quant equivalence, permutation equivalence, quota feasibility,
              quantizer segment-routing unit check
  control   — float reference run (PPL + fp16 noise-floor layer errors)
  run       — one (model, strategy, scope, r, seed) point
  alloc     — dump the full allocation (stable channel ids) for one point
  aggregate — scan data/*.json -> runs.csv / layer_errors.csv / allocations.jsonl
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))

import transformers
import datasets as hf_datasets

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2, get_c4
from promix.quantize import quant_utils as qu
from promix.quantize import gptq as gptqmod
from promix.models.loader import load_model, install_column_order_hooks
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
from promix.quantize.quant_utils import (
    ActQuantWrapper, ActQuantizer, _quant_enabled, add_actquant, find_qlayers,
    fake_quantize_nvfp4, fake_quantize_mxfp8,
)

ELIGIBLE = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
OUT_DIR = os.path.join(REPO, "docs", "motivation_3_1")
DATA_DIR = os.path.join(OUT_DIR, "data")
SCORES_DIR = os.path.join(OUT_DIR, "scores")
NVFP4_BLK, MXFP8_BLK = 16, 32
FIT_NSAMPLES = 512          # the frozen seed0 fit window sequence
SCORE_TOKENS = 4096         # score-collection cap (= first 2 windows)
HELDOUT_SEED = 31001
HELDOUT_N = 32
EPS = 1e-12

MODELS = {
    "1b": dict(name="unsloth/Llama-3.2-1B-Instruct", short="Llama-3.2-1B-Instruct",
               cfg="promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml"),
    "3b": dict(name="/home/luofan/models/Llama-3.2-3B", short="Llama-3.2-3B",
               cfg="promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml"),
    "8b": dict(name="/home/luofan/models/llama-3-8b", short="llama-3-8b",
               cfg="promix/configs/llama-3-8b-mxfp8-nvfp4.yaml"),
}


# ---------------------------------------------------------------------------
# config / model builds
# ---------------------------------------------------------------------------
def resolve(mk):
    mi = MODELS[mk]
    cfg = ptqmod.load_config(os.path.join(REPO, mi["cfg"]))
    if os.path.isdir(mi["name"]):
        cfg["model"]["name"] = mi["name"]
    rotdir = os.path.join(REPO, "rotation")
    cfg["paths"]["basis"] = os.path.join(rotdir, f"U-wikitext-512-{mi['short']}.bin")
    cfg["paths"]["rotation"] = os.path.join(
        rotdir, f"R-high-0.125-low-0.0-sparse-0.0-{mi['short']}.bin")
    cfg["quantize"]["o_proj_pca"] = "per_head"
    cfg["quantize"]["high_fraction"] = 0.125   # fallback only; per-Linear override wins
    cfg["quantize"]["low_fraction"] = 0.0
    return cfg


def rep_for(strategy):
    return "pca" if strategy == "av" else "raw"


def build_float_model(mk, rep, cfg=None):
    """Representation-equivalent float model (quantizers at pass-through 16)."""
    cfg = cfg or resolve(mk)
    model = load_model(cfg["model"]["name"], dtype=torch.float16)
    fuse_layer_norms(model)
    if rep == "pca":
        for p in (cfg["paths"]["basis"], cfg["paths"]["rotation"]):
            assert os.path.exists(p), f"missing artifact: {p}"
        fuse_basis_to_model(model, basis_path=cfg["paths"]["basis"],
                            rotation_path=cfg["paths"]["rotation"],
                            high_fraction=0.125, low_fraction=0.0,
                            identity_main_rotation=True)
        rearrange_columns(model, high_fraction=0.125, low_fraction=0.0,
                          o_proj_pca="per_head")
    cleanup_memory()
    add_actquant(model)
    if rep == "pca":
        ptqmod.setup_down_proj_hadamard(model)
        install_column_order_hooks(model)
    model.config.use_cache = False
    model.seqlen = cfg["eval"]["max_length"]
    return model, cfg


def eligible_wrappers(model):
    out = {}
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        if ".layers." in name and name.split(".")[-1] in ELIGIBLE:
            out[name] = w
    return out


def get_tokenizer(cfg):
    return transformers.AutoTokenizer.from_pretrained(cfg["model"]["name"])


# ---------------------------------------------------------------------------
# score collection (fit statistics)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_scores(mk, rep, max_tokens=SCORE_TOKENS):
    os.makedirs(SCORES_DIR, exist_ok=True)
    path = os.path.join(SCORES_DIR, f"{mk}_{rep}.pt")
    if os.path.exists(path):
        print(f"[collect] exists: {path}")
        return path
    model, cfg = build_float_model(mk, rep)
    model.to(DEV).eval()
    tok = get_tokenizer(cfg)
    nwin = math.ceil(max_tokens / 2048)
    # first nwin windows of the frozen seed0 FIT_NSAMPLES sequence (same
    # random.seed(0) randint stream -> exact prefix subset of the fit set)
    calib = get_wikitext2(nsamples=nwin, seed=cfg["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok, eval_mode=False)
    wrappers = eligible_wrappers(model)
    stats, handles = {}, []

    def mk_hook(name):
        def hook(_m, inp):
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).to(torch.float64)
            st = stats.get(name)
            if st is None:
                d = x.shape[1]
                st = dict(n=0, d=d,
                          s1=torch.zeros(d, dtype=torch.float64),
                          s2=torch.zeros(d, dtype=torch.float64),
                          sa=torch.zeros(d, dtype=torch.float64))
                stats[name] = st
            take = min(x.shape[0], max_tokens - st["n"])
            if take <= 0:
                return
            x = x[:take]
            st["s1"] += x.sum(0).cpu()
            st["s2"] += x.pow(2).sum(0).cpu()
            st["sa"] += x.abs().sum(0).cpu()
            st["n"] += take
        return hook

    module_order = list(wrappers.keys())   # deterministic model-traversal order
    for name in module_order:
        handles.append(wrappers[name].module.register_forward_pre_hook(mk_hook(name)))
    for inp, _ in calib:
        model(inp.to(DEV))
    for h in handles:
        h.remove()
    assert all(stats[n]["n"] == max_tokens for n in module_order), \
        {n: stats[n]["n"] for n in module_order}
    meta = dict(model=mk, representation=rep, max_tokens=max_tokens,
                nwindows=nwin, fit_seed=cfg["calibration"].get("seed", 0),
                fit_nsamples=FIT_NSAMPLES, model_name=cfg["model"]["name"],
                basis=os.path.basename(cfg["paths"]["basis"]) if rep == "pca" else None,
                note="s1/s2/sa are float64 per-channel sums over the first "
                     "nwindows of the frozen seed0 fit window sequence; "
                     "scores: ss=s2/n, ma=sa/n, av=clamp(s2/n-(s1/n)^2,0)")
    torch.save(dict(meta=meta, module_order=module_order, stats=stats), path)
    print(f"[collect] wrote {path} ({len(module_order)} modules)")
    del model
    cleanup_memory()
    return path


def load_scores(mk, rep):
    path = os.path.join(SCORES_DIR, f"{mk}_{rep}.pt")
    if not os.path.exists(path):
        collect_scores(mk, rep)
    return torch.load(path, weights_only=False)


def module_scores(strategy, st, mk=None, name=None, seed=0):
    n = st["n"]
    if strategy == "ss":
        return st["s2"] / n
    if strategy == "ma":
        return st["sa"] / n
    if strategy == "av":
        mean = st["s1"] / n
        return (st["s2"] / n - mean * mean).clamp_min(0.0)
    if strategy == "rnd":
        d = st["d"]
        h = int(hashlib.md5(f"m31:{mk}:{name}:seed{seed}".encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(h)
        perm = torch.randperm(d, generator=g)
        # fixed per-channel priority: channel perm[k] has priority d-k
        s = torch.empty(d, dtype=torch.float64)
        s[perm] = torch.arange(d, 0, -1, dtype=torch.float64)
        return s
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# quotas and selection
# ---------------------------------------------------------------------------
def quota_within(d, r):
    """Feasible h (mult of 32, (d-h)%16==0 holds since d%32==0) nearest r*d;
    tie -> smaller."""
    assert d % 32 == 0, d
    q = r * d
    lo = int(math.floor(q / 32.0)) * 32
    cands = [c for c in (lo, lo + 32) if 0 <= c <= d]
    return min(cands, key=lambda c: (abs(c - q), c))


def select_top(s, h):
    """Top-h channels by score; ties broken by LOWER stable channel id
    (stable sort on -s). Returns ascending-sorted selected ids."""
    if h == 0:
        return torch.empty(0, dtype=torch.long)
    idx = torch.argsort(-s, stable=True)
    return torch.sort(idx[:h]).values


def compute_alloc(mk, strategy, scope, r, seed, scores_bundle):
    """Returns (alloc {name: dict(d,h,sel)}, H, total, r_actual)."""
    module_order = scores_bundle["module_order"]
    stats = scores_bundle["stats"]
    s_all = {n: module_scores(strategy, stats[n], mk=mk, name=n, seed=seed)
             for n in module_order}
    quotas = {n: quota_within(stats[n]["d"], r) for n in module_order}
    H = sum(quotas.values())
    total = sum(stats[n]["d"] for n in module_order)
    if scope == "across":
        # spec section 6: per-layer descending-sorted prefix 32-chunk
        # increments (non-increasing within a layer), global greedy top H/32
        # chunks; ties -> earlier prefix, then stable layer order.
        chunks = []
        sort_idx = {}
        for li, n in enumerate(module_order):
            s = s_all[n]
            idx = torch.argsort(-s, stable=True)
            sort_idx[n] = idx
            inc = s[idx].reshape(-1, 32).sum(1)
            for c in range(inc.numel()):
                chunks.append((-float(inc[c]), li, c))
        chunks.sort()
        assert H % 32 == 0
        cnt = {n: 0 for n in module_order}
        for _, li, c in chunks[: H // 32]:
            cnt[module_order[li]] += 1
        alloc = {}
        for n in module_order:
            h = cnt[n] * 32
            sel = torch.sort(sort_idx[n][:h]).values
            alloc[n] = dict(d=stats[n]["d"], h=h, sel=sel)
    else:
        alloc = {n: dict(d=stats[n]["d"], h=quotas[n],
                         sel=select_top(s_all[n], quotas[n]))
                 for n in module_order}
    Hs = sum(a["h"] for a in alloc.values())
    assert Hs == H, (Hs, H)
    r_actual = H / total
    # score summaries (spec section 6: per-layer score mean/max/sum + selected frac)
    for n in module_order:
        s = s_all[n]
        a = alloc[n]
        a["score_summary"] = dict(
            mean=float(s.mean()), max=float(s.max()), sum=float(s.sum()),
            min=float(s.min()), sel_frac=a["h"] / a["d"],
            sel_score_sum=float(s[a["sel"]].sum()) if a["h"] else 0.0)
    return alloc, H, total, r_actual


def alloc_hash(rep, alloc):
    hsh = hashlib.sha256()
    hsh.update(rep.encode())
    for n in sorted(alloc):
        a = alloc[n]
        hsh.update(n.encode())
        hsh.update(str(a["d"]).encode())
        hsh.update(str(a["h"]).encode())
        hsh.update(a["sel"].to(torch.int32).numpy().tobytes())
    return hsh.hexdigest()


def ids_sha(sel):
    return hashlib.sha256(sel.to(torch.int32).numpy().tobytes()).hexdigest()[:16]


def bytes_accounting(alloc):
    """Storage accounting over eligible Linears only (weights + block scales +
    permutation metadata). out_features inferred later; here per input channel
    counts are enough for ratios — we fill exact bytes in run() where the model
    is available."""
    return None  # computed in run() with real out_features


# ---------------------------------------------------------------------------
# quantized build
# ---------------------------------------------------------------------------
def apply_allocation(model, alloc):
    wrappers = find_qlayers(model, layers=[ActQuantWrapper])
    for name, w in wrappers.items():
        if ".layers." not in name:
            continue
        short = name.split(".")[-1]
        if short not in ELIGIBLE:
            w.module._m31_skip = True   # o_proj: weights stay FP16 (GPTQ skip)
            continue
        a = alloc[name]
        d, h, sel = a["d"], a["h"], a["sel"]
        W = w.module.weight.data
        assert W.shape[1] == d, (name, W.shape, d)
        mask = torch.zeros(d, dtype=torch.bool)
        mask[sel] = True
        order = torch.cat([torch.arange(d)[~mask], torch.arange(d)[mask]])
        if not torch.equal(order, torch.arange(d)):
            dev = W.device
            w.module.weight.data = W[:, order.to(dev)].contiguous()
            w._column_order = order.to(dev)
            w.reorder_before_quant = True
        w.module._edriven_khigh = int(h)


def patch_gptq():
    orig_init = gptqmod.GPTQ.__init__
    orig_fq = gptqmod.GPTQ.fasterquant

    def pinit(self, layer, mixed_precision=False, high_bits_length=0,
              low_bits_length=0):
        k = getattr(layer, "_edriven_khigh", None)
        if k is not None:
            high_bits_length = int(k)
            mixed_precision = True
        orig_init(self, layer, mixed_precision, high_bits_length, low_bits_length)

    def pfq(self, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False):
        if getattr(self.layer, "_m31_skip", False):
            self.H = None
            self.Q_int = None
            return
        return orig_fq(self, blocksize=blocksize, percdamp=percdamp,
                       groupsize=groupsize, actorder=actorder)

    gptqmod.GPTQ.__init__ = pinit
    gptqmod.GPTQ.fasterquant = pfq
    return orig_init, orig_fq


def unpatch_gptq(saved):
    gptqmod.GPTQ.__init__, gptqmod.GPTQ.fasterquant = saved


def configure_m31_quantizers(model, alloc):
    """Activation quantizers on eligible Linears only. Pure endpoints use the
    single-format branch (h=0 pure NVFP4; h=d pure MXFP8). Everything else
    (o_proj, lm_head, out_quantizers/KV) stays at pass-through 16."""
    for name, w in eligible_wrappers(model).items():
        a = alloc[name]
        d, h = a["d"], a["h"]
        if h == d:
            w.quantizer.configure(bits="mxfp8", groupsize=-1, sym=False,
                                  clip_ratio=1.0, high_bits_length=0,
                                  high_bits="mxfp8", low_bits_length=0,
                                  low_bits="nvfp4")
        else:
            w.quantizer.configure(bits="nvfp4", groupsize=-1, sym=False,
                                  clip_ratio=1.0, high_bits_length=h,
                                  high_bits="mxfp8", low_bits_length=0,
                                  low_bits="nvfp4")


def build_quant_model(mk, rep, alloc, cfg=None):
    model, cfg = build_float_model(mk, rep, cfg)
    apply_allocation(model, alloc)
    saved = patch_gptq()
    try:
        ptqmod.run_gptq_if_enabled(model, cfg, DEV)
    finally:
        unpatch_gptq(saved)
    configure_m31_quantizers(model, alloc)
    return model, cfg


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------
def tensor_sha(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def eval_wiki(model, cfg, tok, batch_size=None):
    testenc = get_wikitext2(seed=cfg["calibration"].get("seed", 0),
                            seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = batch_size or cfg["eval"]["batch_size"]
    ppl = float(evaluate_ppl(model, testenc, DEV, batch_size=bs))
    nwin = testenc.input_ids.numel() // model.seqlen
    ntok = nwin * (model.seqlen - 1)
    return dict(ppl=ppl, nll_sum=math.log(ppl) * ntok, target_tokens=ntok,
                nwindows=nwin, data_sha=tensor_sha(
                    testenc.input_ids[:, :nwin * model.seqlen]))


def eval_c4(model, cfg, tok, batch_size=None):
    c4 = get_c4(nsamples=256, seed=cfg["calibration"].get("seed", 0),
                seqlen=model.seqlen, tokenizer=tok, eval_mode=True)
    bs = batch_size or cfg["eval"]["batch_size"]
    ppl = float(evaluate_ppl(model, c4, DEV, batch_size=bs))
    nwin = c4.input_ids.numel() // model.seqlen
    ntok = nwin * (model.seqlen - 1)
    return dict(ppl=ppl, nll_sum=math.log(ppl) * ntok, target_tokens=ntok,
                nwindows=nwin, data_sha=tensor_sha(c4.input_ids))


# ---------------------------------------------------------------------------
# held-out diagnostic set + teacher-input layer errors (spec section 7.2)
# ---------------------------------------------------------------------------
def heldout_windows(cfg, tok, n=HELDOUT_N, seqlen=2048):
    traindata = hf_datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    trainenc = tok(text="\n\n".join(traindata["text"]), return_tensors="pt")
    L = trainenc.input_ids.shape[1]
    random.seed(cfg["calibration"].get("seed", 0))
    fit = []
    for _ in range(FIT_NSAMPLES):
        i = random.randint(0, L - seqlen - 1)
        fit.append((i, i + seqlen))
    random.seed(HELDOUT_SEED)
    wins, offs = [], []
    while len(wins) < n:
        i = random.randint(0, L - seqlen - 1)
        j = i + seqlen
        if any(i < fj and fi < j for fi, fj in fit):
            continue
        if any(i < wj and wi < j for wi, wj in offs):
            continue
        offs.append((i, j))
        wins.append(trainenc.input_ids[:, i:j])
    return wins, offs


@torch.no_grad()
def layer_errors(fmodel, qmodel, cfg, tok, fp16_control=False):
    """Stream the held-out windows through the FLOAT model; at each eligible
    inner Linear, teacher Y = X.float() @ Wf.float().T (+bias) and candidate
    Yhat = quantized path on the SAME teacher X (reorder -> act fake-quant ->
    quantized weights), matmul in fp32. Accumulate SSE/ENERGY in float64.

    fp16_control=True: candidate = the real fp16 GEMM on the float weights
    (numeric noise floor of the float path itself)."""
    wins, offs = heldout_windows(cfg, tok)
    fmodel.to(DEV).eval()
    fw = eligible_wrappers(fmodel)
    qw = eligible_wrappers(qmodel) if qmodel is not None else None
    acc = {n: dict(sse=0.0, energy=0.0, n_tokens=0, out=fw[n].module.weight.shape[0])
           for n in fw}
    handles = []

    def mk_hook(name):
        wf = fw[name]
        wq = qw[name] if qw is not None else None

        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            Wf = wf.module.weight
            Y = x.float() @ Wf.float().t()
            if wf.module.bias is not None:
                Y = Y + wf.module.bias.float()
            if fp16_control:
                Yh = (x @ Wf.t())
                if wf.module.bias is not None:
                    Yh = Yh + wf.module.bias
                Yh = Yh.float()
            else:
                xc = x
                order = getattr(wq, "_column_order", None)
                if order is not None and wq.reorder_before_quant:
                    xc = xc[..., order]
                qz = wq.quantizer
                if _quant_enabled(qz.bits):
                    qz.find_params(xc)
                    xq = qz(xc).to(x.dtype)
                    qz.free()
                else:
                    xq = xc
                Wq = wq.module.weight
                Yh = xq.float() @ Wq.float().t()
                if wq.module.bias is not None:
                    Yh = Yh + wq.module.bias.float()
            a = acc[name]
            a["sse"] += float((Y - Yh).pow(2).sum(dtype=torch.float64))
            a["energy"] += float(Y.pow(2).sum(dtype=torch.float64))
            a["n_tokens"] += x.shape[0]
        return hook

    for name in fw:
        handles.append(fw[name].module.register_forward_pre_hook(mk_hook(name)))
    for w_ in wins:
        fmodel(w_.to(DEV))
    for h in handles:
        h.remove()

    out = {}
    for name, a in acc.items():
        nel = a["n_tokens"] * a["out"]
        out[name] = dict(
            projection_type=name.split(".")[-1],
            n_tokens=a["n_tokens"], n_output_elements=nel,
            SSE=a["sse"], ENERGY=a["energy"],
            MSE=a["sse"] / nel,
            relative_squared_error=a["sse"] / (a["energy"] + EPS))
    return out, offs


def layer_error_stats(errs):
    es = [v["relative_squared_error"] for v in errs.values()]
    L = len(es)
    mean = sum(es) / L
    std = math.sqrt(sum((e - mean) ** 2 for e in es) / L)   # ddof=0, layer spread
    return mean, std


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------
def run_path(rid):
    return os.path.join(DATA_DIR, f"{rid}.json")


def run_complete(rid):
    p = run_path(rid)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return d.get("status", "").startswith(("ok", "reused"))
    except Exception:
        return False


def find_physical(mk, rep, ahash):
    for p in sorted(glob.glob(os.path.join(DATA_DIR, f"{mk}_*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if (d.get("status") == "ok" and d.get("representation") == rep
                and d.get("alloc_hash") == ahash):
            return d
    return None


def recipe_meta():
    return dict(
        gptq=dict(nsamples=128, percdamp=0.01, w_clip=True, act_order=False,
                  calib="wikitext2-train seed0 first-128 windows of the "
                        "frozen 512-window fit sequence"),
        act_mse=qu.ACT_MSE_SCALE_SEARCH, act_mse_steps=qu.ACT_MSE_STEPS,
        act_mse_steps_up=qu.ACT_MSE_STEPS_UP, act_mse_fast=qu.ACT_MSE_FAST,
        weight_mse=qu.WEIGHT_MSE_SCALE_SEARCH, weight_mse_steps=qu._MSE_STEPS,
        weight_two_level=qu.WEIGHT_TWO_LEVEL,
        formats=dict(main="nvfp4 blk16", high="mxfp8 blk32"),
        o_proj="fp16 (excluded, all configs)", kv="fp16 (KV16)",
        lm_head="fp16", had_mode=os.environ.get("PROMIX_HAD_MODE", "fp16"))


def cmd_run(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    mk, strat, scope, seed = args.model, args.strategy, args.scope, args.assignment_seed
    rep = rep_for(strat)
    rid = f"{mk}_{strat}_{scope}_s{seed}_r{args.ratio:g}"
    if run_complete(rid) and not args.force:
        print(f"[run] {rid} complete, skip")
        print("M31_DONE")
        return
    t_start = time.time()
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    scores = load_scores(mk, rep)
    alloc, H, total, r_actual = compute_alloc(mk, strat, scope, args.ratio,
                                              seed, scores)
    ahash = alloc_hash(rep, alloc)
    base = dict(run_id=rid, model=mk, strategy=strat, scope=scope,
                representation=rep, seed=seed, target_ratio=args.ratio,
                actual_ratio=r_actual, H=H, total_channels=total,
                alloc_hash=ahash, recipe=recipe_meta(),
                alloc={n: dict(d=a["d"], h=a["h"], ids_sha256=ids_sha(a["sel"]),
                               packed_order="unselected_asc+selected_asc",
                               transform_id=rep,
                               score_summary=a["score_summary"])
                       for n, a in alloc.items()})

    phys = find_physical(mk, rep, ahash)
    if phys is not None and phys["run_id"] != rid:
        base.update(status="reused", physical_run_id=phys["run_id"],
                    wiki=phys["wiki"], c4=phys["c4"],
                    layer_errors=phys["layer_errors"],
                    error_mean_layers=phys["error_mean_layers"],
                    error_std_layers=phys["error_std_layers"],
                    bytes=phys["bytes"], heldout_offsets=phys["heldout_offsets"],
                    seconds=round(time.time() - t_start, 1))
        json.dump(base, open(run_path(rid), "w"), indent=1)
        print(f"M31_RESULT {rid} reused<-{phys['run_id']} "
              f"wiki={phys['wiki']['ppl']:.4f} c4={phys['c4']['ppl']:.4f}")
        print("M31_DONE")
        return

    cfg = resolve(mk)
    print(f"[run] {rid} r_actual={r_actual:.4f} H={H}/{total} rep={rep}", flush=True)
    t0 = time.time()
    qmodel, cfg = build_quant_model(mk, rep, alloc, cfg)
    t_build = time.time() - t0
    print(f"[run] built+GPTQ ({t_build:.0f}s)", flush=True)

    # bytes accounting (weights + block scales + permutation metadata)
    wbytes = sbytes = mbytes = 0
    for name, w in eligible_wrappers(qmodel).items():
        a = alloc[name]
        out = w.module.weight.shape[0]
        d, h = a["d"], a["h"]
        wbytes += out * ((d - h) * 0.5 + h * 1.0)
        sbytes += out * ((d - h) / NVFP4_BLK + h / MXFP8_BLK)
        if 0 < h < d:
            mbytes += d * 4   # column permutation indices (int32)
    base["bytes"] = dict(weight_bytes=int(wbytes), scale_bytes=int(sbytes),
                         padding_bytes=0, metadata_bytes=int(mbytes))

    # teacher-input layer errors (float twin + quant model both on GPU)
    tok = get_tokenizer(cfg)
    t0 = time.time()
    fmodel, _ = build_float_model(mk, rep, resolve(mk))
    qmodel.to(DEV)
    errs, offs = layer_errors(fmodel, qmodel, cfg, tok)
    emean, estd = layer_error_stats(errs)
    base.update(layer_errors=errs, error_mean_layers=emean,
                error_std_layers=estd, heldout_offsets=offs)
    del fmodel
    cleanup_memory()
    t_err = time.time() - t0
    print(f"[run] layer_errors mean={emean:.6g} std={estd:.6g} ({t_err:.0f}s)",
          flush=True)

    bs = args.batch_size or (8 if mk == "8b" else cfg["eval"]["batch_size"])
    t0 = time.time()
    base["wiki"] = eval_wiki(qmodel, cfg, tok, batch_size=bs)
    print(f"[run] wiki={base['wiki']['ppl']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    base["c4"] = eval_c4(qmodel, cfg, tok, batch_size=bs)
    print(f"[run] c4={base['c4']['ppl']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    nan_layers = [n for n, v in errs.items()
                  if math.isnan(v["relative_squared_error"])]
    base.update(status="ok", nan_error_layers=nan_layers,
                timings=dict(build_gptq=round(t_build, 1),
                             layer_errors=round(t_err, 1)),
                seconds=round(time.time() - t_start, 1))
    json.dump(base, open(run_path(rid), "w"), indent=1)
    print(f"M31_RESULT {rid} wiki={base['wiki']['ppl']:.4f} "
          f"c4={base['c4']['ppl']:.4f} r_act={r_actual:.4f} "
          f"E_mean={emean:.6g} E_std={estd:.6g} nan={len(nan_layers)}")
    print("M31_DONE")


def cmd_control(args):
    """Float reference for one model x representation: PPL + fp16-GEMM noise
    floor of the layer-error machinery."""
    os.makedirs(DATA_DIR, exist_ok=True)
    mk, rep = args.model, args.rep
    rid = f"{mk}_float_{rep}"
    if run_complete(rid) and not args.force:
        print(f"[control] {rid} complete, skip")
        print("M31_DONE")
        return
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    cfg = resolve(mk)
    fmodel, cfg = build_float_model(mk, rep, cfg)
    tok = get_tokenizer(cfg)
    errs, offs = layer_errors(fmodel, None, cfg, tok, fp16_control=True)
    emean, estd = layer_error_stats(errs)
    bs = args.batch_size or (8 if mk == "8b" else cfg["eval"]["batch_size"])
    wiki = eval_wiki(fmodel, cfg, tok, batch_size=bs)
    c4 = eval_c4(fmodel, cfg, tok, batch_size=bs)
    out = dict(run_id=rid, model=mk, strategy="float", scope="none",
               representation=rep, seed=0, target_ratio=None, actual_ratio=None,
               H=0, total_channels=None,
               status="ok", wiki=wiki, c4=c4, layer_errors=errs,
               error_mean_layers=emean, error_std_layers=estd,
               heldout_offsets=offs, recipe=recipe_meta(),
               note="fp16-GEMM noise floor control (quant off)")
    json.dump(out, open(run_path(rid), "w"), indent=1)
    print(f"M31_RESULT {rid} wiki={wiki['ppl']:.4f} c4={c4['ppl']:.4f} "
          f"E_floor_mean={emean:.3g}")
    print("M31_DONE")


# ---------------------------------------------------------------------------
# selfcheck (spec section 9.3)
# ---------------------------------------------------------------------------
def cmd_selfcheck(args):
    mk = args.model
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    cfg = resolve(mk)
    tok = get_tokenizer(cfg)
    res = {}

    # (0) quantizer segment-routing unit check
    torch.manual_seed(7)
    x = torch.randn(4, 128, dtype=torch.float16, device=DEV) * 3
    for h in (0, 32, 64, 128):
        q = ActQuantizer()
        if h == 128:
            q.configure(bits="mxfp8", groupsize=-1, sym=False, clip_ratio=1.0,
                        high_bits_length=0, high_bits="mxfp8",
                        low_bits_length=0, low_bits="nvfp4")
            ref = fake_quantize_mxfp8(x)
        else:
            q.configure(bits="nvfp4", groupsize=-1, sym=False, clip_ratio=1.0,
                        high_bits_length=h, high_bits="mxfp8",
                        low_bits_length=0, low_bits="nvfp4")
            if h == 0:
                ref = fake_quantize_nvfp4(x)
            else:
                ref = torch.cat([fake_quantize_nvfp4(x[..., :128 - h]),
                                 fake_quantize_mxfp8(x[..., 128 - h:])], -1)
        got = q(x)
        assert torch.equal(got, ref), f"segment routing mismatch h={h}"
    res["segment_routing"] = "PASS"
    print("[selfcheck] segment routing PASS", flush=True)

    # (1) quota feasibility + endpoint totals + across H equality
    scores_raw = load_scores(mk, "raw")
    for r in [i / 10 for i in range(11)]:
        for strat in ("ss", "ma"):
            a_w, H_w, tot, _ = compute_alloc(mk, strat, "within", r, 0, scores_raw)
            a_x, H_x, _, _ = compute_alloc(mk, strat, "across", r, 0, scores_raw)
            assert H_w == H_x
            for n, a in list(a_w.items()) + list(a_x.items()):
                assert a["h"] % 32 == 0 and (a["d"] - a["h"]) % 16 == 0
                assert len(a["sel"]) == a["h"]
    a0, H0, tot, r0 = compute_alloc(mk, "ss", "within", 0.0, 0, scores_raw)
    a1, H1, _, r1 = compute_alloc(mk, "ss", "within", 1.0, 0, scores_raw)
    assert H0 == 0 and r0 == 0.0 and H1 == tot and r1 == 1.0
    # endpoint mask identity across raw strategies (structural)
    for strat in ("rnd", "ma"):
        for r, ref in ((0.0, a0), (1.0, a1)):
            a, _, _, _ = compute_alloc(mk, strat, "within", r, 0, scores_raw)
            assert all(torch.equal(a[n]["sel"], ref[n]["sel"]) for n in a)
    res["quota_checks"] = "PASS"
    print("[selfcheck] quota/endpoint structural checks PASS", flush=True)

    # (2) common float reference (raw rep, no quant)
    fr, _ = build_float_model(mk, "raw", resolve(mk))
    ppl_raw = eval_wiki(fr, cfg, tok)["ppl"]
    res["ppl_float_raw"] = ppl_raw
    print(f"[selfcheck] float raw (norm-fused fp16) wiki={ppl_raw:.4f}", flush=True)

    # (3) permutation-machinery equivalence (random alloc, quant OFF)
    scores_rnd = scores_raw
    alloc, _, _, _ = compute_alloc(mk, "rnd", "within", 0.5, 123, scores_rnd)
    apply_allocation(fr, alloc)          # permutes weights + activation orders
    ppl_perm = eval_wiki(fr, cfg, tok)["ppl"]
    res["ppl_perm_noquant"] = ppl_perm
    res["perm_delta"] = ppl_perm - ppl_raw
    print(f"[selfcheck] perm-noquant wiki={ppl_perm:.4f} "
          f"(delta={ppl_perm - ppl_raw:+.5f})", flush=True)
    del fr
    cleanup_memory()

    # (4) PCA rep, no quant (legal-transform equivalence)
    fp, _ = build_float_model(mk, "pca", resolve(mk))
    ppl_pca = eval_wiki(fp, cfg, tok)["ppl"]
    res["ppl_float_pca"] = ppl_pca
    res["pca_delta"] = ppl_pca - ppl_raw
    print(f"[selfcheck] float pca wiki={ppl_pca:.4f} "
          f"(delta={ppl_pca - ppl_raw:+.5f})", flush=True)
    del fp
    cleanup_memory()

    # (5) plain fp16 (no norm fuse) disclosure anchor
    plain = load_model(cfg["model"]["name"], dtype=torch.float16)
    plain.config.use_cache = False
    plain.seqlen = cfg["eval"]["max_length"]
    ppl_plain = eval_wiki(plain, cfg, tok)["ppl"]
    res["ppl_plain_fp16"] = ppl_plain
    print(f"[selfcheck] plain fp16 wiki={ppl_plain:.4f}", flush=True)
    del plain
    cleanup_memory()

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"selfcheck_{mk}.json"), "w"),
              indent=2)
    print(f"M31_SELFCHECK {json.dumps(res)}")
    print("M31_DONE")


# ---------------------------------------------------------------------------
# alloc dump (audit: reconstruct full stable channel ids for any point)
# ---------------------------------------------------------------------------
def cmd_alloc(args):
    scores = load_scores(args.model, rep_for(args.strategy))
    alloc, H, total, r_actual = compute_alloc(
        args.model, args.strategy, args.scope, args.ratio,
        args.assignment_seed, scores)
    out = dict(model=args.model, strategy=args.strategy, scope=args.scope,
               seed=args.assignment_seed, target_ratio=args.ratio,
               actual_ratio=r_actual, H=H, total_channels=total,
               alloc_hash=alloc_hash(rep_for(args.strategy), alloc),
               modules={n: dict(d=a["d"], h=a["h"], ids_sha256=ids_sha(a["sel"]),
                                stable_selected_channel_ids=a["sel"].tolist())
                        for n, a in alloc.items()})
    path = args.out or os.path.join(
        OUT_DIR, f"alloc_dump_{args.model}_{args.strategy}_{args.scope}"
                 f"_s{args.assignment_seed}_r{args.ratio:g}.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def cmd_aggregate(_args):
    runs, lerr, allo = [], [], []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "run_id" not in d:
            continue
        runs.append(d)
    with open(os.path.join(OUT_DIR, "runs.csv"), "w") as f:
        f.write("model,strategy,scope,representation,target_ratio,actual_ratio,"
                "seed,H,total_channels,weight_bytes,scale_bytes,padding_bytes,"
                "metadata_bytes,wiki_ppl,c4_ppl,error_mean_layers,"
                "error_std_layers,status,run_id\n")
        for d in runs:
            b = d.get("bytes") or {}
            f.write(",".join(str(x) for x in [
                d["model"], d["strategy"], d["scope"], d["representation"],
                d.get("target_ratio"), d.get("actual_ratio"), d.get("seed"),
                d.get("H"), d.get("total_channels"),
                b.get("weight_bytes", ""), b.get("scale_bytes", ""),
                b.get("padding_bytes", ""), b.get("metadata_bytes", ""),
                (d.get("wiki") or {}).get("ppl", ""),
                (d.get("c4") or {}).get("ppl", ""),
                d.get("error_mean_layers", ""), d.get("error_std_layers", ""),
                d.get("status"), d["run_id"]]) + "\n")
    with open(os.path.join(OUT_DIR, "layer_errors.csv"), "w") as f:
        f.write("run_id,module_id,projection_type,n_tokens,n_output_elements,"
                "SSE,ENERGY,MSE,relative_squared_error\n")
        for d in runs:
            for n, v in (d.get("layer_errors") or {}).items():
                f.write(f"{d['run_id']},{n},{v['projection_type']},"
                        f"{v['n_tokens']},{v['n_output_elements']},"
                        f"{v['SSE']!r},{v['ENERGY']!r},{v['MSE']!r},"
                        f"{v['relative_squared_error']!r}\n")
    with open(os.path.join(OUT_DIR, "allocations.jsonl"), "w") as f:
        for d in runs:
            for n, a in (d.get("alloc") or {}).items():
                f.write(json.dumps(dict(run_id=d["run_id"], module_id=n, **a))
                        + "\n")
    print(f"aggregated {len(runs)} runs -> runs.csv / layer_errors.csv / "
          f"allocations.jsonl")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--model", required=True, choices=list(MODELS))
    pc.add_argument("--rep", required=True, choices=["raw", "pca"])

    ps = sub.add_parser("selfcheck")
    ps.add_argument("--model", default="3b", choices=list(MODELS))

    pr = sub.add_parser("run")
    pr.add_argument("--model", required=True, choices=list(MODELS))
    pr.add_argument("--strategy", required=True,
                    choices=["rnd", "ss", "ma", "av"])
    pr.add_argument("--scope", required=True, choices=["within", "across"])
    pr.add_argument("--ratio", type=float, required=True)
    pr.add_argument("--assignment_seed", type=int, default=0)
    pr.add_argument("--batch_size", type=int, default=None)
    pr.add_argument("--force", action="store_true")

    pf = sub.add_parser("control")
    pf.add_argument("--model", required=True, choices=list(MODELS))
    pf.add_argument("--rep", required=True, choices=["raw", "pca"])
    pf.add_argument("--batch_size", type=int, default=None)
    pf.add_argument("--force", action="store_true")

    pa = sub.add_parser("alloc")
    pa.add_argument("--model", required=True, choices=list(MODELS))
    pa.add_argument("--strategy", required=True,
                    choices=["rnd", "ss", "ma", "av"])
    pa.add_argument("--scope", required=True, choices=["within", "across"])
    pa.add_argument("--ratio", type=float, required=True)
    pa.add_argument("--assignment_seed", type=int, default=0)
    pa.add_argument("--out", default=None)

    pg = sub.add_parser("aggregate")

    args = ap.parse_args()
    if args.cmd == "collect":
        ptqmod.init_distributed_for_ptq_main_if_needed()
        transformers.set_seed(0)
        collect_scores(args.model, args.rep)
        print("M31_DONE")
    elif args.cmd == "selfcheck":
        cmd_selfcheck(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "control":
        cmd_control(args)
    elif args.cmd == "alloc":
        cmd_alloc(args)
    elif args.cmd == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
