"""AdaMix Motivation E1a — local controlled representation x precision diagnostic.

Task book §4 E1a. CPU-only, isolated-linear analysis on fixed FP16 teacher
inputs X and weights W for the eligible linears (q/k/v/gate/up/down; o_proj
EXCLUDED per §3.1). Four representations (orthogonal input-space transforms T):

  R0 = I                      (original representation)
  R1 = blockdiag(R_g)         (intra-group rotation only; G=512 Sylvester-Had)
  R2 = P                      (PCA change-of-basis only)
  R3 = P . blockdiag(R_g)     (PCA + intra-group rotation)

P definition (documented, honest):
  - q/k/v/gate/up : the SHIPPED shared PCA basis `attn_mlp` (3072x3072) from the
    production U file -> this IS the pipeline representation for the residual
    stream. Faithful.
  - down_proj : a LOCAL full PCA (per-layer, variance-sorted eigenbasis of the
    layer's own fit-split down-input covariance). NOTE: the shipped down path is
    256-block PCA + online Hadamard, NOT a single global PCA, so down R2/R3 are
    an ISOLATED change-of-basis probe, not the shipped representation. Flagged.

Two precision settings reused inside every representation (4x2 matrix):
  U4     : all eligible channels NVFP4.
  LDelta : per-layer fixed k_l = round(1/8 * n_groups) groups upgraded to MXFP8,
           chosen by measured per-group Delta = d_fp4 - d_fp8 on the fit split.

Quantization is RTN with the deployed NVFP4 / MXFP8 fake-quant (promix
quant_utils.fake_quantize_nvfp4 / fake_quantize_mxfp8) applied to BOTH X and W.
NO GPTQ (§4: avoid weight compensation masking the representation effect).
Weights use single-level RTN fake-quant (documented; the deployed two-level
weight refinement is a weight-only option, disabled here for a clean control).

fit/diagnostic split (§3.2): disjoint wikitext-2 train windows; masks + Delta
selected on fit, re-measured (NOT re-selected) on diagnostic.

Outputs per task book §9: manifest.json, metrics.json, group_stats.csv
(long table with STABLE group IDs = group index in the transformed space),
allocation.json, plus a compact summary printed to stdout.
"""
import argparse
import json
import os
import sys
import time
import hashlib
import csv

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import transformers

from promix.models.loader import load_model
from promix.quantize.fuse_norm import fuse_layer_norms
from promix.quantize.ring_rotation import build_uniform_ring_R1, compose_rotation
from promix.eval.data import get_wikitext2
from promix.quantize import quant_utils as qu

torch.set_grad_enabled(False)
DEV = "cpu"

# Eligible linears (o_proj excluded per §3.1). q/k/v share attn-input; gate/up
# share mlp-input; down has its own input space.
ATTN = ("q_proj", "k_proj", "v_proj")
MLP = ("gate_proj", "up_proj")
SPACE_OF = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn",
            "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "down"}


def _nvfp4(x):
    return qu.fake_quantize_nvfp4(x)


def _mxfp8(x):
    return qu.fake_quantize_mxfp8(x)


def file_sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def collect_X(model, samples, targets, max_tokens):
    """One forward pass; grab the fp16 (->fp32) input into one representative
    linear per (layer, space): q_proj for attn, gate_proj for mlp, down_proj for
    down. Returns {(layer_idx, space): Tensor(tokens, in)}."""
    store = {}
    handles = []

    def mk(key):
        def hook(_m, inp):
            if store.get(key) is not None and store[key][0] >= max_tokens:
                return
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()
            cur = store.get(key)
            if cur is None:
                store[key] = [0, []]
                cur = store[key]
            take = min(x.shape[0], max_tokens - cur[0])
            cur[1].append(x[:take].clone())
            cur[0] += take
        return hook

    layers = model.model.layers
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(mk((li, "attn"))))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(mk((li, "mlp"))))
        if "down_proj" in targets:
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(mk((li, "down"))))

    model.eval()
    for (inp, _tar) in samples:
        model(inp.to(DEV))
        if all(store.get(k, [0])[0] >= max_tokens for k in store):
            if len(store) >= len(layers) * (3 if "down_proj" in targets else 2):
                break
    for h in handles:
        h.remove()
    return {k: torch.cat(v[1], 0)[:max_tokens] for k, v in store.items()}


def local_pca(X):
    """Variance-sorted eigenbasis (columns) of the (centered? no - raw 2nd
    moment, matching PCA-over-activations convention) covariance of X.
    Returns T (in,in) with columns = components, descending eigenvalue."""
    Xf = X.to(torch.float64)
    C = (Xf.t() @ Xf) / Xf.shape[0]
    evals, evecs = torch.linalg.eigh(C)       # ascending
    idx = torch.argsort(evals, descending=True)
    return evecs[:, idx].to(torch.float32)


def act_pack(Xt, G):
    """Per-group activation stats + cached NVFP4/MXFP8 block tensors for a
    transformed activation Xt (tokens,in). Quantizes X ONCE. Returns a dict with
    per-group stats (var/maxabs/rms/d4/d8/delta) and reshaped q4/q8/raw blocks
    (n,ng,G) reused for every downstream metric so X is never re-quantized."""
    n, d = Xt.shape
    ng = d // G
    Xb = Xt.reshape(n, ng, G)
    mean = Xb.mean(0)
    var_c = (Xb.pow(2).mean(0) - mean.pow(2)).clamp_min(0)
    var_g = var_c.mean(1)
    maxabs = Xb.abs().amax(dim=(0, 2))
    rms = Xb.pow(2).mean(dim=(0, 2)).sqrt()
    q4 = _nvfp4(Xt).reshape(n, ng, G)
    q8 = _mxfp8(Xt).reshape(n, ng, G)
    d4 = (Xb - q4).pow(2).sum(dim=(0, 2)) / (n * G)
    d8 = (Xb - q8).pow(2).sum(dim=(0, 2)) / (n * G)
    return dict(var=var_g, maxabs=maxabs, rms=rms, d4=d4, d8=d8,
                delta=(d4 - d8), ng=ng, n=n, d=d, Xb=Xb, q4=q4, q8=q8)


def clip_rate(Xt, fmt, G):
    """Fraction of elements clamped under the deployed base block scale.
    fmt in {fp4,fp8}. Base scale = amax_block / MAX (ceil to grid); clip iff
    |x|/scale > MAX before clamp. Labeled clip_rate_basescale (MSE search can
    only increase it)."""
    if fmt == "fp4":
        s = qu.select_block_scales_nvfp4(Xt, block_size=16)   # (.., k16)
        bs, MAX = 16, qu.FP4_E2M1_MAX
    else:
        s = qu.select_block_scales_mxfp8(Xt, block_size=32)
        bs, MAX = 32, qu.FP8_E4M3_MAX
    n, d = Xt.shape
    xb = Xt.reshape(n, d // bs, bs)
    sc = s.reshape(n, d // bs, 1)
    over = (xb.abs() / sc) > (MAX + 1e-6)
    return float(over.float().mean())


def _mix_blocks(q4b, q8b, fp8_groups):
    """Assemble mixed quant from cached (n,ng,G) q4/q8 blocks: fp8_groups use
    MXFP8, rest NVFP4. Returns flat (n,in)."""
    n, ng, G = q4b.shape
    out = q4b.clone()
    if fp8_groups:
        idx = torch.tensor(sorted(fp8_groups), dtype=torch.long)
        out[:, idx, :] = q8b[:, idx, :]
    return out.reshape(n, ng * G)


def metrics_from_cache(Xt, Wt, Xq4b, Xq8b, Wq4b, Wq8b, fp8_groups):
    """All §4 metrics using PRE-QUANTIZED block caches (no re-quantization).
    Xt/Wt raw (flat); X*b/W*b are (.,ng,G) caches."""
    n, ngX, G = Xq4b.shape
    Xh = _mix_blocks(Xq4b, Xq8b, fp8_groups)
    Wh = _mix_blocks(Wq4b, Wq8b, fp8_groups)
    act_mse = float((Xt - Xh).pow(2).mean())
    w_mse = float((Wt - Wh).pow(2).mean())
    Y = Xt @ Wt.t()
    Yh = Xh @ Wh.t()
    out_mse = float((Y - Yh).pow(2).mean())
    rel_sq = float((Y - Yh).pow(2).sum() / Y.pow(2).sum().clamp_min(1e-12))
    return dict(act_mse=act_mse, weight_mse=w_mse, out_mse=out_mse,
                out_rel_sq=rel_sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/luofan/models/Llama-3.2-3B")
    ap.add_argument("--short", default="Llama-3.2-3B")
    ap.add_argument("--basis", default="rotation/U-wikitext-512-Llama-3.2-3B.bin")
    ap.add_argument("--G", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--fit_samples", type=int, default=3)
    ap.add_argument("--diag_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_down", action="store_true")
    ap.add_argument("--out_dir", default="docs/adamix_motivation/runs/e1a_3b")
    args = ap.parse_args()

    os.environ["PROMIX_LOAD_DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    G = args.G
    targets = list(ATTN + MLP) + (["down_proj"] if args.include_down else [])
    print(f"START E1a model={args.short} G={G} targets={targets} "
          f"max_tokens={args.max_tokens}", flush=True)

    transformers.set_seed(args.seed)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    nsmp = args.fit_samples + args.diag_samples
    allsmp = get_wikitext2(nsamples=nsmp, seed=args.seed, seqlen=2048,
                           tokenizer=tok, eval_mode=False)
    fit_smp = allsmp[:args.fit_samples]
    diag_smp = allsmp[args.fit_samples:nsmp]

    model = load_model(args.model, dtype=torch.float32)
    fuse_layer_norms(model)
    model.eval()
    print(f"[t+{time.time()-t_start:.0f}s] model loaded+norm-fused", flush=True)

    Xfit = collect_X(model, fit_smp, targets, args.max_tokens)
    print(f"[t+{time.time()-t_start:.0f}s] fit X collected: "
          f"{len(Xfit)} (layer,space) tensors", flush=True)
    Xdiag = collect_X(model, diag_smp, targets, args.max_tokens)
    print(f"[t+{time.time()-t_start:.0f}s] diag X collected", flush=True)

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size

    # --- transforms for the 3072 (attn/mlp shared) space --------------------
    Ubin = torch.load(args.basis, weights_only=False)
    P_shared = Ubin["attn_mlp"].to(torch.float32)          # (3072,3072)
    assert P_shared.shape[0] == hidden, P_shared.shape
    ring_h = build_uniform_ring_R1(hidden, ring_size=G, dtype=torch.float64).to(torch.float32)
    T_hidden = {
        "R0": torch.eye(hidden),
        "R1": ring_h,
        "R2": P_shared,
        "R3": compose_rotation(P_shared.double(), ring_h.double()).float(),
    }

    # example layers rule (written to manifest): shallow/mid/deep
    ex_layers = sorted({1, n_layers // 2, n_layers - 2})

    rows = []          # group_stats long table
    summary = []       # per (layer, module, rep, precision) metrics
    alloc = {}         # per (layer,module,rep): fit mask + diag check

    def process_space(li, space, mods_weights, Xf, Xd, T_map, pca_note):
        """All eligible linears sharing one input space. Activation transform +
        quant + stats done ONCE per (rep,split); only weights vary per linear."""
        for rep, T in T_map.items():
            Xtf = Xf @ T
            Xtd = Xd @ T
            pf = act_pack(Xtf, G)
            pd = act_pack(Xtd, G)
            ng = pf["ng"]
            k_l = int(round((1.0 / 8.0) * ng))
            fp8 = set(torch.topk(pf["delta"], k_l).indices.tolist()) if k_l > 0 else set()
            cr = {"fit": (clip_rate(Xtf, "fp4", G), clip_rate(Xtf, "fp8", G)),
                  "diag": (clip_rate(Xtd, "fp4", G), clip_rate(Xtd, "fp8", G))}
            for mod_short, W in mods_weights:
                mod_name = f"model.layers.{li}.{'self_attn' if space=='attn' else 'mlp'}.{mod_short}"
                Wt = W @ T
                n = Wt.shape[1]
                Wq4b = _nvfp4(Wt).reshape(W.shape[0], ng, G)
                Wq8b = _mxfp8(Wt).reshape(W.shape[0], ng, G)
                for split, Xt_, pk in (("fit", Xtf, pf), ("diag", Xtd, pd)):
                    mU = metrics_from_cache(Xt_, Wt, pk["q4"], pk["q8"], Wq4b, Wq8b, set())
                    mL = metrics_from_cache(Xt_, Wt, pk["q4"], pk["q8"], Wq4b, Wq8b, fp8)
                    cr4, cr8 = cr[split]
                    for pol, m in (("U4", mU), ("LDelta", mL)):
                        summary.append(dict(
                            layer=li, module=mod_name, space=space, rep=rep,
                            split=split, policy=pol, G=G, n_groups=ng, k_l=k_l,
                            rho_actual=round(len(fp8) / ng, 4),
                            act_mse=m["act_mse"], weight_mse=m["weight_mse"],
                            out_mse=m["out_mse"], out_rel_sq=m["out_rel_sq"],
                            clip_fp4=cr4, clip_fp8=cr8,
                            n_tokens=Xt_.shape[0], out_features=W.shape[0],
                            fp8_groups=";".join(map(str, sorted(fp8))),
                            pca_note=pca_note))
                for split, pk in (("fit", pf), ("diag", pd)):
                    for gid in range(ng):
                        rows.append(dict(
                            split=split, module_name=mod_name, rep=rep,
                            group_id_before_pack=gid, channel_start=gid * G, width=G,
                            n_tokens=pk["n"],
                            variance=float(pk["var"][gid]),
                            maxabs=float(pk["maxabs"][gid]),
                            rms=float(pk["rms"][gid]),
                            mse_fp4=float(pk["d4"][gid]),
                            mse_fp8=float(pk["d8"][gid]),
                            signed_delta=float(pk["delta"][gid]),
                            selected_fp8=int(gid in fp8)))
                alloc[f"{mod_name}|{rep}"] = dict(
                    k_l=k_l, n_groups=ng, fp8_groups=sorted(fp8),
                    rho_actual=len(fp8) / ng)
        print(f"  [t+{time.time()-t_start:.0f}s] done L{li} {space}", flush=True)

    # --- q/k/v/gate/up : shared-basis transforms -----------------------------
    for li in range(n_layers):
        layer = model.model.layers[li]
        attn_w = [(ms, getattr(layer.self_attn, ms).weight.data.float()) for ms in ATTN]
        process_space(li, "attn", attn_w, Xfit[(li, "attn")], Xdiag[(li, "attn")],
                      T_hidden, "shared_attn_mlp_basis")
        mlp_w = [(ms, getattr(layer.mlp, ms).weight.data.float()) for ms in MLP]
        process_space(li, "mlp", mlp_w, Xfit[(li, "mlp")], Xdiag[(li, "mlp")],
                      T_hidden, "shared_attn_mlp_basis")

    # --- down_proj : LOCAL PCA (documented isolated probe) -------------------
    down_status = "skipped"
    if args.include_down:
        down_status = "ok"
        try:
            inter = model.config.intermediate_size
            ring_d = build_uniform_ring_R1(inter, ring_size=G, dtype=torch.float64).to(torch.float32)
            for li in range(n_layers):
                Xf = Xfit[(li, "down")]
                P_loc = local_pca(Xf)
                T_down = {
                    "R0": torch.eye(inter),
                    "R1": ring_d,
                    "R2": P_loc,
                    "R3": compose_rotation(P_loc.double(), ring_d.double()).float(),
                }
                W = model.model.layers[li].mlp.down_proj.weight.data.float()
                process_space(li, "down", [("down_proj", W)], Xf, Xdiag[(li, "down")],
                              T_down, "local_full_pca_NOT_shipped_down_repr")
        except Exception as e:  # never discard the qkv/gate/up deliverables
            import traceback
            down_status = f"FAILED: {type(e).__name__}: {e}"
            print("DOWN_PHASE_FAILED:\n" + traceback.format_exc(), flush=True)

    # --- write deliverables --------------------------------------------------
    run_id = f"e1a_{args.short}_G{G}"
    manifest = dict(
        run_id=run_id, kind="E1a", model=args.model, model_short=args.short,
        basis_file=os.path.abspath(args.basis),
        basis_sha16=file_sha(args.basis), G=G, max_tokens=args.max_tokens,
        fit_samples=args.fit_samples, diag_samples=args.diag_samples,
        seqlen=2048, calib="wikitext-2-train", seed=args.seed,
        example_layers=ex_layers,
        example_layers_rule="shallow=1, mid=n_layers//2, deep=n_layers-2",
        reps={"R0": "I", "R1": "blockdiag(Sylvester-Hadamard-%d)" % G,
              "R2": "P (shared attn_mlp PCA for qkv/gate/up; local full PCA for down)",
              "R3": "P . blockdiag(R_g)"},
        precision={"U4": "all NVFP4", "LDelta": "per-layer top-k_l=round(ng/8) MXFP8 by fit Delta"},
        quant="RTN deployed NVFP4(bs16)/MXFP8(bs32) on X and W; single-level weight; NO GPTQ",
        act_mse_search=qu.ACT_MSE_SCALE_SEARCH, act_mse_fast=qu.ACT_MSE_FAST,
        weight_two_level=qu.WEIGHT_TWO_LEVEL, weight_mse_search=qu.WEIGHT_MSE_SCALE_SEARCH,
        down_included=args.include_down,
        down_status=down_status,
        down_caveat="down R2/R3 use LOCAL full PCA, NOT the shipped 256-block PCA + online Hadamard",
        eligible="q/k/v/gate/up/down (o_proj excluded per 3.1)",
        device="cpu", torch=torch.__version__, tf=transformers.__version__,
        wall_s=round(time.time() - t_start, 1))
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "allocation.json"), "w") as f:
        json.dump(alloc, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # group_stats.csv
    gcols = ["split", "module_name", "rep", "group_id_before_pack",
             "channel_start", "width", "n_tokens", "variance", "maxabs", "rms",
             "mse_fp4", "mse_fp8", "signed_delta", "selected_fp8"]
    with open(os.path.join(args.out_dir, "group_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # summary.csv (per-linear metrics, all layers = full distribution)
    scols = list(summary[0].keys())
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    # --- compact stdout aggregate (mean out_mse / act_mse over all layers) ---
    import statistics as st
    print("=" * 70, flush=True)
    print("E1a AGGREGATE (mean over all eligible q/k/v/gate/up layers, fit split)", flush=True)
    print(f"{'rep':>4} {'policy':>7} | {'act_mse':>10} {'out_mse':>10} "
          f"{'out_rel_sq':>11} {'clip_fp4':>9}", flush=True)
    for rep in ("R0", "R1", "R2", "R3"):
        for pol in ("U4", "LDelta"):
            sub = [s for s in summary if s["rep"] == rep and s["policy"] == pol
                   and s["split"] == "fit" and s["space"] in ("attn", "mlp")]
            if not sub:
                continue
            print(f"{rep:>4} {pol:>7} | "
                  f"{st.mean(s['act_mse'] for s in sub):10.3e} "
                  f"{st.mean(s['out_mse'] for s in sub):10.3e} "
                  f"{st.mean(s['out_rel_sq'] for s in sub):11.3e} "
                  f"{st.mean(s['clip_fp4'] for s in sub):9.4f}", flush=True)
    print(f"WROTE {args.out_dir} wall={time.time()-t_start:.0f}s", flush=True)
    print("E1A_DONE", flush=True)


if __name__ == "__main__":
    main()
