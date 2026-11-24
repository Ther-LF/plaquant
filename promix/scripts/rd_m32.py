#!/usr/bin/env python
"""rd_m32.py — Motivation §3.2: fixed allocation, within-precision-group rotation.

Two conditions on top of the frozen M31 AV-within allocation (same PCA/ResQ
representation, same precision mask, same PTQ recipe):

  Direct  — M31 AV-within results reused (verified, not just PPL numbers);
            activation-side A / magnitude stats computed here (stats32).
  Rotated — each eligible Linear's high group H_l and low group L_l are each
            multiplied by a deterministic random orthogonal matrix
            (float64 Gaussian -> QR -> diagonal sign fix; seed derived from
            model/module/ratio/group/rotation_seed). Weights paired as
            V_g R_g; activation side applies Z_g R_g at runtime (explicit
            algorithm-level reference implementation — NOT claimed zero-cost).

Hard gates (spec section 8): no ring path (identity_main_rotation build, ring_r1
False), no reuse of the fixed r=1/8 R1/R2 matrices as group rotations, per-run
transform manifest, equivalence.json (orthogonality residual / group energy /
layer outputs / logits / PPL bridge with quantization off).

Subcommands:
  run32     — one Rotated point (build + GPTQ + E_l/A/magnitude + wiki/C4)
  stats32   — Direct-side A + magnitude stats for all 9 ratios in one pass
  equiv32   — no-quant equivalence checks -> equivalence.json entry
  verify32  — Direct-reuse verification (mask/recipe vs M31 runs) -> manifest
  aggregate32 — runs.csv / layer_metrics.csv / magnitude_stats.csv
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))

import transformers

import rd_m31 as m31
from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.quantize import quant_utils as qu
from promix.quantize.hadamard import matmul_hadU_cuda
from promix.quantize.quant_utils import (
    ActQuantWrapper, _quant_enabled, find_qlayers,
    fake_quantize_nvfp4, fake_quantize_mxfp8,
)

OUT32 = os.path.join(REPO, "docs", "motivation_3_2")
DATA32 = os.path.join(OUT32, "data")
RATIOS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1..0.9 non-degenerate
EPS = 1e-12
RESERVOIR_N = 200_000       # per module x group abs-value samples
RESERVOIR_BLK_N = 100_000   # per module x group block max/RMS samples
RESERVOIR_SEED = 320001
QUANTS = (0.5, 0.9, 0.99, 0.999)


# ---------------------------------------------------------------------------
# deterministic group rotations
# ---------------------------------------------------------------------------
def rotation_seed_int(mk, module, ratio, group, rot_seed):
    s = f"m32:{mk}:{module}:r{ratio:g}:{group}:rot{rot_seed}"
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def make_rotation(mk, module, ratio, group, rot_seed, n):
    """float64 Gaussian -> QR -> diagonal sign correction (Haar-ish)."""
    g = torch.Generator().manual_seed(rotation_seed_int(mk, module, ratio,
                                                        group, rot_seed))
    A = torch.randn(n, n, dtype=torch.float64, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def apply_group_rotations(model, alloc, mk, ratio, rot_seed):
    """Rotate the packed [low | high] weight columns of each eligible Linear by
    blockdiag(R_low, R_high) (float64), and stash fp16 copies on the wrapper
    for the runtime activation-side Z_g R_g. Returns the transform manifest."""
    tman = {}
    for name, w in m31.eligible_wrappers(model).items():
        a = alloc[name]
        d, h = a["d"], a["h"]
        split = d - h
        Rl = make_rotation(mk, name, ratio, "low", rot_seed, split) \
            if split > 0 else None
        Rh = make_rotation(mk, name, ratio, "high", rot_seed, h) \
            if h > 0 else None
        W = w.module.weight.data
        dev = W.device          # NB: after fuse_basis the weights live on CPU
        dt = W.dtype
        W64 = W.to(DEV, torch.float64)   # rotate on GPU
        resid = 0.0
        if Rl is not None:
            Rld = Rl.to(DEV)
            W64[:, :split] = W64[:, :split] @ Rld
            resid = max(resid, float((Rld @ Rld.T
                                      - torch.eye(split, dtype=torch.float64,
                                                  device=DEV)).abs().max()))
        if Rh is not None:
            Rhd = Rh.to(DEV)
            W64[:, split:] = W64[:, split:] @ Rhd
            resid = max(resid, float((Rhd @ Rhd.T
                                      - torch.eye(h, dtype=torch.float64,
                                                  device=DEV)).abs().max()))
        w.module.weight.data = W64.to(dev, dt)
        w._m31_group_rot = (
            Rl.to(DEV, torch.float16) if Rl is not None else None,
            Rh.to(DEV, torch.float16) if Rh is not None else None,
            split,
        )
        tman[name] = dict(
            d=d, h=h, split=split,
            seed_low=rotation_seed_int(mk, name, ratio, "low", rot_seed)
            if split else None,
            seed_high=rotation_seed_int(mk, name, ratio, "high", rot_seed)
            if h else None,
            ortho_residual_fp64=resid)
    return tman


# ---------------------------------------------------------------------------
# wrapper forward with group rotation (process-local patch)
# ---------------------------------------------------------------------------
_ORIG_FORWARD = ActQuantWrapper.forward


def _m32_forward(self, x, column_order=None, **kwargs):
    rot = getattr(self, "_m31_group_rot", None)
    if rot is None:
        return _ORIG_FORWARD(self, x, column_order=column_order, **kwargs)
    x_dtype = x.dtype
    if self.online_full_had:
        if self.fp32_had:
            x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
        else:
            had_dtype = getattr(self, "had_dtype", None)
            xin = x if had_dtype is None else x.to(had_dtype)
            x = matmul_hadU_cuda(
                xin, self.had_K, self.K,
                kmat_fp32=getattr(self, "had_mixed", False)).to(x_dtype)
    order = getattr(self, "_column_order", None)
    if order is not None and self.reorder_before_quant:
        x = x[..., order]
    x = apply_act_rotation(x, rot).to(x_dtype)
    if _quant_enabled(self.quantizer.bits):
        self.quantizer.find_params(x)
        x = self.quantizer(x).to(x_dtype)
        self.quantizer.free()
    x = self.module(x).to(x_dtype)
    if _quant_enabled(self.out_quantizer.bits):
        self.out_quantizer.find_params(x)
        x = self.out_quantizer(x).to(x_dtype)
        self.out_quantizer.free()
    return x


def apply_act_rotation(x, rot):
    Rl, Rh, split = rot
    parts = []
    if split > 0:
        xl = x[..., :split]
        parts.append(xl @ Rl.to(xl.device, xl.dtype) if Rl is not None else xl)
    if x.shape[-1] - split > 0:
        xh = x[..., split:]
        parts.append(xh @ Rh.to(xh.device, xh.dtype) if Rh is not None else xh)
    return torch.cat(parts, dim=-1)


def install_m32_forward():
    ActQuantWrapper.forward = _m32_forward


# ---------------------------------------------------------------------------
# group statistics (spec sections 5.1 / 5.2)
# ---------------------------------------------------------------------------
class GroupStat:
    """Streaming per (module, group) stats: A (SSE/ENERGY, float64), abs-value
    reservoir + exact max, scale-block max/RMS reservoir + zero-block count."""

    def __init__(self, fmt, blk, seed):
        self.fmt, self.blk = fmt, blk
        self.sse = 0.0
        self.energy = 0.0
        self.nelem = 0
        self.absmax = 0.0
        self.zero_blocks = 0
        self.total_blocks = 0
        self.vals = []
        self.ratios = []
        self.g = torch.Generator(device="cuda").manual_seed(seed)

    def _sample(self, t, cap):
        if t.numel() <= cap:
            return t
        idx = torch.randint(0, t.numel(), (cap,), generator=self.g,
                            device=t.device)
        return t.reshape(-1)[idx]

    def update(self, z, nwin):
        """z: (tokens, g) fp16 group slice (teacher representation)."""
        zf = z.float()
        q = fake_quantize_mxfp8(z) if self.fmt == "mxfp8" \
            else fake_quantize_nvfp4(z)
        self.sse += float((zf - q.float()).pow(2).sum(dtype=torch.float64))
        self.energy += float(zf.pow(2).sum(dtype=torch.float64))
        self.nelem += zf.numel()
        a = zf.abs()
        self.absmax = max(self.absmax, float(a.max()) if a.numel() else 0.0)
        self.vals.append(self._sample(a, max(1, RESERVOIR_N // nwin)).cpu())
        blk = zf.reshape(-1, zf.shape[-1] // self.blk, self.blk)
        bmax = blk.abs().amax(-1)
        rms = blk.pow(2).mean(-1).sqrt()
        zero = rms == 0
        self.zero_blocks += int(zero.sum())
        self.total_blocks += rms.numel()
        ratio = torch.where(zero, torch.zeros_like(rms), bmax / rms.clamp_min(1e-30))
        self.ratios.append(self._sample(ratio, max(1, RESERVOIR_BLK_N // nwin)).cpu())

    def finalize(self, full_cdf=False):
        vals = torch.cat(self.vals) if self.vals else torch.zeros(1)
        rat = torch.cat(self.ratios) if self.ratios else torch.zeros(1)
        out = dict(
            format=self.fmt, block=self.blk,
            A=self.sse / (self.energy + EPS), SSE=self.sse, ENERGY=self.energy,
            n_elements=self.nelem, MSE=self.sse / max(self.nelem, 1),
            abs_quantiles={f"p{q*100:g}": float(torch.quantile(vals, q))
                           for q in QUANTS},
            abs_max=self.absmax,
            blk_maxrms_quantiles={f"p{q*100:g}": float(torch.quantile(rat, q))
                                  for q in (0.5, 0.9, 0.99)},
            blk_maxrms_max=float(rat.max()), blk_maxrms_mean=float(rat.mean()),
            zero_blocks=self.zero_blocks, total_blocks=self.total_blocks,
            n_sampled_vals=int(vals.numel()), n_sampled_blocks=int(rat.numel()),
            reservoir_seed=RESERVOIR_SEED)
        if full_cdf:
            qs = torch.linspace(0, 1, 101)
            out["abs_quantile_curve"] = [float(torch.quantile(vals, float(q)))
                                         for q in qs]
        return out


def representative_modules(model_layers, module_order):
    """Pre-declared shallow/mid/deep blocks x {q_proj, down_proj}."""
    L = model_layers
    picks = []
    for li in (2, L // 2, L - 3):
        picks += [f"model.layers.{li}.self_attn.q_proj",
                  f"model.layers.{li}.mlp.down_proj"]
    return [p for p in picks if p in module_order]


@torch.no_grad()
def layer_errors32(fmodel, qmodel, cfg, tok, alloc, mk, rotated, ratio,
                   rot_seed, rep_modules):
    """Held-out-teacher pass computing, per eligible module:
       - E_l (full quantized output vs float teacher output; same as M31)
       - A per precision group on Z' (teacher representation after packing
         [+ group rotation when rotated])
       - magnitude stats per group
    qmodel=None -> stats only (Direct side without re-running E/PPL)."""
    wins, offs = m31.heldout_windows(cfg, tok)
    nwin = len(wins)
    fmodel.to(DEV).eval()
    fw = m31.eligible_wrappers(fmodel)
    qw = m31.eligible_wrappers(qmodel) if qmodel is not None else None
    eacc = {n: dict(sse=0.0, energy=0.0, n_tokens=0,
                    out=fw[n].module.weight.shape[0]) for n in fw}
    gacc, rots = {}, {}
    for n in fw:
        a = alloc[n]
        split = a["d"] - a["h"]
        if rotated:
            Rl = make_rotation(mk, n, ratio, "low", rot_seed, split).to(
                DEV, torch.float16) if split > 0 else None
            Rh = make_rotation(mk, n, ratio, "high", rot_seed, a["h"]).to(
                DEV, torch.float16) if a["h"] > 0 else None
            rots[n] = (Rl, Rh, split)
        else:
            rots[n] = None
        gacc[n] = {}
        if split > 0:
            gacc[n]["low"] = GroupStat("nvfp4", 16, RESERVOIR_SEED)
        if a["h"] > 0:
            gacc[n]["high"] = GroupStat("mxfp8", 32, RESERVOIR_SEED)
    handles = []

    def mk_hook(name):
        wf = fw[name]
        wq = qw[name] if qw is not None else None
        a = alloc[name]
        split = a["d"] - a["h"]
        order = None
        sel = a["sel"]
        mask = torch.zeros(a["d"], dtype=torch.bool)
        mask[sel] = True
        order_t = torch.cat([torch.arange(a["d"])[~mask],
                             torch.arange(a["d"])[mask]]).to(DEV)

        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            xc = x[..., order_t]
            if rots[name] is not None:
                xc = apply_act_rotation(xc, rots[name]).to(x.dtype)
            # group stats on the teacher representation Z'
            if split > 0:
                gacc[name]["low"].update(xc[..., :split], nwin)
            if a["h"] > 0:
                gacc[name]["high"].update(xc[..., split:], nwin)
            if wq is None:
                return
            Wf = wf.module.weight
            Y = x.float() @ Wf.float().t()
            if wf.module.bias is not None:
                Y = Y + wf.module.bias.float()
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
            e = eacc[name]
            e["sse"] += float((Y - Yh).pow(2).sum(dtype=torch.float64))
            e["energy"] += float(Y.pow(2).sum(dtype=torch.float64))
            e["n_tokens"] += x.shape[0]
        return hook

    for name in fw:
        handles.append(fw[name].module.register_forward_pre_hook(mk_hook(name)))
    for w_ in wins:
        fmodel(w_.to(DEV))
    for h in handles:
        h.remove()

    errs = None
    if qmodel is not None:
        errs = {}
        for name, e in eacc.items():
            nel = e["n_tokens"] * e["out"]
            errs[name] = dict(projection_type=name.split(".")[-1],
                              n_tokens=e["n_tokens"], n_output_elements=nel,
                              SSE=e["sse"], ENERGY=e["energy"],
                              MSE=e["sse"] / nel,
                              relative_squared_error=e["sse"] / (e["energy"] + EPS))
    groups = {}
    for name, gs in gacc.items():
        groups[name] = {g: st.finalize(full_cdf=(name in rep_modules))
                        for g, st in gs.items()}
    return errs, groups, offs


def group_mean_std(groups, key):
    vals = [g[key]["A"] for g in groups.values() if key in g]
    L = len(vals)
    if L == 0:
        return None, None
    mean = sum(vals) / L
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / L)
    return mean, std


# ---------------------------------------------------------------------------
# transform manifest (spec section 8)
# ---------------------------------------------------------------------------
def transform_manifest(variant, tman=None):
    return dict(
        common_both_sides=dict(
            norm_fusion=True,
            residual_pca="U_attn fused, identity_main_rotation=True (R1 OFF)",
            ring_r1=False,
            per_head_value_R2="applied to v_proj-out/o_proj-in (o_proj excluded "
                              "from eligible set; FP16)",
            down_proj_online_hadamard="fp16 mode (representation transform for "
                                      "the intermediate dim; both sides)",
            column_permutation="shared AV-within precision mask; packed "
                               "[low_asc | high_asc]"),
        changed_by_experiment=(
            dict(group_rotation="per-Linear blockdiag(R_low, R_high) on the "
                                "packed layout; float64 QR construction, fp16 "
                                "runtime activation matmul (explicit reference "
                                "implementation, non-zero overhead)",
                 modules=tman) if variant == "rotated"
            else dict(group_rotation=None)),
        scope_note="rotation applied to ALL eligible linear projections "
                   "(q/k/v/gate/up/down), inside each precision group only")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def build_rotated_quant(mk, alloc, ratio, rot_seed, cfg):
    install_m32_forward()
    model, cfg = m31.build_float_model(mk, "pca", cfg)
    m31.apply_allocation(model, alloc)
    tman = apply_group_rotations(model, alloc, mk, ratio, rot_seed)
    saved = m31.patch_gptq()
    try:
        ptqmod.run_gptq_if_enabled(model, cfg, DEV)
    finally:
        m31.unpatch_gptq(saved)
    m31.configure_m31_quantizers(model, alloc)
    return model, cfg, tman


def cmd_run32(args):
    os.makedirs(DATA32, exist_ok=True)
    mk, r, rs = args.model, args.ratio, args.rotation_seed
    rid = f"{mk}_rotated_rot{rs}_r{r:g}"
    outp = os.path.join(DATA32, f"{rid}.json")
    if os.path.exists(outp) and not args.force:
        try:
            if json.load(open(outp)).get("status") == "ok":
                print(f"[run32] {rid} complete, skip")
                print("M32_DONE")
                return
        except Exception:
            pass
    t_start = time.time()
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    scores = m31.load_scores(mk, "pca")
    alloc, H, total, r_actual = m31.compute_alloc(mk, "av", "within", r, 0,
                                                  scores)
    ahash = m31.alloc_hash("pca", alloc)
    cfg = m31.resolve(mk)
    print(f"[run32] {rid} r_actual={r_actual:.4f} H={H}/{total}", flush=True)
    t0 = time.time()
    qmodel, cfg, tman = build_rotated_quant(mk, alloc, r, rs, cfg)
    t_build = time.time() - t0
    print(f"[run32] built+GPTQ ({t_build:.0f}s)", flush=True)

    tok = m31.get_tokenizer(cfg)
    fmodel, _ = m31.build_float_model(mk, "pca", m31.resolve(mk))
    qmodel.to(DEV)
    nlayers = len(fmodel.model.layers)
    rep_mods = representative_modules(nlayers, list(alloc))
    t0 = time.time()
    errs, groups, offs = layer_errors32(fmodel, qmodel, cfg, tok, alloc, mk,
                                        rotated=True, ratio=r, rot_seed=rs,
                                        rep_modules=rep_mods)
    del fmodel
    cleanup_memory()
    emean, estd = m31.layer_error_stats(errs)
    ah_m, ah_s = group_mean_std(groups, "high")
    al_m, al_s = group_mean_std(groups, "low")
    print(f"[run32] E={emean:.6g}±{estd:.6g} A_high={ah_m:.6g} "
          f"A_low={al_m:.6g} ({time.time()-t0:.0f}s)", flush=True)

    bs = args.batch_size or (8 if mk == "8b" else cfg["eval"]["batch_size"])
    wiki = m31.eval_wiki(qmodel, cfg, tok, batch_size=bs)
    print(f"[run32] wiki={wiki['ppl']:.4f}", flush=True)
    c4 = m31.eval_c4(qmodel, cfg, tok, batch_size=bs)
    print(f"[run32] c4={c4['ppl']:.4f}", flush=True)

    out = dict(run_id=rid, model=mk, variant="rotated", rotation_seed=rs,
               strategy="av", scope="within", representation="pca",
               target_ratio=r, actual_ratio=r_actual, H=H,
               total_channels=total, alloc_hash=ahash,
               alloc={n: dict(d=a["d"], h=a["h"],
                              ids_sha256=m31.ids_sha(a["sel"]))
                      for n, a in alloc.items()},
               recipe=m31.recipe_meta(),
               transform_manifest=transform_manifest("rotated", tman),
               rep_modules=rep_mods,
               wiki=wiki, c4=c4, layer_errors=errs, groups=groups,
               error_mean_layers=emean, error_std_layers=estd,
               A_high_mean=ah_m, A_high_std=ah_s,
               A_low_mean=al_m, A_low_std=al_s,
               heldout_offsets=offs, status="ok",
               seconds=round(time.time() - t_start, 1),
               timings=dict(build_gptq=round(t_build, 1)))
    json.dump(out, open(outp, "w"), indent=1)
    print(f"M32_RESULT {rid} wiki={wiki['ppl']:.4f} c4={c4['ppl']:.4f} "
          f"E={emean:.6g} A_h={ah_m:.6g} A_l={al_m:.6g}")
    print("M32_DONE")


def cmd_stats32(args):
    """Direct-side A + magnitude for all ratios in ONE held-out pass per ratio
    (model built once)."""
    os.makedirs(DATA32, exist_ok=True)
    mk = args.model
    outp = os.path.join(DATA32, f"{mk}_direct_stats.json")
    if os.path.exists(outp) and not args.force:
        print(f"[stats32] {outp} exists, skip")
        print("M32_DONE")
        return
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    scores = m31.load_scores(mk, "pca")
    cfg = m31.resolve(mk)
    fmodel, cfg = m31.build_float_model(mk, "pca", cfg)
    tok = m31.get_tokenizer(cfg)
    nlayers = len(fmodel.model.layers)
    res = {}
    for r in RATIOS:
        alloc, H, total, r_actual = m31.compute_alloc(mk, "av", "within", r,
                                                      0, scores)
        rep_mods = representative_modules(nlayers, list(alloc))
        _, groups, offs = layer_errors32(fmodel, None, cfg, tok, alloc, mk,
                                         rotated=False, ratio=r, rot_seed=0,
                                         rep_modules=rep_mods)
        ah_m, ah_s = group_mean_std(groups, "high")
        al_m, al_s = group_mean_std(groups, "low")
        res[f"{r:g}"] = dict(
            actual_ratio=r_actual, H=H, total_channels=total,
            alloc_hash=m31.alloc_hash("pca", alloc), groups=groups,
            A_high_mean=ah_m, A_high_std=ah_s,
            A_low_mean=al_m, A_low_std=al_s, rep_modules=rep_mods)
        print(f"[stats32] {mk} direct r={r} A_h={ah_m:.6g} A_l={al_m:.6g}",
              flush=True)
    json.dump(dict(model=mk, variant="direct", heldout_seed=m31.HELDOUT_SEED,
                   transform_manifest=transform_manifest("direct"),
                   ratios=res), open(outp, "w"), indent=1)
    print(f"M32_RESULT {mk} direct_stats written")
    print("M32_DONE")


def cmd_equiv32(args):
    """No-quant equivalence: orthogonality residual, group energy preservation,
    layer-output floor, whole-network logits + PPL bridge (float PCA twin vs
    rotated-no-quant model)."""
    os.makedirs(OUT32, exist_ok=True)
    mk, r, rs = args.model, args.ratio, args.rotation_seed
    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    scores = m31.load_scores(mk, "pca")
    alloc, H, total, r_actual = m31.compute_alloc(mk, "av", "within", r, 0,
                                                  scores)
    cfg = m31.resolve(mk)
    install_m32_forward()
    rmodel, cfg = m31.build_float_model(mk, "pca", cfg)
    m31.apply_allocation(rmodel, alloc)
    tman = apply_group_rotations(rmodel, alloc, mk, r, rs)
    ortho_max = max(v["ortho_residual_fp64"] for v in tman.values())

    fmodel, _ = m31.build_float_model(mk, "pca", m31.resolve(mk))
    tok = m31.get_tokenizer(cfg)
    # group energy preservation + layer-output floor via the standard pass
    nlayers = len(fmodel.model.layers)
    rep_mods = representative_modules(nlayers, list(alloc))
    rmodel.to(DEV)
    errs, groups_rot, _ = layer_errors32(fmodel, rmodel, cfg, tok, alloc, mk,
                                         rotated=True, ratio=r, rot_seed=rs,
                                         rep_modules=rep_mods)
    _, groups_dir, _ = layer_errors32(fmodel, None, cfg, tok, alloc, mk,
                                      rotated=False, ratio=r, rot_seed=rs,
                                      rep_modules=rep_mods)
    emean, estd = m31.layer_error_stats(errs)
    energy_ratios = []
    for n in groups_rot:
        for g in groups_rot[n]:
            er = groups_rot[n][g]["ENERGY"] / (groups_dir[n][g]["ENERGY"] + EPS)
            energy_ratios.append(er)
    # logits bridge on one held-out window
    wins, _ = m31.heldout_windows(cfg, tok, n=1)
    w0 = wins[0].to(DEV)
    fmodel.to(DEV)
    lf = fmodel(w0).logits.float()
    lr = rmodel(w0).logits.float()
    logit_maxdiff = float((lf - lr).abs().max())
    logit_cos = float(torch.nn.functional.cosine_similarity(
        lf.reshape(-1), lr.reshape(-1), dim=0))
    del fmodel
    cleanup_memory()
    bs = args.batch_size or (8 if mk == "8b" else cfg["eval"]["batch_size"])
    ppl_rot = m31.eval_wiki(rmodel, cfg, tok, batch_size=bs)["ppl"]
    ctrl = os.path.join(m31.DATA_DIR, f"{mk}_float_pca.json")
    ppl_float = json.load(open(ctrl))["wiki"]["ppl"] if os.path.exists(ctrl) \
        else None
    entry = dict(model=mk, ratio=r, rotation_seed=rs, actual_ratio=r_actual,
                 ortho_residual_fp64_max=ortho_max,
                 group_energy_ratio_min=min(energy_ratios),
                 group_energy_ratio_max=max(energy_ratios),
                 layer_output_floor_mean=emean, layer_output_floor_std=estd,
                 logits_max_abs_diff=logit_maxdiff, logits_cos=logit_cos,
                 wiki_ppl_rotated_noquant=ppl_rot,
                 wiki_ppl_float_pca=ppl_float,
                 ppl_delta=(ppl_rot - ppl_float) if ppl_float else None,
                 compute_dtype="fp16 runtime (fp64 construction)",
                 note="rotated model here has quantization OFF")
    path = os.path.join(OUT32, "equivalence.json")
    all_e = json.load(open(path)) if os.path.exists(path) else {}
    all_e[f"{mk}_r{r:g}_rot{rs}"] = entry
    json.dump(all_e, open(path, "w"), indent=2)
    print(f"M32_EQUIV {json.dumps(entry)}")
    print("M32_DONE")


def cmd_verify32(args):
    """Direct-reuse verification: recompute the AV-within alloc per ratio and
    compare against the stored M31 run (mask hash + recipe + representation +
    data protocol), not just PPL."""
    mk = args.model
    scores = m31.load_scores(mk, "pca")
    rows = {}
    ok_all = True
    for r in RATIOS:
        alloc, H, total, r_actual = m31.compute_alloc(mk, "av", "within", r,
                                                      0, scores)
        ahash = m31.alloc_hash("pca", alloc)
        p = os.path.join(m31.DATA_DIR, f"{mk}_av_within_s0_r{r:g}.json")
        if not os.path.exists(p):
            rows[f"{r:g}"] = dict(status="missing_m31_run")
            ok_all = False
            continue
        d = json.load(open(p))
        match = dict(
            alloc_hash=(d.get("alloc_hash") == ahash),
            representation=(d.get("representation") == "pca"),
            recipe=(d.get("recipe") == m31.recipe_meta()),
            status=d.get("status"))
        rows[f"{r:g}"] = dict(m31_run=d["run_id"], ahash=ahash, **match)
        ok_all = ok_all and all(v is True for k, v in match.items()
                                if k != "status")
    path = os.path.join(OUT32, f"direct_reuse_verification_{mk}.json")
    os.makedirs(OUT32, exist_ok=True)
    json.dump(dict(model=mk, all_match=ok_all, ratios=rows),
              open(path, "w"), indent=2)
    print(f"M32_VERIFY {mk} all_match={ok_all}")
    print("M32_DONE")


def cmd_aggregate32(_args):
    rows = []
    lm, mg = [], []

    def push_layer_metrics(rid, errs, groups):
        for n in (groups or {}):
            for g, s in groups[n].items():
                lm.append(dict(run_id=rid, module_id=n, group=g,
                               format=s["format"], n_elements=s["n_elements"],
                               SSE=s["SSE"], ENERGY=s["ENERGY"], A=s["A"],
                               MSE=s["MSE"],
                               out_SSE=(errs or {}).get(n, {}).get("SSE"),
                               out_ENERGY=(errs or {}).get(n, {}).get("ENERGY"),
                               out_E=(errs or {}).get(n, {}).get(
                                   "relative_squared_error")))
                mg.append(dict(run_id=rid, module_id=n, group=g,
                               **{k: s[k] for k in
                                  ("abs_quantiles", "abs_max",
                                   "blk_maxrms_quantiles", "blk_maxrms_max",
                                   "blk_maxrms_mean", "zero_blocks",
                                   "total_blocks", "ENERGY",
                                   "n_sampled_vals", "reservoir_seed")}))

    for p in sorted(glob.glob(os.path.join(DATA32, "*_rotated_*.json"))):
        d = json.load(open(p))
        rows.append(dict(run_id=d["run_id"], model=d["model"],
                         ratio_target=d["target_ratio"],
                         ratio_actual=d["actual_ratio"], variant="rotated",
                         rotation_seed=d["rotation_seed"],
                         wiki_ppl=d["wiki"]["ppl"], c4_ppl=d["c4"]["ppl"],
                         A_high_mean=d["A_high_mean"], A_high_std=d["A_high_std"],
                         A_low_mean=d["A_low_mean"], A_low_std=d["A_low_std"],
                         E_mean=d["error_mean_layers"],
                         E_std=d["error_std_layers"], status=d["status"]))
        push_layer_metrics(d["run_id"], d.get("layer_errors"), d.get("groups"))
    for mk in ("3b", "8b"):
        sp = os.path.join(DATA32, f"{mk}_direct_stats.json")
        if not os.path.exists(sp):
            continue
        ds = json.load(open(sp))
        for r, rv in ds["ratios"].items():
            p31 = os.path.join(m31.DATA_DIR, f"{mk}_av_within_s0_r{r}.json")
            d31 = json.load(open(p31)) if os.path.exists(p31) else {}
            rid = f"{mk}_direct_r{r}"
            rows.append(dict(run_id=rid, model=mk, ratio_target=float(r),
                             ratio_actual=rv["actual_ratio"], variant="direct",
                             rotation_seed="",
                             wiki_ppl=(d31.get("wiki") or {}).get("ppl", ""),
                             c4_ppl=(d31.get("c4") or {}).get("ppl", ""),
                             A_high_mean=rv["A_high_mean"],
                             A_high_std=rv["A_high_std"],
                             A_low_mean=rv["A_low_mean"],
                             A_low_std=rv["A_low_std"],
                             E_mean=d31.get("error_mean_layers", ""),
                             E_std=d31.get("error_std_layers", ""),
                             status="reused_m31:" + d31.get("run_id", "MISSING")))
            push_layer_metrics(rid, d31.get("layer_errors"), rv["groups"])
    cols = ["run_id", "model", "ratio_target", "ratio_actual", "variant",
            "rotation_seed", "wiki_ppl", "c4_ppl", "A_high_mean", "A_high_std",
            "A_low_mean", "A_low_std", "E_mean", "E_std", "status"]
    with open(os.path.join(OUT32, "runs.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for d in rows:
            f.write(",".join(str(d.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(OUT32, "layer_metrics.csv"), "w") as f:
        cols2 = ["run_id", "module_id", "group", "format", "n_elements", "SSE",
                 "ENERGY", "A", "MSE", "out_SSE", "out_ENERGY", "out_E"]
        f.write(",".join(cols2) + "\n")
        for d in lm:
            f.write(",".join(str(d.get(c, "")) for c in cols2) + "\n")
    with open(os.path.join(OUT32, "magnitude_stats.csv"), "w") as f:
        cols3 = ["run_id", "module_id", "group", "abs_quantiles", "abs_max",
                 "blk_maxrms_quantiles", "blk_maxrms_max", "blk_maxrms_mean",
                 "zero_blocks", "total_blocks", "ENERGY", "n_sampled_vals",
                 "reservoir_seed"]
        f.write(",".join(cols3) + "\n")
        for d in mg:
            f.write(",".join(json.dumps(d.get(c, "")).replace(",", ";")
                             for c in cols3) + "\n")
    print(f"aggregated {len(rows)} rows -> {OUT32}/runs.csv etc.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run32")
    pr.add_argument("--model", required=True, choices=["1b", "3b", "8b"])
    pr.add_argument("--ratio", type=float, required=True)
    pr.add_argument("--rotation_seed", type=int, default=0)
    pr.add_argument("--batch_size", type=int, default=None)
    pr.add_argument("--force", action="store_true")
    ps = sub.add_parser("stats32")
    ps.add_argument("--model", required=True, choices=["1b", "3b", "8b"])
    ps.add_argument("--force", action="store_true")
    pe = sub.add_parser("equiv32")
    pe.add_argument("--model", required=True, choices=["1b", "3b", "8b"])
    pe.add_argument("--ratio", type=float, default=0.5)
    pe.add_argument("--rotation_seed", type=int, default=0)
    pe.add_argument("--batch_size", type=int, default=None)
    pv = sub.add_parser("verify32")
    pv.add_argument("--model", required=True, choices=["1b", "3b", "8b"])
    pg = sub.add_parser("aggregate32")
    args = ap.parse_args()
    if args.cmd == "run32":
        cmd_run32(args)
    elif args.cmd == "stats32":
        cmd_stats32(args)
    elif args.cmd == "equiv32":
        cmd_equiv32(args)
    elif args.cmd == "verify32":
        cmd_verify32(args)
    elif args.cmd == "aggregate32":
        cmd_aggregate32(args)


if __name__ == "__main__":
    main()
