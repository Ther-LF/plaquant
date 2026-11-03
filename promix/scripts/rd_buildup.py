"""rd-buildup — E5a cumulative component build-up ablation (paper centerpiece).

ONE clean cumulative ladder on a single substrate/protocol: start from the
no-rotation RTN pure-NVFP4 cliff and add ONE component per row, measuring
wiki (+C4) PPL at each step. All mixed-precision rows run at the matched
avg FP8 = 1/8 budget; rows before mixed precision are pure NVFP4.

Rows (cumulative; each row = previous row + ONE component):
  1 norot      RTN pure-NVFP4, NO rotation (true vanilla cliff)
  2 pca_u      + PCA U rotation (identity main rotation, r1_off) + the
                 protocol substrate (norm fusion, column reorder, down_proj
                 online Hadamard) — concentration effect
  3 ring512    + ring512 R1 (T = U @ ring512_R1) — ring substrate
  4 uniform18  + uniform-1/8 mixed precision (positional tail) — partition
  5 gptq       + GPTQ weight compensation (per-model calibration:
                 1B n=1024 / 3B n=512,percdamp=.025 / 8B n=512)
  6 wscale     + two-level weight scale + micro-MSE search
                 (PROMIX_WEIGHT_MSE=1 + 2LEVEL=1, steps 7 on 3B/8B, 3 on 1B)
  7 actmse     + act-MSE scale search d3u2 (PROMIX_ACT_MSE=1, STEPS=3, UP=2)
  8 measured   + measured-DeltaD allocation (replaces uniform; global
                 threshold at the same 1/8 budget) == the shipped recipe;
                 on 3B must reproduce the rd_acctune anchor 8.2180/11.2549
  9 w4a16      row-8 weights, activation quant OFF (the decode-route W4A16
                 PPL; measured in the SAME invocation as row 8 via --also_w4a16)

Env knobs are process-level (read at quant_utils import), so the queue
script launches ONE row per process with the row's env; this driver VERIFIES
the live env matches what the row requires and aborts loudly on mismatch.

Rows 1-3 are pure NVFP4 (quantizer K_high=0 everywhere incl o_proj) but keep
the protocol's 1/8 substrate geometry (fuse/rearrange at high_fraction=0.125)
so the ONLY thing that changes at row 4 is the quantizer partition.

Reuses rd_validity build/collect/alloc/eval machinery + rd_ron.resolve_paths
(same protocol as rd_acctune; bridges digit-exact).

Usage (one row per process; see rd_buildup_queue.sh):
  PROMIX_ACT_MSE=0 PROMIX_WEIGHT_MSE=0 PROMIX_WEIGHT_2LEVEL=0 \
  python promix/scripts/rd_buildup.py \
      --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
      --row norot --point 3b_r1_norot --c4 --out OUT.json
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

from promix.utils import DEV, cleanup_memory
from promix.eval import ptq as ptqmod
from promix.eval.evaluator import evaluate_ppl
from promix.eval.data import get_wikitext2, get_c4
from promix.quantize import quant_utils as qu
from promix.quantize import gptq as gptqmod
from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
from promix.models.loader import load_model

import rd_validity as rv
import rd_ron

RING = 512  # the shipped ring substrate width = allocation atom

# Per-model GPTQ calibration recipe (rd_acctune promoted stacks).
MODEL_KNOBS = {
    "1b": dict(gptq_nsamples=1024, percdamp=0.01, wmse_steps=3),
    "3b": dict(gptq_nsamples=512, percdamp=0.025, wmse_steps=7),
    "8b": dict(gptq_nsamples=512, percdamp=0.01, wmse_steps=7),
}

# Row ladder. substrate: none | u | ring. hf: quantizer high_fraction.
# weights: rtn | gptq. alloc: pure | uniform | measured.
ROWS = {
    "norot":     dict(substrate="none", hf=0.0, weights="rtn", alloc="pure",
                      env="off"),
    "pca_u":     dict(substrate="u", hf=0.0, weights="rtn", alloc="pure",
                      env="off"),
    "ring512":   dict(substrate="ring", hf=0.0, weights="rtn", alloc="pure",
                      env="off"),
    "uniform18": dict(substrate="ring", hf=0.125, weights="rtn",
                      alloc="uniform", env="off"),
    "gptq":      dict(substrate="ring", hf=0.125, weights="gptq",
                      alloc="uniform", env="off"),
    "wscale":    dict(substrate="ring", hf=0.125, weights="gptq",
                      alloc="uniform", env="wmse"),
    "actmse":    dict(substrate="ring", hf=0.125, weights="gptq",
                      alloc="uniform", env="full"),
    "measured":  dict(substrate="ring", hf=0.125, weights="gptq",
                      alloc="measured", env="full"),
    # CONTROL (not a ladder row): strongest NO-ROTATION baseline = pure
    # NVFP4 + GPTQ + two-level weight scale + act-MSE. Isolates the net
    # end-point value of the rotation+partition substrate vs just stacking
    # weight/activation compensation on the raw model.
    "norot_full": dict(substrate="none", hf=0.0, weights="gptq",
                       alloc="pure", env="full"),
}


def model_key(config):
    n = config["model"]["name"].lower()
    if "13b" in n:
        raise SystemExit("13B not in the build-up matrix")
    for k in ("3b", "8b", "1b"):
        if k in n:
            return k
    raise SystemExit(f"cannot key model {n}")


def expected_env(row_env, knobs):
    """Returns dict of (qu-constant name -> required value)."""
    if row_env == "off":
        return dict(ACT_MSE_SCALE_SEARCH=False, WEIGHT_MSE_SCALE_SEARCH=False,
                    WEIGHT_TWO_LEVEL=False)
    if row_env == "wmse":
        return dict(ACT_MSE_SCALE_SEARCH=False, WEIGHT_MSE_SCALE_SEARCH=True,
                    WEIGHT_TWO_LEVEL=True, _MSE_STEPS=knobs["wmse_steps"])
    if row_env == "full":
        return dict(ACT_MSE_SCALE_SEARCH=True, ACT_MSE_STEPS=3,
                    ACT_MSE_STEPS_UP=2, WEIGHT_MSE_SCALE_SEARCH=True,
                    WEIGHT_TWO_LEVEL=True, _MSE_STEPS=knobs["wmse_steps"])
    raise SystemExit(f"bad env class {row_env}")


def verify_env(row_env, knobs):
    exp = expected_env(row_env, knobs)
    bad = {k: (getattr(qu, k), v) for k, v in exp.items()
           if getattr(qu, k) != v}
    if bad:
        raise SystemExit(f"ENV MISMATCH for row env '{row_env}': "
                         f"{{name: (live, required)}} = {bad}. "
                         "Set PROMIX_* env before launching this process.")
    return exp


def patch_gptq_rtn():
    """GPTQ -> exact RTN: identity Hessian => zero cross-column error
    propagation => each column rounds to nearest on the SAME frozen-scale
    grid find_params froze. Restores via the returned original."""
    orig = gptqmod.GPTQ.add_batch

    def rtn_add_batch(self, inp, out):
        if self.nsamples == 0:
            self.H = torch.eye(self.columns, device=self.dev,
                               dtype=self.H.dtype)
            self.nsamples = 1
    gptqmod.GPTQ.add_batch = rtn_add_batch
    return orig


def build_norot(config):
    """TRUE vanilla baseline: no norm fusion, no basis, no reorder, no
    online Hadamard — just ActQuantWrapper on the stock fp16 model."""
    model = load_model(config["model"]["name"], dtype=torch.float16)
    add_actquant(model)
    return model


def run_weights(model, config, mode):
    """RTN or GPTQ via the SAME gptq machinery (frozen-scale grid)."""
    restore = patch_gptq_rtn() if mode == "rtn" else None
    try:
        ptqmod.run_gptq_if_enabled(model, config, DEV)
    finally:
        if restore is not None:
            gptqmod.GPTQ.add_batch = restore


def eval_c4_ppl(model, config, seed=0):
    model.to(DEV)
    model.seqlen = config["eval"]["max_length"]
    tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    c4 = get_c4(nsamples=256, seed=seed, seqlen=model.seqlen,
                tokenizer=tok, eval_mode=True)
    return float(evaluate_ppl(model, c4, DEV,
                              batch_size=config["eval"]["batch_size"]))


def disable_act_quant(model):
    """Row 9: decode-route W4A16 — weights stay quantized (GPTQ already
    applied in-place), activation quantizers off. Reorder/Hadamard stay
    (they are part of the fused-rotation model math)."""
    n = 0
    for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
        if isinstance(getattr(w.quantizer, "bits", None), str):
            w.quantizer.bits = 16
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--row", required=True, choices=sorted(ROWS))
    ap.add_argument("--point", required=True)
    ap.add_argument("--also_w4a16", action="store_true",
                    help="after the row eval, flip activation quant OFF and "
                         "re-eval (row 9, decode-route W4A16; pair with "
                         "--row measured)")
    ap.add_argument("--calib_seed", type=int, default=None,
                    help="override calibration seed (eval sets stay seed-0)")
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row = ROWS[args.row]
    config = ptqmod.load_config(os.path.join(REPO, args.config))
    config = rd_ron.resolve_paths(config)
    mk = model_key(config)
    knobs = MODEL_KNOBS[mk]
    env_live = verify_env(row["env"], knobs)

    if args.calib_seed is not None:
        config["calibration"]["seed"] = args.calib_seed

    # Weight-quant calibration: RTN is data-independent -> tiny loader;
    # GPTQ rows use the per-model promoted calibration.
    if row["weights"] == "rtn":
        config["calibration"]["nsamples"] = 8
    else:
        config["calibration"]["nsamples"] = knobs["gptq_nsamples"]
        config["quantize"]["percdamp"] = knobs["percdamp"]

    if row["substrate"] != "none":
        for p in (config["paths"]["basis"], config["paths"]["rotation"]):
            assert os.path.exists(p), f"missing artifact: {p}"

    ptqmod.init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    meta = dict(point=args.point, row=args.row, model=config["model"]["name"],
                substrate=row["substrate"], hf=row["hf"],
                weights=row["weights"], alloc=row["alloc"],
                gptq_nsamples=config["calibration"]["nsamples"],
                percdamp=config["quantize"].get("percdamp", 0.01),
                calib_seed=config["calibration"].get("seed", 0),
                env={k: getattr(qu, k) for k in
                     ("ACT_MSE_SCALE_SEARCH", "ACT_MSE_STEPS",
                      "ACT_MSE_STEPS_UP", "WEIGHT_MSE_SCALE_SEARCH",
                      "_MSE_STEPS", "WEIGHT_TWO_LEVEL")})
    results = {"meta": meta}
    print(f"START rd_buildup {json.dumps(meta)}", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    ring_r1 = row["substrate"] == "ring"
    r1_off = row["substrate"] == "u"

    # --- allocation (measured rows only) --------------------------------
    chosen = None
    if row["alloc"] == "measured":
        tok = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
        calib = get_wikitext2(nsamples=args.calib_samples,
                              seed=config["calibration"].get("seed", 0),
                              seqlen=2048, tokenizer=tok, eval_mode=False)
        cmodel = rv.build_base(config, r1_off=r1_off, ring_r1=ring_r1,
                               ring_size=RING)
        cmodel.to(DEV)
        stats = rv.collect_stats(cmodel, calib,
                                 max_tokens=args.collect_tokens, ring=RING)
        del cmodel
        cleanup_memory()
        total_rings = sum(stats[n]["nr"] for n in stats)
        budget = int(round(0.125 * total_rings))
        chosen = rv.measured_alloc(stats, budget, total_rings)
        nsel = sum(len(v) for v in chosen.values())
        print(f"ALLOC total_rings={total_rings} sel={nsel} "
              f"(frac={nsel/total_rings:.4f})", flush=True)
        results["alloc_meta"] = dict(total_rings=total_rings, sel_rings=nsel)

    # --- build ----------------------------------------------------------
    t0 = time.time()
    config["quantize"]["high_fraction"] = row["hf"]  # quantizer partition
    if row["substrate"] == "none":
        model = build_norot(config)
        run_weights(model, config, row["weights"])
        ptqmod.configure_quantizers(model, config)
    else:
        # rv.build_eval runs GPTQ internally -> replicate it here so the
        # RTN patch wraps it. build_base keeps the 1/8 substrate geometry.
        model = rv.build_base(config, r1_off, ring_r1=ring_r1,
                              ring_size=RING)
        override = {}
        if chosen is not None:
            import ablation_acc_per_bit as abl
            for name, w in find_qlayers(model, layers=[ActQuantWrapper]).items():
                if not rv._is_alloc_target(name):
                    continue
                S = chosen.get(name, set())
                nr = w.module.weight.shape[1] // RING
                non_s = [r for r in range(nr) if r not in S]
                s_lst = [r for r in range(nr) if r in S]
                ring_order = non_s + s_lst
                order = torch.cat(
                    [torch.arange(r * RING, (r + 1) * RING)
                     for r in ring_order]).to(w.module.weight.device)
                W = w.module.weight.data
                w.module.weight.data = W[:, order].contiguous()
                w._column_order = order
                w.reorder_before_quant = True
                kh = len(s_lst) * RING
                w.module._edriven_khigh = int(kh)
                override[name] = int(kh)
            model._rd_khigh_override = override
            gptq_restore = abl.patch_gptq_edriven()
        else:
            gptq_restore = None
        try:
            run_weights(model, config, row["weights"])
        finally:
            if gptq_restore is not None:
                gptqmod.GPTQ.__init__ = gptq_restore
        ptqmod.configure_quantizers(model, config)
        from promix.quantize.kv_quant import setup_k_quant
        setup_k_quant(model, config, basis_path=config["paths"]["basis"],
                      rotation_path=config["paths"]["rotation"])

    frac, hi, tot = rv.avg_fp8_fraction(model)
    print(f"BUILT row={args.row} avg_fp8={frac:.4f} ({hi}/{tot} ch) "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["avg_fp8"] = frac

    # --- eval -----------------------------------------------------------
    t0 = time.time()
    ppl_wiki = rv.eval_ppl(model, config)
    print(f"RESULT {args.point} wikitext={ppl_wiki:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results["wikitext"] = ppl_wiki
    save()
    if args.c4:
        ppl_c4 = eval_c4_ppl(model, config, seed=0)
        print(f"RESULT {args.point} c4={ppl_c4:.4f}", flush=True)
        results["c4"] = ppl_c4
        save()

    if args.also_w4a16:
        n = disable_act_quant(model)
        print(f"W4A16 act-quant disabled on {n} wrappers", flush=True)
        ppl_w = rv.eval_ppl(model, config)
        results["w4a16_wikitext"] = ppl_w
        print(f"RESULT {args.point} w4a16_wikitext={ppl_w:.4f}", flush=True)
        save()
        if args.c4:
            ppl_wc = eval_c4_ppl(model, config, seed=0)
            results["w4a16_c4"] = ppl_wc
            print(f"RESULT {args.point} w4a16_c4={ppl_wc:.4f}", flush=True)
            save()

    del model
    cleanup_memory()
    save()
    print(f"BUILDUP_RESULT point={args.point} row={args.row} "
          f"wiki={ppl_wiki:.4f} c4={results.get('c4')} "
          f"w4a16={results.get('w4a16_wikitext')}", flush=True)
    print("BUILDUP_DONE", flush=True)


if __name__ == "__main__":
    main()
