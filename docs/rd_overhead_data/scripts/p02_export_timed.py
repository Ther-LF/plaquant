"""P0-2 sub-task A: timed ring512+measured export (mirrors promix.vllm_integ.export
main measured path exactly, with wall-clock per stage; skips PPL evals / T2 snapshot
which are gate artifacts, not offline preparation)."""
import argparse, json, os, sys, time
import torch

REPO = "/home/luofan/plaquant-densesweep"
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
sys.path.insert(0, os.path.join(REPO, "promix", "scripts"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

import transformers
from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config, init_distributed_for_ptq_main_if_needed
from promix.eval.data import get_wikitext2
from promix.quantize import quant_utils as qu
from promix.eval.full_network_realkernel import load_kernels
import promix.vllm_integ.export as ex
import rd_ron
import rd_validity as rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gptq_nsamples", type=int, required=True)
    ap.add_argument("--percdamp", type=float, default=None)
    ap.add_argument("--ring_size", type=int, default=512)
    ap.add_argument("--collect_tokens", type=int, default=4096)
    ap.add_argument("--calib_samples", type=int, default=16)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    t = {}
    config = load_config(args.config)
    config["_path"] = args.config
    config = rd_ron.resolve_paths(config)
    config["calibration"]["nsamples"] = args.gptq_nsamples
    if args.percdamp is not None:
        config["quantize"]["percdamp"] = args.percdamp
    config["quantize"]["high_fraction"] = 0.125
    ex.HIGH_FRACTION = 0.125
    budget_frac = 0.125
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(config["calibration"].get("seed", 0))

    s = time.time()
    nv, mx, cd, qpf = load_kernels()
    t["kernel_jit_s"] = time.time() - s   # NOT an offline-prep stage (one-time JIT)

    # ---- Stage 2: global deltaD collection + allocation ----------------------
    s = time.time()
    tok0 = transformers.AutoTokenizer.from_pretrained(config["model"]["name"])
    calib = get_wikitext2(nsamples=args.calib_samples,
                          seed=config["calibration"].get("seed", 0),
                          seqlen=2048, tokenizer=tok0, eval_mode=False)
    t["s2a_calib_load_s"] = time.time() - s
    s2 = time.time()
    cmodel = rv.build_base(config, r1_off=False, ring_r1=True,
                           ring_size=args.ring_size)
    cmodel.to(DEV)
    t["s2b_build_base_s"] = time.time() - s2
    s2 = time.time()
    stats = rv.collect_stats(cmodel, calib, max_tokens=args.collect_tokens,
                             ring=args.ring_size)
    del cmodel
    cleanup_memory()
    t["s2c_collect_stats_s"] = time.time() - s2
    s2 = time.time()
    total_rings = sum(stats[n]["nr"] for n in stats)
    budget = int(round(budget_frac * total_rings))
    chosen = rv.clamp_chosen(rv.measured_alloc(stats, budget, total_rings), stats)
    # fused-group unify (verbatim export.py)
    import re as _re
    groups = {}
    for nm in chosen:
        m = _re.search(r"layers\.(\d+)\..*\.(q|k|v|gate|up)_proj", nm)
        if m:
            gid = (m.group(1), "qkv" if m.group(2) in ("q", "k", "v") else "gateup")
            groups.setdefault(gid, []).append(nm)
    bumped = 0
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        tgt = max(len(chosen[nm]) for nm in members)
        tgt = min(tgt, min(stats[nm]["nr"] for nm in members) - 1)
        for nm in members:
            need = tgt - len(chosen[nm])
            if need <= 0:
                if len(chosen[nm]) > tgt:
                    dd = stats[nm]["deltaD"]
                    keep = sorted(chosen[nm], key=lambda r: -float(dd[r]))[:tgt]
                    bumped += len(chosen[nm]) - tgt
                    chosen[nm] = set(keep)
                continue
            dd = stats[nm]["deltaD"]
            rest = sorted((r for r in range(stats[nm]["nr"]) if r not in chosen[nm]),
                          key=lambda r: -float(dd[r]))
            chosen[nm] |= set(rest[:need])
            bumped += need
    nsel = sum(len(v) for v in chosen.values())
    t["s2d_alloc_s"] = time.time() - s2
    t["stage2_deltaD_alloc_s"] = time.time() - s
    print(f"ALLOC total_rings={total_rings} sel={nsel} "
          f"(frac={nsel/total_rings:.4f}, fused-unify moved {bumped} rings)", flush=True)

    # ---- Stage 3: frozen-scale GPTQ + two-level scale + weight/act-MSE -------
    s = time.time()
    model = rv.build_eval(config, r1_off=False, chosen=chosen, ring_r1=True,
                          ring_size=args.ring_size)
    model.to(DEV)
    torch.cuda.synchronize()
    t["stage3_gptq_scales_s"] = time.time() - s

    # ---- Stage 4: weight pre-pack + checkpoint export -------------------------
    s = time.time()
    packed_sg, meta, linear_sf = ex.pack_all_targets_shared_g(
        model, nv, mx, cd, verbose=False)
    torch.cuda.synchronize()
    t["s4a_pack_s"] = time.time() - s
    s4 = time.time()
    provenance = {"substrate": "ring", "ring_size": args.ring_size,
                  "alloc": "measured", "budget_frac": budget_frac,
                  "gptq_nsamples": config["calibration"].get("nsamples"),
                  "percdamp": config["quantize"].get("percdamp"),
                  "weight_mse_steps": int(qu._MSE_STEPS),
                  "act_mse_steps_up": int(os.environ.get("PROMIX_ACT_MSE_STEPS_UP", "0")),
                  "p02_timed_run": True}
    ex.export_checkpoint(model, packed_sg, meta, linear_sf, args.out, config,
                         ref_ppl_sharedg=None, ref_ppl_perproj=None,
                         provenance=provenance)
    t["s4b_write_ckpt_s"] = time.time() - s4
    t["stage4_pack_export_s"] = time.time() - s

    t["total_offline_s234"] = (t["stage2_deltaD_alloc_s"]
                               + t["stage3_gptq_scales_s"]
                               + t["stage4_pack_export_s"])
    t["_meta"] = {"config": args.config, "model": config["model"]["name"],
                  "gptq_nsamples": args.gptq_nsamples,
                  "percdamp": config["quantize"].get("percdamp"),
                  "weight_mse_steps": int(qu._MSE_STEPS),
                  "act_mse_steps_up": os.environ.get("PROMIX_ACT_MSE_STEPS_UP"),
                  "ring_size": args.ring_size, "budget_frac": budget_frac,
                  "alloc_frac": nsel / total_rings,
                  "gpu": torch.cuda.get_device_name(0)}
    with open(args.json_out, "w") as f:
        json.dump(t, f, indent=2)
    print("P02_TIMING " + json.dumps(t), flush=True)
    print("P02_EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
