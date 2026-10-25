"""Round-26 edge-fusion harness: GEMM epilogue bias+cast (lever A) and fused
residual-add + RMSNorm + quant/pack producer (lever B) on the 1B real model.

Phases
  --phase probe : per-lever numerics on the REAL activations of every layer.
                  A:  fused-epilogue GEMM (fp16 out + bias + bf16 round) vs the
                      current path (bf16 GEMM -> .to(fp16) -> + bias); counts
                      mismatched elements, must be 0 (bit-identical).
                  B:  fused add+RMSNorm+pack vs the reference chain
                      (r = residual + x; LlamaRMSNorm; qpf pack); reports the
                      norm-output max |diff| / code mismatch counts, and the
                      whole model's logit cosine with norm-pack on vs off.
  --phase perf  : full-model prefill (CUDA-graph, min-of-N) same-run A/B for
                  {baseline, +A, +B, +A+B} vs FP16, M = 2048/4096/8192.
  --phase ppl   : wikitext-2 PPL for the shipping config (+A+B) vs baseline.

Remote-only. GPU chosen by idle-waiter (util <= 8).
"""
import argparse, collections, json, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
import transformers

from promix.utils import DEV, cleanup_memory
from promix.eval.ptq import load_config, prepare_ptq_model, init_distributed_for_ptq_main_if_needed
from promix.quantize import full_network as fn
from promix.quantize.hadamard import matmul_hadU_cuda
from promix.eval.data import get_wikitext2
from promix.eval.full_network_realkernel import (
    load_kernels, _build_fp16_model, cosine, HIGH_FRACTION)
from promix.eval.prefill_graph import capture_forward, time_eager, time_graph
from promix.eval.hadamard_swiglu_fusion import load_hpk, load_qpfm, pick_free_gpu


def load_new():
    """round-26 extensions: lever A epilogue (fp16 out + bias) and lever B norm->pack."""
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    cd_bias = load(name="codesign_fused_bias_ext",
                   sources=[os.path.join(KERN, "codesign_fused_bias_ext.cu")],
                   extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    arn = load(name="add_rmsnorm_pack_ext",
               sources=[os.path.join(KERN, "add_rmsnorm_pack_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    return cd_bias, arn


def load_cd_bias():
    """round-26 lever A extension only (epilogue bias + output-cast fold).
    Bit-exact to `gemm -> .to(fp16) -> + bias` (probe: 0 / 400M mismatches)."""
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="codesign_fused_bias_ext",
                sources=[os.path.join(KERN, "codesign_fused_bias_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


def build(config, nv, mx, cd, qpf, qpfm, hpk, cd_bias, arn, gptq, norm_pack=False):
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = prepare_ptq_model(config, DEV, run_gptq=gptq)
    model.to(DEV); model.config.use_cache = False
    packed = fn.pack_all_targets(model, nv, mx, cd, high_fraction=HIGH_FRACTION, verbose=False)
    fn.wire_real_kernel(model, packed, nv, mx, cd, qpf=qpf, qpfm=qpfm, hpk=hpk,
                        cd_bias=cd_bias, arn=arn, norm_pack=norm_pack)
    fn.wire_fused_mlp(model)
    model.to(DEV).eval()
    for m in model.modules():
        for attr in ("_column_order", "new_column_order"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and t.device.type != "cuda":
                setattr(m, attr, t.to(DEV))
    return model, packed


def set_levers(model, a=None, b=None):
    """Toggle lever A (epilogue bias/cast) and/or lever B (fused norm->pack)."""
    na = nb = 0
    for _, _, w in fn.iter_target_wrappers(model):
        m = w.module
        if m.__class__.__name__ != "MixedQuantLinear":
            continue
        if a is not None:
            m.bias_in_epilogue = bool(a) and m._cd_bias is not None
            na += 1
        if b is not None:
            w.norm_pack = bool(b) and getattr(m, "_arn", None) is not None \
                and not w.online_full_had
            nb += 1
    return na, nb


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_probe(model, config, qpfm, nv, mx, cd, cd_bias, arn):
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    ids = get_wikitext2(seed=0, seqlen=2048, tokenizer=tok, eval_mode=True).input_ids[:, :2048].to(DEV)
    set_levers(model, b=False)

    # --- capture REAL (norm input, residual) pairs from every layer's norms ---
    caps = {i: {} for i in range(len(model.model.layers))}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        # capture the MLP's post-attention-norm input BEFORE the layer rewires it
        # (self.mlp is called with the normed tensor, so hook the pre-norm tensor
        # from the attention residual add instead: hook post_attention_layernorm's
        # input, which IS the residual sum).
        for nm in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, nm)
            def mk(i=i, nm=nm):
                def hook(m, args, kwargs):
                    xin = args[0] if args else kwargs.get("hidden_states")
                    res = kwargs.get("norm_residual")
                    if res is not None and nm not in caps[i]:
                        caps[i][nm] = (xin.detach().clone(), res.detach().clone())
                return hook
            hooks.append(mod.register_forward_pre_hook(mk(), with_kwargs=True))
    # force a layer-forward pass that passes norm_residual (installed by wiring)
    model(ids)
    for h in hooks:
        h.remove()

    # --- lever A probe: epilogue bias/cast vs current path, on REAL GEMM inputs ---
    mismatch = collections.defaultdict(int)
    total = 0
    exact_all = True
    for i in (0, 15):
        layer = model.model.layers[i]
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mql = getattr(layer.self_attn, nm).module
            x = torch.randn(2048, mql.K, device=DEV, dtype=torch.float16) * 0.5
            A = mql._pack_activation(x.contiguous())
            ref = mql._gemm(*A, 2048)                      # bf16 GEMM + cast (+ bias)
            got = mql._cd_bias.fused_gemm_prepacked_bias(
                A[0], A[1], mql.W_lo_packed, mql.physSFB_lo,
                A[2], A[3], mql.W_hi_bytes, mql.physSFB_hi, mql.bias,
                2048, mql.N, mql.K_low, mql.K_high)
            mm = int((ref.float() != got.float()).sum())
            mismatch[nm] += mm; total += ref.numel(); exact_all &= (mm == 0)
        for nm in ("gate_proj", "up_proj", "down_proj"):
            mql = getattr(layer.mlp, nm).module
            x = torch.randn(2048, mql.K, device=DEV, dtype=torch.float16) * 0.5
            A = mql._pack_activation(x.contiguous())
            ref = mql._gemm(*A, 2048)
            got = mql._cd_bias.fused_gemm_prepacked_bias(
                A[0], A[1], mql.W_lo_packed, mql.physSFB_lo,
                A[2], A[3], mql.W_hi_bytes, mql.physSFB_hi, mql.bias,
                2048, mql.N, mql.K_low, mql.K_high)
            mm = int((ref.float() != got.float()).sum())
            mismatch[nm] += mm; total += ref.numel(); exact_all &= (mm == 0)
    print(f"PROBE_A epilogue_bias_cast bit-identical={exact_all} mismatches={mismatch} of {total}",
          flush=True)

    # --- lever B probe: fused add+RMSNorm+pack vs the reference chain ----------
    # reference: r = residual + x (fp16); t = r.float(); var = t.pow(2).mean(-1);
    #            xn = (t*rsqrt(var+eps)).to(fp16); pack via qpfm
    eps = model.config.rms_norm_eps
    nb_bad = 0; nb_codes = 0; nb_tot = 0; max_absdiff = 0.0
    for i in (0, 1, 15):
        for nm, (xin, res) in caps[i].items():
            w = getattr(model.model.layers[i], nm)
            if nm == "input_layernorm":
                mql = w.module if False else model.model.layers[i].self_attn.q_proj.module
            else:
                mql = model.model.layers[i].mlp.gate_proj.module
            x2d = xin.reshape(-1, xin.shape[-1]).contiguous()
            r2d = res.reshape(-1, res.shape[-1]).contiguous()
            # Llama is pre-norm: the norm input IS the residual stream.
            ref_n = w.norm(x2d)                            # reference RMSNorm
            ref_pk = qpfm.quant_pack_float_mixed(ref_n.contiguous(), ref_n.shape[0],
                                                 mql.N, mql.K_low, mql.K_high)
            out = arn.add_rmsnorm_pack_mixed(x2d, None, mql.N, mql.K_low, mql.K_high, eps, True)
            clo, slo, chi, shi, dbg = out[1], out[2], out[3], out[4], out[5]
            rm = 0
            nd = (dbg - ref_n.float()).abs().max().item()
            max_absdiff = max(max_absdiff, nd)
            # codes: identical only if the norm output matched exactly
            c_lo = int((clo != ref_pk[0]).sum()); s_lo = int((slo != ref_pk[1]).sum())
            c_hi = int((chi != ref_pk[2]).sum()); s_hi = int((shi != ref_pk[3]).sum())
            bad = c_lo + s_lo + c_hi + s_hi
            nb_bad += bad
            nb_codes += clo.numel() + slo.numel() + chi.numel() + shi.numel()
            nb_tot += 1
            print(f"  PROBE_B {i:2d} {nm:24s} resid_mm={rm} norm_maxabs={nd:.3e} "
                  f"pack_mm={bad} (lo {c_lo}/{s_lo} hi {c_hi}/{s_hi})", flush=True)
    print(f"PROBE_B fused_addrmsnormpack: sites={nb_tot} pack_mismatch={nb_bad}/{nb_codes} "
          f"norm_maxabs={max_absdiff:.3e}", flush=True)
    # --- whole-model logit cosine: +B vs baseline ---
    set_levers(model, b=False)
    logits_ref = model(ids).logits.clone()
    set_levers(model, b=True)
    logits_b = model(ids).logits.clone()
    cos_b = cosine(logits_ref, logits_b)
    set_levers(model, a=False, b=True)
    logits_ab = model(ids).logits.clone()
    cos_ab = cosine(logits_ref, logits_ab)
    print(f"PROBE_B logit cos normpack={cos_b:.7f} normpack+bias_epi={cos_ab:.7f}", flush=True)
    return dict(probe_a_exact=exact_all, probe_a_mismatch=mismatch,
                probe_b_pack_mismatch=nb_bad, probe_b_norm_maxabs=max_absdiff,
                cos_normpack=cos_b, cos_both=cos_ab)


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_perf(model, fp16, config, Ms, args, out):
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    wiki = get_wikitext2(seed=0, seqlen=max(Ms), tokenizer=tok, eval_mode=True).input_ids
    rows = []
    for M in Ms:
        ids = wiki[:, :M].contiguous().to(DEV)
        t_fp16 = time_eager(lambda: fp16(ids).logits, iters=(10 if M <= 4096 else 6), warmup=4)
        rec = dict(M=M, fp16_ms=t_fp16)
        ref = None
        all_cfgs = (("base", False, False), ("A", True, False),
                    ("B", False, True), ("AB", True, True))
        want_b = bool(args.with_b)
        cfgs = tuple(c for c in all_cfgs if bool(c[2]) == want_b)
        for tag, a, b in cfgs:
            set_levers(model, a=a, b=b)
            logits = model(ids).logits.clone()
            if ref is None: ref = logits.clone()
            cos = cosine(ref, logits)
            g, sl = capture_forward(model, ids)
            g.replay(); torch.cuda.synchronize()
            cos_g = cosine(ref, sl.clone())
            t_g = time_graph(g, iters=(30 if M <= 4096 else 15), warmup=5)
            del g
            rec[tag] = dict(graph_ms=t_g, cos=cos_g, speedup_vs_fp16=t_fp16 / t_g)
            print(f"  M={M:5d} {tag:5s} graph={t_g:8.3f} ms  ({t_fp16/t_g:5.3f}x fp16) "
                  f"cos={cos_g:.7f}", flush=True)
            cleanup_memory()
        b = rec["base"]["graph_ms"]
        parts = []
        for tag in ("A", "B", "AB"):
            if tag in rec:
                parts.append(f"{tag}={b - rec[tag]['graph_ms']:+.3f}")
        print(f"  M={M:5d} saved ms vs base: " + "  ".join(parts), flush=True)
        rows.append(rec)
        del ids, ref
        cleanup_memory()
    out["perf"] = rows


# ---------------------------------------------------------------------------
# ppl
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_ppl(model, config, out):
    from promix.eval.evaluator import evaluate_ppl
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    model.seqlen = config['eval']['max_length']
    tl = get_wikitext2(seed=config['calibration'].get('seed', 0), seqlen=model.seqlen,
                       tokenizer=tok, eval_mode=True)
    bs = config['eval']['batch_size']
    res = {}
    for tag, a, b in (("base", False, False), ("AB", True, True)):
        set_levers(model, a=a, b=b)
        res[tag] = evaluate_ppl(model, tl, DEV, batch_size=bs)
        print(f"PPL_{tag} {res[tag]:.4f}", flush=True)
    out["ppl"] = res
    out["ppl_delta"] = res["AB"] - res["base"]


@torch.no_grad()
def run_probe_b(model, config, qpfm, arn):
    """Lever-B bit-exactness on the REAL norm inputs of every layer, plus the
    per-N SFA re-scatter used to share one pack across a group."""
    tok = transformers.AutoTokenizer.from_pretrained(config['model']['name'])
    ids = get_wikitext2(seed=0, seqlen=2048, tokenizer=tok, eval_mode=True).input_ids[:, :2048].to(DEV)
    set_levers(model, a=False, b=False)
    caps = {}
    hk = []
    for i, layer in enumerate(model.model.layers):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            def mk(i=i, nm=nm):
                def hook(m, args, kwargs):
                    if nm not in caps.setdefault(i, {}):
                        caps[i][nm] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
                return hook
            hk.append(getattr(layer, nm).register_forward_pre_hook(mk(), with_kwargs=True))
    model(ids)
    for h in hk:
        h.remove()

    eps = float(model.config.rms_norm_eps)
    # consumers per norm group, with their N
    groups = {"input_layernorm": ("q_proj", "k_proj", "v_proj"),
              "post_attention_layernorm": ("gate_proj", "up_proj")}
    tot = 0
    n_mis = 0
    n_norm_exact = 0
    n_norm_total = 0
    for i in sorted(caps):
        for nm, (x,) in caps[i].items():
            x2d = x.reshape(-1, x.shape[-1]).contiguous()
            M = x2d.shape[0]
            ref_n = model.model.layers[i].__getattr__(nm).norm(x2d) if hasattr(
                model.model.layers[i].__getattr__(nm), "norm") else getattr(
                model.model.layers[i], nm)(x2d)
            ref_n = ref_n.contiguous()
            # fused producer, no residual add (Llama is pre-norm)
            out = arn.add_rmsnorm_pack_mixed(x2d, None, 2048, 1792, 256, eps, True)
            clo, slo, chi, shi, dbg = out[1], out[2], out[3], out[4], out[5]
            n_norm_total += 1
            exact = bool((dbg == ref_n.float()).all().item())
            n_norm_exact += int(exact)
            for short in groups[nm]:
                mql = (model.model.layers[i].self_attn if short.endswith("_proj") and short in
                       ("q_proj", "k_proj", "v_proj") else model.model.layers[i].mlp)
                mql = getattr(mql, short).module
                N = int(mql.N)
                if short in ("q_proj", "k_proj", "v_proj") and N != int(mql.N):
                    pass
                ref_pk = qpfm.quant_pack_float_mixed(ref_n, M, N, mql.K_low, mql.K_high)
                got = arn.norm_pack_for_n(list(out[1:5]), N, mql.K_low, mql.K_high)
                mm = [int((a != b).sum()) for a, b in zip(got, ref_pk)]
                tot += 1
                bad = sum(mm)
                n_mis += bad
                print(f"  PROBE_B {i:2d} {nm:24s} {short:10s} N={N:5d} norm_exact={int(exact)} "
                      f"pack_mm={mm}", flush=True)
    print(f"PROBE_B2 sites={tot} pack_mismatch={n_mis} norms_exact={n_norm_exact}/{n_norm_total}",
          flush=True)
    out = dict(probe_b2_pack_mismatch=n_mis, probe_b2_norms_exact=n_norm_exact,
               probe_b2_norm_total=n_norm_total)
    # whole-model: lever B on vs off (cosine + nan check)
    set_levers(model, a=False, b=False)
    lo0 = model(ids).logits.clone()
    set_levers(model, a=False, b=True)
    lo1 = model(ids).logits.clone()
    out["cos_b"] = cosine(lo0, lo1)
    out["nan_b"] = bool(torch.isnan(lo1).any().item())
    print(f"PROBE_B2 logit cos={out['cos_b']:.7f} nan={out['nan_b']}", flush=True)
    set_levers(model, a=True, b=True)
    lo2 = model(ids).logits.clone()
    out["cos_ab"] = cosine(lo0, lo2)
    out["nan_ab"] = bool(torch.isnan(lo2).any().item())
    print(f"PROBE_B2 (A+B) logit cos={out['cos_ab']:.7f} nan={out['nan_ab']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "promix/configs/llama-3.2-1b-fp-perhead.yaml"))
    ap.add_argument("--phase", default="perf", choices=["probe", "probeb", "perf", "ppl", "all"])
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", dest="gptq", action="store_false")
    ap.add_argument("--with-b", action="store_true", default=False,
                    help="also measure the norm->pack lever (still under debug)")
    ap.add_argument("--out", default=os.path.join(KERN, "edge_fusion.json"))
    args = ap.parse_args()
    gpu = pick_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"EDGE-FUSION harness | phase={args.phase} gpu={gpu}", flush=True)
    config = load_config(args.config)
    nv, mx, cd, qpf = load_kernels()
    qpfm = load_qpfm()
    hpk = load_hpk()
    cd_bias, arn = load_new()
    print("kernels loaded", flush=True)
    model, packed = build(config, nv, mx, cd, qpf, qpfm, hpk, cd_bias, arn, args.gptq,
                          norm_pack=bool(args.with_b))
    fp16 = _build_fp16_model(config['model']['name'])
    out = {"config": args.config, "gpu": gpu, "ms": args.ms}
    if args.phase in ("probe", "all"):
        out.update(run_probe(model, config, qpfm, nv, mx, cd, cd_bias, arn))
    if args.phase in ("probeb", "all"):
        out.update(run_probe_b(model, config, qpfm, arn))
    if args.phase in ("perf", "all"):
        run_perf(model, fp16, config, [int(x) for x in args.ms.split(",")], args, out)
    if args.phase in ("ppl", "all"):
        run_ppl(model, config, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("EDGE_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
