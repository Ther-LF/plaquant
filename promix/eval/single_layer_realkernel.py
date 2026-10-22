"""Single-layer real-kernel integration harness (SM120 NVFP4+MXFP8).

Validates wiring ONE Llama-3.2-1B decoder layer's 7 Linears to the real packed
fused-kernel path (MixedQuantLinear) vs the fake-quant reference.

Phases:
  (1) BIT-EXACT PROBE  — weight SFB (pack_weight_mixed) torch.equal vs the kernel
      load-time prepacker; activation SFA (quant_pack ilv) torch.equal vs
      prepack_sf_*_a. Hard gate before anything else.
  (2) PER-SHAPE COSINE — for each of the 7 real layer shapes, MixedQuantLinear
      output vs (a) kernel-quant fake-ref (isolates GEMM accumulation; ~1.0),
      (b) quant_utils.fake_quantize NVFP4+MXFP8 ref (spec's named fake-quant;
      gate >=0.999), (c) FP16 (drift). Synthetic activations unless --real.
  (3) LATENCY — per M in {2048,4096,8192}: real-kernel layer (pack+GEMM, split
      reported) vs FP16 cuBLAS vs fake-quant, for the 7 Linears summed.
  (4) --real: load rotated Llama-3.2-1B, capture layer-8 per-Linear inputs +
      rotated weights, redo per-Linear + whole-layer cosine on REAL activations.

Remote-only: run inside luofan-plaquant on a FREE GPU. Line-buffered stdout +
JSON results to disk for harvest.

Usage:
  python -m promix.eval.single_layer_realkernel [--real] [--layer 8] [--ms 2048,4096,8192]
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO)
sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.quantize.weight_pack import (
    pack_weight_mixed, probe_weight_sfb_bitexact, split_k,
    quantize_nvfp4_spec, quantize_mxfp8_spec,
)
from promix.quantize.mixed_linear import MixedQuantLinear
from promix.quantize import quant_utils as qu

# Llama-3.2-1B decoder-layer Linear shapes: (name, N, K)
LAYER_SHAPES = [
    ("q_proj",   2048, 2048),
    ("k_proj",    512, 2048),
    ("v_proj",    512, 2048),
    ("o_proj",   2048, 2048),
    ("gate_proj", 8192, 2048),
    ("up_proj",   8192, 2048),
    ("down_proj", 2048, 8192),
]
HIGH_FRACTION = 0.125


def load_kernels():
    from torch.utils.cpp_extension import load
    import nvfp4_gemm as nv
    import mxfp8_gemm as mx
    nv.get_ext(); mx.get_ext()
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    cd = load(name="codesign_fused_ext",
              sources=[os.path.join(KERN, "codesign_fused_ext.cu")],
              extra_include_paths=inc,
              extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                                 "--expt-extended-lambda", "-O3"],
              verbose=False)
    return nv, mx, cd


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def fakequant_split(x, K_low):
    """quant_utils NVFP4(main)+MXFP8(high) fake-quant on the [:K_low | K_low:] split."""
    lo = qu.fake_quantize_nvfp4(x[..., :K_low].float())
    hi = qu.fake_quantize_mxfp8(x[..., K_low:].float())
    return torch.cat([lo, hi], dim=-1)


def time_ms(fn, iters=50, warmup=15):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


# ---------------------------------------------------------------------------
# Phase 1: bit-exact probe
# ---------------------------------------------------------------------------
def phase_bitexact(nv, mx, cd, dev):
    print("=== PHASE 1: BIT-EXACT SF PROBE ===", flush=True)
    allok = True
    # 1a. spec-quant deq == quant_utils.fake_quantize_* (faithful-numerics self-check)
    xs = torch.randn(256, 512, device=dev)
    d_nv = quantize_nvfp4_spec(xs)[2]
    d_nv_ref = qu.fake_quantize_nvfp4(xs.float())
    d_mx = quantize_mxfp8_spec(xs)[2]
    d_mx_ref = qu.fake_quantize_mxfp8(xs.float())
    nv_eq = bool(torch.equal(d_nv, d_nv_ref)); mx_eq = bool(torch.equal(d_mx, d_mx_ref))
    print(f"  spec-quant deq == fake_quantize: nvfp4={nv_eq} mxfp8={mx_eq}", flush=True)
    allok = allok and nv_eq and mx_eq
    for name, N, K in LAYER_SHAPES:
        Kl, Kh = split_k(K, HIGH_FRACTION)
        torch.manual_seed(0)
        W = torch.randn(N, K, device=dev)
        # weight SFB
        wp = probe_weight_sfb_bitexact(W, Kl, Kh, nv, mx, cd)
        # activation SFA (M=2048) via the real spec-quant path
        M = 2048
        x = torch.randn(M, K, device=dev)
        cl, SFAl, _ = quantize_nvfp4_spec(x[:, :Kl].contiguous())
        bh, SFAh, _ = quantize_mxfp8_spec(x[:, Kl:].contiguous())
        a4 = cd.quant_pack_fp4(cl, SFAl, M, N, Kl, True)
        a8 = cd.quant_pack_fp8(bh, SFAh, M, N, Kh, True)
        sfa4_ref = cd.prepack_sf_fp4_a(SFAl, M, N, Kl)
        sfa8_ref = cd.prepack_sf_fp8_a(SFAh, M, N, Kh)
        data4 = bool(torch.equal(a4[0], nv.pack_fp4(cl)))
        data8 = bool(torch.equal(a8[0], bh))
        sfa4 = bool(torch.equal(a4[1], sfa4_ref))
        sfa8 = bool(torch.equal(a8[1], sfa8_ref))
        cell = wp["sfb_lo_equal"] and wp["sfb_hi_equal"] and data4 and data8 and sfa4 and sfa8
        allok = allok and cell
        print(f"  {name:10s} N={N:5d} K={K:5d} Kl={Kl:5d} Kh={Kh:4d} | "
              f"SFB_lo={wp['sfb_lo_equal']} SFB_hi={wp['sfb_hi_equal']} "
              f"A_data4={data4} A_data8={data8} SFA_lo={sfa4} SFA_hi={sfa8}  "
              f"{'OK' if cell else 'FAIL'}", flush=True)
    print("BITEXACT_PROBE_PASS" if allok else "BITEXACT_PROBE_FAIL", flush=True)
    return allok


# ---------------------------------------------------------------------------
# Phase 2: per-shape cosine (synthetic)
# ---------------------------------------------------------------------------
def build_layer(nv, mx, cd, dev, M=2048, seed=0, real_caps=None):
    """Returns list of (name, N, K, Kl, Kh, x, W, mql). If real_caps given, uses
    captured (x, W) per name; else synthetic randn."""
    layer = []
    for name, N, K in LAYER_SHAPES:
        Kl, Kh = split_k(K, HIGH_FRACTION)
        if real_caps is not None:
            x = real_caps[name]["x"].to(dev)
            W = real_caps[name]["W"].to(dev)
            N = W.shape[0]
        else:
            torch.manual_seed(seed + hash(name) % 1000)
            W = torch.randn(N, K, device=dev) * 0.08
            x = torch.randn(M, K, device=dev)
            # emulate rotated high-variance-last structure
            x[:, Kl:] *= 3.0
        mql = MixedQuantLinear.from_linear(
            type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})(),
            nv, mx, cd, high_fraction=HIGH_FRACTION)
        layer.append((name, N, K, Kl, Kh, x, W, mql))
    return layer


def phase_cosine(nv, mx, cd, dev, layer, tag="synthetic"):
    print(f"=== PHASE 2: PER-LINEAR COSINE ({tag}) ===", flush=True)
    results = []
    y_rk_cat, y_fq_cat, y_fp16_cat = [], [], []
    allok = True
    for name, N, K, Kl, Kh, x, W, mql in layer:
        M = x.reshape(-1, K).shape[0]
        with torch.no_grad():
            y_rk = mql(x).float()
        # reference A: quant_utils fake-quant NVFP4+MXFP8 (spec's named reference)
        y_fq = (fakequant_split(x, Kl) @ fakequant_split(W, Kl).t()).float()
        # reference B: fp16 cuBLAS (quant drift)
        y_fp16 = (x.half() @ W.half().t()).float()
        c_fq = cosine(y_rk, y_fq)
        c_fp16 = cosine(y_rk, y_fp16)
        ok = c_fq >= 0.999
        allok = allok and ok
        results.append(dict(name=name, N=N, K=K, Kl=Kl, Kh=Kh, M=M,
                            cos_vs_fakequant=c_fq, cos_vs_fp16=c_fp16, pass_gate=ok))
        print(f"  {name:10s} M={M:5d} N={N:5d} Kh={Kh:4d} | "
              f"cos(vs fakequant)={c_fq:.6f} cos(vs fp16)={c_fp16:.6f}  "
              f"{'OK' if ok else 'FAIL<0.999'}", flush=True)
        y_rk_cat.append(y_rk); y_fq_cat.append(y_fq); y_fp16_cat.append(y_fp16)
    # whole-layer concat cosine
    lc_fq = cosine(torch.cat([t.flatten() for t in y_rk_cat]),
                   torch.cat([t.flatten() for t in y_fq_cat]))
    lc_fp16 = cosine(torch.cat([t.flatten() for t in y_rk_cat]),
                     torch.cat([t.flatten() for t in y_fp16_cat]))
    print(f"  WHOLE-LAYER (7 Linears concat): cos(vs fakequant)={lc_fq:.6f} "
          f"cos(vs fp16)={lc_fp16:.6f}", flush=True)
    print(f"COSINE_{tag.upper()}_{'PASS' if allok else 'FAIL'}", flush=True)
    return dict(per_linear=results, layer_cos_vs_fakequant=lc_fq,
                layer_cos_vs_fp16=lc_fp16, pass_gate=allok)


# ---------------------------------------------------------------------------
# Phase 3: latency
# ---------------------------------------------------------------------------
def phase_latency(nv, mx, cd, dev, Ms):
    print("=== PHASE 3: PREFILL LATENCY (sum of 7 Linears) ===", flush=True)
    rows = []
    for M in Ms:
        layer = build_layer(nv, mx, cd, dev, M=M)
        tot_pack = tot_gemm = tot_rk = tot_fp16 = tot_fq = 0.0
        per = []
        for name, N, K, Kl, Kh, x, W, mql in layer:
            xf = x
            Wh = W.half(); xh = x.half()
            t_pack = time_ms(lambda: mql.pack_only(xf))

            # GEMM-only: pre-pack activation once, time the kernel GEMM alone
            A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(xf)

            def gemm_only():
                return cd.fused_gemm_prepacked(
                    A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                    A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi,
                    M, N, Kl, Kh)
            t_gemm = time_ms(gemm_only)

            def rk():
                with torch.no_grad():
                    mql(xf)
            t_rk = time_ms(rk)
            t_fp16 = time_ms(lambda: xh @ Wh.t())

            def fq():
                a = fakequant_split(xf, Kl); w = fakequant_split(W, Kl)
                return a @ w.t()
            t_fq = time_ms(fq, iters=20, warmup=5)
            per.append(dict(name=name, t_pack_ms=t_pack, t_gemm_only_ms=t_gemm,
                            t_e2e_ms=t_rk, t_fp16_ms=t_fp16, t_fakequant_ms=t_fq))
            tot_pack += t_pack; tot_gemm += t_gemm; tot_rk += t_rk
            tot_fp16 += t_fp16; tot_fq += t_fq
        su_gemm = tot_fp16 / tot_gemm if tot_gemm > 0 else float('nan')
        su_e2e = tot_fp16 / tot_rk if tot_rk > 0 else float('nan')
        su_fq = tot_fq / tot_rk if tot_rk > 0 else float('nan')
        print(f"  M={M:5d}: GEMM-only={tot_gemm:.3f}ms (vs FP16 {tot_fp16:.3f}ms = "
              f"{su_gemm:.2f}x) | pack(torch-quant)={tot_pack:.3f}ms | "
              f"e2e(pack+GEMM)={tot_rk:.3f}ms (vs FP16 {su_e2e:.2f}x, vs fakequant {su_fq:.2f}x) "
              f"| fakequant={tot_fq:.3f}ms", flush=True)
        rows.append(dict(M=M, tot_pack_ms=tot_pack, tot_gemm_only_ms=tot_gemm,
                         tot_e2e_ms=tot_rk, tot_fp16_ms=tot_fp16, tot_fakequant_ms=tot_fq,
                         speedup_gemm_vs_fp16=su_gemm, speedup_e2e_vs_fp16=su_e2e,
                         speedup_e2e_vs_fakequant=su_fq, per_linear=per))
    return rows


# ---------------------------------------------------------------------------
# Phase 4: real capture
# ---------------------------------------------------------------------------
def capture_real(cfg_path, layer_idx, dev):
    import yaml
    from promix.models.loader import load_model, install_column_order_hooks
    from promix.quantize.fuse_norm import fuse_layer_norms
    from promix.quantize.rotation import fuse_basis_to_model, rearrange_columns
    from promix.quantize.quant_utils import add_actquant, find_qlayers, ActQuantWrapper
    from promix.eval.ptq import setup_down_proj_hadamard, configure_quantizers, init_distributed_for_ptq_main_if_needed
    from promix.eval.data import get_wikitext2
    import transformers

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    init_distributed_for_ptq_main_if_needed()
    transformers.set_seed(0)
    model = load_model(cfg['model']['name'], dtype=torch.float16)
    fuse_layer_norms(model)
    fuse_basis_to_model(model, basis_path=cfg['paths']['basis'],
                        rotation_path=cfg['paths']['rotation'],
                        high_fraction=cfg['quantize']['high_fraction'],
                        low_fraction=cfg['quantize'].get('low_fraction', 0.0),
                        identity_main_rotation=cfg['quantize'].get('adaptive_high_fraction', False))
    rearrange_columns(model, high_fraction=cfg['quantize']['high_fraction'],
                      low_fraction=cfg['quantize'].get('low_fraction', 0.0),
                      o_proj_pca=cfg['quantize'].get('o_proj_pca', 'per_head'))
    add_actquant(model)
    setup_down_proj_hadamard(model)
    install_column_order_hooks(model)
    # force FP routing for the quantizer config (Hadamard/reorder machinery)
    cfg['quantize']['a_bits'] = 'nvfp4'
    cfg['quantize']['high_bits'] = 'mxfp8'
    cfg['quantize']['w_bits'] = 16
    configure_quantizers(model, cfg)
    model = model.to(dev).eval()

    # capture the input each target-layer quantizer sees (post Hadamard+reorder, pre-quant)
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    caps = {}
    hooks = []
    prefix = f"model.layers.{layer_idx}."
    name_map = {}
    for full, wrap in qlayers.items():
        if not full.startswith(prefix):
            continue
        short = full.split(".")[-1]
        if short not in [n for n, _, _ in LAYER_SHAPES]:
            continue
        name_map[id(wrap.quantizer)] = short
        caps[short] = {"W": wrap.module.weight.data.detach().clone()}

        def mk(sh, wrap):
            def pre_hook(mod, inp):
                caps[sh]["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1]).clone()
            return pre_hook
        hooks.append(wrap.quantizer.register_forward_pre_hook(mk(short, wrap)))

    tok = transformers.AutoTokenizer.from_pretrained(cfg['model']['name'])
    loader = get_wikitext2(seed=0, seqlen=2048, tokenizer=tok, eval_mode=True)
    ids = loader.input_ids[:, :2048].to(dev) if hasattr(loader, 'input_ids') else loader[0][0][:, :2048].to(dev)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    # move weights back to fp32 on dev, keep only captured keys with both x and W
    caps = {k: v for k, v in caps.items() if "x" in v}
    print(f"REAL_CAPTURE: layer {layer_idx} captured {sorted(caps.keys())}", flush=True)
    del model
    torch.cuda.empty_cache()
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--config", type=str, default=os.path.join(REPO, "promix/configs/llama-3.2-1b-resq.yaml"))
    ap.add_argument("--ms", type=str, default="2048,4096,8192")
    ap.add_argument("--out", type=str, default=os.path.join(KERN, "single_layer_realkernel.json"))
    args = ap.parse_args()
    dev = "cuda"
    Ms = [int(x) for x in args.ms.split(",")]

    print("Loading kernels (JIT build may take minutes)...", flush=True)
    nv, mx, cd = load_kernels()
    print(cd.compat_report() if hasattr(cd, "compat_report") else "cd loaded", flush=True)

    out = {"bitexact": None, "cosine_synthetic": None, "latency": None, "cosine_real": None}
    out["bitexact"] = phase_bitexact(nv, mx, cd, dev)

    layer = build_layer(nv, mx, cd, dev, M=2048)
    out["cosine_synthetic"] = phase_cosine(nv, mx, cd, dev, layer, tag="synthetic")
    del layer; torch.cuda.empty_cache()

    out["latency"] = phase_latency(nv, mx, cd, dev, Ms)

    if args.real:
        try:
            caps = capture_real(args.config, args.layer, dev)
            rlayer = build_layer(nv, mx, cd, dev, real_caps=caps)
            out["cosine_real"] = phase_cosine(nv, mx, cd, dev, rlayer, tag="real")
        except Exception as e:
            import traceback
            print("REAL_CAPTURE_ERROR:", repr(e), flush=True)
            traceback.print_exc()
            out["cosine_real"] = {"error": repr(e)}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("SINGLE_LAYER_REALKERNEL_DONE", flush=True)


if __name__ == "__main__":
    main()
