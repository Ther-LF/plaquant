"""M33-ext (Motivation 3.3): naive 2-launch mixed-precision GEMM latency sweep
over the high-precision fraction r in [0, 1], plus white-box breakdown of the
four runtime system taxes that Section 4.3's fusion eliminates (1-to-1):

  1. online_dense_rot_ms      — online dense basis change X @ (K x K) BF16 mm
                                (eliminated by model-level shared offline absorption)
  2. online_gather_ms         — per-channel gather X.index_select(1, perm)
                                (eliminated by fused activation prep)
  3. standalone_quant_pack_ms — two independent NVFP4(K_low) + MXFP8(K_high)
                                quant+pack passes (eliminated by single-pass
                                dual-format fused pack)
  4. online_hadamard_ms       — down_proj-only standalone online Hadamard
                                (eliminated by Hadamard+pack fusion, Step 4)

Measured paths are STANDARD/naive only (bf16 / pure NVFP4 / pure MXFP8 /
naive 2-launch via torch._scaled_mm_v2) — the AdaMix fused kernel is
deliberately NOT measured here (anti-circularity, see task spec
docs/paper_motivation_3_3_ext_experiments.md).

Timing protocol, quantizers, swizzle layouts and the torchmixed2 implementation
(explicit `lo + hi` add) are imported VERBATIM from promix/eval/gemm_h2h_torchfp4.py,
the script that produced docs/rd_buildup_data/h2h_naive.json, so the r = 0.125
point is directly comparable (Gate 1).

Writes docs/motivation_3_3_ext/naive_rsweep.{json,csv} + manifest.json.
"""
import os, sys, json, csv, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.eval.gemm_h2h_torchfp4 import (  # noqa: E402
    quant_nvfp4_ref, to_blocked, native_fp4_mm, native_mxfp8_mm,
    bench_cell, shapes_for, MODELS, native_fp4_validate)
from promix.eval.full_network_realkernel import act_mse_cflags, CUTLASS  # noqa: E402
from promix.quantize.hadamard import get_hadK, matmul_hadU_cuda  # noqa: E402


def load_qpf():
    """qpf (quant_pack_float_ext) ONLY — same include paths/flags as
    full_network_realkernel.load_kernels(), but deliberately does NOT build or
    load the fused GEMM (codesign_fused_ext): this benchmark measures naive
    paths only (anti-circularity). PROMIX_CUTLASS env overrides the CUTLASS
    checkout (worktrees may have an uninitialized submodule)."""
    from torch.utils.cpp_extension import load
    cut = os.environ.get("PROMIX_CUTLASS", CUTLASS)
    inc = [KERN, os.path.join(cut, "include"),
           os.path.join(cut, "tools/util/include"),
           os.path.join(cut, "examples/common")]
    flags = ["-std=c++20" if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 13) else "-std=c++17",
             "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name="quant_pack_float_ext",
                sources=[os.path.join(KERN, "quant_pack_float_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags + act_mse_cflags(),
                verbose=False)

DEV = "cuda"
R_GRID = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]

FIELDS = ["model", "shape", "M", "N", "K", "K_low", "K_high", "r_target",
          "r_actual", "bf16_ms", "torchfp4_ms", "torchfp8_ms",
          "torchmixed2_ms", "torchmixed2_min_ms", "part_fp4_ms",
          "part_fp8_ms", "part_add_ms", "online_dense_rot_ms",
          "online_gather_ms", "standalone_quant_pack_ms",
          "online_hadamard_ms", "ideal_linear_ms",
          "slowdown_vs_fp4", "slowdown_vs_ideal", "slowdown_vs_fp8"]


def k_split_r(K, r):
    """K_high = round(r*K/32)*32 (endpoints exact), K_low = K - K_high."""
    if r <= 0.0:
        kh = 0
    elif r >= 1.0:
        kh = K
    else:
        kh = int(round(r * K / 32.0)) * 32
    kl = K - kh
    assert kh % 32 == 0 and kl % 16 == 0, (K, r, kl, kh)
    return kl, kh


def e8m0_ones(rows, cols):
    return to_blocked(torch.ones(rows, cols, device=DEV)).to(torch.float8_e8m0fnu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="8B,3B")
    ap.add_argument("--ms", default="8192,128")
    ap.add_argument("--outdir", default=os.path.join(REPO, "docs", "motivation_3_3_ext"))
    args = ap.parse_args()
    ms_list = [int(x) for x in args.ms.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    qpf = load_qpf()   # quant_pack_float_ext only; fused GEMM NOT used
    print("qpf kernel loaded", flush=True)
    exact, cos = native_fp4_validate()
    print(f"NATIVE_FP4_VALIDATION exact={exact} cosine={cos:.6f}", flush=True)

    rows = []
    for mdl in args.models.split(","):
        c = MODELS[mdl]
        for name, N, K in shapes_for(c["H"], c["I"], c["nheads"], c["nkv"]):
            is_down = (name == "down")
            torch.manual_seed(0)
            W = torch.randn(N, K, device=DEV) * 0.08
            Wbf = W.to(torch.bfloat16).contiguous()

            # full-K NVFP4 weight (r=0 path), same recipe as h2h_naive
            Wp, Wsc, _ = quant_nvfp4_ref(W)
            Bfp4_t = Wp.view(torch.float4_e2m1fn_x2).t()
            sB4 = to_blocked(Wsc).to(torch.float8_e4m3fn)
            # full-K MXFP8 weight (r=1 path), unit e8m0 scales (same as h2h hi-seg)
            W8_t = W.to(torch.float8_e4m3fn).t()
            sB8 = e8m0_ones(N, K // 32)
            # dense rotation matrix (timing-only; orthogonality irrelevant to mm cost)
            Rrot = None if is_down else torch.randn(K, K, device=DEV, dtype=torch.bfloat16)
            perm = torch.randperm(K, device=DEV)
            had_K, Khad = (None, 1)
            if is_down:
                had_K, Khad = get_hadK(K)
                if had_K is not None:
                    had_K = had_K.to(DEV)

            for M in ms_list:
                torch.manual_seed(1)
                x = torch.randn(M, K, device=DEV)
                x[:, k_split_r(K, 0.125)[0]:] *= 3.0  # same hi-var boost as h2h (r=1/8 split)
                xbf = x.to(torch.bfloat16).contiguous()

                # ---- r-independent cell: bf16 / pure fp4 / pure fp8 / taxes ----
                Ap, Asc, _ = quant_nvfp4_ref(x)
                Afp4 = Ap.view(torch.float4_e2m1fn_x2)
                sA4 = to_blocked(Asc).to(torch.float8_e4m3fn)
                A8 = x.to(torch.float8_e4m3fn)
                sA8 = e8m0_ones(M, K // 32)

                def bf16mm():
                    return xbf @ Wbf.t()

                def torchfp4():
                    return native_fp4_mm(Afp4, Bfp4_t, sA4, sB4)

                def torchfp8():
                    return native_mxfp8_mm(A8, W8_t, sA8, sB8)

                def gather():
                    return xbf.index_select(1, perm)

                fns = dict(bf16=bf16mm, fp4=torchfp4, fp8=torchfp8, gather=gather)
                if is_down:
                    fns["had"] = lambda: matmul_hadU_cuda(xbf, had_K, Khad, fold_scale=True)
                else:
                    fns["rot"] = lambda: xbf @ Rrot
                # endpoint standalone quant+pack (r=0: fp4 full K; r=1: fp8 full K)
                fns["qp0"] = lambda: qpf.quant_pack_float_fp4(xbf, M, N, K, K, 0)
                fns["qp1"] = lambda: qpf.quant_pack_float_fp8(xbf, M, N, K, K, 0)
                cc = bench_cell(fns, spin_fn=bf16mm)
                t_bf, _ = cc["bf16"]
                t4, min4 = cc["fp4"]
                t8, min8 = cc["fp8"]
                t_gather, _ = cc["gather"]
                t_rot = cc["rot"][0] if "rot" in cc else None
                t_had = cc["had"][0] if "had" in cc else None

                def mkrow(r_t, kl, kh, tmix, tmix_min, p4, p8, padd, qp):
                    r_a = kh / K
                    ideal = (1.0 - r_a) * t4 + r_a * t8
                    return dict(model=mdl, shape=name, M=M, N=N, K=K,
                                K_low=kl, K_high=kh, r_target=r_t, r_actual=r_a,
                                bf16_ms=t_bf, torchfp4_ms=t4, torchfp8_ms=t8,
                                torchmixed2_ms=tmix, torchmixed2_min_ms=tmix_min,
                                part_fp4_ms=p4, part_fp8_ms=p8, part_add_ms=padd,
                                online_dense_rot_ms=t_rot, online_gather_ms=t_gather,
                                standalone_quant_pack_ms=qp, online_hadamard_ms=t_had,
                                ideal_linear_ms=ideal,
                                slowdown_vs_fp4=tmix / t4,
                                slowdown_vs_ideal=tmix / ideal,
                                slowdown_vs_fp8=tmix / t8)

                for r_t in R_GRID:
                    kl, kh = k_split_r(K, r_t)
                    if kh == 0:      # endpoint: torchmixed2 := pure NVFP4
                        rows.append(mkrow(r_t, kl, kh, t4, min4, t4, 0.0, 0.0, cc["qp0"][0]))
                    elif kl == 0:    # endpoint: torchmixed2 := pure MXFP8
                        rows.append(mkrow(r_t, kl, kh, t8, min8, 0.0, t8, 0.0, cc["qp1"][0]))
                    else:
                        # per-r weights: NVFP4 on K_low + MXFP8 on K_high (h2h recipe)
                        Wp_lo, Wsc_lo, _ = quant_nvfp4_ref(W[:, :kl].contiguous())
                        Blo_t = Wp_lo.view(torch.float4_e2m1fn_x2).t()
                        sB_lo = to_blocked(Wsc_lo).to(torch.float8_e4m3fn)
                        Whi8_t = W[:, kl:].contiguous().to(torch.float8_e4m3fn).t()
                        sB_hi = e8m0_ones(N, kh // 32)
                        # per-r activations (prequantized, GEMM-only timing — h2h口径)
                        Ap_lo, Asc_lo, _ = quant_nvfp4_ref(x[:, :kl].contiguous())
                        Alo = Ap_lo.view(torch.float4_e2m1fn_x2)
                        sA_lo = to_blocked(Asc_lo).to(torch.float8_e4m3fn)
                        Ahi8 = x[:, kl:].contiguous().to(torch.float8_e4m3fn)
                        sA_hi = e8m0_ones(M, kh // 32)
                        Ylo = native_fp4_mm(Alo, Blo_t, sA_lo, sB_lo)
                        Yhi = native_mxfp8_mm(Ahi8, Whi8_t, sA_hi, sB_hi)

                        def mix2():
                            lo = native_fp4_mm(Alo, Blo_t, sA_lo, sB_lo)
                            hi = native_mxfp8_mm(Ahi8, Whi8_t, sA_hi, sB_hi)
                            return lo + hi

                        def p4():
                            return native_fp4_mm(Alo, Blo_t, sA_lo, sB_lo)

                        def p8():
                            return native_mxfp8_mm(Ahi8, Whi8_t, sA_hi, sB_hi)

                        def padd():
                            return Ylo + Yhi

                        def qpack():  # two INDEPENDENT standalone quant+pack passes
                            a = qpf.quant_pack_float_fp4(xbf, M, N, kl, K, 0)
                            b = qpf.quant_pack_float_fp8(xbf, M, N, kh, K, kl)
                            return a, b

                        cr = bench_cell(dict(mix2=mix2, p4=p4, p8=p8, padd=padd,
                                             qpack=qpack), spin_fn=bf16mm)
                        rows.append(mkrow(r_t, kl, kh, cr["mix2"][0], cr["mix2"][1],
                                          cr["p4"][0], cr["p8"][0], cr["padd"][0],
                                          cr["qpack"][0]))
                        del Blo_t, sB_lo, Whi8_t, sB_hi, Alo, sA_lo, Ahi8, sA_hi, Ylo, Yhi
                        torch.cuda.empty_cache()
                    rr = rows[-1]
                    print(f"{mdl} {name:8s} M={M:5d} r={r_t:6.4f} Kh={kh:5d}: "
                          f"mix2 {rr['torchmixed2_ms']:.4f}ms | fp4 {t4:.4f} fp8 {t8:.4f} "
                          f"bf16 {t_bf:.4f} ideal {rr['ideal_linear_ms']:.4f} | "
                          f"x_fp4 {rr['slowdown_vs_fp4']:.2f} x_ideal {rr['slowdown_vs_ideal']:.2f} "
                          f"x_fp8 {rr['slowdown_vs_fp8']:.2f} | qp {rr['standalone_quant_pack_ms']:.4f} "
                          f"gth {t_gather:.4f} rot {t_rot if t_rot else 0:.4f} "
                          f"had {t_had if t_had else 0:.4f}", flush=True)
                del x, xbf, Ap, Asc, Afp4, sA4, A8, sA8
                torch.cuda.empty_cache()
            del W, Wbf, Wp, Wsc, Bfp4_t, sB4, W8_t, sB8, Rrot
            torch.cuda.empty_cache()

    out_json = os.path.join(args.outdir, "naive_rsweep.json")
    with open(out_json, "w") as f:
        json.dump(dict(native_fp4_exact=exact, native_fp4_cosine=cos, rows=rows), f, indent=2)
    with open(os.path.join(args.outdir, "naive_rsweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in FIELDS})

    import subprocess
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    manifest = dict(
        task="M33-ext naive 2-launch r-sweep + 4-tax white-box breakdown",
        spec="docs/paper_motivation_3_3_ext_experiments.md",
        gpu=torch.cuda.get_device_name(0),
        driver=None,
        torch=torch.__version__, cuda=torch.version.cuda,
        commit=sha,
        r_grid=R_GRID,
        k_rule="K_high = round(r*K/32)*32 (endpoints exact); K_low = K - K_high",
        torchmixed2_impl=("explicit elementwise add `lo + hi` after two torch._scaled_mm_v2 "
                          "launches (NVFP4 BlockWise1x16 e4m3 scales on K_low + MXFP8 "
                          "BlockWise1x32 unit e8m0 scales on K_high), SWIZZLE_32_4_4 — "
                          "imported verbatim from promix/eval/gemm_h2h_torchfp4.py "
                          "(producer of docs/rd_buildup_data/h2h_naive.json). NOT beta=1 epilogue."),
        endpoints="r=0 row records torchmixed2:=pure NVFP4; r=1 row records torchmixed2:=pure MXFP8",
        timing=("bench_cell from gemm_h2h_torchfp4.py: 400ms bf16 clock warm-spin, "
                "warmup=10 per fn, iters=50, candidates interleaved round-robin per "
                "iteration, CUDA events, median+min reported. NO explicit L2 flush — "
                "identical cache treatment to h2h_naive.json. (Task spec suggested "
                "warmup=20; warmup=10 kept deliberately to match h2h_naive.json exactly.)"),
        activation=("x = randn(M,K) seed 1 fp32; last K/8 channels *3.0 (same hi-variance "
                    "boost as h2h, applied at the fixed r=1/8 split for all r so the input "
                    "tensor is identical across the sweep); GEMM paths time prequantized "
                    "activations (GEMM-only), matching h2h_naive.json."),
        taxes=dict(
            online_dense_rot_ms="BF16 (M,K)@(K,K) torch.mm (cuBLAS); qkv/o_proj/gate_up only, null for down",
            online_gather_ms="xbf.index_select(1, randperm(K))",
            standalone_quant_pack_ms=("two independent qpf CUDA kernel launches: "
                                      "quant_pack_float_fp4 over K_low + quant_pack_float_fp8 over "
                                      "K_high (col_off slices of the same (M,K) bf16 buffer); "
                                      "r=0/r=1 endpoints: single full-K fp4/fp8 pack"),
            online_hadamard_ms=("down only: matmul_hadU_cuda(xbf, had_K, K, fold_scale=True) "
                                "(fast_hadamard_transform; 8B K=14336 -> had28 kron, 3B K=8192 pure WHT)"),
        ),
        anti_circularity="AdaMix fused kernel (fused_ms) NOT measured; standard/naive paths only",
        models=args.models, Ms=ms_list,
        native_fp4_exact=exact, native_fp4_cosine=cos,
    )
    try:
        import subprocess as sp
        manifest["driver"] = sp.run(["nvidia-smi", "--query-gpu=driver_version",
                                     "--format=csv,noheader"], capture_output=True,
                                    text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"RESULTS_WRITTEN {out_json}\nM33EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
