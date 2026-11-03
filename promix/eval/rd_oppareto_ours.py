"""rd-oppareto (E4): ProMix fused GEMM latency-vs-r FULL curve on the FINAL
post-R5 kernel, through the SHIPPING dispatch (MixedQuantLinear._gemm — the
exact serving launch path: per-shape split-K/small-M windows as they'd fire at
this M, r5 cast-fold bias epilogue outside the windows, single-phase NVFP4 for
K_high==0).

Grid r in {0, 1/32, 1/16, 1/12, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1}, K_high
snapped to the MXFP8 block (32); r=1 is served as ALL-BUT-ONE-RING (K_low=64,
one RING=64 NVFP4 ring kept) because K_low==0 (pure-MXFP8) has no kernel path
— noted in the output rows. GEMM-only, prepacked activations, cudaEvent
median-of-50 (15 warmup) like the rd-bench M=8192 suite, PLUS a CUDA-graph
replay median (serving replays captured graphs). Idle GPU, run inside
container `luofan-plaquant`.

Usage: python3 promix/eval/rd_oppareto_ours.py --ms 2048,8192 --out ours_curve.json
"""
import os, sys, json, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.eval.full_network_realkernel import load_kernels  # noqa: E402
from promix.vllm_integ.kernels import load_smallm, load_splitk, load_cd_bias  # noqa: E402
from promix.quantize.weight_pack import pack_weight_mixed  # noqa: E402
from promix.quantize.mixed_linear import (MixedQuantLinear, _SPLITK_TABLE,  # noqa: E402
                                          _SMALLM_TABLE)

DEV = "cuda"
RING = 64   # rd_validity RING — one NVFP4 ring kept at the r=1 end

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]

R_GRID = [0.0, 1 / 32, 1 / 16, 1 / 12, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2,
          3 / 4, 1.0]


def snap_khigh(K, r):
    """K_high for nominal r: multiple of 32 (MXFP8 block), K_low >= RING.
    r=0 -> 0 (single-phase pure-NVFP4); r=1 -> K-RING (all-but-one-ring,
    K_low==0 unsupported)."""
    if r == 0.0:
        return 0
    if r == 1.0:
        return K - RING
    kh = int(round(r * K / 32)) * 32
    return max(32, min(kh, K - RING))


def median_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort(); n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def graph_median_ms(fn, iters=50, warmup=5):
    """Capture fn into a CUDA graph, time graph.replay() (serving-mode number)."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort(); n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def dispatch_tag(mql, M):
    """Replicate MixedQuantLinear._gemm routing to label which path fires."""
    if mql.is_pure_nvfp4:
        return "single_phase_nvfp4"
    skc = mql._splitk_cfg(M)
    if skc is not None:
        return f"splitk_cfg{skc[0]}_s{skc[1]}_d{skc[2]}"
    cfg = mql._smallm_cfg(M)
    if cfg is not None:
        return "smallm_128x64"
    if mql._cd_bias is not None:
        return "bias_castfold_128x128"
    return "fused_prepacked_128x128"


def eff_bits(K_low, K_high):
    """Effective element+scale bits: NVFP4 4+8/16=4.5, MXFP8 8+8/32=8.25."""
    return (K_low * 4.5 + K_high * 8.25) / (K_low + K_high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="2048,8192")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]

    nv, mx, cd, qpf = load_kernels()
    cd_sm, cd_sk, cd_bias = load_smallm(), load_splitk(), load_cd_bias()
    print(f"kernels loaded (cd_sm={cd_sm is not None} cd_sk={cd_sk is not None} "
          f"cd_bias={cd_bias is not None})", flush=True)

    rows = []
    for mdl, name, N, K in SHAPES:
        torch.manual_seed(0)
        W = (torch.randn(N, K, device=DEV) * 0.08)
        for M in ms:
            x = torch.randn(M, K, device=DEV)
            ref = None
            for r_nom in R_GRID:
                K_high = snap_khigh(K, r_nom)
                K_low = K - K_high
                packed = pack_weight_mixed(W, K_low, K_high, nv, mx, cd)
                mql = MixedQuantLinear(packed, nv, mx, cd, qpf=qpf,
                                       cd_bias=cd_bias, cd_sm=cd_sm, cd_sk=cd_sk)
                A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

                def gemm():
                    return mql._gemm(A_lo, pSFA_lo, A_hi, pSFA_hi, M)

                y = gemm()
                if ref is None:
                    ref = x.float() @ W.float().t()
                cos = cosine(y, ref)
                tag = dispatch_tag(mql, M)
                t_med, t_min = median_ms(gemm)
                if args.no_graph:
                    g_med, g_min = float("nan"), float("nan")
                else:
                    g_med, g_min = graph_median_ms(gemm)
                flops = 2.0 * M * N * K
                rec = dict(model=mdl, shape=name, M=M, N=N, K=K,
                           r_nominal=r_nom, K_high=K_high, K_low=K_low,
                           r_exact=K_high / K, dispatch=tag,
                           gemm_ms=t_med, gemm_min_ms=t_min,
                           graph_ms=g_med, graph_min_ms=g_min,
                           tflops=flops / (t_med * 1e-3) / 1e12,
                           eff_bits=eff_bits(K_low, K_high),
                           cosine_vs_fp32=cos,
                           note=("r=1 served as all-but-one-ring (K_low=64; "
                                 "K_low==0 pure-MXFP8 unsupported)")
                           if r_nom == 1.0 else "")
                rows.append(rec)
                print(f"{mdl} {name:8s} M={M} r={r_nom:.4f} K_high={K_high:5d} "
                      f"[{tag}]: {t_med:.4f}ms graph {g_med:.4f}ms "
                      f"({rec['tflops']:.0f} TF, cos={cos:.6f})", flush=True)
                del mql, packed, A_lo, pSFA_lo, A_hi, pSFA_hi, y
                torch.cuda.empty_cache()
            # fp16 cuBLAS anchor per shape x M
            Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
            Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
            t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))
            rows.append(dict(model=mdl, shape=name, M=M, N=N, K=K,
                             r_nominal=-1, K_high=-1, K_low=-1, r_exact=-1,
                             dispatch="fp16_cublas", gemm_ms=t_fp16,
                             gemm_min_ms=min_fp16, graph_ms=float("nan"),
                             graph_min_ms=float("nan"),
                             tflops=2.0 * M * N * K / (t_fp16 * 1e-3) / 1e12,
                             eff_bits=16.0, cosine_vs_fp32=float("nan"),
                             note="fp16 anchor"))
            print(f"{mdl} {name:8s} M={M} fp16: {t_fp16:.4f}ms", flush=True)
            del x, ref, Xf, Wf
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nOURS_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
