"""rd-bench: ARCQuant NVFP4(+residual channels) W4A4 GEMM at M=8192, as shipped
(/home/luofan/baselines/peers/arcquant/ARCQuant, agemm extension).

ARCQuant appends S residual channels along K (GEMM runs K+S). S is
calibration-adaptive per layer (threshold = 0.125*max, repo utilize.py; paper
caps S<=512). We time its GEMM at S in {0, 256, 512} per shape to bracket the
actual deployed work. Timing matches the repo's own kernels/bench.py pattern:
reorder_quantize_x/w with KE, then agemm.matmul cudaEvent-timed GEMM-only.
Run with container python (torch 2.11 cu130).
"""
import sys, json, argparse
sys.path.insert(0, "/home/luofan/baselines/peers/arcquant/ARCQuant/kernels/build")
import torch
import agemm

DEV = "cuda"

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]
S_LIST = [0, 256, 512]


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


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M = args.m

    rows = []
    for mdl, name, N, K in SHAPES:
        torch.manual_seed(0)
        W = (torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.08)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.08
        reorder_index = torch.arange(K, dtype=torch.int16, device=DEV)
        ref = x.float() @ W.float().t()

        Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
        Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
        t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))

        for S in S_LIST:
            qw, sfw = agemm.reorder_quantize_w(W, reorder_index, S)
            qx, sfx = agemm.reorder_quantize_x(x, reorder_index, S)

            def gemm():
                return agemm.matmul(qx, qw, sfx, sfw, 1.0)
            y = gemm()
            cos = cosine(y, ref)
            t_g, min_g = median_ms(gemm)

            flops_base = 2.0 * M * N * K          # useful work
            flops_actual = 2.0 * M * N * (K + S)  # work the kernel actually does
            rec = dict(method="arcquant", model=mdl, shape=name, M=M, N=N, K=K, S=S,
                       K_gemm=K + S, cosine_vs_fp32=cos,
                       gemm_ms=t_g, gemm_min_ms=min_g,
                       fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                       gemm_tflops_base=flops_base / (t_g * 1e-3) / 1e12,
                       gemm_tflops_actual=flops_actual / (t_g * 1e-3) / 1e12,
                       gemm_vs_fp16=t_fp16 / t_g)
            rows.append(rec)
            print(f"{mdl} {name:8s} M={M} N={N:6d} K={K:6d} S={S:4d}: "
                  f"gemm {t_g:.4f}ms ({rec['gemm_tflops_base']:.0f} TF-base, cos={cos:.6f}) | "
                  f"fp16 {t_fp16:.4f}ms | gemm/fp16 {rec['gemm_vs_fp16']:.2f}x", flush=True)
            del qw, sfw, qx, sfx, y
            torch.cuda.empty_cache()
        del W, x, ref, Xf, Wf
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nARCQUANT_DONE", flush=True)


if __name__ == "__main__":
    main()
