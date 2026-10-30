"""rd-bench: MicroMix mixed MXFP4/6/8 GEMM at M=8192, as shipped
(/home/luofan/baselines/peers/micromix/MicroMix, mixedgemm extension, built on
CUDA 13 where its own --validate FAILS — correctness NOT validated here; we
also record output cosine vs fp32 reference to quantify the breakage).

Configs per shape (KN=MXFP4 ch, KS=MXFP6 ch, KO=MXFP8 ch):
  pure_fp4:  (K, 0, 0)          — its W4A4 floor, 4.0 bits
  paper_avg: (K/2, K/4, K/4)    — ~5.5 avg bits (paper's Llama-3.1-8B average)
Timing follows the repo's own test.py call pattern: reorder_quantize_x/w4 then
matmul (single launch per precision pair inside), cudaEvent GEMM-only.
Run with container python (torch 2.11 cu130).
"""
import sys, json, argparse
sys.path.insert(0, "/home/luofan/baselines/peers/micromix/MicroMix/mgemm/build")
import torch
import mixedgemm

DEV = "cuda"

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]


def configs_for(K):
    return [
        ("pure_fp4", K, 0, 0, 4.0),
        ("paper_avg_5.5b", K // 2, K // 4, K // 4, 5.5),
    ]


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
        W = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.08
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.08
        reorder_index = torch.arange(K, dtype=torch.int16, device=DEV)
        ref = x.float() @ W.float().t()

        Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
        Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
        t_fp16, min_fp16 = median_ms(lambda: torch.matmul(Xf, Wf.t()))

        for cfg, KN, KS, KO, bits in configs_for(K):
            try:
                AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(
                    x, reorder_index, KN, KS, KO)
                BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w4(
                    W, reorder_index, KN, KS, KO)

                def gemm():
                    return mixedgemm.matmul(AN, BN, AS, BS, AO, BO,
                                            SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)
                y = gemm()
                cos = cosine(y, ref)
                t_g, min_g = median_ms(gemm)
            except Exception as e:
                rows.append(dict(method="micromix", model=mdl, shape=name, M=M, N=N, K=K,
                                 config=cfg, KN=KN, KS=KS, KO=KO, avg_bits=bits,
                                 error=repr(e)))
                print(f"{mdl} {name} {cfg}: FAILED {e!r}", flush=True)
                continue

            flops = 2.0 * M * N * K
            rec = dict(method="micromix", model=mdl, shape=name, M=M, N=N, K=K,
                       config=cfg, KN=KN, KS=KS, KO=KO, avg_bits=bits,
                       cosine_vs_fp32=cos, validated=False,
                       gemm_ms=t_g, gemm_min_ms=min_g,
                       fp16_ms=t_fp16, fp16_min_ms=min_fp16,
                       gemm_tflops=flops / (t_g * 1e-3) / 1e12,
                       gemm_vs_fp16=t_fp16 / t_g)
            rows.append(rec)
            print(f"{mdl} {name:8s} M={M} N={N:6d} K={K:6d} {cfg:14s}: "
                  f"gemm {t_g:.4f}ms ({rec['gemm_tflops']:.0f} TF, cos={cos:.6f}) | "
                  f"fp16 {t_fp16:.4f}ms | gemm/fp16 {rec['gemm_vs_fp16']:.2f}x", flush=True)
            del AN, AS, AO, SFAN, SFAS, SFAO, BN, BS, BO, SFBN, SFBS, SFBO, y
            torch.cuda.empty_cache()
        del W, x, ref, Xf, Wf
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nMICROMIX_DONE", flush=True)


if __name__ == "__main__":
    main()
