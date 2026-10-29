"""QuaRot INT4 CUTLASS GEMM at large M on SM120 (RTX Pro 6000) — peer datapoint.

QuaRot's quarot._CUDA.matmul (cutlass INT4 s4 GEMM, built for SM120 in
/home/luofan/baselines/peers/quarot) vs fp16 cuBLAS at the same logical
(M, N, K) Llama shapes (vLLM merged decomposition), M = 2048..32768.
GEMM-only, warm, cudaEvent median of 30. On Blackwell s4 lowers to the INT8
IMMA rate (no INT4 speedup) — this measures how far the result lands from fp16.
"""
import argparse
import json

import torch
import quarot

DEV = "cuda"


def shapes_for(H, I, nheads, nkv):
    hd = H // nheads
    q = nheads * hd
    kv = nkv * hd
    return [("qkv", q + 2 * kv, H), ("o_proj", H, q),
            ("gate_up", 2 * I, H), ("down", H, I)]


MODELS = {"3B": dict(H=3072, I=8192, nheads=24, nkv=8),
          "8B": dict(H=4096, I=14336, nheads=32, nkv=8)}


def median_ms(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort()
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3B,8B")
    ap.add_argument("--ms", default="2048,4096,8192,16384,32768")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    print("device:", torch.cuda.get_device_name(0),
          "cc:", torch.cuda.get_device_capability(0), flush=True)

    rows = []
    for mdl in args.models.split(","):
        c = MODELS[mdl]
        for name, N, K in shapes_for(c["H"], c["I"], c["nheads"], c["nkv"]):
            torch.manual_seed(0)
            B = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=DEV)
            for M in ms:
                try:
                    A = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8,
                                      device=DEV)
                    t_i4, min_i4 = median_ms(lambda: quarot._CUDA.matmul(A, B))
                    Xf = torch.randn(M, K, device=DEV, dtype=torch.float16)
                    Wf = torch.randn(N, K, device=DEV, dtype=torch.float16)
                    t_f, min_f = median_ms(lambda: torch.matmul(Xf, Wf.t()))
                    flops = 2.0 * M * N * K
                    rec = dict(model=mdl, shape=name, M=M, N=N, K=K,
                               int4_ms=t_i4, int4_min_ms=min_i4,
                               fp16_ms=t_f, fp16_min_ms=min_f,
                               int4_tops=flops / (t_i4 * 1e-3) / 1e12,
                               fp16_tflops=flops / (t_f * 1e-3) / 1e12,
                               int4_vs_fp16=t_f / t_i4)
                    rows.append(rec)
                    print(f"{mdl} {name:8s} M={M:5d} N={N:6d} K={K:6d}: "
                          f"int4 {t_i4:.4f}ms ({rec['int4_tops']:.0f} TOPS) | "
                          f"fp16 {t_f:.4f}ms ({rec['fp16_tflops']:.0f} TF) | "
                          f"int4/fp16 {rec['int4_vs_fp16']:.3f}x", flush=True)
                    del A, Xf, Wf
                except torch.cuda.OutOfMemoryError:
                    print(f"{mdl} {name} M={M}: OOM", flush=True)
                    rows.append(dict(model=mdl, shape=name, M=M, N=N, K=K,
                                     oom=True))
                    torch.cuda.empty_cache()
            del B
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nQUAROT_GEMM_DONE", flush=True)


if __name__ == "__main__":
    main()
