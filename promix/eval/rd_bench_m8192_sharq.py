"""rd-bench: SharQ (NVFP4 4:8 sparse backbone + dense FP4 residual) at M=8192,
as shipped, using its own CUDA harness `bench_sharq_linear` (preallocated
buffers, its own warmup+avg timing). We parse the "GEMM only" section:
  - NVFP4 dense        : its plain dense NVFP4 GEMM reference
  - SHARQ fused-accum  : sparse-main GEMM + residual GEMM accumulating (the
                         method's actual GEMM work per Linear, 2 launches)
Run with container python; binary at kernels/build_cmake_sm120a/.
"""
import json, re, subprocess, argparse

BIN = "/home/luofan/baselines/peers/sharq/SharQ/kernels/build_cmake_sm120a/bench_sharq_linear"

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]

FIELDS = {
    "nvfp4_gemm_ms": r"NVFP4 dense\s*:\s*([0-9.eE+-]+) ms",
    "sharq_sparse_main_ms": r"SHARQ sparse main\s*:\s*([0-9.eE+-]+) ms",
    "sharq_residual_ms": r"SHARQ residual\s*:\s*([0-9.eE+-]+) ms",
    "sharq_gemm_fused_ms": r"SHARQ fused-accum\s*:\s*([0-9.eE+-]+) ms",
    "nvfp4_whole_ms": r"Whole linear[\s\S]*?NVFP4\s*:\s*([0-9.eE+-]+) ms",
    "sharq_whole_ms": r"Whole linear[\s\S]*?SHARQ\s*:\s*([0-9.eE+-]+) ms",
    "sharq_quant_ms": r"Quantize only[\s\S]*?SHARQ\s*:\s*([0-9.eE+-]+) ms",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M = args.m

    rows = []
    for mdl, name, N, K in SHAPES:
        cmd = [BIN, "--m", str(M), "--n", str(N), "--k", str(K),
               "--warmup", str(args.warmup), "--iters", str(args.iters)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout
        if p.returncode != 0:
            rows.append(dict(method="sharq", model=mdl, shape=name, M=M, N=N, K=K,
                             error=p.stderr.strip()[:500]))
            print(f"{mdl} {name}: FAILED rc={p.returncode} {p.stderr[:200]}", flush=True)
            continue
        rec = dict(method="sharq", model=mdl, shape=name, M=M, N=N, K=K)
        for key, pat in FIELDS.items():
            m = re.search(pat, out)
            rec[key] = float(m.group(1)) if m else None
        flops = 2.0 * M * N * K
        if rec["sharq_gemm_fused_ms"]:
            rec["gemm_ms"] = rec["sharq_gemm_fused_ms"]
            rec["gemm_tflops_base"] = flops / (rec["gemm_ms"] * 1e-3) / 1e12
        rows.append(rec)
        print(f"{mdl} {name:8s} M={M} N={N:6d} K={K:6d}: "
              f"SHARQ gemm(fused sparse+resid) {rec['sharq_gemm_fused_ms']}ms "
              f"(sparse {rec['sharq_sparse_main_ms']} + resid {rec['sharq_residual_ms']}) | "
              f"its NVFP4 dense {rec['nvfp4_gemm_ms']}ms", flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nSHARQ_DONE", flush=True)


if __name__ == "__main__":
    main()
