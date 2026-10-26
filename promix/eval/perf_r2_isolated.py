"""Isolated harness for PERF round-2 Task 1: the 8B (I=14336) fused SwiGLU +
Hadamard(28x512) + NVFP4/MXFP8 pack producer.

Standalone, no model load: synthetic gate/up with realistic magnitudes exercise
the full numeric path.  Two things measured:

  --phase gate : byte-exactness of the fused kernel's 4 output tensors
                 (code_lo, sfa_lo, code_hi, sfa_hi) vs the reference chain
                 silu(g)*u -> matmul_hadU_cuda(h.float(), had28, 28, fold_scale=True)
                 .half() -> quant_pack_float_mixed.  The HARD gate (0 bytes).
  --phase perf : fused-kernel time vs the 4-op reference chain at the 8B
                 down_proj shape (I=14336, K_low=12544, K_high=1792, N=4096),
                 min-of-N, event-synchronized.

Remote-only (RTX PRO 6000, SM120).  Pick an idle GPU via CUDA_VISIBLE_DEVICES.
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.quantize.hadamard import matmul_hadU_cuda, get_hadK

DEV = "cuda"
I8B = 14336
KLOW = 12544
KHIGH = 1792
NGEMM = 4096   # 8B down_proj output width (hidden_size)


def _load(name, src):
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(REPO, "third_party/cutlass/include"),
           os.path.join(REPO, "third_party/cutlass/tools/util/include"),
           os.path.join(REPO, "third_party/cutlass/examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    return load(name=name, sources=[os.path.join(KERN, src)],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)


def ref_chain(g, u, had_K, K, qpfm, M):
    h = torch.nn.functional.silu(g) * u
    y = matmul_hadU_cuda(h.float(), had_K, K, fold_scale=True).to(torch.float16)
    return qpfm.quant_pack_float_mixed(y.contiguous(), M, NGEMM, KLOW, KHIGH)


def mins(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


@torch.no_grad()
def run_gate(h28, qpfm, out, dtypes=("bf16", "fp16")):
    had_cpu, K = get_hadK(I8B)          # had28 (28x28 +-1), K=28
    print(f"get_hadK({I8B}) -> K={K}, had shape={tuple(had_cpu.shape)}", flush=True)
    had_K = had_cpu.to(DEV).float()
    results = []
    ok_all = True
    for dt in dtypes:
        tdt = torch.bfloat16 if dt == "bf16" else torch.float16
        for M in (512, 2048, 4096):
            torch.manual_seed(M)
            g = (torch.randn(M, I8B, device=DEV) * 3).to(tdt)
            u = (torch.randn(M, I8B, device=DEV) * 2).to(tdt)
            u[::3] *= 8.0                                        # outlier stress
            a_ref = ref_chain(g, u, had_K, K, qpfm, M)
            a = h28.hadamard28_swiglu_pack_mixed(g, u, had_cpu.cpu(), M, NGEMM, KLOW, KHIGH)
            mm = [int((x != y).sum().item()) for x, y in zip(a_ref, a)]
            tot = [int(x.numel()) for x in a_ref]
            good = all(m == 0 for m in mm)
            ok_all &= good
            print(f"  {dt} M={M:5d} {'PASS' if good else 'FAIL'} "
                  f"mism code_lo/sfa_lo/code_hi/sfa_hi={mm} of {tot}", flush=True)
            results.append(dict(dtype=dt, M=M, mismatch=mm, total=tot, pass_=good))
            del g, u, a_ref, a
            torch.cuda.empty_cache()
    out["gate"] = dict(ok=ok_all, rows=results)
    print("GATE", "PASS" if ok_all else "FAIL", flush=True)
    return ok_all


@torch.no_grad()
def run_perf(h28, qpfm, out, Ms):
    had_cpu, K = get_hadK(I8B)
    had_K = had_cpu.to(DEV).float()
    rows = []
    for M in Ms:
        g = (torch.randn(M, I8B, device=DEV) * 3).bfloat16()
        u = (torch.randn(M, I8B, device=DEV) * 2).bfloat16()
        had_cpu_c = had_cpu.cpu()
        f_fused = lambda: h28.hadamard28_swiglu_pack_mixed(g, u, had_cpu_c, M, NGEMM, KLOW, KHIGH)
        f_cur = lambda: ref_chain(g, u, had_K, K, qpfm, M)
        tf = mins(f_fused)
        tc = mins(f_cur)
        # fused traffic: read 2*M*I*2B, write M*(K_low/2 + K_high) codes + SF
        rb = 2 * M * I8B * 2 + M * (KLOW // 2 + KHIGH) + 300000
        tbps = rb / (tf * 1e-3) / 1e12
        print(f"  M={M:5d} fused={tf:7.3f} ms  ref_chain={tc:7.3f} ms  "
              f"speedup={tc/tf:4.2f}x  traffic={rb/1e6:7.1f} MB -> {tbps:5.3f} TB/s",
              flush=True)
        rows.append(dict(M=M, fused_ms=tf, ref_chain_ms=tc, speedup=tc / tf,
                         traffic_MB=rb / 1e6, eff_TBps=tbps))
        del g, u
        torch.cuda.empty_cache()
    out["perf"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["gate", "perf", "all"])
    ap.add_argument("--ms", default="512,2048,4096,8192")
    ap.add_argument("--out", default=os.path.join(KERN, "perf_r2_hadamard28.json"))
    args = ap.parse_args()
    print(f"Task1 harness | phase={args.phase} | dev={torch.cuda.get_device_name(0)}", flush=True)
    h28 = _load("hadamard28_pack_ext", "hadamard28_pack_ext.cu")
    qpfm = _load("quant_pack_fused_ext", "quant_pack_fused_ext.cu")
    print("kernels loaded", flush=True)
    out = {}
    if args.phase in ("gate", "all"):
        run_gate(h28, qpfm, out)
    if args.phase in ("perf", "all"):
        run_perf(h28, qpfm, out, [int(x) for x in args.ms.split(",")])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("TASK1_DONE", flush=True)


if __name__ == "__main__":
    main()
