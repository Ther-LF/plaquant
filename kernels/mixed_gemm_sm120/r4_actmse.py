"""round-4 CHEAP activation-MSE harness (isolated, remote RTX PRO 6000 / SM120).

Builds the 8B fused SwiGLU+Hadamard(28x512)+NVFP4/MXFP8 producer (hadamard28_pack_ext)
AND the plain packer (quant_pack_fused_ext) in several act-MSE flavors via distinct
cpp_extension names + -D flags:

  OFF    : act-MSE off              (baseline pack)
  OLD    : round-3 search (true-div, exact RTN, 3 steps)  -DPROMIX_ACT_MSE_FAST=0
  FAST3  : round-4 fast (recip+branchless, 3 steps)        -DPROMIX_ACT_MSE_FAST=1
  FAST2  : round-4 fast, 2 steps
  FAST1  : round-4 fast, 1 step

Measures (8B down_proj shape I=14336, K_low=12544, K_high=1792, N=4096):
  * PERF  : isolated hadamard28 pack ms (min-of-N), overhead vs OFF.
  * BITGATE: hadamard28[flavor] vs ref_chain packed with quant_pack_fused_ext[flavor]
             (same-flavor CUDA self-consistency; 0 byte mismatch required).
Harvest JSON from disk.
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
I8B, KLOW, KHIGH, NGEMM = 14336, 12544, 1792, 4096
CUTLASS = os.path.join(REPO, "third_party/cutlass")

FLAVORS = {
    "OFF":   [],
    "OLD":   ["-DPROMIX_ACT_MSE=1", "-DPROMIX_ACT_MSE_FAST=0", "-DPROMIX_ACT_MSE_STEPS=3"],
    "FAST3": ["-DPROMIX_ACT_MSE=1", "-DPROMIX_ACT_MSE_FAST=1", "-DPROMIX_ACT_MSE_STEPS=3"],
    "FAST2": ["-DPROMIX_ACT_MSE=1", "-DPROMIX_ACT_MSE_FAST=1", "-DPROMIX_ACT_MSE_STEPS=2"],
    "FAST1": ["-DPROMIX_ACT_MSE=1", "-DPROMIX_ACT_MSE_FAST=1", "-DPROMIX_ACT_MSE_STEPS=1"],
}

def _load(name, src, defs):
    from torch.utils.cpp_extension import load
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"),
           os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"] + defs
    return load(name=name, sources=[os.path.join(KERN, src)],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)

def mins(fn, iters=40, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3

@torch.no_grad()
def ref_chain(g, u, had_K, qpfm, M):
    h = torch.nn.functional.silu(g) * u
    y = matmul_hadU_cuda(h.float(), had_K, 28, fold_scale=True).to(torch.float16)
    return qpfm.quant_pack_float_mixed(y.contiguous(), M, NGEMM, KLOW, KHIGH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="2048,4096,8192")
    ap.add_argument("--out", default=os.path.join(KERN, "r4_actmse.json"))
    args = ap.parse_args()
    print(f"dev={torch.cuda.get_device_name(0)}", flush=True)
    Ms = [int(x) for x in args.ms.split(",")]

    h28 = {f: _load(f"h28_{f.lower()}", "hadamard28_pack_ext.cu", d) for f, d in FLAVORS.items()}
    qpfm = {f: _load(f"qpfm_{f.lower()}", "quant_pack_fused_ext.cu", d) for f, d in FLAVORS.items()}
    print("all flavors built", flush=True)

    had_cpu, K = get_hadK(I8B)
    had_K = had_cpu.to(DEV).float()
    had_cpu_c = had_cpu.cpu()
    out = {"perf": [], "bitgate": []}

    # ---- BIT GATE: hadamard28[f] == ref_chain via qpfm[f] (same flavor) ----
    for f in FLAVORS:
        torch.manual_seed(1234)
        M = 2048
        g = (torch.randn(M, I8B, device=DEV) * 3).to(torch.float16)
        u = (torch.randn(M, I8B, device=DEV) * 2).to(torch.float16)
        u[::3] *= 8.0
        a_ref = ref_chain(g, u, had_K, qpfm[f], M)
        a = h28[f].hadamard28_swiglu_pack_mixed(g, u, had_cpu_c, M, NGEMM, KLOW, KHIGH)
        mm = [int((x != y).sum().item()) for x, y in zip(a_ref, a)]
        good = all(m == 0 for m in mm)
        print(f"  BITGATE {f:5s} {'PASS' if good else 'FAIL'} mism={mm}", flush=True)
        out["bitgate"].append(dict(flavor=f, mismatch=mm, pass_=good))
        del g, u, a_ref, a; torch.cuda.empty_cache()

    # ---- PERF: isolated hadamard28 pack, overhead vs OFF ----
    for M in Ms:
        g = (torch.randn(M, I8B, device=DEV) * 3).bfloat16()
        u = (torch.randn(M, I8B, device=DEV) * 2).bfloat16()
        row = {"M": M}
        for f in FLAVORS:
            t = mins(lambda: h28[f].hadamard28_swiglu_pack_mixed(g, u, had_cpu_c, M, NGEMM, KLOW, KHIGH))
            row[f] = t
        base = row["OFF"]
        row_ovh = {f: row[f] / base for f in FLAVORS}
        print(f"  M={M:5d} ms={{" + ", ".join(f'{f}:{row[f]:.3f}' for f in FLAVORS) + "}  "
              f"ovh={{" + ", ".join(f'{f}:{row_ovh[f]:.2f}x' for f in FLAVORS if f!='OFF') + "}", flush=True)
        row["overhead"] = row_ovh
        out["perf"].append(row)
        del g, u; torch.cuda.empty_cache()

    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"RESULTS_WRITTEN {args.out}", flush=True)
    print("R4_ACTMSE_DONE", flush=True)

if __name__ == "__main__":
    main()
