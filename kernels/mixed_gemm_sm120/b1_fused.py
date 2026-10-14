"""B1 — TRUE single-kernel fused mixed-precision GEMM (SM120). Loader + validator.

Reuses nvfp4_gemm.py / mxfp8_gemm.py quantizers for data + oracle. Builds the
hand-rolled CuTe kernel in b1_fused_ext.cu and validates vs the oracle.

Run inside luofan-plaquant on a FREE GPU:
    CUDA_VISIBLE_DEVICES=<free> python b1_fused.py
"""
import csv
import os
import torch
from torch.utils.cpp_extension import load

import nvfp4_gemm as nv

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="b1_fused_ext",
            sources=[os.path.join(HERE, "b1_fused_ext.cu")],
            extra_include_paths=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
                os.path.join(CUTLASS, "examples/common"),
            ],
            extra_cuda_cflags=[
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-O3",
            ],
            verbose=True,
        )
    return _ext


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


# ---- Step 1: NVFP4-only ----
def validate_step1():
    print("=== B1 Step 1: NVFP4-only hand kernel vs oracle ===")
    ext = get_ext()
    ok = True
    # (M, N, K) ; K multiple of 64
    cases = [
        (1,    512,  1792),
        (16,   2048, 1792),
        (256,  512,  1792),
        (2048, 2048, 1792),
        (16,   3072, 2688),
        (16,   2048, 7168),   # down_proj K_low
    ]
    for (M, N, K) in cases:
        torch.manual_seed(M + K)
        dev = "cuda"
        A = torch.randn(M, K, device=dev)
        B = torch.randn(N, K, device=dev)
        A_codes, SFA, A_deq = nv.quantize_nvfp4(A)
        B_codes, SFB, B_deq = nv.quantize_nvfp4(B)
        A_packed = nv.pack_fp4(A_codes)
        B_packed = nv.pack_fp4(B_codes)
        D = ext.b1_nvfp4_only(A_packed, SFA, B_packed, SFB, M, N, K).float()
        ref = A_deq @ B_deq.t()
        cos = cosine(D, ref)
        rel = ((D - ref).norm() / (ref.norm() + 1e-12)).item()
        status = "PASS" if cos >= 0.999 else "FAIL"
        if cos < 0.999:
            ok = False
        print(f"  M={M:5d} N={N:5d} K={K:5d}: cosine={cos:.6f} rel={rel:.3e} "
              f"Dn={D.norm().item():.3e} refn={ref.norm().item():.3e} [{status}]")
    print("STEP1 ALL PASS" if ok else "STEP1 SOME FAILED")
    return ok


# ---- Full B1: dual-phase fused kernel vs mixed oracle ----
def b1_single(p, M, N, K_low, K_high):
    ext = get_ext()
    return ext.b1_fused(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, K_low, K_high)


def validate_fused():
    import mixed_gemm as mg
    print("\n=== B1 full fused kernel vs mixed oracle ===")
    ext = get_ext()
    ok = True
    val_shapes = {
        "q_proj_1B": (2048, 1792, 256),
        "down_proj_1B": (2048, 7168, 1024),
        "q_proj_3B": (3072, 2688, 384),
    }
    for name, (N, K_low, K_high) in val_shapes.items():
        for M in [1, 16, 256, 2048]:
            p, oracle = mg.prep_mixed(M, N, K_low, K_high, seed=M)
            D = b1_single(p, M, N, K_low, K_high).float()
            cos = cosine(D, oracle)
            rel = ((D - oracle).norm() / (oracle.norm() + 1e-12)).item()
            status = "PASS" if cos >= 0.999 else "FAIL"
            if cos < 0.999:
                ok = False
            print(f"  {name:14s} M={M:5d} N={N:5d} Klow={K_low:5d} Khigh={K_high:5d}: "
                  f"cosine={cos:.6f} rel={rel:.3e} [{status}]")
    print("B1 FUSED ALL PASS" if ok else "B1 FUSED SOME FAILED")
    return ok


def benchmark_b1():
    """Honest benchmark: same _time_ms harness as mixed_gemm.benchmark(); appends
    b1_single rows to benchmark.csv. Naive B1 being slower than rungB_presf is
    expected at this stage (one warp per 16x8 tile, no TMA/ldmatrix/pipelining)."""
    import mixed_gemm as mg
    print("\n=== B1 single-kernel benchmark (appending b1_single rows) ===")
    rows = []
    all_shapes = {**mg.SHAPES_1B, **mg.SHAPES_3B}
    for name, (N, K_low, K_high) in all_shapes.items():
        for M in mg.M_LIST:
            p, _ = mg.prep_mixed(M, N, K_low, K_high, seed=1)
            lat = mg._time_ms(lambda: b1_single(p, M, N, K_low, K_high))
            print(f"  {name:14s} M={M:5d} N={N:5d}: b1_single={lat:.4f} ms")
            rows.append((name, M, "b1_single", f"{lat:.5f}"))
    csv_path = os.path.join(HERE, "benchmark.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["shape", "M", "path", "latency_ms"])
        w.writerows(rows)
    print(f"Appended {len(rows)} b1_single rows to {csv_path}")


if __name__ == "__main__":
    print("Building b1_fused_ext (first build may take minutes)...")
    get_ext()
    print("Built.\n")
    ok1 = validate_step1()
    ok2 = validate_fused()
    if ok1 and ok2:
        benchmark_b1()
    print("\nB1 DONE" if (ok1 and ok2) else "\nB1 VALIDATION FAILED")
