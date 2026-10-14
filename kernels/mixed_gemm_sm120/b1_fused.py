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
                "-Xptxas", "-v",          # register/smem usage in build log
            ],
            verbose=True,
        )
    return _ext


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


_presf_ext = None


def get_presf_ext():
    """T0-presf bounds extension: pure-format full-K GEMMs with pre-interleaved
    weight SF (band yardstick per project-lead directive; see pure_presf_ext.cu)."""
    global _presf_ext
    if _presf_ext is None:
        _presf_ext = load(
            name="pure_presf_ext",
            sources=[os.path.join(HERE, "pure_presf_ext.cu")],
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
    return _presf_ext


def prep_bounds_presf(M, N, K_total, seed=0):
    """Pure-format bound inputs with the weight SF prepacked OUTSIDE the timed
    region (same methodology as rungB_presf / b1_fused). Returns (pack4, pack8)."""
    import mixed_gemm as mg
    ext = get_presf_ext()
    p4, o4 = mg.prep_pure_fp4(M, N, K_total, seed=seed)
    p4["physSFB"] = ext.prepack_sf_fp4_b(p4["SFB"], N, K_total)
    p4["oracle"] = o4
    p8, o8 = mg.prep_pure_fp8(M, N, K_total, seed=seed)
    p8["physSFB"] = ext.prepack_sf_fp8_b(p8["SFB"], N, K_total)
    p8["oracle"] = o8
    return p4, p8


def pure_fp4_presf(p4, M, N, K_total):
    return get_presf_ext().nvfp4_gemm_presf(
        p4["A_packed"], p4["SFA"], p4["B_packed"], p4["physSFB"], M, N, K_total)


def pure_fp8_presf(p8, M, N, K_total):
    return get_presf_ext().mxfp8_gemm_presf(
        p8["A_bytes"], p8["SFA"], p8["B_bytes"], p8["physSFB"], M, N, K_total)


def validate_bounds_presf():
    """Correctness gate for the presf bounds vs each format's own oracle."""
    print("\n=== T0-presf bounds correctness ===")
    ok = True
    for (N, K_total) in [(2048, 2048), (2048, 8192), (3072, 3072)]:
        for M in [1, 16, 2048]:
            p4, p8 = prep_bounds_presf(M, N, K_total, seed=M)
            c4 = cosine(pure_fp4_presf(p4, M, N, K_total).float(), p4["oracle"])
            c8 = cosine(pure_fp8_presf(p8, M, N, K_total).float(), p8["oracle"])
            status = "PASS" if min(c4, c8) >= 0.999 else "FAIL"
            if min(c4, c8) < 0.999:
                ok = False
            print(f"  N={N:5d} K={K_total:5d} M={M:5d}: fp4_presf={c4:.6f} fp8_presf={c8:.6f} [{status}]")
    print("BOUNDS PRESF ALL PASS" if ok else "BOUNDS PRESF SOME FAILED")
    return ok


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
def b1_single(p, M, N, K_low, K_high, stages=0, cfg=0):
    """stages: 0=auto, 1=sync staging, 2/3=cp.async pipeline depth (cfg10: 3/4-deep).
    cfg: 0=auto by (M,N), 1=16x64x4w, 2=16x32x2w, 3=32x64x4w, 4=128x128x8w,
    5=128x128x8w+KMUL2, 6=64x128x8w+KMUL2, 10=warp-specialized 128x128+KMUL2."""
    ext = get_ext()
    return ext.b1_fused(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, K_low, K_high, stages, cfg)


def validate_fused(stages=0):
    import mixed_gemm as mg
    print(f"\n=== B1 full fused kernel vs mixed oracle (stages={stages}) ===")
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
            D = b1_single(p, M, N, K_low, K_high, stages=stages).float()
            cos = cosine(D, oracle)
            rel = ((D - oracle).norm() / (oracle.norm() + 1e-12)).item()
            status = "PASS" if cos >= 0.999 else "FAIL"
            if cos < 0.999:
                ok = False
            print(f"  {name:14s} M={M:5d} N={N:5d} Klow={K_low:5d} Khigh={K_high:5d}: "
                  f"cosine={cos:.6f} rel={rel:.3e} [{status}]")
    print("B1 FUSED ALL PASS" if ok else "B1 FUSED SOME FAILED")
    return ok


def validate_all_cfgs():
    """Gate every (cfg, stages) kernel instantiation, incl. ragged M (padding).
    cfg10 = warp-specialized kernel (stages<=2 -> 3-deep, 3 -> 4-deep)."""
    import mixed_gemm as mg
    print("\n=== B1 per-config gate (cfg x stages) ===")
    ok = True
    cases = [
        ("q_proj_1B", 2048, 1792, 256, 256, [1, 2, 3, 4, 5, 6, 10]),
        ("down_proj_1B", 2048, 7168, 1024, 49, [1, 2, 3, 4, 5, 6, 10]),
        ("q_proj_3B", 3072, 2688, 384, 49, [10]),
    ]
    for name, N, K_low, K_high, M, cfgs in cases:
        p, oracle = mg.prep_mixed(M, N, K_low, K_high, seed=M)
        for cfg in cfgs:
            for st in [1, 2, 3]:
                D = b1_single(p, M, N, K_low, K_high, stages=st, cfg=cfg).float()
                cos = cosine(D, oracle)
                if cos < 0.999:
                    ok = False
                    print(f"  FAIL {name} M={M} cfg={cfg} stages={st}: cosine={cos:.6f}")
        print(f"  {name:14s} M={M:4d}: all cfg x stages cosine>=0.999" if ok else "", "")
    print("CFG GATE ALL PASS" if ok else "CFG GATE SOME FAILED")
    return ok


def benchmark_b1(path_name, stages, ref_paths=True, bounds=True):
    """Honest benchmark: same _time_ms harness as mixed_gemm.benchmark(); appends
    `path_name` rows to benchmark.csv. If ref_paths, also times rungB_presf fresh
    in the same session; if bounds, times the pure NVFP4/MXFP8 full-K T0 bounds
    and prints the band position (0%=FP4 floor, 100%=FP8 ceiling)."""
    import mixed_gemm as mg
    print(f"\n=== B1 benchmark: path={path_name} stages={stages} ===")
    rows = []
    all_shapes = {**mg.SHAPES_1B, **mg.SHAPES_3B}
    for name, (N, K_low, K_high) in all_shapes.items():
        K_total = K_low + K_high
        for M in mg.M_LIST:
            p, _ = mg.prep_mixed(M, N, K_low, K_high, seed=1)
            lat = mg._time_ms(lambda: b1_single(p, M, N, K_low, K_high, stages=stages))
            line = f"  {name:14s} M={M:5d} N={N:5d}: {path_name}={lat:.4f} ms"
            rows.append((name, M, path_name, f"{lat:.5f}"))
            if ref_paths:
                mg.prepack_weight_sf(p, N, K_low, K_high)
                lat_bp = mg._time_ms(lambda: mg.mixed_gemm_fused_b_presf(p, M, N, K_low, K_high))
                line += f"  presf={lat_bp:.4f} ({lat/lat_bp:.2f}x)"
                rows.append((name, M, f"rungB_presf@{path_name}", f"{lat_bp:.5f}"))
            if bounds:
                p4, p8 = prep_bounds_presf(M, N, K_total, seed=1)
                l4 = mg._time_ms(lambda: pure_fp4_presf(p4, M, N, K_total))
                l8 = mg._time_ms(lambda: pure_fp8_presf(p8, M, N, K_total))
                band = mg.band_position(lat, l4, l8)
                line += f"  fp4p={l4:.4f} fp8p={l8:.4f} band={band:.0f}%"
                rows.append((name, M, f"pure_fp4_fullK_presf@{path_name}", f"{l4:.5f}"))
                rows.append((name, M, f"pure_fp8_fullK_presf@{path_name}", f"{l8:.5f}"))
            print(line)
    csv_path = os.path.join(HERE, "benchmark.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["shape", "M", "path", "latency_ms"])
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    import sys
    path_name = sys.argv[1] if len(sys.argv) > 1 else "b1_opt"
    stages = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("Building b1_fused_ext (first build may take minutes)...")
    get_ext()
    print("Built.\n")
    ok1 = validate_step1()
    ok2 = validate_fused(stages=stages)
    ok3 = validate_all_cfgs()
    ok4 = validate_bounds_presf()
    if ok1 and ok2 and ok3 and ok4:
        benchmark_b1(path_name, stages)
    print("\nB1 DONE" if (ok1 and ok2 and ok3 and ok4) else "\nB1 VALIDATION FAILED")
