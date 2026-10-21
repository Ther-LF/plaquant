"""Round-20 STAGE-1 isolation A/B: prove the OVERLAP kernel is bit-exact and measure
the seam effect APPLES-TO-APPLES. A 128x64 tile halves B smem so the de-unioned
overlap fits SM120's ~101 KB at NV2/MX2 -- the SAME tile + SAME stages as a serial
cd_cfg build. The ONLY difference is union+phase-bars (serial) vs de-union+no-bars
(overlap), so overlap/serial at 128x64 isolates the seam. Production serial 128x128
NV4/MX2 is the absolute reference. gate.8192 r=1/2 and down.8192 r=1/2.
"""
import os, csv, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import mixed_gemm as mg
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1

INC = [HERE, os.path.join(CUTLASS, "include"),
       os.path.join(CUTLASS, "tools/util/include"),
       os.path.join(CUTLASS, "examples/common")]
F = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]


def serial_prod():  # 128x128 NV4/MX2 production (cached)
    return load(name="codesign_fused_ext", sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
                extra_include_paths=INC, extra_cuda_cflags=F, verbose=False)


def serial_cfg(tn, nvs, mxs):  # cd_cfg serial kernel at a chosen tile/stage
    return load(name=f"cd_cfg_t128x{tn}_nv{nvs}mx{mxs}", sources=[os.path.join(HERE, "cd_cfg_ext.cu")],
                extra_include_paths=INC,
                extra_cuda_cflags=F + [f"-DCDCFG_TN={tn}", f"-DCDCFG_NVST={nvs}", f"-DCDCFG_MXST={mxs}"],
                verbose=False)


def overlap_cfg(tn, nvs, mxs):
    return load(name=f"codesign_overlap_ext_t128x{tn}_nv{nvs}mx{mxs}",
                sources=[os.path.join(HERE, "codesign_overlap_ext.cu")], extra_include_paths=INC,
                extra_cuda_cflags=F + [f"-DCDO_TN={tn}", f"-DCD_NV_STAGES={nvs}", f"-DCD_MX_STAGES={mxs}"],
                verbose=False)


def prep(M, N, Kl, Kh, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    Xl = torch.randn(M, Kl, device=dev); Wl = torch.randn(N, Kl, device=dev)
    Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    Xh = torch.randn(M, Kh, device=dev); Wh = torch.randn(N, Kh, device=dev)
    Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
    Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
    s_x = (0.5 + torch.rand(M, device=dev)).float(); s_w = (0.5 + torch.rand(N, device=dev)).float()
    oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t()) * s_x.view(M, 1) * s_w.view(1, N)
    return dict(Al_codes=Al_codes, Al_packed=Al_packed, SFAl=SFAl, Bl_packed=Bl_packed, SFBl=SFBl,
                Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh, s_x=s_x, s_w=s_w), oracle


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m * 1000.0


CELLS = [("gate", 8192, 8192, 1024, 1024), ("down", 8192, 2048, 4096, 4096)]

prod = serial_prod(); print("PROD_BUILD_OK", flush=True)
ser64 = serial_cfg(64, 2, 2); print("SER64_BUILD_OK " + ser64.compat_report().replace(chr(10), " | "), flush=True)
ov64 = overlap_cfg(64, 2, 2); print("OV64_BUILD_OK " + ov64.compat_report().replace(chr(10), " | "), flush=True)

CSV = os.path.join(HERE, "r20_overlap_iso.csv")
f = open(CSV, "w", newline="", buffering=1); w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kl", "Kh",
            "cos_prod128", "cos_ser64", "cos_ov64",
            "prod128_us", "ser64_us", "ov64_us", "ov64_over_ser64", "ov64_over_prod128"])
allgate = True
for name, M, N, Kl, Kh in CELLS:
    p, oracle = prep(M, N, Kl, Kh)
    # Shared pre-interleaved inputs (atom layout is tile-independent; built once via overlap ext).
    o4 = ov64.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl)
    o8 = ov64.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh)
    pBlo = ov64.prepack_sf_fp4_b(p["SFBl"], N, Kl); pBhi = ov64.prepack_sf_fp8_b(p["SFBh"], N, Kh)
    args = (o4[0], o4[1], p["Bl_packed"], pBlo, o8[0], o8[1], p["Bh_bytes"], pBhi, M, N, Kl, Kh)
    sx = p["s_x"].view(M, 1); sw = p["s_w"].view(1, N)

    def cosof(D):
        return b1.cosine(D.float() * sx * sw, oracle)
    c_prod = cosof(prod.fused_gemm_prepacked(*args))
    c_ser = cosof(ser64.fused_gemm_prepacked(*args))
    c_ov = cosof(ov64.fused_gemm_overlap_prepacked(*args))
    for nm, c in [("prod", c_prod), ("ser64", c_ser), ("ov64", c_ov)]:
        if c < 0.999:
            allgate = False; print(f"  COSINE FAIL {name} {nm}: {c:.6f}", flush=True)
    t_prod = med(lambda: prod.fused_gemm_prepacked(*args))
    t_ser = med(lambda: ser64.fused_gemm_prepacked(*args))
    t_ov = med(lambda: ov64.fused_gemm_overlap_prepacked(*args))
    w.writerow([name, M, N, Kl, Kh, f"{c_prod:.6f}", f"{c_ser:.6f}", f"{c_ov:.6f}",
                f"{t_prod:.2f}", f"{t_ser:.2f}", f"{t_ov:.2f}",
                f"{t_ov/t_ser:.4f}", f"{t_ov/t_prod:.4f}"])
    print(f"{name} M{M}: cos prod={c_prod:.6f} ser64={c_ser:.6f} ov64={c_ov:.6f} | "
          f"prod128={t_prod:.1f} ser64={t_ser:.1f} ov64={t_ov:.1f} | "
          f"ov/ser64={t_ov/t_ser:.3f} ov/prod128={t_ov/t_prod:.3f}", flush=True)
    del p, oracle, o4, o8; torch.cuda.empty_cache()
f.close()
print("R20_ISO_ALL_COSINE_PASS" if allgate else "R20_ISO_COSINE_FAILED", flush=True)
print("R20_ISO_DONE", flush=True)
