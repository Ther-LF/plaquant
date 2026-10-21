"""Round-20 STAGE-1 PoC: cross-phase OVERLAP vs SERIAL dual-format fused GEMM.

For each (NV,MX) stage config (de-union needs shallower stages than the serial union):
  1. build codesign_overlap_ext with -DCD_NV_STAGES/-DCD_MX_STAGES (unique ext name),
  2. compat_report -> SharedStorageSize (does the de-unioned config fit ~101 KB?),
  3. gate cosine vs the SAME oracle as r19 (bit-exact shared-accumulator math), and
  4. median latency A/B: overlap GEMM-only vs serial fused_gemm_prepacked (both pre-
     interleaved activation SF -> pure GEMM mainloop) at the two mid-r cells where the
     round-19 bulge is largest: gate.8192 r=1/2 and down.8192 r=1/2.

Verdict: does removing the phase seam (overlap) beat the serial kernel at mid-r? If
not (expected from the ncu root-cause: DUAL already <= endpoint-sum, compute-bound),
that bounds the structural ceiling. CSV line-buffered; harvest from disk.
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
FLAGS = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]


def get_serial():
    return load(name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=INC, extra_cuda_cflags=FLAGS, verbose=False)


def get_overlap(nvs, mxs):
    return load(name=f"codesign_overlap_ext_nv{nvs}mx{mxs}",
        sources=[os.path.join(HERE, "codesign_overlap_ext.cu")],
        extra_include_paths=INC,
        extra_cuda_cflags=FLAGS + [f"-DCD_NV_STAGES={nvs}", f"-DCD_MX_STAGES={mxs}"],
        verbose=False)


def prep_mixed_codes(M, N, K_low, K_high, seed=1):
    torch.manual_seed(seed); dev = "cuda"
    Xl = torch.randn(M, K_low, device=dev); Wl = torch.randn(N, K_low, device=dev)
    Al_codes, SFAl, Al_deq = nv.quantize_nvfp4(Xl)
    Bl_codes, SFBl, Bl_deq = nv.quantize_nvfp4(Wl)
    Al_packed = nv.pack_fp4(Al_codes); Bl_packed = nv.pack_fp4(Bl_codes)
    Xh = torch.randn(M, K_high, device=dev); Wh = torch.randn(N, K_high, device=dev)
    Ah_bytes, SFAh, Ah_deq = mx.quantize_mxfp8(Xh)
    Bh_bytes, SFBh, Bh_deq = mx.quantize_mxfp8(Wh)
    s_x = (0.5 + torch.rand(M, device=dev)).float()
    s_w = (0.5 + torch.rand(N, device=dev)).float()
    oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t()) * s_x.view(M, 1) * s_w.view(1, N)
    return dict(Al_codes=Al_codes, Al_packed=Al_packed, SFAl=SFAl, Bl_packed=Bl_packed, SFBl=SFBl,
                Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh, s_x=s_x, s_w=s_w), oracle


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m * 1000.0  # us


CELLS = [("gate", 8192, 8192, 1024, 1024), ("down", 8192, 2048, 4096, 4096)]
CONFIGS = [(2, 2), (3, 2), (2, 1), (3, 1), (4, 2)]

serial = get_serial()
print("SERIAL_BUILD_OK", flush=True)

CSV = os.path.join(HERE, "r20_overlap.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["nv_stages", "mx_stages", "smem_bytes", "fits", "shape", "M", "N", "Kl", "Kh",
            "cos_overlap", "serial_gemm_us", "overlap_gemm_us", "overlap_over_serial"])

# Pre-build serial prepacked inputs + oracle per cell (shared across configs).
cell_data = {}
for name, M, N, Kl, Kh in CELLS:
    p, oracle = prep_mixed_codes(M, N, Kl, Kh)
    physB_lo_s = serial.prepack_sf_fp4_b(p["SFBl"], N, Kl)
    physB_hi_s = serial.prepack_sf_fp8_b(p["SFBh"], N, Kh)
    o4i_s = serial.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
    o8i_s = serial.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
    D_s = serial.fused_gemm_prepacked(o4i_s[0], o4i_s[1], p["Bl_packed"], physB_lo_s,
                                      o8i_s[0], o8i_s[1], p["Bh_bytes"], physB_hi_s,
                                      M, N, Kl, Kh).float()
    cos_s = b1.cosine(D_s * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
    t_serial = med(lambda: serial.fused_gemm_prepacked(o4i_s[0], o4i_s[1], p["Bl_packed"], physB_lo_s,
                                                       o8i_s[0], o8i_s[1], p["Bh_bytes"], physB_hi_s,
                                                       M, N, Kl, Kh))
    print(f"SERIAL {name} M{M} r0.5 cos={cos_s:.6f} gemm={t_serial:.2f}us", flush=True)
    cell_data[name] = (p, oracle, t_serial, (physB_lo_s, physB_hi_s, o4i_s, o8i_s))

allgate = True
for (nvs, mxs) in CONFIGS:
    try:
        ov = get_overlap(nvs, mxs)
    except Exception as e:
        print(f"CONFIG nv{nvs}mx{mxs} BUILD_FAIL: {type(e).__name__} {str(e).splitlines()[0][:100]}", flush=True)
        continue
    rep = ov.compat_report()
    smem = None
    for ln in rep.splitlines():
        if "SharedStorageSize" in ln:
            try:
                smem = int(ln.split("=")[1].strip().split()[0])
            except Exception:
                pass
    print(f"CONFIG nv{nvs}mx{mxs} smem={smem}", flush=True)
    for name, M, N, Kl, Kh in CELLS:
        p, oracle, t_serial, _ = cell_data[name]
        fits = "?"; cos_o = float('nan'); t_ov = float('nan')
        try:
            physB_lo = ov.prepack_sf_fp4_b(p["SFBl"], N, Kl)
            physB_hi = ov.prepack_sf_fp8_b(p["SFBh"], N, Kh)
            o4i = ov.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl)
            o8i = ov.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh)
            D = ov.fused_gemm_overlap_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                M, N, Kl, Kh).float()
            cos_o = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
            fits = "yes"
            del D
            t_ov = med(lambda: ov.fused_gemm_overlap_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                               o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                               M, N, Kl, Kh))
        except Exception as e:
            fits = "no"
            print(f"  nv{nvs}mx{mxs} {name}: {type(e).__name__} {str(e).splitlines()[0][:90]}", flush=True)
        if cos_o == cos_o and cos_o < 0.999:
            allgate = False
            print(f"  OVERLAP COSINE FAIL nv{nvs}mx{mxs} {name}: cos={cos_o:.6f}", flush=True)
        ratio = (t_ov / t_serial) if (t_ov == t_ov) else float('nan')
        w.writerow([nvs, mxs, smem, fits, name, M, N, Kl, Kh,
                    f"{cos_o:.6f}", f"{t_serial:.2f}", f"{t_ov:.2f}", f"{ratio:.4f}"])
        print(f"  nv{nvs}mx{mxs} {name} M{M}: fits={fits} cos={cos_o:.6f} "
              f"serial={t_serial:.2f} overlap={t_ov:.2f} ov/ser={ratio:.3f}", flush=True)
    torch.cuda.empty_cache()
f.close()
print("R20_OVERLAP_ALL_COSINE_PASS" if allgate else "R20_OVERLAP_COSINE_FAILED", flush=True)
print("R20_OVERLAP_DONE", flush=True)
