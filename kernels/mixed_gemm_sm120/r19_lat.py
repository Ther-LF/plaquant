"""Round-19 DENSE-r MASTER TABLE (fixed current config). For EVERY shape x M x a
dense r sweep (step 1/16, 0..1 = 17 points), measure the realistic e2e of BOTH
production kernels and the best = min(cd_e2e, ship_e2e), vs the zero-overhead
envelope t_ideal(r)=t_fp4*(1-r)+t_fp8*r (CUTLASS pure-presf endpoints, same session).

NO kernel logic changes. Reuses the cached codesign_fused_ext .so (md5-matched) and
the r18 fused-pack + GEMM machinery. This is step 1 of round-19: locate headroom with
the CURRENT fixed co-design config (128x128x128, NV4/MX2). Step 2 (grid config search)
is a separate detached build+measure (cd_cfg_ext.cu).

r validity: for KT in {2048,8192}, k/16 splits give K_high=k*(KT/16) = k*128 or k*512,
both multiples of 64 (so K_low mult 64) and 32 (K_high) -> ALL 17 points land EXACTLY
on valid K-splits, no snapping. r=0 pure-NVFP4 (co-design single path), r=1 pure-MXFP8
(co-design has no exposed K_low=0 path -> endpoint reference only).

Per cell:
  * t_fp4,t_fp8,ideal(r)
  * ship_e2e = quant_pack_nat ; b1_fused_tma_dual   (shipping per-call)
  * cd_e2e   = quant_pack_ilv ; fused_gemm_prepacked (co-design ceiling, folded pack)
  * best = min(ship_e2e, cd_e2e) ; best_kernel ; excess_best = best/ideal - 1
  * cd_gemmonly / ship_gemm (GEMM-only envelope context)
  * cos_cd (hard gate)
CSV line-buffered; harvest from disk.
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
import b1_tma as bt


def get_cd():
    return load(name="codesign_fused_ext",
        sources=[os.path.join(HERE, "codesign_fused_ext.cu")],
        extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools/util/include"),
                             os.path.join(CUTLASS, "examples/common")],
        extra_cuda_cflags=["-std=c++17", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-O3"],
        verbose=True)


cd = get_cd()
print("CD_BUILD_OK", flush=True)
tma = bt.get_tma()
b1.get_ext(); b1.get_presf_ext()
nv.get_ext(); mx.get_ext()
print("ALL_BUILD_OK", flush=True)

# (name, N, KT, tile, sw, pers) -- shipping per-shape config (round-12/15)
SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
MLIST = [2048, 8192, 4096, 512]
R = [k / 16.0 for k in range(17)]
ITERS = 30
WARMUP = 10


def prep_mixed_codes(M, N, K_low, K_high, seed=0):
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
    oracle = (Al_deq @ Bl_deq.t() + Ah_deq @ Bh_deq.t())
    oracle = oracle * s_x.view(M, 1) * s_w.view(1, N)
    pack = dict(Al_codes=Al_codes, Al_packed=Al_packed, SFAl=SFAl, Bl_packed=Bl_packed, SFBl=SFBl,
                Ah_bytes=Ah_bytes, SFAh=SFAh, Bh_bytes=Bh_bytes, SFBh=SFBh,
                s_x=s_x, s_w=s_w)
    return pack, oracle


def ship_data(Al_packed, SFAl, Ah_bytes, SFAh, p, M, N, Kl, Kh, tile, sw, pers, kp):
    return tma.b1_fused_tma_dual(
        Al_packed, SFAl, p["Bl_packed"], p["SFBl"],
        Ah_bytes, SFAh, p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, 1, 0, 0, 0)


def med(fn):
    m, _ = mg._time_ms_median(fn, iters=ITERS, warmup=WARMUP)
    return m * 1000.0  # us


CSV = os.path.join(HERE, "r19_lat.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "cos_cd",
            "t_fp4_us", "t_fp8_us", "ideal_us",
            "ship_gemm_us", "cd_gemmonly_us",
            "ship_e2e_us", "cd_e2e_us", "best_us", "best_kernel",
            "excess_best_pct", "ex_ship_e2e_pct", "ex_cd_e2e_pct",
            "cd_e2e_over_ship_e2e", "cd_gemmonly_over_ship_gemm"])
hdr = (f"{'shape':9s}{'M':>6s}{'r':>6s}{'cos':>9s}{'t4':>7s}{'t8':>7s}{'ideal':>8s}"
       f"{'shipE':>8s}{'cdE':>8s}{'best':>8s}{'win':>6s}{'exB%':>7s}{'E/E':>6s}")
print(hdr, flush=True)
allgate = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in MLIST:
        p4b, p8b = b1.prep_bounds_presf(M, N, KT, seed=1)
        t4 = med(lambda: b1.pure_fp4_presf(p4b, M, N, KT))
        t8 = med(lambda: b1.pure_fp8_presf(p8b, M, N, KT))
        del p4b, p8b; torch.cuda.empty_cache()
        for r in R:
            Kh = KT if r == 1.0 else (0 if r == 0.0 else int(round(r * KT)))
            Kl = KT - Kh
            kp = 0 if r == 0.0 else 1
            p, oracle = prep_mixed_codes(M, N, Kl, Kh, seed=1)
            ideal = t4 * (1 - r) + t8 * r
            cos_cd = float('nan')
            shipG = cdG = shipE = cdE = float('nan')
            try:
                if r == 1.0:
                    # pure MXFP8: co-design has no K_low=0 path. ship kernel also needs Kl>0
                    # for dual path -> record endpoint (ideal=t_fp8) only; no kernel best.
                    pass
                elif r == 0.0:
                    # pure NVFP4: co-design single path (SF interleaved internally).
                    physB = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    D = cd.single_nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB,
                                             M, N, Kl).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                    def cd_e2e0():
                        a4 = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
                        return cd.single_nvfp4_gemm(a4[0], p["SFAl"], p["Bl_packed"], physB, M, N, Kl)
                    # NOTE: single path re-interleaves SF internally; cd_e2e0 pack cost is the
                    # fold model, GEMM is the NVFP4 endpoint kernel. ship has no pure-fp4 dual.
                    cdG = med(lambda: cd.single_nvfp4_gemm(p["Al_packed"], p["SFAl"],
                                                           p["Bl_packed"], physB, M, N, Kl))
                    cdE = med(cd_e2e0)
                else:
                    physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
                    o4i = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
                    o8i = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
                    D = cd.fused_gemm_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                M, N, Kl, Kh).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                    shipG = med(lambda: ship_data(p["Al_packed"], p["SFAl"], p["Ah_bytes"], p["SFAh"],
                                                  p, M, N, Kl, Kh, tile, sw, pers, kp))
                    cdG = med(lambda: cd.fused_gemm_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                              o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                              M, N, Kl, Kh))
                    def ship_e2e():
                        a4 = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, False)
                        a8 = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, False)
                        return ship_data(a4[0], a4[1], a8[0], a8[1], p, M, N, Kl, Kh, tile, sw, pers, kp)
                    def cd_e2e():
                        a4 = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
                        a8 = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
                        return cd.fused_gemm_prepacked(a4[0], a4[1], p["Bl_packed"], physB_lo,
                                                       a8[0], a8[1], p["Bh_bytes"], physB_hi,
                                                       M, N, Kl, Kh)
                    shipE = med(ship_e2e); cdE = med(cd_e2e)
                    del o4i, o8i
            except Exception as e:
                print(f"  ERR {name} M{M} r{r:.4f}: {type(e).__name__} "
                      f"{str(e).splitlines()[0][:90]}", flush=True)
            if cos_cd == cos_cd and cos_cd < 0.999:
                allgate = False
                print(f"  CD COSINE FAIL {name} M{M} r{r:.4f}: cos={cos_cd:.6f}", flush=True)
            # best = min of the realistic e2e that exist
            cands = []
            if shipE == shipE: cands.append((shipE, "ship"))
            if cdE == cdE: cands.append((cdE, "codesign"))
            if cands:
                best, bk = min(cands, key=lambda x: x[0])
            else:
                best, bk = float('nan'), ""
            exB = (best - ideal) / ideal * 100 if best == best else float('nan')
            exSE = (shipE - ideal) / ideal * 100 if shipE == shipE else float('nan')
            exCE = (cdE - ideal) / ideal * 100 if cdE == cdE else float('nan')
            EE = (cdE / shipE) if (cdE == cdE and shipE == shipE) else float('nan')
            GG = (cdG / shipG) if (cdG == cdG and shipG == shipG) else float('nan')
            w.writerow([name, M, N, KT, f"{r:.4f}", Kl, Kh, f"{cos_cd:.6f}",
                        f"{t4:.4f}", f"{t8:.4f}", f"{ideal:.4f}",
                        f"{shipG:.4f}", f"{cdG:.4f}",
                        f"{shipE:.4f}", f"{cdE:.4f}", f"{best:.4f}", bk,
                        f"{exB:.2f}", f"{exSE:.2f}", f"{exCE:.2f}",
                        f"{EE:.4f}", f"{GG:.4f}"])
            print(f"{name:9s}{M:6d}{r:6.3f}{cos_cd:9.6f}{t4:7.1f}{t8:7.1f}{ideal:8.1f}"
                  f"{shipE:8.1f}{cdE:8.1f}{best:8.1f}{bk:>6s}{exB:7.1f}{EE:6.2f}", flush=True)
            del p, oracle; torch.cuda.empty_cache()
f.close()
print("R19_LAT_ALL_COSINE_PASS" if allgate else "R19_LAT_COSINE_FAILED", flush=True)
print("R19_LAT_DONE", flush=True)
