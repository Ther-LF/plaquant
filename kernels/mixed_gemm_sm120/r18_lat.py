"""Round-18 REALISTIC-CEILING fair A/B: fold the activation-SF interleave into the
quantize/pack step, then measure the fair end-to-end A/B at the ceiling.

Context (from round-17): the co-design wins the GEMM-only ceiling 5/6 shapes, but a
residual ~9-20us per-call standalone activation-SF interleave (cd_fill_sf_block_kernel)
kept cd_full behind the shipping kernel on small shapes. In a REAL inference pipeline the
activation is quantized/packed per call anyway -- that step touches every element and
emits the block SF. If it emits the SF DIRECTLY in the interleaved atom layout there is
NO separate interleave launch and the co-design reaches its ceiling for free.

This harness measures the FOLD:
  * cd.quant_pack_fp4/fp8(codes/bytes, SF, M,N,K, ilv): ONE pass = data pack + SF layout.
    ilv=True  -> interleaved atom SF (co-design consumes via fused_gemm_prepacked, NO
                 separate interleave; GEMM-only ceiling reached per call).
    ilv=False -> natural contiguous SF (shipping kernel's native format).
    Data pack is byte-identical for both; only the SF store address differs.

Gate (BEFORE timing):
  (A) fold bit-exactness: quant_pack(ilv=True).SF == standalone prepack_sf_*_a(SF) and
      quant_pack(ilv=*).data == torch pack; quant_pack(ilv=False).SF == natural SF.
  (B) cosine (hard gate) all cells: co-design GEMM fed the fused-pack ILV output
      (fused_gemm_prepacked) vs the float oracle.

Timing (warm, median-of-50, matched clock, free GPU), per shape x M{2048,8192} x r:
  * t_q_nat = quant_pack NAT (lo fp4 + hi fp8)   -- what shipping's per-call pack costs
  * t_q_ilv = quant_pack ILV (lo fp4 + hi fp8)   -- folded interleaved-SF pack
  * delta_q = t_q_ilv - t_q_nat                  -- IS THE FOLD FREE?
  * ship_gemm / cd_gemmonly (prepacked) / cd_full (r17 separate-interleave reference)
  * REALISTIC e2e (combined, one timed region, captures no-overlap dependency):
      ship_e2e = quant_pack_nat ; ship_gemm
      cd_e2e   = quant_pack_ilv ; fused_gemm_prepacked   (ceiling, folded)
  * excess vs t_ideal(r)=t_fp4(1-r)+t_fp8*r (CUTLASS pure-presf endpoints, same session).
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
print(cd.compat_report(), flush=True)

# (name, N, KT, tile, sw, pers) -- shipping per-shape config (round-12/15)
SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
R = [0.0, 0.125, 0.5, 1.0]


def prep_mixed_codes(M, N, K_low, K_high, seed=0):
    """Mirror mixed_gemm.prep_mixed EXACTLY (same seed + op order so oracle/SF match the
    r17-gated path) but ALSO keep the NVFP4 codes / MXFP8 bytes the fused-pack kernel needs."""
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
    m, _ = mg._time_ms_median(fn, iters=50, warmup=10)
    return m * 1000.0  # us


# ---- GATE A: fold bit-exactness ----
def probe_fold():
    print("=== GATE A: fold bit-exactness (quant_pack ILV SF == standalone interleave; data == torch pack) ===", flush=True)
    ok = True
    for name, N, KT, *_ in SHAPES:
        M = 2048
        Kh = KT // 8; Kl = KT - Kh
        p, _ = prep_mixed_codes(M, N, Kl, Kh, seed=1)
        # fp4
        o4i = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
        o4n = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, False)
        sf4_ref = cd.prepack_sf_fp4_a(p["SFAl"], M, N, Kl)
        d4 = bool(torch.equal(o4i[0], p["Al_packed"])) and bool(torch.equal(o4n[0], p["Al_packed"]))
        s4i = bool(torch.equal(o4i[1], sf4_ref))
        s4n = bool(torch.equal(o4n[1], p["SFAl"].contiguous().view(-1)))
        # fp8
        o8i = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
        o8n = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, False)
        sf8_ref = cd.prepack_sf_fp8_a(p["SFAh"], M, N, Kh)
        d8 = bool(torch.equal(o8i[0], p["Ah_bytes"])) and bool(torch.equal(o8n[0], p["Ah_bytes"]))
        s8i = bool(torch.equal(o8i[1], sf8_ref))
        s8n = bool(torch.equal(o8n[1], p["SFAh"].contiguous().view(-1)))
        cell = d4 and s4i and s4n and d8 and s8i and s8n
        print(f"  {name:4s} fp4[data={d4} sfILV={s4i} sfNAT={s4n}] "
              f"fp8[data={d8} sfILV={s8i} sfNAT={s8n}]  {'OK' if cell else 'FAIL'}", flush=True)
        ok = ok and cell
        del p, o4i, o4n, o8i, o8n, sf4_ref, sf8_ref; torch.cuda.empty_cache()
    print("FOLD_BITEXACT_PASS" if ok else "FOLD_BITEXACT_FAIL", flush=True)
    return ok


fold_ok = probe_fold()

CSV = os.path.join(HERE, "r18_lat.csv")
f = open(CSV, "w", newline="", buffering=1)
w = csv.writer(f)
w.writerow(["shape", "M", "N", "Kt", "r", "Kl", "Kh", "cos_cd",
            "t_q_nat_us", "t_q_ilv_us", "delta_q_us",
            "ship_gemm_us", "cd_gemmonly_us", "cd_full_us",
            "ship_e2e_us", "cd_e2e_us", "ideal_us",
            "ex_ship_e2e_pct", "ex_cd_e2e_pct", "cd_e2e_over_ship_e2e",
            "cd_gemmonly_over_ship_gemm"])
hdr = (f"{'shape':9s}{'M':>6s}{'r':>6s}{'cos':>9s}{'qNAT':>7s}{'qILV':>7s}{'dQ':>6s}"
       f"{'shipG':>8s}{'cdG':>8s}{'cdF':>8s}{'shipE':>8s}{'cdE':>8s}{'ideal':>8s}"
       f"{'exSE':>7s}{'exCE':>7s}{'E/E':>6s}{'G/G':>6s}")
print(hdr, flush=True)
allgate = True
for name, N, KT, tile, sw, pers in SHAPES:
    for M in (2048, 8192):
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
            tqN = tqI = dQ = shipG = cdG = cdF = shipE = cdE = float('nan')
            try:
                if r == 1.0:
                    pass  # co-design has no pure-MXFP8 (K_low=0) path
                elif r == 0.0:
                    # pure NVFP4: co-design single path interleaves SF internally (no fold
                    # ceiling variant); gate cosine via the existing single path, report the
                    # fp4 fold cost (NAT vs ILV) and cd_full = single GEMM.
                    physB = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    D = cd.single_nvfp4_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB,
                                             M, N, Kl).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                    tqN = med(lambda: cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, False)[0])
                    tqI = med(lambda: cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)[0])
                    dQ = tqI - tqN
                    cdF = med(lambda: cd.single_nvfp4_gemm(p["Al_packed"], p["SFAl"],
                                                           p["Bl_packed"], physB, M, N, Kl))
                else:
                    physB_lo = cd.prepack_sf_fp4_b(p["SFBl"], N, Kl)
                    physB_hi = cd.prepack_sf_fp8_b(p["SFBh"], N, Kh)
                    # fused-pack ILV output -> co-design ceiling (NO separate interleave)
                    o4i = cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)
                    o8i = cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)
                    D = cd.fused_gemm_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                M, N, Kl, Kh).float()
                    cos_cd = b1.cosine(D * p["s_x"].view(M, 1) * p["s_w"].view(1, N), oracle)
                    del D
                    # fold cost: NAT vs ILV quant/pack (lo fp4 + hi fp8)
                    tqN = (med(lambda: cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, False)[0])
                           + med(lambda: cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, False)[0]))
                    tqI = (med(lambda: cd.quant_pack_fp4(p["Al_codes"], p["SFAl"], M, N, Kl, True)[0])
                           + med(lambda: cd.quant_pack_fp8(p["Ah_bytes"], p["SFAh"], M, N, Kh, True)[0]))
                    dQ = tqI - tqN
                    # GEMM-only components
                    shipG = med(lambda: ship_data(p["Al_packed"], p["SFAl"], p["Ah_bytes"], p["SFAh"],
                                                  p, M, N, Kl, Kh, tile, sw, pers, kp))
                    cdG = med(lambda: cd.fused_gemm_prepacked(o4i[0], o4i[1], p["Bl_packed"], physB_lo,
                                                              o8i[0], o8i[1], p["Bh_bytes"], physB_hi,
                                                              M, N, Kl, Kh))
                    cdF = med(lambda: cd.fused_gemm(p["Al_packed"], p["SFAl"], p["Bl_packed"], physB_lo,
                                                    p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], physB_hi,
                                                    M, N, Kl, Kh))
                    # REALISTIC e2e: pack+GEMM in one timed region (dependent, no overlap)
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
                print(f"  CD ERR {name} M{M} r{r}: {type(e).__name__} "
                      f"{str(e).splitlines()[0][:90]}", flush=True)
            if cos_cd == cos_cd and cos_cd < 0.999:
                allgate = False
                print(f"  CD COSINE FAIL {name} M{M} r{r}: cos={cos_cd:.6f}", flush=True)
            exSE = (shipE - ideal) / ideal * 100 if shipE == shipE else float('nan')
            exCE = (cdE - ideal) / ideal * 100 if cdE == cdE else float('nan')
            EE = (cdE / shipE) if (cdE == cdE and shipE == shipE) else float('nan')
            GG = (cdG / shipG) if (cdG == cdG and shipG == shipG) else float('nan')
            w.writerow([name, M, N, KT, f"{r:.3f}", Kl, Kh, f"{cos_cd:.6f}",
                        f"{tqN:.4f}", f"{tqI:.4f}", f"{dQ:.4f}",
                        f"{shipG:.4f}", f"{cdG:.4f}", f"{cdF:.4f}",
                        f"{shipE:.4f}", f"{cdE:.4f}", f"{ideal:.4f}",
                        f"{exSE:.2f}", f"{exCE:.2f}", f"{EE:.4f}", f"{GG:.4f}"])
            print(f"{name:9s}{M:6d}{r:6.3f}{cos_cd:9.6f}{tqN:7.2f}{tqI:7.2f}{dQ:6.2f}"
                  f"{shipG:8.2f}{cdG:8.2f}{cdF:8.2f}{shipE:8.2f}{cdE:8.2f}{ideal:8.2f}"
                  f"{exSE:7.1f}{exCE:7.1f}{EE:6.2f}{GG:6.2f}", flush=True)
            del p, oracle; torch.cuda.empty_cache()
f.close()
print("R18_FOLD_BITEXACT_OK" if fold_ok else "R18_FOLD_BITEXACT_FAIL", flush=True)
print("R18_LAT_ALL_COSINE_PASS" if allgate else "R18_LAT_COSINE_FAILED", flush=True)
print("R18_LAT_DONE", flush=True)
