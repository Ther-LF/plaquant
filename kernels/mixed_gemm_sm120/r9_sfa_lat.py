"""Round-9 Lever-1 SFASYNC latency+excess before/after (no ncu). Compares sfa0 vs
sfa1 at the round-8 best kpipe per r (r=0 -> kp0, else kp1), per-shape best cfg,
3 shapes x M{2048,8192} x r{0,1/8,1/2,1}. Excess vs same-session CUTLASS presf
bounds ideal(r)=t_fp4*(1-r)+t_fp8*r. Self-contained; row-flushed CSV.
"""
import csv, os, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "r9_sfa_lat.csv")
ext = t.get_tma(); b1.get_ext(); b1.get_presf_ext()
print("BUILD_OK", flush=True)

def dual(p, M, N, Kl, Kh, tile, sw, pers, kp, sfa):
    return ext.b1_fused_tma_dual(
        p["Al_packed"], p["SFAl"], p["Bl_packed"], p["SFBl"],
        p["Ah_bytes"], p["SFAh"], p["Bh_bytes"], p["SFBh"],
        p["s_x"], p["s_w"], M, N, Kl, Kh, 3, tile, sw, 2, 1, pers, kp, sfa)

def med(fn):
    m, _ = mg._time_ms_median(fn, iters=30, warmup=8); return m

SHAPES = [("qkv", 2048, 2048, 0, 0, 0),
          ("down", 2048, 8192, 0, 0, 0),
          ("gate", 8192, 2048, 0, 8, 1)]
R = [0.0, 0.125, 0.5, 1.0]
f = open(CSV, "w", newline="", buffering=1); w = csv.writer(f)
w.writerow(["shape","M","N","Kt","r","Kl","Kh","kp","lat_sfa0","lat_sfa1","ideal",
            "ex_sfa0","ex_sfa1","delta_pp","speedup_pct"])
for name, N, KT, tile, sw, pers in SHAPES:
    for M in (2048, 8192):
        p4, p8 = b1.prep_bounds_presf(M, N, KT, seed=1)
        t4 = med(lambda: b1.pure_fp4_presf(p4, M, N, KT))
        t8 = med(lambda: b1.pure_fp8_presf(p8, M, N, KT))
        del p4, p8
        for r in R:
            Kh = KT if r == 1.0 else (0 if r == 0.0 else int(round(r*KT)))
            Kl = KT - Kh
            kp = 0 if r == 0.0 else 1
            p, _ = mg.prep_mixed(M, N, Kl, Kh, seed=1)
            ideal = t4*(1-r) + t8*r
            l0 = med(lambda: dual(p, M, N, Kl, Kh, tile, sw, pers, kp, 0))
            l1 = med(lambda: dual(p, M, N, Kl, Kh, tile, sw, pers, kp, 1))
            e0 = (l0-ideal)/ideal*100; e1 = (l1-ideal)/ideal*100
            sp = (l0-l1)/l0*100
            w.writerow([name,M,N,KT,f"{r:.3f}",Kl,Kh,kp,f"{l0:.5f}",f"{l1:.5f}",
                        f"{ideal:.5f}",f"{e0:.2f}",f"{e1:.2f}",f"{e1-e0:.2f}",f"{sp:.2f}"])
            print(f"{name} M{M} r{r:.3f} kp{kp}: sfa0={l0:.4f}({e0:+.1f}%) sfa1={l1:.4f}({e1:+.1f}%) "
                  f"d={e1-e0:+.2f}pp sp={sp:+.1f}%", flush=True)
            del p; torch.cuda.empty_cache()
f.close()
print("SFA_LAT_DONE", flush=True)
