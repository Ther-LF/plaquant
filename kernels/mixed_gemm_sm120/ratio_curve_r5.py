"""Round-5 dense Pareto curve: dual-TMA (fixed) vs cp.async vs CUTLASS presf
bounds, excess vs ideal interpolation. M in {2048, 8192}, regular K_high grids
(lead directive). Flushes per row to ratio_curve_r5.csv for plotting.
    CUDA_VISIBLE_DEVICES=<free> python ratio_curve_r5.py
"""
import csv, os, torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as bt
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "ratio_curve_r5.csv")
SCEN = [("qkv_class",2048,2048),("down_class",2048,8192),("gate_class",8192,2048)]  # (name,N,K_total)
MS = [2048, 8192]

def kh_list(KT):
    if KT == 2048: ks = set(range(0, KT+1, 128))
    else: ks = set(range(0, KT+1, 512)) | {128, 384}
    return sorted(k for k in ks if 0<=k<=KT and k%64==0 and (KT-k)%128==0)

def med(fn): 
    m,_ = mg._time_ms_median(fn, iters=50, warmup=10); return m

def dual(p,M,N,Kl,Kh,st):
    return bt.get_tma().b1_fused_tma_dual(p["Al_packed"],p["SFAl"],p["Bl_packed"],p["SFBl"],
        p["Ah_bytes"],p["SFAh"],p["Bh_bytes"],p["SFBh"],p["s_x"],p["s_w"],M,N,Kl,Kh,st)

def main():
    b1.get_ext(); b1.get_presf_ext(); bt.get_tma()
    f=open(CSV,"w",newline=""); w=csv.writer(f)
    w.writerow(["scenario","M","N","K_total","K_high","r","dual_ms","dual_st","cp_ms",
                "best_ms","best_variant","t_fp4","t_fp8","t_ideal","excess_best","excess_dual","cosine"])
    f.flush()
    for (scen,N,KT) in SCEN:
        for M in MS:
            p4,p8=b1.prep_bounds_presf(M,N,KT,seed=1)
            c4=b1.cosine(b1.pure_fp4_presf(p4,M,N,KT).float(),p4["oracle"])
            c8=b1.cosine(b1.pure_fp8_presf(p8,M,N,KT).float(),p8["oracle"])
            t4=med(lambda:b1.pure_fp4_presf(p4,M,N,KT)); t8=med(lambda:b1.pure_fp8_presf(p8,M,N,KT))
            print(f"=== {scen} M={M} N={N} K={KT}  t4={t4:.4f} t8={t8:.4f} boundscos={c4:.5f},{c8:.5f}",flush=True)
            del p4,p8; torch.cuda.empty_cache()
            for Kh in kh_list(KT):
                Kl=KT-Kh; r=Kh/KT
                p,orc=mg.prep_mixed(M,N,Kl,Kh,seed=1)
                best=1e9; bst=-1; cosbad=False
                for st in [3,4,5]:
                    D=dual(p,M,N,Kl,Kh,st).float(); c=b1.cosine(D,orc); del D
                    if c<0.999: cosbad=True; continue
                    lt=med(lambda:dual(p,M,N,Kl,Kh,st))
                    if lt<best: best,bst=lt,st
                cos=c
                lcp=med(lambda:b1.b1_single(p,M,N,Kl,Kh))
                bl,bv=(best,f"dualTMA_st{bst}") if best<=lcp else (lcp,"cp.async")
                ti=t4*(1-r)+t8*r
                exb=(bl-ti)/ti*100; exd=(best-ti)/ti*100
                w.writerow([scen,M,N,KT,Kh,f"{r:.4f}",f"{best:.5f}",bst,f"{lcp:.5f}",f"{bl:.5f}",bv,
                            f"{t4:.5f}",f"{t8:.5f}",f"{ti:.5f}",f"{exb:.2f}",f"{exd:.2f}",f"{cos:.6f}"])
                f.flush()
                print(f"  Kh={Kh:5d} r={r:.3f} dual={best:.4f}(st{bst},{exd:+.1f}%) cp={lcp:.4f} best={bv} exBest={exb:+.1f}% cos={cos:.6f}",flush=True)
                del p,orc; torch.cuda.empty_cache()
    f.close(); print("R5 CURVE DONE",flush=True)

if __name__=="__main__": main()
