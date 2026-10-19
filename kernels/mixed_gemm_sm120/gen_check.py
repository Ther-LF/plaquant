import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t
ext = t.get_tma(); b1.get_ext(); b1.get_presf_ext()
def dual(p,M,N,Kl,Kh,st):
    return ext.b1_fused_tma_dual(p["Al_packed"],p["SFAl"],p["Bl_packed"],p["SFBl"],
        p["Ah_bytes"],p["SFAh"],p["Bh_bytes"],p["SFBh"],p["s_x"],p["s_w"],M,N,Kl,Kh,st)
def med(fn): return t.median_ms(fn)
SC=[("qkv",2048,2048,2048),("down",2048,2048,8192),("gate",2048,8192,2048)]
print("=== generalization: dual-TMA excess vs CUTLASS ideal (all cosine-gated) ===")
for name,M,N,KT in SC:
    p4,p8=b1.prep_bounds_presf(M,N,KT,seed=1)
    t4=med(lambda:b1.pure_fp4_presf(p4,M,N,KT)); t8=med(lambda:b1.pure_fp8_presf(p8,M,N,KT))
    rs=[0.0,0.125,0.5,1.0]
    print(f"-- {name} N={N} K={KT}  t4(fp4)={t4:.4f} t8(fp8)={t8:.4f}")
    for r in rs:
        Kh=int(round(KT*r)); Kh-=Kh%64; Kl=KT-Kh
        if Kl%128!=0: continue
        p,orc=mg.prep_mixed(M,N,Kl,Kh,seed=1)
        best=1e9;bst=-1
        for st in [3,4]:
            D=dual(p,M,N,Kl,Kh,st).float(); c=b1.cosine(D,orc)
            if c<0.999: print(f"   r={r} FAIL cos={c}"); continue
            lt=med(lambda:dual(p,M,N,Kl,Kh,st))
            if lt<best: best,bst=lt,st
        lcp=med(lambda:b1.b1_single(p,M,N,Kl,Kh))
        ideal=t4*(1-r)+t8*r
        print(f"   r={r:.3f} Kl={Kl:5d} Kh={Kh:5d}: ideal={ideal:.4f} dualTMA(st{bst})={best:.4f}({(best-ideal)/ideal*100:+.1f}%) cp.async={lcp:.4f}({(lcp-ideal)/ideal*100:+.1f}%)")
print("GEN DONE")
