"""Round-5 T3 grid sweep: dual-TMA tile (128x128 / 128x256 / 256x128) x stages,
on the overhead-dominated gate (N8192 K2048) + down (N2048 K8192), M in {2048,8192}.
Cosine-gated; excess vs same-session CUTLASS presf bounds."""
import torch
import mixed_gemm as mg
import b1_fused as b1
import b1_tma as t
ext = t.get_tma(); b1.get_ext(); b1.get_presf_ext()
def dual(p,M,N,Kl,Kh,st,tile):
    return ext.b1_fused_tma_dual(p["Al_packed"],p["SFAl"],p["Bl_packed"],p["SFBl"],
        p["Ah_bytes"],p["SFAh"],p["Bh_bytes"],p["SFBh"],p["s_x"],p["s_w"],M,N,Kl,Kh,st,tile)
def med(fn): return t.median_ms(fn)
CASES=[("gate",8192,2048),("down",2048,8192)]
RS=[0.0,0.125,1.0]
TILES={0:"128x128",1:"128x256",2:"256x128"}
for name,N,KT in CASES:
    for M in [2048,8192]:
        p4,p8=b1.prep_bounds_presf(M,N,KT,seed=1)
        t4=med(lambda:b1.pure_fp4_presf(p4,M,N,KT)); t8=med(lambda:b1.pure_fp8_presf(p8,M,N,KT))
        print(f"=== {name} M={M} N={N} K={KT}  t4={t4:.4f} t8={t8:.4f}",flush=True)
        for r in RS:
            Kh=int(round(KT*r)); Kh-=Kh%64; Kl=KT-Kh
            if Kl%128!=0: continue
            p,orc=mg.prep_mixed(M,N,Kl,Kh,seed=1); ideal=t4*(1-r)+t8*r
            line=f"  r={r:.3f}:"
            for tile in [0,1,2]:
                if tile==1 and N%256: continue
                if tile==2 and M%256: continue
                best=1e9;bst=-1;cb=False
                for st in [3,4]:
                    try:
                        D=dual(p,M,N,Kl,Kh,st,tile).float()
                    except Exception as e:
                        line+=f" {TILES[tile]}=ERR"; cb=True; break
                    c=b1.cosine(D,orc); del D
                    if c<0.999: cb=True; break
                    lt=med(lambda:dual(p,M,N,Kl,Kh,st,tile))
                    if lt<best:best,bst=lt,st
                if cb: continue
                ex=(best-ideal)/ideal*100
                line+=f" {TILES[tile]}(st{bst})={best:.4f}({ex:+.1f}%)"
            print(line,flush=True)
            del p,orc; torch.cuda.empty_cache()
print("TILE SWEEP DONE",flush=True)
