import torch, json, sys
E=torch.load('/home/luofan/plaquant/rotation/E-wikitext-512-llama-3-8b.bin', weights_only=False)
d=4096; NL=32
layvar=[torch.as_tensor(E[f'layer.{i}.attn_mlp_var']).double().flatten() for i in range(NL)]
Ks=[4096,4096,4096,4096,4096,4096,14336]  # q,k,v,o,gate,up,down
# legal K_high per K: mult of 32 AND K_low=K-K_high mult of 64. K%64==0 -> K_high mult 64.
def legal_khighs(K):
    return [kh for kh in range(0,K+1,32) if kh%32==0 and (K-kh)%64==0]
# shared per-layer r grid legal for ALL Ks: r*K must be a legal K_high for every K.
# both 3072,8192 %64==0 -> need r*K mult of 64 for each -> r mult of 1/16 (since lcm).
RGRID=[i/16 for i in range(0,9)]  # 0,1/16,...,1/2
def tail_energy(var,r):
    k=int(round(r*d)); return 0.0 if k==0 else float(var[-k:].sum()/var.sum())
def ideal_r(var,tau):
    # continuous-ish: smallest r on a FINE grid reaching tau (for reporting)
    fine=[i/96 for i in range(0,49)]
    for r in fine:
        if tail_energy(var,r)>=tau: return r
    return fine[-1]
def snap_r(var,tau):
    for r in RGRID:
        if tail_energy(var,r)>=tau: return r
    return RGRID[-1]
def meanr_snap(tau): return sum(snap_r(layvar[i],tau) for i in range(NL))/NL
def plan_for(targetbits):
    target=(targetbits-4.5)/3.75
    lo,hi=0.0,1.0
    for _ in range(80):
        mid=(lo+hi)/2
        if meanr_snap(mid)<target: lo=mid
        else: hi=mid
    tau=(lo+hi)/2
    ri={i:snap_r(layvar[i],tau) for i in range(NL)}
    ideal={i:ideal_r(layvar[i],tau) for i in range(NL)}
    return tau,ri,ideal
for tb in [4.734,4.852,4.969]:
    tau,ri,ideal=plan_for(tb)
    mr=sum(ri.values())/NL; B=4.5+3.75*mr
    import collections
    hist=collections.Counter(round(v,4) for v in ri.values())
    snapgap=sum(abs(ri[i]-ideal[i]) for i in range(NL))/NL
    plan={str(i):ri[i] for i in range(NL)}
    name=f'/home/luofan/round13_results/planL8b_{tb}.json'
    json.dump(plan,open(name,'w'))
    # verify all legal
    ok=all(all((int(round(ri[i]*K))%32==0 and (K-int(round(ri[i]*K)))%64==0) for K in Ks) for i in range(NL))
    print(f'targetbits={tb} tau={tau:.4f} mean_r={mr:.5f} realbits={B:.4f} legal_ALL={ok}')
    print('   snapped r_i dist:',dict(sorted(hist.items())),' mean|ideal-snap|=%.4f'%snapgap)
    print('   wrote',name)
