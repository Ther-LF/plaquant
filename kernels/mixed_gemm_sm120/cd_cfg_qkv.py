"""Round-19 qkv config confirm: clean de-noised dense-r GEMM-only A/B of the two viable
128x128 co-design configs for the qkv shape -- NV4/MX2 (shipping) vs NV3/MX2 (config
search candidate winner). Both extensions already built/cached. GEMM-only (prepacked),
cosine-gated, median-of-50 on a FREE GPU with no co-tenant. Confirms whether qkv should
dispatch to NV3/MX2 and quantifies the per-r GEMM speedup."""
import os, csv, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = "/home/luofan/plaquant/third_party/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
sys.path.insert(0, HERE)
from torch.utils.cpp_extension import load
import nvfp4_gemm as nv
import mxfp8_gemm as mx
import b1_fused as b1
import mixed_gemm as mg
nv.get_ext(); mx.get_ext(); b1.get_presf_ext()

def build(lbl, NVST, MXST):
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3",
             "-DCDCFG_TM=128", "-DCDCFG_TN=128", "-DCDCFG_TK=128",
             "-DCDCFG_CM=1", "-DCDCFG_CN=1", "-DCDCFG_CK=1",
             f"-DCDCFG_NVST={NVST}", f"-DCDCFG_MXST={MXST}"]
    return load(name="cd_cfg_" + lbl, sources=[os.path.join(HERE, "cd_cfg_ext.cu")],
                extra_include_paths=[HERE, os.path.join(CUTLASS, "include"),
                                     os.path.join(CUTLASS, "tools/util/include"),
                                     os.path.join(CUTLASS, "examples/common")],
                extra_cuda_cflags=flags, verbose=False)

nv4 = build("t128x128_nv4mx2", 4, 2)
nv3 = build("t128x128_nv3mx2", 3, 2)
print("BUILD_OK both", flush=True)

N, KT = 2048, 2048  # qkv
MLIST = [512, 2048, 4096, 8192]
R = [k/16.0 for k in range(1, 16)]  # interior (dual path)

def prep(M, Kl, Kh, seed=1):
    torch.manual_seed(seed); dev="cuda"
    Xl=torch.randn(M,Kl,device=dev); Wl=torch.randn(N,Kl,device=dev)
    Alc,SFAl,Ald=nv.quantize_nvfp4(Xl); Blc,SFBl,Bld=nv.quantize_nvfp4(Wl)
    Alp=nv.pack_fp4(Alc); Blp=nv.pack_fp4(Blc)
    Xh=torch.randn(M,Kh,device=dev); Wh=torch.randn(N,Kh,device=dev)
    Ahb,SFAh,Ahd=mx.quantize_mxfp8(Xh); Bhb,SFBh,Bhd=mx.quantize_mxfp8(Wh)
    s_x=(0.5+torch.rand(M,device=dev)).float(); s_w=(0.5+torch.rand(N,device=dev)).float()
    oracle=(Ald@Bld.t()+Ahd@Bhd.t())*s_x.view(M,1)*s_w.view(1,N)
    return dict(Alp=Alp,SFAl=SFAl,Blp=Blp,SFBl=SFBl,Ahb=Ahb,SFAh=SFAh,Bhb=Bhb,SFBh=SFBh,s_x=s_x,s_w=s_w),oracle

def med(fn):
    m,_=mg._time_ms_median(fn,iters=50,warmup=10); return m*1000.0

def run(ext, p, M, Kl, Kh):
    pBlo=ext.prepack_sf_fp4_b(p["SFBl"],N,Kl); pBhi=ext.prepack_sf_fp8_b(p["SFBh"],N,Kh)
    pAlo=ext.prepack_sf_fp4_a(p["SFAl"],M,N,Kl); pAhi=ext.prepack_sf_fp8_a(p["SFAh"],M,N,Kh)
    fn=lambda: ext.fused_gemm_prepacked(p["Alp"],pAlo,p["Blp"],pBlo,p["Ahb"],pAhi,p["Bhb"],pBhi,M,N,Kl,Kh)
    return fn, pBlo,pBhi,pAlo,pAhi

CSV=os.path.join(HERE,"cd_cfg_qkv.csv")
f=open(CSV,"w",newline="",buffering=1); w=csv.writer(f)
w.writerow(["shape","M","r","Kl","Kh","cos_nv4","cos_nv3","gemm_nv4_us","gemm_nv3_us","nv3_over_nv4"])
print(f"{'M':>6s}{'r':>7s}{'nv4':>9s}{'nv3':>9s}{'nv3/nv4':>8s}", flush=True)
allok=True
for M in MLIST:
    for r in R:
        Kh=int(round(r*KT)); Kl=KT-Kh
        p,oracle=prep(M,Kl,Kh)
        fn4,*b4=run(nv4,p,M,Kl,Kh); fn3,*b3=run(nv3,p,M,Kl,Kh)
        D4=fn4().float(); c4=b1.cosine(D4*p["s_x"].view(M,1)*p["s_w"].view(1,N),oracle); del D4
        D3=fn3().float(); c3=b1.cosine(D3*p["s_x"].view(M,1)*p["s_w"].view(1,N),oracle); del D3
        if c4<0.999 or c3<0.999: allok=False; print(f"  COSFAIL M{M} r{r} c4={c4} c3={c3}",flush=True)
        t4=med(fn4); t3=med(fn3)
        w.writerow([ "qkv",M,f"{r:.4f}",Kl,Kh,f"{c4:.6f}",f"{c3:.6f}",f"{t4:.4f}",f"{t3:.4f}",f"{t3/t4:.4f}"])
        print(f"{M:6d}{r:7.3f}{t4:9.2f}{t3:9.2f}{t3/t4:8.3f}",flush=True)
        del p,oracle,b4,b3; torch.cuda.empty_cache()
f.close()
print("QKV_CFG_ALL_COSINE_PASS" if allok else "QKV_CFG_COSINE_FAIL",flush=True)
print("QKV_CFG_DONE",flush=True)
