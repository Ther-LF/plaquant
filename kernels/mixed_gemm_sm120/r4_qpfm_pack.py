"""round-4: pure quant_pack overhead (the path round-3's '5.3x' was measured on).
Times quant_pack_float_mixed (the single-launch NVFP4-low+MXFP8-high packer, NO
Hadamard) OFF vs OLD(round-3) vs FAST3/2/1 at 8B down shape. Reuses the cached
flavor builds from r4_actmse.py (same names)."""
import os, sys, time, json
import torch
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(HERE,"..",".."))
KERN=os.path.join(REPO,"kernels","mixed_gemm_sm120"); CUT=os.path.join(REPO,"third_party/cutlass")
sys.path.insert(0,REPO); sys.path.insert(0,KERN); os.environ.setdefault("TORCH_CUDA_ARCH_LIST","12.0a")
from torch.utils.cpp_extension import load
FL={"OFF":[],"OLD":["-DPROMIX_ACT_MSE=1","-DPROMIX_ACT_MSE_FAST=0","-DPROMIX_ACT_MSE_STEPS=3"],
    "FAST3":["-DPROMIX_ACT_MSE=1","-DPROMIX_ACT_MSE_FAST=1","-DPROMIX_ACT_MSE_STEPS=3"],
    "FAST2":["-DPROMIX_ACT_MSE=1","-DPROMIX_ACT_MSE_FAST=1","-DPROMIX_ACT_MSE_STEPS=2"],
    "FAST1":["-DPROMIX_ACT_MSE=1","-DPROMIX_ACT_MSE_FAST=1","-DPROMIX_ACT_MSE_STEPS=1"]}
inc=[KERN,os.path.join(CUT,"include"),os.path.join(CUT,"tools/util/include"),os.path.join(CUT,"examples/common")]
def ld(f,d): return load(name=f"qpfm_{f.lower()}",sources=[os.path.join(KERN,"quant_pack_fused_ext.cu")],
    extra_include_paths=inc,extra_cuda_cflags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3"]+d,verbose=False)
def mins(fn,it=50,wu=12):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    return min(ts)*1e3
m={f:ld(f,d) for f,d in FL.items()}; print("built",flush=True)
I,KLOW,KHIGH,NGEMM=14336,12544,1792,4096
out=[]
for M in (2048,4096,8192):
    y=(torch.randn(M,I,device="cuda")*2).half().contiguous(); y[::3]*=4
    row={"M":M}
    for f in FL: row[f]=mins(lambda: m[f].quant_pack_float_mixed(y,M,NGEMM,KLOW,KHIGH))
    b=row["OFF"]; row["ovh"]={f:round(row[f]/b,3) for f in FL if f!="OFF"}
    print(f"M={M} "+" ".join(f'{f}:{row[f]:.3f}' for f in FL)+f"  ovh={row['ovh']}",flush=True)
    out.append(row); del y; torch.cuda.empty_cache()
json.dump(out,open(os.path.join(KERN,"r4_qpfm_pack.json"),"w"),indent=2)
print("R4_QPFM_DONE",flush=True)
