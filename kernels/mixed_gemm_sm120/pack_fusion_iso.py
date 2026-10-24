"""Lever 1 (pack fusion) gate: bit-exactness + isolated latency of the SINGLE-LAUNCH
fused activation pack (quant_pack_float_mixed) vs the current TWO-launch pack
(quant_pack_float_fp4 + quant_pack_float_fp8).

Bit-exact gate: torch.equal on all four outputs (data_lo, sfa_lo, data_hi, sfa_hi)
across the 7 Llama-3.2-1B Linear shapes x M in {2048,4096,8192} and dtype fp16.
Isolated latency: min-of-N per shape, summed x16 layers, for both packers.
"""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
DEV = "cuda"

from torch.utils.cpp_extension import load

LAYER_SHAPES = [("q_proj",2048,2048),("k_proj",512,2048),("v_proj",512,2048),
                ("o_proj",2048,2048),("gate_proj",8192,2048),("up_proj",8192,2048),
                ("down_proj",2048,8192)]
HIGH_FRACTION = 0.125

def split_k(K, hf=HIGH_FRACTION):
    Kh = int(round(hf*K)); Kl = K-Kh
    assert Kl%16==0 and Kh%32==0
    return Kl, Kh

def build():
    inc=[KERN, os.path.join(CUTLASS,"include"), os.path.join(CUTLASS,"tools/util/include"),
         os.path.join(CUTLASS,"examples/common")]
    flags=["-std=c++17","--expt-relaxed-constexpr","--expt-extended-lambda","-O3"]
    qpf = load(name="quant_pack_float_ext",
               sources=[os.path.join(KERN,"quant_pack_float_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    qpfm = load(name="quant_pack_fused_ext",
                sources=[os.path.join(KERN,"quant_pack_fused_ext.cu")],
                extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    return qpf, qpfm

def tms(fn_, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup): fn_()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn_()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3

@torch.no_grad()
def main():
    qpf, qpfm = build()
    print("built qpf (2-launch) + qpfm (1-launch)", flush=True)
    Ms=[2048,4096,8192]
    rows=[]; all_exact=True
    for M in Ms:
        two=one=0.0
        for name,N,K in LAYER_SHAPES:
            Kl,Kh = split_k(K)
            torch.manual_seed(hash(name)%1000)
            x = torch.randn(M,K,device=DEV,dtype=torch.float16)
            x[:,Kl:]*=3.0
            x=x.contiguous()
            a_lo = qpf.quant_pack_float_fp4(x, M, N, Kl, K, 0)
            a_hi = qpf.quant_pack_float_fp8(x, M, N, Kh, K, Kl)
            mix = qpfm.quant_pack_float_mixed(x, M, N, Kl, Kh)
            eq = (torch.equal(a_lo[0],mix[0]) and torch.equal(a_lo[1],mix[1]) and
                  torch.equal(a_hi[0],mix[2]) and torch.equal(a_hi[1],mix[3]))
            all_exact = all_exact and eq
            def two_launch():
                qpf.quant_pack_float_fp4(x, M, N, Kl, K, 0)
                qpf.quant_pack_float_fp8(x, M, N, Kh, K, Kl)
            def one_launch():
                qpfm.quant_pack_float_mixed(x, M, N, Kl, Kh)
            t2=tms(two_launch); t1=tms(one_launch)
            two+=t2; one+=t1
            print(f"  M={M:5d} {name:10s} exact={eq} two={t2*1e3:7.1f}us one={t1*1e3:7.1f}us", flush=True)
        two16=two*16; one16=one*16
        print(f"  M={M:5d} SUM x16: two-launch={two16:.3f}ms one-launch={one16:.3f}ms "
              f"save={two16-one16:+.3f}ms ({100*(two16-one16)/two16:+.1f}%)", flush=True)
        rows.append(dict(M=M, two_launch_ms=two16, one_launch_ms=one16, save_ms=two16-one16))
    print(f"BITEXACT_ALL {all_exact}", flush=True)
    out=os.path.join(KERN,"pack_fusion_iso.json")
    json.dump(dict(bitexact=all_exact, rows=rows), open(out,"w"), indent=2)
    print(f"RESULTS_WRITTEN {out}", flush=True)
    print("PACK_FUSION_ISO_DONE", flush=True)

if __name__=="__main__":
    main()
