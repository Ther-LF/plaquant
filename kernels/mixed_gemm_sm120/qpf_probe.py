"""Bit-exact probe for quant_pack_float (NEW fused float->code+pack+ILV-SF kernel).

GATE: the packed data AND interleaved SF produced by quant_pack_float_fp4/fp8 MUST be
torch.equal to the existing torch-quant (quantize_nvfp4_spec/quantize_mxfp8_spec) +
cd.quant_pack_fp4/fp8(ilv=True) path. If equal, the fused-GEMM output is unchanged.

Run inside luofan-plaquant on a FREE GPU. Line-buffered; prints QPF_BITEXACT_PASS/FAIL.
"""
import os, sys, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = HERE
CUTLASS = os.path.join(REPO, "third_party", "cutlass")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from promix.quantize.weight_pack import split_k, quantize_nvfp4_spec, quantize_mxfp8_spec

LAYER_SHAPES = [
    ("q_proj", 2048, 2048), ("k_proj", 512, 2048), ("v_proj", 512, 2048),
    ("o_proj", 2048, 2048), ("gate_proj", 8192, 2048), ("up_proj", 8192, 2048),
    ("down_proj", 2048, 8192),
]
HIGH_FRACTION = 0.125


def load_exts():
    from torch.utils.cpp_extension import load
    import nvfp4_gemm as nv, mxfp8_gemm as mx
    nv.get_ext(); mx.get_ext()
    inc = [KERN, os.path.join(CUTLASS, "include"),
           os.path.join(CUTLASS, "tools/util/include"), os.path.join(CUTLASS, "examples/common")]
    flags = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"]
    cd = load(name="codesign_fused_ext", sources=[os.path.join(KERN, "codesign_fused_ext.cu")],
              extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    qpf = load(name="quant_pack_float_ext", sources=[os.path.join(KERN, "quant_pack_float_ext.cu")],
               extra_include_paths=inc, extra_cuda_cflags=flags, verbose=False)
    return nv, mx, cd, qpf


def main():
    dev = "cuda"
    print("Loading extensions (JIT build may take minutes)...", flush=True)
    nv, mx, cd, qpf = load_exts()
    allok = True
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        print(f"=== dtype={dtype} ===", flush=True)
        for name, N, K in LAYER_SHAPES:
            Kl, Kh = split_k(K, HIGH_FRACTION)
            for M in (2048, 4096):
                torch.manual_seed(1234 + M)
                x = torch.randn(M, K, device=dev, dtype=dtype)
                x[:, Kl:] *= 3.0  # emulate high-variance-last
                x_lo = x[:, :Kl].contiguous()
                x_hi = x[:, Kl:].contiguous()
                # reference (torch quant + existing fused pack, ilv=True)
                cl, SFAl, _ = quantize_nvfp4_spec(x_lo.float())
                bh, SFAh, _ = quantize_mxfp8_spec(x_hi.float())
                ref4 = cd.quant_pack_fp4(cl, SFAl, M, N, Kl, True)
                ref8 = cd.quant_pack_fp8(bh, SFAh, M, N, Kh, True)
                # new fused float kernel (contiguous segments)
                new4 = qpf.quant_pack_float_fp4(x_lo, M, N, Kl)
                new8 = qpf.quant_pack_float_fp8(x_hi, M, N, Kh)
                # strided path (full x read in place, col_off) — what MixedQuantLinear uses
                xc = x.contiguous()
                st4 = qpf.quant_pack_float_fp4(xc, M, N, Kl, row_stride=K, col_off=0)
                st8 = qpf.quant_pack_float_fp8(xc, M, N, Kh, row_stride=K, col_off=Kl)
                d4 = bool(torch.equal(new4[0], ref4[0])) and bool(torch.equal(st4[0], ref4[0]))
                s4 = bool(torch.equal(new4[1], ref4[1])) and bool(torch.equal(st4[1], ref4[1]))
                d8 = bool(torch.equal(new8[0], ref8[0])) and bool(torch.equal(st8[0], ref8[0]))
                s8 = bool(torch.equal(new8[1], ref8[1])) and bool(torch.equal(st8[1], ref8[1]))
                cell = d4 and s4 and d8 and s8
                allok = allok and cell
                tag = "OK" if cell else "FAIL"
                if not cell:
                    # diagnostics: count mismatches
                    nd4 = (new4[0] != ref4[0]).sum().item()
                    ns4 = (new4[1] != ref4[1]).sum().item()
                    nd8 = (new8[0] != ref8[0]).sum().item()
                    ns8 = (new8[1] != ref8[1]).sum().item()
                    tag += f" [data4_mm={nd4} sf4_mm={ns4} data8_mm={nd8} sf8_mm={ns8}]"
                print(f"  {name:10s} M={M:5d} Kl={Kl:5d} Kh={Kh:4d} | "
                      f"data4={d4} sf4={s4} data8={d8} sf8={s8}  {tag}", flush=True)
    print("QPF_BITEXACT_PASS" if allok else "QPF_BITEXACT_FAIL", flush=True)
    print("QPF_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
