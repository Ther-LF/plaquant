"""rd-oppareto (E4): torch-NATIVE operator points on the unmerged per-projection
shapes — the framework's own tensor-core paths, no custom kernels:

  torchfp4  — torch._scaled_mm_v2 NVFP4 (float4_e2m1fn_x2 data, BlockWise1x16
              e4m3 scales, SWIZZLE_32_4_4) over full K  (cuBLASLt)
  torchfp8  — torch._scaled_mm_v2 MXFP8 (float8_e4m3fn data, BlockWise1x32
              e8m0 scales, SWIZZLE_32_4_4) over full K  (cuBLASLt)
  bf16      — torch.mm bf16 cuBLAS anchor

Quantizers are the validated reference recipes from gemm_h2h_torchfp4.py
(exactness probe + cosine gate). GEMM-only, cudaEvent median-of-50 (15 warmup),
idle GPU, container `luofan-plaquant`.

Usage: python3 promix/eval/rd_oppareto_torch.py --m 8192 --out torch_points.json
"""
import json, argparse
import torch
from torch._C import _ScalingType as ST, _SwizzleType as SW

DEV = "cuda"

E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)

SHAPES = [
    ("3B", "qkvo",    3072, 3072),
    ("3B", "gate_up", 8192, 3072),
    ("3B", "down",    3072, 8192),
    ("8B", "qkvo",    4096, 4096),
    ("8B", "gate_up", 14336, 4096),
    ("8B", "down",    4096, 14336),
]


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(m):
    """cuBLASLt SWIZZLE_32_4_4 block-scale layout (pytorch test-suite recipe)."""
    rows, cols = m.shape
    nrb, ncb = ceil_div(rows, 128), ceil_div(cols, 4)
    pr, pc = nrb * 128, ncb * 4
    if (rows, cols) != (pr, pc):
        p = torch.zeros(pr, pc, dtype=m.dtype, device=m.device)
        p[:rows, :cols] = m
        m = p
    blocks = m.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quant_nvfp4_ref(x):
    """Per-1x16-block e4m3 scale + e2m1 codes -> (packed u8, scales fp32, dequant)."""
    R, C = x.shape
    xb = x.view(R, C // 16, 16)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    v = xb / scale
    sign = v.sign()
    idx = torch.bucketize(v.abs().clamp(max=6.0), E2M1_MIDS)
    q = E2M1_VALS[idx] * sign
    codes = idx.to(torch.uint8) + (sign < 0).to(torch.uint8) * 8
    codes = codes.view(R, C)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    deq = (q * scale).view(R, C)
    return packed, scale.view(R, C // 16), deq


def quant_mxfp8_ref(x):
    """Per-1x32-block ue8m0 scale + e4m3 codes -> (e4m3 tensor, e8m0 scales, dequant)."""
    R, C = x.shape
    xb = x.view(R, C // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(min=-127.0, max=127.0)
    scale = torch.exp2(e)
    codes = (xb / scale).to(torch.float8_e4m3fn)
    deq = (codes.float() * scale).view(R, C)
    sc8 = scale.view(R, C // 32).to(torch.float8_e8m0fnu)
    return codes.view(R, C), sc8, deq


def native_fp4_mm(Afp4, Bfp4_t, sA, sB):
    return torch._scaled_mm_v2(
        Afp4, Bfp4_t,
        [sA], [ST.BlockWise1x16], [SW.SWIZZLE_32_4_4],
        [sB], [ST.BlockWise1x16], [SW.SWIZZLE_32_4_4],
        None, torch.bfloat16)


def native_mxfp8_mm(A8, B8_t, sA, sB):
    return torch._scaled_mm_v2(
        A8, B8_t,
        [sA], [ST.BlockWise1x32], [SW.SWIZZLE_32_4_4],
        [sB], [ST.BlockWise1x32], [SW.SWIZZLE_32_4_4],
        None, torch.bfloat16)


def median_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort(); n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M = args.m

    rows = []
    for mdl, name, N, K in SHAPES:
        torch.manual_seed(0)
        W = (torch.randn(N, K, device=DEV) * 0.08)
        x = torch.randn(M, K, device=DEV)
        ref = x @ W.t()
        flops = 2.0 * M * N * K

        # torch-native NVFP4
        pa, sa, _ = quant_nvfp4_ref(x)
        pb, sb, _ = quant_nvfp4_ref(W)
        A4 = pa.view(torch.float4_e2m1fn_x2)
        B4 = pb.view(torch.float4_e2m1fn_x2)
        sA4 = to_blocked(sa).to(torch.float8_e4m3fn)
        sB4 = to_blocked(sb).to(torch.float8_e4m3fn)

        def fp4():
            return native_fp4_mm(A4, B4.t(), sA4, sB4)
        cos4 = cosine(fp4(), ref)
        t4, m4 = median_ms(fp4)
        rows.append(dict(method="torch_fp4", model=mdl, shape=name, M=M, N=N,
                         K=K, gemm_ms=t4, gemm_min_ms=m4,
                         tflops=flops / (t4 * 1e-3) / 1e12, eff_bits=4.5,
                         cosine_vs_fp32=cos4))
        # torch-native MXFP8
        a8, sa8, _ = quant_mxfp8_ref(x)
        b8, sb8, _ = quant_mxfp8_ref(W)
        sA8 = to_blocked(sa8.float()).to(torch.float8_e8m0fnu)
        sB8 = to_blocked(sb8.float()).to(torch.float8_e8m0fnu)

        def fp8():
            return native_mxfp8_mm(a8, b8.t(), sA8, sB8)
        cos8 = cosine(fp8(), ref)
        t8, m8 = median_ms(fp8)
        rows.append(dict(method="torch_fp8", model=mdl, shape=name, M=M, N=N,
                         K=K, gemm_ms=t8, gemm_min_ms=m8,
                         tflops=flops / (t8 * 1e-3) / 1e12, eff_bits=8.25,
                         cosine_vs_fp32=cos8))
        # bf16 cuBLAS
        Xb = x.to(torch.bfloat16).contiguous()
        Wb = W.to(torch.bfloat16).contiguous()
        tb, mb = median_ms(lambda: torch.mm(Xb, Wb.t()))
        rows.append(dict(method="bf16_cublas", model=mdl, shape=name, M=M, N=N,
                         K=K, gemm_ms=tb, gemm_min_ms=mb,
                         tflops=flops / (tb * 1e-3) / 1e12, eff_bits=16.0,
                         cosine_vs_fp32=cosine(torch.mm(Xb, Wb.t()), ref)))
        print(f"{mdl} {name:8s} M={M}: torchfp4 {t4:.4f}ms (cos {cos4:.4f}) | "
              f"torchfp8 {t8:.4f}ms (cos {cos8:.4f}) | bf16 {tb:.4f}ms", flush=True)
        del W, x, ref, pa, sa, pb, sb, A4, B4, sA4, sB4, a8, sa8, b8, sb8, Xb, Wb
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nTORCH_POINTS_DONE", flush=True)


if __name__ == "__main__":
    main()
