"""rd-torchfp4: isolate-our-contribution GEMM head-to-head (QServe methodology).

Compares, at the EXACT vLLM GEMM decomposition shapes of Llama-3.2-3B / Llama-3-8B
across M, GEMM-only, warm, median, idle GPU:

  1. ours      — our fused dual-format NVFP4+MXFP8 W4A4 GEMM (prepacked, r=1/8)
  2. torchfp4  — torch-NATIVE NVFP4 tensor-core GEMM: torch._scaled_mm_v2 with
                 float4_e2m1fn_x2 data + BlockWise1x16 e4m3 scales (SWIZZLE_32_4_4),
                 i.e. the framework's own FP4 path (cuBLASLt), no custom kernel.
  3. dequant   — "no kernel work" floor: NVFP4-packed weight dequantized to bf16
                 in plain torch per call, then torch.mm in bf16 (naive W4 path).
  4. bf16      — torch.mm bf16 cuBLAS (same-stack FP16-class anchor).

All paths time the GEMM proper (activation/weight quantization is outside the
timed region for 1 and 2, matching how the fused kernel is benchmarked; path 3
times dequant+mm because the dequant IS the naive path's per-call cost).

Also validates torch-native FP4 numerically (exactness probe + real-data cosine
vs a dequantized reference) so the baseline is known-correct, not just running.

Writes JSON for docs/rd_torchfp4_report.md.
"""
import os, sys, json, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

from torch._C import _ScalingType as ST, _SwizzleType as SW  # noqa: E402
from promix.eval.full_network_realkernel import load_kernels, HIGH_FRACTION  # noqa: E402
from promix.quantize.weight_pack import split_k  # noqa: E402
from promix.quantize.mixed_linear import MixedQuantLinear  # noqa: E402
import nvfp4_gemm as nvmod  # noqa: E402

DEV = "cuda"

E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)
E2M1_LUT16 = torch.cat([E2M1_VALS, -E2M1_VALS])  # code 0..15 -> value


# (name, N, K) — vLLM GEMM decomposition (qkv/gate_up merged), same as rd-largem.
def shapes_for(H, I, nheads, nkv):
    hd = H // nheads
    q = nheads * hd; kv = nkv * hd
    return [
        ("qkv",     q + 2 * kv, H),
        ("o_proj",  H,          q),
        ("gate_up", 2 * I,      H),
        ("down",    H,          I),
    ]


MODELS = {
    "3B": dict(H=3072, I=8192, nheads=24, nkv=8),
    "8B": dict(H=4096, I=14336, nheads=32, nkv=8),
}


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
    """Reference NVFP4 quantizer: per-1x16-block e4m3 scale + e2m1 codes.
    Returns packed bytes (R, C/2) uint8, scales fp32 (R, C/16) (post-e4m3-rounding),
    and the dequantized fp32 tensor."""
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


def native_fp4_mm(Afp4, Bfp4_t, sA, sB):
    return torch._scaled_mm_v2(
        Afp4, Bfp4_t,
        [sA], [ST.BlockWise1x16], [SW.SWIZZLE_32_4_4],
        [sB], [ST.BlockWise1x16], [SW.SWIZZLE_32_4_4],
        None, torch.bfloat16)


def native_fp4_validate():
    """(a) exactness: all codes=1.0, unit scales -> out == K.
       (b) real data: quantized A,B with real block scales vs dequant reference."""
    M, N, K = 256, 256, 512
    A = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device=DEV).view(torch.float4_e2m1fn_x2)
    B = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device=DEV).view(torch.float4_e2m1fn_x2)
    sA = to_blocked(torch.ones(M, K // 16, device=DEV)).to(torch.float8_e4m3fn)
    sB = to_blocked(torch.ones(N, K // 16, device=DEV)).to(torch.float8_e4m3fn)
    out = native_fp4_mm(A, B.t(), sA, sB)
    exact = bool((out == float(K)).all().item())

    torch.manual_seed(7)
    M, N, K = 2048, 1024, 2048
    x = torch.randn(M, K, device=DEV)
    w = torch.randn(N, K, device=DEV) * 0.08
    pa, sa, da = quant_nvfp4_ref(x)
    pb, sb, db = quant_nvfp4_ref(w)
    out = native_fp4_mm(
        pa.view(torch.float4_e2m1fn_x2), pb.view(torch.float4_e2m1fn_x2).t(),
        to_blocked(sa).to(torch.float8_e4m3fn), to_blocked(sb).to(torch.float8_e4m3fn))
    ref = da @ db.t()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    return exact, cos


def median_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        s.append(st.elapsed_time(en))
    s.sort()
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), s[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3B,8B")
    ap.add_argument("--ms", default="2048,8192,16384,32768")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]

    nv, mx, cd, qpf = load_kernels()
    nvext = nvmod.get_ext()
    print("kernels loaded", flush=True)

    exact, cos = native_fp4_validate()
    print(f"NATIVE_FP4_VALIDATION exact={exact} cosine_vs_dequant_ref={cos:.6f}", flush=True)
    lut = E2M1_LUT16.to(DEV)

    rows = []
    for mdl in args.models.split(","):
        c = MODELS[mdl]
        for name, N, K in shapes_for(c["H"], c["I"], c["nheads"], c["nkv"]):
            K_low, K_high = split_k(K, HIGH_FRACTION)
            torch.manual_seed(0)
            W = (torch.randn(N, K, device=DEV) * 0.08)
            lin = type("L", (), {"weight": type("T", (), {"data": W})(), "bias": None})()
            mql = MixedQuantLinear.from_linear(lin, nv, mx, cd, high_fraction=HIGH_FRACTION, qpf=qpf)

            # torch-native FP4 weight: packed codes + swizzled e4m3 block scales
            Wp, Wsc, _ = quant_nvfp4_ref(W)
            Bfp4 = Wp.view(torch.float4_e2m1fn_x2)
            sB = to_blocked(Wsc).to(torch.float8_e4m3fn)

            # dequant-path weight storage: packed uint8 codes + fp32 block scales
            Wp_u8 = Wp  # (N, K/2) uint8
            Wsc_f = Wsc  # (N, K/16) fp32

            # bf16 weight
            Wbf = W.to(torch.bfloat16).contiguous()

            # our pure-NVFP4 full-K weight (r=0 structural reference)
            Bc, SFB_nv, _ = nvmod.quantize_nvfp4(W)
            Bp_nv = nvmod.pack_fp4(Bc)

            for M in ms:
                x = torch.randn(M, K, device=DEV)
                x[:, K_low:] *= 3.0

                # 1) ours fused (prepacked, GEMM only)
                A_lo, pSFA_lo, A_hi, pSFA_hi = mql.pack_only(x)

                def fused():
                    return cd.fused_gemm_prepacked(
                        A_lo, pSFA_lo, mql.W_lo_packed, mql.physSFB_lo,
                        A_hi, pSFA_hi, mql.W_hi_bytes, mql.physSFB_hi,
                        M, N, K_low, K_high)
                t_fused, min_fused = median_ms(fused)

                # 2) torch-native FP4 (prequantized A, GEMM only)
                Ap, Asc, _ = quant_nvfp4_ref(x)
                Afp4 = Ap.view(torch.float4_e2m1fn_x2)
                sA = to_blocked(Asc).to(torch.float8_e4m3fn)
                Bt = Bfp4.t()

                def torchfp4():
                    return native_fp4_mm(Afp4, Bt, sA, sB)
                t_tfp4, min_tfp4 = median_ms(torchfp4)

                # 2b) our pure-NVFP4 full-K GEMM (r=0, apples-to-apples vs torch native)
                Xq, SFA_nv, _ = nvmod.quantize_nvfp4(x)
                Ap_nv = nvmod.pack_fp4(Xq)

                def purenv():
                    return nvext.nvfp4_gemm(Ap_nv, SFA_nv, Bp_nv, SFB_nv, M, N, K)
                t_nv, min_nv = median_ms(purenv)

                # 3) naive dequant-FP4 -> bf16 mm (dequant inside timed region)
                xbf = x.to(torch.bfloat16).contiguous()

                def dequant_mm():
                    lo = (Wp_u8 & 0xF).long()
                    hi = (Wp_u8 >> 4).long()
                    vals = torch.stack((lut[lo], lut[hi]), dim=-1).view(N, K)
                    w = (vals.view(N, K // 16, 16) * Wsc_f.unsqueeze(-1)).view(N, K)
                    return xbf @ w.to(torch.bfloat16).t()
                t_dq, min_dq = median_ms(dequant_mm, iters=30, warmup=8)

                # 4) bf16 cuBLAS
                def bf16mm():
                    return xbf @ Wbf.t()
                t_bf, min_bf = median_ms(bf16mm)

                flops = 2.0 * M * N * K
                tf = lambda t: flops / (t * 1e-3) / 1e12
                rec = dict(model=mdl, shape=name, N=N, K=K, K_low=K_low, K_high=K_high, M=M,
                           fused_ms=t_fused, fused_min_ms=min_fused,
                           torchfp4_ms=t_tfp4, torchfp4_min_ms=min_tfp4,
                           pure_nvfp4_ms=t_nv, pure_nvfp4_min_ms=min_nv,
                           dequant_ms=t_dq, dequant_min_ms=min_dq,
                           bf16_ms=t_bf, bf16_min_ms=min_bf,
                           fused_tflops=tf(t_fused), torchfp4_tflops=tf(t_tfp4),
                           pure_nvfp4_tflops=tf(t_nv),
                           dequant_tflops=tf(t_dq), bf16_tflops=tf(t_bf),
                           fused_vs_bf16=t_bf / t_fused,
                           fused_vs_torchfp4=t_tfp4 / t_fused,
                           fused_vs_dequant=t_dq / t_fused,
                           torchfp4_vs_bf16=t_bf / t_tfp4,
                           purenv_vs_torchfp4=t_tfp4 / t_nv)
                rows.append(rec)
                print(f"{mdl} {name:8s} M={M:5d} N={N:6d} K={K:6d}: "
                      f"ours {t_fused:.4f}ms ({rec['fused_tflops']:.0f}TF) | "
                      f"torchFP4 {t_tfp4:.4f}ms ({rec['torchfp4_tflops']:.0f}TF) | "
                      f"pureNV {t_nv:.4f}ms ({rec['pure_nvfp4_tflops']:.0f}TF) | "
                      f"dequant {t_dq:.4f}ms | bf16 {t_bf:.4f}ms | "
                      f"ours/bf16 {rec['fused_vs_bf16']:.2f}x "
                      f"ours/torchFP4 {rec['fused_vs_torchfp4']:.2f}x "
                      f"ours/dequant {rec['fused_vs_dequant']:.2f}x", flush=True)
                del x, xbf, Ap, Asc, Afp4, sA, Xq, SFA_nv, Ap_nv
                torch.cuda.empty_cache()
            del mql, W, Wp, Wsc, Bfp4, sB, Wbf, Bc, SFB_nv, Bp_nv
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(dict(native_fp4_exact=exact, native_fp4_cosine=cos, rows=rows), f, indent=2)
    print(f"RESULTS_WRITTEN {args.out}\nTORCHFP4_DONE", flush=True)


if __name__ == "__main__":
    main()
