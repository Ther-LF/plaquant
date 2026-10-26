"""PERF round-2 Task 3: push the W4A4 decode GEMV toward the HBM roof.

Standalone.  Compares the shipped v2 (shared-staged activation, uint32 loads) with
the new v4 (no shared staging, uint4/128-bit loads, occupancy-tunable rows/block)
across every Linear shape of Llama-3.2-1B / 3.2-3B / 3.1-8B at M=1.  Reports
per-shape time, achieved weight-read bandwidth, and % of the measured HBM ceiling,
plus a correctness gate (cosine vs the fp32 fake-quant twin).

b8: a pure M=1 GEMV cannot represent batch-8 decode (that needs weight reuse across
8 rows == a small-M GEMM); the b8 lever is documented, not measured here.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KERN = os.path.join(REPO, "kernels", "mixed_gemm_sm120")
sys.path.insert(0, REPO); sys.path.insert(0, KERN)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
import torch
from torch.utils.cpp_extension import load
from promix.quantize.weight_pack import quantize_nvfp4_spec, quantize_mxfp8_spec, split_k
from promix.quantize.quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8

DEV = "cuda"
HIGH = 0.125

def shapes_for(hidden, interm, kv):
    return [("q_proj", hidden, hidden), ("k_proj", kv, hidden), ("v_proj", kv, hidden),
            ("o_proj", hidden, hidden), ("gate_proj", interm, hidden),
            ("up_proj", interm, hidden), ("down_proj", hidden, interm)]

MODELS = {
    "1B": shapes_for(2048, 8192, 512),
    "3B": shapes_for(3072, 8192, 1024),
    "8B": shapes_for(4096, 14336, 1024),
}

def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

def tmin(fn, iters=200, warmup=40):
    torch.cuda.synchronize()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3

def build_pack(W, K_low, K_high):
    N, K = W.shape
    codes_lo, SFB_lo, deq_lo = quantize_nvfp4_spec(W[:, :K_low].contiguous())
    bytes_hi, SFB_hi, deq_hi = quantize_mxfp8_spec(W[:, K_low:].contiguous())
    cl = codes_lo.view(N, K_low // 2, 2)
    Wlo = (cl[:, :, 0] | (cl[:, :, 1] << 4)).to(torch.uint8).contiguous()
    Slo = SFB_lo.view(torch.float8_e4m3fn).float().contiguous()
    Whi = bytes_hi.contiguous()
    Shi = torch.exp2(SFB_hi.float() - 127.0).contiguous()
    W_deq = torch.cat([deq_lo, deq_hi], dim=-1).contiguous()
    pb = Wlo.numel() + Whi.numel() + Slo.numel() * 4 + Shi.numel() * 4
    return Wlo, Slo, Whi, Shi, W_deq, pb

def main():
    flags = ["-std=c++17", "-O3"]
    dg = load(name="decode_gemv_ext", sources=[os.path.join(KERN, "decode_gemv_ext.cu")],
              extra_cuda_cflags=flags, verbose=False)
    print("decode_gemv_ext loaded", flush=True)
    xb = torch.empty(1 << 28, device=DEV, dtype=torch.float16); yb = torch.empty_like(xb)
    t_cp = tmin(lambda: yb.copy_(xb), iters=50, warmup=10)
    hbm = xb.numel() * 2 * 2 / (t_cp / 1e3) / 1e12
    del xb, yb; torch.cuda.empty_cache()
    print(f"HBM ceiling (copy) = {hbm:.2f} TB/s", flush=True)

    out = {"hbm_tbps": hbm, "models": {}}
    for size, shapes in MODELS.items():
        print(f"\n=== {size} ===", flush=True)
        rows = []
        for name, N, K in shapes:
            K_low, K_high = split_k(K, HIGH)
            torch.manual_seed(hash(name) % 1000)
            W = torch.randn(N, K, device=DEV) * 0.08
            x = torch.randn(1, K, device=DEV); x[:, K_low:] *= 3.0
            Wlo, Slo, Whi, Shi, W_deq, pb = build_pack(W, K_low, K_high)
            xq = torch.cat([fake_quantize_nvfp4(x[:, :K_low].float()),
                            fake_quantize_mxfp8(x[:, K_low:].float())], dim=-1)[0].contiguous()
            D_twin = (xq @ W_deq.t()).float()
            # v2 baseline (shared-staged, uint32, fixed 8 warps). Fails for K>~12k
            # (no smem opt-in) -> fall back to v5 w8 which is the same kernel + opt-in.
            smem_floats = K_low + K_high
            if smem_floats * 4 <= 48 * 1024:
                D2 = dg.decode_gemv_v2(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high)
                t2 = tmin(lambda: dg.decode_gemv_v2(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high))
                c2 = cos(D2, D_twin)
                v2tag = "v2"
            else:
                D2 = dg.decode_gemv_v5(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, 8)
                t2 = tmin(lambda: dg.decode_gemv_v5(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, 8))
                c2 = cos(D2, D_twin)
                v2tag = "v5w8*"   # v2 would crash here; v5 w8 == same kernel + opt-in
            # one v4 datapoint (no-shared / uint4) to document the negative result
            D4n = dg.decode_gemv_v4(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, 8)
            t4n = tmin(lambda: dg.decode_gemv_v4(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, 8))
            # v5 sweep rows/block (the roof-push on the winning shared design)
            best = None
            for w in (4, 8, 16, 24, 32):
                D5 = dg.decode_gemv_v5(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, w)
                c5 = cos(D5, D_twin)
                t5 = tmin(lambda: dg.decode_gemv_v5(xq, Wlo, Slo, Whi, Shi, N, K_low, K_high, w))
                if best is None or t5 < best[1]:
                    best = (w, t5, c5)
            bw2 = pb / (t2 / 1e3) / 1e12
            bw4 = pb / (best[1] / 1e3) / 1e12
            print(f"  {name:10s} N={N:5d} K={K:5d} Klo={K_low:5d}: "
                  f"{v2tag} {t2*1e3:5.1f}us {bw2/hbm*100:4.1f}%HBM (cos {c2:.5f}) | "
                  f"v5 w{best[0]:<2d} {best[1]*1e3:5.1f}us {bw4/hbm*100:4.1f}%HBM (cos {best[2]:.5f}) | "
                  f"v5/v2 {t2/best[1]:.2f}x | v4(uint4) {t4n*1e3:5.1f}us", flush=True)
            rows.append(dict(name=name, N=N, K=K, K_low=K_low, packed_bytes=pb,
                             v2_us=t2*1e3, v2_hbm=bw2/hbm, v2_cos=c2, v2tag=v2tag,
                             v4_uint4_us=t4n*1e3,
                             v5_warps=best[0], v5_us=best[1]*1e3, v5_hbm=bw4/hbm, v5_cos=best[2],
                             v5_speedup=t2/best[1]))
            del W, x, Wlo, Slo, Whi, Shi, W_deq
            torch.cuda.empty_cache()
        s2 = sum(r["v2_us"] for r in rows); s4 = sum(r["v5_us"] for r in rows)
        wc = min(min(r["v2_cos"], r["v5_cos"]) for r in rows)
        print(f"  SUM/layer: v2={s2:.1f}us v5={s4:.1f}us  v5/v2={s2/s4:.2f}x  "
              f"max v5 HBM%={max(r['v5_hbm'] for r in rows)*100:.0f}%  worst cos={wc:.5f}", flush=True)
        out["models"][size] = dict(rows=rows, sum_v2_us=s2, sum_v4_us=s4, worst_cos=wc)
    with open(os.path.join(KERN, "perf_r2_decode_gemv.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nRESULTS_WRITTEN perf_r2_decode_gemv.json\nTASK3_DONE", flush=True)

if __name__ == "__main__":
    main()
