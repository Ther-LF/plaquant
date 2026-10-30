#!/usr/bin/env python3
"""vllm-perf r1 probe: batched W4A16 decode GEMV numerics + M-crossover bench.

Per real merged-Linear shape (3B/8B), on synthetic packed weights:
  1. BIT gate   decode_gemv_w4a16_mb row m == decode_gemv_w4a16 on row m
                (torch.equal — the batched kernel must be bit-identical per row).
  2. COS gate   GEMV output vs x_fp32 @ W_deq.T (fp32 reference) >= 0.999999.
  3. BENCH      decode GEMV (warps 8/16/32) vs the fused W4A4 GEMM path
                (quant_pack + fused_gemm_prepacked) at M in {1,2,4,8,16,24,32}
                -> picks the route threshold + warp default per shape.

Run (container base env or vLLM venv, idle GPU):
  CUDA_VISIBLE_DEVICES=7 python -m promix.vllm_integ.decode_probe [--out out.json]
"""
import argparse
import json
import time

import torch

from promix.vllm_integ.kernels import load_kernels, load_decode_gemv

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# merged vLLM shapes: (tag, N, K, K_high)
SHAPES = [
    ("3b_qkv",     5120,  3072,  384),
    ("3b_o",       3072,  3072,  384),
    ("3b_gate_up", 16384, 3072,  384),
    ("3b_down",    3072,  8192,  1024),
    ("8b_qkv",     6144,  4096,  512),
    ("8b_o",       4096,  4096,  512),
    ("8b_gate_up", 28672, 4096,  512),
    ("8b_down",    4096,  14336, 1792),
]


def fp4_decode_codes(codes):
    lut = FP4_LUT.to(codes.device)
    mag = lut[(codes & 7).long()]
    return torch.where((codes & 8).bool(), -mag, mag)


def make_packed(N, K, K_high, cd, dev="cuda"):
    """Synthetic packed weight with realistic scale magnitudes."""
    from promix.quantize.weight_pack import PackedWeight
    K_low = K - K_high
    g = 0.01
    codes = torch.randint(0, 16, (N, K_low), dtype=torch.uint8, device=dev)
    # E4M3 scale bytes in a sane positive range (exp 120..130 -> ~0.25..4)
    sf_lo = torch.randint(0x30, 0x48, (N, K_low // 16), dtype=torch.uint8, device=dev)
    bytes_hi = torch.randint(0, 255, (N, K_high), dtype=torch.uint8, device=dev)
    bytes_hi[bytes_hi == 0x7F] = 0  # avoid NaN E4M3
    bytes_hi[bytes_hi == 0xFF] = 0
    sf_hi = torch.randint(124, 131, (N, K_high // 32), dtype=torch.uint8, device=dev)

    cl = codes.view(N, K_low // 2, 2)
    W_lo_packed = (cl[:, :, 0] | (cl[:, :, 1] << 4)).to(torch.uint8).contiguous()
    S_lo_dec = (sf_lo.view(torch.float8_e4m3fn).float() * g).contiguous()
    S_hi_dec = (torch.exp2(sf_hi.float() - 127.0) * g).contiguous()
    physSFB_lo = cd.prepack_sf_fp4_b(sf_lo.contiguous(), N, K_low)
    physSFB_hi = cd.prepack_sf_fp8_b(sf_hi.contiguous(), N, K_high)

    # fp32 dequant reference
    deq_lo = fp4_decode_codes(codes) * S_lo_dec.repeat_interleave(16, dim=1)
    deq_hi = bytes_hi.view(torch.float8_e4m3fn).float() * S_hi_dec.repeat_interleave(32, dim=1)
    W_deq = torch.cat([deq_lo, deq_hi], dim=1).contiguous()

    pk = PackedWeight(N=N, K=K, K_low=K_low, K_high=K_high,
                      W_lo_packed=W_lo_packed, physSFB_lo=physSFB_lo,
                      W_hi_bytes=bytes_hi.contiguous(), physSFB_hi=physSFB_hi,
                      W_deq=W_deq, bias=None, g_w=g,
                      W_lo_dec=W_lo_packed, S_lo_dec=S_lo_dec,
                      W_hi_dec=bytes_hi.contiguous(), S_hi_dec=S_hi_dec)
    # raw 1-byte scale streams (r2 sf8 probe)
    pk.sf_lo_raw = sf_lo.contiguous()
    pk.sf_hi_raw = sf_hi.contiguous()
    return pk


def bench(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def bench_graph(fn, reps=30, inner=10):
    """Per-iteration cost under CUDA-graph REPLAY (the serving regime): capture
    `inner` iterations into one graph (amortizes the per-replay launch floor the
    way the engine's big whole-model graph does), replay, report us/iter.
    Eager timing overstates multi-launch paths by 50-80us of host/launch cost
    that graph replay erases — it picks the WRONG route threshold (r1 lesson)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / inner * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    torch.manual_seed(0)
    cd, qpf = load_kernels()
    dg = load_decode_gemv()
    assert dg is not None and hasattr(dg, "decode_gemv_w4a16_mb")

    from promix.quantize.mixed_linear import MixedQuantLinear
    results = {}
    for tag, N, K, K_high in SHAPES:
        pk = make_packed(N, K, K_high, cd)
        mql = MixedQuantLinear(pk, nv=None, mx=None, cd=cd, qpf=qpf, dg=dg).cuda()
        res = {"N": N, "K": K, "K_high": K_high}

        # ---- numerics ----
        x = (torch.randn(32, K, device="cuda") * 0.5).half()
        bit_ok, min_cos = True, 1.0
        for M in (1, 2, 3, 8, 16):
            xm = x[:M].contiguous()
            Dmb = dg.decode_gemv_w4a16_mb(xm, pk.W_lo_dec, pk.S_lo_dec,
                                          pk.W_hi_dec, pk.S_hi_dec,
                                          N, pk.K_low, pk.K_high, 16)
            for m in range(M):
                D1 = dg.decode_gemv_w4a16(xm[m], pk.W_lo_dec, pk.S_lo_dec,
                                          pk.W_hi_dec, pk.S_hi_dec,
                                          N, pk.K_low, pk.K_high, 8)
                if not torch.equal(Dmb[m], D1):
                    bit_ok = False
            ref = xm.float() @ pk.W_deq.t()
            c = torch.nn.functional.cosine_similarity(
                Dmb.flatten(), ref.flatten(), dim=0).item()
            min_cos = min(min_cos, c)
        res["bit_mb_vs_m1"] = bit_ok
        res["min_cos_vs_ref"] = min_cos
        print(f"{tag:12s} bit={bit_ok} min_cos={min_cos:.8f}", flush=True)

        # ---- bench (graph-replay timing): GEMV (per warps) vs fused GEMM ----
        timing = {}
        for M in (1, 2, 4, 8, 16):
            xm = x[:M].contiguous()
            row = {}
            if M == 1:
                for w in (8, 24):
                    row[f"gemv_w{w}"] = bench_graph(lambda w=w: dg.decode_gemv_w4a16(
                        xm[0], pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec, pk.S_hi_dec,
                        N, pk.K_low, pk.K_high, w))
            else:
                for w in (8, 16, 32):
                    row[f"gemv_w{w}"] = bench_graph(lambda w=w: dg.decode_gemv_w4a16_mb(
                        xm, pk.W_lo_dec, pk.S_lo_dec, pk.W_hi_dec, pk.S_hi_dec,
                        N, pk.K_low, pk.K_high, w))

            def fused():
                a = mql._pack_activation(xm)
                return mql._gemm(*a, M)
            row["fused"] = bench_graph(fused)
            timing[M] = row
            best = min((v, k) for k, v in row.items() if k != "fused")
            print(f"  M={M:2d} fused={row['fused']:7.1f}us best_gemv={best[0]:7.1f}us "
                  f"({best[1]}) ratio={row['fused'] / best[0]:.2f}x", flush=True)
        res["timing_us"] = timing
        results[tag] = res

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
