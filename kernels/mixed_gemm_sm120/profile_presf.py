"""Standalone ncu harness: profile the post-T1 best path (mixed_gemm_fused_b_presf).

Usage: python profile_presf.py <decode|prefill>
  decode  -> q_proj_1B  M=16   (N=2048, Klow=1792, Khigh=256)
  prefill -> gate_proj_1B M=2048 (N=8192, Klow=1792, Khigh=256)

Prepacks the weight SF ONCE (outside the measured region) then issues exactly ONE
presf call so ncu captures each per-call kernel (fill_sf_merged2, NVFP4 GEMM,
MXFP8 GEMM, scale_only). Run under ncu in the luofan-ncu container.
"""
import sys
import torch
import mixed_gemm as mg

SCEN = {
    "decode":  ("q_proj_1B",    16,   2048, 1792, 256),
    "prefill": ("gate_proj_1B", 2048, 8192, 1792, 256),
}

def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else "decode"
    name, M, N, K_low, K_high = SCEN[scen]
    mg.get_fused_ext()
    p, _ = mg.prep_mixed(M, N, K_low, K_high, seed=1)
    mg.prepack_weight_sf(p, N, K_low, K_high)   # once, before measured region
    torch.cuda.synchronize()
    # one measured call -> ncu captures ONLY this (run with --profile-from-start off)
    torch.cuda.profiler.start()
    out = mg.mixed_gemm_fused_b_presf(p, M, N, K_low, K_high)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"{scen}: {name} M={M} N={N} -> out {tuple(out.shape)} {out.dtype}")

if __name__ == "__main__":
    main()
