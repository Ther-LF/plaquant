# RD Re-Latency — low-budget e2e latency of the measured-MSE allocation, with the K_high==0 kernel fix

**Branch:** `feat/rd-relat` (off `origin/feat/rd-alloc`, which includes the
kernel fix 7368a28/76da36a) · **HW:** RTX PRO 6000 Blackwell (SM120), docker
`luofan-plaquant`, dedicated worktree `/home/luofan/plaquant-relat`, idle GPU
(1B/3B: GPU 5; 8B: GPU 2) · **Date:** 2026-10-29 ·
**Harness:** `promix/scripts/rd_relat.py` · **Raw data:** `docs/rd_relat_data/`

## TL;DR

With K_high==0 Linears now served by the tuned **single-phase pure-NVFP4 kernel**
(instead of the old silent uniform-1/8 fallback), the low-budget **prefill curve
is genuinely downward-sloping**: cutting the measured-MSE FP8 budget now buys
real full-model prefill latency, monotonically on 3B and 8B. The old "latency
flat below 1/8" was the fallback timing a relabeled 1/8 model.

| | r=0 (pure NVFP4) | iso-accuracy budget | 1/8 (anchor) |
|---|---|---|---|
| **3B prefill ×** | **1.050** | **1.012** @ 1/16 | 0.996 |
| **8B prefill ×** | **1.064** | **1.003** @ 1/12 (1.010 @ 1/16) | 0.993 |
| **1B prefill ×** | **1.061** | — (1B iso only at 1/8) | 0.994 |

Decode is the honest negative: every K_high==0 Linear routes M=1 through the
prefill GEMM (no pure-NVFP4 decode-GEMV exists), so decode is ×0.72–0.84 at r=0
and ×0.88–0.93 at the other budgets. A K_high==0-capable decode GEMV is the
named follow-up.

## Protocol (contention-robust)

The shared box corrupts absolute latency, so `rd_relat.py` times the uniform-1/8
baseline and each budget's measured-MSE model **interleaved in one process**
(A,B,A,B,…, cudaEvent-timed, M=2048 prefill / 256+40 decode, 25 iters). The
primary statistic is the **median of paired per-iteration ratios**
(base_i/meas_i, timed microseconds apart under identical instantaneous load);
min-of-each is the cross-check. ×>1 = measured-MSE faster than uniform-1/8.
Latency-only builds (no GPTQ, contiguous-tail K_high pack — the ring *selection*
is folded offline into the PCA rotation in deployment, so only the K_high
*count* matters for latency; PPL is measured separately by `rd_sweep`).

Memory note: two resident 8B real-kernel models OOMed the 95 GiB card until
`PackedWeight.W_deq` — the (N,K) FP32 fake-twin reference the real kernel never
reads, ~26 GiB per 8B model copy — is freed after each build (`_free_wdeq`,
baseline residency 93 → 23.5 GiB).

## Results

`n_khigh0` counts reallocated Linears (o_proj pinned at 1/8, excluded) served by
the single-phase pure-NVFP4 kernel. `avg_fp8` is the **realized** FP8 ring
fraction read back from the wired kernel — it now tracks the budget (the old
bug reported 0.125 everywhere). prefill×/decode× are paired-median vs
uniform-1/8.

### Llama-3.2-3B (168 reallocated Linears, M=2048)

| budget | K_high==0 | avg_fp8 | bits/wt | prefill × | decode × |
|---|---|---|---|---|---|
| 0 | **168/168** | 0.0144 | 4.554 | **1.050** | 0.748 |
| 1/32 | 28/168 | 0.0421 | 4.658 | **1.035** | 0.905 |
| **1/16 (iso)** | 28/168 | 0.0697 | 4.761 | **1.012** | 0.905 |
| 1/12 | 28/168 | 0.0882 | 4.831 | 1.008 | 0.895 |
| 1/8 | 28/168 | 0.1250 | 4.969 | 0.996 | 0.902 |

### Llama-3-8B (192 reallocated Linears, M=2048)

| budget | K_high==0 | avg_fp8 | bits/wt | prefill × | decode × |
|---|---|---|---|---|---|
| 0 | **192/192** | 0.0132 | 4.549 | **1.064** | 0.719 |
| 1/32 | 32/192 | 0.0411 | 4.654 | **1.022** | 0.881 |
| 1/16 | 32/192 | 0.0691 | 4.759 | **1.010** | 0.885 |
| **1/12 (iso)** | 32/192 | 0.0877 | 4.829 | 1.003 | 0.889 |
| 1/8 | 32/192 | 0.1250 | 4.969 | 0.993 | 0.886 |

(8B r=0 cross-check: a first run under a 37 GiB co-tenant gave paired-med
×1.038 — same sign and ballpark as the clean-GPU ×1.064, i.e. the paired
interleaved protocol degrades gracefully but contention still compresses the
ratio; clean-GPU values are reported. Clean base prefill 83.05 ms vs measured
78.10 ms at M=2048.)

### Llama-3.2-1B (96 reallocated Linears, M=2048)

| budget | K_high==0 | avg_fp8 | bits/wt | prefill × | decode × |
|---|---|---|---|---|---|
| 0 | **96/96** | 0.0125 | 4.547 | **1.061** | 0.838 |
| 1/32 | 16/96 | 0.0406 | 4.652 | 1.025 | 0.925 |
| 1/16 | 16/96 | 0.0688 | 4.758 | 0.998 | 0.920 |
| 1/12 | 16/96 | 0.0875 | 4.828 | 1.008 | 0.911 |
| 1/8 | 16/96 | 0.1250 | 4.969 | 0.994 | 0.863 |

1B is noise-limited (±0.01 wiggle at 1/16 vs 1/12); the endpoints (×1.061 at
r=0, ×0.994 at 1/8) still bracket a downward slope, and the 1B r=0 value
reproduces the kfix report's full-model ×1.07.

### Old (buggy fallback) vs new — the curve un-flattened

Old `RD_SWEEP_3b_lat` (cross-process absolutes, fallback active): realized
avg_fp8 at labeled r=0 was **0.125** (a relabeled 1/8 model) and the "ratio" vs
uniform-1/8 wobbled flat around 1.0 (0.997 / 0.973 / 1.016 / 0.976 / 0.991 for
budgets 0…1/8); even measured@1/8 realized 0.164 because the fallback FORCED ≥1
FP8 ring on every Linear. New realized avg_fp8: 0.014 / 0.042 / 0.070 / 0.088 /
0.125 — tracks the budget, and the prefill ratio slopes 1.050 → 0.996 (3B),
1.064 → 0.993 (8B).
## Money point — iso-accuracy speedup

From the PPL sweep (`rd_pareto_report.md`): 3B holds iso-accuracy vs uniform-1/8
down to **1/16** (8.3774 ≤ 8.3924); 8B down to **1/12** (6.6272 ≤ 6.6367;
1/16 = 6.6663 just exceeds); 1B cannot drop below 1/8.

At those iso points the measured-MSE allocation now delivers, **for free in
accuracy**:

| model | iso budget | prefill × | bits/wt saved vs uniform-1/8 |
|---|---|---|---|
| 3B | 1/16 | **×1.012** (+1.2%) | 4.969 → 4.761 (−0.21) |
| 8B | 1/12 | **×1.003** (+0.3%; ×1.010 at 1/16 for +0.03 PPL) | 4.969 → 4.829 (−0.14) |

The prefill win is real but small, because at M=2048 the fused GEMMs are only
part of the forward (attention/norm/Hadamard/pack are format-independent): the
kfix GEMM-level r=0 saving is 18%, the full-model r=0 saving is 5–6%, and the
iso-budget point only converts ~0.4 of the r-range, giving ~1%. The larger
durable win of the low-budget allocation remains the **weight-memory/bpw cut**
plus the (now-unblocked) pure-NVFP4 serving path.

## Decode-path limitation (named follow-up)

`MixedQuantLinear.forward` routes M=1 through the single-phase prefill GEMM for
K_high==0 Linears (`mixed_linear.py:249-253`) because `decode_gemv_ext` has no
pure-NVFP4 entry (its weight-high buffers are empty). Consequence: decode is
×0.72–0.84 at r=0 (all Linears penalized) and ×0.88–0.93 at budgets ≥1/32
(the 16/28/32 K_high==0 Linears penalized) — the old sweep's decode≈1.0 was the
fallback masking this. Extending the decode GEMV to skip the high segment
(stream `W_lo_dec/S_lo_dec` only) should recover decode ≈1.0 at all budgets and
plausibly >1.0 at r=0 (less weight traffic); it is a kernel-wrapper change, not
a mainloop change.

## Verdict

**Re-confirmed and strengthened:** the K_high==0 fix turns the measured-MSE
low-budget regime from "accuracy-only, latency-flat" into a genuine (small)
latency win in prefill — the budget→prefill curve is now monotonically
downward-sloping on 3B (×0.996→×1.050) and 8B (×0.993→×1.064), with realized
avg_fp8 finally matching the budget. At iso-accuracy the free prefill gain is
~1.2% (3B @ 1/16) and ~0.3% (8B @ 1/12) on top of the 0.14–0.21 bpw saving;
the full 5–6% prefill gain sits at r=0, which costs accuracy (3B PPL 10.91 vs
8.39) and currently decode speed. Decode needs the pure-NVFP4 GEMV before any
low-budget configuration is deployable for generation-heavy serving.
