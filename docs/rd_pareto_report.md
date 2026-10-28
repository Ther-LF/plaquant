# Rate-Distortion Mixed-Precision Allocation — Full Pareto Sweep (R1-off)

**Branch:** `feat/rd-sweep` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant` · **Date:** 2026-10-28

**Question.** Characterize the θ/budget knob of the rate-distortion mixed-precision
allocation **completely**, in both directions: *below* 1/8 (can we serve more of
the network in NVFP4 at iso-accuracy and go faster?) and *above* 1/8 (can we spend
latency to buy accuracy?). Three model sizes form a scaling ladder:
Llama-3.2-1B-Instruct (d=2048), Llama-3.2-3B (d=3072), Llama-3-8B (d=4096).

**Harness.** `promix/scripts/rd_sweep.py` (model-agnostic driver) reusing the
validated `rd_validity` PPL path (measured-MSE ΔD allocation → ring reorder →
GPTQ → wikitext-2) and the `rd_perf` real packed NVFP4+MXFP8 fused-kernel e2e
latency path (prefill TTFT M=2048, decode TPOT M=1, cudaEvent median). Regime:
**R1-off** (pure PCA, identity main rotation) — the cheapest, rotation-consistent
regime, same as the validity gate. CSV: `docs/rd_pareto_sweep.csv`.

**Allocations.** *measured* = global single-threshold ΔD = D_fp4 − D_fp8 ranking
(`adaptive.allocate_by_budget`) over the 6 reallocated Linears q/k/v/gate/up/down_proj
at each budget; o_proj held at uniform 1/8 (controlled constant). *uniform* =
every reallocated Linear at the same ring fraction (1/8 anchor, 1/16 reference).
Budget grid (avg FP8 ring-fraction over the reallocated rings):
{0, 1/32, 1/24, 1/16, 1/12, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1}. r=0 = pure NVFP4,
r=1 = (all-but-one-ring) pure MXFP8. `bits_per_weight` is exact from the realized
K_high map (NVFP4=4.5 b, MXFP8=8.25 b/weight).

## Validation gate (reproduced before trusting any delta)

| model | bf16 bridge (target) | uniform-1/8 | measured-1/8 | measured-1/16 |
|---|---|---|---|---|
| 1B | 13.157 (13.16) ✓ | 14.6423 | 14.4202 | 14.8728 |
| 3B | 7.8148 (7.82) ✓ | **8.3924** | **8.3124** | **8.3774** |
| 8B | 6.1371 (6.14) ✓ | 6.6367 | *(pending)* | *(pending)* |

The 3B uniform-1/8 = **8.3924**, measured-1/8 = **8.3124**, measured-1/16 =
**8.3774** reproduce the prior `rd_validity`/`rd_perf` numbers exactly, confirming
the harness is on the proven path. All three bf16 bridges match their committed
baselines, so every delta below is trustworthy.

---

## Llama-3.2-1B-Instruct (d=2048) — COMPLETE

| alloc | budget | avg FP8 | bits/w | PPL | prefill ms | decode ms | prefill × | decode × |
|---|---|---|---|---|---|---|---|---|
| measured | 0      | 0.0125 | 4.547 | 17.8383 | 28.24 | 14.19 | 0.995 | 0.988 |
| measured | 1/32   | 0.0406 | 4.652 | 15.4401 | 27.58 | 14.48 | 1.019 | 0.968 |
| measured | 1/24   | 0.0500 | 4.688 | 15.2980 | 28.23 | 13.98 | 0.995 | 1.003 |
| measured | 1/16   | 0.0688 | 4.758 | 14.8728 | 27.80 | 13.98 | 1.011 | 1.002 |
| measured | 1/12   | 0.0875 | 4.828 | 14.7048 | 27.97 | 14.01 | 1.004 | 1.000 |
| **measured** | **1/8** | 0.1250 | 4.969 | **14.4202** | 28.27 | 13.82 | 0.994 | 1.014 |
| measured | 3/16   | 0.1812 | 5.180 | 14.2500 | 28.81 | 14.03 | 0.975 | 0.999 |
| measured | 1/4    | 0.2375 | 5.391 | 14.1605 | 29.22 | 13.93 | 0.961 | 1.006 |
| measured | 3/8    | 0.3494 | 5.810 | 14.0894 | 28.67 | 14.00 | 0.980 | 1.001 |
| measured | 1/2    | 0.4562 | 6.211 | 13.9806 | 28.63 | 13.98 | 0.981 | 1.003 |
| measured | 3/4    | 0.6705 | 7.014 | 13.7359 | 29.69 | 13.94 | 0.946 | 1.005 |
| measured | 1 (≈MXFP8) | 0.8938 | 7.852 | 13.6272 | 30.96 | 14.00 | 0.907 | 1.001 |
| uniform  | 1/16   | 0.0688 | 4.758 | 15.3938 | 27.43 | 13.93 | 1.024 | 1.006 |
| **uniform** | **1/8** | 0.1250 | 4.969 | **14.6423** | 28.09 | 14.01 | 1.000 | 1.000 |

**Pareto readings (1B).**
- **Iso-accuracy** (smallest budget with measured PPL ≤ uniform-1/8 = 14.6423):
  **1/8** (measured 14.4202). 1/12 already exceeds (14.7048 > 14.6423), so for the
  small model you **cannot drop below the 1/8 budget** at iso-accuracy. The
  allocation's win here is **accuracy at fixed budget** (−0.22 PPL at 1/8), not a
  budget/speed reduction.
- **Iso-speed** (best PPL within ±1% prefill latency of uniform-1/8): **measured-1/8
  = 14.4202** (−0.22 vs uniform). Higher-accuracy points (3/16…1) cost latency and
  fall outside the ±1% band.
- Latency is **flat (×1.00 ± ~2% noise) for all budgets ≤ 1/8**, and rises in the
  upper wing to **prefill ×0.907 at r=1** (≈10% slower, pure MXFP8). Decode ≈ flat.

---

## Llama-3.2-3B (d=3072) — COMPLETE

| alloc | budget | avg FP8 | bits/w | PPL | prefill ms | decode ms | prefill × | decode × |
|---|---|---|---|---|---|---|---|---|
| measured | 0      | 0.0144 | 4.554 | 10.9085 | 54.39 | 24.05 | 0.997 | 1.031 |
| measured | 1/32   | 0.0421 | 4.658 | 8.4465 | 55.67 | 25.94 | 0.974 | 0.956 |
| measured | 1/24   | 0.0513 | 4.692 | 8.4328 | 54.17 | 25.26 | 1.001 | 0.982 |
| **measured** | **1/16** | 0.0697 | 4.761 | **8.3774** | 53.33 | 23.85 | 1.016 | 1.039 |
| measured | 1/12   | 0.0882 | 4.831 | 8.3740 | 55.55 | 26.21 | 0.976 | 0.946 |
| **measured** | **1/8** | 0.1250 | 4.969 | **8.3124** | 54.70 | 25.10 | 0.991 | 0.988 |
| measured | 3/16   | 0.1803 | 5.176 | 8.2975 | 56.89 | 25.24 | 0.953 | 0.982 |
| measured | 1/4    | 0.2356 | 5.383 | 8.2841 | 57.94 | 24.84 | 0.935 | 0.998 |
| measured | 3/8    | 0.3462 | 5.798 | 8.2219 | 57.90 | 25.35 | 0.936 | 0.978 |
| measured | 1/2    | 0.4558 | 6.209 | 8.1935 | 58.34 | 25.48 | 0.929 | 0.973 |
| measured | 3/4    | 0.6653 | 6.995 | 8.1213 | 59.73 | 25.00 | 0.907 | 0.992 |
| measured | 1 (≈MXFP8) | 0.8846 | 7.817 | 8.0385 | 64.16 | 26.06 | 0.845 | 0.952 |
| uniform  | 1/16   | 0.0697 | 4.761 | 8.4613 | 54.35 | 24.52 | 0.997 | 1.011 |
| **uniform** | **1/8** | 0.1250 | 4.969 | **8.3924** | 54.20 | 24.80 | 1.000 | 1.000 |

**Pareto readings (3B).**
- **Iso-accuracy**: **1/16** (measured 8.3774 ≤ uniform-1/8 8.3924) — **half the FP8
  budget** at iso-accuracy; 1/24 (8.4328) already exceeds. Speedup at 1/16: prefill
  ×1.016, decode ×1.039 (≈flat + noise — the low region is flat, see note).
- **Iso-speed** (±1% prefill of uniform-1/8): **measured-1/8 = 8.3124** (−0.08).
- **Upper wing (spend latency, buy accuracy):** 1/8 → r=1 improves PPL
  **8.3124 → 8.0385 (−0.27)** for **+18% prefill** (prefill ×0.845). The accuracy
  gain is smooth and monotone across the whole upper wing.
- **Pure NVFP4 (r=0) is a cliff:** 10.9085 — but a single 1/32 FP8 ring-budget
  recovers almost all of it (8.4465). The knee is extremely sharp at the bottom.
- Latency flat ≤ 1/8; upper wing rises to **prefill ×0.845 at r=1** (≈18% slower).

---

## Llama-3-8B (d=4096) — all latency DONE; measured PPL filling in (sharded across GPUs)

| alloc | budget | avg FP8 | bits/w | PPL | prefill ms | decode ms | prefill × | decode × |
|---|---|---|---|---|---|---|---|---|
| measured | 0      | —      | —     | *(pending)* | 82.69 | 31.42 | 1.009 | 0.962 |
| measured | 1/16   | 0.0691 | 4.759 | 6.6663 | 85.33 | 31.18 | 0.978 | 0.970 |
| **measured** | **1/8** | 0.1250 | 4.969 | **6.5996** | 84.75 | 31.18 | 0.984 | 0.970 |
| measured | 3/8    | 0.3487 | 5.808 | 6.5211 | 93.61 | 31.44 | 0.891 | 0.962 |
| measured | 3/4    | 0.6754 | 7.033 | 6.3877 | 96.81 | 31.87 | 0.862 | 0.949 |
| measured | 1 (≈MXFP8) | —  | —     | *(pending)* | 103.23 | 32.52 | 0.808 | 0.930 |
| uniform  | 1/16   | 0.0691 | 4.759 | 6.7116 | 82.88 | 31.13 | 1.007 | 0.971 |
| **uniform** | **1/8** | 0.1250 | 4.969 | **6.6367** | 83.42 | 30.23 | 1.000 | 1.000 |

**Pareto readings (8B, preliminary — remaining measured PPLs still computing).**
- **Iso-accuracy so far**: **1/8** (measured 6.5996 ≤ uniform-1/8 6.6367; measured-1/16
  = 6.6663 just exceeds). Like 1B, 8B does **not** drop below 1/8 at iso-accuracy —
  the measured win is −0.037 PPL at the matched 1/8 budget (1/12 result pending may
  refine this).
- **Upper wing:** 1/8 → 3/4 improves PPL 6.5996 → 6.3877 (−0.21) for +16% prefill.
- Latency flat ≤ 1/8, upper wing rises to **prefill ×0.808 at r=1** (≈24% slower) —
  the steepest of the three (bigger model → bigger K → steeper budget→latency slope).

(Full 8B grid in the CSV; measured PPL being filled by 4 GPU-sharded resume jobs.)

---

## Cross-model scaling & honest notes

- **Latency payoff/cost of the budget knob grows with model size.** Pushing the
  budget from 1/8 → r=1 (pure MXFP8) costs **+10% (1B) / +18% (3B) / +24% (8B)**
  prefill. Symmetrically the *savings* from 1/8 → pure NVFP4 are small on all three
  because the low-budget region is **flat** (the fused 128-tile kernel runs a fixed
  total-K mainloop; shrinking K_high below ~2–3 tiles removes no tensor work — see
  `docs/rd_perf_diagnostic.md`). **The real latency movement lives in the UPPER
  wing**, and it scales up with K (model size). Decode (bandwidth-bound) stays ≈flat.
- **Accuracy headroom shrinks with model size at low budget.** 3B holds iso-accuracy
  down to **1/16** (half the FP8); 1B cannot drop below **1/8** at iso-accuracy
  (small models are more NVFP4-sensitive). The measured-MSE allocation still beats
  uniform at the matched 1/8 budget for both (1B −0.22, 3B −0.08 PPL).
- **Kernel edge behaviour (K_low==0 vs K_high==0).** The fused kernel handles
  **K_high==0 (pure-NVFP4) gracefully** (r=0 ran with `fallback=False`), but **does
  NOT handle K_low==0 (pure-MXFP8)** — a Linear with all rings FP8 is rejected and
  (latency) falls back to a near-uniform allocation retaining one FP4 ring, and
  (PPL) the two-level NVFP4 weight quantizer reduces an empty block (`amax` crash).
  The driver therefore **clamps every Linear to ≥1 NVFP4 ring**, so r=1 is
  "all-but-one-ring MXFP8" — a faithful pure-MXFP8 endpoint anchor. This asymmetry
  is worth fixing in the kernel if a true pure-MXFP8 path is ever wanted.
- **R1-off lower bound.** All PPLs use R1-off (identity main rotation); the shipped
  R-on recipe is slightly better in absolute PPL and will shift the iso-accuracy
  knee. This sweep is the **R1-off characterization** (controlled lower-bound
  curve); the parallel `feat/rd-ron` work provides the R-on shift. Specifically the
  R-on **ring128 substrate shifts the accuracy column down ~0.08–0.1 PPL at 1/8**
  (e.g. 3B measured-1/8 ≈ 8.289 vs 8.3124 here, and it even overturns the
  uniform-optimal verdict at 3B@1/8) — those absolute numbers live on the `feat/rd-ron`
  branch. **Latency is rotation-independent, so the latency columns in this report
  are THE latency data for both the R1-off and R-on curves.** The measured-MSE-vs-
  uniform *deltas* here are the controlled signal.
- **4th model size:** the remote HF cache holds only the three llama sizes already
  swept (1B/3B/8B) plus Qwen3-8B (different architecture, same ~8B size class,
  incompatible with the llama-specific rotation/basis pipeline). No 7B/13B/3.1
  variants with weights on disk; nothing downloaded. The three llama sizes
  (d=2048/3072/4096) already form a clean scaling ladder.

## Reproduce

```bash
# per model (resumable; writes incremental JSON):
python3 promix/scripts/rd_sweep.py --model {1b|3b|8b} \
  --config promix/configs/llama-3.2-{1b|3b}-mxfp8-nvfp4.yaml \
  --phase {ppl|lat} \
  --budgets 0,0.03125,0.041667,0.0625,0.083333,0.125,0.1875,0.25,0.375,0.5,0.75,1.0
# then assemble:
python3 promix/scripts/rd_sweep_merge.py <json_dir> docs/rd_pareto_sweep.csv
```
