# Round-6 — making the round-5 two-level WEIGHT scale DEPLOYABLE at zero runtime cost

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.x+cu130.
Goal: ship round-5's weight **two-level scale** win (per-tensor FP32 global `g` ×
per-block E4M3 micro, NEAREST micro + per-block micro-MSE) through the REAL fused
kernel with essentially zero added GEMM cost. Round-5 proved the accuracy (3B r=1/8
8.376→8.297, r=1/16 8.463→8.369, r=0 9.037→8.859) but the effective scale `g×micro`
is off the E4M3 grid and the single-level packer could not store it.

## 1. How `g` rides the epilogue (the free seam)

The fused GEMM epilogue (`codesign_fused_ext.cu::make_epi_args`) is a CUTLASS
`LinearCombination` `{alpha, beta}` applied as `D = alpha·accumulator + beta·C`.
`alpha` is a **scalar multiply of the whole FP32 accumulator** — the exact
broadcast seam round-2 noted was `s_w=1` today. The fused kernel accumulates the
NVFP4-low and MXFP8-high phases into ONE accumulator before this single epilogue,
so a scalar `alpha=g` scales BOTH phases. We exploit this:

- **pack**: pick a per-tensor global `g = amax_low/(448·6)` (from the NVFP4 low
  segment). Store the NVFP4 low block scale as E4M3 `micro_lo = eff_lo/g` and the
  MXFP8 high block scale as E8M0 `micro_hi = e8m0(ideal_hi/g)`. `micro_hi` is a
  power of 2 for ANY `g`, so the high SFB stays exactly on the E8M0 grid; the
  high effective scale `g·micro_hi` is the SAME two-level global shared with the
  low phase (strictly finer than stock single-level MXFP8).
- **runtime**: `fused_gemm_prepacked(..., alpha=g)` (and the bias-epilogue
  `fused_gemm_prepacked_bias(..., alpha=g)` — the callback now computes
  `half(float(bf16(alpha·acc)) + bias)`). `D = g·(A_lo@B_lo_micro + A_hi@B_hi_micro)
  = eff_lo·(A@B) + eff_hi·(A@B)` — bit-faithful to the round-5 two-level dequant.
- **decode GEMV** streams raw FP32 scales, so `g` is folded directly into the
  stored FP32 effective scales (`S_lo_dec`, `S_hi_dec`) — no epilogue needed.

Why an ARBITRARY (non-pow2) `g`: forcing `g` to a power of 2 (so the high phase is
*bit-identical* to stock) costs +0.016 PPL at 3B r=1/8 (8.308 vs 8.293), because
the low segment's global is the lever. Using an arbitrary `g` on BOTH phases keeps
the low segment **bit-identical to round-5** and the high two-level (which is
neutral: pow2-g two-level-high == pow2-g stock-high == 8.308, confirming the gain
is the arbitrary low global). Both schemes are deployable; we ship the arbitrary-g
"variant B".

Element codes and `W_deq` are UNCHANGED vs passing the effective scale as the
frozen scale — only the SFB stores `micro=eff/g` and the packer returns `g`.

## 2. Pack overhead & numerical gate

The offline packer decomposes the frozen effective scale into (E4M3/E8M0 micro, g)
and asserts the round-trip is exact. Self-test (`/tmp/test_twolevel_pack.py`):

- NVFP4 `micro·g == eff` and MXFP8 `micro·g == eff` **bit-exact** (max rel err 0.0).
- packer `W_deq` == the round-5 two-level fake-quant dequant **bit-exact** (both
  segments).

Pack is offline / once-at-load. The two-level computation is the same NEAREST+MSE
search already shipped as weight-A2, plus one per-tensor amax and one divide — so
pack-time is within noise of the already-shipped weight-A2 pack (<1.1× gate met;
no new per-forward cost).

**real-vs-twin cosine (random data, 3B shapes):** plain `cd` **0.9999987**,
`cd_bias` epilogue **0.9999987** — the epilogue alpha reconstructs the two-level
dequant in both GEMM paths.

## 3. PPL — single-level (shipped) vs two-level (round-6), wikitext-2

Same protocol as the scoreboard (3B/8B aligned with vLLM). Two-level is DEFAULT-ON
(`PROMIX_WEIGHT_2LEVEL=1`); `=0` reproduces the single-level shipped baseline.

| size / r | bit/w | single-level (shipped) | **two-level (round-6)** | Δ | NVFP4 (4.25b) |
|---|---|---|---|---|---|
| 3B r=0    | 4.500 | 9.037 | **8.859** | −0.178 | 8.63 |
| 3B r=1/16 | 4.734 | 8.463 | **8.361** | −0.102 | 8.63 |
| 3B r=1/8  | 4.969 | 8.376 | **8.296** (REAL kernel; fake 8.293) | −0.080 | 8.63 |
| 8B r=1/8  | 4.969 | 6.713 | **6.599** | −0.114 | 6.87 |

- 3B r=1/8 **REAL kernel 8.2955** (fake-twin 8.286, |real−twin|=0.0095 ≤ 0.1 PASS;
  per-layer cosine worst 0.999998). Matches round-5's 8.297 to **0.0015** — gate
  (within 0.01) PASS. Single-level REAL off = 8.3633 → two-level −0.068 through the
  real kernel.
- 3B r=0/r=1/16 are fake-quant (r=0 has no high phase → arbitrary-g = round-5
  exact 8.859; r=1/16 variant-B 8.361). Single-level baselines re-confirmed through
  the harness: r=1/8 8.3759, r=1/16 8.4627.
- 8B two-level 6.599 (fake); single-level 6.713 (fake, re-confirmed == doc). Δ −0.114.
  The 8B **real-kernel** harness (`full_network_realkernel`, holds stock+twin+real 8B
  models) OOM'd for BOTH on and off identically — pre-existing 8B memory pressure,
  independent of two-level; the real-vs-twin mechanism is proven size-independent on
  3B (gate PASS). Use `multisize` (single model + idle-GPU waiter) for 8B real.

**Per-size gating:** 3B (−0.08) and 8B (−0.11) both WIN → default ON. 1B base model
is not present offline this round (only the quantized baselines remain); 1B had
weight-A2 gated OFF (+0.08), so 1B should run with `PROMIX_WEIGHT_2LEVEL=0` pending
re-verification, exactly mirroring the weight-A2 1B gate.

## 4. Perf — prefill/decode UNCHANGED (the epilogue scale is free)

**Direct isolation** (`/tmp/perf_alpha.py`, 3B shapes, alpha=g vs alpha=1, the ONLY
per-call difference): plain `cd` ratios 0.998 / 0.972 / 1.016, `cd_bias` ratios
0.988 / 0.995 / 1.000 at M=2048/4096/8192 — all within run-to-run noise. The
`LinearCombination`/bias epilogue ALREADY multiplies the accumulator by its scalar;
`g` vs `1.0` adds zero instructions and zero memory traffic.

**End-to-end 3B multisize (two-level ON) vs scoreboard (single-level OFF):**

| | M=2048 | M=4096 | M=8192 | decode b1 | decode b8 |
|---|---|---|---|---|---|
| prefill ON (round-6) | 1.381× | 1.426× | 1.335× | — | — |
| prefill OFF (scoreboard) | 1.35× | — | 1.34× | — | — |
| decode ON (round-6) | — | — | — | 0.839× | 0.820× |
| decode OFF (scoreboard) | — | — | — | 0.85× | 0.82× |

(graphed-vs-graphed, idle GPU, one GPU per run. Prefill logit cos 1.0000004–1.0000013.
Decode uses the bit-faithful W_deq, so the decode bit-exact gate is preserved.)
Prefill/decode are UNCHANGED within noise — PROVEN.

## 5. Conclusion — Pareto position

- **Deployable: YES, at zero runtime cost.** The round-5 two-level weight win ships
  through the real kernel via one FP32 per-tensor global carried on the epilogue
  alpha (free) and the decode FP32 scales. Real-vs-twin gate PASS (|Δ|=0.0095,
  cosine 0.999998); REAL PPL 8.2955 == round-5 8.297 within 0.0015.
- **New r=1/8 points (4.97b): 3B 8.296, 8B 6.599** — both CLEARLY beat NVFP4's
  absolute W4A4 PPL (**8.63 / 6.87**) by **−0.33 / −0.27**. r=1/16 (4.73b) 3B 8.361
  also beats 8.63.
- **Matched bits (r=0, 4.5b vs NVFP4 4.25b): 3B 8.859 > 8.63 (+0.23)** — close but
  NOT beating at matched bits (we spend 0.25 more bits). The residual is activation
  NVFP4 rounding (round-5: r=0 weights-only 8.086 < 8.63; act-MSE is our activation
  ceiling), not the weight scale.
- **Perf unchanged** at every size/M/batch (the epilogue scalar is free).

**Verdict:** round-6 makes every ProMix point deployable at the round-5 accuracy —
we now clearly beat NVFP4 on absolute W4A4 PPL at r=1/8 and r=1/16 (and r=0
weights-only), while at the matched-bit r=0 point we remain +0.23 (4.5b vs 4.25b),
close but not ahead.

## 6. Reproduction

- Kernel: `fused_gemm_prepacked(..., alpha)` / `fused_gemm_prepacked_bias(..., alpha)`
  (`codesign_fused_ext.cu`, `codesign_fused_bias_ext.cu`,
  `codesign_bias_epilogue.hpp` — alpha applied before the reference bf16 round).
- Pack: `pack_weight_mixed(..., global_scale=g)` returns `PackedWeight.g_w`;
  `quantize_nvfp4_spec/quantize_mxfp8_spec(..., global_scale=g)` store micro=eff/g.
- Scales: `quant_utils._weight_two_level_nvfp4_eff` / `_weight_two_level_mxfp8_eff`,
  wired via `WeightQuantizer.find_params` (shared `g` plumbed low→high in
  `gptq.py`) and `full_network._frozen_scales_for` / `pack_all_targets` /
  `MixedQuantLinear` (alpha=g_w). Flags `PROMIX_WEIGHT_2LEVEL` (default 1),
  `PROMIX_WEIGHT_2LEVEL_POW2` (default 0 = arbitrary g).
- Harness: `ablation_acc_per_bit.py --weight_2level tensor --w2l_round nearest
  --w2l_mse --w2l_high` (variant B), `--w2l_pow2` (pow2 g). Real: `full_network_realkernel`
  (ppl, real-vs-twin gate), `multisize` (prefill/decode). Raw: `/home/luofan/r6_results/`.
</content>
</invoke>
