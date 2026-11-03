# rd-acctune: Accuracy Tuning Campaign (push PPL below the shipped anchors)

**Branch:** `feat/rd-acctune` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, remote dir
`/home/luofan/plaquant-acctune` · **Date:** 2026-11-01 · **Status: COMPLETE**

## Headline

Every model gets a strictly better anchor at the SAME 1/8 FP8 budget, same
eval protocol (bridges digit-exact), zero new online cost (all levers are
offline: rotation substrate, scale-search width, GPTQ calibration):

| model | shipped wiki | **new wiki** | Δ | shipped C4 | **new C4** | Δ |
|---|---|---|---|---|---|---|
| 1B | 14.4717 | **14.1694** | **−0.302** | 20.1864 | **19.7110** | **−0.475** |
| 3B | 8.2678 | **8.2180** | **−0.050** | 11.3194 | **11.2549** | **−0.065** |
| 8B | 6.5882 (uniform) | **6.5642** | **−0.024** | 9.6355 | **9.5639** | **−0.072** |

New shipping recipes (per model — the stack is model-dependent):

- **1B**: ring512 substrate + measured-MSE @1/8 + act-MSE d3u2 + GPTQ nsamples
  1024 (`--substrate ring --ring_size 512 --alloc measured --gptq_nsamples
  1024`, `PROMIX_ACT_MSE_STEPS_UP=2`)
- **3B**: ring512 + measured + act-MSE d3u2 + weight-MSE depth 7 + GPTQ 512 +
  percdamp 0.025 (`PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7`,
  `--gptq_nsamples 512 --percdamp 0.025`)
- **8B**: ring512 + measured + act-MSE d3u2 + weight-MSE depth 7 + GPTQ 512
  (as 3B, without d025)

Common core promoted for ALL sizes: **ring512 R-on substrate + global
measured-ΔD allocation + act-MSE up-steps + GPTQ calibration ≥512**.

## Protocol gates (all PASS)

| gate | expected | measured |
|---|---|---|
| 3B bf16 bridge | 7.8148 | 7.8148 |
| 1B bf16 bridge | 13.1569 | 13.1569 |
| 8B bf16 bridge | 6.1368 | 6.1368 |
| 3B dense+measured @1/8 | 8.2678 | 8.2678 (digit-exact) |
| 1B dense+measured @1/8 | 14.4717 | 14.4717 (digit-exact) |
| 3B ring512+measured @1/8 (rd_ron repro) | 8.2328 | 8.2328 (digit-exact) |

## Per-lever isolated results

### Lever 1 — ring512 R-on substrate (T = U @ ring512_R1) + measured-MSE @1/8 → **PROMOTE**

| point | 1B wiki | 1B C4 | 3B wiki | 3B C4 | 8B wiki | 8B C4 |
|---|---|---|---|---|---|---|
| dense + measured (shipped) | 14.4717 | 20.1864 | 8.2678 | 11.3194 | 6.6072 | 9.6393 |
| dense uniform | 14.5483 | 20.3150 | 8.2974 | 11.3915 | 6.5882 | 9.6355 |
| ring512 uniform | 14.8557 | — | 8.4494 (rd_ron) | — | 6.5970 (rd_ron) | — |
| **ring512 + measured** | **14.2392** | **19.8676** | **8.2328** | **11.2968** | 6.5890 | **9.6076** |

- 1B had never been ring-tested: −0.233 wiki / −0.319 C4 — the single largest
  lever of the campaign. Mechanism as on 3B: the ring-uniform rotation cost is
  large (ε = +0.31) but the measured allocation gain (−0.62) dominates.
- 8B: wiki tie (+0.0008 vs uniform), C4 −0.028 → safe to promote universally.
- K alignment holds everywhere (512 % 32 = 0).

### Lever 2 — ring-variance ring selection (per-Linear uniform count) → **REJECT (subsumed)**

| point | 1B wiki | 3B wiki |
|---|---|---|
| var_uniform (per-Linear top-var rings, dense R) | 14.4285 | 8.2736 |
| dd_uniform (per-Linear top-ΔD rings, dense R) | 14.5145 | 8.2756 |
| ring512 + var_uniform | 15.5846 | 8.2486 |

- var_uniform reproduces the rd_whydecay footnote (14.4285) and does beat the
  dense measured allocation on 1B by 0.043 — but ring512+measured beats it by
  0.19, and on the ring substrate the per-Linear-uniform budget collapses on
  1B (15.58) and loses on 3B (8.2486 vs 8.2328). The global measured-ΔD
  threshold is what makes the ring substrate work → lever 2 is strictly
  dominated by lever 1.
- Within a fixed per-Linear budget, variance SELECTION beats ΔD selection
  (1B 14.4285 vs 14.5145; 3B 8.2736 vs 8.2756) — so measured-ΔD does NOT
  subsume the uniform-config selection choice; but the question is moot since
  the global-ΔD-threshold configuration dominates both.

### Lever 3 — calibration size → **PROMOTE GPTQ-512 (3B/8B) / GPTQ-1024 (1B); basis stays 512**

| variant (3B, dense+measured unless noted) | wiki | Δ |
|---|---|---|
| gptq512 (GPTQ 128→512, shipped basis) | 8.2512 | −0.017 |
| gptq1024 (GPTQ 128→1024, shipped basis) | 8.2421 | −0.026 |
| calib1024 full (per-head basis-1024 + GPTQ-1024 + ΔD 16k tok) | 8.2403 | −0.028 |

- GPTQ calibration is the active ingredient; recomputing the Step-0 PCA
  covariance at 1024 samples adds only −0.002 over GPTQ-1024 alone, and the
  4× ΔD collect changes nothing measurable → keep the shipped 512-sample
  basis, raise GPTQ nsamples.
- In-stack the optimum is model-dependent and non-monotone: 1B best at 1024
  (g512 14.2035 / g1024 **14.1694** / g2048 14.2333), 3B best at 512 (stack
  +g512+d025 8.2180 vs +g1024+d025 8.2346), 8B best at 512 (6.5642 vs
  g1024 6.5681). 2048 rejected.
- Pitfall for reproduction: the Step-0 basis MUST be generated with the
  fp-perhead config (`o_proj_pca: per_head`); generating it with the
  mxfp8-nvfp4 config (full_global) destroys the model (PPL 440k) under the
  per-head eval protocol.

### Lever 4 — activation NVFP4 scale-search widening → **PROMOTE d3u2**

Candidates = base (clip-free ceil) + D lower + U upper E4M3 steps
(`PROMIX_ACT_MSE_STEPS` / new `PROMIX_ACT_MSE_STEPS_UP`); 3B dense+measured:

| variant | candidates | 3B wiki | Δ |
|---|---|---|---|
| shipped | d3 u0 (4) | 8.2678 | — |
| **up2** | d3 u2 (6) | **8.2362** | **−0.032** |
| d3u3 | d3 u3 (7) | 8.2576 | −0.010 |
| d7u3 | d7 u3 (11) | 8.2627 | −0.005 |
| d0u2 (up-only) | d0 u2 (3) | 8.3411 | +0.073 |

The round-2 "7-step ≈ 3-step → saturated" conclusion was a DOWN-only
artifact: the win was hiding in the UP direction, and exactly 2 upper steps
(u3 already worse; down deeper than 3 also worse). Down steps remain
essential. On the ring512 substrate the isolated −0.032 shrinks (3B stack
−0.001) but it still helps 1B (−0.030 on r512) and never hurts → promoted
everywhere. Deployment note: the CUDA producer search (act_mse.cuh) needs
the 2 extra upper candidates (6 total vs 4) — same branchless structure,
~1.5× the (cheap) selection cost.

### Lever 5 — weight micro-scale depth + GPTQ knobs → **PROMOTE w7 (3B/8B) + d025 (3B only); REJECT act_order**

| variant (3B, dense+measured) | wiki | Δ |
|---|---|---|
| **wmse7** (`PROMIX_WEIGHT_MSE_STEPS=7`) | **8.2426** | −0.025 |
| wmse15 (in final stack) | 8.2180 | = w7 digit-exact → search saturates at 7 |
| percdamp 0.025 | 8.2457 | −0.022 |
| percdamp 0.05 | 8.2643 | −0.004 |
| act_order=true | 22.2607 | BROKEN — incompatible with the reorder / frozen-block-scale pipeline; do not enable without a dedicated fix |

Model-dependence in the stack: w7 helps 3B (−0.008 on r512) and 8B, but HURTS
1B (+0.094 on r512). d025 helps 3B only when combined with g512 (without:
8.2312), hurts 1B (+0.061), and is a wash on 8B (wiki −0.002 / C4 +0.010).
w7 is weight-repack-only (zero forward cost).

### Lever 6 — KV4 re-measure → **BLOCKED (mechanical)** *(fixed post-campaign: see docs/rd_kv4_fix.md)*

`MXFP8 quantization requires K (16) divisible by block_size (32)`: with
k_groupsize = head_dim = 128 and high_fraction 1/8, the per-head KV high
segment is 16 channels < the MXFP8 block of 32. The current KV-quant path
cannot express the kv4 configs under this harness; needs a dedicated fix
(e.g. per-head high segment ≥32 or NVFP4-only KV), out of campaign scope.
The W4A4KV4 row is NOT updated.

*Post-campaign resolution (rd-kv4, 2026-11-03): fixed on `feat/rd-kv4` by
rounding the per-head KV high segment up to 32 (`kv_fp_high: up`) +
per-head FP groupsize support in ActQuantizer; W4A4KV4 measured on the
final recipes above — see `docs/rd_kv4_fix.md` for the table.*

## Stacking table (ring512+measured substrate; winners bold)

| stack | 1B wiki | 1B C4 | 3B wiki | 3B C4 | 8B wiki | 8B C4 |
|---|---|---|---|---|---|---|
| shipped anchor (dense; measured for 1B/3B, uniform for 8B) | 14.4717 | 20.1864 | 8.2678 | 11.3194 | 6.5882 | 9.6355 |
| r512 | 14.2392 | 19.8676 | 8.2328 | 11.2968 | 6.5890 | 9.6076 |
| r512+up2 | 14.2093 | 19.8041 | 8.2320 | 11.2802 | 6.5815 | 9.5915 |
| r512+w7 | 14.3334 | — | 8.2245 | — | — | — |
| r512+up2+w7 | 14.2851 | 19.8356 | 8.2214 | 11.2822 | 6.5785 | 9.5868 |
| r512+up2+w7+g512 | — | — | 8.2203 | 11.2782 | **6.5642** | **9.5639** |
| r512+up2+w7+g512+d025 | — | — | **8.2180** | **11.2549** | 6.5624 | 9.5735 |
| r512+up2+w7+g1024(±d025) | — | — | 8.2346 | 11.2683 | 6.5681 | 9.5778 |
| r512+up2+w7+d025 (no g512) | — | — | 8.2312 | — | — | — |
| r512+up2+g512 | 14.2035 | 19.7121 | — | — | — | — |
| **r512+up2+g1024** | **14.1694** | **19.7110** | — | — | — | — |
| r512+up2+g2048 | 14.2333 | 19.8427 | — | — | — | — |
| r512+up2+d025 | 14.2704 | 19.9047 | — | — | — | — |

Cross-lever interactions are real and model-dependent; the harness noise
floor is ~±0.005 wiki, so each final stack choice is supported by both
datasets moving together.

## What the new SHIPPED anchors should be

- **W4A4(FP) 1/8-budget anchors: 1B 14.17 / 19.71 · 3B 8.22 / 11.25 ·
  8B 6.56 / 9.56 (wiki/C4)**, recipes above. All offline-only changes: same
  kernel, same K_high alignment, same online cost as the previous ship.
- 8B note: the previous ship was uniform (allocation OFF at 8B per rd_ron);
  the new 8B recipe turns the measured allocation back ON via the ring512
  substrate — the full stack is what first beats uniform on 8B on both
  datasets.
- The dense-R-on "simplest story" (dense + measured) remains the fallback for
  contexts that cannot bake a new rotation; its anchors are unchanged.

## Reproduce

```bash
# driver: one point per invocation (env knobs isolate per process)
# 1B final
PROMIX_ACT_MSE_STEPS_UP=2 python promix/scripts/rd_acctune.py \
  --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml --point 1b_final \
  --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 1024 --c4 --out OUT.json
# 3B final
PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 python promix/scripts/rd_acctune.py \
  --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --point 3b_final \
  --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 512 --percdamp 0.025 --c4 --out OUT.json
# 8B final
PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 python promix/scripts/rd_acctune.py \
  --config promix/configs/llama-3-8b-mxfp8-nvfp4.yaml --point 8b_final \
  --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 512 --c4 --out OUT.json
```

Raw JSONs/logs: `/home/luofan/acctune/*.{json,log}` (remote ICT90). Points of
record: `1b_stack_r512_up2_g1024`, `3b_stack_full_g512_d025`
(`3b_stack_w15_g512_d025` identical), `8b_stack_full_g512`.
