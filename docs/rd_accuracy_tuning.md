# rd-acctune: Accuracy Tuning Campaign (push PPL below the shipped anchors)

**Branch:** `feat/rd-acctune` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, remote dir
`/home/luofan/plaquant-acctune` · **Date:** 2026-11-01 · **Status: IN PROGRESS**

## Mission

Shipped accuracy anchors (dense-R-on, measured-MSE @1/8, our protocol):

| model | wiki (shipped) | C4 (shipped) | bf16 bridge |
|---|---|---|---|
| 1B | 14.4717 | 20.1864 | 13.1569 |
| 3B | 8.2678 | 11.3194 | 7.8148 |
| 8B | 6.5882 (uniform; measured 6.6072) | 9.6355 | 6.1368 |

Measure each lever ISOLATED on 3B, stack winners, validate the stack on
1B/3B/8B (wiki + C4). Protocol gates: bridges 13.1569 / 7.8148 / 6.1368 must
reproduce (PASS, see below).

## Protocol gates

| gate | expected | measured | verdict |
|---|---|---|---|
| 3B bf16 bridge | 7.8148 | 7.8148 | PASS |
| 1B bf16 bridge | 13.1569 | 13.1569 | PASS |
| 3B dense+measured @1/8 | 8.2678 | 8.2678 | PASS (digit-exact) |
| 1B dense+measured @1/8 | 14.4717 | 14.4717 | PASS (digit-exact) |
| 3B ring512+measured @1/8 (rd_ron repro) | 8.2328 | 8.2328 | PASS (digit-exact) |

## Levers (isolated, 3B wiki unless noted)

### Lever 1 — ring512 R-on substrate (T = U @ ring512_R1) + measured-MSE @1/8

| point | 1B wiki | 3B wiki |
|---|---|---|
| dense + measured (shipped anchor) | 14.4717 | 8.2678 |
| ring512 uniform | 14.8557 | 8.4494 (rd_ron) |
| **ring512 + measured** | **14.2392 (−0.233)** | **8.2328 (−0.035)** |

- 1B (never ring-tested before): ring512+measured −0.233 vs the shipped
  anchor — the largest single accuracy lever found in the campaign.
  Same mechanism as 3B: huge uniform rotation cost (ε = +0.31) but the
  measured allocation gain (−0.62 from ring-uniform) dominates.
- 3B: digit-exact repro of the rd_ron ablation (−0.035 vs shipped measured,
  −0.065 vs dense uniform).
- C4 fully corroborates (isolated lever-1 runs): 1B 19.8676 (shipped 20.1864,
  **−0.319**); 3B 11.2968 (shipped 11.3194, −0.023); 8B 9.6076 (shipped
  uniform 9.6355, −0.028). 8B wiki 6.5890 ties uniform 6.5882 (+0.0008),
  bridge 6.1368 reproduced.
- **VERDICT: PROMOTE ring512+measured as the universal shipping substrate**
  (wins 1B/3B on both datasets, ties-or-wins 8B; K_high alignment holds,
  512 % 32 = 0).

### Lever 2 — ring-variance ring selection (per-Linear uniform count)

| point | 1B wiki | 3B wiki |
|---|---|---|
| var_uniform (per-Linear top-var rings, dense R) | 14.4285 | 8.2736 |
| dd_uniform (per-Linear top-ΔD rings, dense R) | 14.5145 | 8.2756 |
| ring512 + var_uniform | 15.5846 | 8.2486 |

- var_uniform reproduces the rd_whydecay footnote (14.4285) and beats the
  dense measured allocation on 1B by 0.043 — BUT it is beaten by
  ring512+measured (14.2392) by 0.19, and on the ring512 substrate the
  per-Linear-uniform selection collapses (15.58): the global measured-ΔD
  threshold is what makes the ring substrate work. Verdict: lever 2 is
  SUBSUMED by lever 1 — reject as a shipping config, keep as the
  dense-substrate fallback note.
- dd_uniform (ΔD selection at per-Linear uniform count) is WORSE than
  var_uniform by +0.086 on 1B: within a fixed per-Linear budget the variance
  ranking is the better selector; ΔD's edge is only as a global
  budget-distribution signal.

### Lever 4 — activation NVFP4 scale-search widening (3B, dense+measured base 8.2678)

Candidates = base E4M3 scale + D lower steps + U upper steps
(`PROMIX_ACT_MSE_STEPS` / `PROMIX_ACT_MSE_STEPS_UP`):

| variant | candidates | 3B wiki | Δ vs base |
|---|---|---|---|
| shipped | d3 u0 (4) | 8.2678 | — |
| **up2** | d3 u2 (6) | **8.2362** | **−0.032** |
| d3u3 | d3 u3 (7) | 8.2576 | −0.010 |
| d7u3 | d7 u3 (11) | 8.2627 | −0.005 |
| d0u2 (up-only) | d0 u2 (3) | 8.3411 | +0.073 |

The round-2 "7-step ≈ 3-step" saturation was a DOWN-only artifact: the win was
hiding in the UP direction, and exactly 2 upper steps (u3 is already worse, and
deepening down past 3 hurts). But down steps remain essential (up-only is
+0.073). Isolated verdict: **d3u2 wins −0.032 on the dense substrate** —
interaction with ring512 measured in the stack section.

### Lever 5 — weight micro-scale MSE deepening + GPTQ knobs (3B)

| variant | 3B wiki | Δ vs base |
|---|---|---|
| **wmse7** (`PROMIX_WEIGHT_MSE_STEPS=7`) | **8.2426** | **−0.025** |
| act_order=true | 22.2607 | +14 (BROKEN — incompatible with the reorder/frozen-scale pipeline; reject, do not re-try without a dedicated fix) |
| percdamp 0.025 / 0.05 | TBD | — |

### Lever 3 — calibration size

| variant | 3B wiki | Δ |
|---|---|---|
| gptq512 (GPTQ nsamples 128→512, shipped basis) | 8.2512 | −0.017 |
| calib1024 (basis 1024 + GPTQ 1024 + ΔD 16k tokens) | TBD | — |

First calib1024 attempt EXPLODED (PPL 440k): the 1024-sample basis had been
generated with the mxfp8-nvfp4 config (o_proj_pca full_global) while the eval
protocol forces per_head — basis regenerated with the fp-perhead config.
Note for reproduction: Step-0 basis MUST be generated with
`llama-3.2-3b-fp-perhead.yaml` (per-head o_proj), not the mxfp8-nvfp4 config.

## Stacking (ring512+measured substrate)

| stack | 1B wiki | 1B C4 | 3B wiki | 3B C4 | 8B wiki | 8B C4 |
|---|---|---|---|---|---|---|
| shipped anchor (dense+measured; 8B uniform) | 14.4717 | 20.1864 | 8.2678 | 11.3194 | 6.5882 | 9.6355 |
| r512 | 14.2392 | 19.8676 | 8.2328 | 11.2968 | 6.5890 | 9.6076 |
| r512+up2 | **14.2093** | **19.8041** | 8.2320 | 11.2802 | TBD | TBD |
| r512+w7 | 14.3334 | — | 8.2245 | — | — | — |
| r512+up2+w7 | 14.2851 | 19.8356 | **8.2214** | 11.2822 | **6.5785** | **9.5868** |

Interactions are real and model-dependent:
- up2's dense-substrate −0.032 shrinks to −0.001 on 3B ring512 (the ring
  Hadamard reshapes activation blocks), but still −0.030 on 1B ring512.
- w7 stacks on 3B (−0.008 on r512) and helps the 8B stack, but HURTS 1B
  (+0.094 on r512) — per-model stack selection required.
- 8B: the full stack is the first configuration to beat the shipped uniform
  anchor on BOTH datasets (wiki −0.010, C4 −0.049).



## Best stack (1B/3B/8B × wiki/C4) vs shipped anchors

(TBD)

## Reproduce

```bash
# one point per invocation (env knobs isolate per process)
python promix/scripts/rd_acctune.py --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
    --point 3b_ring512_measured --substrate ring --ring_size 512 --alloc measured \
    --out /home/luofan/acctune/P.json
```
Raw JSONs: `/home/luofan/acctune/*.json` (remote).
