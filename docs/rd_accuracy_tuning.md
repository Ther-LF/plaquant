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
  anchor — the largest single accuracy lever found in the campaign so far.
  Same mechanism as 3B: huge uniform rotation cost (ε = +0.31) but the
  measured allocation gain (−0.62 from ring-uniform) dominates.
- 3B: digit-exact repro of the rd_ron ablation (−0.035 vs shipped measured,
  −0.065 vs dense uniform).
- 8B (rd_ron): 6.5890 vs 6.5882 — tie; C4 at stack validation.

### Lever 2 — ring-variance ring selection (per-Linear uniform count)

| point | 1B wiki | 3B wiki |
|---|---|---|
| var_uniform (per-Linear top-var rings, dense R) | 14.4285 | TBD |
| dd_uniform (per-Linear top-ΔD rings, dense R) | 14.5145 | TBD |
| ring512 + var_uniform | 15.5846 | — |

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

### Levers 3-5 (TBD)


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
