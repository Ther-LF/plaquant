# R-on Integration (Task 6) — measured-MSE allocation on the uniform 64-ring R1

**Branch** `feat/rd-ron` · **date** 2026-10-28 · **spec** `docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md` §3

## What this is

Rounds 11/13 concluded **4×** that per-layer precision allocation "cannot beat
R-on uniform" — but every time, the allocation machinery was forced to run
**R1-OFF** (identity main rotation, `T = U`), throwing away the dense R1's
0.09–0.13 PPL gain. This report is the **first test of the flip lever**: wire the
uniform 64-ring block-diagonal R1 (`T = U @ ring_R1`, fixed Hadamard per
64-ring) into the PTQ pipeline so the rotation is **preserved** while allocation
still works per-ring, then run measured-MSE allocation **on top of R-on**.

Only the hidden-dim residual-stream rotation is swapped (embeddings/lm_head/
q/k/v/gate/up weights + o_proj/down_proj output). Per-head R2 / value / key
rotations and the down_proj online Hadamard are **unchanged**. o_proj is held at
uniform 1/8 in every config (controlled constant). Opt-in `ring_r1` flag; default
off = byte-identical shipped path.

## Sanity gates

| gate | 3B | 8B | verdict |
|---|---|---|---|
| (a) bf16 bridge == floor | 7.8148 (floor 7.82) | 6.1368 (floor 6.14) | PASS |
| fp16 bridge | 7.8113 | 6.1346 | — |
| (b) ring-rotation-only, **no quant** == fp16 (lossless) | 7.8121 (Δ +0.0008) | 6.1347 (Δ +0.0001) | PASS |

Gate (b) is the fusing-bug catch: an orthogonal rotation must leave PPL
unchanged. Δ < 0.001 on both 3B and 8B → the ring-R1 fuse (incl. the 64-ring
block-diagonal Hadamard baked into embeddings + per-layer weight absorption +
tie-word-embeddings untie/clone) is **numerically exact**. 8B (hidden=4096 =
64 rings of 64) uses the identical machinery — confirmed lossless.

## 3B — the @1/8 three-way ranking + ring-vs-dense cost

Matched average FP8 ring-budget = 1/8 (93184 / 745472 channels) across the
q/k/v/gate/up/down reallocated units. Same protocol as `rd_validity`; bf16
bridge reproduces 7.82, dense-R-on uniform reproduces the published 8.293.

| config @ 1/8 | PPL | note |
|---|---|---|
| dense-R-on uniform (shipped floor) | **8.2974** | published 8.293 reproduced |
| ring-R-on uniform | 8.3452 | gate (c): between R1-off 8.392 and dense 8.297 |
| ring-R-on variance-proxy | 8.4632 | worse than uniform (mis-picks rings) |
| ring-R-on measured-MSE | **8.3097** | best of the three ring configs |

- **3-way ranking @1/8 (best→worst): measured 8.3097 < uniform 8.3452 <
  variance 8.4632.** The measured-MSE < variance-proxy < uniform?-no:
  measured < uniform < variance. The gate ranking **measured < variance AND
  measured ≤ uniform HOLDS under ring-R-on** (re-verified: the ring Hadamard's
  per-64-ring internal mixing did not invert the R1-off ranking). The
  variance-proxy remains a bad allocator (as in R1-off: 8.47 there).
- **Ring-vs-dense rotation cost** = ring uniform − dense uniform =
  8.3452 − 8.2974 = **+0.0478 PPL**. This is the price of the 64-ring R1 mixing
  less than the dense full-top-block R1.
- **Allocation gain on ring-R-on** = ring uniform − ring measured =
  8.3452 − 8.3097 = **0.0355 PPL** (measured-MSE allocation recovers this over
  uniform at matched budget).

## 3B — the three key answers

**(i) Does ring-R-on + measured-MSE @1/8 beat dense-R-on uniform (8.2974 /
published 8.293)?**
**NO.** measured-MSE @1/8 = 8.3097 vs dense uniform 8.2974 → **ε = +0.0123**
(vs published 8.293 → +0.0167). The allocation gain (0.0355) does **not** cover
the ring-vs-dense rotation cost (0.0478); net +0.012 worse.

**VERDICT — the "uniform near-optimal" conclusion STANDS, flip lever exhausted.**
This is the first run where allocation kept R (via the ring substrate), and it
still does not beat dense-R-on uniform at matched 1/8. The reason is now
quantified and specific: the ring R1 is a *weaker* rotation than the dense R1
(+0.048), and measured-MSE allocation only buys back +0.036 at iso-budget.
The lever is real (allocation works: measured beats ring-uniform by 0.036) but
too small to overcome the rotation it had to trade away.

**(ii) Iso-accuracy (PPL ≤ 8.2974) minimum budget on ring-R-on:** _[sweep in
progress]_ — but note measured-MSE @1/8 already sits **above** the dense floor
(8.3097 > 8.2974), and PPL only rises as the budget drops, so no budget ≤ 1/8
reaches the dense floor. (Crossing happens on the upper wing, budget > 1/8.)

**(iii) Best PPL at budget 1/8 (iso-speed):** min(uniform, variance, measured) =
**8.3097** (measured-MSE). At iso-speed (same 1/8 FP8 count), measured-MSE
allocation is the best ring-R-on operating point, 0.036 better than uniform.

## 3B — measured-MSE budget sweep (ring-R-on)

_[populated as the sweep completes — downward iso-accuracy hunt
{1/12, 1/16, 1/24, 1/32} then upper Pareto wing {3/16, 1/4, 3/8, 1/2, 1.0}]_

| budget | FP8 rings | avg FP8 | PPL |
|---|---|---|---|
| 1/8 (from @1/8 block) | 1288 | 0.1250 | 8.3097 |

## 8B

_[gates PASS above; @1/8 ranking + sweep {1/32,1/16,1/12,1/8,1/4} in progress]_

## Code (opt-in, zero-regression)

- `promix/quantize/rotation.py` — `build_main_transform` / `fuse_basis_to_model`
  gain `ring_r1` + `ring_size` kwargs. `ring_r1=True` → `T = U @ ring_R1`.
  Default off = byte-identical dense/identity path.
- `promix/eval/ptq.py` — `prepare_ptq_model` reads `quantize.ring_r1` /
  `quantize.ring_size` (default off).
- `promix/scripts/rd_validity.py` — `build_base`/`build_eval` gain `ring_r1`.
- `promix/scripts/rd_ron.py` — the R-on driver (gates, @1/8 three-way,
  measured-MSE sweep, flip-lever verdict) + generalized `resolve_paths` (1B/3B/8B).
- `tests/test_ring_rotation.py` — 6 `build_main_transform` wiring tests (24 total pass).
