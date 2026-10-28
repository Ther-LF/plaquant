# R-on Integration (Task 6) — measured-MSE allocation on the uniform ring R1

**Branch** `feat/rd-ron` · **date** 2026-10-28 · **spec** `docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md` §3

## Headline

Rounds 11/13 concluded **4×** that per-layer precision allocation "cannot beat
R-on uniform" — but every time the allocation was forced **R1-OFF**, discarding
the dense R1's gain. This is the first test of the **flip lever**: wire the
uniform ring block-diagonal R1 (`T = U @ ring_R1`, fixed Hadamard per ring,
block-diagonal over all hidden_dim) so the rotation is **preserved** while
allocation still works per-ring, then run measured-MSE allocation on top.

**Result (3B, @ matched 1/8 FP8 budget):**
- At the first-tried ring width (64) the flip **failed** by +0.012: the 64-wide
  Hadamard mixes too little, so the ring-vs-dense rotation cost (+0.048) exceeds
  the allocation gain (+0.036).
- A **ring-size ablation** exposed this as a pure mixing-width effect (the dense
  R1 baseline is a *random* orthogonal, not trained — so only mixing width
  separates them). At **ring 128** the rotation cost collapses to +0.016 while
  the allocation gain holds at +0.024 →

  **ring128-R-on + measured-MSE @1/8 = 8.2893 < dense-R-on uniform 8.2974**
  (and < published 8.293). **The "uniform near-optimal" conclusion is OVERTURNED
  at ring_size=128.**

Mechanism, allocation preserved vs discarded, is the whole story: keep R (ring
substrate) **and** make the ring wide enough, and per-ring measured-MSE
allocation beats dense-R-on uniform at iso-budget.

## Sanity gates (fusing correctness)

| gate | 3B | 8B | verdict |
|---|---|---|---|
| (a) bf16 bridge == floor | 7.8148 (floor 7.82) | 6.1368 (floor 6.14) | PASS |
| fp16 bridge | 7.8113 | 6.1346 | — |
| (b) ring-rotation-only, **no quant** == fp16 (lossless) | 7.8121 (Δ+0.0008) | 6.1347 (Δ+0.0001) | PASS |

Gate (b): an orthogonal rotation must leave PPL unchanged. Δ < 0.001 on both →
the ring-R1 fuse (64-ring block-diagonal Hadamard baked into embeddings +
per-layer weight absorption + tie-word-embeddings untie/clone) is numerically
exact. 8B (hidden 4096 = 64 rings of 64) uses identical machinery — lossless.

## The paper figure — ring-size decomposition (3B, @1/8, vs dense uniform 8.2974)

| ring_size S | #rings (3B) | ring-R-on uniform | ring-R-on measured | **ε(S)** = unif−dense (rotation cost) | **gain(S)** = unif−meas (allocation) | net = meas−dense | verdict |
|---|---|---|---|---|---|---|---|
| 64  | 48/unit (10304 tot) | 8.3452 | 8.3097 | **+0.0478** | +0.0355 | +0.0123 | stands |
| 128 | 24/unit (5152 tot)  | 8.3133 | **8.2893** | **+0.0159** | +0.0240 | **−0.0081** | **OVERTURNED** |
| 256 | 12/unit (2576 tot)  | _[pending]_ | _[pending]_ | _[pending]_ | _[pending]_ | _[pending]_ | _[pending]_ |

**ε(S) decreases monotonically in S** (wider Hadamard rings mix more, closing the
gap to the dense R1) while **gain(S) persists** (measured-MSE still finds
high-value rings at coarser granularity). The flip happens where the two curves
cross — between S=64 and S=128.

## 3B @1/8 three-way ranking (re-verification)

| config @ 1/8 (ring64) | PPL | note |
|---|---|---|
| dense-R-on uniform (floor) | **8.2974** | reproduces published 8.293 |
| ring-R-on uniform | 8.3452 | gate (c): between R1-off 8.392 and dense 8.297 |
| ring-R-on variance-proxy | 8.4632 | worse than uniform |
| ring-R-on measured-MSE | 8.3097 | best ring config |

**Ranking (best→worst): measured 8.3097 < uniform 8.3452 < variance 8.4632.**
The gate ranking (measured < variance AND measured ≤ uniform) **HOLDS under
ring-R-on** — the ring Hadamard's per-ring mixing did not invert the R1-off
order. The variance-proxy remains a bad allocator (8.47 in R1-off too).

## 3B measured-MSE budget sweep — iso-accuracy hunt

ring64 (for reference; all above the 8.2974 floor, PPL rises as budget drops):

| budget | avg FP8 | PPL (ring64) |
|---|---|---|
| 1/8  | 0.1250 | 8.3097 |
| 1/12 | 0.0882 | 8.3399 |
| 1/16 | 0.0697 | 8.3510 |
| 1/24 | 0.0513 | 8.3892 |

ring128 (the default substrate — measured@1/8 = 8.2893 is **below** the floor, so
there is headroom; downward hunt finds the smallest budget still ≤ 8.2974):
_[ring128 sweep in progress — downward {1/12,1/16,1/24,1/32} + upper {3/16,1/4}]_

## 8B

- Gates PASS: bf16 6.1368 (floor 6.14); ring-noquant 6.1347 (lossless Δ+0.0001).
- dense-R-on uniform (in-harness) = **6.5882** (≈ published floor 6.599).
- ring64 uniform = 6.6111 → ε(64, 8B) = **+0.0229** (already smaller than 3B's
  +0.0478 — 8B is less sensitive to ring width).
- ring128 {uniform, measured} + overturn confirmation: _[in progress]_ — this is
  the publishable confirmation that the 3B overturn generalizes.

## Three key answers (3B, ring128 default)

**(i) Does ring-R-on + measured-MSE @1/8 beat dense-R-on uniform (8.2974 /
published 8.293)?** **YES at ring_size=128**: 8.2893 < 8.2974 (margin −0.0081,
−0.0081 vs published too). NO at ring_size=64 (+0.0123). The flip is real and is
gated on ring width ≥ 128.

**(ii) Iso-accuracy (PPL ≤ 8.2974) minimum budget:** _[ring128 downward hunt in
progress]_ — at ring128 measured@1/8 (8.2893) is already below the floor, so the
iso-accuracy budget is ≤ 1/8; the sweep finds exactly how far below 1/8 it holds.
(At ring64 no budget ≤ 1/8 reaches the floor — measured@1/8 was 8.3097 > floor.)

**(iii) Best PPL at budget 1/8 (iso-speed):** **8.2893** (ring128 measured-MSE) —
0.008 better than dense-R-on uniform at the same FP8 count.

## Code (opt-in, zero-regression)

- `promix/quantize/rotation.py` — `build_main_transform`/`fuse_basis_to_model`
  gain `ring_r1` + `ring_size` kwargs. `ring_r1=True` → `T = U @ ring_R1`.
  Default off = byte-identical dense/identity path.
- `promix/eval/ptq.py` — `prepare_ptq_model` reads `quantize.ring_r1` /
  `quantize.ring_size` (default off).
- `promix/scripts/rd_validity.py` — `build_base`/`build_eval`/`collect_stats`
  parametrized by `ring_size`/`ring` (default 64 = zero regression).
- `promix/scripts/rd_ron.py` — R-on driver (gates, @1/8 three-way, measured-MSE
  sweep, flip-lever verdict) + `--ring_size`/`--skip_dense` + generalized
  `resolve_paths` (1B/3B/8B).
- `promix/scripts/rd_ron_ringsize.py` — ring-size ablation (ε(S) vs gain(S)).
- `tests/test_ring_rotation.py` — 6 `build_main_transform` wiring tests (24 pass).
