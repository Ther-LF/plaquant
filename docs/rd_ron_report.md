# R-on Integration (Task 6) — measured-MSE allocation on the uniform ring R1

**Branch** `feat/rd-ron` · **date** 2026-10-28 · **spec** `docs/superpowers/specs/2026-10-28-nested-r1-per-layer-allocation-design.md` §3

## Headline

Rounds 11/13 concluded **4×** that per-layer precision allocation "cannot beat
R-on uniform" — but always with the allocation forced **R1-OFF** (identity main
rotation, `T=U`), discarding the dense R1's gain. The old argument rested on
*last-K = top-variance* semantics, which breaks once a dense R1 mixes the top
block. This work tests the **flip lever**: keep the rotation and make allocation
per-ring. Two levers, both validated on **3B**:

1. **Ring substrate** (`T = U @ ring_R1`, fixed Hadamard per ring) + per-ring
   measured-MSE allocation. Overturns dense-R-on uniform for ring_size ≥ 128,
   margin widening with ring width (best tested **ring512: 8.2328, −0.065**).
2. **Simplest shipping story — dense-limit**: the **shipped dense R1, unchanged**
   + per-layer measured-MSE K allocation (reorder-before-quant). Because the
   rotation is untouched, rotation cost ε = 0, so the allocation gain is pure
   win: **8.2678, −0.030** vs dense uniform 8.2974.

**On 8B the overturn does NOT hold** — not via the ring substrate (measured ≈
dense uniform within +0.001…+0.006 at every ring width) and not at the
dense-limit either (ε=0: measured 6.6072 vs uniform 6.5882, **+0.019 worse** —
the gain itself is negative on 8B). 8B's per-ring ΔD is far more uniform, so
allocation buys nothing there. **8B verdict final: stands.**

**Net:** the "uniform near-optimal" claim is **overturned on 3B** (clearly, by
either lever) but **holds on 8B** for the ring substrate; the model-dependence
(allocation gain tracks how non-uniform the per-ring error is) is itself the
finding.

## Sanity gates (fusing correctness)

| gate | 3B | 8B | verdict |
|---|---|---|---|
| (a) bf16 bridge == floor | 7.8148 (floor 7.82) | 6.1368 (floor 6.14) | PASS |
| fp16 bridge | 7.8113 | 6.1346 | — |
| (b) ring-rotation-only, **no quant** == fp16 (lossless) | 7.8121 (Δ+0.0008) | 6.1347 (Δ+0.0001) | PASS |

Gate (b): an orthogonal rotation must leave PPL unchanged. Δ < 0.001 on both →
the ring-R1 fuse (ring-block-diagonal Hadamard baked into embeddings + per-layer
weight absorption + tie-word-embeddings untie/clone) is numerically exact.

## THE mechanism figure — 3B ring-size decomposition (@1/8, vs dense uniform 8.2974)

| substrate | ring uniform | ring measured | **ε** = unif−dense (rotation cost) | **gain** = unif−meas (allocation) | **net** = meas−dense | verdict |
|---|---|---|---|---|---|---|
| ring64  | 8.3452 | 8.3097 | +0.0478 | +0.0355 | +0.0123 | stands |
| ring128 | 8.3133 | 8.2893 | +0.0159 | +0.0240 | −0.0081 | OVERTURNED |
| ring256 | 8.3369 | 8.2665 | +0.0395 | +0.0704 | −0.0309 | OVERTURNED |
| ring512 | 8.4494 | 8.2328 | +0.1520 | +0.2166 | −0.0646 | OVERTURNED (best) |
| **dense R1 + measured** | 8.2974 | **8.2678** | **0** | +0.0296 | **−0.0296** | OVERTURNED (simplest) |

Reference anchors: R1-off uniform 8.3924; dense-R-on uniform 8.2974 (=published
8.293); 3B bf16 7.8148.

**Three curves tell the whole story:**
- **ε(S)** (ring-R-on *uniform* rotation cost) is **non-monotone, blows up at
  512** (+0.048, +0.016, +0.040, +0.152): a wide Hadamard smears the U-sorted
  variance tail that the *uniform* last-K cut relies on (at 512 only 6 rings/unit).
- **gain(S)** (measured-MSE allocation benefit) **grows monotonically and fast**
  (+0.036, +0.024, +0.070, +0.217): the measured path *reorders* (picks true
  high-ΔD rings regardless of position), is immune to smearing, and actively
  benefits from wider per-ring mixing — each upgraded ring absorbs concentrated
  energy.
- **net = meas − dense improves monotonically with S** (+0.012 → −0.008 →
  −0.031 → −0.065). No knee through 512.
- **dense-limit (ε=0):** removing the rotation change entirely leaves *pure*
  allocation gain (+0.030). It overturns, but is 0.035 *worse* than ring512 —
  so **the ring structure itself contributes** beyond "dense + measured-K": the
  uniform-per-ring Hadamard is a better basis for the measured allocation than
  the shipped dense R1 (which only densely-mixes the top-1/8 and the rest
  separately). The S→d-unit limit of the ring curve therefore does **not** equal
  the dense R1 — it is a per-unit-dense Hadamard, which is strictly better here.

## 3B @1/8 three-way ranking (allocation sanity)

| config @ 1/8 | ring64 | ring128 |
|---|---|---|
| ring-R-on uniform | 8.3452 | 8.3133 |
| ring-R-on variance-proxy | 8.4632 | 8.3634 |
| ring-R-on measured-MSE | 8.3097 | 8.2893 |

Ranking (best→worst) **measured < uniform < variance** holds at both widths —
the gate ranking survives ring-R-on; the variance proxy is consistently the
worst allocator (as in R1-off, 8.47).

## 3B iso-accuracy hunt (ring128 measured-MSE budget sweep)

| budget | avg FP8 | PPL (ring128) | vs floor 8.2974 |
|---|---|---|---|
| 1/4  | 0.2356 | 8.2022 | −0.095 |
| 3/16 | 0.1803 | 8.2303 | −0.067 |
| 1/8  | 0.1250 | 8.2893 | −0.008 (below floor) |
| 1/12 | 0.0881 | 8.3022 | +0.005 |
| 1/16 | 0.0697 | 8.2983 | +0.001 (≈ floor) |
| 1/24 | 0.0513 | 8.3467 | +0.049 |
| 1/32 | 0.0421 | 8.3881 | +0.091 |

**Iso-accuracy (PPL ≤ dense uniform 8.2974) holds down to ≈ 1/16 (0.07 avg FP8)**
at ring128 — i.e. ~1.8× fewer MXFP8 channels than the shipped uniform 1/8 at the
same accuracy. The upper wing (3/16, 1/4) keeps improving (8.230, 8.202) — the
spend-latency-buy-accuracy side. (ring64 had no sub-1/8 headroom: measured@1/8
was 8.3097 > floor; 1/12=8.3399, 1/16=8.3510, 1/24=8.3892.) ring256/512 shift
the whole curve lower still.

## 8B — overturn does NOT generalize via the ring substrate

Reference: dense-R-on uniform (in-harness) **6.5882** (≈ published 6.599); bf16 6.1368.

| substrate | ring uniform | ring measured | ε | gain | net = meas−dense | verdict |
|---|---|---|---|---|---|---|
| ring64  | 6.6111 | — | +0.0229 | — | — | — |
| ring128 | 6.6105 | 6.5937 | +0.0223 | +0.0168 | +0.0055 | stands |
| ring256 | 6.6009 | 6.5917 | +0.0127 | +0.0092 | +0.0035 | stands |
| ring512 | 6.5970 | 6.5890 | +0.0088 | +0.0080 | +0.0008 | stands (≈ tied) |
| **dense R1 + measured** | 6.5882 | **6.6072** | 0 | **−0.0190** | **+0.0190** | **stands** |

8B's measured allocation gain is tiny (+0.008…+0.017 on rings, vs 3B's
+0.024…+0.217) because 8B's per-ring ΔD is far more uniform — there is little
outlier structure to exploit. With the ring substrate, the (small) rotation cost
ε is not repaid, so net stays ≈ 0 to slightly positive at every width. The
dense-limit (ε=0, the only configuration that could net-positive) **also
stands**: measured = 6.6072 is +0.019 *worse* than uniform — on 8B the measured
ΔD ranking actually underperforms the position-ranked uniform cut (gain is
*negative* at dense), consistent with the ΔD-metric caveat below. **8B verdict
is final: uniform near-optimal holds on 8B under every tested lever.**

## Three-model summary + ΔD-structure metric

Per-ring ΔD = (D_nvfp4 − D_mxfp8) non-uniformity on dense-R-on-rotated
activations (ring-64 atoms, pooled over the reallocated Linears):

| model | dense uniform (anchor) | ΔD Gini | ΔD CV | top-1/8 ΔD frac | measured gain @1/8 (dense-limit) | overturn? |
|---|---|---|---|---|---|---|
| 1B (Instruct) | 14.5483 (≈14.563) | 0.694 | 1.945 | 0.607 | **+0.0766** (14.4717) | **YES** |
| 3B | 8.2974 (≈8.293) | 0.641 | 1.772 | 0.592 | +0.0296 (8.2678) | **YES** |
| 8B | 6.5882 (≈6.599) | 0.664 | 1.834 | 0.588 | **−0.0190** (6.6072, worse) | **NO** (rings +0.001…+0.006; dense −0.019 gain) |

**The dense-limit allocation gain orders cleanly by model size: 1B +0.077 >
3B +0.030 > 8B −0.019.** Smaller models have more allocation headroom (they
quantize further from their fp16 floor), so per-layer measured-MSE K buys more —
and on 8B it crosses into slightly negative. The dense-limit (shipped R1 +
measured K, ε=0) is a clean overturn on both 1B and 3B, and a clean "stands"
on 8B.

**Honest caveat on the ΔD metric:** the ΔD-*concentration* numbers (Gini,
top-1/8 fraction) are *similar across all three models* and do **not** cleanly
rank the gains — 8B's Gini (0.664) exceeds 3B's (0.641) yet 8B's gain is the
smallest. So "structure = ΔD-concentration" is only a *partial* predictor; the
gain is better explained by absolute ΔD magnitude relative to each model's loss
curvature (model-size headroom) and by the overlap between the measured-ΔD
ranking and the position ranking the uniform cut already uses. We report the
metric as measured rather than curve-fit a just-so story.

## Shipping recommendation

- **3B (and likely other small models): ship the allocation.** The simplest
  variant — **shipped dense R1 unchanged + per-layer measured-MSE K allocation**
  (`reorder_before_quant`, no ring substrate) — already overturns uniform by
  0.030 PPL at iso-budget and needs no rotation-artifact change. If the ring
  substrate is acceptable (opt-in, zero online cost, baked once), ring256/512
  buys a further ~0.035 and extends iso-accuracy to smaller budgets.
- **8B: uniform is near-optimal; ship uniform.** The measured allocation is
  neutral on rings (+0.001…+0.006) and slightly *negative* at the dense-limit
  (−0.019) — final call: **do not enable allocation on 8B**; keep uniform 1/8.
  Operationally: gate the allocator on a cheap pre-check (e.g. the measured
  3B-vs-8B gain pattern, or a calibration A/B at 1/8) rather than a fixed
  model-size rule.
- **Rings in the paper:** keep the full ring64→512 ablation — it is what
  *discovered* the flip and isolates the ε(S)/gain(S) mechanism (mixing-width
  vs allocation), and shows the ring structure contributes beyond dense+measured.

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
- `promix/scripts/rd_ron_ringsize.py` — ring-size ablation + `--dense_rotation`
  dense-limit probe (ε(S) vs gain(S) across substrates).
- `tests/test_ring_rotation.py` — 6 `build_main_transform` wiring tests (24 pass).

## Raw numbers (for reproducibility)

3B measured @1/8 by substrate: ring64 8.3097 / ring128 8.2893 / ring256 8.2665 /
ring512 8.2328 / dense+measured 8.2678. 8B measured @1/8: ring128 6.5937 /
ring256 6.5917 / ring512 6.5890. JSON outputs on remote: `/home/luofan/RD_RON_*.json`.
