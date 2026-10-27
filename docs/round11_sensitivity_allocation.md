# Round-11 (ALGORITHM axis) — MEASURED per-layer error-sensitivity allocation

Goal (project lead's idea, done right this time): allocate the high-precision
(MXFP8) budget by **MEASURED** per-layer error sensitivity under a global bit
budget, instead of the fixed 1/8-per-layer uniform split. The two prior attempts
used HEURISTIC signals (round-1 `edriven`: weight-error / act-variance;
2026-10-09 adaptive: activation-variance vary-cut-length) and older baselines.
This round measures the **real** ΔPPL of each layer's high segment on the
**current shipped** recipe and solves the allocation from that.

- Model: Llama-3.2-3B (28 decoder layers), wikitext-2 PPL, SM120 RTX Pro 6000.
- Shipped recipe = act-MSE on + two-level weight scale (variant B: shared
  per-tensor FP32 global × E4M3/E8M0 micro, nearest + micro-MSE) + GPTQ frozen
  scales + PCA(U)+R1 rotation + per-head o_proj + online-Hadamard(down), r=1/8.
- Harness: `promix/scripts/round11_sensitivity.py` (reuses the round-1 ablation
  build + GPTQ `_edriven_khigh` patch to set per-(layer) K_high on all 7 Linears
  for BOTH the weight GPTQ split and the activation quantizer). Raw JSON on the
  box: `/home/luofan/round11_results/`.

## 0. Anchor reproduction (gate — PASS)

| run | PPL | bits/w |
|---|---|---|
| shipped uniform r=1/8 (round-6 fake anchor) | 8.293 | 4.969 |
| `ablation_acc_per_bit rsweep --r 0.125 --weight_2level … --w2l_high` | **8.29254** | 4.9688 |
| `round11_sensitivity --mode anchor --base_r 0.125` (my harness) | **8.29254** | 4.9688 |

My per-layer harness, with all 28 layers at r=1/8, reproduces the shipped anchor
**bit-identically** (8.292540550231934 both). The uniform r=1/16 baseline also
reproduces exactly: `rsweep --r 0.0625` = **8.3613** (scoreboard 8.361). Gate PASS.

All PPLs here are **fake-quant** (the round-1 ablation build, no kernel build).
They are directly comparable to the scoreboard's fake-quant column; the shipped
real-vs-twin gate (round-6 |Δ|=0.0095 @ r=1/8, round-19/24 bit-exact for
arbitrary K_low/K_high incl. K_high=0 pure-NVFP4) carries these to the real
kernel unchanged — so the winning prune config is fake-quant-measured but
real-kernel-deployable (see §4).

## 1. The 28-layer MEASURED sensitivity vector

ΔPPL_i = PPL(layer i high segment REMOVED → r=0 for that layer only, all others
r=1/8) − anchor(8.29254). Positive = that layer's 0.47-bit high segment is
load-bearing; negative = it HURTS. Each run is a full GPTQ+PPL rebuild (~12 min;
28 runs across 7 GPUs ≈ 4 waves). All at 4.952 bits/w (one layer's high dropped).

| layer | ΔPPL | layer | ΔPPL | layer | ΔPPL | layer | ΔPPL |
|---|---|---|---|---|---|---|---|
| 0 | +0.0188 | 7 | +0.0199 | 14 | +0.0358 | 21 | +0.0461 |
| 1 | **−0.0407** | 8 | +0.0008 | 15 | +0.0364 | 22 | +0.0547 |
| 2 | **+0.1555** | 9 | +0.0119 | 16 | +0.0448 | 23 | +0.0199 |
| 3 | +0.0927 | 10 | +0.0093 | 17 | +0.0409 | 24 | +0.0305 |
| 4 | +0.0685 | 11 | +0.0108 | 18 | +0.0385 | 25 | +0.0314 |
| 5 | +0.0253 | 12 | +0.0207 | 19 | +0.0367 | 26 | +0.0371 |
| 6 | +0.0377 | 13 | +0.0242 | 20 | +0.0522 | 27 | +0.0357 |

Sorted (ascending): **L1 −0.041**, L8 +0.001, L10 +0.009, L11 +0.011, L9 +0.012,
L0 +0.019, L23 +0.020, L7 +0.020, L12 +0.021, L13 +0.024, L5 +0.025, L24 +0.031,
L25 +0.031, L27 +0.036, L14 +0.036, L15 +0.036, L19 +0.037, L26 +0.037,
L6 +0.038, L18 +0.039, L17 +0.041, L16 +0.045, L21 +0.046, L20 +0.052,
L22 +0.055, L4 +0.068, L3 +0.093, **L2 +0.156**.

- **Spread = 0.196** (max L2 +0.156, min L1 −0.041); ~190× between the top and
  the near-zero layers. Mean +0.0356, median +0.031. The signal is real and
  strongly heterogeneous — FAR more structure than the heuristic signals found.
- **Early layers dominate**: L2/L3/L4 (+0.156/+0.093/+0.068) carry most of the
  high-segment value. **L1 is NEGATIVE** — its high segment (MXFP8 block-32 on
  the rotated top-variance channels) *increases* error vs quantizing those
  channels as NVFP4 (finer block-16 scale + GPTQ). L8 is ~0 (dead weight).
- **Subadditivity**: Σ ΔPPL_i = **+0.996**, but removing ALL high at once
  (r=0, 4.50 b) costs only **+0.566** (8.293→8.859). The high segments are
  ~1.76× redundant — the marginal value of high precision is concave; the FIRST
  high channels per layer are the valuable ones.

(Per-projection 28×7 sensitivity was NOT run: at ~12 min/run it is 196 runs, and
the layer-level verdict below — allocation ≤ uniform — made it low-value. Round-1
already measured intra-Linear channel-SET selection and it LOST to the variance
tail.)

## 2. Allocation solved from the measured vector + equal-bits validation

**Important validity constraint discovered this round.** The shipped rotation
`R1 = block_diag(R1_1 main, R1_2 high)` has its subspace boundary fixed at
high_fraction=1/8. Per-layer r ∈ {0, 1/8} is rotation-consistent (drop the whole
rotated high subspace, or keep all of it). Intermediate r (1/16, 1/4) takes a
*sub-tail of the R1_2-rotated block*, which is no longer monotonic by variance →
invalid with the 1/8 rotation (a fixed-1/8-rotation "uniform 1/16" gives 8.77,
not 8.36). So the honest, deployable adaptive scheme on the shipped recipe is
**per-layer r ∈ {0, 1/8}** (keep/drop the high segment per layer). Intermediate-r
reallocation is only valid with a matched rotation (tested separately, R1-off).

### 2a. Adaptive {0,1/8} — concentrate budget on the top-K measured layers

Keep the high segment on the K most-sensitive layers (r=1/8), drop it (r=0) on
the rest. bits/w = 4.5 + (K/28)·0.469.

| scheme | K | bits/w | PPL | uniform @ same bits |
|---|---|---|---|---|
| adaptive top-7 | 7 | 4.617 | 8.814 | — |
| adaptive top-14 | 14 | 4.734 | **8.531** | **uniform r=1/16 = 8.361** |
| adaptive top-21 | 21 | 4.852 | 8.365 | — |
| **anti**-adaptive bottom-14 (control) | 14 | 4.734 | **9.159** | (same 8.361) |
| uniform r=1/8 | 28 | 4.969 | 8.293 | 8.293 |
| uniform r=0 | 0 | 4.500 | 8.859 | 8.859 |

- **At the matched 4.734-bit budget, adaptive {0,1/8} (8.531) LOSES to uniform
  r=1/16 (8.361) by +0.170.** The whole adaptive {0,1/8} curve sits ABOVE the
  uniform (r=0 / r=1/16 / r=1/8) curve; top-21 (8.365 @ 4.852 b) is even
  Pareto-dominated by uniform r=1/16 (8.361 @ 4.734 b).
- The sensitivity SIGNAL is nonetheless correct and strong: top-14 (8.531) beats
  bottom-14 (9.159) by **0.63** at identical bits. The problem is not the signal;
  it is that "whole high-segment on/off per layer" is the wrong lever —
  concentrating budget wastes it (concave per-layer value) while fully starving
  half the layers hurts a lot (the first high channels are the valuable ones).
  Uniform fractional-high beats concentrated allocation.

### 2b. Equal-bits (4.969 b) reallocation with valid intermediate-r (R1-off)

To test the budget-neutral reallocation the lead asked for (per-layer r ∈
{0,1/16,1/8,1/4} at equal 4.969 b) one needs intermediate r, which requires a
matched rotation. R1-off (identity main rotation) keeps the channel order
monotonic by variance from U, making any r valid. Round-5 showed R1-off costs
only +0.03 at r=1/8, so it is a near-shipped recipe.

| scheme (R1-off) | bits/w | PPL |
|---|---|---|
| uniform r=1/8 | 4.969 | 8.4257 |
| reallocation: L2,L3,L4 → 1/4; L1,L8,L10 → 0 (equal bits) | 4.969 | **8.4155** |

Equal-bits reallocation gains **−0.010** — negligible, and the R1-off base is
already +0.133 worse than the shipped R1-on 8.293, so it does not come close to
the shipped number. (A rotation-contaminated R1-on version of the same
reallocation read 8.276, −0.017, but it is invalid per §2 and not counted.)

### 2c. The one genuine win — PRUNE the measured ≤0-sensitivity layers

Dropping the high segment only on layers whose measured sensitivity is ≤0 (L1) or
~0 (L8, …) is a strict Pareto improvement over uniform r=1/8 (valid {0,1/8}):

| scheme | dropped layers | bits/w | PPL | vs uniform 1/8 (8.293 @ 4.969) |
|---|---|---|---|---|
| **drop-1** | L1 | 4.952 | **8.2518** | **−0.041 PPL AND −0.017 bits** |
| drop-2 | L1,L8 | 4.935 | 8.2608 | −0.032 PPL, −0.034 bits |
| drop-5 | L1,L8,L10,L11,L9 | 4.885 | 8.2930 | ≈0 PPL, **−0.084 bits** |

drop-1 is the lowest PPL of every config tested. This is a small, real,
deployable free lunch: the measurement *finds and removes* high segments that
cost bits and buy nothing (or hurt). It is NOT "spend the budget smarter"; it is
"don't spend it where it is provably useless."

## 3. Structural 3-segment variant (4|8|4) — not pursued

The per-LAYER sensitivity vector says nothing about intra-Linear channel
placement, and round-1's A1 already measured intra-Linear channel-SET selection
(weight-energy / combined) and found it WORSE than the plain variance tail. A
3-segment (NVFP4 | MXFP8-middle | NVFP4) layout with the middle at
"highest-error channels" is therefore unmotivated by this data and would add
kernel/packer complexity (3 segments + non-tail high). Not run.

## 4. Deployability

- **Prune scheme (per-layer r ∈ {0,1/8})** — the only win — is the MOST
  deployable: the fused kernel already takes per-Linear K_low/K_high (round-19/24
  bit-exact with arbitrary lengths; per-layer cos 0.999999 in multisize);
  `split_k` accepts K_high=0 and `pack_weight_mixed` emits a pure-NVFP4 Linear.
  **Kernel UNCHANGED.** Needed: (1) a per-layer K_high table in the config
  (the `adaptive_high_fraction` infra in `configure_quantizers` already carries a
  per-layer khigh map on the main path — extend to all 7 projections); (2) GPTQ's
  per-layer `high_bits_length` (gptq_fwrd uses a single global high_fraction; the
  `_edriven_khigh` tag used here is the mechanism) ; (3) confirm the dispatcher
  routes a K_high=0 Linear to the pure-NVFP4 GEMM path. No Step-0/1 rerun, re-pack
  only.
- **Intermediate-r per-layer (1/16, 1/4)** is NOT deployable on the shipped
  recipe without per-fraction rotations (the block-diag R1 boundary is fixed per
  high_fraction) — a Step-0/1 change. Given the −0.010 payoff, not worth it.

## 5. Verdict

**Does measured-sensitivity allocation beat uniform? Essentially NO — third
independent confirmation — with one small Pareto nugget.**

- The measured per-layer sensitivity is real and highly heterogeneous (spread
  **0.196**, L2 +0.156 … L1 −0.041; signal validated by the anti-adaptive control
  losing by 0.63). This is a much stronger, cleaner signal than the heuristics
  round-1 / 2026-10-09 used.
- **Reallocating the budget by it does not beat uniform at equal bits.**
  Concentrating on sensitive layers at {0,1/8} LOSES to uniform r=1/16 by
  **+0.170** at 4.734 b (8.531 vs 8.361). Fine-grained equal-4.969-bit
  reallocation (valid only R1-off) gains a negligible **−0.010**. The mechanism is
  now understood: per-layer high-precision value is concave + the segments are
  1.76× redundant, so uniform fractional-high dominates concentrated allocation.
  This matches and extends round-1 ("chasing smart allocation is a dead end") and
  the 2026-10-09 adaptive negative result, but now with MEASURED sensitivity and
  the shipped baseline — so the conclusion is no longer attributable to a weak
  signal or a stale baseline.
- **The only real win is PRUNING**, not allocating: dropping the high segment on
  the measured ≤0-sensitivity layer(s) is a strict Pareto improvement over uniform
  r=1/8 — **drop-L1 = 8.2518 @ 4.952 b (−0.041 PPL and −0.017 bits)**, or
  drop-5 = 8.293 @ 4.885 b (same PPL, −0.084 bits). Small, free, deployable with
  the kernel unchanged. Recommendation: ship the prune of L1 (and audit per model;
  it is model-specific — an early layer whose rotated high channels don't need
  8-bit), and do NOT invest further in per-layer budget *reallocation*; the
  accuracy-efficiency lever remains the base quantizer quality (round-5/6), not
  the partition.

## 6. Reproduction

- Harness: `promix/scripts/round11_sensitivity.py` (`--mode anchor|leaveone|
  addone|plan`, `--plan_json`, `--r1_off`); reuses `ablation_acc_per_bit.py`
  helpers. Anchor gate: `--mode anchor --base_r 0.125` = 8.29254 (== shipped).
- Raw JSON: `/home/luofan/round11_results/3b_{leaveone,plan,anchor}_*.json`,
  `3b_rsweep_unif16proper.json`; plans `plan*.json`; launcher `r11_launch.sh`.
- Each run ~12 min (GPTQ rebuild + sequential wikitext-2 eval), 7× RTX Pro 6000.
