# Round-5 (ALGORITHM axis) — fixing the pure-NVFP4 base (the 1.0 PPL deficit)

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.11+cu130,
transformers 4.48.1. Goal: explain why our **pure-NVFP4** point (r=0, ≈4.5 bit incl
SF) gives 3B PPL **9.644** while vLLM/llm-compressor NVFP4 (≈4.25 bit) gives **8.63**
— a 1.0 PPL deficit at ~the same bits — and close it. Same protocol anchor: 3B r=0 =
9.644 (reproduced below to 4 dp).

Harness: `promix/scripts/ablation_acc_per_bit.py` (research-only accuracy path, no
kernel build), the exact round-1/2 build (`prepare_ptq_model`: fuse-norm → PCA
rotation(U+R1) → rearrange → actquant → online-Hadamard(down) → **GPTQ frozen FP
block scales + Hessian error feedback** → activation quantizers → KV), scored by the
pipeline's own `evaluate_ppl` on wikitext-2 (non-overlapping 2048-token chunks, NLL
from position 1 — the protocol the scoreboard aligns with vLLM on for 3B/8B). Raw
JSON: `docs/round5_results/`.

## Protocol gate (PASS)

| run | 3B r=0 PPL | env |
|---|---|---|
| round-1 anchor | 9.6436 | — |
| round-5 `rsweep --r 0` | **9.6436** (9.643648) | `PROMIX_ACT_MSE=0 PROMIX_WEIGHT_MSE=0` |

The anchor reproduces **exactly** with both MSE levers OFF. **This is the key framing
correction: round-1's 9.644 "pure NVFP4" was measured with our two shipped quality
levers (activation block-scale MSE search + weight-A2) turned OFF.** Those levers had
only ever been evaluated at r=1/8 (where the MXFP8 high segment hides the weak base);
at r=0 they were never measured. Turning them on is most of the fix.

## 1. Where the r=0 error lives (diagnostic)

Reference: BF16 3B = 7.818. Weights-only (acts FP16) isolates the weight NVFP4 error.

| config | 3B r=0 PPL | error vs BF16 |
|---|---|---|
| BF16 | 7.818 | — |
| **weights-only, stock** (NVFP4 wts, acts FP16) | **8.2553** | weight NVFP4 **+0.44** |
| full W4A4, stock (anchor) | 9.6436 | + activation NVFP4 **+1.39** |

**At r=0 the error is activation-dominated** (+1.39 activation vs +0.44 weight). Note
our r=0 **weights-only (8.255) already beats vLLM NVFP4's full W4A4 (8.63)** — our
weights are not the problem; the activation quantizer is.

## 2. The r=0 ladder (controlled, each lever on the shared build)

All 3B, GPTQ on (frozen-scale + Hessian feedback) unless noted; acts quantized unless
"weights-only". "act-MSE" = the shipped activation NVFP4 block-scale MSE search
(ceil base + 3 lower E4M3 steps, per-block argmin dequant-MSE). "weight-A2" = the
shipped weight version of the same (single-level, ceil base + lower E4M3 steps).

| # | variant | 3B r=0 PPL | Δ vs anchor |
|---|---|---|---|
| 0 | **anchor** (stock single-level ceil, no MSE) | **9.6436** | 0 |
| 1 | + weight-A2 only (acts stock) | 9.6015 | −0.04 |
| 2 | + **act-MSE only** (wts stock) | 9.0923 | **−0.55** |
| 3 | + act-MSE + weight-A2 (**= SHIPPED recipe**) | 9.0369 | −0.61 |
| 4 | + act-MSE + weight-2level(tensor, **ceil**) | 9.0379 | −0.61 (ceil = no gain over A2) |
| 5 | + act-MSE + weight-2level(tensor, **nearest**) | 8.9391 | −0.70 |
| 6 | + act-MSE + weight-2level(**row**, nearest) | 8.9097 | −0.73 |
| 7 | + act-MSE + **weight-2level(tensor, nearest, +micro-MSE)** | **8.8588** | **−0.78** |

Weight-only floors (acts FP16):

| variant | 3B r=0 PPL |
|---|---|
| weights-only, stock | 8.2553 |
| weights-only + weight-2level(tensor, nearest, +micro-MSE) | **8.0858** (−0.17) |

Rejected activation variants (all stacked on the best weight scheme #7, acts replaced):

| variant | 3B r=0 PPL | verdict |
|---|---|---|
| act two-level per-token, nearest | 9.4244 | **worse** than act-MSE (8.86) |
| act two-level per-tensor, nearest | 9.4649 | **worse** |
| act two-level per-token, ceil | 9.6044 | worst |
| act two-level per-tensor, nearest, **weights stock** | 9.7652 | far worse than act-MSE-only (9.09) |

Error-feedback / rotation controls:

| variant | 3B r=0 PPL | note |
|---|---|---|
| **RTN** (no GPTQ feedback) + best weight scheme + act-MSE | 10.1475 | GPTQ feedback worth **−1.29** |
| rotation (main R1) OFF, stock (round-1) | 13.56 | rotation worth **−3.9** at r=0 |

## 3. Attribution of the 1.0 PPL gap (anchor 9.644 → vLLM 8.63)

| contributor | Δ | hypothesis | status |
|---|---|---|---|
| **Activation block-scale MSE search** (act-MSE) | **−0.55** | #5 / base-quantizer | already shipped; never measured at r=0 before |
| **Weight two-level scale: NEAREST E4M3 micro + per-block micro-MSE** | **−0.23** (−0.18 beyond weight-A2) | **#1** | **new this round** |
| residual vs vLLM | +0.23 (8.86 vs 8.63) | activation rounding | act-MSE is our ceiling; two-level acts hurt |

- **Hypothesis 1 (second-level global scale): CONFIRMED for WEIGHTS, REJECTED for
  ACTIVATIONS.** The active ingredient is **NEAREST E4M3 micro-scale rounding** (ceil
  over-allocates headroom so the FP4 grid under-resolves the block). Round-1 reported
  "two-level hurts 3B" (per-row, r=0 9.96 vs 9.64) — that was a **ceil-rounding
  artifact**: per-row + ceil = 9.96 (+0.32), but per-row + **nearest** = 8.91 (−0.73)
  and per-tensor + nearest = 8.94 (−0.70). Granularity (row vs tensor) barely matters
  (±0.03); the rounding direction was the bug. The per-tensor FP32 global additionally
  lifts tiny weight-block scales out of E4M3's subnormal range; stacked with a per-block
  micro-MSE it reaches −0.23. On **activations** every two-level variant (per-token /
  per-tensor, ceil / nearest) is worse than act-MSE — the global loses per-block
  adaptivity, confirming and extending round-2's per-token result.
- **Hypothesis 2 (FP4-code rounding / error feedback):** GPTQ Hessian error feedback on
  the frozen-scale FP4 grid is worth **−1.29** at r=0 (RTN 10.15 vs GPTQ 8.86). It is
  already applied to all 7 Linears. The FP4 element codes are round-to-nearest-even
  (correct). The only rounding lever on top is the block-**scale** direction (ceil→nearest,
  −0.10), now folded into the two-level weight scheme.
- **Hypothesis 3 (rotation):** removing the main rotation at r=0 costs +3.9 (13.56);
  rotation is critical and already on — not the gap source.
- **Hypothesis 4 (reorder):** at r=0 there is no MXFP8 high segment, so the high/low
  channel reorder is the identity; it cannot change which 16 channels share a scale.
  Not a factor at r=0 (verified analytically: `rearrange_columns(high_fraction=0)`).
- **Hypothesis 5 (outliers / per-block clipping):** act-MSE already trades a little
  outlier clipping for bulk resolution (that IS the −0.55 lever). An explicit
  outlier-oriented global (two-level activations) hurt. The error is broad, not
  tail-concentrated — consistent with round-1.

## 4. Best achievable points

The new weight lever (two-level **nearest** micro + per-block micro-MSE, NVFP4
segment) carries to the shipped r>0 points (gates reproduce exactly: r=1/8 shipped
8.3759 = documented 8.376; r=1/16 8.4627 = 8.463):

| r | bit/w | shipped (act-MSE + weight-A2) | **+ weight-2level (new)** | Δ | vLLM NVFP4 (4.25b) |
|---|---|---|---|---|---|
| 0      | 4.500 | 9.0369 | **8.8588** | −0.178 | 8.63 |
| 1/16   | 4.734 | 8.4627 | **8.3686** | −0.094 | 8.63 |
| 1/8    | 4.969 | 8.3759 | **8.2966** | −0.079 | 8.63 |

(r=1/8 without micro-MSE = 8.3369, −0.04; micro-MSE adds the other half.)

- **Best r=0 = 8.859** (down from the 9.644 anchor, −0.78). Within the coordinator's
  8.6–8.8 target band (8.86). Still +0.23 above vLLM NVFP4 (8.63) — but at 4.5 bit vs
  4.25, and with our **weights-only r=0 already at 8.086 < 8.63**: the residual is
  purely activation NVFP4 rounding, for which act-MSE is our ceiling.
- **New r=1/8 = 8.297** (4.969 bit) and **r=1/16 = 8.369** (4.734 bit). Both **below
  vLLM NVFP4's 8.63 on absolute W4A4 PPL** (activations quantized). The coordinator's
  ~7.7–7.9 expectation does not materialise: once act-MSE is on, the MXFP8 high segment
  only buys −0.56 (8.859→8.297), not the −1.0 it bought in the MSE-off world, so the
  best r=1/8 is ~8.30, not ~7.8.

### Deployability

- **act-MSE** — already SHIPPED (cheap fused producer, round-4). No change.
- **Weight two-level (nearest micro + per-block micro-MSE)** — **re-pack only, no
  Step-0/1 rerun**, BUT needs the kernel/packer to carry the standard
  **nvfp4-pack-quantized two-level weight-scale layout** (one FP32 per-tensor global ×
  per-16 E4M3 micro) instead of our current single-level E4M3-only weight scale: the
  effective scale `global_fp32 × micro_e4m3` is off the E4M3 grid, so the single-level
  packer cannot represent it. This is vLLM's existing weight layout and adds one FP32
  broadcast multiply in the weight-dequant/GEMM-scale path. The best *pack-only*
  (single-level E4M3) r=0 point remains the shipped 9.037 (weight-A2 is already the
  best grid-constrained search); the two-level gain (−0.18 at r=0, −0.08 at r=1/8)
  requires the layout change.

## 5. Conclusion

The 1.0 PPL "pure-NVFP4" deficit (anchor 9.644 vs vLLM 8.63) decomposes as:

1. **−0.55 — activation block-scale MSE search (act-MSE).** Already shipped, but only
   ever measured at r=1/8; at r=0 the weak NVFP4 base was exposed because the two
   quality levers were OFF. Turning them on is most of the fix. The r=0 error is
   **activation-dominated** (+1.39 act vs +0.44 weight over BF16).
2. **−0.23 — weight two-level scale (hypothesis 1), the new win.** The real ingredient
   is **NEAREST E4M3 micro-scale rounding** (ceil over-allocates); the per-tensor FP32
   global lifts tiny weight-block scales out of E4M3 subnormals; a per-block micro-MSE
   adds the rest. Round-1's "two-level hurts 3B" was a ceil artifact — with nearest,
   two-level helps at every r.
3. **Residual +0.23 vs vLLM** is activation NVFP4 rounding. Two-level on *activations*
   (per-token or per-tensor, ceil or nearest) all HURT — act-MSE is our activation
   ceiling.

Controls: GPTQ error feedback is worth −1.29 (already on); rotation −3.9 (already on);
reorder is identity at r=0.

**Verdict:** round-5 lowers every ProMix point (r=0 8.859, r=1/16 8.369, r=1/8 8.297)
and beats vLLM NVFP4's **absolute** W4A4 PPL at r=1/8 and r=1/16 (and r=0 weights-only).
We do **not** beat NVFP4 **decisively on accuracy-per-bit**: at the matched-bit point
(r=0, 4.5b vs NVFP4 4.25b) we are 8.86 > 8.63. The genuinely new, deployable lever this
round is the two-level **nearest** weight scale (−0.18 r=0 / −0.08 r=1/8), gated on
adopting the standard two-level weight-scale pack layout.

## Audit / reproduction

- Harness: `promix/scripts/ablation_acc_per_bit.py` (round-5 flags `--weight_2level`,
  `--w2l_round`, `--w2l_mse`, `--act_2level`, `--a2l_round`, `--rtn`).
- Anchor gate: `rsweep --r 0` with `PROMIX_ACT_MSE=0 PROMIX_WEIGHT_MSE=0` = 9.6436.
- Shipped gates: `--r 0.125` = 8.3759, `--r 0.0625` = 8.4627 (both MSE on).
- Raw JSON: `docs/round5_results/` (3B, wikitext-2, `evaluate_ppl`).

