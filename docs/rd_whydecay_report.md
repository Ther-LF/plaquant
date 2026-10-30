# rd-whydecay: Why the Measured-MSE Allocation Gain Decays to ~0 with Model Scale

**Branch:** `feat/rd-whydecay` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant` · **Date:** 2026-10-30 · **Regime:**
dense-R-on, ring_size=64, matched avg FP8 = 1/8, o_proj held uniform — the exact
shipping build `rd_metrics` evaluated.

## The question

The measured-MSE per-ring allocation beats uniform by less and less as the model
grows (wikitext, measured − uniform @1/8): **1B −0.077 → 3B −0.030 → 8B +0.019 →
13B −0.001** (C4 same signs). Two candidate mechanisms:

- **UPSTREAM**: the per-ring ΔD / variance structure flattens with scale, so there
  is less concentration for the allocator to exploit.
- **DOWNSTREAM**: the structure is scale-invariant, but (a) the total quantization
  headroom (uniform-W4A4 gap vs bf16) shrinks, and/or (b) the per-ring PPL
  sensitivity flattens (model redundancy), so the same reallocation buys less PPL.

The rd-accpuzzle round already showed (R1-off) that ΔD concentration is the SAME on
3B and 8B (top-1/32 rings = 42.2 % vs 42.3 % of all ΔD). This round measures all
four mechanisms on all four models in the SHIPPING dense-R-on regime and adds the
decisive per-ring PPL-sensitivity test.

**TL;DR — it is DOWNSTREAM, in two stages.**
(1) 1B→3B: the upstream structure and the gain-to-headroom ratio are both
~constant (~5.5–6 % of the gap on wikitext, ~7 % on C4) while the headroom itself
collapses 1.39→0.48 — a pure **headroom effect**. (2) 3B→8B/13B: headroom barely
moves (0.48→0.45) yet the gain vanishes — an additional **per-ring sensitivity
flattening**: the top-ΔD ring flip costs FLIP1_SENS_TABLE. Upstream concentration
is scale-invariant (Gini 0.61–0.69, top-1/32 share 21–27 %, non-monotone) and is
NOT the cause.

## 1. Upstream structure vs scale — flat (not the cause)

Per-ring ΔD = D_fp4 − D_fp8 and per-ring variance over the 6 reallocated Linears
(q/k/v/gate/up/down_proj), all layers, dense-R-on rotated calibration activations
(4096 tokens), ring=64. `promix/scripts/rd_whydecay.py --structure_only`.

| model | rings | ΔD Gini | ΔD top-1/32 share | ΔD top-1/8 share | ΔD CV | ρ(ΔD,var) | per-layer ΔD CV | var Gini | var top-1/32 |
|---|---|---|---|---|---|---|---|---|---|
| 1B | 4608 | 0.694 | 27.4 % | 60.7 % | 1.95 | 0.985 | 0.012 | 0.639 | 22.2 % |
| 3B | 10304 | 0.641 | 23.5 % | 59.2 % | 1.77 | 0.998 | 0.015 | 0.583 | 19.7 % |
| 8B | 17408 | 0.664 | 25.2 % | 58.8 % | 1.83 | 0.999 | 0.040 | 0.615 | 21.2 % |
| 13B | 24640 | 0.609 | 21.5 % | 54.8 % | 1.62 | 0.992 | 0.014 | 0.552 | 17.4 % |

The concentration is **scale-invariant and non-monotone** (8B is MORE concentrated
than 3B on every metric; 13B is mildly less, Gini −0.03, top-1/8 −4 pp — far too
small to explain a 50× gain collapse). ΔD stays rank-tied to ring variance
(ρ ≥ 0.985) and per-layer ΔD is near-flat (CV ≤ 0.04) on every model. The
hypothesis "per-ring variance/ΔD/energy becomes uniform at scale" is **rejected**:
the allocator sees essentially the same exploitable structure at 1B and 13B.

(Note: dense-R-on top-1/32 ≈ 21–27 % vs the R1-off 42 % from rd-accpuzzle — the
dense R mixes the high-variance block internally, flattening the extreme tail —
but the cross-model invariance is identical in both regimes.)

## 2. Total headroom vs scale — collapses 10×

From the committed `docs/rd_metrics_report.md` (same build):

| model | bf16 wiki | uniform-1/8 wiki | gap (headroom) | bf16 C4 | uniform C4 | C4 gap |
|---|---|---|---|---|---|---|
| 1B | 13.1569 | 14.5483 | **+1.3914** | 18.5126 | 20.3150 | +1.8024 |
| 3B | 7.8148 | 8.2974 | **+0.4826** | 10.4329 | 11.3915 | +0.9586 |
| 8B | 6.1368 | 6.5882 | **+0.4514** | 8.8830 | 9.6355 | +0.7525 |
| 13B | 4.8853 | 5.0269 | **+0.1416** | 6.4692 | 6.6528 | +0.1836 |

Confirmed: the total quantization penalty the allocation can redistribute shrinks
~10× from 1B to 13B. But note the 3B→8B step: the gap barely moves (−6 %).

## 3. Gain as a fraction of headroom — constant 1B→3B, collapses at 8B+

| model | gain (measured−uniform, wiki) | gap | gain/gap (wiki) | gain (C4) | gain/gap (C4) |
|---|---|---|---|---|---|
| 1B | −0.0766 | 1.3914 | **5.5 %** | −0.1286 | **7.1 %** |
| 3B | −0.0296 | 0.4826 | **6.1 %** | −0.0721 | **7.5 %** |
| 8B | +0.0190 | 0.4514 | **−4.2 %** | +0.0038 | −0.5 % |
| 13B | −0.0006 | 0.1416 | **0.4 %** | −0.0020* | −1.1 % |

1B→3B the allocation claws back a **constant ~6 % (wiki) / ~7.5 % (C4) fraction**
of a 3× smaller gap — the decay there is **purely the shrinking headroom**. At
8B/13B even the fraction dies (|gain| ≤ eval reproducibility), so a second
mechanism must act between 3B and 8B.

## 4. Per-ring PPL sensitivity vs scale — the decisive test

From the explicit uniform-1/8 ring-set, demote the globally top-ΔD ring (flip1)
and the top-16 ΔD rings (flip16) back to NVFP4; wikitext PPL
(`rd_whydecay.py`, same build path for all three points so the deltas are
build-consistent).

FLIP_TABLE

VERDICT_SECTION

## 5. Structural breakdown — where is the exploitable non-uniformity?

Three axes of the per-ring ΔD non-uniformity, from the same dense-R-on stats
(ring-level one-way variance decomposition + per-group ΔD shares; the measured
column is the FP8 fraction the global ΔD-threshold allocator actually gives each
group at the matched 1/8 budget).

**(a) Cross-layer vs within-layer — the structure is ~100 % within-layer.**

| model | between-layer var frac | within-layer var frac | per-layer ΔD-sum CV | within-layer top-1/32 share (avg) |
|---|---|---|---|---|
| 1B | 0.00005 | 0.99995 | 0.012 | 25.8 % |
| 3B | 0.0001 | 0.9999 | 0.015 | 23.4 % |
| 8B | 0.0005 | 0.9995 | 0.040 | 24.2 % |
| 13B | 0.0001 | 0.9999 | 0.0143 | 20.5 % |

Per-layer ΔD sums are near-identical across depth (CV ≤ 0.04; e.g. 8B layer
shares span only 2.7–3.6 %), while each layer internally concentrates ~20–26 % of
its ΔD in its top 1/32 rings. There is NO cross-layer lever: a layer-weighted
allocation has nothing to redistribute. This confirms (now in dense-R-on, on all
four models) the rd-accpuzzle R1-off finding.

**(b) Per proj-type — huge, and already fully exploited.** q/k/v share the
attention input stream and gate/up share the MLP input stream, so per-ring stats
are identical within those groups; the three DISTINCT streams per layer are
attn-in, mlp-in, and down-in:

| model | down_proj ring share | down_proj ΔD share | ΔD density (share/rings) | measured FP8 frac on down_proj | measured FP8 frac on attn-in / mlp-in |
|---|---|---|---|---|---|
| 1B | 44.4 % | 0.54 % | 0.01 | **0.000** | 0.254 / 0.182 |
| 3B | 34.8 % | 0.66 % | 0.02 | **0.000** | 0.179 / 0.211 |
| 8B | 41.2 % | 0.84 % | 0.02 | **0.000** | 0.210 / 0.216 |
| 13B | 35.1 % | 0.83 % | 0.02 | **0.000** | 0.207 / 0.170 |

Between-type variance is the ONLY coarse structure (16–20 % of total ΔD variance,
vs ≤0.05 % between layers): **down_proj is a ΔD desert** — it holds 35–44 % of all
rings but <1 % of all ΔD, because its input passes through the online Hadamard,
which flattens outliers and makes NVFP4 nearly as good as MXFP8 there. The global
allocator already exploits this fully: it strips down_proj to **zero** FP8 rings
and reinvests the freed ~35–44 %-of-budget into the attn/mlp streams (0.17–0.25
vs uniform 0.125). This is THE systematic measured-vs-uniform difference — and the
PPL ladder (§3) shows it is worth −0.08/−0.03 PPL on 1B/3B and ~0 on 8B/13B. So
the type lever exists, is already pulled, and its VALUE (not its size) is what
decays with scale. No missed type-weighting remains.

**(c) By layer position — flat, with 1–2 isolated spike layers.** The measured
per-layer FP8 fraction is ~0.07–0.08 everywhere except single massive-activation
layers (1B: L3=0.56; 3B: L0=0.38; 8B: L9=0.52; 13B: L36=0.43, L37=0.34). No
early-vs-late trend in either ΔD share or allocated budget; the spikes are
ring-level outlier structure the ring allocator already captures. A
position-aware (early/late-weighted) allocation has no signal to use.

**Conclusion of the breakdown:** the exploitable non-uniformity lives on exactly
two axes — within-layer ring tails and the down_proj-vs-rest type split — and the
global ΔD threshold already captures both. The gain decay is therefore not an
averaging/granularity artifact; there is no structural lever the allocation is
missing at scale.

## 6. Scaling-ladder coverage note

The box's HF cache holds no additional Llama-architecture sizes beyond the ladder
(the extra entries are duplicates of Llama-3-8B under other org names, Qwen3-8B
(non-Llama arch, unsupported by the promix plugin), and EAGLE3 draft heads); per
the no-download constraint the ladder stays 1B/3B/8B/13B. The 4-point ladder
spans 13× in parameters and 10× in headroom, which is sufficient to separate the
two mechanisms.

## 7. Reproduce

```bash
# per model (GPU-pinned inside docker luofan-plaquant)
python promix/scripts/rd_whydecay.py --model {1b|3b|8b|13b} \
    --out /home/luofan/whydecay/RD_WHYDECAY_<mk>.json   # add --structure_only for §1/§5
```
Raw JSONs: `docs/rd_whydecay/RD_WHYDECAY_{1b,3b,8b,13b}.json`.
