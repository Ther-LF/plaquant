# Adaptive High-Fraction — Results

**Date:** 2026-10-10
**Spec:** `docs/superpowers/specs/2026-10-09-adaptive-high-fraction-design.md`
**Plan:** `docs/superpowers/plans/2026-10-09-adaptive-high-fraction.md`
**Model:** Llama-3.2-1B-Instruct, wikitext-2 PPL, FP (NVFP4 main + MXFP8 high),
per-head o_proj, identity-R1 on the main path, nsamples=512, Pro 6000 (SM120).

## Pareto comparison (main-path average bit-width vs PPL)

| scheme | main-path avg bits | PPL |
|---|---|---|
| fixed 1/8 | 4.50 | 14.92 |
| fixed 1/4 | 5.00 | 14.43 |
| fixed 1/2 | 6.00 | 14.03 |
| adaptive τ=0.50 | 4.29 | 15.59 |
| adaptive τ=0.60 | 4.53 | 15.32 |
| adaptive τ=0.70 | 4.96 | 14.95 |
| adaptive τ=0.80 | 5.61 | 14.87 |
| adaptive τ=0.85 | 6.06 | 14.81 |

Matched-bit-width comparisons (adaptive vs fixed):
- ~4.5 bits: 15.32 (τ0.60) vs 14.92 (1/8) → adaptive **+0.40 worse**
- ~5.0 bits: 14.95 (τ0.70) vs 14.43 (1/4) → adaptive **+0.52 worse**
- ~6.0 bits: 14.81 (τ0.85) vs 14.03 (1/2) → adaptive **+0.78 worse**

## Verdict: negative result

The adaptive per-layer allocation is **dominated by uniform allocation** at
every matched bit-width, and the gap widens with bit-width. This is the
valid negative outcome anticipated in spec §7.

Caveat (makes the negative result conservative, not weaker): the fixed
baseline runs varied high_fraction globally (o_proj/down_proj also scaled
with f), so at a given main-path avg bit-width the fixed runs actually spent
*more* total bits than the main-path-only accounting credits them. So the
fixed baseline is if anything *stronger* than tabulated — adaptive losing to
it is robust, not an artifact. (The 1/8 fixed point is the cleanest: there
o_proj/down are at 1/8, matching adaptive's o_proj/down, and adaptive still
loses by 0.40.)

## Diagnosis

Two mechanisms, the second more likely dominant:

1. **Per-layer variance-in-shared-basis is a poor proxy for quantization
   sensitivity.** Allocating 8-bit budget by this variance did worse than
   splitting it evenly — the "high-variance layers need more 8-bit"
   assumption does not hold in the shared-U coordinate system.

2. **"Vary cut length only, not channel set" is too weak (likely main
   cause).** High-precision channels were pinned to the *tail of the shared
   average ordering*; each layer could only lengthen/shorten that tail. But a
   layer's true outlier channels are not necessarily in the average-ordering
   tail. A layer squeezed to K_high=128 gets a 128-tail that may not cover
   its own outliers — worse than a uniform 256-tail. At τ=0.60 the per-layer
   K_high spanned [128, 512]; the starved layers hurt more than the boosted
   layers helped.

What is **not** refuted: "adaptive allocation" as an idea. What **is**
refuted: "shared basis + vary-length-only," on 1B.

## Next increments (from spec §7, now data-justified)

1. **Per-layer channel-set selection** — each layer picks the high-precision
   channels by its *own* variance ordering (needs per-layer reorder), instead
   of the shared average tail. This directly addresses mechanism (2).
2. **Sensitivity-based proxy** — allocate 8-bit budget by GPTQ/Hessian
   reconstruction error rather than activation variance. Addresses (1).
3. Larger models (3B/8B) where layer heterogeneity and redundancy differ —
   the 1B conclusion may not transfer.

## Engineering status

All code (Tasks 1–4) passed task review with zero regressions; the
non-adaptive path is byte-for-byte unchanged. The scheme is correct and
runnable — it is the *hypothesis* that the data rejected, which is the
experiment working as intended.
