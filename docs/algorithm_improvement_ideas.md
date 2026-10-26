# Algorithm Improvement Ideas — toward Pareto-dominating NVFP4/AWQ/GPTQ

> 2026-10-26. Read-only design doc. Goal (user's framing): **accuracy no worse
> than related work, performance better, and a LARGER accuracy/performance
> trade-off space** so our curve Pareto-dominates NVFP4/AWQ/GPTQ at equal bits.
> This doc proposes concrete, testable *algorithm* ideas to (a) raise
> accuracy-per-bit at ≤4.5 bits and (b) add per-layer / per-component / per-token
> knobs that map to real hardware cost. It does **not** touch code — `promix/`
> and `kernels/` are being edited by another agent.
>
> Companions: `docs/scoreboard.md` (the judging table), `docs/multisize_report.md`
> (1B/3B/8B real-kernel numbers), `docs/reference_deploy_code_reading.md`
> (QuaRot/SpinQuant/Atom/QServe mechanisms), `docs/superpowers/specs/2026-10-09-adaptive-high-fraction-results.md`
> (the per-layer-adaptive experiment that **failed**, and *why*).

---

## 0. Where we stand and where the bits go

Current ProMix W4A4 point (same-protocol 3B/8B rows of the scoreboard):

| | bits/weight | 3B PPL | 8B PPL |
|---|---|---|---|
| vLLM NVFP4 W4A4 | ≈4.25 | 8.63 | 6.87 |
| **ProMix (r=1/8)** | **4.969** | 8.615 | 6.823 |
| GPTQ W4A16 | 4.0 | 8.20 | 6.52 |

We spend **+0.72 bit/weight** over NVFP4 and only **tie** it. Decompose the 4.969:

```
4.969  = 7/8·(4 + 0.5)        # NVFP4 block-16: 4-bit elem + E4M3 scale 8b/16 = 0.5
       + 1/8·(8 + 0.25)       # MXFP8 block-32: 8-bit elem + UE8M0 scale 8b/32 = 0.25
```

So the premium over a pure-NVFP4 4.5-bit point is the **+0.5 bit** we pay to put
1/8 of channels in MXFP8 (the "high" segment). The whole Pareto argument reduces
to one question: **can we buy the same high-variance protection for fewer bits,
and can we spend the bits we do keep more accurately?**

Two independent levers, both pursued below:

- **Accuracy-per-bit** (make each bit worth more): MSE-optimal block scales,
  block-local rotation, error-feedback rounding on the MX grid, sensitivity-driven
  channel selection.
- **Enlarge the trade-off space** (expose dials that map to real cost): per-layer
  `K_high`, decoupled weight/activation precision, high-segment bit-width (8→6→4),
  outlier-channel count, KV-cache bits. Each dial is a Pareto *axis*, not a single
  point — this is the part related work does not have (NVFP4 is one fixed point,
  AWQ/GPTQ are W4A16 single points).

### What has already been tried (do not re-propose naively)

- **Fixed 1/8 split, variance-ordered channels, absmax frozen scales, GPTQ error
  feedback on both segments** — current shipping recipe (4.969 bits). Confirmed by
  code read: high channels = the activation-covariance eigenvalue *tail*
  (`basis.py:67` sorts ascending, `rotation.py:136-194` reorders to `[low|main|high]`),
  uniform `high_fraction` across all layers (`gptq.py:366`); the MXFP8 high segment
  *does* get genuine GPTQ error feedback (`gptq.py:166`, shared Hessian), only its
  *scale* is frozen absmax, not the elements. So "add GPTQ to the high segment" is
  **not** a gap — the gap is scale optimality and channel/bit *allocation*.
- **Per-layer adaptive `K_high` by variance-in-shared-basis, vary-cut-length only**
  — **FAILED**, dominated by uniform at every matched bit-width (−0.40 PPL at 4.5
  bits). Diagnosed causes (results doc §Diagnosis): (1) variance in the *shared* U
  basis is a poor sensitivity proxy; (2) pinning high channels to the shared
  *average* tail and only varying its length starves layers whose true outliers
  are elsewhere. **The fix is per-layer own-ordering + a sensitivity proxy — not
  "turn adaptive off."**
- **RTN + frozen absmax scales** is current (`select_block_scales_{nvfp4,mxfp8}`,
  `quant_utils.py:617/647`, both pure absmax→ceil-to-grid); **MSE/clip-searched
  scales are not.** Note the `w_clip`/`mse` flag is a **no-op for FP formats** —
  `WeightQuantizer.configure` returns early on a string `bits` (`kv_quant.py:43-44`),
  so the MSE grid search (`kv_quant.py:105-127`) only ever touches INT quantizers.
  The hook exists; filling it in for the FP block grids is idea I3.
- Step-1 learned rotation (SGD on Stiefel) gives **no measurable 1B gain**.

---

## 1. Ranked idea table

Ranked by (accuracy-per-bit at ≤4.5 bits) × (how much it enlarges the trade-off
space) ÷ cost. "Rerun" = which pipeline stage; "pack" = packing/calibration only,
no kernel change, no Step 0/1.

| # | Idea | Mechanism | Expected gain | Cost / deployability | How to test cheaply | Novelty |
|---|---|---|---|---|---|---|
| **I1** | **Block-local rotation / permutation to equalize per-block dynamic range** | Within each MX block (16 for FP4 / 32 for FP8), apply a block-diagonal Hadamard (cheap butterfly) or a balanced channel permutation so every block has a tight, uniform absmax → the one shared block scale wastes far less of the FP4 grid | **0.2–0.5 PPL at fixed bits, or ≈0.3–0.5 fewer bits at fixed PPL** — this attacks the dominant block-scaled error source | Permutation: fold into weights (free); block-diag rotation: free on W, one 16-wide butterfly in the pack epilogue on A. Step-0 reorder + Step-2 repack; kernel only needs the butterfly in the pack (producer) epilogue | Offline: for a few layers, measure per-block `max/mean` ratio and MSE(FP4) before/after a size-16 Hadamard; then 1B/3B PPL, pack-only | **HIGH** — block-aware rotation *targeting the MX block grid* is new; QuaRot/SpinQuant rotate for per-tensor/per-token INT, not microscaling blocks |
| **I2** | **Per-layer + per-component channel selection by *sensitivity*, each layer using its OWN ordering** | Replace shared-U average tail with per-layer high-precision channel sets chosen by diag-Hessian × reconstruction-error; per component (q/k/v/o/gate/up/down) independently. Fixes BOTH diagnosed failure causes | Recovers the **+0.40 PPL** the naive adaptive lost and plausibly goes below uniform; enables honest per-layer `K_high` dial | Step-0 per-layer reorder index + Step-2 repack. Kernel already supports variable `K_low/K_high`. No new kernel | 1B first (where naive adaptive failed), then 3B/8B for heterogeneity. Metric: matched-bit PPL vs uniform 1/8 | Medium — sensitivity-driven *dual-format* per-layer split; MixLLM does per-output-channel bits by Fisher, not PCA-block |
| **I3** | **MSE-optimal (clip-searched) block scales for NVFP4 & MXFP8, inside GPTQ** | Current scales are absmax→ideal→frozen. Replace with a per-block grid clip search minimizing L^p error (Atom `gptq.py:146`, grid=100, maxshrink=0.8). Block-16 FP4 is *maximally* sensitive to this: one large elem inflates the scale and crushes the other 15 | **0.1–0.3 PPL at fixed bits**, free; larger at block-16 than the usual per-channel W4 because the block is tiny | **Pack-only.** Calibration/packer change; frozen-scale thread already exists (`accuracy_frozen_scale_report.md`). Kernel consumes whatever scale it is given | 1B: sweep clip grid {absmax, p=0.9..0.6}, PPL + per-block MSE. ~minutes | Medium — MSE scales are known; *MSE scales on dual-grid NVFP4+MXFP8 interacting with the FP8 second-level scale* is underexplored |
| **I4** | **Decoupled weight/activation precision + per-layer bit allocation by measured sensitivity** | Today W and A share one split. Decouple `K_high^W` and `K_high^A` and allocate per layer by sensitivity. Weights (GPTQ-compensated) often tolerate smaller high-segment than activations (dynamic, outlier-prone), or vice versa | **Biggest "reduce the 0.72 bit" lever**: frees bits where they are wasted; plausibly 4.97 → ~4.4 bits at equal PPL | Kernel: weight split and activation split are both along K; verify the GEMM can take different `K_high^W` vs `K_high^A` (may need a packing/layout tweak). Step-2 repack | 1B ablation holding total bits: grid over (`K_high^W`,`K_high^A`) ∈ {1/16,1/8,1/4} | High — explicit W/A precision decoupling on a block-scaled dual-format kernel |
| **I5** | **Outlier-channel keep (tiny set at MXFP8) to shrink the variance segment** | In addition to / instead of the 1/8 variance segment, pull a *handful* (16–32) of true max\|activation\| outlier channels to MXFP8; the remaining NVFP4 blocks then use a much smaller scale. Atom keeps 128 INT8 channels for exactly this reason | Lets `r` drop 1/8 → 1/16 at equal PPL ≈ **−0.23 bits**; stacks with I1/I3 | Reorder only; our dual-format kernel already does mixed precision. Step-0 outlier list + Step-2 repack | 1B: r=1/8 variance-split vs r=1/16 + 16 outlier channels, matched-bit PPL | Medium — combines PCA-variance split with magnitude-outlier keep in one kernel |
| **I6** | **Scale-overhead reduction (two-level global + coarser per-block scale)** | The 4.969 carries ≈0.5+ bit of scale metadata. Adopt NVFP4's two-level scheme (per-tensor FP32 global × per-16 E4M3) and/or share/compress the MXFP8 UE8M0 scale so the per-block scale can be lower-precision without PPL loss | **0.1–0.3 bits for ≈free** — pure accounting win straight off the 0.72 gap | Pack-only + minor kernel SFB layout (two-level multiply in epilogue, like QServe progressive scales) | 1B/3B PPL with global+E4M3 two-level vs current; measure bits and PPL | Low-med — standard for NVFP4; new *for our dual-grid* |
| **I7** | **Lower-bit high segment (MXFP6/INT6) as a dial** | The +0.5-bit premium assumes the high segment needs 8 bits. Make high-segment width a knob {8,6,4}. 1/8 at 6-bit = +0.25 bit (half the premium) if high channels don't need full FP8 | **≈0.25 bits** at ≤0.05 PPL if 6-bit suffices; adds a clean Pareto axis | New 6-bit kernel path (cost) **or** emulate in fake-quant first to decide before building | Fake-quant 1B/3B: high={8,6,4}, PPL vs bits curve. Build kernel only if 6-bit holds | Med — "mixed FP4/FP6" block-scaled dual-format |
| **I8** | **KV-cache quantization — accuracy knob AND decode lever** | KV/attention are FP16 today (`multisize_report.md`: decode never nears the BW ceiling because KV is uncompressed). Add KV4/KV8 per-token/per-head + SmoothAttention (QServe per-head K scaling, inverse on Q) | Decode **+20–40%** at long context (the *only* lever that moves decode past parity per the report); KV4 costs ≈0.05–0.15 PPL — an independent trade-off axis | Needs fused RoPE+KV-write + INT4 KV read in attention (flagged as the biggest decode gap). Algorithm (`promix/quantize/kv_quant.py`) exists; kernel does not | 8B long-context decode tok/s + PPL at KV∈{FP16,8,4} | Low (algorithm known); value is closing our specific decode gap |
| **I9** | **Error-feedback rounding on the MX grid (GPTQ block-aware + AdaRound)** | We have GPTQ, but against frozen absmax scales. Stack on I3: GPTQ with MSE scales, Hessian error propagation respecting 16/32 block boundaries; optional AdaRound (learned per-weight rounding) within each block | **0.1–0.2 PPL** on top of GPTQ+MSE | Pack/calibration only (Step-2). AdaRound adds a short per-layer optimize | 1B: GPTQ-RTN vs GPTQ+MSE vs +AdaRound, PPL | Med — AdaRound-on-microscaling-blocks is largely unpublished |
| **I10** | **Rotation objective = block-scaled reconstruction error (Step-1 retarget)** | Step-1 SGD currently targets generic quant error (null on 1B). Retarget to minimize summed per-block MX-grid MSE, and/or learn a block-structured rotation (ties to I1) | Uncertain on 1B (Step-1 already null); plausibly matters at 7B+ | Step-1 rerun (objective change). No kernel change | 7B/8B only; 1B is known-null for Step-1 | Med — rotation learned *for* the MX grid specifically |

---

## 2. Detail on the highest-value ideas

### I1 — Block-local rotation / permutation (the novel contribution)

**What.** A block-scaled format quantizes each contiguous group (16 for NVFP4, 32
for MXFP8) with a *single* shared scale. The quantization error of a block is set
by that block's **internal dynamic range**: if a block holds one element of
magnitude 8 and fifteen of magnitude 0.5, the shared scale is sized for 8 and the
small elements land on 2–3 FP4 codes. A **global** rotation (our PCA + random
orthogonal, or a full-tensor Hadamard) improves the *whole-tensor* distribution
but has **no control over per-block balance** — it is blind to the 16-element grid
the kernel actually uses.

Two concrete mechanisms, both cheap:

1. **Balanced permutation (free, weights).** Sort channels by calibration absmax,
   then interleave so each 16-block receives a balanced mix (one large, one medium,
   … "snake"/round-robin assignment). Fold the permutation into the weight columns
   (free) and into the activation reorder we already pay (`reorder_before_quant`).
   Equalizes per-block absmax → every block's scale is near the global optimum.
2. **Block-diagonal rotation (cheap, activations).** Apply a size-16 (or 32)
   Hadamard *within each block*. On weights it folds in for free; on activations
   it is a `log2(16)=4`-stage butterfly in the pack epilogue (far cheaper than the
   full `d_ff` down_proj Hadamard we already run). It whitens *inside* each block,
   which is exactly where the MX scale lives.

**Why it helps us specifically.** We are the only stack whose kernel is built
around block-scaled dual-format *and* already runs online rotations. Everyone else
rotates for per-tensor/per-token INT (QuaRot/SpinQuant) where there is no block
grid to balance. For block-scaled FP4 the per-block range is the whole ballgame,
and nobody is optimizing it. This is a clean, defensible **paper contribution**:
"rotation/permutation co-designed with the microscaling block size."

**Expected gain & basis.** Block-16 FP4 MSE is roughly set by `(block_absmax /
block_rms)`. Our calibration can measure this ratio per block; balancing it toward
1 typically halves per-block MSE for peaky blocks, which on W4 regimes is
0.2–0.5 PPL (comparable to what full-tensor Hadamard buys, but *stackable* because
it operates at a different granularity).

**Cost / test.** Pack-only for the permutation variant (no kernel change). First
run an **offline oracle**: for 3–4 layers, histogram per-block `absmax/rms`,
compute FP4 MSE before/after permutation and before/after a size-16 Hadamard.
If MSE drops materially, run 1B/3B PPL pack-only. Build the activation-side
butterfly in the pack epilogue only if the weight-side permutation alone already
shows a win.

### I2 — Per-layer, per-component, sensitivity-driven channel selection

The failed adaptive experiment pinned high channels to the *shared average* tail
and only varied its length. Its own diagnosis names the fix: **(a)** each layer
selects its high-precision channels from *its own* ordering (needs per-layer
reorder, which we can fold into weights), and **(b)** order by a **sensitivity**
proxy, not variance-in-shared-basis.

**Cheap sensitivity proxy.** We already collect a calibration covariance (Step 0)
and a Hessian for GPTQ. Per channel `j`, use `s_j = H_jj · (w_j quantized-err)²`
summed over the layer's output rows — i.e., diagonal-Hessian-weighted
reconstruction error (the Optimal Brain Surgeon / GPTQ saliency, also Atom's
`metric='hessian'` act-stat, `outlier.py:18`). Channels with large `s_j` go to
MXFP8. This is one extra pass over data already loaded; no training.

**Knob.** Per-layer `K_high` becomes a real dial that maps to kernel cost (our
GEMM already takes a variable split). Allocate a global 8-bit budget across layers
by total `Σ s_j` saved per 8-bit channel (a water-filling allocation), rather than
uniform. This is the per-layer bit-allocation the user asked for, now with a proxy
the data did not reject.

### I3 — MSE-optimal block scales (cheapest immediate win)

Current packer: `select_block_scales_{nvfp4,mxfp8}` derives the *ideal* (absmax)
scale per block and freezes it; GPTQ then rounds Hessian-corrected columns against
that frozen scale (`quantize_column_with_frozen_scale`). The scale itself is never
optimized. Atom/GPTQ `find_params(mse=True)` instead sweeps a shrink factor
`p ∈ [maxshrink,1]`, quantizes, and keeps the `p` minimizing `Σ|q−x|^norm` per row
(`Atom/model/gptq.py:146-161`). For **block-16** this matters more than the usual
per-channel case: a 16-wide block has little averaging, so a single outlier
dictating the absmax scale is common, and a 5–10 % clip recovers many codes for
the other 15. Do the search **per block**, for both the NVFP4 and MXFP8 grids,
and freeze the clipped scale into the same thread the frozen-scale path already
uses. **No kernel change** — the kernel consumes whatever scale byte the packer
writes.

### I4 — Decoupled weight/activation precision

Weight error is static and GPTQ-compensated; activation error is dynamic,
per-token, and outlier-driven. There is no reason the *same* 1/8 split is optimal
for both. Expose `(K_high^W, K_high^A)` and allocate per layer by the respective
sensitivities (weight: GPTQ saliency; activation: per-token diag-Hessian). The
likely finding is that weights can run a smaller high segment (even `K_high^W=0`
in some layers, i.e. pure NVFP4 weights) while activations keep theirs — directly
shaving the 0.72-bit weight premium while holding PPL. This is the single biggest
structural lever on the *weight* bit count (the scoreboard axis), and it is a
genuine new axis in the trade-off space.

**Plumbing already half-exists.** The opt-in `adaptive_high_fraction` path
(`adaptive.py`, `eval/ptq.py:48-68,125-132`) already applies a per-layer `K_high`
to the **activation** quantizers *only* and is **not wired into weight GPTQ**
(`gptq.py:366` still uses the global `high_fraction`). So W and A high-splits can
already diverge structurally — I4 is mostly about choosing them by sensitivity and
extending the override to the weight path, not building new machinery. The current
adaptive path also varies only the *count* of high channels (not the format, and
excluding o_proj/down_proj via `ptq.py:216`), which is exactly why I7 (per-layer
*format*) and I2 (per-layer *channel set*, all components) are the net-new pieces.

---

## 3. What to try in the next 2 rounds

**Round A (accuracy, pack-only, no kernel, no Step 0/1 rerun) — prove the curve moves.**
Stack the three zero-kernel-cost levers and measure a bit-vs-PPL curve against the
NVFP4 point on the *same-protocol* 3B & 8B rows:
1. **I3** MSE-optimal block scales (biggest certainty, minutes).
2. **I1** balanced per-block permutation (weight-side only, free).
3. **I5** pull 16–32 outlier channels to MXFP8 and drop `r` to 1/16.
Report PPL at r ∈ {1/16, 1/8} with and without I1+I3. **Target:** r=1/16
(≈4.73 bits) matching the current r=1/8 (4.969) PPL → a strict accuracy-per-bit
improvement; and ideally r=1/16 + I6 scale-trim landing ≤4.5 bits at ≤ NVFP4 PPL.

**Round B (enlarge the space) — add the dials.**
4. **I2** per-layer own-ordering sensitivity selection (Step-0 reorder, Step-2
   repack) — turn the failed adaptive result positive.
5. **I4** decoupled `K_high^W`/`K_high^A` ablation (fake-quant first to pick the
   grid, then one repack).
6. **I7** high-segment width sweep {8,6,4} in fake-quant to decide if a 6-bit
   kernel path is worth building.
Keep **I8 (KV quant)** as the parallel *performance*-round lever — it is the only
thing that moves decode and is also an independent accuracy axis; it needs the
fused KV kernel, so it runs on the kernel agent's track, not this accuracy track.

---

## 4. The single experiment to run first (to prove Pareto dominance)

**MSE-optimal block scales (I3) + balanced per-block weight permutation (I1),
packing-only, on 3B and 8B, swept at r ∈ {1/16, 1/8}, plotted as bits-vs-PPL
against the vLLM-NVFP4 point.**

Rationale:
- **Cheapest decisive test.** Pure packer/calibration change: no kernel edit, no
  Step-0/1 rerun, runs in the existing `full_network_realkernel.py --phases ppl`
  harness in minutes per config.
- **Directly targets the Pareto claim.** NVFP4 currently *ties* us at 3B/8B while
  spending 0.72 fewer bits. If I1+I3 let **r=1/16 (≈4.73 bits) match the current
  r=1/8 (4.969) PPL**, we have moved our point down-and-left — a strict
  accuracy-per-bit gain — and with I6 scale-trim the same recipe aims at ≤4.5 bits
  at PPL ≤ NVFP4, which is Pareto dominance on the accuracy axis at equal bits.
- **Same-protocol rows.** 3B/8B are already aligned to vLLM's absolute PPL
  protocol, so the comparison is admissible to the scoreboard without the 1B
  protocol caveat.

Gate: pack-only PPL must beat the matched-bit uniform baseline, and the real-vs-twin
cosine/PPL gate (`|real−twin| ≤ 0.1`) must still pass — both already wired.
