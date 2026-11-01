# rd-calib: Calibration-set robustness of the rate-distortion allocation pipeline

**Branch:** `feat/rd-calib` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, isolated workdir
`/home/luofan/plaquant-calib` · **Date:** 2026-11-01 · **Model:** Llama-3.2-3B

**Worry being tested.** Every offline calibration artifact in the shipped
pipeline — the Step-0 PCA basis U/E, the GPTQ Hessians, and the measured-ΔD
allocation statistics — is fit on **wikitext-2 train** windows (512×2048,
seed 0). If the pipeline's accuracy (and specifically the measured-vs-uniform
allocation gain) were calibration-set-specific, it would evaporate on other
text. This is the standard quant-paper calibration-robustness ablation
(cf. GPTQ/QuaRot appendices), run on our strongest-gain model (3B).

**Design.** 2×2 (+ decomposition cells + one 0-shot spot check), dense-R-on
shipping regime (ring=64, matched avg FP8 = 0.125 exactly, o_proj uniform —
the exact `rd_metrics` recipe, so the wikitext-calibrated row is the committed
`docs/rd_metrics/RD_METRICS2_3b.json` numbers):

- **Calib ∈ {wikitext-512, C4-512}** — the C4 column re-runs the FULL offline
  pipeline on `allenai/c4` train: Step-0 basis `U-c4-512-Llama-3.2-3B.bin`
  (512×2048 windows, same seed/sampling as wikitext, via the new
  `basis.py --calib_dataset c4` override), GPTQ Hessians on 128 C4 windows,
  and ΔD collected on C4-rotated activations (16×2048, 4096 tokens/Linear).
  The rotation R file is **reused** (seed-fixed random, not data-fit) so only
  the data-fit artifacts vary. Act-MSE / weight-MSE scale searches are runtime
  searches (no offline artifact) — nothing to swap.
- **Config ∈ {uniform-1/8, measured-MSE-1/8}** — does the ALLOCATION gain
  specifically survive calibration shift?
- **Eval ∈ {wikitext-2 PPL, C4 PPL}** (committed protocol loaders) + one
  zero-shot 6-task AVG spot check on the C4-calibrated measured point.

Harness: `promix/scripts/rd_calib.py` (`--basis_calib/--gptq_calib/--alloc_calib`
control the three calibration entry points independently; `--mode overlap`
computes ring-selection stability) + `promix/scripts/rd_calib_basis_overlap.py`
(U-subspace principal angles, CPU). Raw JSONs: `docs/rd_calib/`.

---

## The 2×2 (+ spot check)

Llama-3.2-3B, matched 4.97 bit/weight, identical speed; bf16 bridge
7.8148 wiki / 10.4329 C4 / 68.11 AVG.

| calib (basis+GPTQ+ΔD) | config | wikitext PPL | C4 PPL | 0-shot AVG |
|---|---|---|---|---|
| wikitext-512 *(committed)* | uniform-1/8 | **8.2974** | 11.3915 | 66.89 |
| wikitext-512 *(committed)* | measured-1/8 | **8.2678** | **11.3194** | 66.77 |
| C4-512 *(this run)* | uniform-1/8 | 8.4086 | 11.3113 | — |
| C4-512 *(this run)* | measured-1/8 | 8.3897 | **11.2777** | **66.95** |

Deltas that answer the key questions:

1. **Cross-cell degradation (home-field effect).**
   - C4-calib → wiki-eval vs wiki-calib → wiki-eval: **+0.111** (uniform) /
     **+0.122** (measured) ≈ 1.3–1.5 %.
   - wiki-calib → C4-eval vs C4-calib → C4-eval: **+0.080** (uniform) /
     **+0.042** (measured) ≈ 0.4–0.7 %.
   So there IS a measurable but modest home-field effect (0.04–0.12 PPL,
   ~20 % of the total quantization gap of ~0.5–0.9 PPL). It never changes any
   ordering: even the worst cross cell (C4-calib uniform on wiki, 8.409) stays
   well ahead of the same-protocol peers (NVFP4 8.63, ResQ 8.71, QuaRot 10.72).
2. **The allocation gain survives calibration shift.** measured − uniform:
   - wiki-calib: **−0.030** wiki / **−0.072** C4;
   - C4-calib: **−0.019** wiki / **−0.034** C4.
   Same sign on every eval under both calibrations — including the fully
   shifted cell (C4-calibrated pipeline evaluated on wikitext). The gain
   shrinks somewhat under C4 calibration but never inverts.
3. **Zero-shot spot check:** C4-calibrated measured-1/8 AVG = **66.95** vs
   wikitext-calibrated measured 66.77 and uniform 66.89 — flat (±0.2 pp,
   well inside task noise). Calibration shift costs nothing on zero-shot.

Reproducibility bonus: the C4-calib uniform cell was built twice from scratch
(a mid-run GPU OOM from an unrelated job forced a retry on a different GPU) —
wikitext 8.4086 / C4 11.3113 both times, digit-for-digit.

## Decomposition: which artifact carries the sensitivity?

Single-knob swap cells on wiki-basis (uniform unless noted), wiki-eval /
C4-eval:

| cell | wikitext PPL | C4 PPL | reading |
|---|---|---|---|
| all-wikitext uniform (anchor) | 8.2974 | 11.3915 | — |
| **ΔD→C4 only** (measured, wiki basis+GPTQ) | 8.2570 | 11.3199 | vs all-wiki measured 8.2678/11.3194: **−0.011/+0.0005 — nil** |
| **GPTQ→C4 only** (uniform) | 8.3652 | 11.2992 | **+0.068** of the +0.111 wiki-eval penalty; on C4-eval it alone captures the whole home-field gain |
| full C4 pipeline (uniform) | 8.4086 | 11.3113 | residual basis-U effect ≈ **+0.043** wiki-eval |

Attribution of the C4-calib wiki-eval penalty (+0.111): **GPTQ Hessians
+0.068 > basis U +0.043 > ΔD allocation −0.011 (zero)**. The one artifact this
campaign actually introduced — the measured-ΔD allocation — is the SINGLE most
calibration-robust piece of the pipeline: measuring ΔD on C4 and plugging it
into the otherwise-wikitext pipeline reproduces (actually slightly beats) the
all-wikitext measured PPL. The home-field sensitivity lives in the
pre-existing GPTQ/PCA stages, at the magnitude the GPTQ literature reports for
calib-set swaps.

## Artifact-stability statistics

- **Ring-selection overlap** (ΔD measured on wikitext vs C4, same wiki-basis
  model → same coordinates, both at 1288/10304 FP8 rings): **81.4 %** identical
  rings (random-baseline 12.5 %). Per unit: q/k/v 86.7 %, gate/up 74.6 %,
  down_proj 0 rings under BOTH (the allocation's known down_proj-starving
  behavior is calib-independent). Full-vector ΔD Spearman is lower
  (mean 0.34) because ranks are noise-dominated in the near-zero-ΔD bulk;
  the budgeted tail — the only thing the allocation consumes — agrees at 81 %.
- **PCA basis subspace overlap** (top-1/8 eigenvector span, wiki-U vs C4-U;
  overlap = mean squared principal-angle cosine; random baseline 0.125):
  shared `attn_mlp` **0.689** (204/384 principal cosines > 0.9, median 0.91),
  per-layer down_proj mean **0.676** (min 0.496), per-head value mean
  **0.661**. The top-variance subspace is largely but not perfectly shared —
  consistent with the small basis term (+0.04) in the decomposition.

## Verdict

**Robust, with an honest asterisk.** The measured-MSE allocation — the
contribution under test — is calibration-set robust by every measure: 81 %
ring overlap across calib sets, zero PPL cost when ΔD is measured on a
different distribution than eval (8.2570 vs 8.2678), gain sign preserved in
all four cells, zero-shot flat. The full pipeline carries a modest
0.04–0.12 PPL (~1 %) home-field effect that decomposes into GPTQ (+0.068) and
PCA-basis (+0.043) calibration — shared-with-prior-work machinery, ~20 % of
the quantization gap, and never ordering-changing vs peers or between
configs. Not run: 8B C4 row (the allocation is already break-even at 8B, so
there is no 8B gain whose robustness needs establishing; box contention).

**One-line paper claim this supports:** *"Our measured-MSE allocation is
calibration-robust: swapping the entire offline calibration set from
WikiText-2 to C4 preserves the allocation's gain over uniform on both
WikiText-2 and C4 perplexity (and leaves 0-shot accuracy unchanged, 66.95 vs
66.77), the selected FP8 rings overlap 81 % across calibration sets (random:
12.5 %), and measuring ΔD on C4 while evaluating on WikiText-2 matches the
fully in-distribution result to within 0.01 PPL; the residual ≤0.12 PPL (~1 %)
calibration sensitivity is attributable to the standard GPTQ/PCA stages, not
to the allocation."*

## Reproduction

```bash
# Step 0 on C4 (writes U/E-c4-512-Llama-3.2-3B.bin; R reused if present)
python -m promix.quantize.basis --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
    --output_dir ./rotation --nsamples 512 --calib_dataset c4 \
    --model /home/luofan/models/Llama-3.2-3B
# C4-calibrated row (uniform + measured + 0-shot on measured)
python promix/scripts/rd_calib.py --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
    --basis_calib c4 --gptq_calib c4 --alloc_calib c4 \
    --configs uniform,measured --zs_configs measured --out RD_CALIB_3b_c4row.json
# decomposition cells
python promix/scripts/rd_calib.py --config ... --basis_calib wikitext \
    --gptq_calib wikitext --alloc_calib c4 --configs measured   # ΔD-only swap
python promix/scripts/rd_calib.py --config ... --basis_calib wikitext \
    --gptq_calib c4 --alloc_calib wikitext --configs uniform    # GPTQ-only swap
# stability stats
python promix/scripts/rd_calib.py --config ... --mode overlap   # ring overlap
python promix/scripts/rd_calib_basis_overlap.py \
    --basis_a rotation/U-wikitext-512-Llama-3.2-3B.bin \
    --basis_b rotation/U-c4-512-Llama-3.2-3B.bin --frac 0.125   # U subspace
```
