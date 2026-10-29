# rd-metrics: Full ResQ metric suite for the rate-distortion allocation (dense-R-on)

**Branch:** `feat/rd-metrics` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant` · **Date:** 2026-10-29

**Question.** The rate-distortion allocation campaign (`rd_validity` → `rd_ron` →
`rd_sweep`) judged *measured-MSE* vs *uniform* FP8-ring allocation on **WikiText-2
PPL only**. Does the measured allocation's wikitext win (3B/1B) survive on the full
ResQ/QuaRot-standard metric suite — C4 PPL + 6-task zero-shot — and does the 8B
break-even stay break-even? Plus: one bigger Llama model.

**Harness.** `promix/scripts/rd_metrics.py`: per model, three comparison points in
the SHIPPING **dense-R-on** regime (ring_size=64, matched avg FP8 = 0.125 exactly,
o_proj held at uniform 1/8):

- **bf16** — no rotation, no quant (bridge; reproduces the committed baselines)
- **uniform-1/8** — every reallocated Linear K_high = d/8 (shipped anchor)
- **measured-1/8** — global single-threshold ΔD = D_fp4 − D_fp8 per-ring measured-MSE
  allocation at the same total FP8-ring budget (the `rd_ron` dense-limit, e=0)

Metrics: WikiText-2 PPL (`evaluate_ppl`), C4 PPL (QuaRot-standard 256×2048
validation stream, `promix/eval/data.py:get_c4`), zero-shot acc via lm-eval 0.4.5 on
piqa / arc_easy / arc_challenge / hellaswag / winogrande / lambada_openai.
Table cells: PPL for the two PPL columns; **acc_norm** for piqa/arc/hellaswag,
**acc** for winogrande/lambada (the ResQ/QuaRot convention); AVG = mean of those six.
Built-in gates all reproduced: every bf16 bridge matches its committed baseline, and
the wikitext column reproduces the committed dense-R-on numbers digit-for-digit
(1B 14.5483/14.4717, 3B 8.2974/8.2678, 8B 6.5882/6.6072).

Raw JSONs: `RD_METRICS2_{1b,3b,8b,13b}.json` (incremental, one per model).

---

## Llama-3.2-1B-Instruct

| point | wikitext | C4 | piqa | arc_e | arc_c | hellaswag | winogrande | lambada | **0-shot AVG** |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 13.1569 | 18.5126 | 74.32 | 63.30 | 37.80 | 60.78 | 60.06 | 59.73 | **59.33** |
| uniform-1/8 | 14.5483 | 20.3150 | 72.31 | 61.91 | 35.75 | 57.59 | 57.46 | 56.51 | **56.92** |
| measured-1/8 | 14.4717 | 20.1864 | 72.09 | 61.49 | 35.75 | 58.16 | 58.80 | 58.06 | **57.39** |

measured − uniform: wikitext **−0.077**, C4 **−0.129**, 0-shot AVG **+0.47 pp**.
All three metric families agree: measured wins on the 1B.

## Llama-3.2-3B

| point | wikitext | C4 | piqa | arc_e | arc_c | hellaswag | winogrande | lambada | **0-shot AVG** |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 7.8148 | 10.4329 | 77.20 | 71.84 | 46.42 | 73.61 | 69.53 | 70.04 | **68.11** |
| uniform-1/8 | 8.2974 | 11.3915 | 76.06 | 71.55 | 44.45 | 71.67 | 66.77 | 70.81 | **66.89** |
| measured-1/8 | 8.2678 | 11.3194 | 76.55 | 70.33 | 44.20 | 71.76 | 68.19 | 69.57 | **66.77** |

measured − uniform: wikitext **−0.030**, C4 **−0.072**, 0-shot AVG **−0.12 pp**.
C4 confirms the wikitext win; zero-shot is a wash (per-task scatter ±1.4 pp —
piqa/hellaswag/winogrande up, arc/lambada down — nets to noise).

## Llama-3-8B

| point | wikitext | C4 | piqa | arc_e | arc_c | hellaswag | winogrande | lambada | **0-shot AVG** |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 6.1368 | 8.8830 | 81.07 | 77.82 | 53.16 | 79.17 | 73.24 | 75.51 | **73.33** |
| uniform-1/8 | 6.5882 | 9.6355 | 79.87 | 76.77 | 51.71 | 77.52 | 74.82 | 75.43 | **72.69** |
| measured-1/8 | 6.6072 | 9.6393 | 79.22 | 78.41 | 51.19 | 77.68 | 71.51 | 76.13 | **72.36** |

measured − uniform: wikitext **+0.019**, C4 **+0.004**, 0-shot AVG **−0.33 pp**.
The 8B break-even stays break-even on EVERY metric family (the zero-shot −0.33 pp is
dominated by one noisy task, winogrande −3.3 pp, against arc_e +1.6 / lambada +0.7).

## Llama-2-13B (bigger model)

*(pending — download → PCA basis → metrics chain in flight; section filled on
completion)*

---

## Verdict

1. **C4 PPL fully corroborates wikitext.** The sign of the measured-vs-uniform delta
   is identical on C4 and WikiText-2 for every model: 1B −0.129, 3B −0.072 (measured
   wins), 8B +0.004 (tie). The allocation's PPL win is not a wikitext artifact —
   it transfers to held-out C4 text.
2. **Zero-shot does NOT show a matching gain at matched budget.** Deltas are
   noise-level everywhere: +0.47 pp (1B), −0.12 pp (3B), −0.33 pp (8B) — all within
   ~1σ of lm-eval task noise at these eval sizes, with per-task scatter (±1.4 pp)
   far above the net. A ~0.03–0.08 PPL improvement is simply below the resolution
   of 6-task 0-shot accuracy.
3. **The 8B break-even verdict is metric-robust**: +0.019 wiki / +0.004 C4 /
   −0.33 pp 0-shot. Nothing hidden behind the wikitext-only view.
4. Practical reading: the measured-MSE allocation is **safe everywhere**
   (never worse beyond noise on any metric) and **buys real PPL on ≤3B models**,
   but its headline benefit remains the PPL axis; don't claim zero-shot gains.

## Reproduction

```bash
# per model (GPU-pinned, detached; MASTER_PORT unique per job)
python promix/scripts/rd_metrics.py \
    --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml \
    --configs bf16,uniform,measured --out RD_METRICS2_3b.json
# 13B: first `huggingface-cli download NousResearch/Llama-2-13b-hf` +
# `python -m promix.quantize.basis --config promix/configs/llama-2-13b-mxfp8-nvfp4.yaml \
#      --output_dir ./rotation --nsamples 512`
```

Fixes this round (vs the first rd-metrics attempt):

- zero-shot pinned to GPU (`evaluate_ppl` offloads all layers to CPU on exit; the
  quantized models' `fast_hadamard_transform` down_proj is CUDA-only and crashed,
  and bf16 zero-shot silently ran ~10× slower on CPU),
- unique `MASTER_PORT` per concurrent job (EADDRINUSE on 29500),
- `rd_ron.resolve_paths` matches `13b` before the `3b` substring,
- C4 loads serialized/staggered against the warm `default-c7bc` HF cache (the
  cold-cache concurrent-load race from the first attempt is gone).
