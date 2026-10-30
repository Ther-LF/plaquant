# rd-newacc: NEW W4A4 competitors measured in OUR protocol (ARCQuant / SharQ / MicroMix / CoQuant)

**Branch:** `feat/rd-newacc` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell
(SM120, CUDA 13.0), container `luofan-plaquant` · **Date:** 2026-10-30

**Question.** The 2025–26 Blackwell-FP4 wave (ARCQuant, SharQ, MicroMix) and the ResQ-successor
CoQuant previously existed in our docs only as PAPER numbers on THEIR models/protocols
(`docs/rd_related_work.md`). This round runs each method **as shipped** (official W4A4 recipe,
no tuning by us) on **our models** (`/home/luofan/models/Llama-3.2-3B`, `llama-3-8b`, plus the
rd-campaign 1B = unsloth/Llama-3.2-1B-Instruct) and scores it in **our committed protocol**, so
they sit on the same ground as ProMix and the old peers.

**Protocol** (identical to the committed scoreboard): wikitext-2-raw-v1 test, bf16 model,
`"\n\n".join`, non-overlapping 2048-token chunks, token-pooled exp(mean NLL); C4 =
QuaRot-standard 256×2048 windows from `en/c4-validation.00000-of-00008`, seed 0.
Harness: `promix/scripts/rd_newacc_peer.py` (quantizes via each peer's own `model_utils`,
evaluates via each peer's own GPTQ-standard PPL loop — mathematically identical to
`promix/eval/evaluator.py` for equal-length chunks) + `rd_newacc_mm_actscales.py`
(MicroMix gap-fill, see caveats). CoQuant runs its own shipped two-step pipeline (ResQ fork,
ResQ evaluator = the ancestor of ours; its repo-shipped `get_c4` is verbatim our C4 protocol).
Raw JSONs: `docs/rd_newacc_data/`.

**Bridge gate (MUST pass before any delta is trusted).** The FP16 (bf16) model run through the
exact peer-harness path reproduces the committed bridges:

| model | wikitext (committed) | C4 (committed) |
|---|---|---|
| 1B | 13.1586 (13.1569) ✓ | 18.5169 (18.5126) ✓ |
| 3B | 7.8135 (7.8148) ✓ | 10.4367 (10.4329) ✓ |
| 8B | 6.1356 (6.1368) ✓ | 8.8870 (8.8830) ✓ |

---

## Headline tables — W4A4 accuracy, our protocol, our models

Sorted by wikitext PPL per model. "eff bits/wt" = params-weighted effective weight bits
**including block-scale overhead**. KV = KV-cache bits during eval. Δ = PPL − bf16 bridge.
Bold **measured (new)** rows are this round's runs; "committed" rows are copied from the
committed scoreboard (`docs/rd_metrics_report.md`, `docs/baselines_on_hardware.md`, peer
RESULTS.md) for the complete picture.

### Llama-3.2-3B

| method | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 7.81 | — | 10.43 | — | measured (this round) |
| **ProMix measured-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **8.27** | **+0.45** | 11.32 | +0.88 | committed (rd_metrics) |
| ProMix uniform-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 8.30 | +0.48 | 11.39 | +0.95 | committed (rd_metrics) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual W4A4 | ~4.5 (†) | 16 | 8.39 | +0.57 | **11.29** | **+0.85** | **measured (new)** |
| **ARCQuant** (real kernels) | NVFP4+residual-ch W4A4 | **5.12** | 16 | 8.54 | +0.73 | 11.55 | +1.12 | **measured (new)** |
| vLLM-NVFP4 (modelopt) | plain NVFP4 W4A4 | 4.56 | 16 | 8.63 | +0.81 | n/a | — | committed (baselines) |
| **MicroMix** (‡ caveat) | MXFP4/6/8 channel mix | **5.23** | 16 | 9.05 | +1.23 | 12.28 | +1.85 | **measured (new)** |
| **CoQuant a_cov control** (=ResQ criterion) | INT4 + 1/8 ch INT8 | ~4.6 | **4** | 9.18 | +1.37 | 12.82 | +2.38 | **measured (new)** |
| **CoQuant wa_cov** (its own criterion) | INT4 + 1/8 ch INT8 | ~4.6 | **4** | 9.56 | +1.74 | 13.25 | +2.81 | **measured (new)** |
| Atom W4A4(≈4.2b)KV4 | INT4 + 128 INT8 ch | ~4.3 | 4 | 9.70 | +1.88 | n/a | — | committed (peer RESULTS) |
| SpinQuant (Hadamard) | INT4 W4A4KV4 | 4.0 | 4 | 10.09 | +2.28 | n/a | — | committed (peer RESULTS) |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 11.23 | +3.41 | n/a | — | committed (peer RESULTS) |

### Llama-3-8B

| method | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 6.14 | — | 8.89 | — | measured (this round) |
| **ProMix uniform-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **6.59** | **+0.45** | 9.64 | +0.75 | committed (rd_metrics) |
| ProMix measured-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 6.61 | +0.47 | 9.64 | +0.75 | committed (rd_metrics) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual W4A4 | ~4.5 (†) | 16 | 6.68 | +0.54 | **9.58** | **+0.69** | **measured (new)** |
| **ARCQuant** (real kernels) | NVFP4+residual-ch W4A4 | **5.14** | 16 | 6.81 | +0.67 | 9.79 | +0.91 | **measured (new)** |
| vLLM-NVFP4 (modelopt) | plain NVFP4 W4A4 | 4.56 | 16 | 6.87 | +0.73 | n/a | — | committed (baselines) |
| **CoQuant a_cov control** (=ResQ criterion) | INT4 + 1/8 ch INT8 | ~4.6 | **4** | 7.35 | +1.21 | 10.79 | +1.90 | **measured (new)** |
| **MicroMix** (‡ caveat) | MXFP4/6/8 channel mix | **5.18** | 16 | 7.39 | +1.26 | 10.56 | +1.68 | **measured (new)** |
| Atom W4A4(≈4.2b)KV4 | INT4 + 128 INT8 ch | ~4.3 | 4 | 7.57 | +1.43 | n/a | — | committed (peer RESULTS) |
| **CoQuant wa_cov** | INT4 + 1/8 ch INT8 | ~4.6 | **4** | 7.69 | +1.55 | 11.18 | +2.30 | **measured (new)** |
| SpinQuant (Hadamard) | INT4 W4A4KV4 | 4.0 | 4 | 7.88 | +1.75 | n/a | — | committed (peer RESULTS) |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 8.86 | +2.72 | n/a | — | committed (peer RESULTS) |

### Llama-3.2-1B-Instruct (bonus round)

| method | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 13.16 | — | 18.52 | — | measured (this round) |
| **ProMix measured-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **14.47** | **+1.31** | **20.19** | **+1.67** | committed (rd_metrics) |
| ProMix uniform-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 14.55 | +1.39 | 20.32 | +1.80 | committed (rd_metrics) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual | ~4.5 (†) | 16 | 14.79 | +1.63 | 20.96 | +2.44 | **measured (new)** |
| **ARCQuant** (real kernels) | NVFP4+residual-ch | **5.22** | 16 | 15.22 | +2.07 | 20.79 | +2.28 | **measured (new)** |
| vLLM-NVFP4 (modelopt) | plain NVFP4 | 4.56 | 16 | 15.65 | +2.49 | n/a | — | committed (baselines) |
| SpinQuant (learned R) | INT4 W4A4KV4 | 4.0 | 4 | 16.14 | +2.99 | n/a | — | committed (peer RESULTS) |
| **CoQuant wa_cov** | INT4 + 1/8 ch INT8 | ~4.6 | **4** | 16.64 | +3.48 | 23.32 | +4.81 | **measured (new)** |
| Atom | INT4 + 128 INT8 ch | ~4.3 | 4 | 18.40 | +5.24 | n/a | — | committed (peer RESULTS) |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 26.00 | +12.84 | n/a | — | committed (peer RESULTS) |

---

## Bits accounting (how each "eff bits/wt" was computed)

- **ProMix 4.97** = 7/8·NVFP4(4 + 8/16 scale) + 1/8·MXFP8(8 + 8/32 scale); matches the
  measured packed-weight memory (4.97 bit/wt, decode report).
- **ARCQuant 5.12 / 5.14 / 5.22 (3B/8B/1B)** = params-weighted mean of the method's OWN
  `saved/*_average_bits_wikitext2_max.pt` (= 4.5·(K+S)/K per Linear; S = adaptive residual
  channels, S̄ ≈ 385 (3B) / 483 (8B) per Linear; gate/up_proj reach 5.3–5.8 bits).
  ARCQuant duplicates the top-S channels along K — its "unified format" costs ~0.15–0.25
  bits/wt MORE than our dual-format 1/8 split.
- **SharQ ~4.5 (†)** = single shared NVFP4 weight payload (4 + g32 shared scales ×2 views).
  († the weight payload is single, but every activation is processed twice — sparse 2:4
  backbone + dense residual GEMM per Linear, ~1.5–2× GEMM work and two activation streams;
  weight bits alone understate its runtime cost.)
- **MicroMix 5.23 / 5.18** = params-weighted (4·p4 + 6·p6 + 8·p8)/K + 0.25 UE8M0 scale, from
  its own saved p6/p8 allocations.
- **vLLM-NVFP4 4.56** = measured checkpoint size (6.05 GB / 8B).
- **CoQuant ~4.6** = ResQ INT recipe: 7/8·INT4 + 1/8·INT8 + per-group scales.

## As-shipped execution notes (what actually ran)

- **ARCQuant** (`actypedef/ARCQuant`, ACL'26): official pipeline — `reorder_indices.py`
  (128×2048 wikitext calib, act_sort_metric=max) → `reorder_model_llama` NVFP4 through its
  **real `agemm` CUDA kernels** (SM120a/CUDA13 build, smoke-passed). KV bf16 (its
  `evaluate.sh` does not pass `--kv_cache`).
- **SharQ** (`actypedef/SharQ`): calibration-free; `quantize_model_llama(quant_type=SHARQ)`
  through its **real fused `sharq_ops` kernels** (built this round per repo
  SERVER_BUILD_NOTES, SM120a/CUDA13). KV bf16 (default).
- **MicroMix** (`lwy2020/MicroMix`): `reorder_indices.py` calibration → SmoothQuant
  `smooth_lm` → `reorder_model_llama` through its **real `mixedgemm` kernels**.
  **‡ Two caveats:** (1) its kernels FAIL their own `--validate` on CUDA 13 (repo pins
  CUDA 12.8) — these PPLs run through those kernels and are plausibly degraded: the paper
  reports Δwiki +0.48 on Llama-3.1-8B vs our measured +1.26 on Llama-3-8B. Treat as the
  method's honest as-shipped result on THIS box and a pessimistic bound on its true accuracy.
  (2) the repo ships NO generator for the required SmoothQuant `saved/{model}.pt` act-scales —
  gap-filled with the standard SmoothQuant abs-max generator
  (`promix/scripts/rd_newacc_mm_actscales.py`, wikitext 128×2048).
- **CoQuant** (`Zachary5895/CoQuant`, ResQ fork): official two-step
  (`get_basis.py` 512×2048 wikitext → `ptq.py` W4A4**KV4** INT, GPTQ+clip, per-layer rotation
  granularity, random seeded R, no rotation training — their shipped recipe; note KV4 where
  the FP rows are KV16, and INT formats, so it is the one row NOT format-matched to the
  NVFP4 group). 4 minimal compat patches recorded in `peers/coquant/PATCHES.md`: dead
  `FSDPTrainer` import (file not shipped), hardcoded `/dz/dataset/wikitext` →
  `Salesforce/wikitext`, +C4 eval via its own repo-shipped `get_c4` (verbatim our protocol),
  tokenizer `use_fast` False→True (Llama-3 has no slow tokenizer). 3B/8B use
  `k/v_groupsize=128` (= head_dim, required by its own K-cache wrapper assert; the shipped
  script's 64 is the 1B head_dim).
- Env pins per peer (venvs over container torch 2.11.0+cu130): ARCQuant/SharQ
  transformers 4.45.2 (repos pin 4.44.0, which cannot load Llama-3.2's tokenizer.json —
  minimal bump), MicroMix 4.50.0 (as pinned), CoQuant on container transformers 4.48.1.
  Same model weights as the rd campaign. lm-eval zero-shot phases were skipped (PPL scope).

## CoQuant deep-dive: the claimed subspace win does NOT reproduce

CoQuant's claim is that its joint weight+activation weighted-PCA subspace (wa_cov) beats the
ResQ activation-only criterion (a_cov) by 0.2–0.4 PPL (paper Table 1: 3B wa_cov 8.93 vs
a_cov/ResQ 9.18; FP16 7.81 = our bridge, same protocol). We ran BOTH criteria through its own
pipeline, same seed (R generation seeded by `transformers.set_seed(0)` in `get_basis.py`,
so the only difference is the basis):

| model | wa_cov (CoQuant) wiki / C4 | a_cov (ResQ control) wiki / C4 | paper claim |
|---|---|---|---|
| 3B | 9.5552 / 13.2484 | **9.1803 / 12.8181** | wa_cov 8.93 < a_cov 9.18 |
| 8B | 7.6908 / 11.1848 | **7.3510 / 10.7912** | n/a (paper has no 8B) |

Our a_cov control lands **exactly on the paper's ResQ number (9.18)** — the pipeline and
protocol match — but wa_cov comes out 0.38 WORSE instead of 0.25 better, and the same
reversal repeats on 8B (a_cov better by 0.34 wiki / 0.39 C4). On BOTH our models the
paper's ordering REVERSES: the "better-than-variance" subspace criterion did not reproduce.
(Both variants are in any case far above ProMix 8.27 and even above MicroMix.) As an ablation
target for our Phase-2 "error-driven partition" plans, wa_cov as published is not evidence
that joint-W+A PCA beats the activation-variance subspace at 1/8.

## Verdict — where does ProMix sit on accuracy-per-bit vs the NEW competitors?

1. **vs ARCQuant (the closest competitor, ACL'26, same GPU): ProMix wins outright.**
   Lower PPL on all three models and both datasets (3B 8.27 vs 8.54; 8B 6.59 vs 6.81;
   1B 14.47 vs 15.22; C4 likewise) at FEWER effective bits (4.97 vs 5.12–5.22).
   ARCQuant's residual channels do beat plain NVFP4 (8.54/6.81 vs 8.63/6.87) but recover
   only ~¼ of the NVFP4→FP16 gap, while its channel duplication costs more bits than our
   1/8 MXFP8 subspace. On accuracy-per-bit ProMix strictly dominates ARCQuant.
2. **vs SharQ (the strongest NEW accuracy competitor): honest split, ProMix ahead overall.**
   ProMix is better on wikitext at every size (3B −0.12, 8B −0.09, 1B −0.32); SharQ is
   slightly better on C4 at 3B/8B (−0.03/−0.06) but worse at 1B (+0.77), at ~4.5 weight bits
   vs our 4.97. SharQ's decomposition pays elsewhere: two activation streams + sparse+dense
   GEMM per Linear (~1.5–2× GEMM work). Net: roughly accuracy-parity-minus on fewer weight
   bits but higher compute — it does NOT dominate ProMix; ProMix wins the headline metric
   (wikitext) everywhere.
3. **vs MicroMix: ProMix wins at ~0.25 FEWER bits** on this box (9.05/7.39 vs 8.27/6.59) —
   with the honest caveat that MicroMix's kernels mis-validate under CUDA 13; its paper
   numbers (Δ+0.48 on Llama-3.1-8B at ~5.2 bits) would still sit at or behind ProMix's
   Δ+0.45 at 4.97 bits.
4. **vs CoQuant: not competitive as shipped** (9.18–9.56 on 3B, 7.35–7.69 on 8B, INT KV4),
   and its core claimed improvement over the ResQ a_cov criterion reverses in its own
   pipeline on BOTH our models (same seed, basis-only difference).
5. **Overall:** on these models the accuracy-per-bit frontier of the entire W4A4 field
   measured on this box is ProMix (4.97b) → SharQ (4.5b weights, C4-competitive, compute
   overhead) → ARCQuant (5.1b) → vLLM-NVFP4 (4.56b) → MicroMix (5.2b, caveat) → INT methods.
   No NEW competitor matches ProMix on wikitext at any bit budget; SharQ is the only one
   that touches it on any metric (C4, by ≤0.06).
