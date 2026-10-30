# rd-newacc: NEW W4A4 competitors measured in OUR protocol (ARCQuant / SharQ / MicroMix / CoQuant)

**Branch:** `feat/rd-newacc` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell
(SM120, CUDA 13.0), container `luofan-plaquant` · **Date:** 2026-10-30

**Question.** The 2025–26 Blackwell-FP4 wave (ARCQuant, SharQ, MicroMix) and the ResQ-successor
CoQuant previously existed in our docs only as PAPER numbers on THEIR models/protocols
(`docs/rd_related_work.md`). This round runs each method **as shipped** (official W4A4 recipe,
no tuning by us) on **our models** (`/home/luofan/models/Llama-3.2-3B`, `llama-3-8b`) and scores
it in **our committed protocol**, so they sit on the same ground as ProMix and the old peers.

**Protocol** (identical to the committed scoreboard): wikitext-2-raw-v1 test, bf16 model,
`"\n\n".join`, non-overlapping 2048-token chunks, token-pooled exp(mean NLL); C4 =
QuaRot-standard 256×2048 windows from `en/c4-validation.00000-of-00008`, seed 0.
Harness: `promix/scripts/rd_newacc_peer.py` (quantizes via each peer's own `model_utils`,
evaluates via each peer's own GPTQ-standard PPL loop — mathematically identical to
`promix/eval/evaluator.py` for equal-length chunks) + `rd_newacc_mm_actscales.py`
(MicroMix gap-fill, see caveats). Raw JSONs: `docs/rd_newacc_data/`.

**Bridge gate (MUST pass before any delta is trusted).** The FP16 (bf16) model run through the
exact peer-harness path reproduces the committed bridges:

| model | wikitext (committed) | C4 (committed) |
|---|---|---|
| 3B | 7.8135 (7.8148) ✓ | 10.4367 (10.4329) ✓ |
| 8B | 6.1356 (6.1368) ✓ | 8.8870 (8.8830) ✓ |

---

## Headline table — W4A4 accuracy, our protocol, our models

Sorted by wikitext PPL per model. "eff bits/wt" = params-weighted effective weight bits
**including block-scale overhead**. KV column = KV-cache bits during eval.
ΔFP16 = wikitext PPL − bf16 bridge (3B 7.81 / 8B 6.14; C4 10.43 / 8.88).

### Llama-3.2-3B

| method | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 7.81 | — | 10.43 | — | measured (this round) |
| **ProMix measured-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **8.27** | **+0.45** | 11.32 | +0.89 | committed (rd_metrics) |
| ProMix uniform-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 8.30 | +0.48 | 11.39 | +0.96 | committed (rd_metrics) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual W4A4 | ~4.5 (†) | 16 | 8.39 | +0.57 | **11.29** | **+0.85** | **measured (this round)** |
| **ARCQuant** (real kernels) | NVFP4+residual-ch W4A4 | **5.12** | 16 | 8.54 | +0.73 | 11.55 | +1.12 | **measured (this round)** |
| vLLM-NVFP4 (modelopt) | plain NVFP4 W4A4 | 4.56 | 16 | 8.63 | +0.81 | n/a | — | committed (baselines_on_hardware) |
| **MicroMix** (‡ caveat) | MXFP4/6/8 channel mix | **5.23** | 16 | 9.05 | +1.23 | 12.28 | +1.85 | **measured (this round)** |
| **CoQuant** (wa_cov) | INT4 + 1/8 ch INT8 | ~4.6 | **4** | PENDING | | PENDING | | measured (this round) |
| Atom W4A4(≈4.2b)KV4 | INT4 + 128 INT8 ch | ~4.3 | 4 | 9.70 | +1.88 | n/a | — | committed (peer RESULTS) |
| SpinQuant (Hadamard) | INT4 W4A4KV4 | 4.0 | 4 | 10.09 | +2.28 | n/a | — | committed (peer RESULTS) |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 11.23 | +3.41 | n/a | — | committed (peer RESULTS) |

### Llama-3-8B

| method | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 6.14 | — | 8.89 | — | measured (this round) |
| **ProMix uniform-1/8** | NVFP4+MXFP8 W4A4 | **4.97** | 16 | **6.59** | **+0.45** | 9.64 | +0.75 | committed (rd_metrics) |
| ProMix measured-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 6.61 | +0.47 | 9.64 | +0.76 | committed (rd_metrics) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual W4A4 | ~4.5 (†) | 16 | 6.68 | +0.54 | **9.58** | **+0.69** | **measured (this round)** |
| **ARCQuant** (real kernels) | NVFP4+residual-ch W4A4 | **5.14** | 16 | 6.81 | +0.67 | 9.79 | +0.91 | **measured (this round)** |
| vLLM-NVFP4 (modelopt) | plain NVFP4 W4A4 | 4.56 | 16 | 6.87 | +0.73 | n/a | — | committed (baselines_on_hardware) |
| **CoQuant** (wa_cov) | INT4 + 1/8 ch INT8 | ~4.6 | **4** | PENDING | | PENDING | | measured (this round) |
| **MicroMix** (‡ caveat) | MXFP4/6/8 channel mix | **5.18** | 16 | 7.39 | +1.26 | 10.56 | +1.68 | **measured (this round)** |
| Atom W4A4(≈4.2b)KV4 | INT4 + 128 INT8 ch | ~4.3 | 4 | 7.57 | +1.43 | n/a | — | committed (peer RESULTS) |
| SpinQuant (Hadamard) | INT4 W4A4KV4 | 4.0 | 4 | 7.88 | +1.75 | n/a | — | committed (peer RESULTS) |
| QuaRot | INT4 W4A4KV4 | 4.0 | 4 | 8.86 | +2.72 | n/a | — | committed (peer RESULTS) |

---

## Bits accounting (how each "eff bits/wt" was computed)

- **ProMix 4.97** = 7/8·NVFP4(4 + 8/16 scale) + 1/8·MXFP8(8 + 8/32 scale); matches the
  measured packed-weight memory (4.97 bit/wt, decode report).
- **ARCQuant 5.12 / 5.14** = params-weighted mean of the method's OWN
  `saved/*_average_bits_wikitext2_max.pt` (= 4.5·(K+S)/K per Linear, S = residual channels,
  adaptive; S̄ ≈ 385 (3B) / 483 (8B) per Linear; gate/up_proj reach 5.3–5.8 bits).
  ARCQuant duplicates the top-S channels along K — its "unified format" costs ~0.6 bits/wt
  MORE than our dual-format split at 1/8.
- **SharQ ~4.5 (†)** = single shared NVFP4 weight payload (4 + g32 shared scales ×2 views).
  († the weight payload is single, but every activation is processed twice — sparse backbone
  (2:4) + dense residual — i.e., ~1.5–2× GEMM work and 2 activation streams per Linear; weight
  bits alone understate its runtime cost.)
- **MicroMix 5.23 / 5.18** = params-weighted (4·p4 + 6·p6 + 8·p8)/K + 0.25 UE8M0 scale, from
  its own saved p6/p8 allocations.
- **vLLM-NVFP4 4.56** = measured checkpoint size (6.05 GB / 8B).
- **CoQuant ~4.6** = ResQ recipe: 7/8·INT4 + 1/8·INT8 + per-group scales (same as ResQ-INT).

## As-shipped execution notes (what actually ran)

- **ARCQuant** (`actypedef/ARCQuant`, ACL'26): official pipeline — `reorder_indices.py`
  (128×2048 wikitext calib, act_sort_metric=max) → `reorder_model_llama` NVFP4 through its
  **real `agemm` CUDA kernels** (built SM120a/CUDA13, smoke-passed earlier). KV left bf16
  (its `evaluate.sh` does not pass `--kv_cache`).
- **SharQ** (`actypedef/SharQ`): calibration-free; `quantize_model_llama(quant_type=SHARQ)`
  through its **real fused `sharq_ops` kernels** (built this round, SM120a/CUDA13, per repo
  SERVER_BUILD_NOTES). KV bf16 (default).
- **MicroMix** (`lwy2020/MicroMix`): `reorder_indices.py` calibration → SmoothQuant
  `smooth_lm` → `reorder_model_llama` through its **real `mixedgemm` kernels**.
  **‡ Two caveats:** (1) its kernels FAIL their own `--validate` on CUDA 13 (repo pins
  CUDA 12.8) — these PPLs run through those kernels and are plausibly degraded: the paper
  reports Δwiki +0.48 on Llama-3.1-8B vs our measured +1.26 on Llama-3-8B. Treat as the
  method's honest as-shipped result on THIS box, a lower bound on its true accuracy.
  (2) the repo ships NO generator for the required SmoothQuant `saved/{model}.pt` act-scales —
  we gap-filled with the standard SmoothQuant abs-max generator
  (`promix/scripts/rd_newacc_mm_actscales.py`, wikitext 128×2048).
- **CoQuant** (`Zachary5895/CoQuant`, ResQ fork): official two-step
  (`get_basis.py` wa_cov 512×2048 → `ptq.py` W4A4**KV4** GPTQ+clip, per-layer rotation
  granularity — their shipped recipe; note KV4 where our ProMix rows are KV16).
  3 minimal compat patches (recorded in `peers/coquant/PATCHES.md`): dead `FSDPTrainer`
  import (file not shipped), hardcoded `/dz/dataset/wikitext` path → `Salesforce/wikitext`,
  and +C4 eval via its own repo-shipped `get_c4` (verbatim our protocol).
- Env pins per peer (venvs over container torch 2.11.0+cu130): ARCQuant/SharQ
  transformers 4.45.2 (repo pins 4.44.0, which cannot load Llama-3.2's tokenizer.json —
  minimal bump), MicroMix 4.50.0 (as pinned). Same model weights as the rd campaign.

## Verdict — where does ProMix sit vs the NEW competitors?

PENDING-FINAL (filled after CoQuant lands); current picture from ARCQuant/SharQ/MicroMix:

1. **vs ARCQuant (the closest competitor): ProMix wins outright.** Lower PPL on both models
   and both datasets (3B 8.27 vs 8.54; 8B 6.59 vs 6.81; C4 likewise) at FEWER effective bits
   (4.97 vs 5.12–5.14). ARCQuant's residual channels do beat plain NVFP4 (8.54/6.81 vs
   8.63/6.87) but recover only ~1/4 of the NVFP4→FP16 gap, while costing more bits than our
   1/8 MXFP8 split. On accuracy-per-bit, ProMix dominates ARCQuant.
2. **vs SharQ: honest split decision.** ProMix is better on wikitext (3B −0.12, 8B −0.09);
   SharQ is slightly better on C4 (3B −0.03, 8B −0.06) at ~4.5 weight bits. SharQ is the
   strongest NEW accuracy competitor — but its decomposition doubles the activation work
   (sparse+dense GEMM per Linear), so equal-ish accuracy comes at a different system cost
   point, not a dominating one.
3. **vs MicroMix: ProMix wins at ~0.25 FEWER bits** (as measured on this box; see ‡ caveat —
   on CUDA 12.8 hardware the paper's numbers would be closer, at its ~5.2-bit budget).

