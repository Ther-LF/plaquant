# rd-mmlu (E1): MMLU 5-shot + NEW-recipe 0-shot — the two accuracy-metric gaps

**Branch:** `feat/rd-mmlu` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell
(SM120, CUDA 13.0), container `luofan-plaquant`, GPU lane 4 · **Date:** 2026-11-03/04
· **Status: DONE** (all 10 MMLU cells + 3 zero-shot cells complete; 3B pm/arc/sharq
lm-eval stages were rerun once after OOM batch-size fixes — `*_r2` logs, all `rc=0`)

Fills `docs/paper_experiments.md` block-1 gap **E1**:

1. **MMLU 5-shot** (ARCQuant reports it — we must match) for
   {BF16, vLLM-NVFP4, ProMix NEW recipe, ARCQuant, SharQ} on 3B and 8B.
2. **NEW-recipe 0-shot 6-task average** for ProMix 1B/3B/8B (the committed 0-shot
   numbers in `rd_metrics_report.md` are the OLD dense anchors).

## Protocol

- **Harness:** `promix/scripts/rd_mmlu.py` (ProMix/BF16; lm-eval **0.4.5** HFLM, the
  exact rd_metrics GPU-pinning convention) and `promix/scripts/rd_newacc_peer.py
  --lmeval_tasks mmlu` (ARCQuant/SharQ as-shipped in `venv_acl44`, lm_eval pinned to
  **0.4.5** this round) · stage driver `promix/scripts/rd_mmlu_run.sh`.
- **vLLM-NVFP4:** llm-compressor checkpoints (`baselines/quant/*-nvfp4`) through
  lm-eval's vllm backend (lm-eval 0.4.13 + vLLM 0.32.0 — the same serving stack that
  produced the committed NVFP4 PPL rows; installed into `baselines/venv` without
  touching any pinned package). Cross-harness bridge: BF16 run through the SAME vllm
  harness must match the HF-0.4.5 BF16 number within noise.
- **MMLU:** task group `mmlu` (57 subtasks), `num_fewshot=5`, metric `acc`; headline
  = lm-eval's own group aggregate (= micro average; macro also recorded).
- **0-shot:** piqa / arc_easy / arc_challenge / hellaswag / winogrande /
  lambada_openai; acc_norm for piqa/arc/hellaswag, acc for winogrande/lambada
  (identical to `rd_metrics_report.md`).
- **Reproduction gate:** each ProMix NEW-recipe build must reproduce its committed
  acctune wikitext PPL (1B 14.1694 / 3B 8.2180 / 8B 6.5642) digit-for-digit before
  lm-eval runs (`--expect_wiki`, auto-abort on mismatch).
- ProMix NEW recipe = `rd_accuracy_tuning.md` shipping recipes: ring512 R-on
  substrate + global measured-ΔD @1/8 + act-MSE d3u2 + (3B/8B) weight-MSE-7 +
  GPTQ 512 (+percdamp 0.025 on 3B) / (1B) GPTQ 1024.

## 1. MMLU 5-shot (acc %, lm-eval group aggregate = micro; macro in parentheses)

| model | BF16 | vLLM-NVFP4 | ProMix (new recipe) | ARCQuant | SharQ |
|---|---|---|---|---|---|
| Llama-3.2-3B | 56.11 (57.49) | 52.63 (53.93) | **54.12** (55.70) | 52.64 (54.07) | 53.42 (54.82) |
| Llama-3-8B | 65.28 (66.61) | 62.34 (63.73) | 62.85 (63.97) | 62.19 (63.45) | **62.87** (64.12) |

**Cross-harness bridge (verified both models):** BF16 run through the same
vLLM-0.32.0/lm-eval-0.4.13 stack reads **56.47** on 3B (vs HF-0.4.5 56.11, **+0.36 pp**)
and **65.45** on 8B (vs 65.28, **+0.17 pp**) — the vLLM harness reads slightly HIGH, so
the raw vLLM-NVFP4 numbers above are, if anything, flattered. Bridge-adjusted NVFP4 on
the HF scale: 3B ≈ **52.27**, 8B ≈ **62.17**.

Δ vs BF16 (micro, pp; NVFP4 bridge-adjusted in parentheses):

| model | vLLM-NVFP4 | ProMix (new) | ARCQuant | SharQ |
|---|---|---|---|---|
| 3B | −3.48 (−3.84) | **−1.99** | −3.47 | −2.69 |
| 8B | −2.94 (−3.11) | −2.43 | −3.09 | **−2.41** |

**Verdict:** on 3B ProMix is the clear best quantized point — **+0.70 pp over SharQ,
+1.48 pp over ARCQuant and vLLM-NVFP4** (which land on top of each other, 52.64 vs
52.63). On 8B ProMix 62.85 vs SharQ 62.87 is a dead tie (0.02 pp ≪ the ~0.4 pp MMLU
sampling noise), both **+0.66–0.68 pp over ARCQuant** and ~+0.7 pp over bridge-adjusted
NVFP4. ProMix's reproduction gates passed digit-for-digit before each lm-eval run
(3B wikitext 8.2180, 8B 6.5642), and both ProMix builds hit avg FP8 = 0.125 exactly.

## 2. NEW-recipe 0-shot 6-task (acc %, rd_metrics convention)

| model | bf16 (committed) | old anchor (committed) | **NEW recipe** | Δ vs old anchor | Δ vs bf16 |
|---|---|---|---|---|---|
| 1B | 59.33 | 57.39 (measured-1/8) | 57.01 | −0.38 | −2.32 |
| 3B | 68.11 | 66.77 (measured-1/8) | 66.33 | −0.44 | −1.78 |
| 8B | 73.33 | 72.69 (uniform-1/8) | **73.02** | **+0.33** | −0.31 |

Per-task (acc_norm for piqa/arc/hellaswag, acc for winogrande/lambada; Δ vs the old
anchor in parentheses):

| model | wikitext | piqa | arc_e | arc_c | hellaswag | winogrande | lambada | **AVG** |
|---|---|---|---|---|---|---|---|---|
| 1B NEW | 14.1694 | 72.03 (−0.06) | 61.03 (−0.46) | 34.73 (−1.02) | 58.02 (−0.14) | 58.56 (−0.24) | 57.68 (−0.38) | **57.01** |
| 3B NEW | 8.2180 | 76.61 (+0.06) | 68.64 (−1.69) | 43.34 (−0.86) | 71.54 (−0.22) | 68.67 (+0.48) | 69.18 (−0.39) | **66.33** |
| 8B NEW | 6.5642 | 79.87 (0.00) | 77.53 (+0.76) | 53.07 (+1.36) | 77.54 (+0.02) | 73.56 (−1.26) | 76.54 (+1.11) | **73.02** |

**Verdict (honest):** the NEW recipe's clear wikitext-PPL wins over the old anchors
(1B 14.17 vs 14.47, 3B 8.218 vs 8.268, 8B 6.564 vs 6.588) do NOT translate into
0-shot average gains on 1B/3B — both tick down ~0.4 pp (1B: six small uniform
declines; 3B: dominated by arc_easy −1.69 pp), within the ±1.4 pp per-task scatter
already documented in `rd_metrics_report.md`. The 8B improves to +0.33 pp over the
old anchor and closes to −0.31 pp vs BF16. Combined with section 1, the NEW recipe's
headline strength on downstream accuracy is **MMLU** (3B best-in-class by a wide
margin, 8B tied-best), not the 0-shot suite.

## Raw data

`docs/rd_mmlu_data/`: `bf16_{3b,8b}.json`, `pm_{1b,3b,8b}.json` (ProMix NEW recipe —
MMLU for 3B/8B, 0-shot for all three, wikitext gate), `arc_{3b,8b}_mmlu.json`,
`sharq_{3b,8b}_mmlu.json`, `nv_{3b,8b}_mmlu_results.json` (vLLM-NVFP4),
`nvb_{3b,8b}_mmlu_results.json` (BF16-via-vLLM bridge). Full logs on ICT90
`/home/luofan/mmlu/` (`run_*.log`, `*_r2.log` = post-OOM-fix reruns, all
`RD_MMLU_STAGE_DONE rc=0`).
