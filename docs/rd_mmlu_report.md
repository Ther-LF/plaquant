# rd-mmlu (E1): MMLU 5-shot + NEW-recipe 0-shot — the two accuracy-metric gaps

**Branch:** `feat/rd-mmlu` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell
(SM120, CUDA 13.0), container `luofan-plaquant`, GPU lane 4 · **Date:** 2026-11-03
· **Status: RUNNING (skeleton)**

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

## 1. MMLU 5-shot (acc %, lm-eval group aggregate)

| model | BF16 | vLLM-NVFP4 | ProMix (new recipe) | ARCQuant | SharQ |
|---|---|---|---|---|---|
| Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | TBD | TBD | TBD | TBD | TBD |

TBD deltas + verdict.

## 2. NEW-recipe 0-shot 6-task (acc %, rd_metrics convention)

| model | bf16 (committed) | old anchor (committed) | **NEW recipe** | Δ vs old anchor | Δ vs bf16 |
|---|---|---|---|---|---|
| 1B | 59.33 | 57.39 (measured-1/8) | TBD | TBD | TBD |
| 3B | 68.11 | 66.77 (measured-1/8) | TBD | TBD | TBD |
| 8B | 73.33 | 72.69 (uniform-1/8) | TBD | TBD | TBD |

TBD per-task table + verdict.

## Raw data

`docs/rd_mmlu_data/` (JSONs per point; logs on ICT90 `/home/luofan/mmlu/`).
