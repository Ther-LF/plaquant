# rd-peers70b: ARCQuant + SharQ at 70B — filling the 70B W4A4 peer gap

**Branch:** `feat/rd-peers70b` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120, 96 GB), host ICT90, container `luofan-plaquant`, GPU lanes 0-2
· **Date:** 2026-11-05 · **Status:** COMPLETE

**Mission.** `docs/rd_scale_report.md` (E2) extended the accuracy ladder to
Meta-Llama-3-70B-Instruct, but its 70B accuracy table only had {bf16, NVFP4,
ProMix}. This round runs the two closest W4A4 rivals — **ARCQuant** (ACL'26,
NVFP4 + adaptive residual channels, real `agemm` kernels) and **SharQ**
(all-NVFP4 sparse-2:4 + residual decomposition, real fused `sharq_ops`
kernels), both previously measured at 1B/3B/8B in `docs/rd_newpeers_accuracy.md`
— on the SAME 70B model in the SAME protocol, so the 70B row has real peers.

## 1. The 70B memory problem and what was (and wasn't) changed

70B fp16 = 140 GB > 96 GB GPU; both peers' shipped eval loads the whole model
onto one GPU. Resolution, in line with the minimal-intervention policy:

- **Quantization: UNTOUCHED.** Both peers' as-shipped quantizers are already
  layer-offloading (move one decoder layer to GPU, wrap/quantize, move back);
  they ran at 70B unmodified. Their packed NVFP4 weights live GPU-resident
  (~40-41 GB at 70B — the quantized model fits a single 96 GB card).
- **Eval loop: the ONE substitution.** Each peer's `model/eval.py:eval_ppl` is
  already a GPTQ-standard layer-offload loop EXCEPT one whole-model
  `model.to("cuda:0")` line (OOM at 70B). `promix/scripts/rd_peers70b.py`
  (= `rd_newacc_peer.py` 70B-safe) re-implements that loop with the
  whole-model move removed and `rotary_emb` handled — identical chunking and
  NLL math (non-overlapping 2048-token chunks, exp(sum nll / total tokens)),
  i.e. the same layer-offload evaluation used for every committed 70B row
  (`promix/eval/evaluator.py:evaluate_ppl` via rd_scale).
- **ARCQuant calibration, one output-identical device patch.** Its
  `reorder_indices.py`/`get_act_stats` pre-registers hook weights
  (`qkv/gate_up_weight_combined`) on GPU for ALL layers up front — ~88.6 GB at
  70B. With the shipped `act_sort_metric=max` these tensors are never read by
  `stat_tensor` (the weight-using code is commented out upstream), so the patch
  keeps them on CPU. Recipe output is unchanged; diff recorded in
  `docs/rd_peers70b_data/arcquant_utilize_calib.patch`.

**Gates (all passed before any 70B number was trusted):**

| gate | result |
|---|---|
| 8B FP16 through the new offload loop | wiki **6.1356** == committed 6.1356 (bit-identical) |
| 8B ARCQuant NVFP4 through it | wiki **6.8054** == rd-newacc 6.805419 (bit-identical) |
| 8B SharQ through it | wiki **6.6769** == rd-newacc 6.676935 (bit-identical) |
| **70B bf16 bridge** | wiki **5.3182** / C4 **9.2050** vs committed 5.3159 / 9.2036 (Δ 0.0023 / 0.0014 — same 3rd-decimal band as every accepted rd-newacc bridge gate; batch_size=1 vs 8 reduction-order wiggle) |

## 2. What ran (as-shipped recipes)

- **ARCQuant**: official pipeline on the 70B — `reorder_indices.py` (128×2048
  wikitext calib, act_sort_metric=max, 9.6 min) → `reorder_model_llama` NVFP4
  through its real `agemm` SM120 kernels (quantize 133 s). Adaptive residual:
  mean select_num 725 ch/Linear → **5.14 eff bits/wt** (params-weighted from
  its own saved `*_average_bits` file, 4.5·(K+S)/K — same accounting as
  rd-newacc; its log's unweighted mean prints 4.79). KV bf16 (its evaluate.sh
  does not pass `--kv_cache`).
- **SharQ**: calibration-free — `quantize_model_llama(quant_type=SHARQ,
  extra_fusion=False)` through its real fused `sharq_ops` kernels (quantize
  132 s). ~4.5 weight bits (single shared NVFP4 payload; the standing caveat:
  every activation is processed twice — sparse 2:4 backbone + dense residual
  GEMM per Linear, ~1.5-2× GEMM work). KV bf16.
- Env: venv_acl44 (torch 2.11.0+cu130, transformers 4.45.2), the exact
  rd-newacc peer env; `PYTHONPATH=kernels/build` for agemm. Single GPU each;
  GPU peak 53-60 GB (quantized evals), 19 GB (bf16 bridge).

## 3. Results — the extended 70B table (wiki / C4 PPL, our protocol, same Meta-Llama-3-70B-Instruct)

Sorted by wikitext. Δ vs the committed bf16 bridge (5.3159 / 9.2036).
{bf16, NVFP4, ProMix} rows are the committed rd_scale numbers; bold rows are
this round's runs.

| point | class | eff bits/wt | KV | wikitext | Δwiki | C4 | ΔC4 | source |
|---|---|---|---|---|---|---|---|---|
| bf16 bridge | — | 16 | 16 | 5.3159 | — | 9.2036 | — | committed (rd_scale) |
| **SharQ** (fused kernels) | all-NVFP4 sparse+residual W4A4 | **~4.5 (†)** | 16 | **5.7693** | **+0.453** | **9.6590** | **+0.455** | **measured (new)** |
| vLLM NVFP4 W4A4 | plain NVFP4 | 4.50 | 16 | 5.9279 | +0.612 | 9.7461 | +0.542 | committed (rd_scale) |
| ProMix ring512 measured-1/8 | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 5.9732 | +0.657 | 9.8917 | +0.688 | committed (rd_scale) |
| ProMix ring512 measured + up2 + GPTQ512 (tuned) | NVFP4+MXFP8 W4A4 | 4.97 | 16 | 5.9829 | +0.667 | 9.8549 | +0.651 | committed (rd_scale) |
| **ARCQuant** (real kernels) | NVFP4+residual-ch W4A4 | **5.14** | 16 | **6.1834** | **+0.868** | **10.0487** | **+0.845** | **measured (new)** |

(† single weight payload, but ~1.5-2× GEMM work / two activation streams per
Linear — weight bits alone understate SharQ's runtime cost.)

**Headline (honest):** at 70B **SharQ leads the whole measured W4A4 field** —
−0.159 wiki / −0.087 C4 below plain NVFP4 and −0.204 wiki / −0.233 C4 below
ProMix's best standard point, at fewer weight bits (~4.5 vs 4.97), paid for in
GEMM compute rather than bits. **ARCQuant flips to WORSE than plain NVFP4 at
70B** (+0.256 wiki / +0.303 C4 vs NVFP4) despite spending 5.14 bits/wt — its
residual-channel selection, which beat NVFP4 at 1B/3B/8B, stops paying at 70B.

## 4. Scale trend (Δwiki vs each model's own bf16 bridge)

| method | 1B | 3B | 8B | 70B |
|---|---|---|---|---|
| SharQ | +1.63 | +0.57 | +0.54 | **+0.45** |
| ProMix (best standard point) | +1.31 | +0.45 | +0.45 | +0.66 |
| vLLM NVFP4 | +2.49 | +0.81 | +0.73 | +0.61 |
| ARCQuant | +2.07 | +0.73 | +0.67 | **+0.87** |

(1B/3B/8B from `docs/rd_newpeers_accuracy.md` / committed scoreboard; 70B this
round + rd_scale. 70B is the Instruct model — deltas against its own bridge.)

The rd_scale finding sharpens: ProMix's wiki lead shrinks monotonically with
scale and flips at 70B, where it is overtaken not only by plain NVFP4 but by
SharQ, whose delta keeps improving with scale (+1.63 → +0.45). ARCQuant is the
mirror image: competitive at mid scale, degrades at 70B. The two 70B-robust
designs (NVFP4-plain, SharQ) are exactly the ones that spend their quality
budget on finer/decomposed NVFP4 of ALL channels rather than promoting a
channel subset — consistent with rd_scale's observation that outlier structure
flattens at 70B.

## 5. Cite-only peers (NOT run — paper numbers, protocol caveat)

These are base-model (non-Instruct) numbers from the papers' own pipelines
(INT formats, KV4, their eval conventions) — NOT comparable in absolute terms
to our Instruct-70B table; deltas are indicative only.

| method | model | paper FP16 wiki | paper W4A4 wiki | Δ | source |
|---|---|---|---|---|---|
| QuaRot (GPTQ, W4A4KV4) | Llama-3-70B base | 2.86 | 6.66 (5.51 with 128G groups) | +3.80 / +2.65 | QuaRot v2 App. A.8 Tab. 11 |
| SpinQuant (learned R, GPTQ) | Llama-3-70B base | 2.8 | 5.5 (4-4-16 and 4-4-4; RTN variant 4.1) | +2.7 (+1.3 RTN) | SpinQuant Tab. 5 |
| Atom (INT4 + 128 INT8 ch) | Llama-1-65B (no L3-70B in paper) | 3.53 | 3.89 (C4 5.62 → 5.92) | +0.36 | Atom Tab. 2 |

Even with the protocol caveat, the INT rotation methods' Llama-3-70B deltas
(+2.7 to +3.8 wiki) are an order larger than the FP4 field measured above
(+0.45 to +0.87): the 1B/3B/8B ordering (FP4 methods >> INT W4A4) clearly
persists at 70B. **MicroMix skipped** as planned: needs SmoothQuant calib
scales the repo doesn't ship a generator for, and its kernels fail their own
`--validate` on this box's CUDA 13 (see rd_newpeers caveats) — a 70B number
through mis-validating kernels would be meaningless.

## 6. Runtime / memory ledger (single 96 GB GPU each, idle box)

| step | time | GPU peak |
|---|---|---|
| ARCQuant calib (reorder_indices, 128×2048 + select search) | 9.6 min | ~13 GB |
| ARCQuant quantize (80 layers, agemm pack) | 133 s | 41.2 GB resident after |
| ARCQuant wiki / C4 eval | 96 s / 168 s | 53.0 / 60.2 GB |
| SharQ quantize (calibration-free) | 132 s | 39.9 GB resident after |
| SharQ wiki / C4 eval | 100 s / 189 s | 51.7 / 58.9 GB |
| bf16 bridge wiki / C4 (layer-offload, CPU-resident model) | 248 s / 368 s | 11.8 / 19.0 GB |

Both quantized 70B models fit and evaluate on ONE 96 GB GPU (packed weights
~40 GB); only the bf16 bridge needs the CPU-offload path end-to-end.

## 7. Raw data index

`docs/rd_peers70b_data/`: 70b_bf16_bridge.json · 70b_sharq.json ·
70b_arcquant.json · smoke_8b_{fp16,arc,sharq}.json ·
arcquant_utilize_calib.patch (the one calib device patch).
Remote: logs `/home/luofan/rdpeers70b/` (host ICT90), worktree
`/home/luofan/plaquant-peers70b`, ARCQuant 70B saved calib artifacts in
`/home/luofan/baselines/peers/arcquant/ARCQuant/saved/meta-llama-3-70b-instruct_*`.
Harness: `promix/scripts/rd_peers70b.py`.
