# rd-scale (paper E2): extending the model ladder to 70B

**Branch:** `feat/rd-scale` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120, 96 GB), host ICT90, container `luofan-plaquant`, GPU lane 0-2
· **Dates:** 2026-11-03/05 · **Status:** COMPLETE

**Mission.** The paper's accuracy ladder (1B/3B/8B/13B) tops out at 13B. E2 adds
a ≥32B point: feasibility → {bf16 bridge, uniform-1/8, measured-1/8, vLLM-NVFP4}
wiki+C4 PPL → serving-as-feasible. Key paper questions: (1) does ProMix's
accuracy lead over NVFP4 hold at 70B-scale, (2) does the allocation gain stay
~0 as the 13B trend predicts.

---

## 1. Feasibility (STEP 1 verdicts)

### Qwen3-32B — NO-GO (architecture), model IS cached

`/home/luofan/latches/models/Qwen3-32B` (66 GB, full safetensors) is on disk —
no download needed. But its config breaks the pipeline's structural assumption:

| field | Qwen3-32B | pipeline requirement |
|---|---|---|
| hidden_size | 5120 | — |
| num_attention_heads | 64 | — |
| head_dim | **128** | **hidden // heads = 80** |
| intermediate_size | 25600 | needs K×2^n Hadamard: 25600 = 100×256 or 200×128 — **no had100/had200 in hadamard_data.json** (had12/28/108 only) |

This is exactly the Qwen3-4B blocker documented in `docs/rd_qwen_report.md`
("all rotation/ptq sites derive head_dim = hidden//heads … plugin asserts"):
q/k/v out-features are heads×head_dim = 8192 ≠ hidden 5120, so the R2/value
rotation and o_proj reorder math as coded do not close. Qwen3-8B passed only
because 4096/32 = 128. Unblocking 32B = (a) thread `config.head_dim` through
basis/rotation/ptq (bounded but risky refactor), + (b) generate a had100
(order-100 Hadamard exists; Paley-type construction like the had108 precedent).
Deferred — 70B covers the "scale" ask on the pipeline-native Llama arch.

### Llama-3-70B — GO (already on disk, no download)

`/home/luofan/latches/models/Meta-Llama-3-70B-Instruct` (150 GB, 30/30 shards +
index + tokenizer) is already on the box. This supersedes the "download
Llama-3.3-70B (~140 GB)" plan — which would NOT have fit anyway: `/` is 99%
full (86 GB free at mission start, 45 GB after the NVFP4 export) and `/data`
(6 TB free) is root-owned, not writable by luofan.

Architecture check (config.json): hidden 8192 / 64 heads → head_dim 128 =
hidden//heads ✓; GQA 8 KV heads (same as Llama-3-8B, supported) ✓;
intermediate 28672 = 28×1024 → had28 already in hadamard_data.json ✓;
untied embeddings ✓. Ring512 substrate: 8192 % 512 = 0 ✓.

NOTE: the only 70B on disk is the **Instruct** model. All six 70B points below
(bridge, NVFP4, 4 ProMix points) are measured on the SAME model, so every
comparison is internally fair; absolute PPLs are not comparable to a base-model
70B (instruct models have higher wiki PPL).

### Memory math (503 GB host RAM, 96 GB/GPU) — all verified in practice

| stage | host RAM | GPU | verdict |
|---|---|---|---|
| Step-0 basis (fp32 load, layer-by-layer hooks) | 280 GB model + 43 GB covariances (2×80×8192² fp32) | inps+outs 512×2048×8192 ×2 + layer + fp64 cov temps ≈ 75-84 GB | ✓ ran 2026-11-03/04, 80 layers × ~725 s = 16.1 h, GPU1 |
| rotation fuse | 140 GB fp16 model CPU-resident; per-weight GPU round-trip | ≤ 9 GB transient | ✓ ~22 min |
| GPTQ (layer-by-layer, 128×2048 calib) | 140 GB | inps/outs ~8.6 GB + Hessian 3.3 GB + layer | ✓ ~95 min |
| PPL eval (evaluate_ppl layer-offload) | 140 GB | inps/outs ≤ 17 GB + layer + logits | ✓ wiki ~6 min idle (~37 min contended) |
| zero-shot (lm-eval) | — | needs WHOLE fake-quant model GPU-resident = 140 GB fp16 | ✗ → **PPL-only at 70B** (limitation) |
| llm-compressor NVFP4 | low (weights straight to GPU) | sharded across 2 GPUs (device_map=balanced) | ✓ 26 min |

One fix was required and is the only code change to shared files:
`load_model()` whole-model `.cuda()` OOMs at 70B →
**`PROMIX_LOAD_DEVICE=cpu` opt-in** (promix/models/loader.py; default
unchanged). Every downstream build step was already device-safe for a
CPU-resident model (rotation fusers move one weight at a time; GPTQ and
evaluate_ppl are layer-offloading).

## 2. Protocol

Driver: `promix/scripts/rd_scale.py` — rd_acctune.py semantics with every
full-model `.to(DEV)` removed (layer-offload eval; stats hooks driven through
`evaluate_ppl`'s offload loop at batch_size=1 so token order — and hence the
allocation — is identical). Committed protocol knobs unchanged: GPTQ
nsamples=128 / percdamp 0.01 / no act_order; act_mse_steps 3/0;
weight_mse_steps 3; collect_tokens 4096; calib_samples 16; KV16; wikitext-2
token-pooled 2048-chunk PPL; C4 = QuaRot 256×2048 validation windows. NVFP4 =
llm-compressor `scheme="NVFP4"` (W4A4, g16, lm_head ignored), 256×2048 wikitext
calib, scored in vLLM 0.32 via prompt_logprobs (same chunking/scoring
convention); C4 windows replicated verbatim from `promix/eval/data.py:get_c4`.

**1B regression gate (passed before any 70B number):** rd_scale reproduces the
committed acctune points digit-for-digit — ring512 uniform **14.8557**
(== 14.855731), ring512 measured **14.2392** (== 14.239173), alloc sel_rings
**72/576** (== committed; the offload stats path yields the identical
allocation). Raw: `docs/rd_scale_data/gate_1b_ring512_{uniform,measured}.json`.

## 3. Results — 70B accuracy (wiki / C4 PPL, same protocol as the committed ladder)

| point | bits/wt | wikitext | C4 | Δwiki vs bf16 |
|---|---|---|---|---|
| bf16 bridge | 16 | **5.3159** | **9.2036** | — |
| vLLM NVFP4 W4A4 | 4.50 | **5.9279** | **9.7461** | +0.612 (+11.5%) |
| ProMix dense uniform-1/8 | 4.97 | 6.0749 | 9.9087 | +0.759 (+14.3%) |
| ProMix dense measured-1/8 | 4.97 | 6.0975 | 9.9660 | +0.782 |
| ProMix ring512 uniform-1/8 | 4.97 | 6.0738 | 9.8980 | +0.758 |
| ProMix ring512 measured-1/8 | 4.97 | **5.9732** | **9.8917** | +0.657 (+12.4%) |
| ProMix ring512 measured + up2 + GPTQ512 (shipped tuned recipe) | 4.97 | 5.9829 | **9.8549** | +0.667 |

All ProMix points: avg_fp8 = 0.1250 exactly (778240/6225920 channels), GPTQ
frozen-scale protocol, o_proj held at uniform 1/8. Raw JSONs in
`docs/rd_scale_data/`.

**Headline (honest):** at 70B the ProMix accuracy lead over vLLM-NVFP4 does
NOT hold under the standard recipe: best standard point (ring512+measured)
trails NVFP4 by **+0.045 wiki / +0.146 C4** while spending 0.47 more bits/wt.
Compare the smaller models (wiki, ProMix shipped vs NVFP4): 1B 14.56 vs 15.65
(**−1.09**), 3B 8.30 vs 8.63 (**−0.33**), 8B 6.60 vs 6.87 (**−0.27**), 70B
5.97 vs 5.93 (**+0.045**). The lead shrinks monotonically with scale and flips
sign at 70B. The tuned-recipe cell (up2 + GPTQ 512, the acctune-promoted
common core) does NOT recover it: wiki 5.9829 (+0.010 vs standard measured,
within noise) / C4 9.8549 (−0.037, a real but small C4 gain) — ProMix's best
70B cells remain +0.045 wiki (standard) / +0.109 C4 (tuned) behind NVFP4.
The acctune levers were tuned on 1B/3B/8B and transfer only on C4 here.

## 4. Scaling-trend verdict (allocation gain, uniform − measured, wiki)

| substrate | 1B | 3B | 8B | 13B | **70B** |
|---|---|---|---|---|---|
| dense (committed ladder) | −0.077 | −0.030 | +0.019 | −0.0006 | **+0.023** |
| ring512 | −0.617 | −0.217 | −0.008 | — | **−0.101** (C4 −0.006) |

(negative = measured wins)

1. **Dense substrate: the scaling law is CONFIRMED at 70B** — measured-MSE
   allocation is dead break-even with uniform (+0.023 wiki / +0.057 C4, same
   sign and size as the 8B/13B noise band). Uniform stays optimal at scale on
   the dense substrate.
2. **Ring512 substrate: NON-monotonic surprise** — the wiki allocation gain
   re-emerges at 70B (−0.101) after vanishing at 8B (−0.008), and
   ring512+measured is the only ProMix point below 6.0, while the ring-vs-dense
   rotation cost ε is FREE at 70B (ring512 uniform 6.0738 vs dense uniform
   6.0749). C4 gain is ~0 (−0.006), so the wiki win is real but does not
   transfer to C4 — report both, don't oversell.
3. Quantization difficulty does NOT keep shrinking with size on this ladder:
   Δwiki(uniform, dense) = 1B +1.39 / 3B +0.48 / 8B +0.45 / 13B +0.14 / 70B
   +0.76. The 70B point is an Instruct model (the 13B/8B/3B points are base),
   which plausibly explains the uptick — flagged, not resolved.

## 5. Serving (STEP 3)

Decision taken: quantized-70B serving is reported via the **vLLM NVFP4 export
on ONE 96 GB GPU vs vLLM BF16 at TP=2** (the honest caveat applies: different
parallelism — BF16 70B cannot fit one GPU, and the ProMix serving plugin has no
TP). ProMix-70B serving export is additionally **disk-blocked** (needs ~44 GB;
45 GB free) — accuracy-only for ProMix at 70B, serving deferred.

vLLM 0.32, max_model_len 8192, same bench (`bench_vllm.py`), median of 5:

| config | GPUs | TTFT b1 2048/4096/8064 (s) | TTFT b8 (s) | decode tok/s b1 / b8 |
|---|---|---|---|---|
| NVFP4 W4A4 (40 GB) | **1** | 0.253 / 0.535 / 1.254 | 2.00 / 4.32 / 10.66 | **32.6 / 258.4** |
| BF16 (140 GB) | 2 (TP=2) | 0.542 / 1.082 / 2.389 | 4.17 / 8.59 / 19.18 | 19.4 / 149.0 |

Quantized 70B on a single GPU beats BF16 on two GPUs outright: TTFT ~1.9-2.1×
faster, decode 1.68× (b1) / 1.73× (b8) — i.e. **~3.4-4.2× per-GPU serving
efficiency**, plus it is the only way to serve 70B on one 96 GB card at all.
Raw: `docs/rd_scale_data/{70b_nvfp4_serve,70b_bf16_tp2_serve}.json`.

## 6. Limitations / notes

- 70B zero-shot (lm-eval) not run: the fake-quant model must be GPU-resident
  (140 GB fp16 > 96 GB). Would need multi-GPU device_map through the wrapper
  stack — untested, out of E2 scope.
- Single calibration seed (seed 0), as in the committed ladder; the E6
  error-bar protocol was not extended to 70B (each 70B point costs ~4 h).
- 70B = Meta-Llama-3-**Instruct** (only 70B on disk; downloads blocked by
  disk). All 70B rows share the model, so comparisons stand.
- Runtime ledger (for planning): basis 16.1 h; per quant point ≈ rotate 22 min
  + GPTQ 95 min (128 calib) + wiki 6-37 min + C4 10-64 min (contention-
  dependent); NVFP4 quantize 26 min; each PPL-only eval of a built model ~15
  min on an idle GPU.

## 7. Raw data index

`docs/rd_scale_data/`: gate_1b_ring512_{uniform,measured}.json ·
70b_bf16_bridge.json · 70b_nvfp4_ppl.json · 70b_{ring512,dense}_{uniform,measured}.json ·
70b_ring512_measured_up2_g512.json (tuned recipe) · 70b_nvfp4_serve.json ·
70b_bf16_tp2_serve.json. Remote logs/scripts: `/home/luofan/rdscale/` (host
ICT90), worktree `/home/luofan/plaquant-scale` (branch feat/rd-scale).
