# rd-scale (paper E2): extending the model ladder to 70B

**Branch:** `feat/rd-scale` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000
Blackwell (SM120, 96 GB), host ICT90, container `luofan-plaquant`, GPU lane 0-2
· **Date started:** 2026-11-03 · **Status:** IN PROGRESS

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
full (86 GB free) and `/data` (6 TB free) is root-owned, not writable by luofan.

Architecture check (config.json): hidden 8192 / 64 heads → head_dim 128 =
hidden//heads ✓; GQA 8 KV heads (same as Llama-3-8B, supported) ✓;
intermediate 28672 = 28×1024 → had28 already in hadamard_data.json ✓;
untied embeddings ✓. Ring512 substrate: 8192 % 512 = 0 ✓.

### Memory math (503 GB host RAM, 96 GB/GPU)

| stage | host RAM | GPU | verdict |
|---|---|---|---|
| Step-0 basis (fp32 load, layer-by-layer hooks) | 280 GB model + 43 GB covariances (2×80×8192² fp32) | inps+outs 512×2048×8192 fp16 ×2 ≈ 34 GB + layer 3.4 GB | ✓ (must run WITHOUT a concurrent 70B job) |
| rotation fuse | 140 GB fp16 model (CPU-resident; per-weight GPU matmul round-trip) | ≤ 4 GB transient | ✓ |
| GPTQ (layer-by-layer, 128×2048 calib) | 140 GB | inps/outs 128×2048×8192 fp16 ×2 ≈ 8.6 GB + Hessian 28672² fp32 3.3 GB + layer | ✓ |
| PPL eval (evaluate_ppl layer-offload) | 140 GB | inps/outs ≤ 17 GB (C4 256×2048) + layer + logits | ✓ |
| zero-shot (lm-eval) | — | needs WHOLE fake-quant model GPU-resident = 140 GB fp16 | ✗ → **PPL-only at 70B** (limitation) |
| llm-compressor NVFP4 | ~180 GB | sharded across GPU lane 0-2 (288 GB) | ✓ |

Runtime calibration from the 13B chain logs: basis 300 s/layer × 40 = 3h20m →
70B ≈ 768 s/layer × 80 ≈ **17 h**; 13B quant build 1437 s, wiki eval 707 s,
C4 1105 s → 70B ≈ 2 h build + ~1 h wiki + ~1.7 h C4 per point.

### Serving call (STEP 3)

Quantized 70B (~44 GB at 4.97 bit/wt) fits one 96 GB GPU; BF16 70B needs TP=2
which the ProMix serving plugin does not implement. Per the E2 spec, the honest
options are "quantized single-GPU vs vLLM BF16 TP=2 (different parallelism)" or
accuracy-only. Decision recorded here once accuracy lands.

---

## 2. Protocol

Driver: `promix/scripts/rd_scale.py` — rd_acctune.py semantics with every
full-model `.to(DEV)` removed (layer-offload eval; stats hooks driven through
`evaluate_ppl`'s offload loop at batch_size=1 so token order — and hence the
allocation — is identical). Committed protocol knobs unchanged: GPTQ
nsamples=128 / percdamp 0.01 / no act_order; act_mse_steps 3/0;
weight_mse_steps 3; collect_tokens 4096; calib_samples 16; KV16; wikitext-2
token-pooled 2048-chunk PPL; C4 = QuaRot 256×2048 validation windows.

**1B regression gate (must pass before any 70B number is trusted):**
rd_scale reproduces the committed acctune points digit-for-digit —
ring512 uniform 14.8557 / ring512 measured 14.2392 (+ sel_rings 72/576).

- [x] gate PASSED 2026-11-03: uniform **14.8557** (== committed 14.855731),
  measured **14.2392** (== committed 14.239173), alloc sel_rings **72/576**
  (== committed) — the offload stats path yields the identical allocation.
  Raw: `docs/rd_scale_data/gate_1b_ring512_{uniform,measured}.json`.

## 3. Results (70B)

| point | wikitext | C4 | notes |
|---|---|---|---|
| bf16 bridge | **5.3159** | **9.2036** | 803 s, layer-offload eval; Meta-Llama-3-70B-**Instruct** |
| ring512 uniform-1/8 | — | — | |
| ring512 measured-1/8 | — | — | |
| dense uniform-1/8 (committed-ladder substrate) | — | — | stretch |
| dense measured-1/8 | — | — | stretch |
| vLLM NVFP4 W4A4 | **5.9279** | **9.7461** | llm-compressor quantize (GPUs 1-2, device_map=balanced) + vLLM prompt_logprobs; scripts: `/home/luofan/rdscale/{quantize_70b_nvfp4,ppl_70b_nvfp4}.py` |

## 4. Scaling verdict

(pending)
