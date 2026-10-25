# Related-work baselines on our actual hardware (RTX PRO 6000 Blackwell, SM120)

Date: 2026-10-25. All numbers measured on the target machine, not taken from papers.
Harness, quantized checkpoints and raw JSON live on the GPU host in `/home/luofan/baselines`
(logs in `/home/luofan/baselines/logs`, results in `/home/luofan/baselines/results`).

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, SM 12.0 (compute capability 12,0) |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Host | 8x RTX PRO 6000, shared with other tenants (see noise caveat in §6) |
| Container | `luofan-plaquant` (image `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`), `/home/luofan` bind-mounted |
| Python | 3.12.3 (venv `/home/luofan/baselines/venv`, created with `uv`) |
| vLLM | **0.32.0** (single manylinux wheel, cu130 build; `torch 2.13.0+cu130`) |
| transformers | 5.17.0 (downgraded from 5.19.0 by llm-compressor) |
| compressed-tensors | 0.19.0 |
| llm-compressor | 0.14.0.1 |
| attention backend | `FLASH_ATTN` (FlashAttention 2) in every run |
| sampling | FlashInfer (top-p/top-k); FlashInfer JIT cubins built on first use |

SM120 is fully supported by this vLLM build: BF16, FP8 and NVFP4 all ran, with no
capability fallbacks and no compile errors.

### Setup commands (reproducible)

```bash
# inside container luofan-plaquant, Python 3.12 available as python3.12
mkdir -p /home/luofan/baselines && cd /home/luofan/baselines
uv venv --python 3.12 venv
VIRTUAL_ENV=$PWD/venv uv pip install --python $PWD/venv/bin/python vllm==0.32.0
VIRTUAL_ENV=$PWD/venv uv pip install --python $PWD/venv/bin/python datasets llmcompressor
# models (already present in /home/luofan/hf_cache):
#   unsloth/Llama-3.2-1B-Instruct, unsloth/Llama-3.2-3B, unsloth/llama-3-8b (all BF16, full weights)
```

## 2. Protocol (identical for every baseline and model)

**Accuracy — wikitext-2 PPL**
* `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test` split (the standard test split).
* Model's own HF tokenizer, `add_special_tokens=False`; all rows joined with a blank line.
* Token stream cut into **non-overlapping 2048-token chunks** → 141 chunks, 288,768 tokens
  (288,627 scored: the first token of each chunk has no context; **no sliding window, no
  chunk re-use**).
* NLL of every token from position 1 onward scored via vLLM `prompt_logprobs=1`; PPL = exp(mean NLL).
  0 missing logprob slots in every run.
* No lm-eval tasks in round 1 (PPL only, as agreed).

**Prefill — TTFT**
* Prompt = exact length L (2048 / 4096 / 8192 tokens) of repeated English filler, batch 1 and 8,
  distinct requests, `max_tokens=1`, greedy.
* **Prefix caching disabled**; 1 warmup + **median of 5** wall-clock iterations (CUDA-synchronised).
* The same 8192-token prefill segment is repeated in all 8 batch-1 prompts (one shared prefix);
  batch-8 prompts are identical copies.

**Decode**
* 32-token prompt, `max_tokens=128`, greedy, KV cache on, **median of 5** after warmup.
* tok/s = generated tokens / wall time (prefill included; negligible at 32 prompt tokens).

**Memory / weight size**
* Weight size = bytes on disk of the served checkpoint.
* "Model loading took X GiB" = vLLM's measured weight + non-torch runtime memory.
* Peak GPU memory was sampled from `nvidia-smi` in a background thread during a
  8192-prefill + b8-decode workload. **Caveat:** the peak is dominated by vLLM's KV-cache
  reservation (`gpu_memory_utilization`), so it measures the configured footprint, not a
  minimal one. Disk/mmap weight numbers are the clean cross-baseline comparison.

**Environment settings for every run:** `gpu_memory_utilization` 0.45–0.80 (space-dependent),
`max_model_len` 12288 (8192 for 8B; the 8B config caps `max_position_embeddings` at 8192),
CUDA graphs on (default FULL_AND_PIECEWISE), chunked prefill on, prefix caching off.

## 3. Headline table (best/latest run per config)

PPL is wikitext-2 test; TTFT is batch-1 unless stated; tok/s is batch-1 decode.
8B rows use the quiet-GPU reruns (`awq_r2__llama-3-8b`) where available.

| Model | Baseline | PPL | TTFT 2048 (ms) | TTFT 4096 (ms) | TTFT 8192 (ms) | TTFT 8192 b8 (ms) | Decode b1 (tok/s) | Decode b8 (tok/s) | Weights (GB) | GEMM backend |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B | BF16 | **13.16** | 22 | 37 | 77 | 541 | 435 | 3070 | 2.49 | BF16 cuBLAS/CUTLASS via torch.compile |
| 1B | GPTQ W4A16 | 14.22 | 23 | 42 | 90 | 618 | 663 | 4822 | 1.04 | MarlinLinearKernel |
| 1B | AWQ W4A16 | 15.16 | 22 | 41 | 85 | 610 | 673 | 4856 | 1.04 | MarlinLinearKernel |
| 1B | FP8 W8A8 | 13.30 | 15 | 28 | 59 | 446 | 537 | 4296 | 1.52 | CutlassFP8ScaledMM |
| 1B | **NVFP4 W4A4** | 15.65 | 34 | 35 | **47** | **305** | 645 | 4640 | 1.09 | FlashInferCutlassNvFp4 (fp4_gemm_cutlass_sm120) |
| 3B | BF16 | **7.82** | 47 | 88 | 187 | 1446 | 183 | 1432 | 6.44 | BF16 cuBLAS/CUTLASS |
| 3B | GPTQ W4A16 | 8.20 | 60–88* | 158 | 441–512* | 3473–3546 | 129–208* | 1175–1706* | 2.26 | MarlinLinearKernel |
| 3B | AWQ W4A16 | 8.48 | 49 | 98 | 209 | 1607 | 320 | 2848 | 2.26 | MarlinLinearKernel |
| 3B | **NVFP4 W4A4** | 8.63 | 36 | 44 | **101** | **751** | 334 | 2344 | 2.39 | FlashInferCutlassNvFp4 |
| 8B | BF16 | **6.14** | 93 | 186 | 392 | 3136 | 87 | 703 | 16.07 | BF16 cuBLAS/CUTLASS |
| 8B | GPTQ W4A16 | 6.52 | 103 | 207 | 441 | 3473 | 218 | 1733 | 5.72 | MarlinLinearKernel |
| 8B | AWQ W4A16 | 6.73 | 103 | 207 | 442 | 4144* | 220 | 1725 | 5.72 | MarlinLinearKernel |
| 8B | NVFP4 W4A4 | 6.87 | 42 | 78 | **177** | **1611** | 140 | 1020 | 6.05 | FlashInferCutlassNvFp4 |

\* 3B GPTQ has two runs; the first was contention-polluted. Quiet-GPU rerun values are
60 / 158 / 208 tok/s; the 8192 TTFT point (441 ms) and b8 (3473 ms) come from the first run.
PPL is identical to 2 dp across both runs (8.20).

Perplexity delta vs BF16 (lower is better):

| Model | GPTQ W4A16 | AWQ W4A16 | FP8 W8A8 | NVFP4 W4A4 |
|---|---|---|---|---|
| 1B | +8.1% | +15.2% | +1.1% | +18.9% |
| 3B | +4.9% | +8.4% | – | +10.4% |
| 8B | +6.2% | +9.6% | – | +11.9% |

## 4. What each launcher actually runs (fairness accounting)

| Baseline | Weight quant | **Activation quant** | KV quant | Attention | CUDA graphs | Paged KV | Notes |
|---|---|---|---|---|---|---|---|
| BF16 | none | none | none (bf16) | FlashAttention 2 | yes (FULL+PIECEWISE) | yes | torch.compile on the model; FlashInfer for sampling |
| GPTQ W4A16 | 4-bit, g128, sym | **none (bf16)** | none | FlashAttention 2 | yes | yes | served by vLLM **Marlin** kernel; GPTQ `dampening_frac=0.01`, 256×2048 wikitext calib |
| AWQ W4A16 | 4-bit, g128, sym | **none (bf16)** | none | FlashAttention 2 | yes | yes | AWQ grid search + W4A16; **also served by Marlin** in vLLM 0.32 (AWQ and GPTQ share the Marlin path on this build) |
| FP8 W8A8 | FP8 e4m3 | **FP8, dynamic per-tensor** | none | FlashAttention 2 | yes | yes | `CutlassFP8ScaledMMLinearKernel`; DeepGEMM present but dense path uses CUTLASS |
| NVFP4 W4A4 | NVFP4 e2m1, g16, fp8-e4m3 scales | **NVFP4, static tensor-group quant** (scale per 16 elems) | none | FlashAttention 2 | yes | yes | `FlashInferCutlassNvFp4LinearKernel`, JIT module `fp4_gemm_cutlass_sm120`; `fuse_act_quant` fusion enabled; weight-only variant would route to `B12xNvFp4W4A16` |

Key fairness points for our ProMix comparison:

1. **GPTQ/AWQ are W4A16** — only weights are 4-bit; activations stay BF16. They are *not*
   a W4A4 comparison and will always look better on PPL per bit. The apples-to-apples
   accuracy baseline for ProMix W4A4 is **NVFP4 W4A4**; FP8 W8A8 is the accuracy ceiling reference.
2. **NVFP4 activations are static per-tensor-group** (calibrated min/max per group of 16,
   kv-cache untouched). Our ProMix split is per-channel high/low with a learned rotation —
   different activation treatment, so PPL differences reflect both format and scaling strategy.
3. All baselines use paged attention + continuous batching + CUDA graphs + torch.compile,
   i.e. the same serving stack ours must eventually match. **KV cache is BF16 everywhere**;
   no baseline here uses KV quantization.
4. TTFT includes scheduling + (uncached) prefill; prefix caching is disabled so the numbers
   are not flattered by cross-request reuse.
5. Decode b8 tok/s is aggregate; with 128 output tokens per request, per-request latency
   ≈ 8 × 128 / (aggregate tok/s).
6. **Not measured in round 1:** lm-eval accuracy tasks, KV-quantized baselines, W4A8,
   tensor-parallel, throughput at concurrency > 8, energy.

## 5. Findings

* **SM120 support is real end-to-end.** vLLM 0.32 (cu130 wheel) ran BF16, FP8 W8A8,
  GPTQ/AWQ W4A16 and NVFP4 W4A4 on the RTX PRO 6000 with no patches. The earlier concern
  that NVFP4 activation quant is gated to SM100 does **not** apply to this build's dense path:
  `FlashInferCutlassNvFp4LinearKernel` + `fp4_gemm_cutlass_sm120` executed. The SM100-only
  restriction we saw in source is on the *online MoE* NVFP4 path (and vLLM issue #36094
  concerns MoE backends); Llama is dense, so it does not affect these numbers.
* **NVFP4 W4A4 has the best prefill on SM120.** At 8192 tokens batch 1 it beats BF16 by
  1.6x (1B), 1.9x (3B), 2.2x (8B) and beats GPTQ/AWQ W4A16 at every model size
  (8B: 177 ms vs 441 ms). FP8 is close behind NVFP4 on 1B (59 ms) but uses ~40% more weight memory.
* **W4A16 Marlin wins decode at b1** on 1B (663–673 vs 645 NVFP4 vs 435 BF16) and 8B
  (218–220 vs 140 NVFP4 vs 87 BF16). At b8 AWQ/GPTQ reach 4.8k tok/s (1B) and 1.7k (8B).
  NVFP4 decode is limited by activation quantization overhead (an extra quantize + scale
  per layer) and wins only where the GEMM is large enough — at 3B it already leads BF16
  (334 vs 183 tok/s) but trails W4A16.
* **Accuracy ordering is stable:** BF16 < FP8 ≪ GPTQ < AWQ < NVFP4 in PPL degradation.
  NVFP4 costs +11.9% PPL on 8B (6.14 → 6.87), +10.4% on 3B. That is the number ProMix W4A4
  must beat at the same or better speed.
* **Memory:** 4-bit weights are 1/4 of BF16 on disk (8B: 5.7 GB vs 16.1 GB; 1B: 1.04 vs 2.49 GB).
  NVFP4 weights (g16 + fp8 scales) are ~5-6% larger than INT4 g128 (6.05 vs 5.72 GB at 8B).
  FP8 is 1/2 of BF16 (1.52 GB at 1B).

## 6. Caveats / measurement quality

* **Shared GPU host.** Eight GPUs are used by several tenants; several runs were visibly
  affected (e.g. AWQ-8B first pass TTFT@8192 b1 963 ms vs 442 ms on a quiet GPU; 3B GPTQ
  decode 129→208 tok/s between runs; 1B FP8 decode 225→537 tok/s). The table uses the
  cleanest run per config; treat TTFT and decode as ±10–25% unless marked. The **8B b8
  TTFT points are the noisiest**: BF16 re-confirmed at 3136 ms, but AWQ-8B b8 came out
  3947 / 4144 ms in two quiet passes vs 3476 ms under load, and its decode is *faster*
  than BF16 — treat that one cell as unstable pending a dedicated re-run.
* **Memory peak** from `nvidia-smi` reflects vLLM's KV reservation (`gpu_memory_utilization`),
  not a minimal footprint; use the weight-size column for cross-baseline comparison.
* **PPL absolute values** are higher than the ~14.7 quoted for the ProMix pipeline on
  Llama-3.2-1B because that number uses a different protocol (sliding/windowed eval over
  a shorter or differently tokenized stream). Cross-baseline comparisons in this table are
  self-consistent (same tokenizer, same 141 chunks, same scoring).
* **8B context:** `max_position_embeddings=8192`; TTFT@8192+b1 needed `max_model_len=12288`
  with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` so prompt+1 fits. No NaNs observed.
* **FlashInfer JIT:** first model init takes 3–5 min (autotune + cubin compile). Warm runs
  were used; a fresh container will pay this once.

## 7. Reproduction

Scripts (on the GPU host, in `/home/luofan/baselines/scripts`; copies committed in
`docs/baselines_scripts/`):

| File | Purpose |
|---|---|
| `bench_vllm.py` | unified PPL / TTFT / decode / memory harness (main entry) |
| `bench_quant.sh` | wraps `bench_vllm.py` for a quantized checkpoint; waits for free memory |
| `quantize.py` | llm-compressor one-shot: `gptq_w4a16`, `awq_w4a16`, `nvfp4`, `fp8` |
| `quant_gen.sh` | name-a-checkpoint wrapper around `quantize.py` |
| `mem_probe.py` | nvidia-smi sampling of peak memory for one config |
| `mem_sweep.sh` | all configs sequentially on one GPU |
| `ttft_extra.py` | fills in the 8192 TTFT point when the model config caps at 8192 |
| `waiter.sh` | block until a GPU has N MiB free and <25% util |
| `harvest.py` | build the markdown table from `results/*.json` |

```bash
# 1) BF16 baseline (repeat per model)
cd /home/luofan/baselines
source venv/bin/activate
HF_HOME=/home/luofan/hf_cache CUDA_VISIBLE_DEVICES=2 \
  python scripts/bench_vllm.py \
  --model /home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/ \
  --tag bf16__llama-3.2-1b --out results/bf16__llama-3.2-1b.json

# 2) quantize (one scheme per run; outputs to quant/<repo>-<scheme>)
CUDA_VISIBLE_DEVICES=2 bash scripts/quant_gen.sh nvfp4 \
  "/home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/"
CUDA_VISIBLE_DEVICES=2 bash scripts/quant_gen.sh gptq_w4a16 \
  "/home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-3B/snapshots/*/"
CUDA_VISIBLE_DEVICES=2 bash scripts/quant_gen.sh awq_w4a16 \
  "/home/luofan/hf_cache/hub/models--unsloth--llama-3-8b/snapshots/*/" --nsamples 96

# 3) benchmark a quantized checkpoint (auto-detected quantization; pass the method to pin it)
MML=8192 GMU=0.45 NEED=45000 bash scripts/bench_quant.sh 1 nvfp4__llama-3-8b \
  /home/luofan/baselines/quant/unsloth--llama-3-8b-nvfp4

# 4) fill 8192 TTFT where the config caps at 8192 (8B)
CUDA_VISIBLE_DEVICES=2 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  python scripts/ttft_extra.py --model quant/unsloth--llama-3-8b-gptq_w4a16 \
  --merge-into results/gptq_w4a16__llama-3-8b.json --quantization compressed-tensors

# 5) memory sweep + table
bash scripts/mem_sweep.sh 1
python scripts/harvest.py
```

Exact per-run logs: `/home/luofan/baselines/logs/<tag>.log` (vLLM kernel-selection lines included);
raw results: `/home/luofan/baselines/results/<tag>.json`.

## 8. Toolkit status on SM120 (round 1)

| Toolkit | Status | Notes |
|---|---|---|
| vLLM 0.32.0 (cu130 wheel) | **works** | BF16 / FP8 / GPTQ / AWQ / NVFP4 all ran |
| llm-compressor 0.14.0.1 | **works** | GPTQ, AWQ, NVFP4, FP8 recipes all produced loadable compressed-tensors checkpoints |
| AutoGPTQ / AutoAWQ | **not used** | no need: vLLM serves compressed-tensors; prequantized TheBloke repos return HTTP 401 (removed/gated) |
| FlashInfer 0.7.0.post1 | works, JIT | no `flashinfer-cubin`/`jit-cache` wheel installed → builds `fp4_gemm_cutlass_sm120` at first use (minutes) |
