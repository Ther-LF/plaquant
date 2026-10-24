# Blackwell (SM100/SM120) NVFP4/FP8 Production Deployment Survey

> 2026-10-24. Read-only survey for the ProMix path. Scope: **how TensorRT-LLM, vLLM/FlashInfer,
> llm-compressor and ModelOpt actually arrange an NVFP4/FP8 Llama so it beats FP16/FP8 end-to-end**,
> and which of those mechanisms ProMix lacks.
>
> Code evidence was pulled from `main` branches on 2026-10-24 (TRT-LLM, vLLM, FlashInfer,
> llm-compressor, TensorRT-Model-Optimizer). File paths below are relative to each upstream repo
> root. Performance numbers are cited with their baseline; nothing here is our own measurement
> except where marked **[ours]**.

---

## 0. TL;DR

1. **Production stacks keep the residual stream in BF16/FP16, and quantize only at GEMM inputs.**
   TRT-LLM's fused norm kernel returns `output: [M, N] - pre-norm output (input + residual), same
   dtype as input` next to the FP4 tensor (`cpp/tensorrt_llm/thop/fusedAddRMSNormQuant.cpp`).
   There is no FP8/FP4 residual stream anywhere in the dense decode path.
2. **The quantizer is folded into the producer's epilogue**, not run as its own op:
   `add + RMSNorm + quantize` produces the next GEMM's A in FP4/FP8
   (`tensorrt_llm/_torch/modules/rms_norm.py`), `gate_up GEMM + SwiGLU + quantize` produces
   down_proj's A in FP4 (`tensorrt_llm/_torch/modules/gated_mlp.py`), and attention can emit FP4
   directly for o_proj (`tensorrt_llm/_torch/attention/attention.py`). vLLM does the same with
   `fuse_norm_quant`, `fuse_act_quant`, `fuse_attn_quant`, `fuse_allreduce_rms`
   (`vllm/compilation/passes/fusion/*`, docs table in `docs.vllm.ai/en/latest/design/fusions`).
3. **Static, calibrated activation scales let the norm kernel bake the consumer's scale in.**
   TRT-LLM attaches `gate_up_proj.input_scale` to the preceding RMSNorm (`nvfp4_scale`) and the
   norm emits FP4 + block scales directly; the Linear then skips its own quantize
   (`NVFP4LinearMethod._input_prepare` fast-path on `Fp4QuantizedTensor`).
4. **The one big fusion we cannot copy on SM120 is TRT-LLM's.** Its fused
   `add+RMSNorm+NVFP4-quantize` epilogue is gated `100 <= sm_version < 120`
   (`rms_norm.py`, `if not (100 <= sm_version < 120) ...: self.is_nvfp4 = False`), and its
   hardware matrix marks NVFP4 KV cache as SM100/103-only. On SM120 TRT-LLM falls back to the
   generic norm and "let the downstream linear layer handle FP4 quantization" — i.e. SM120
   production still pays a **separate quantize**. vLLM's FlashInfer `fused_add_rms_norm_nvfp4_quant`
   and `silu_and_mul_nvfp4_quant` are not gated off on SM120 (FlashInfer ships
   `gen_fp4_quantization_sm120_module` / `sm120f`), which is the better reference for us.
5. **The published wins are 2.4× vs BF16 at high concurrency** for a 32B model with native FP4
   GEMM (Qwen3-32B, RTX PRO 6000, vLLM) — with only +0.41 PPL, and **only because it used the
   FP4 tensor-core path**; the same 4-bit format on a W4A16 Marlin fallback got 1.34×. At
   batch-1 decode, the win is weight bytes + KV capacity, so it needs ≥7–8B weights and/or long
   context. For our 1B it is not there yet **[ours]**, consistent with our
   `docs/decode_throughput_report.md`.

Our gap is therefore **not "quantize more Linears"** — it is **where the quantize happens
(epilogue vs standalone kernel), what it is fused with (norm, SwiGLU, attention, Hadamard), and
what else moves to low precision (KV cache, lm_head)**. Ranked list in §7.

---

## 1. TensorRT-LLM NVFP4 (SM100 path, and what changes on SM120)

### 1.1 The layer graph

`modeling_llama.py` (`LlamaDecoderLayer.__init__`, forward):

- `post_attention_layernorm = RMSNorm(..., quantize_type="nvfp4")`, then
  `post_attention_layernorm.nvfp4_scale = self.mlp.gate_up_proj.input_scale` (line ~874).
- `next_layer_layernorm.nvfp4_scale = self.next_attn.qkv_proj.input_scale` (line ~949), i.e. the
  norm before layer *n+1* carries layer *n+1*'s qkv static scale.
- With TP, the fusion op is `AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4` for both pre- and
  post-MLP edges (falls back to `RESIDUAL_RMS_NORM` when not quantized).

`rms_norm.py::RMSNorm.forward` → `_fused_nvfp4_quant(hidden_states, residual, sf_scale, ...)`:

- residual + add is applied in high precision, **the residual output keeps the input dtype**
  (BF16/FP16);
- two kernels: `torch.ops.trtllm.fused_add_rms_norm_quant` (warp-specialized, N∈[2048,16384],
  M≥4096 — measured at N=7168 on GB200, "ws first overtakes at M=4096, ~6% faster; M<=3072
  favors reduce_fusion") and `fused_add_rmsnorm_fp4_quantize` (one CTA per row);
- outputs `(Fp4QuantizedTensor, residual_out)`; the FP4 tensor stores the **consumer's static
  input scale** already applied (`sf_scale` arg) and, if needed, the BF16 normed value for a MoE
  gate (`return_hp_output`).

C++ contract (`fusedAddRMSNormQuant.cpp`):

```
// input: [M, N] fp16/bf16 ; residual: [M, N] fp16/bf16 ; gamma: [N]
// Returns:
//   normed_output: [M, N/8]  - FP4 quantized normalized output (uint32 packed)
//   output:        [M, N]    - pre-norm output (input + residual), same dtype as input
//   sf_out:                  - FP8 scale factors, swizzled
//   high_precision_normed_output: [M, N] (optional, for MoE gate)
// NOTE: requires SM90 (Hopper) or SM100 (Blackwell); N in [2048, 16384]
```

This is the single most important architectural fact for question 4: **the residual stream is never
quantized; only the GEMM A operand is.**

### 1.2 The Linear side

`linear.py::NVFP4LinearMethod`:

- weights are packed FP4 + FP8-E4M3 per-16-block scales; `weight_scale_2` is the global weight
  scale; `alpha = input_scale * weight_scale_2` is precomputed for static mode or derived from
  `amax(input)` for dynamic mode;
- `create_weights` reserves `kv_scales`/`inv_kv_scales` (3 floats) — **KV cache scales ride on the
  qkv Linear**;
- `pre_quant_scale` is "computed as an average and merged with the LayerNorm for QKV and Gate/Up
  projection layers when possible … only for o_proj and down_proj" (AWQ smoothing, folded into the
  norm for the two LayerNorm-fed GEMMs);
- `_input_prepare` accepts a pre-quantized `Fp4QuantizedTensor` and returns immediately **with no
  re-quantize**; a plain BF16 tensor triggers `torch.ops.trtllm.fp4_quantize(input, input_scale,
  16, False)` — i.e. dynamic per-block quantize in the GEMM prologue, only when the producer could
  not emit FP4;
- `apply` calls `torch.ops.trtllm.nvfp4_gemm(...)`, a unified dispatcher that auto-tunes across
  `cutlass, cublaslt, cutedsl, cuda_core, marlin` and caches the best (runner, tactic) per shape.
  Bias is fused in the GEMM when N is unpadded.

### 1.3 Fused MLP: gate/up GEMM + SwiGLU → FP4 for down_proj

`gated_mlp.py::GatedMLP`:

- `_can_fuse_gate_up_swiglu_fp4out()` → `_fused_gate_up_swiglu(x, fp4_out=True)` calls
  `torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(act_fp4, W, act_sf,
  weight_scale, alpha, global_sf=self.down_proj.input_scale)`;
- comment: "FC2's input_scale serves as norm_const for SFC quantization … eliminating the
  bf16→fp4 requantization between FC1 and FC2";
- gated to `_FP4OUT_MIN_M = 128` (below that the SFC epilogue can write OOB); under
  `torch.compile` it is always on because the token dim is symbolic.
- the BF16-output version (`cute_dsl_nvfp4_dense_gemm_swiglu_blackwell`) is the fallback.

### 1.4 Attention → o_proj

`attention/attention.py::Attention._use_quantize_output()` returns True iff o_proj is NVFP4 and
`is_static_nvfp4_input_eligible(o_proj)`; the attention kernel then returns
`Fp4QuantizedTensor(output, output_sf)` (line ~1092) and o_proj consumes it without quantizing.
`TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT` is the kill switch. A `TwoFP4` variant
(`NVFP4ARCLinearMethod`, `fp4_quantize_with_reorder_residual`) also exists for fused
attention-output quantization with residual reorder.

### 1.5 What stays high precision

RMSNorm math, residual add, RoPE, softmax/attention math, SiLU **when the fused epilogue is not
used**. The fused SwiGLU epilogue computes silu(gate)*up on FP4-GEMM *outputs* (BF16 accumulators)
then quantizes — the nonlinearity itself is not a 4-bit op. Note `gated_mlp.py` says the fused
CuteDSL NVFP4 SwiGLU epilogue applies **no clamp**, so checkpoints relying on swiglu clamping fall
back.

### 1.6 KV cache and hardware gating (SM120 caveat)

`docs/source/features/quantization.md` hardware matrix:

| | NVFP4 | MXFP4 | FP8 per-tensor | FP8 block-scaling | FP8 rowwise | FP8 KV | **NVFP4 KV** | W4A16 AWQ |
|---|---|---|---|---|---|---|---|---|
| Blackwell (sm120) | Y | Y | Y | . | . | Y | **.** | . |
| Blackwell (sm100/103) | Y | Y | Y | Y | . | Y | **Y** | Y |

NVFP4 KV is enabled by `KvCacheConfig(dtype='nvfp4')` and **requires offline ModelOpt
quantization**; TRT-LLM currently only supports it with FP8 weight/activation (`--quant fp8
--kv_cache_quant nvfp4`). NVFP4 KV halves KV vs FP8 and <1% accuracy loss vs BF16/FP8 on
LiveCodeBench / MMLU-PRO / MBPP / Ruler-64K; values are dequantized to FP8 before attention math
(NVIDIA blog, 2025-12-08). On SM120 it is **not available in TRT-LLM**.

Also gated off on SM120: the fused norm+NVFP4 epilogue itself (`100 <= sm_version < 120`), FP8
block-scaling and rowwise GEMMs. So **TRT-LLM's most copyable mechanism is exactly the one SM120
does not get**; FlashInfer's SM120 FP4 module and vLLM's fusions are the SM120-relevant references.

### 1.7 How they report speedups

`docs/source/performance/perf-overview.md` reports **total output tokens/s** from `trtllm-bench`
(max-load queueing, `--kv_cache_free_gpu_mem_fraction 0.90–0.95`) for ModelOpt-quantized
checkpoints, per (ISL, OSL) pair, on B200/GB200/H200. Example: Llama-3.3-70B FP4, TP=1, B200:
10,614 tok/s at 128/128, 6,984 at 500/2000, 1,330 at 2048/128. The FP8 tables are H100/H200 — i.e.
**TRT-LLM's headline FP4-vs-FP8 comparison is cross-GPU-generation**, which is why independent
same-hardware comparisons (§5) matter. No public SM120 NVFP4 numbers in the repo docs.

---

## 2. vLLM / FlashInfer on Blackwell

### 2.1 NVFP4 model support and the GEMM path

`vllm/model_executor/layers/quantization/modelopt.py`:

- `ModelOptNvFp4Config` accepts `NVFP4` (W4A4, "static fp4 weight + dynamic fp4 activation (has a
  static global input scale). alpha = weight_gs * input_gs") and `W4A16_NVFP4` (FP4 weight,
  **no activation quant → Marlin**);
- a checkpoint labeled `NVFP4` whose `config_groups[*].input_activations` are all null is silently
  rerouted to `W4A16_NVFP4` — this is the "weights-only FP4" trap;
- the generic `ModelOptLinearMethod` resolves a `QuantSpec(weight=kNvfp4Static,
  activation=kNvfp4Dynamic)` and selects a kernel; `expose_input_quant_key(layer, kernel)` publishes
  whether the kernel can consume a **pre-quantized** activation.
- `vllm/model_executor/kernels/linear/nvfp4/cutlass.py::CutlassNvFp4LinearKernel.apply_weights`
  calls `scaled_fp4_quant` then `cutlass_scaled_fp4_mm` — i.e. **quantize then GEMM as two ops in
  the generic path**. The FlashInfer/TRTLLM-Gen kernels fuse activation quant on the fly.

FlashInfer (`flashinfer/quantization/fp4_quantization.py`) offers the primitives to avoid the
separate op:

| API | What it does |
|---|---|
| `nvfp4_quantize(a, a_global_sf, per_token_activation=..., backend="cuda"/"cute-dsl"/"cake")` | BF16/FP16/FP8 → packed FP4 + swizzled SF; SM120 module generated (`gen_fp4_quantization_sm120_module`, `sm120f`, `sm121`) |
| `silu_and_mul_nvfp4_quantize(input, global_scale)` | SwiGLU **and** FP4 quantize in one CuTe-DSL kernel, SM100+ |
| `nvfp4_quantize_paged_kv_cache(k, v, ...)` | paged K/V → NVFP4 for TRTLLM-Gen MHA (SM100 layout) |
| `nvfp4_kv_quantize(input, global_scale)` | KV → FP4 + linear block scales, SM100+ |
| `add_rmsnorm_quant` (via `torch.ops.flashinfer` / vLLM's `_flashinfer_fused_add_rms_norm_nvfp4_quant`) | add + RMSNorm + NVFP4 in one call |

### 2.2 Fusions vLLM applies (torch.compile Inductor passes)

`docs.vllm.ai/en/latest/design/fusions` (2026-10):

| Fusion | Flag | Fuses | Default | Reported e2e | SM100 support |
|---|---|---|---|---|---|
| AllReduce + RMSNorm | `fuse_allreduce_rms` | AR → RMSNorm(+residual) (→quant) | O2 (TP>1) | 5–20% | FP16/BF16, FP8 static, **NVFP4** |
| Attention + Quant | `fuse_attn_quant` | attention out → FP8/NVFP4 | **off** | 3–7% | FP8 static, **NVFP4** |
| RMSNorm + Quant | `fuse_norm_quant` | RMSNorm(+residual) → FP8/FP4 | O1 (conditional) | 1–4% | FP8 static/per-token/per-group; **NVFP4 is the FlashInfer-only variant** |
| SiLU+Mul + Quant | `fuse_act_quant` | SwiGLU → FP8/FP4 | O1 (conditional) | 1–4% | FP8 static; **NVFP4** when `silu_and_mul_nvfp4_quant` is built (`FUSED_OPS[kNvfp4Dynamic]`, `act_quant_fusion.py`) |

Mechanism: a `QuantizedActivation` dataclass (`layers/fusion/quant_activation.py`) carries
`(data, scale, orig_dtype, orig_shape, quant_key)`; producers ask
`get_input_quant_key(consumer)` and emit the quantized payload, consumers validate via
`as_quantized_activation(x, expected_key)` and skip their own quantize. It has a
`weak_ref()` for CUDA-graph replay. This is architecturally the same contract as TRT-LLM's
`Fp4QuantizedTensor`.

`rms_quant_fusion.py::FusedAddRMSNormNvfp4QuantPattern` registers
`_flashinfer_fused_add_rms_norm_nvfp4_quant(result, result_block_scale, residual, input, weight,
input_global_scale, block_scale_unswizzled, is_sf_swizzled_layout, epsilon)` — one FlashInfer call
for add+norm+FP4. The attn fusion (`attn_quant_fusion.py::AttnNvfp4QuantPattern`) passes an
`output_scale` into `Attention.forward` so the attention kernel writes FP4 directly.

### 2.3 CUDA graphs / decode

vLLM captures piecewise CUDA graphs for decode; the fusion passes are designed to be
graph-capturable (the `weak_ref` comment above), and the ATTN fusion explicitly requires "the full
model graph to be visible (Inductor partition or `splitting_ops=[]`)". Our own experience matches:
graphing decode erased ~100% of host dispatch (15.26 → 4.75 ms/token at 1B **[ours]**,
`docs/decode_cudagraph_report.md`), but it cannot remove GPU-side quantize work — which is exactly
why the epilogue fusion matters more than graph capture once you are graphed.

### 2.4 Published speedups and baselines

- **Qwen3-32B, single RTX PRO 6000 Blackwell (SM120), vLLM, AWQ variants** (jarvislabs.ai,
  2026-10-05): NVFP4 **3,079 tok/s** (2.41× BF16's 1,276), FP8 2,076 (1.63×), MXFP8 2,211 (1.73×),
  MXFP4 1,715 (1.34×, Marlin W4A16 fallback). At concurrency 360: NVFP4 2,755 tok/s / 1,133 ms
  mean TTFT vs BF16 1,263 tok/s / 8,606 ms (7.6× TTFT). Accuracy: BF16 PPL 10.21, FP8 +0.03,
  NVFP4 **+0.41**. Baseline is **same-stack BF16, same GPU**. Note the MXFP4 result: a 4-bit
  *format* without the 4-bit *kernel* is slower than FP8 — **the kernel matters more than the bit
  width**.
- vLLM GB200 WideEP (DeepSeek-V3/R1, 2K/2K): 26.2K prefill TPGS, 10.1K decode TPGS with NVFP4
  GEMM/MoE + FP8 GEMM + RoPE+Quant+Q-write and Concat-K fusions; baseline is the previous H200
  deployment (cross-GPU), so it is not an NVFP4-vs-FP8 number.
- InferenceX (SemiAnalysis) DeepSeek-R1 leaderboard: B200 fp4 (SGLang) 4,610 tok/s/GPU at 50
  tok/s/user vs GB200 fp8 (Dynamo SGLang) 2,697 — different engines and hardware, so indicative
  only.

---

## 3. ModelOpt / llm-compressor recipes: what gets quantized

### 3.1 llm-compressor (`examples/quantization_w4a4_fp4/`)

```python
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
oneshot(model=model, dataset="perfectblend", splits="train[:512]",
        recipe=recipe, max_seq_length=2048, num_calibration_samples=20)
model.save_pretrained(SAVE_DIR, save_compressed=True)   # compressed-tensors
```

README: "nvfp4 quantization generates per-tensor global scales and per-group (size 16) local
quantization scales for the weights, as well as **per-tensor global scales for the activations**.
Per-group local activation quantization scales are generated **dynamically during inference**."
And: "**if running inference on a machine that is < SM100, vLLM will not run activation
quantization, only weight-only quantization.**"

So the recipe is: **all Linears except lm_head; weights W4 + dynamic A4; 20 samples to calibrate
global scales only.** No attention, no KV cache, no lm_head. A `quantization_w4a16_fp4` example
exists for weight-only. Accuracy land:

- llm-compressor issue #2922, Llama-3.1-8B-Instruct, vLLM 0.25.1: GSM8K-platinum-cot recovery
  vs FP16 = **93.5% RTN, 94.2% GPTQ, 95.1% AWQ, 95.6% best GPTQ (damp05 + perf-blend
  calibration)**; average over 5 tasks 94.2% (GPTQ) → 95.6%.
- ModelOpt/QAD work (2026) exists for models where PTQ W4A4 is not enough (Qwen3.6 W4A4 QAD).

### 3.2 ModelOpt

`NVFP4_DEFAULT_CFG` is loaded from `configs/ptq/presets/model/nvfp4.yaml` in
TensorRT-Model-Optimizer; the preset is the `QuantizeConfig` used by
`mtq.quantize(model, NVFP4_DEFAULT_CFG)` and by TRT-LLM's `huggingface_example.sh --quant nvfp4`.
The deployment contract is a `hf_quant_config.json` with `quant_algo: NVFP4`, `group_size: 16`,
`kv_cache_quant_algo`, and `exclude_modules` — **KV cache quantization is a separate algo field**
(`kv_cache_quant_algo: FP8` or `NVFP4`), not part of the W4A4 recipe. TRT-LLM's NVFP4-KV path
requires `--quant fp8 --kv_cache_quant nvfp4` today.

---

## 4. The architectural question, answered directly

**Do production stacks keep the residual stream in FP16/BF16 and quantize only at GEMM inputs, or
keep it in FP8?**

- **Residual stream: BF16/FP16.** TRT-LLM's fused norm returns `input + residual` in the input
  dtype; `Fp4QuantizedTensor` optionally carries `unquantized_hidden_states` (BF16) for MoE gates
  and other consumers. vLLM's `QuantizedActivation` carries `orig_dtype` and `can be dropped when
  another consumer branch needs the original activation` (`get_input_quant_key` "hides the key
  when another consumer branch needs the original activation"). Both stacks treat low precision as
  a **GEMM-operand format**, not a storage format for the stream.
- **RMSNorm, RoPE, softmax, attention math: high precision.** The norm kernel does the add and
  normalization in FP32/BF16 and only the *output* is quantized. Attention consumes FP8/BF16
  Q/K/V and, in TRT-LLM, dequantizes NVFP4 KV to FP8 before the context matmul.
- **SiLU: high precision inside the fused epilogue.** FP4 GEMM → BF16 accumulators → silu*up →
  quantize. The nonlinearity is never done in 4/8-bit.
- **What *is* low precision everywhere:** weight storage (FP4), GEMM A operand (FP4/FP8),
  optionally KV cache (FP8/NVFP4 — separate opt-in), and `pre_quant_scale`/`alpha` folded into
  norm weights or a static scalar.

Implication: **you cannot get an e2e win by making the residual stream or non-Linear ops low
precision.** The win comes from (a) GEMM math/bytes, (b) not paying a separate quantize kernel,
(c) KV capacity at long context/high concurrency. That matches our own measurements: at 1B
prefill M=2048 the fused GEMM saves only 0.34 ms of Linear time, while the extra FP32 Hadamard
costs 2.81 ms **[ours]**, `docs/prefill_overhead_report.md`.

---

## 5. When does NVFP4 actually beat FP16/FP8 end-to-end?

There is no vendor-published threshold table, but the evidence lines up on five conditions:

1. **Native FP4 tensor-core kernels, not a W4A16 fallback.** Qwen3-32B SM120: NVFP4 2.41× vs BF16,
   while the same 4-bit format on Marlin (BF16 activations) got 1.34× — worse than FP8's 1.63×.
   Also seen on SM120: vLLM's NVFP4 MoE backend is not available for some models
   and AWQ INT4 sometimes edges out NVFP4 in single-stream decode (`docs/quant_e2e_methodology_survey.md`).
2. **Model size ≥ ~7–8B for batch-1 decode.** The batch-1 win is "weight bytes per token vs HBM
   bandwidth". At 1B (~0.6 GB at 4.5 bits **[ours]**) the per-Linear GPU work is <5 µs and
   launch/occupancy dominates; our graphed 1B decode is 3.66 ms/token for 112 GEMVs, 11.3× off the
   0.419 ms/token bandwidth ceiling, i.e. **not bandwidth-bound** (`docs/decode_cudagraph_report.md`).
   At 8B, FP16 weights ≈ 16 GB vs ≈ 4.5 GB at NVFP4 — the 3.5× byte reduction finally maps to
   decode time.
3. **Prefill / compute-bound, large M.** NVFP4's 2× FP4-vs-FP8 tensor-core rate only helps when the
   GEMM is compute-bound. Our 1B prefill crosses FP16 at M=2048 (1.07×) and loses at M≥4096
   (0.95×/0.88×) mainly because of the FP32 Hadamard, not the GEMM **[ours]**.
4. **Long context / high concurrency.** NVFP4 KV halves KV vs FP8: BF16 Qwen3-32B had 93K tokens
   of KV capacity vs 208–267K for quantized variants; at concurrency 360 that moved the TTFT cliff
   from 8.6 s to 1.1 s. NVIDIA reports up to 3× TTFT improvement and ~2× context budget from
   NVFP4 KV alone, with <1% accuracy loss. **This benefit is independent of whether the weights
   are NVFP4.** For a 1B at short context, KV is tiny; the benefit is ~0.
5. **Accuracy budget.** Uniform W4A4 PTQ on Llama-3.1-8B recovers 94–96% of FP16 on GSM8K/average
   task scores; Qwen3-32B pays +0.41 PPL. Larger models and MoE tolerate it better; smaller models
   need GPTQ/AWQ/QAD or a mixed-precision scheme like ours (8-bit high-variance slice + 4-bit
   main), which we already have (PPL 14.72/15.21 at 1B, `docs/resq_eval_results.md`).

**So: NVFP4 beats FP16 end-to-end when the model is ≥7–8B, the workload is compute-bound prefill
or long-context/high-concurrency decode, the kernels are native FP4, and the quantizer is fused
into producers.** On a 1B at short context / batch 1, the honest ceiling is low — our measured
0.70× vs graphed FP16 at decode and 1.07× at M=2048 prefill are in line with that.

---

## 6. Mechanisms table

Legend: **Saves** = what the mechanism actually removes from the critical path. **Applicable?** =
to ProMix (7 per-layer Linears, INT4+INT8 mixed, separate activation pack kernel, FP32 online
Hadamard on down_proj, no attention/KV quant, SM120 / RTX PRO 6000).

| # | Mechanism | Where in code | What it saves | Applicable to us? |
|---|---|---|---|---|
| 1 | **add + RMSNorm + quantize → next GEMM's A** | TRT-LLM `modules/rms_norm.py::_fused_nvfp4_quant`, `cpp/.../thop/fusedAddRMSNormQuant.cpp`; vLLM `compilation/passes/fusion/rms_quant_fusion.py::FusedAddRMSNormNvfp4QuantPattern`; FlashInfer `add_rmsnorm_quant` | One kernel launch + one full read+write of `[M,H]` per norm; removes the standalone activation-quant kernel entirely | **Yes, highest value.** We do norm (FP16) then a separate pack kernel. Note TRT-LLM's kernel is SM100-only; vLLM/FlashInfer are the SM120 references |
| 2 | **gate/up GEMM + SwiGLU + quantize → down_proj's A (FP4 out)** | TRT-LLM `modules/gated_mlp.py::_fused_gate_up_swiglu(fp4_out=True)`, `cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell`; FlashInfer `silu_and_mul_nvfp4_quantize`; vLLM `act_quant_fusion.py` | Removes the BF16 round-trip of the intermediate `[M, 2/7·I]` tensor and the down_proj input quantize | **Yes.** Our down_proj is exactly this edge; we currently write BF16 h, run FP32 Hadamard, then pack |
| 3 | **Attention output → FP4/FP8 for o_proj** | TRT-LLM `attention/attention.py::_use_quantize_output` (returns `Fp4QuantizedTensor`); vLLM `attn_quant_fusion.py::AttnNvfp4QuantPattern` (`output_scale` into Attention) | One kernel + one `[M,H]` read/write for o_proj's A; 3–7% e2e in vLLM | **Yes, later**; our o_proj input is currently FP16 + a column reorder |
| 4 | **Static calibrated activation scale + pre-quantized-activation contract** | TRT-LLM `Fp4QuantizedTensor` / `module.input_scale` / `alpha`; vLLM `QuantizedActivation` + `expose_input_quant_key` / `as_quantized_activation` | Lets the producer emit the consumer's quantized format directly; skips the consumer's quantize and its amax reduction | **Partially.** We use per-token dynamic scales, so a producer cannot know the exact FP8/FP4 scale; but a *contract* (producer emits codes + scales) is still implementable and would let us fuse 1–3 |
| 5 | **AllReduce + RMSNorm (+quant) fused (TP)** | TRT-LLM `AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4`; vLLM `fuse_allreduce_rms` (5–20% e2e) | Collective + norm + quant in one comm kernel | **No today** (we run TP=1 on one 96 GB GPU). Relevant only if we scale out |
| 6 | **KV cache quant (FP8, then NVFP4) + paged KV quant kernel** | TRT-LLM `KvCacheConfig(dtype='fp8'/'nvfp4')`, `doc/source/features/quantization.md`; vLLM `kv_cache.py`, FlashInfer `nvfp4_quantize_paged_kv_cache` / `nvfp4_kv_quantize` | 2× (FP8) / 4× (NVFP4) KV bytes → more context/ concurrency, lower decode bandwidth, higher cache hit → up to 3× TTFT | **Yes, but not first.** ~0 benefit at 1B/short context; material at 8B, ≥32K context, or high concurrency. SM120: TRT-LLM **does not** support NVFP4 KV; vLLM does via FlashInfer (PR #59791, closed, `--kv-cache-dtype nvfp4`, fa2 prefill + XQA decode, head sizes 128/256/512) |
| 7 | **Fused FP8/NVFP4 GEMM with epilogue bias, autotuned backend** | TRT-LLM `torch.ops.trtllm.nvfp4_gemm` (cutlass/cublaslt/cutedsl/cuda_core/marlin autotune); vLLM `cutlass_scaled_fp4_mm`, FlashInfer TRTLLM-Gen | Removes separate bias add, picks best tactic per M; avoids the W4A16 fallback trap | **Partially.** Our fused INT4+INT8 GEMM is already one launch; we still do `.to(fp16)` + bias + reshape outside the kernel and do not autotune per-M |
| 8 | **lm_head / embedding quantized** | llm-compressor and vLLM/ModelOpt **ignore `lm_head`** by default; our `kernels/mixed_gemm_sm120/lmhead_mxfp8_ext.cu` exists | lm_head is ~1 GB at 8B (vocab 128K × H 4096, 8-bit) → ~3% of batch-1 weight traffic | **Yes, small.** More relevant at 8B; at 1B the lm_head GEMM is 1/16 of traffic |
| 9 | **CUDA graphs / piecewise graphs for decode** | vLLM piecewise graphs + `weak_ref` for captured quantized activations; TRT-LLM graphs | Host dispatch (~100% erased in our measurement; 1B decode 0.64× → 2.06× vs FP16 eager) | **Already done [ours]**: `docs/decode_cudagraph_report.md`, `docs/prefill_overhead_report.md` |
| 10 | **QAT/QAD fallback when PTQ W4A4 is not enough** | ModelOpt QAD (e.g. Qwen3.6 W4A4 QAD, 2026-09) | Recovers accuracy so NVFP4 is deployable at all | **No** (our mixed 8/4-bit already preserves PPL; QAT is a fallback, not a fusion) |

---

## 7. Ranked list of fusions we lack (expected e2e impact, Llama 1B / 8B, SM120)

All 1B numbers are against our measured baselines on RTX PRO 6000 SM120:
prefill FP16 20.94 / 41.14 / 86.12 ms at M=2048/4096/8192; real graphed 19.52 / 43.42 / 98.29 ms
(pack 3.18/5.60/9.97 ms, GEMM 8.10/11.27/19.44 ms, FP32 Hadamard 2.81/9.78/20.71 ms) **[ours]**.

### Rank 1 — Fold activation quantize into the producing epilogue (norm+quant for qkv/gate_up, SwiGLU+quant for down_proj)

- **Evidence:** TRT-LLM `rms_norm.py` `_fused_nvfp4_quant` + `gated_mlp.py`
  `_fused_gate_up_swiglu(fp4_out=True)`; vLLM `fuse_norm_quant` (1–4%), `fuse_act_quant` (1–4%);
  FlashInfer `add_rmsnorm_quant`, `silu_and_mul_nvfp4_quantize`.
- **What we pay now:** a separate `quant_pack` kernel per Linear (3.18 ms at M=2048, 9.97 ms at
  M=8192, 112 Linears) plus a full read/write of `x`.
- **Expected:** removes ~50–70% of the pack bucket (the remaining part is unavoidable compute),
  i.e. **~1.5–2 ms @ M=2048 (+8–10%)**, **~4–6 ms @ M=4096 (+10–14%)**, **~6–8 ms @ M=8192
  (+6–8%)**. At M=2048 that moves 19.52 → ~17.6 ms, i.e. 1.07× → **~1.19× vs FP16**.
  At 8B the Linear share of prefill time is larger (7 × H·I GEMMs vs 16 layers, H=4096, I=14336),
  so the same removal is an **Amdahl extrapolation of ~8–12% prefill e2e**, not a measurement; do
  not quote it as one without an 8B run.
- **SM120 caveat:** TRT-LLM's fused norm is SM100-only; implement against FlashInfer's SM120
  FP4 module or our own epilogue. vLLM's `fuse_norm_quant` FP8 patterns are gated to SM100, but the
  NVFP4 `FusedAddRMSNormNvfp4QuantPattern` is registered when
  `current_platform.has_device_capability(100)` and FlashInfer's `add_rmsnorm_fp4quant` imports —
  a >= 100 check that **includes SM120**, so this is the SM120-viable reference path.

### Rank 2 — Kill or fuse the FP32 online Hadamard on down_proj

- **Evidence:** TRT-LLM fuses SwiGLU + FP4-out in the GEMM epilogue, so **there is no standalone
  rotation pass at all**; QServe (A100) folds activation quant into the preceding LN and is the
  closest published analogue of "rotation/quant fused with producer"
  (`docs/quant_e2e_methodology_survey.md`).
- **What we pay now:** `matmul_hadU_cuda(x.float(), ...)` in `promix/inference/real_forward.py`
  — 2.81 / 9.78 / 20.71 ms at M=2048/4096/8192, the single largest non-GEMM term, and the reason
  we lose to FP16 at M≥4096. Note this is **not arithmetic-bound**: one Hadamard pass over
  `[M, I]` × 16 layers is only ~0.018% of the model's GEMM FLOPs at 1B (≈0.009% at 8B), so the
  measured 9.78 ms at M=4096 is a **BF16→FP32 cast + two full `[M, I]` round-trips through HBM**
  (and the standalone kernel launch), not the butterfly math. FP16 Hadamard fails our numerics
  gate **[ours]** (`docs/prefill_overhead_report.md` §3), so the fix is not "cast it down" — it is
  "never materialize a separate FP32 copy".
- **Expected:** produce the SwiGLU output in a form that is already rotated (or fold the rotation
  into the fused epilogue) and write FP4/FP8 codes once. That removes one full `[M, I]` FP32
  write + read per layer: **~2–3 ms @ M=2048**, **~5–7 ms @ M=4096**, **~10–12 ms @ M=8192**.
  Combined with Rank 1: 43.42 → ~33–35 ms at M=4096 (**~1.15–1.2× vs FP16**) and 98.29 → ~79–82 ms
  at M=8192 (**~1.05–1.10×**).
- **How to fuse it:** the Hadamard must stay on the *input* side (it is `H·x`); folding it into the
  down_proj weight would instead compute `x·(H^T W)`, which is not the same transform and also
  breaks the offline INT4/INT8 packing. The correct placement is the **producer epilogue**: GEMM →
  BF16 acc → SwiGLU → Hadamard over each `[M, I]` tile → quantize → write codes. That is one
  kernel and one write of `h`. The FP32 accumulation precision must be preserved (our FP16 gate
  failed), i.e. keep the rotation math FP32 but never write an FP32 tensor to HBM. If a full-tile
  Hadamard in the epilogue is impractical, the fallback is a single fused `hadamard+quantize`
  kernel that reads BF16 `h` once and writes only codes (no FP32 materialization).
- **Why ranked 2 not 1:** it is down_proj-only (1 of 7 Linears) but its absolute cost is larger
  than the pack bucket. Do it together with Rank 1's `silu_and_mul + Hadamard + quant` epilogue.

### Rank 3 — KV cache quantization (FP8 first, NVFP4 later) + attention-side quant

- **Evidence:** TRT-LLM `KvCacheConfig(dtype='fp8'|'nvfp4')`; vLLM `--kv-cache-dtype` with
  FlashInfer `nvfp4_quantize_paged_kv_cache`; NVIDIA NVFP4-KV blog (2× KV capacity vs FP8, <1%
  accuracy loss, up to 3× TTFT via cache hits); Qwen3-32B SM120 data (93K → 208–267K KV tokens,
  TTFT 8.6 s → 1.1 s at c=360).
- **What we pay now:** nothing — we don't touch attention or KV (`promix/quantize/kv_quant.py`
  exists but is wired only into `ptq.py`/GPTQ, not the real forward).
- **Expected:** at **1B, batch 1, short context: ~0** (KV is tens of MB; decode is
  launch-bound, not bandwidth-bound — our own report says the 4-bit weight win only materializes
  once fixed costs are amortized). At **1B, 32K+ context or batch ≫1: 5–15%** via more KV capacity
  and lower decode read traffic. At **8B: decode becomes genuinely weight+KV bandwidth-bound**;
  FP8 KV is the safer first step (SM120 supports it in both stacks), NVFP4 KV only via vLLM/
  FlashInfer and still experimental on SM120 (vLLM PR #59791 closed; TRT-LLM marks it SM100-only).
- **Caveat:** `KV-dtype nvfp4` on SM120 has a documented accuracy cliff on small models
  (Qwen3-0.6B GSM8K 0.37 → 0.18–0.21 in PR #59791) — test on 1B before shipping.

### Rank 4 — Fuse the epilogue (`.to(fp16)`, bias, reshape, o_proj column reorder) into the GEMM; per-M kernel selection

- **Evidence:** TRT-LLM fuses bias in `nvfp4_gemm` and autotunes per M across 5 backends;
  vLLM/FlashInfer select TRTLLM-Gen vs cutlass vs Marlin per shape. We launch the GEMM, then
  `D.to(fp16) + bias` and a Python-level `x[..., order]` for o_proj (the latter forced a harness
  fix under graph capture).
- **Expected:** small on GPU time (<1 ms each) but removes graph-capture hazards and the
  o_proj reorder pass; **~1–2% e2e**, more at high M. Also: our prefill kernel uses a 128×128 tile
  at M=1 (0.45× FP16) — already fixed by the decode-GEMV **[ours]**, but the fused kernel should
  select a low-M variant automatically.

### Rank 5 — Quantize lm_head / keep embeddings out of the hot path

- **Evidence:** vLLM/llm-compressor **ignore `lm_head`** by default (it is numerically sensitive);
  TRT-LLM's NVFP4 path also leaves it out. Our `lmhead_mxfp8_ext.cu` exists but is not in the
  default forward.
- **Expected:** at 1B, lm_head is 1/16 of weight traffic and the logits GEMM is bandwidth-trivial
  — **<1% e2e**; at 8B with a 128K vocab it is ~1 GB of weights (~3% at batch-1) and **2–3%**.
  Low priority; do it only after 1–3, and verify accuracy separately (lm_head quantization is the
  usual accuracy footgun).

### Not missing / already correct

- Residual stream in FP16, RMSNorm/RoPE/softmax/SiLU high precision — matches production.
- Mixed 8/4-bit (our differentiator vs uniform NVFP4) — accuracy is *better* than uniform W4A4 at
  the same average bit width; do not "simplify" to uniform FP4.
- CUDA graphs for prefill and decode — done and measured.

---

## 8. SM120-specific bottom line

- NVFP4 **GEMM** is supported on SM120 by both TRT-LLM and vLLM/FlashInfer (FlashInfer has
  `gen_fp4_quantization_sm120_module`, `sm120f`, `sm121`).
- TRT-LLM's **fused RMSNorm→NVFP4 epilogue is SM100-only** (`100 <= sm_version < 120`); on SM120 it
  falls back to "generic norm + downstream linear quantizes", i.e. the very pattern we have.
  So **do not take TRT-LLM's SM100 fused-norm code as the SM120 plan** — use FlashInfer's
  `add_rmsnorm_quant` / `silu_and_mul_nvfp4_quantize` (SM100+, compiled for SM120) or write the
  epilogue in our own fused kernel.
- **NVFP4 KV cache is not available in TRT-LLM on SM120**; vLLM has an experimental FlashInfer
  path (fa2 prefill + XQA decode, head sizes 128/256/512 only).
- Reported same-hardware SM120 wins are ~2.4× vs BF16 for 32B at high concurrency with native FP4
  GEMM; for our 1B the honest targets are **~1.2× prefill at M=2048–4096** with Ranks 1–2, and
  **long-context/high-batch KV wins** with Rank 3; the 3–4× headlines belong to 7B+ decode.

---

## 9. Sources (fetched 2026-10-24)

- TRT-LLM: `tensorrt_llm/_torch/modules/rms_norm.py`, `linear.py`, `gated_mlp.py`,
  `tensorrt_llm/_torch/attention/attention.py`, `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`,
  `cpp/tensorrt_llm/thop/fusedAddRMSNormQuant.cpp`,
  `docs/source/features/quantization.md`, `docs/source/features/kv-cache-compression.md`,
  `docs/source/performance/perf-overview.md`.
- vLLM: `vllm/model_executor/layers/quantization/modelopt.py`,
  `kv_cache.py`, `fusion/quant_activation.py`,
  `kernels/linear/nvfp4/cutlass.py`,
  `vllm/compilation/passes/fusion/{rms_quant_fusion,act_quant_fusion,attn_quant_fusion}.py`;
  docs `docs.vllm.ai/en/latest/design/fusions`.
- FlashInfer: `flashinfer/quantization/fp4_quantization.py` (`nvfp4_quantize`,
  `silu_and_mul_nvfp4_quantize`, `nvfp4_quantize_paged_kv_cache`, `nvfp4_kv_quantize`,
  `gen_fp4_quantization_sm120_module`).
- llm-compressor: `examples/quantization_w4a4_fp4/{README.md,llama3_example.py}`,
  issue #2922 (Llama-3.1-8B NVFP4 accuracy).
- TensorRT-Model-Optimizer: `NVFP4_DEFAULT_CFG` (`configs/ptq/presets/model/nvfp4.yaml`).
- NVIDIA: "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference";
  "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache" (2025-12-08);
  vLLM blog "Driving vLLM WideEP … on Blackwell" (2026-02-03).
- Independent SM120: jarvislabs.ai "Four Blackwell Formats, One Model: Quantizing Qwen3-32B on
  RTX PRO 6000" (2026-10-05); vLLM PR #59791 (NVFP4 KV on SM120/SM121); SemiAnalysis InferenceX.
- Ours: `docs/prefill_overhead_report.md`, `docs/decode_throughput_report.md`,
  `docs/decode_cudagraph_report.md`, `docs/quant_e2e_methodology_survey.md`,
  `docs/resq_eval_results.md`, `kernels/mixed_gemm_sm120/docs/c1_nvfp4_spec.md`.
