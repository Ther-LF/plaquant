# vLLM Integration Feasibility — ProMix Dual-Format (NVFP4+MXFP8) Fused GEMM

**Spike date:** 2026-10-30. **Scope:** feasibility only — vLLM source read on the Pro 6000
(`/home/luofan/baselines/venv`, vLLM **0.32.0**, torch 2.13.0+cu130, runs inside the
`luofan-plaquant` container; this is the exact stack that produced the existing
`/home/luofan/vllm_e2e/{nvfp4,bf16}_{3b,8b}.json` baselines via
`/home/luofan/baselines/scripts/bench_vllm.py`). One throwaway probe was run
(`/home/luofan/vllm_spike/sf_probe.py`, result below). No vLLM code was modified.

---

## 1. VERDICT: custom quant method — NO FORK NEEDED

vLLM 0.32 has a first-class out-of-tree quantization mechanism:

- `@register_quantization_config("promix")` in
  `vllm/model_executor/layers/quantization/__init__.py:61` registers a custom
  `QuantizationConfig` subclass by name, appends it to `QUANTIZATION_METHODS` and the
  platform's supported list. Explicitly documented for "when a quantization method is not
  supported by vllm".
- Loading it: either a tiny pip package exposing a `vllm.general_plugins` entry point
  (`vllm/plugins/__init__.py`, loaded in engine-core AND worker processes — required because
  vLLM spawns workers), or, for benchmarking, simply `import promix_vllm` at the top of the
  bench script **plus** `VLLM_WORKER_MULTIPROC_METHOD`-safe plugin loading (the entry-point
  route is the robust one since the engine runs in a spawned process; the run.log shows
  spawn is forced on this box).
- Selection: our exported checkpoint's `config.json` carries
  `"quantization_config": {"quant_method": "promix", ...}`; `ModelConfig` resolves it through
  `get_quant_config()` (`model_executor/model_loader/weight_utils.py:192`) →
  `get_quantization_config("promix")` → our class. `--quantization promix` also works.

Everything else — model definition, paged attention, scheduler, sampler, torch.compile,
CUDA graphs, the benchmark harness — is reused untouched. **Feasible, no fork.**

## 2. Integration point (how vLLM plugs in a quantized linear)

Lifecycle (all verified in the installed 0.32.0 source):

1. **`QuantizationConfig`** (`quantization/base_config.py:106`) — parsed from checkpoint
   config via `from_config(cls, config: dict)`; `get_config_filenames()`; capability gate
   `get_min_capability()` (we return 120).
2. **`get_quant_method(self, layer, prefix)`** (`base_config.py:200`) — called once per
   module **with the layer's full state-dict prefix** (e.g.
   `model.layers.7.mlp.down_proj`). Returning a `LinearMethodBase` quantizes that layer;
   returning `None`/`UnquantizedLinearMethod` leaves it FP16/BF16. **This is exactly the
   hook for our per-layer error-driven K partition**: the config holds a
   `{layer_name: K_high}` map (from our offline allocator) and hands each layer a
   `PromixLinearMethod(K_high=...)`. lm_head/embeddings stay unquantized by prefix.
3. **`LinearMethodBase.create_weights(layer, input_size_per_partition,
   output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs)`**
   (`layers/linear.py:123`) — registers empty parameters on the layer with
   `weight_loader` attributes; the loader fills them from safetensors by name.
4. **`process_weights_after_loading(layer)`** — called by the model loader
   (`model_loader/utils.py:133`) after all weights land. This is where vLLM's own NVFP4
   path swizzles SF and pads weights — and where we run our load-time prepack.
5. **`apply(layer, x, bias)`** — the per-forward GEMM call.

**vLLM's existing NVFP4 path** (the baseline we compare against, confirmed active in
run.log): checkpoint `quant_method=compressed-tensors` → scheme
`CompressedTensorsW4A4Fp4`
(`quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`) → kernel
abstraction `init_nvfp4_linear_kernel()` → **`FlashInferCutlassNvFp4LinearKernel`**
(`model_executor/kernels/linear/nvfp4/flashinfer.py`; log: "Using
FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM"). Its weight contract:

- `weight_packed`: `(ΣN, K/2)` uint8 (2×FP4/byte), `ModelWeightParameter(input_dim=1, output_dim=0)`
- `weight_scale`: `(ΣN, K/16)` `float8_e4m3fn`, `GroupQuantScaleParameter`
- `weight_global_scale` / `input_global_scale`: `PerTensorScaleParameter`,
  **one entry per merged shard** (len(output_partition_sizes)) — see hard part #2.
- `process_weights_after_loading`: `swizzle_blockscale()` (CUTLASS 128×4 SF atom) + pad;
  `apply_weights`: `scaled_fp4_quant` (activation quant, swizzled SF) →
  `flashinfer_scaled_fp4_mm` → slice/bias. There is also a `fuse_act_quant` compile pass
  (`input_quant_key`/`QuantizedActivation`) that hoists the activation quant out of the
  linear — an optional later optimization for us, not needed for correctness.

## 3. Can our fused kernel drop in? YES — and the layout probe PASSED

Our offline `PackedWeight` (promix/quantize/weight_pack.py): per linear, split along K →
`W_lo_packed (N, K_low/2)` NVFP4 codes + interleaved SFB, `W_hi_bytes (N, K_high)` MXFP8 +
interleaved SFB, per-tensor `g_w` on the epilogue alpha, plus a decode-GEMV pack.

**PROBE RESULT (the one spike probe, `/home/luofan/vllm_spike/sf_probe.py`):** vLLM's
`swizzle_blockscale()` output is **byte-identical (`torch.equal`)** to our kernel's
load-time prepackers `cd.prepack_sf_fp4_b` AND `cd.prepack_sf_fp8_b` on all tested
LLM shapes (qkv/gate/down splits, block-16 and block-32). Both implement the same CUTLASS
`Sm1xx` SF atom (`reshape(M/128,4,32,K/4,4).permute(0,1,4,3,2,5)`). Consequences:

- Our packed-weight SF convention **is** vLLM's standard NVFP4 convention. Zero layout
  translation.
- Checkpoint design: store SF **linear-layout** `(N, K/16)` / `(N, K/32)` (like
  compressed-tensors does) and swizzle in `process_weights_after_loading` — this keeps
  vLLM's shard-merging weight loaders working (the swizzled layout permutes rows within
  128-row tiles and cannot be loaded shard-by-shard).
- Weight-loader fit: all our per-linear tensors are "N-major, packed along K", which is
  exactly the `ModelWeightParameter(input_dim=1, output_dim=0)` pattern the W4A4 scheme
  already uses, so QKV/gate-up merged loading (narrow along output_dim) works unchanged.
  We register: `weight_lo_packed`, `weight_lo_scale`, `weight_hi_bytes`, `weight_hi_scale`,
  `weight_global_scale`, and read `K_high` from config.

So: `PromixLinearMethod.create_weights` registers those params;
`process_weights_after_loading` runs `prepack_sf_*_b` (== vLLM's swizzle, probe-proven) and
any weight padding; `apply` runs our `quant_pack_float` (activation float→codes+interleaved
SFA, one pass) + `fused_gemm_prepacked` (single-launch dual-phase, shared FP32 accumulator,
alpha=g) — the identical kernels already validated at PPL 14.85 in
promix/quantize/mixed_linear.py.

## 4. Minimal-change point

A single out-of-tree package `promix_vllm/` (new code, lives in our repo):

1. `PromixConfig(QuantizationConfig)` + `@register_quantization_config("promix")`.
2. `PromixLinearMethod(LinearMethodBase)` — ~300 lines, mirrors
   `CompressedTensorsW4A4Fp4` + `FlashInferCutlassNvFp4LinearKernel` structure but calls
   our ext ops.
3. Our CUDA exts compiled against the venv's torch 2.13.0+cu130 (current .so's are built
   against the container's torch 2.11 — ABI rebuild required, routine) and registered as
   torch custom ops via `direct_register_custom_op` (`vllm/utils/torch_utils.py:1068`)
   with `fake_impl`s so torch.compile can trace them.
4. An exporter in promix that writes the quantized+rotated model as safetensors with the
   above param names + `config.json` `quantization_config` (incl. the per-layer
   `{layer: K_high}` map and Hadamard/reorder flags).
5. Entry point `vllm.general_plugins` in the package's pyproject.

vLLM itself: **zero lines changed**. Model/attention/sampler/paged-attn/CUDA-graph all
come for free. Day-1 bring-up with `enforce_eager=True`; graphs/compile enabled after the
custom-op fake-impls are in (hard part #3) — and they MUST be enabled before perf numbers,
because the NVFP4/BF16 baselines run `FULL_AND_PIECEWISE` CUDA graphs + inductor.

## 5. ★ ACCURACY-ALIGNMENT GUARANTEE (gate before any perf number)

Principle: both paths consume **the same bytes** and run **the same GEMM kernel**, so the
linear path can be gated at **bit-exact**, and divergence can only enter through
non-linear glue (attention backend, dtype, RoPE impl) — which is gated statistically.

**(a) Same weights, verified at byte level — Gate T0 (torch.equal).**
One offline export (promix → safetensors) feeds BOTH paths: the vLLM run loads it via the
plugin; the promix reference (`promix/quantize/full_network.py` real-kernel path) loads the
SAME file (new loader shim, replacing its in-process pack). Probe: after vLLM
`process_weights_after_loading`, dump every layer's
`{weight_lo_packed, physSFB_lo, weight_hi_bytes, physSFB_hi, g}` and assert
`torch.equal` against the promix `PackedWeight` buffers, all 112+ linears. Any mismatch =
loader/merge bug, hard fail.

**(b) Identical inputs, per-layer — Gate T1 (bit-exact kernel I/O).**
Forward hooks on every `LinearBase` in the vLLM worker (eager mode, batch of fixed
wikitext prompts) capture `(x, y)` per linear. Offline, feed the captured `x` into the
promix `MixedQuantLinear` holding the T0-verified identical pack. Assert
`torch.equal(y_vllm, y_promix)` — same input bytes + same packed weights + same kernel
⇒ bit-exact is the correct bar, not 0.999. Fallback bar if dtype plumbing differs
(bf16 vs fp16 activation entry): cosine ≥ 0.9999 per layer AND a root-caused explanation;
cosine ≥ 0.999 alone is NOT a pass. This isolates: activation quant+pack (incl. down_proj
Hadamard + o_proj reorder inside apply), GEMM, epilogue.

**(c) End-to-end — Gate T2 (logits + PPL).**
  1. **Logits:** same prompt token-ids, vLLM `prompt_logprobs` full-vocab logits (or a
     worker hook on lm_head input) vs promix model forward: top-1 agreement ≥ 99.9% over
     ≥50k tokens, per-token logit cosine ≥ 0.999 (tolerance exists ONLY for the
     attention/norm glue: vLLM FLASH_ATTN-bf16 vs promix HF-fp16).
  2. **PPL:** wikitext-2, the EXACT protocol already in `bench_vllm.py` (non-overlapping
     2048-token chunks, `prompt_logprobs`): vLLM-ProMix PPL vs promix real-kernel
     reference PPL (1B = 14.85). Gate: |Δ| ≤ 0.1. Run the same PPL for the shared-g
     re-reference (hard part #2) — the gate compares against the re-validated reference,
     not the old number.
- **Order is mandatory:** T0 → T1 → T2 → only then perf. Each gate scripted
  (`promix_vllm/tests/gate_t{0,1,2}.py`), re-run on every kernel/exporter change, results
  committed with the perf numbers. A perf table without the three green gates in the same
  commit is invalid by convention.
- **CUDA-graph equivalence probe:** after enabling graphs/compile, re-run T2-logits
  eager-vs-graphed (same engine, `enforce_eager` toggle): bit-equal expected (graph replay
  runs identical kernels; this reproduces the promix prefill-graph experience, cos
  ≥ 0.9999999).

## 6. Hard parts & risk

| # | Item | Risk | Notes |
|---|------|------|-------|
| 1 | SF layout (weight side) | **RETIRED** | Probe: byte-identical to vLLM/CUTLASS convention, fp4 & fp8. Activation-side SFA is produced by our own `quant_pack_float` → internal to our op, no vLLM contract. |
| 2 | **Merged QKV / gate-up vs per-projection per-tensor `g_w`** | **HIGH (accuracy)** | vLLM models hardcode `QKVParallelLinear`/`MergedColumnParallelLinear` = ONE GEMM for q+k+v and gate+up. Our kernel takes one epilogue alpha per GEMM; q/k/v currently have different per-tensor g (E4M3 SF cannot be rescaled post-hoc without renumerics). Resolution: re-quantize offline with **shared g per merged group** (vLLM's own W4A4 path has the same constraint and warns about it), then re-validate the promix reference PPL with shared-g — this re-referenced number becomes the T2 gate target. Fallback: 2–3 kernel calls on sub-slices inside one apply (perf cost at small N). |
| 3 | CUDA-graph / torch.compile of custom ops | MED | Baselines run `FULL_AND_PIECEWISE` graphs + inductor; fair comparison requires ours do too. Need `direct_register_custom_op` + `fake_impl` for ~3 ops (quant_pack_float, fused_gemm_prepacked(+bias), decode GEMV); our ext is already stream-capture-safe (`cudaStreamIsCapturing` guard, workspace cached). Day-1 eager fallback de-risks bring-up. M=1 GEMV-vs-fused dispatch branches on runtime M inside the op — safe per captured graph size. |
| 4 | ResQ rotation | LOW | R1/R2/U are fused into weights **offline** by the exporter (incl. untie-embeddings fix) — vLLM sees an ordinary rotated checkpoint, no hook. Online residue: (i) down_proj input Hadamard (fast_hadamard_transform — must be compiled for the venv py3.12/torch2.13; graph-safe), (ii) o_proj per-head channel reorder (fixed gather). Both live inside `PromixLinearMethod.apply`, keyed by prefix via config flags. (vLLM even ships a QuaRot-style transform subsystem — `compressed_tensors/transform/` with Hadamard + QuTLASS NVFP4 — usable as reference, not needed.) |
| 5 | Per-layer K-split plumbing | LOW | `get_quant_method(layer, prefix)` receives the full prefix → direct lookup in the config's `{layer: K_high}` map. K_high=0 (pure-FP4 layers, rd-alloc budgets) already supported by MixedQuantLinear. |
| 6 | Torch ABI rebuild + packaging | LOW | Rebuild exts against venv torch 2.13.0+cu130 (JIT-load cache or AOT wheel); same nvcc 13.0 / sm_120a flags. |
| 7 | Reference-vs-vLLM glue numerics (attention bf16 flash-attn vs HF fp16) | MED (gate design) | Handled by T1 bit-exact isolation + T2 statistical bars; if T2 misses, T1 proves whether the quant path or the glue is responsible. |

## 7. Effort estimate

~**2–3 weeks** one person (serial), on top of existing validated promix components:

- **P1 — Exporter + re-reference (2–3 d):** safetensors exporter (param names, config.json,
  per-layer K map, shared-g per merged group); re-run promix 1B PPL with shared-g (new
  reference); rebuild exts vs torch 2.13.
- **P2 — Plugin (3–4 d):** PromixConfig/PromixLinearMethod, custom-op registration,
  eager-mode bring-up on 1B.
- **P3 — Alignment gates (2 d):** T0/T1/T2 harness, all green on 1B (eager).
- **P4 — Graphs/compile + perf hygiene (3–5 d):** fake-impls, FULL_AND_PIECEWISE capture,
  graph-equivalence probe, per-shape dispatch (config_dispatch), decode-GEMV path.
- **P5 — Benchmark campaign (1–2 d):** 1B smoke + 3B/8B headline sweeps, three-way table.
- Contingency: hard part #2 fallback or fuse_act_quant work (+3–4 d).

## 8. Comparison harness (QServe-style, same strong stack)

Reuse the EXISTING harness that produced the current baselines —
`/home/luofan/baselines/scripts/bench_vllm.py` + `bench_quant.sh` — unchanged, so
ProMix-in-vLLM / vLLM-NVFP4 (compressed-tensors, FlashInferCutlassNvFp4) / vLLM-BF16 share
identical engine config, protocol, and box:

- **PPL:** wikitext-2, 2048-token non-overlapping chunks (doubles as gate T2b).
- **TTFT:** prefill lens {2048, 4096, 8192} × batch {1, 8}, 1 generated token, median-of-reps.
- **Decode TPOT/throughput:** 32-token prompt → 128 generated, batch {1, 8} (extend sweep
  to {16, 32, 64} for the serving-scale number); tok/s and ms/token.
- **Peak memory + weight bytes** (our 3.22× weight compression shows here).
- **Serving-scale addendum (QServe methodology):** `vllm bench throughput` /
  `vllm bench serve` at fixed in/out lens (e.g. 1024/512) and max sustainable batch, same
  three engines — this is where the 2–3× regime claim is tested, on **8B** (1B is W4A4's
  structurally weakest regime: decode there is host-latency-bound, cf. decode report).
- Models: Llama-3.2-1B (gates, fast iteration) → Llama-3.2-3B / Llama-3-8B (headline),
  matching the already-benchmarked `nvfp4_3b/8b` + `bf16_3b/8b` baselines.

## 9. Phased plan OUTLINE (to be expanded into a full plan)

0. ~~Spike~~ (this doc): custom-method verdict, SF-layout probe PASS.
1. **Export & re-reference:** shared-g requant → exporter → new promix reference PPL.
2. **Plugin bring-up (eager):** load + forward on 1B, T0 green.
3. **Alignment gates:** T1 bit-exact per-layer, T2 logits+PPL green (eager).
4. **Graphs/compile:** custom-op fake-impls → FULL_AND_PIECEWISE → graph-equivalence
   re-gate.
5. **Perf parity work:** per-shape dispatch, decode GEMV, optional fuse_act_quant
   producer fusion.
6. **Benchmark campaign:** three-way 1B/3B/8B sweep, gated table = the deliverable number.
7. (Stretch) KV-cache NVFP4 + lm_head W8 as separately gated ablations.
