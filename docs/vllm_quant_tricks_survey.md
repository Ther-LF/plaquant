# vLLM 0.32.0 quantization-tricks survey → ProMix gap checklist

**Purpose.** Per the directive "their tricks we must have; tricks they don't have we must
also have": an exhaustive inventory of the optimization techniques used by vLLM 0.32.0's
*other* quantization methods — runtime AND offline/load-time weight processing — mapped
onto ProMix's current vLLM integration as a HAVE / MISSING / N-A checklist, prioritized
by expected gain. Feeds the running vllm-perf loop queue (`docs/vllm_perf_round1.md` §4).

**Method.** Read vLLM 0.32.0 (`__commit_id__ g5df63a720`) source at
`/home/luofan/baselines/venv/lib/python3.12/site-packages/vllm` (the exact install our
three-way benchmarks run on). All `file:line` refs below are relative to that `vllm/`
root. ProMix side = `promix/vllm_integ/{plugin,ops,kernels,export}.py`,
`docs/vllm_integ_phase23.md`, `docs/vllm_perf_round1.md`, `docs/perf_round2.md`.

**Architecture note (0.32 refactor).** Quant methods are now thin *schemes* that
delegate both load-time repack and the GEMM to pluggable kernel classes under
`model_executor/kernels/linear/` chosen from per-platform priority lists
(`model_executor/kernels/linear/__init__.py:443` fp8, `:523` mixed-precision, `:574`
nvfp4, `:607` mxfp4). Every kernel class owns a `process_weights_after_loading`
(offline repack) and `apply_weights` (runtime). `--linear-backend` overrides
(`__init__.py:273`), `VLLM_DISABLED_KERNELS` filters.

---

## 1. ModelOpt NVFP4 (the direct rival — W4A4, 1.7-1.9x faster than us e2e)

| # | technique | mechanism | refs |
|---|---|---|---|
| 1.1 | Alpha precompute at load | `layer.alpha = input_global_scale * weight_global_scale` folded into one GEMM scalar; `input_global_scale_inv` precomputed for runtime act quant | `quantization/modelopt.py:2373,2048,2090` |
| 1.2 | Block-scale swizzle at load | `swizzle_blockscale`: pad M→128/K→4, reshape `(B,M/128,4,32,K/4,4)` + permute → tcgen05 MMA SF layout; no runtime gather | `quantization/utils/nvfp4_utils.py:13` |
| 1.3 | Weight/act padding to tile multiples | pad N,K→32 (64 cute-dsl) at load; pad act K at runtime, slice output N back | `nvfp4_utils.py:61,97,104,116` |
| 1.4 | Kernel registry + per-arch dispatch | priority FlashInfer CuteDSL → FlashInfer CUTLASS → B12x (SM120!) → CUTLASS → W4A16 CuteDSL → Marlin → TRTLLM → cuDNN → fbgemm → torch → emulation; screened by `is_supported()/can_implement()` per layer at load | `kernels/linear/__init__.py:574,1128` |
| 1.5 | One fused act-quant op | `scaled_fp4_quant` custom op quantizes + writes swizzled block scales in one kernel; `.out` variant for compile buffer reuse; small-batch (m≤32) 8×4 SF layout for TRTLLM | `_custom_ops.py:1513,1561` |
| 1.6 | Act-quant hoisted into producer | FlashInfer CUTLASS/CuteDSL kernels accept a pre-quantized `QuantizedActivation` (`as_quantized_activation`), letting the compile passes move the fp4 quant into RMSNorm/SiLU/attention (see §6) | `kernels/linear/nvfp4/flashinfer.py:112,185`; `layers/fusion/quant_activation.py` |
| 1.7 | TRTLLM epilogue-tile weight shuffle | `shuffle_matrix_a`/`shuffle_matrix_sf_a` prepack at load for the trtllm-gen kernel | `kernels/linear/nvfp4/flashinfer.py:280` |
| 1.8 | W4A16 auto-degrade | same checkpoint runs weight-only (CuteDSL BF16×FP4 on SM100/103, Marlin elsewhere) when A4 unsupported — multi-arch portability from one artifact | `kernels/linear/__init__.py:1177`; `kernels/linear/nvfp4/marlin.py:18` |
| 1.9 | FlashInfer autotune at warmup | `autotune` ctx manager run during engine warmup, cached (`flashinfer_autotune_cache.py`); enabled at O1-O3 | `utils/flashinfer.py:354`; `model_executor/warmup/kernel_warmup.py:261` |
| 1.10 | Dual streams | **absent** in the dense quant GEMM path (only MoE shared-experts overlap) | `fused_moe/runner/shared_experts.py` |

## 2. compressed-tensors family (W4A4Fp4 / W4A4Mxfp4 / W4A8 / W8A8)

| # | technique | mechanism | refs |
|---|---|---|---|
| 2.1 | Declarative scheme dispatch | `_get_scheme_from_parts` picks the scheme from checkpoint QuantizationArgs (format predicates `_is_nvfp4_format` etc.); capability-gated with graceful fallback (fp8 w8a8 → w8a16 weight-only below SM89) | `compressed_tensors/compressed_tensors.py:783,419-779,878` |
| 2.2 | Static-vs-dynamic act quant threading | `is_static_input_scheme = input_quant and not input_quant.dynamic` per layer; static → load `input_scale`, fold `max()` across fused shards; dynamic → None | `compressed_tensors.py:900-932`; `schemes/compressed_tensors_w8a8_fp8.py:186` |
| 2.3 | NVFP4 divisor inversion + alpha | CT stores global scales as 1/scale → inverted once at load; `alpha = input_gs * weight_gs` + `input_global_scale_inv` precomputed; warns on mismatched fused-shard global scales | `schemes/compressed_tensors_w4a4_nvfp4.py:110-141,100-124` |
| 2.4 | W4A8-FP8 (Hopper) | int4 weight + fp8 act, group 128, via MPLinearKernel framework → CutlassW4A8; scales fp8-packed/expanded at load | `schemes/compressed_tensors_w4a8_fp8.py:51-95,127` |
| 2.5 | INT8 azp_adj precompute | asymmetric act quant: `azp_adj = weight.sum(dim=0)` (weight-only term of the zero-point correction) computed once at load; static zp folded in | `kernels/linear/scaled_mm/cutlass.py:104-121` |
| 2.6 | Weight transpose for CUTLASS at load | `.t()` to (K,N) + re-tag dims for loader-v2 | `scaled_mm/cutlass.py:52-57`; `schemes/compressed_tensors_w8a8_fp8.py:148-192` |
| 2.7 | **Online transforms (SpinQuant/QuIP-style)** | `CompressedTensorsLinearTransformMethod` wraps any inner quant method with input/output rotations; `HadamardTransform` module uses `ops.hadacore_transform` (no materialized matrix) or dense GEMM; 1/√n **folded into the matrix once at load** (dedup-guarded for shared matrices); rotation matrix loaded once, shared across fused partitions (`SharedWeightParameter`); TP>1 unsupported | `compressed_tensors/transform/linear.py:36-170`; `transform/module.py:75-124` |
| 2.8 | **Fused rotation+quant (QuTLASS)** | when scheme = W4A4-NVFP4 + input Hadamard with head_dim ∈ {16,32,64,128} on SM100: ONE kernel `fusedQuantizeNv(x, hadamard_matrix, global_scale)` does rotation + NVFP4 quant + block scales; `fused_alpha = weight_gs/6.0` precomputed at load; then triton `to_blocked` SF swizzle + `flashinfer_scaled_fp4_mm` | `transform/schemes/linear_qutlass_nvfp4.py:49,95-158`; `qutlass_utils.py:23,87,154` |

## 3. FP8 W8A8 (+ DeepGEMM)

| # | technique | mechanism | refs |
|---|---|---|---|
| 3.1 | Max-scale requant of merged shards | `requantize_with_max_scale`: qkv/gate_up shards dequant→requant to one per-tensor scale at load so ONE GEMM covers the fused weight (the in-engine analog of our offline shared-g) | `utils/w8a8_utils.py:75`; `utils/fp8_utils.py:1454` |
| 3.2 | Per-token dynamic act quant preferred | activation key auto-picks `kFp8DynamicTokenSym` when CUTLASS present (per-token beats per-tensor for accuracy at same speed) | `quantization/fp8.py:274-280` |
| 3.3 | Fused norm+quant custom ops | `rms_norm_dynamic_per_token_quant`, `rms_norm_static_fp8_quant`, `rms_norm_per_block_quant` — norm + quant + scales in one kernel (consumed by compile passes, §6) | `_custom_ops.py:423,443` |
| 3.4 | Pad-to-16 at load | CUTLASS fp8 kernel pads weight K,N→16 at load with a wrapping `padded_weight_loader`; pads act at runtime, slices output | `kernels/linear/scaled_mm/cutlass.py:184-274` |
| 3.5 | DeepGEMM UE8M0 requant + TMA scale layout | block weights requantized in place so block scales are powers of two (UE8M0 → cheap in-kernel dequant); scales transformed to TMA-aligned layout at load | `utils/fp8_utils.py:1096,1169-1254` |
| 3.6 | Warmup providers / JIT caches | layers register `deep_gemm_warmup_provider`; warmup enumerates M values × block configs so no JIT at serve time; flashinfer autotune + cutedsl warmup caches | `kernels/linear/scaled_mm/deep_gemm.py:103`; `model_executor/warmup/deep_gemm_warmup.py:35` |
| 3.7 | Pre-processed-weights fast path | if checkpoint already in runtime layout, `process_weights_after_loading` skips all transforms | `quantization/fp8.py:359-371` |
| 3.8 | QuantFP8 CustomOp | one CustomOp for per-tensor/token/channel/group act quant with native (compile-fusable) + CUDA + HIP impls; reciprocal-multiply instead of divide in native path | `quantization/input_quant_fp8.py:29-231` |

## 4. Weight-only (AWQ / GPTQ / Marlin / Machete / Exllama) — the offline-repack canon

| # | technique | mechanism | refs |
|---|---|---|---|
| 4.1 | MPLinearKernel abstraction + auto-upgrade | `auto_awq`/`auto_gptq` claim plain awq/gptq checkpoints (`override_quantization_method`) and pick the best kernel per layer: CutlassW4A8 → Machete → Marlin → Conch → Exllama → Triton → Humming | `quantization/auto_awq.py:263,288`; `auto_gptq.py:214`; `kernels/linear/__init__.py:523,865` |
| 4.2 | Marlin tile-interleaved repack at load | `ops.gptq_marlin_repack` → MMA-fragment-native layout; zero-pad to tile-aligned (padded_n,padded_k); **scale permutation** `marlin_permute_scales` matches scales to fragment order | `kernels/linear/mixed_precision/marlin.py:74-101`; `utils/marlin_utils.py:514,524,257` |
| 4.3 | Zero-point folding/permute | AWQ ZP de-interleave + repack (`awq_to_marlin_zero_points`); Machete pre-folds `-scale*zp` so dequant is one FMA | `marlin_utils.py:567,589`; `mixed_precision/machete.py:80` |
| 4.4 | Act-quant inside Marlin (W4A8) | int8 act: scales rescaled ×4096 to int16 range + compensating `input_global_scale`; fp8 act: `marlin_int4_fp8_preprocess` + scales ×512 | `mixed_precision/marlin.py:84-88,135-140`; `marlin_utils.py:544` |
| 4.5 | Persistent workspace manager | marlin workspace from a global `current_workspace_manager()` sized by threadblocks (no per-layer alloc) | `marlin_utils.py:408-418` |
| 4.6 | Atomic-add reduce heuristic | `should_use_atomic_add_reduce` picks atomic-add epilogue vs split-k at apply time by shape | `marlin_utils.py:752` |
| 4.7 | Exllama `gptq_shuffle` + checkpoint-bug ZP synth | column shuffle at load; synthesizes `weight_type.bias-1` ZPs to compensate the historical GPTQ "+1" format bug | `mixed_precision/exllama.py:88-110` |
| 4.8 | **NVFP4-via-Marlin scale re-encode (S0E5M3)** | to run an NVFP4 checkpoint weight-only on pre-Blackwell: fp8-E4M3 weight scales re-encoded ×2^7, sign bit becomes exponent MSB (`view(int16)<<1`), power-of-2 `scale_factor` folded into global scale so every nonzero scale has MSB=1 → wider range + faster in-kernel dequant | `utils/marlin_utils_fp4.py:37,59,143,219-282` |
| 4.9 | act_order/g_idx | modern Marlin path handles activation reordering entirely in the load-time repack (runtime `g_idx=None`) | `marlin_utils.py:752` |

## 5. Rotation-fused W4A4 cousins + KV-cache quant

| # | technique | mechanism | refs |
|---|---|---|---|
| 5.1 | FP-Quant fused Hadamard+quant | `FPQuantConfig(hadamard_group_size=32, forward_dtype mxfp4/nvfp4, forward_method abs_max/quest)`; block-diagonal Hadamard applied *inside* the activation-quantize kernel (`fused_quantize_mx/nv` custom ops with fake impls → compile-safe); GEMM `matmul_mxf4_bf16_tn` / `cutlass_scaled_fp4_mm`; alpha = 1/(w_gs·a_gs) | `quantization/fp_quant.py:30-96,231-380` |
| 5.2 | Humming | WNA16(+FP8/INT8 act) GEMM backend with per-layer `hadamard_block_size` in the weight schema (rotation is part of the linear op); supports **online requant** of unquantized checkpoints via a weight-loader intercept; auto-preferred over Marlin on SM90 only | `quantization/humming.py:258,382`; `utils/humming/schema.py:36`; `utils/humming/priority.py:11` |
| 5.3 | TurboQuant KV cache | KV-only quant via `--kv-cache-dtype` presets (k8v4/4bit/k3v4/3bit): Hadamard-rotate each head-dim vector → Lloyd-Max centroids for K (precomputed for N(0,1/d)), uniform for V; norm-correction; boundary-layer skip | `quantization/turboquant/config.py:20,48,83,177`; `centroids.py:30` |
| 5.4 | `online/` family | runtime quantization of *unquantized* checkpoints (fp8/int8/mxfp4/mxfp8/nvfp4) — `--quantization fp8` on any model, weights quantized at load | `quantization/online/*` |

## 6. Cross-cutting machinery (compile passes, graphs, warmup, KV)

| # | technique | mechanism | refs |
|---|---|---|---|
| 6.1 | **RMSNorm+quant fusion pass** | inductor pattern match RMSNorm(+residual add) → quant and replace with one fused op; supports static fp8 / dynamic per-token fp8 / group-128/64 / **NVFP4** (`FusedAddRMSNormNvfp4QuantPattern` → FlashInfer `add_rmsnorm_fp4quant`, SM100+) | `compilation/passes/fusion/rms_quant_fusion.py:158-199,681,690` |
| 6.2 | **SiLU-mul+quant fusion pass** | `SiluMulFp8StaticQuantPattern`, `SiluMulNvfp4QuantPattern` (→ `silu_and_mul_nvfp4_quant`), block variant — kills the full-width intermediate between MLP act and down_proj quant | `passes/fusion/act_quant_fusion.py:33,80,125,179` |
| 6.3 | **Attention-output quant fusion** | folds post-attention quant into the attention op's `output_scale` arg (`AttnNvfp4QuantPattern`) — o_proj input arrives already quantized; requires attention inside a FULL graph | `passes/fusion/attn_quant_fusion.py:39,216`; `config/compilation.py:1165` |
| 6.4 | AllReduce+RMSNorm(+quant) fusion | TP only: one FlashInfer collective replaces allreduce + norm + quant | `passes/fusion/allreduce_rms_fusion.py:158-179` |
| 6.5 | Pass ordering + opt levels | O1: fuse_norm_quant + fuse_act_quant + PIECEWISE graphs; O2/O3: + fuse_attn_quant (if quantized) + allreduce/rope/kvcache fusions + FULL_AND_PIECEWISE | `passes/pass_manager.py:139`; `config/vllm.py:246-310` |
| 6.6 | CUDA-graph modes | `FULL_AND_PIECEWISE` = full graph for decode batches + piecewise for prefill; `cudagraph_capture_sizes` define the static-M buckets; wrapper replays on mode match | `config/compilation.py:53-63,651`; `compilation/cuda_graph.py:145,235` |
| 6.7 | FP8 KV cache | any method subclassing `BaseKVCacheMethod` gets `kv_cache_dtype=fp8`: static per-tensor scales from checkpoint **or default 1.0**; NVFP4-KV path computes per-(token,head) scales dynamically in the cache-write kernel; query quant left as a plain torch op so compile fuses it | `quantization/kv_cache.py:43,76-140`; `modelopt.py:164`; `layers/attention.py:521` |
| 6.8 | Noop/cleanup passes | reshape/slice elimination before lowering so quant patterns stay matchable | `passes/utility/noop_elimination.py` |

---

## 7. THE GAP CHECKLIST

Status: **HAVE** (we do it, possibly differently), **MISSING** (they have it, we don't),
**N-A** (not applicable to our format/stack). Gain = expected effect on the measured
0.53-0.61x pm/nvfp4 mid/large-M gap and decode rows (`docs/vllm_perf_round1.md` §3-4).
Round = suggested vllm-perf loop round (r2 = currently running perf agent; r3+ = queue).

| technique (ref §) | ProMix status | notes / our analog | expected gain | round |
|---|---|---|---|---|
| Out-of-tree method registration + entry point (arch) | HAVE | `plugin.py:66`, pyproject `vllm.general_plugins` | — | done |
| Weight-loader v2 params (arch) | HAVE | `plugin.py:127` | — | done |
| Load-time SF swizzle (1.2) | HAVE | `prepack_sf_fp4_b/fp8_b`, probe-proven byte-identical to `swizzle_blockscale` | — | done |
| Alpha/global-scale fold at load (1.1, 2.3) | HAVE | shared per-tensor g as epilogue alpha; shared-g enforced at load (`plugin.py:184-189`) | — | done |
| Merged-shard scale unification (3.1) | HAVE (stronger) | offline shared-g requant in exporter + PPL re-reference vs their silent in-engine max-scale requant | — | done |
| CUDA-graph compat, FULL_AND_PIECEWISE (6.6) | HAVE | one opaque custom op; 51 PIECEWISE + 51 FULL captured | — | done |
| GPTQ/act-order-style offline weight opt (4.9) | HAVE | GPTQ + PCA/variance ordering + two-level scales in exporter | — | done |
| Decode small-M specialization | HAVE (unique) | M-routed W4A16 GEMV per capture size — vLLM has NO per-M format switching | — | done |
| **Producer-fused act quant: RMSNorm→quant (6.1, 3.3)** | **MISSING** | we pay a standalone `quant_pack_float` launch per linear; vLLM-NVFP4 fuses fp4 quant+SF-write into the producing norm. Our opaque op blocks inductor matching → need our own fused `rms_norm→quant_pack_mixed` kernel (consume the norm) or expose pre-quantized input à la `QuantizedActivation` (1.6) | **high** — named biggest per-layer tax (phase23 §overhead 1); attacks prefill 0.55-0.64x + decode b8/b16 0.53-0.61x | r3 |
| **SiLU·mul→(Hadamard)→quant fusion in-plugin (6.2, 2.8)** | **MISSING (kernel exists!)** | `hadamard28_swiglu_pack_mixed` + 1B variant already bit-identical standalone (perf_round2 Task 1, hadamard_swiglu_fusion_report) — NOT wired into `plugin.py`; replaces 3-4 launches on down_proj | **high**, down_proj is the largest K | r2/r3 |
| **Rotation fused into act-quant kernel (2.8, 5.1)** | PARTIAL | down_proj fusion kernel exists unwired (above); o_proj column gather still a separate torch gather — fold the gather into `quant_pack_float` (index-load) like fusedQuantizeNv folds Hadamard | medium (o_proj small-M rows) | r3 |
| **Per-shape GEMM config dispatch (1.4)** | **MISSING** | round-2 kernel work produced per-shape configs (b1_tma, cfg10 class) — plugin always runs one 128×128 config; vLLM picks kernel per layer at load + FlashInfer autotunes tactics | **high** — lever 1a of round-1 §4; mid-M GEMM regime | r2 (in progress) |
| Autotune/warmup cache (1.9, 3.6) | MISSING | no per-shape tactic cache; JIT ext build at load (minutes, cached by torch ver) — add a probe-once-persist table keyed (shape, M-bucket, arch) like flashinfer autotune / deep_gemm warmup | medium (enables dispatch above; removes first-run jitter) | r2/r3 |
| **In-kernel 1-byte scale decode (3.5, 4.8)** | **MISSING** | decode GEMV streams fp32 effective scales (+~18% weight bytes/token at M≤4); store checkpoint E4M3/UE8M0 bytes + decode in-kernel; consider Marlin's S0E5M3 re-encode + power-of-2 global-scale fold for cheap dequant | medium-high — est. −8-12% decode-b1 TPOT (round-1 §4.2) | r3 |
| **FP8 KV cache (6.7)** | **MISSING (near-free)** | we serve fp16 KV; subclass `BaseKVCacheMethod` (or just don't block `kv_cache_dtype=fp8` — scales default 1.0) → halves KV memory + attention read BW. Gate with T2; note baselines must get the same flag for fair tables | medium (grows with ISL/batch; capacity win immediately) | r4 |
| TurboQuant KV presets (5.3) | MISSING (free ride?) | engine-level `--kv-cache-dtype turboquant_*` — orthogonal to linear quant; we have our own KV4 probe (perf_r2 Task 2) to cross-check accuracy | low-medium | r4 |
| Attention-output quant fusion (6.3) | MISSING | o_proj input quant could ride the attention epilogue — but collides with our o_proj column reorder; prerequisite: fold reorder into weight/quant kernel first | medium, blocked | r4+ |
| Weight/act pad-to-tile at load (1.3, 3.4) | PARTIAL | we assert divisibility (K_low%16, K_high%32) — fine for Llama shapes, blocks odd shapes/TP shards; add pad+slice like `pad_nvfp4_weight_for_cutlass` | low (generality, not speed) | r5 |
| TP>1 shard handling (2.2, 4.2 scale rules) | MISSING | plugin asserts TP=1; needs K/N-shard-aware packing + per-shard g + marlin-style scale replication rules | enables 70B story; no single-GPU gain | r5 |
| Multi-arch fallback (1.8, 4.8) | MISSING | promix checkpoint runs only on SM120; a W4A16-via-Marlin(/our GEMV) fallback would run it anywhere | low (paper: portability claim) | backlog |
| Pre-processed-weights fast path (3.7) | MISSING | re-prepack on every load (seconds — cheap); could cache prepacked buffers | negligible | backlog |
| Online requant of unquantized ckpt (5.4, 5.2) | N-A | our method requires offline rotation+GPTQ; an RTN demo mode possible but pointless for accuracy story | — | — |
| Zero-point folding / azp_adj (4.3, 2.5) | N-A | symmetric formats only | — | — |
| AllReduce+norm fusion (6.4) | N-A (TP=1) | revisit with TP>1 | — | r5+ |
| DeepGEMM UE8M0 weight requant (3.5) | N-A | our MXFP8 high scales are already UE8M0; NVFP4 low scales E4M3 by format | — | — |
| Dual streams in dense GEMM (1.10) | N-A in vLLM | **vLLM doesn't have it either** — our dual-format mainloop could overlap low/high phases = potential edge, not a vLLM trick | — | kernel loop |
| Batch-invariant mode (`VLLM_BATCH_INVARIANT`) | N-A | research feature, not perf | — | — |

## 8. Top-3 highest-ROI missing tricks

1. **Producer-side fused activation quant+pack** (§6.1/6.2/2.8 → checklist rows 9-11).
   The single biggest structural difference vs the rival: vLLM-NVFP4 never launches a
   standalone act-quant kernel — fp4 quant + swizzled-SF write ride the RMSNorm, the
   SiLU·mul, and (O2+) the attention epilogue. We launch `quant_pack_float` per linear
   plus a separate fp16 Hadamard on down_proj. The down_proj fused kernel
   (SiLU·mul→Hadamard→pack, bit-identical) **already exists from perf round 2 — wiring
   it into `plugin.py` is pure integration**; the RMSNorm→quant_pack fused producer for
   qkv/gate_up is new kernel work. Expected: the main lever on the 0.53-0.61x mid/large-M
   and 0.55-0.64x prefill rows (phase23 overhead #1).
2. **Per-shape GEMM config dispatch + a persisted autotune table** (§1.4/1.9/3.6).
   vLLM picks the best kernel per layer at load and FlashInfer autotunes tactics at
   warmup with a cache; we run one 128×128 config for every shape and M. The round-2
   per-shape configs exist; add load-time selection per (N,K) + M-bucket and a
   probe-once JSON cache (graph-timed, per round-1's lesson that eager timings mislead).
   Expected: the remaining small/mid-M GEMM gap (round-1 §4 lever 1a). *Currently
   assigned to the running r2 perf agent — do not double-take.*
3. **In-kernel 1-byte scale decode for the decode GEMV, with Marlin's scale re-encode
   trick** (§3.5/4.8). Stop streaming fp32 effective scales (+18% bytes/token at M≤4):
   stream the checkpoint's 1-byte E4M3/UE8M0 scales, decode in-kernel; adopt the
   `marlin_utils_fp4.py:59` S0E5M3 re-encode + power-of-2 global-scale fold so dequant
   is shift-cheap. Est. −8-12% decode-b1 TPOT on top of round-1's −31%.

Near-free honorable mention: **FP8 KV cache** (§6.7) — subclass `BaseKVCacheMethod`
(scales default to 1.0 without checkpoint calibration; T2-gate it). Capacity doubles at
zero kernel work; TPOT gain grows with context length.

## 9. ProMix edges nobody in vLLM 0.32 has (paper systems section)

1. **Dual-format single-launch GEMM** — every vLLM scheme is single-format per layer
   (NVFP4 *or* MXFP4 *or* FP8); nothing fuses two numeric formats (NVFP4 low + MXFP8
   high ring) into one mainloop with a shared accumulator and one epilogue.
2. **Phase-adaptive numeric format per captured graph size** — vLLM's kernel choice is
   static per layer at load; FlashInfer autotune varies *tactics*, never *format*. Our
   M-routed W4A16-GEMV/W4A4-GEMM switch inside one opaque op (branch resolved at capture
   time, graph-legal) changes the arithmetic itself per phase — and measurably improves
   accuracy on the decode route (ΔPPL −0.11..−0.34) while being 1.7-2.9x faster at M=1.
3. **Tunable accuracy/speed budget curve** — the per-layer `k_high` map in
   `quantization_config` is a deployment-time dial (variance-ordered channel budget);
   vLLM checkpoints carry a fixed format with no per-layer precision budget knob.
4. **Accuracy-gated integration methodology** — exporter ships `t2_reference.pt`
   (per-position reference logprobs/top-1 of the shared-g real-kernel model) inside the
   checkpoint; every perf change passes a signed-ΔPPL gate (T0/T1 bit-exact, T2 ≤0.1).
   No vLLM method ships a numerics reference with the artifact.
5. **act-MSE activation scaling in-kernel** — vLLM activation quant is absmax
   everywhere (FP-Quant's QuEST flag is the only cousin); our compile-flag act-MSE
   refinement (3 steps) is part of the shipped quant kernel.
6. **Offline shared-g re-quantization with measured re-reference** — vLLM silently
   requantizes merged shards to the max scale at load (3.1); we re-derive scales offline
   and re-measure PPL so the deployed artifact's accuracy is known, not assumed.
7. (Candidate, unexploited) **dual-stream low/high overlap** — absent in vLLM's dense
   path (1.10); our two-phase mainloop is a natural fit for inter-phase overlap — a
   kernel-loop idea, not yet implemented.

---
*Survey executed read-only against the live benchmark venv; no runtime code touched
(perf agent r2 owns `promix/vllm_integ/` concurrently). One-doc branch
`feat/vllm-tricks` off `feat/rd-alloc`.*
