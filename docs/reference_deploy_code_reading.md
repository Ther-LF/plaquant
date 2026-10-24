# Reference deploy-path code reading: QuaRot, SpinQuant, Atom, QServe, ResQ

> 2026-10-24. Read-only study of how each reference turns a quantization *algorithm*
> into a runnable model with a measurable speedup, compared against our SM120
> **NVFP4 + MXFP8 mixed W4A4/A8** path. All citations are relative to each repo root;
> repos were cloned into `/tmp/refdeploy/` (QuaRot, SpinQuant, Atom, omniserve = QServe
> + LServe) and `/tmp/project-resq` (ResQ). Nothing in the tracked repo was modified
> except this file.
>
> Companion docs: `docs/blackwell_nvfp4_deploy_survey.md` (what TRT-LLM / vLLM /
> llm-compressor do), `docs/prefill_fusion_validated_report.md` (our own fusion gate).

---

## 0. TL;DR — the four things every deployable stack does that we do not

1. **The quantizer is a module that lives inside the deployed graph, not a research hook.**
   QuaRot: `nn/quantization.py:4-13` (`Quantizer` returns a packed INT4 tensor) +
   `nn/linear.py:42-56` (`Linear4bit.forward` *asserts* its input is a
   `PackedQuantizedTensor`). The quantize step is a real op with a real output type,
   wired by name in `checkpoint_utils/quantize_llama_checkpoint.py:29-54`.
2. **Activation quantization is fused into the producer kernel's epilogue.** QServe:
   `kernels/csrc/activation_kernels.cu:33-82` does `dequant(int32 gate/up) → silu*mul →
   per-token INT8 quant (+ row sum)` in one kernel, so the down_proj input never exists
   as FP16 in DRAM. Atom does the cheap version of the same thing
   (`model/qLlamaLayer.py:137-149`: RMSNorm's output is quantized inside the norm module).
3. **The GEMM's dequant is in registers, in the epilogue, after INT accumulation.**
   Atom `kernels/include/GEMM/Dense_layer_gemm_i4_o16.cuh:404-430` (INT4 main + INT8
   "keeper" in one launch), QServe `kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu:599-627`
   (fp16 weight-scale × fp32 activation-scale applied to `int32` fragments before store).
   Our `MixedQuantLinear` does the same *semantically*, but as a separate pack pass plus a
   dequantized-weight GEMM — see §6.
4. **KV cache quant is fused into the KV-write kernel and its dequant into attention.**
   QuaRot `transformers/kv_cache.py:175-257` + `kernels/flashinfer.cu` (init/append/decode
   templated on INT4 or FP16); QServe
   `csrc/fused_attention/fused_attention_pure_dense/applyBiasRopeUpdateKVCache.h:279-362`
   (RoPE + per-token asymmetric KV4 + scale/zero store in one kernel). We have the
   *algorithm* (`promix/quantize/kv_quant.py:196-280`) but the attention kernel still
   keeps K in INT8 and V in FP16 (`kernels/flash_attn/mixed_flash_attn.cu:44-45`).

---

## 1. QuaRot (`spcl/QuaRot`)

**Deployment status: real.** It ships a deployable HF model (`e2e/quantized_llama/`),
a CUTLASS INT4 GEMM + CUDA quant kernels (`quarot/kernels/`, built via `setup.py`), and
a paged INT4 KV cache with a FlashInfer-derived batch-decode kernel. `fake_quant/` is a
separate directory used only for the accuracy sweep (`fake_quant/main.py`).

### Mechanism

| Question | Answer | Code |
|---|---|---|
| Rotation fused into weights? | Yes, all of R1. Embedding/head/attn-in/attn-out/mlp-in get `W·Q` or `Qᵀ·W`; `down_proj` additionally absorbs the *inverse exact Hadamard* on its input side. | `e2e/checkpoint_utils/rotation_utils.py:45-101` (esp. `:84` `apply_exact_had_to_linear(W, had_dim=-1, output=False)`) |
| Rotation on the activation path? | Only the **down_proj** Hadamard (`OnlineHadamard(intermediate_size)`) and the **per-head o_proj** Hadamard (`OnlineHadamard(num_heads)`), both as `nn.Sequential` elements before the Linear. Both run in FP16 by default (`force_fp32=False`). | `e2e/quantized_llama/modeling_llama.py:99-103`, `:111-115` (MLP sequential at `:111-115`); `quarot/nn/hadamard.py:5-23` |
| Where does activation quant happen? | Inside `Quantizer` modules placed as the *first element* of each quantized `nn.Sequential` (o_proj, down_proj) or explicitly in `LlamaMLP.forward` / attention `forward`. Symmetric per-token INT4, scale = `max|x|/7`. The output is a packed uint8 + scale struct. | `quarot/nn/quantization.py:9-13`; `modeling_llama.py:40`, `:118` |
| Real kernel vs fake quant | Real path: `quarot.matmul` = CUTLASS INT4×INT4→INT32 (`kernels/gemm.cu:16-41`, SM80 tag), dequant via `sym_dequant` (`kernels/quant.cu:64-97`), quant+pack via `sym_quantize_f16_i4_kernel` (`quant.cu:11-61`). Fake path: `fake_quant/quant_utils.py` (STE quant-dequant, PyTorch matmul). | `quarot/nn/linear.py:42-56` |
| QK^T / PV / KV quantized? | **KV only.** Q and K are never rotated over head_dim (no R3/R4) and attention scores are FP16. KV cache is asymmetric INT4 with per-token scale+zero, online Hadamard on K only (Q gets the same Hadamard at read time). | `kv_cache.py:11-33`, `:188-201`, `_AttentionStub.forward:111-127` |
| Residual stream | Kept in FP16, but in the *rotated* basis: `embed_tokens` was rotated, so every residual add is in Q coordinates and each consumer Linear absorbed Q. Residual add itself is never quantized. | `rotation_utils.py:113-123`; `modeling_llama.py:160-183` |
| RMSNorm | Replaced by `quarot.nn.RMSNorm` with **weight = ones** (norm scale fused into the following Linears), so it is a pure normalization — no learned vector to worry about, and no quantizer on its output. | `quarot/nn/normalization.py:3-21`; `rotation_utils.py:25-41` |
| Checkpoint rewrite | The saved state dict renames `o_proj → o_proj.1`, `down_proj → down_proj.2` (the sequential indices), drops the norm weights, and packs weights/`weight_scales` into the `Linear4bit` buffers. `modeling_llama.py` is copied into the checkpoint as `quarot.py` and hooked through `auto_map`. | `e2e/checkpoint_utils/quantize_llama_checkpoint.py:29-81` |
| Attention output | `attn_output.transpose(-1,-2)` → per-head Hadamard → transpose back → reshape → quantizer → o_proj. The transpose is because `OnlineHadamard` rotates the last dim. | `modeling_llama.py:82-84` |

### Caveats observed
- The GEMM is SM80-tagged (`gemm.cu:25`, "just for compiling on my laptop temporarily").
  Prefill never uses it: `e2e/benchmark.py:86-108` calls the HF forward for prefill; only
  decode goes through the paged INT4 kernel (`kv_cache.py:242-257` asserts
  `added_length == 1`).
- Per-token INT4 activation quant (`max/7`) is done in a standalone kernel, not fused into
  the norm — QuaRot's norm is the *only* producer for q/k/v and gate/up, so the quantize
  is one extra launch per Linear. This is exactly what TRT-LLM/vLLM later fused.

---

## 2. SpinQuant (`facebookresearch/SpinQuant`)

**Deployment status: accuracy-first research release.** Upstream `eval_utils/` has *no*
real quantized weight format and *no* real kernel: `ActQuantWrapper` fake-quantizes
activations (`utils/quant_utils.py:201-...`, `forward` at `:245`), the GPTQ `quantizers`
are used only to produce fake-quantized weights, and `QKRotationWrapper` quantizes K and
then **casts it back to the attention dtype** (`eval_utils/rotation_utils.py:39-53`).
The ExecuTorch export (`utils/convert_to_executorch.py`) is the only deploy path and is
not the paper's eval path. Its value for us is the *rotation inventory* (R1/R2/R3/R4) and
the proof that the online down_proj Hadamard is a fixed, fp16-cheap op.

### Mechanism

| Question | Answer | Code |
|---|---|---|
| Rotations | R1 (hidden, Hadamard or *learned* orthogonal), R2 (per-head, learned), R3 (Q/K Hadamard, head_dim), R4 (down_proj Hadamard). | `eval_utils/rotation_utils.py:123-147` (R1, R2), `utils/fuse_norm_utils.py:39-74` (norm fusion), `train_utils/apply_r3_r4.py:24-44` (R4), `eval_utils/rotation_utils.py:150-228` (R3+R2 applied post-RoPE) |
| R1/R2/R4 folding | Same as QuaRot: embeddings/head/qkv/o/gate/up by `R1`/`R1ᵀ`, v_proj/o_proj per head by `R2`, down_proj input by inverse Hadamard. R4 is special-cased when learned (`rotate_mlp_output(..., R4=...)`). | `eval_utils/rotation_utils.py:55-121` |
| Online ops | down_proj: `online_full_had = True`, `had_K`, `K`, optional `fp32_had` (config flag). o_proj Hadamard is applied via the v/o construction, not online. | `eval_utils/main.py:35-40` (inside `ptq_model`), `utils/quant_utils.py` `ActQuantWrapper.forward` |
| QK^T | **Not quantized.** R3 rotates q and k but leaves them fp16; the wrapper only quantizes K for the *cache*. | `eval_utils/rotation_utils.py:30-57` |
| KV cache | K only (no V quantizer). `k_bits`, per-token or per-head groupsize. Applied as a monkeypatched wrapper *after* `apply_rotary_pos_emb` — pre-RoPE quantization raises `NotImplementedError`. | `eval_utils/main.py:142-156`; `rotation_utils.py:60-79`; `modeling_llama.py:549-680` (unmodified FA2 forward) |
| Activation quant placement | `add_actquant(model)` wraps every `nn.Linear` in `ActQuantWrapper`; the wrapper rotates (online had) then fake-quantizes then calls the Linear. | `utils/quant_utils.py:489-523`; `eval_utils/main.py:42-44` |
| Residual | FP16, same rotated-basis trick. Nothing new vs QuaRot. | — |

### What to borrow
- The **`fp32_had` switch** (`eval_utils/main.py:40`, ResQ mirrors it) — a single flag that
  moves the down_proj Hadamard from fp16 to fp32. Useful for our numerics gate, but see
  our own finding in §5: on our path the fp32→fp16 cast is *not* free to move.
- The **monkeypatch-after-`apply_rotary_pos_emb`** pattern (`rotation_utils.py:60-79`) is
  the cleanest way to insert a post-RoPE rotation without forking the attention class. Our
  `promix/quantize/kv_quant.py:296-330` already uses the same monkeypatch helper.

---

## 3. Atom (`efeslab/Atom`)

**Deployment status: real kernels + real serving harness, but fake-quant for accuracy.**
The repo explicitly says "current release features: simulated quantization for accuracy
evaluation" and its TODOs still include "full inference workflow in real production
scenario" (`README.md:21-31`). Its e2e numbers come from substituting kernels into Punica
(`e2e/README.md:6-12`), with random weights. So Atom is the best reference for **kernel
shape**, not for **model plumbing**.

### Mechanism

| Question | Answer | Code |
|---|---|---|
| What kernels ship | INT4 main + INT8 "keeper" mixed GEMM (`kernels/src/GEMM/bench_dense_layer_gemm_i4_o16.cu`, impl `include/GEMM/Dense_layer_gemm_i4_o16.cuh`), fused RMSNorm (`include/RMSNorm/`), fused SwiGLU (`include/Activate/`), reorder (`include/Reorder/`, AWQ-style interleaved packing), FlashInfer decode (`include/flashinfer/`). No separate quant kernel in the fast path — quant is inside RMSNorm/Activate. | `kernels/include/GEMM/Dense_layer_gemm_i4_o16.cuh:1-48` (BLOCK 128×128×128, K-group 128, `KEEPER=128`) |
| How the GEMM gets W4A4 speed | `mma.sync` INT4 (16×8×64) for `K-128` columns and INT8 (16×8×32) for the 128 keeper columns, **in the same CTA/launch**; dequant+fma in registers after INT32 accumulation. | `Dense_layer_gemm_i4_o16.cuh:404-430` (`dequant`), `:712-740` (`DenseLayerGEMM_i4_o16` signature) |
| Activation quant placement | (a) `QLlamaRMSNorm.forward`: norm → **reorder index_select** → dynamic per-token quant, so q/k/v consume a quantized tensor directly. (b) `QLlamaMLP.forward`: `silu(gate)·up` → per-token quant → down_proj. | `model/qLlamaLayer.py:137-149`; `:346-350` |
| Quant math | Dynamic per-token, asymmetric, optional FP8/INT8 keeper for high channels. `Quantizer` supports "static" scales but defaults to dynamic (`args.static == False`). | `model/quant.py:185-232` (`quantize_activation_wrapper`), `:259-290` (`Quantizer.forward`) |
| Attention / KV | Scores are FP16. K is quantized **before** RoPE (comment: "preserve the position embedding info") and V after softmax in the eager path only; the e2e kernel path uses FP16 FlashInfer KV. | `model/qLlamaLayer.py:245-263` (`k_quant` before `apply_rotary_pos_emb`), `:283-285` (`v_quant`) |
| Reorder | Weights and the *activation* are permuted into the quantization-friendly channel order; the reorder is folded into the RMSNorm kernel (index_select) and the o_proj consumes a permuted attn output. | `model/qLinearLayer.py:80-87`; `model/qLlamaLayer.py:137-149`; `:305-306` |
| Reported e2e | 7.73× vs FP16, 2.53× vs INT8, **serving-context** (ShareGPT lengths, continuous batching). In the QServe comparison table Atom-W4A4 reaches 2120 tok/s (Llama-2-7B, L40S) vs QServe-W4A8KV4's 2394 — i.e. even a well-tuned W4A4 kernel lands close to, not far above, W4A8+KV4. | `README.md:26`; omniserve `README.md:252-270` |

### What to borrow
- The **single-launch mixed-precision CTA** (INT4 MMA and INT8 MMA in one kernel over
  different K ranges) is exactly the shape of our `mixed_gemm` idea — but Atom's `KEEPER`
  is a *fixed* 128-column tail, whereas ResQ's split point and the high-group layout are
  variable. Our SM120 kernel already does a variable `K_low/K_high` split, so this is
  confirmation rather than a new mechanism.
- **Quantize inside the norm kernel** (Atom) is the low-effort version of TRT-LLM's
  `add+RMSNorm+quant` fusion. Cheap to adopt on SM120 via a separate epilogue kernel if we
  do not want to touch the CUTLASS mainloop.

---

## 4. QServe (`mit-han-lab/omniserve`, includes QS)

**Deployment status: full serving system** (engine, paged KV, in-flight batching,
pre-quantized model zoo). This is the most instructive reference for *where every op
goes*.

### Mechanism

| Question | Answer | Code |
|---|---|---|
| W4A8 dequant fused into GEMM | Weights stored as **progressive groups**: 4-bit codes with a per-channel `s1` scale and per-group `s2` scale, plus **scaled zeros**; the GEMM applies `wscale·ascale` to INT32 fragments in the epilogue. No separate dequant kernel, no dequantized weight in memory. | `kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu:599-627`; packing `omniserve/modeling/layers/quantized_linear/w4a8_linear.py:17-135` |
| Activation quant placement | **Inside the producer**: `SiluAndMulQuant` runs `dequant(int32) → silu·mul → per-token INT8 quant (+ row-sum for asymmetric correction)` in one kernel; gate_up GEMM writes INT32, down_proj consumes INT8 directly. RMSNorm output is quantized by `fused_kernels` (residual add + norm + quant). | `kernels/csrc/activation_kernels.cu:33-82`; `omniserve/modeling/layers/activation.py:37-78`; `omniserve/modeling/models/llama_w4a8_unpad.py:66-137` |
| Asymmetric handling | The zero-point is folded into a **column-sum correction** (`input_sum`) computed during quant, applied inside the GEMM — so the tensor core sees symmetric signed ints. | `activation_kernels.cu:42-62`; `w4a8_linear.py:111-135` |
| KV cache | KV4 **fused with RoPE and the cache write**: one kernel computes RoPE, per-token (or fine-grained) min/max, scale/zero, quantizes to INT4, stores K and V plus scales. Attention reads INT4 and dequantizes on the fly. Granularity is a config (`per_tensor` / `fine_grained`). | `kernels/csrc/fused_attention/fused_attention_pure_dense/applyBiasRopeUpdateKVCache.h:279-362`; wrapper `omniserve/modeling/layers/ctx_update_kv.py:7-135` |
| Attention quant | Q stays INT8 from the qkv GEMM; **P (scores) are not quantized**; attention output is written FP16 and quantized by the following norm/producer. | `fused_attention_pure_dense/decoderMaskedMultiheadAttention.cu` |
| e2e numbers & settings | Max achievable throughput, ctx 1024 / gen 512, paged attention on, **in-flight batching OFF**, batch decided by memory budget (128–256 for 8B). 1.2× (A100) / 1.39× (L40S) vs TRT-LLM for Llama-3-8B; 2.4–3.5× for Qwen-72B. A100 uses per-channel, L40S per-group g128. | `README.md:248-281`; `scripts/qserve_benchmark/benchmark_a100.sh:1-20`, `benchmark_l40s.sh:19` |
| KV4 accuracy trick | "SmoothAttention": per-head scaling of K before quantization (stored as `kv_scale_quant_orig`), then inverse on Q. | `llama_w4a8_unpad.py:207-208`; `applyBiasRopeUpdateKVCache.h:327-362` |

### What to borrow
- **Progressive scales** (per-channel outer × per-group inner) are a way to get
  per-group weight quantization in a tensor-core-friendly layout. For our NVFP4 path the
  analogue is per-16-block E4M3 scales, which we already have; the *useful* part is the
  epilogue structure (`wscale·ascale` applied to the int accumulator, asymmetries handled
  by a precomputed row/col sum) — see our `MixedQuantLinear` comment on "no extra
  per-tensor scale" (`promix/quantize/mixed_linear.py:15-17`).
- **Fused producer epilogue** is the single biggest structural difference vs our path.
- **KV4 fused with the KV write** (RoPE + quant + store) is exactly the op we lack; we
  quantize K in a Python wrapper (`promix/quantize/kv_quant.py:240-280`) and then the
  attention kernel reads INT8.

---

## 5. ResQ — our own reference implementation (`Ther-LF/project-resq`)

> Note: the tracked local submodule `project-resq/` is an **empty checkout** in this
> working tree (`git submodule status` shows `-6fa3b4f...`, and its worktree has no
> files), so the reading below is from a clean clone of the same URL at
> `/tmp/project-resq` (HEAD `e07322e`). Code paths in `promix/*` that "delegate to resq"
> now correspond to the files cited here. The clone is the `Ther-LF/project-resq` fork,
> which carries additions over the paper release (the SM90 CUTLASS INT8 kernels,
> `triton_gemm.py`, `forward_real_quant`) — these are the parts most relevant to a
> deployment reading, but §5.2/§5.4 conclusions hold for the originals too.

### 5.1 Rotation inventory: folded vs paid per forward

| Rotation | Where applied | Folded into weights? | Runtime cost |
|---|---|---|---|
| Hidden-dim PCA basis `U_attn` (+ learned `R1`) | embedding, lm_head, q/k/v, o_proj input, gate/up, down_proj input | **Yes** — `fuse_basis_shared` multiplies every weight offline | Free |
| Per-head value rotation `U_value` (+ `R2`) | v_proj output, o_proj input | **Yes** — `rotate_ov_proj(..., R2=U_value, per_head=True)` | Free |
| o_proj **column reorder** (high channels → tail) | o_proj weight columns; attn output columns | Weights yes; **activation reorder is a runtime `index_select`** (`promix/quantize/quant_utils.py:388-401`, `reorder_before_quant`) | One gather per attn-output, per layer |
| **down_proj online Hadamard** | MLP intermediate activation (size `intermediate_size`) | **No** — only the *inverse* is folded into `W_down` (`rotate_mlp_output`, `apply_exact_had_to_linear(W, had_dim=-1, output=False)`) | **FP16 by default, optionally FP32** — one full `matmul_hadU_cuda` per forward, per layer |
| K-cache rotation `k_rotation` / Hadamard | q, k after RoPE | **No** — `QKRotationWrapper.forward` does `quantize(q) → q @ k_rotation` (or fp32 Hadamard) every forward | One fp32 Hadamard / matmul per attention layer |
| V-cache | v_proj output rotated by folded `U_value`; out_quantizer mixed 4/8-bit | Rotation yes, quantization is a runtime quantizer | Quantizer per forward |

### 5.2 Is the down_proj Hadamard applied in FP32 per forward? Is there a fused path?

**Yes, per forward; fp32 is opt-in, fp16 is the default.** `utils/quant_utils.py:411-431`
(`ActQuantWrapper.forward`) and the identical `_apply_rotation` at `:669-695`:

```python
if self.online_full_had:
    if self.fp32_had:                     # Full Hadamard in FP32
        x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
    else:                                  # Full Hadamard in FP16
        x = matmul_hadU_cuda(x, self.had_K, self.K)
```

It is set for `down_proj` only (`eval_utils/main.py:38-43`). **There is no fused path**:
the Hadamard is a separate kernel invocation on the full `(M, d_ff)` MLP activation,
followed by a separate per-token quantizer (`:432-436`) and then the Linear. The only
"fused path" in the tree is `forward_real_quant` (`:697-900`), which is *simulated* real
quantization: it still computes the GEMM as **fp32 `@` of quantized integer tensors**
(`:809`, `:822`, `:851-857` — `q_m_flat @ q_w_m.T`), not a tensor-core INT GEMM. The CUTLASS kernels
(`csrc/resq_gemm/resq_gemm_v2.cu`, SM90 INT8) and the Triton kernel
(`triton_gemm.py`) are **only referenced by `bench_gemm.py`** (`:392-430`) — nothing in
`ptq.py` or the model forward calls them. So: ResQ has **no serving path**, only
fake-quant and "simulated-real" evaluation.

### 5.3 What `fuse_norm` actually does

`fake_quant/utils/fuse_norm_utils.py:15-92`:
- subtracts the per-row mean from `embed_tokens` (fold the mean into the residual stream),
- multiplies `q/k/v` and `gate/up` weights by the preceding RMSNorm weight and sets the
  norm weight to **ones** (`:81-84`), same for `model.norm` → `lm_head` (`:87-92`).

This has **nothing to do with the rotation**: it runs *before* `rotate_model` /
`fuse_basis_to_model` so that the rotation can see a weightless RMSNorm (the rotation
commutes with a scalar normalize, not with a diagonal scale). The rotation itself is
folded later and separately (`eval_utils/rotation_utils.py:339-373`). Note ResQ *does
not* fuse the norm into the weights of the producer (the RMSNorm still executes); it
only removes the diagonal so the rotation is invariant.

**Could the down_proj Hadamard be folded into the preceding op?** Not the whole thing —
`down_proj`'s input is `silu(gate(x)) · up(x)`, and SiLU is elementwise nonlinear, so
`H·(silu(g)⊙u)` cannot be written as `silu(g')⊙u'`. Two tractable variants:

1. **Partial fold into the up/gate epilogue**: decompose `H_dff = H_{dff/K} ⊗ H_K`
   (Kronecker, K a power of two, e.g. 128). The cross-chunk factor `H_{dff/K}` is a dense
   `d_ff × d_ff` matrix and can be folded into `down_proj`'s weight (same trick as
   `rotate_mlp_output`); only the **intra-chunk** `H_K` butterfly then has to run online
   on `silu(g)⊙u`, which is `log2(K)` add/sub stages per element instead of the full
   `matmul_hadU_cuda`. For Llama-3.2-1B `d_ff = 8192 = 2^13`, so `K = 128` leaves a
   64-chunk outer matrix that `W_down` absorbs. This is the only fold that respects the
   SiLU nonlinearity, and it is exactly the structure `matmul_hadU_cuda` already exploits
   internally (`get_hadK` returns `(had_K, K)`).
2. **Fuse the butterfly into the SwiGLU epilogue** (a kernel change rather than an
   algebra change): compute `silu·mul`, then the `H_K` butterfly, then the per-token
   NVFP4/MXFP8 quant + pack, all in the gate/up GEMM's epilogue. This is the
   QServe/Atom pattern applied to the one place our path still pays a separate op.

Neither is a "fold the whole Hadamard into weights" win — because of SiLU, only the
`H_{d_ff/K}` half can be folded.

### 5.4 Accuracy configuration actually used

`fake_quant/2_eval_ptq.sh`: `w_bits 16` (weights stay FP16!), `a_bits 4`, `k_bits 4`,
`v_bits 4`, `high_bits 8`, `low_bits 2`, `high_fraction 0.125`, asymmetric a/k/v,
`k/v_groupsize 64`, `rotation_granularity full_shared`. So the paper's headline config is
**W16A4KV4** in terms of the code path exercised here — the "real quant" W4 option is a
separate flag (`ptq.py:342-390`).

---

## 6. What our path is missing vs theirs

Our current path: `promix/quantize/quant_utils.py` (fake-quant) /
`promix/quantize/mixed_linear.py` (`MixedQuantLinear`: torch-side quant+pack →
fused kernel → dequantized-weight GEMM) /
`promix/eval/full_network_realkernel.py` (real-kernel wiring) /
`kernels/mixed_gemm_sm120/` (NVFP4+MXFP8 SM120 kernel).

### 6.1 Activation-quant placement

| Reference | Their placement | Ours | Gap |
|---|---|---|---|
| QuaRot | `Quantizer` module as first element of the consumer Linear; output is a packed tensor the GEMM *requires* | `ActQuantWrapper.forward` quantize-dequantize to FP16 (`quant_utils.py:404-410`) then `MixedQuantLinear` re-quantizes (`mixed_linear.py:118-129`) | We quantize twice in the fake path and the real path quantizes *inside* the consumer, so there is no typed "quantized activation" contract between producer and consumer |
| Atom | inside RMSNorm and SwiGLU modules | separate `quant_pack_float_fp4/fp8` kernel or torch quant | **No producer-side quant.** `full_network_realkernel.py:205` explicitly notes the down_proj Hadamard is still a separate op before the pack |
| QServe | INT8 produced by the `dequant+silu·mul+quant` epilogue; norm output quantized by fused norm | separate pack kernel | **Same gap**: the MLP intermediate and the norm output are materialized in FP16, then read by the pack kernel |
| TRT-LLM/vLLM (from our Blackwell survey) | `add+norm+quant`, `gate_up+swiglu+quant`, `attn+quant` in the producer epilogue | — | biggest single launch-count lever |

### 6.2 Rotation fusion into producer ops

- We fold all weight-side rotations (`rotation.py:31-83`, `:97-133`, `:86-95`), same as
  everyone. Good.
- The **online down_proj Hadamard** is a standalone `matmul_hadU_cuda` call in
  `quant_utils.py:385-394` (and `real_forward.py:78-84`). QuaRot/SpinQuant have exactly
  the same standalone op — so this is *not* a gap vs them, but it *is* a gap vs
  TRT-LLM/vLLM and vs QServe, which fuse the equivalent rotation-free quant into the
  producer. For us the fold is harder because of SiLU (§5.3).
- The **o_proj column reorder** is a runtime `index_select` on the attention output
  (`quant_utils.py:388-401`). Atom folds the equivalent reorder into the RMSNorm kernel
  and QServe keeps the layout in the packed weight. We could fold it into the attention
  output store in our `mixed_flash_attn.cu` (K/V are already stored with an ad-hoc
  layout there: `mixed_flash_attn.cu:44-45`).

### 6.3 KV-cache quantization

| | QuaRot | SpinQuant | Atom | QServe | Ours |
|---|---|---|---|---|---|
| K quant | INT4 asym, per-token, online Hadamard | INT4/8 per-head, fake | INT4 before RoPE (eager only) | INT4 fused with RoPE in the KV-write kernel | mixed 4/8-bit per-head in a Python wrapper (`kv_quant.py:240-280`) |
| V quant | INT4 asym | none | after softmax (eager) | INT4 in the same kernel | via `out_quantizer` on v_proj |
| cache read | FlashInfer batch-decode on INT4 | fp16 | FP16 FlashInfer | INT4 dequant-on-read in FA | K INT8 / V FP16 (`mixed_flash_attn.cu:44-45`) |
| fused with RoPE? | yes (Hadamard in the wrapper, quant in `update`) | after RoPE, separate | before RoPE | **yes, same kernel** | no — separate Python ops then a kernel read |

Our K path is algorithmically the richest (mixed 4/8-bit per head, ResQ-style) but the
**write path is unfused and the read path is not INT4**, so a KV4 decode win cannot show
up. This is the largest *decode-specific* gap.

### 6.4 Attention quantization

- None of QuaRot, SpinQuant, Atom or QServe quantizes QK^T probabilities. QServe keeps Q
  in INT8 (output of the qkv GEMM) and P in FP16. So **not quantizing scores is the
  industry-consistent choice**; our `mixed_flash_attn.cu` INT8-Q path is already more
  aggressive than all four.
- The **attention output → o_proj** quant is where everyone saves: QuaRot puts the
  quantizer inside `o_proj`'s Sequential; QServe fuses it into the norm/producer. Ours is
  a separate `act_quant` step.

### 6.5 Kernel-level fusion we lack

1. Producer epilogue quant (norm+quant, swiglu+had+quant, attn-out+reorder+quant).
2. KV-write fused RoPE+KV4.
3. INT4 KV read / dequant-on-read in attention.
4. GEMM dequant in registers — we have this in `mixed_gemm_sm120` for the packed path,
   but `promix/inference/real_forward.py` still uses the old INT8 GEMM + fp32 dequant
   (`real_forward.py:95-131`), and `MixedQuantLinear` dequantizes the *weight* before the
   GEMM (per its own docstring, `mixed_linear.py:3-17`).

---

## 7. Ranked, actionable mechanisms (for W4A4 NVFP4 + MXFP8, Llama 1B/8B, SM120)

Ranked by expected e2e impact, given our own measurements (1B decode loses to FP16,
`docs/decode_throughput_report.md`; prefill capped at ~1.08×, `docs/prefill_1b_squeeze_report.md`).

> **In-flight note (2026-10-24):** while this doc was being written, an uncommitted
> change in the working tree already implements mechanism #1 as
> `kernels/mixed_gemm_sm120/hadamard_pack_ext.cu` + `promix/eval/hadamard_swiglu_fusion.py`
> (fused `silu·up → Hadamard → NVFP4/MXFP8 pack`, with `wire_fused_mlp` in
> `promix/quantize/full_network.py`). The ranking below is from the pre-fusion state;
> #1 is being validated, not missing.

1. **Fuse the down_proj quant into the SwiGLU producer (and/or the H_K partial fold).**
   Expected: removes one full `(M, d_ff)` fp16 write+read plus the separate Hadamard and
   pack launches per layer. This is the op our own prefill profile flagged as the
   non-kernel bottleneck (`docs/prefill_overhead_report.md`, `prefill_stage_profile.py:268`).
   Evidence: QServe `activation_kernels.cu:33-82`; Atom `qLlamaLayer.py:346-350`; our
   `full_network_realkernel.py:205`.
2. **Producer-side quant for the norm output (add+RMSNorm+quant).** Norm output is the
   q/k/v producer; every reference quantizes it in or immediately after the norm.
   Evidence: Atom `qLlamaLayer.py:137-149`; QServe fused norm; our path pays a separate
   `quant_pack_float_*` (`mixed_linear.py:94-121`).
3. **KV4 write fused with RoPE + KV4 read in attention.** Decode-only impact, but it is
   the only lever that can move our 1B decode from 0.70× (measured loss) toward parity
   at long context, and it is a prerequisite for any 8B long-context claim.
   Evidence: QServe `applyBiasRopeUpdateKVCache.h:279-362`; QuaRot `kv_cache.py:175-257`.
4. **Attention-output reorder+quant fused into the attention epilogue.** Removes a
   gather and a quant launch per layer; cheap and local to `mixed_flash_attn.cu`.
   Evidence: our `quant_utils.py:388-401` (runtime gather); Atom folds the same reorder
   into the norm (`qLlamaLayer.py:137-149`).
5. **Real INT GEMM in the evaluation path replaces fp32 `@`.** ResQ's
   `forward_real_quant` and our `real_forward.py` both simulate INT tensor cores in fp32;
   that inflates measured latency and hides the kernel's real dequant cost. Wire the
   SM120 kernel into `ptq.py`/`real_forward.py` so accuracy *and* speed come from one
   path. Evidence: ResQ `quant_utils.py:809`, `:822`, `:851-857`; callers only in
   `bench_gemm.py:392-430`; our `real_forward.py:95-131`.
6. **Typed quantized-activation contract (packed tensor struct), not quantize-dequantize
   to FP16.** Removes the double quantization in the fake path and makes the consumer
   Linear unable to receive an unquantized tensor by construction. Evidence: QuaRot
   `nn/quantization.py:4-13` + `nn/linear.py:42-56`; Atom `Quantizer` returning a tensor
   the GEMM expects.
7. **Ship a checkpoint converter that bakes the sequential/quantizer structure into the
   state dict** (rename `o_proj`→`o_proj.1`, drop zeroed norms, copy `modeling_llama.py`
   into the checkpoint). We already have `scripts/convert_resq_to_sglang.py`; it should
   also emit the fused-kernel layout so deployment does not re-derive it.
   Evidence: QuaRot `quantize_llama_checkpoint.py:29-81`.
8. **Numerics-gated `fp32_had` switch and per-op precision flags** (SpinQuant/ResQ both
   expose them). Our `fold_had_scale` was validated; `had_out_fp32` and the fp16 epilogue
   both failed our gate — keeping these as explicit flags documented in one place
   prevents re-litigating them. Evidence: SpinQuant `main.py:40`; ResQ
   `quant_utils.py:411-431`; our `full_network.py:189-240`.

---

## 8. Reproduction notes

```bash
# local clones used for this doc (proxy needed when github.com is unreachable)
mkdir -p /tmp/refdeploy && cd /tmp/refdeploy
git clone --depth 1 https://github.com/spcl/QuaRot
git clone --depth 1 https://github.com/facebookresearch/SpinQuant
git clone --depth 1 https://ghfast.top/https://github.com/efeslab/Atom          # Atom -> Atom/model, Atom/kernels
git clone --depth 1 https://ghfast.top/https://github.com/mit-han-lab/omniserve # QServe + LServe
git clone --depth 1 https://ghfast.top/https://github.com/Ther-LF/project-resq  # ResQ (the tracked submodule is empty here)
```

The tracked `project-resq/` submodule pointer is `6fa3b4f3d9a3bdd907c7c2f5cdc32674df5a95a7`;
the clone above is HEAD `e07322e` (`fix: clear _last_gemm_output...`). Line numbers in §5
refer to that clone.
