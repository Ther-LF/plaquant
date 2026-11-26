# Effective bits accounting — exact per-element bits for the Pareto plot

**2026-11-26, feat/rd-alloc.** Machine-readable twin: `docs/effective_bits.json`.

Goal: for ProMix and every related-work point on the accuracy-vs-bits Pareto plot, the
*effective* bits per weight element and per activation element, **including every
scale / zero-point / outlier / residual / metadata overhead**. Rule: measured from code
buffers or checkpoint tensors wherever possible; paper/config-derived values are marked.
`[假设]` marks the one remaining assumption per entry and its impact bound.

Conventions:

- Bits are per **original** element of the quantized Linears (duplicated/residual columns
  are charged back to the original K), params-weighted across Linears when allocation
  varies per layer.
- Per-tensor FP32/BF16 globals amortize to `32/(N·K)` ≤ 1e-4 bits/elem — written in the
  formula, not added numerically.
- CUTLASS interleaved-SF 128-row atom padding is deployment padding, excluded (noted
  where it was measured).
- Tiers: `w4a4` (same-layer comparison group), `weight-only` (W4A16, different marker +
  footnote on the plot), `fp8`, `ref`.

---

## ProMix (ours) — analytic in r (FP8-channel fraction)

Format: K channels split `[K_low NVFP4 | K_high MXFP8]`, r = K_high/K.

Weights (`promix/quantize/weight_pack.py:146-170`, PackedWeight buffers):

| buffer | size | bits/elem |
|---|---|---|
| `W_lo_packed` | N·K_low/2 uint8 (2 E2M1/byte) | 4 on K_low |
| `physSFB_lo` | N·K_low/16 uint8 (E4M3 micro-scale) | 8/16 = 0.5 on K_low |
| `W_hi_bytes` | N·K_high uint8 (E4M3) | 8 on K_high |
| `physSFB_hi` | N·K_high/32 uint8 (E8M0) | 8/32 = 0.25 on K_high |
| `g_w` two-level global | 1 FP32 scalar/tensor | 32/(N·K) ≈ 3e-6 |

```
eff_w(r) = (1-r)·(4 + 8/16) + r·(8 + 8/32) + 32/(N·K) = 4.5 + 3.75·r
```

Activations (`promix/quantize/quant_utils.py:1066-1180`, `fake_quantize_nvfp4` /
`fake_quantize_mxfp8`; block sizes `MXFP8_BLOCK=32`, `NVFP4_BLOCK=16` at :113-115): same
block-scale density, **single-level** (no global on the activation path):

```
eff_a(r) = 4.5 + 3.75·r          eff_avg(r) = 4.5 + 3.75·r
```

11-point r table (nominal r; shipping allocator snaps K_high to 64-rings,
`rd_oppareto_ours.py:snap_khigh`, so use exact r = K_high/K per shape when plotting
measured points; r=1 is served all-but-one-ring, K_low=64):

| r | 0 | 1/32 | 1/16 | 1/12 | 1/8 | 3/16 | 1/4 | 3/8 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eff | 4.5000 | 4.6172 | 4.7344 | 4.8125 | 4.9688 | 5.2031 | 5.4375 | 5.9062 | 6.3750 | 7.3125 | 8.2500 |

**Audit of the existing `eff_bits` in `promix/eval/rd_oppareto_ours.py:140-142`**
(`(4.5·K_low + 8.25·K_high)/K`): **correct**. The only omissions are the per-tensor
two-level global `g_w` (32 bits/tensor, round-6; `quant_utils.py:762-817`,
`weight_pack.py:256,318`) and nothing on the activation side — ≤ 1e-5 bits/elem. No code
change required; the exact formula is `+ 32/(N·K)`.

Excluded with notes:

- **decode-GEMV duplicate pack** (`weight_pack.py:167-170`): `S_lo_dec` FP32 per-16
  (+2 bits) and `S_hi_dec` FP32 per-32 (+1 bit) plus a second code copy — the *same
  information* re-packed for the M=1 path. Pareto counts the format cost once; if
  "resident HBM bytes as currently implemented" were the axis, ProMix weights would be
  ~2·(4.5+3.75r)+3 — flag only, both packs are never the comparison basis for peers
  either.
- **K_low_pad** +64 zero columns (`weight_pack.py:173-183,266-274`): GB10-only, zero
  information, excluded.

**KV cache (kv4, NOT merged into the main axes)** — per-head `[main NVFP4 | high MXFP8]`
with high rounded UP to the MXFP8 block (`quant_utils.py:118-157`, mode="up"):

| head_dim | high | main | KV bits/elem |
|---|---|---|---|
| 128 (3B/8B/Qwen) | 32 | 96 | (96·4.5 + 32·8.25)/128 = **5.4375** |
| 64 (1B) | 32 | 32 | (32·4.5 + 32·8.25)/64 = **6.375** |

---

## NVFP4 (modelopt / llm-compressor / vLLM) — w 4.500 / a 4.5 [measured]

Measured from safetensors headers of `/home/luofan/baselines/quant/Qwen3-8B-NVFP4` and
`unsloth--Llama-3.2-3B-nvfp4` (every quantized Linear):

- `weight_packed` U8 `(N, K/2)` → 4 bits
- `weight_scale` F8_E4M3 `(N, K/16)` → 0.5 bits
- `weight_global_scale` + `input_global_scale` F32 `[1]` → 2·32/(N·K), negligible

Summed over all quantized Linears: **4.5000 bits/elem exact** (6.95e9 elems, Qwen3-8B).
Activations: dynamic NVFP4 (4 + 8/16) + the static per-tensor `input_global_scale` →
**4.5**. avg **4.5**.

> Discrepancy vs tracker "4.56": that number was checkpoint-GB ÷ total params, i.e. it
> smears the bf16 embeddings/norms/lm_head over the weights. Per quantized element the
> format is exactly 4.500 — use 4.5 on the Pareto axis.

---

## ARCQuant — w = a = 5.2204 / 5.1214 / 5.1362 (1B / 3B / 8B) [measured]

Format: everything NVFP4 (E2M1 + UE4M3 scale per 16 = 4.5;
`ARCQuant/model/quantize.py:145-171`) + per-tensor FP32 global (`quantize.py:291`,
negligible). The method's mechanism is K-expansion:

- Weights: top-S channels **duplicated** along K (`quantize.py:287-310`:
  `cat([q(W), q(W[:, topk])])`).
- Activations: NVFP4(x) concat NVFP4(residual error on the same S columns)
  (`quantize.py:312-341`) — same (K+S)/K expansion, so **a_bits = w_bits**.

Per Linear the method's own accounting (`ARCQuant/utilize.py:554,560`):
`bits = 4.5·(K+S)/K`. Params-weighted over its saved per-model allocations
(`saved/*_average_bits_wikitext2_max.pt`, re-weighted by N·K on the server):

| model | weighted bits | S̄/Linear | per-Linear range |
|---|---|---|---|
| Llama-3.2-1B-Instruct (hash 5a8abab4, K inferred 2048/8192 ✓) | **5.2204** | — | — |
| Llama-3.2-3B | **5.1214** | 385 | 4.535–5.813 |
| Llama-3-8B | **5.1362** | 483 | 4.520–5.766 |

> Tracker values 5.12 / 5.14 / 5.22 **confirmed** (source was the same files; now re-derived
> and pinned to utilize.py:554-576). Omitted by the method's own count: per-tensor FP32
> global and the returned per-row absmax (`scale_w`/`scale_x`) — ≤ 0.01 bits/elem.

---

## SharQ — w 4.75 / a 7.5 [measured; biggest correction]

Weights (`SharQ/kernels/src/shared_weight_nvfp4.cu:20-111`): ONE shared E2M1 payload
(K/2 bytes/row) + UE4M3 scale per 32 for the sparse GEMM view, then
`duplicate_sfb32_to_dense_sfb16_kernel` writes a **second physical** per-16 scale buffer
(same values duplicated) for the dense residual GEMM; + per-tensor global
(`model/quantize.py:149-153`).

```
w = 4 + 8/32 (sparse SFB) + 8/16 (dup dense SFB) = 4.75 physical   (4.25 if dedup'd)
```

Activations — measured from its own buffer allocator
`sharq_ops.get_fused_sparse_residual_buffer_sizes(M=2048, N=4096, K=4096)`
(`kernels/src/bindings.cpp:243-253`, `include/sparse_nvfp4.h:117-151`):

| buffer | bits/elem |
|---|---|
| `A_comp` 2:4-compressed E2M1 backbone (K/2 kept) | 2.000 |
| `E` sparsity metadata | 0.500 |
| `SFA_sparse` UE4M3 | 0.531 (0.5 asympt., +128-row pad) |
| `Q_res` dense NVFP4 residual (full K) | 4.000 |
| `SF_res` UE4M3 per 16 | 0.531 (0.5 asympt.) |
| **total** | **7.5625 measured / 7.5 asymptotic** |

avg (4.75 + 7.5)/2 = **6.125**.

> Discrepancy vs tracker "~4.5 (†)": weight physical storage is 4.75, and the † ("two
> activation streams") is now quantified: **7.5 activation bits/elem** — SharQ is not a
> 4.5-bit-class activation method on this axis.

---

## MicroMix — w = a = 5.2256 (3B) / 5.1800 (8B) [measured]

Format: per-Linear channel split into MXFP4 (E2M1) / MXFP6 (E3M2) / MXFP8 (E4M3), all
UE8M0 scale per 32 (`mgemm/src/gemm.cu:13-19`). FP6 is genuinely 6-bit packed in memory
(`mgemm/src/bindings.cpp:69`: `KS = AS.size(1)·4/3`). Same reorder/split for weights and
activations.

```
bits(Linear) = (4·p4 + 6·p6 + 8·p8)/K + 8/32
```

Params-weighted from its own saved allocations
(`saved/{Llama-3.2-3B,llama-3-8b}_p{6,8}_num_wikitext2.pt`):

| model | elem-only | + scale 0.25 |
|---|---|---|
| Llama-3.2-3B | 4.9756 | **5.2256** |
| Llama-3-8B | 4.9300 | **5.1800** |

Tracker 5.23 / 5.18 **confirmed**. No 1B allocation saved (MicroMix was not run on 1B).

---

## ResQ recipe (CoQuant both criteria; = our INT control) — w ≈ 4.51 / a ≈ 4.52

Config as run (`CoQuant/fake_quant/0_eval_ptq_4bit.sh`): `--w_bits 4 --a_bits 4
--high_bits 8 --high_fraction 0.125 --w_clip --a_asym --k_asym --v_asym`, KV4
per-head.

- w: 7/8·INT4 + 1/8·INT8, per-output-channel symmetric scale per segment (2 FP16/row)
  → `4.5 + 32/K`: **4.5124 / 4.5088 / 4.5063** (1B/3B/8B params-weighted).
- a: per-token asym, 2 segments (scale+zero each) → `4.5 + 64/K`:
  **4.5248 / 4.5176 / 4.5126**.
- KV: INT4 asym per-head = `4 + 32/head_dim` (4.5 @64, 4.25 @128).

`[假设]` scale dtype FP16 (fake-quant materializes them in model dtype; FP32 storage
would double only the 32/K and 64/K terms — ≤ +0.02 bits). Tracker "~4.6" → **4.51**.

---

## Atom — w 4.27/4.21/4.17, a 4.33/4.27/4.23 (1B/3B/8B) [paper-config, repo-documented]

As run (`peers/atom/RESULTS.md` §2, its own formula): keeper=128 INT8 channels,
group-128 INT4, channel_group=2, `--a_sym --w_sym`, KV INT4 per-head:

```
w(K) = 4·(1−128/K) + 8·(128/K) + 16/(128·2) + (16/128)·(128/K) = 4.0625 + 528/K
a(K) = 4.125 + 528/K   (per-token group-128 FP16 scales + keeper scale)
```

Params-weighted: w **4.2670 / 4.2075 / 4.1666**, a **4.3295 / 4.2700 / 4.2291**.
`[假设]` FP16 activation scale dtype. Tracker "~4.3" → per-model values above.

## QuaRot — w = a = 4.0062 / 4.0044 / 4.0032 (1B/3B/8B) [as-run config]

`peers/quarot/RESULTS.md:44-59`: weights INT4 GPTQ **per-channel symmetric**
(group_size = −1, `--w_clip`) → `4 + 16/K`; activations INT4 per-token symmetric →
`4 + 16/K`; KV4 per-token sym. `[假设]` FP16 scales.

## SpinQuant — w 4.0062/4.0044/4.0032, a 4.0124/4.0088/4.0063 [as-run config]

`peers/spinquant/RESULTS.md:16-18`: weights as QuaRot (per-channel sym GPTQ);
activations per-token **asym** (`--a_asym`, scale+zero) → `4 + 32/K`; KV INT4 asym
per-head group = head_dim → `4 + 32/hd` (4.5 @1B, 4.25 @3B/8B).

## GPTQ / AWQ W4A16 — w 4.125 / a 16 [measured; weight-only tier]

Both evaluated checkpoints are the symmetric compressed-tensors layout (measured from
safetensors headers, `unsloth--Llama-3.2-3B-{gptq,awq}_w4a16`): `weight_packed` I32
`(N, K/8)` = INT4, `weight_scale` BF16 `(N, K/128)` = g128, **no zero-point tensor**:

```
w = 4 + 16/128 = 4.125        a = 16 (bf16)       avg = 10.0625
```

(Classic asym AWQ would be 4 + 20/128 = 4.156 — not what our ckpt stores.) Plot on the
weight-only tier with its own marker/footnote; never mixed into the W4A4 group.

## FP8 W8A8 (vLLM) — w 8.004 / a ≈ 8.008 [measured / 假设]

Measured (`unsloth--Llama-3.2-3B-fp8`): `weight` F8_E4M3 + `weight_scale` BF16
`(N, 1)` per-output-channel → 8 + 16/K = **8.0044** at 3B. No `input_scale` tensor →
vLLM dynamic per-token activation FP8: `[假设]` FP32 token scale → `8 + 32/K ≈ 8.008`.

## bf16 — 16 / 16 (anchor)

---

## Pareto-plot summary (avg of w/a where single-valued; per-model where it varies)

| method | w bits | a bits | tier | basis |
|---|---|---|---|---|
| QuaRot | 4.003–4.006 | 4.003–4.006 | w4a4 | as-run config |
| SpinQuant | 4.003–4.006 | 4.006–4.012 | w4a4 | as-run config |
| Atom | 4.167–4.267 | 4.229–4.330 | w4a4 | repo formula |
| NVFP4 (modelopt) | 4.500 | 4.5 | w4a4 | measured ckpt |
| ProMix r=0 | 4.500 | 4.500 | w4a4 | measured buffers |
| ResQ / CoQuant | 4.506–4.512 | 4.513–4.525 | w4a4 | as-run config |
| SharQ | 4.75 | 7.50 | w4a4 | measured buffers |
| ProMix r=1/8 | 4.969 | 4.969 | w4a4 | measured buffers |
| ARCQuant | 5.12–5.22 | 5.12–5.22 | w4a4 | measured saved alloc |
| MicroMix | 5.18–5.23 | 5.18–5.23 | w4a4 | measured saved alloc |
| GPTQ / AWQ | 4.125 | 16 | weight-only | measured ckpt |
| FP8 W8A8 | 8.004 | ≈8.008 | fp8 | measured ckpt |
| bf16 | 16 | 16 | ref | — |
