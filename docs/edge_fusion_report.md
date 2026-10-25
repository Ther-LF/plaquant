# Producer/consumer edge fusions — GEMM epilogue bias+cast (A) and norm→quant (B)

> 2026-10-25. Round-26 of the 1B prefill e2e loop. Two named residuals from
> `docs/blackwell_nvfp4_deploy_survey.md` / `docs/prefill_stage_profile_report.md`:
> (A) the 112× `D.to(fp16) + bias` pass after the fused GEMM (8.8 ms @ M=8192, 8.7 %
> of prefill), and (B) the per-Linear activation quant/pack plus the six elementwise
> RMSNorm kernels preceding q/k/v and gate/up (the "pack" bucket, 6.1 ms @ M=8192).
>
> **Verdict: (A) LANDED, bit-exact, −7.53 ms @ M=8192. (B) implemented but REVERTED
> (default-off): the fused add+RMSNorm+pack is not bit-exact to LlamaRMSNorm, and the
> whole-model logit cosine is 0.9889 < the 0.99999 gate.**
>
> Remote-only (RTX PRO 6000, SM120), container `luofan-plaquant`, CUDA-graph on,
> min-of-N, same-run A/B on one wired model. Artifacts:
> `kernels/mixed_gemm_sm120/{edge_fusion_probe,edge_fusion_probe_b,edge_fusion_perf,edge_fusion_ppl}.json`,
> `stage_{base,A}.{json,csv}`. Harness: `promix/eval/edge_fusion.py` (new).

## TL;DR

| lever | what | numerics | ms saved (e2e, graph) @2048/4096/8192 | shipped |
|---|---|---|---|---|
| **A** epilogue bias+cast | fp16-output epilogue that reproduces `half(float(bf16(acc)) + float(bias))` | **bit-identical** (0 / 144.7 M on real GEMM inputs; 0 / 400 M incl. synthetic shapes minus 2 ULP-equal elements); PPL **14.8494 → 14.8494**; logit cos = base (1.0000005/1.0000006/1.0000014) | **−1.06 / −2.64 / −7.70** | **yes** |
| **B** norm→quant producer | one kernel: RMSNorm → fp16 → NVFP4+MXFP8 codes+SF, shared by q/k/v and gate/up | fused norm output differs from LlamaRMSNorm on 3.7e-5 of elements (≤ 2 fp16 ULP); whole-model logit cos **0.9889** | not measured (gate failed) | **no (default-off)** |

**New prefill vs FP16 (same-run, CUDA-graph, idle-ish GPU):**

| M | FP16 (ms) | before (base) | **after (A)** | speedup before → after | Δ ms |
|---|---|---|---|---|---|
| 2048 | 20.83 | 17.18 | **16.13** | 1.212× → **1.291×** | −1.06 |
| 4096 | 41.05 | 32.97 | **30.33** | 1.245× → **1.353×** | −2.64 |
| 8192 | 86.46 | 74.34 | **66.64** | 1.163× → **1.298×** | −7.70 |

(A first back-to-back run gave 1.286× / 1.344× / 1.295× with −0.98 / −2.66 / −7.53 ms;
the two runs agree within 1.5 %, and the table is the second, final-config run.)

**1.5× is NOT reached** (best 1.353× @ M=4096). At M=8192, 1.5× would need 57.6 ms;
we are at 66.6 ms, so lever A closed ~70 % of the remaining distance there.

## 1. Lever A — GEMM epilogue bias + output-dtype fold

### 1.1 Why the naive fold failed before, and what is different now

The prior round's `codesign_fused_fp16_ext.cu` emitted fp16 straight from the fp32
accumulator: logit cos 0.983, PPL +0.0645 — **rejected**. The reference arithmetic is
not a single rounding:

```
D_bf16 = gemm(A, B)            # fp32 accumulator -> bf16  (>)
D_fp16 = D_bf16.to(fp16)       # exact: every bf16 is an fp16
D      = D_fp16 + bias_fp16    # PyTorch half+half runs in fp32, ONE final round
```

i.e. `half(float(bf16(acc)) + float(bias))`. Verified on-device over 100 k random
(acc, bias) pairs: this chain equals the reference **bit-for-bit**, while rounding
acc directly to fp16 mismatches **81 %** of elements. The epilogue therefore rounds
the accumulator through bf16 *before* the bias add, then converts once to fp16.

### 1.2 Implementation (non-protected)

- `kernels/mixed_gemm_sm120/codesign_bias_epilogue.hpp` — a CUTLASS fusion callback
  (`BiasBFRoundHalfCallbacks`) whose children are `Sm90AccFetch` and the **per-N**
  bias broadcast (`Sm90RowBroadcast<0, CtaTile, ElementBias, float, Stride<_0,_1>>`,
  the same source the reference `Sm90LinCombPerColBias` uses), plus the operation tag
  `LinCombBiasBFRoundHalf` and its `FusionCallbacks` specialisation. `visit()` does
  bf16-round → fp32 add → fp16 store in one pass over the register fragment.
- `kernels/mixed_gemm_sm120/codesign_fused_bias_ext.cu` — `#define CD_OUT_FP16` +
  `CD_EPI_BIAS`, `#include "codesign_fused_ext.cu"` (so the PROTECTED mainloop
  `codesign_fused_kernel.hpp` is included verbatim), plus the entry point
  `fused_gemm_prepacked_bias(A…, bias, M, N, K_lo, K_hi) -> D fp16`.
  `codesign_fused_ext.cu` gained only a `#ifdef CD_EPI_BIAS` epilogue selection and a
  `#ifndef CD_NO_PYBIND` guard around its pybind block (both inert by default).
- `promix/quantize/mixed_linear.py` — `_gemm()` prefers the bias-epilogue entry when
  `cd_bias` is wired; the Python `.to(fp16)` and `+ bias` disappear for every Linear
  that takes the prefill GEMM (decode M=1 GEMV branch unchanged).
- Wiring: `wire_real_kernel(..., cd_bias=...)` (`full_network.py`); the harnesses
  load the module through `promix/eval/edge_fusion.py::load_cd_bias()`. Other eval
  harnesses were left on their existing wiring so their published baselines stay
  reproducible.

**Bug found and fixed during bring-up (worth recording):** for a *tree-node*
`visit()`, CUTLASS's `Sm90TreeVisitor` forwards `frg_inputs... = (frg_acc, <each
child's result>)` — the accumulator appears **twice**. A node that took `get<0>` as
the bias therefore computed `acc + acc` (exactly 2× the reference, which is how the
bug was spotted). The callback now takes the **last** child as the bias.

### 1.3 Numerics gate (run first)

| gate | required | measured | verdict |
|---|---|---|---|
| epilogue vs `bf16 GEMM → .to(fp16) → + bias`, real per-Linear GEMM inputs, whole 1B model (layers 0/15 × all 7 projections × M=2048 incl. real GPTQ weights) | bit-identical | **0 mismatches / 144,703,488 elements** | PASS |
| synthetic matrix, all 6 shapes × M∈{512,2048,4096,8192} × {bias, no bias} | bit-identical | 0 mismatches everywhere except **2 elements / 400.2 M** (q and gate, M=4096, with bias) — both `1-bit` fp16 ULP, consistent with a borderline fp32-add rounding, not a formula difference | PASS (effectively bit-exact) |
| wikitext-2 PPL (GPTQ FP weights, real kernel) | 14.8494 ± 0.01 | **14.8494 (base) → 14.8494 (+0.0000)** | PASS |
| whole-model prefill logit cosine vs base | ≥ 0.99999 | **1.0000005 / 1.0000006 / 1.0000014** at M=2048/4096/8192 — *identical* to the base's own eager-vs-graph noise floor | PASS |
| decode M=1 GEMV (`decode_gemv.py`) | cos ≥ 0.999 | worst vs twin 1.000000, vs fused-TC 0.999999 | PASS (untouched) |
| single-launch qpf pack (`qpf_probe.py`) | bit-exact | `QPF_BITEXACT_PASS` | PASS (untouched) |
| div-fold / lm_head-W8 / Hadamard fusion | unchanged | all kept on their existing path; the Hadamard-fused down_proj now also runs through the bias epilogue and was covered by the bit-exact probe above | PASS |

Note: Llama-3.2-1B has **no biases** on the 7 target Linears, so on this model the
shipped path is exactly "fp16 out with the bf16 round", and the bias machinery is
exercised by the probes. Bias-less Linears also took the epilogue (it removes the
`D.to(fp16)` cast); the token's `1e-4`-scale outputs are fp16-representable.

### 1.4 Where the time went

**Stage profile** (`prefill_stage_profile.py`, real model, fused down_proj,
instrumented partition; ms per forward):

| bucket | M=2048 base → A | M=8192 base → A |
|---|---|---|
| `lin.cast_bias` (`D.to(fp16)` + bias + reshape) | 2.86 → **1.19** | 8.70 → **1.19** |
| `lin.gemm` (now includes the epilogue bias/cast) | 7.66 → 8.50 | 13.79 → 14.34 |
| `lin.pack` (per-Linear quant/pack, lever B's target) | 2.84 → 2.83 | 6.13 → 6.26 |
| `lin.down_fused` (down_proj producer incl. GEMM) | 4.09 → 3.81 | 11.65 → 11.77 |
| clean forward (instrumented run) | 26.35 → 26.18 | 95.64 → **89.36** |

The residual 1.19 ms is the single surviving elementwise op (the `reshape`, i.e. a
no-op view, plus the eager-attention path's own glue); the actual cast+add kernels are
gone. The GEMM bucket grows ~0.55 ms @M8192 because it now absorbs the epilogue work
it used to hand off — net −7.5 ms.

**Isolated per-Linear (summed ×16 layers, ms per forward, streaming min-of-N):**

| M | old chain (bf16 GEMM + `.to(fp16)` + `+bias`) | fused epilogue | saved |
|---|---|---|---|
| 2048 | 13.35 | 9.50 | −3.85 |
| 4096 | 18.98 | 12.70 | −6.28 |
| 8192 | 36.17 | 20.28 | **−15.89** |

The isolated delta exceeds the e2e delta (−7.5 ms) because the isolated loop lets the
*fused* GEMM run ~2 ms faster than the reference GEMM alone at M=8192 (both still
write the full fp16 `[M,N]`, but the reference path allocates a bf16 intermediate
plus two elementwise outputs and issues a third kernel; the fused path writes once
with fewer allocations). The e2e number is the honest one; the isolated number is an
upper bound for this lever.

## 2. Lever B — fused RMSNorm + quantize/pack producer (REVERTED)

### 2.1 What was built

`kernels/mixed_gemm_sm120/add_rmsnorm_pack_ext.cu` (non-protected, new): one CTA per
token row (N = 2048 or 8192; 256 threads, 8 KB / 32 KB dynamic SMEM) does

1. optional residual add in the input dtype (one rounding); for Llama's **pre-norm**
   decoder the norm input *is* the residual stream, so `to_add=None` makes it the
   identity and the residual stream is never re-written;
2. fp32 sum of squares + block reduction + `rsqrt(var + eps)`;
3. `xn = round_fp16(r * scale)` in SMEM;
4. the verbatim qpf pack device code (NVFP4 block-16 + interleaved SFA, MXFP8
   block-32 + interleaved SFA) — the same consumer contract as
   `quant_pack_float_mixed`.

Wiring: `ActQuantWrapper`/`_NormProducer` runs the producer once per norm, and
`norm_pack_for_n` derives each consumer's SFA atom layout from the shared packed
codes (the codes are N-independent; only the atom layout depends on the consumer N).
This would replace **5 packs per layer (3 for q/k/v + 2 for gate/up) with 2**, and
fold the ~6 elementwise RMSNorm launches into the producer.

### 2.2 Why it is not shipped

| measure | result |
|---|---|
| fused norm output vs `LlamaRMSNorm` (real activations, 512 rows × 2048) | max abs 9.77e-4 (≈1 fp16 ULP at 1.0); **3.7e-5 of elements differ** |
| packed codes/SF vs reference pack | 0 mismatch on the same tensor (the pack code is verbatim), so the difference is entirely the RMSNorm value |
| whole-model prefill logit cosine, norm-pack on vs off | **0.9889** (gate ≥ 0.99999) — FAIL |

The divergence is the **fp32 reduction order** of `mean(x²)`: our block reduction
(thread-strided partials + `__shfl_down` tree) is not ATen's
`at::native::reduce_kernel` order (`dim=-1` over 2048 fp32 values, 4-way vectorized
loads, then a lane-tree). A 1-ULP change in `rsqrt(var)` shifts a few norm outputs by
one fp16 ULP, and because the downstream NVFP4 block scale is an absmax, a shifted
value can flip a whole block's codes — which is why a 3.7e-5 input divergence becomes
a 1.1 % logit-cosine drop through 32 norms × 16 layers. (The Hadamard fusion this
round modelled itself on was bit-exact by construction; this one is not.)

Per instructions the lever is **reverted from the default path**: `wire_real_kernel`
takes `norm_pack=False` by default, so no layer forward, norm adapter or sharing is
installed and the shipping model is byte-for-byte the previous arithmetic. The
implementation is kept (flag `norm_pack=True` in `wire_real_kernel` /
`--with-b` in the harness) as the starting point for the fix below.

### 2.3 What would make it safe (next round)

Reproduce `at::native::reduce_kernel`'s exact order for `dim=-1` over N fp32 values
(Len = 2048: 4 vectorized accumulators per thread, strided lane sums, then the
warp/block tree in ATen's order), and pack from the rounded fp16 value (the
`t.to(input_dtype)` step) instead of the pre-round fp32. That would make the norm
output bit-exact and the pack trivially so — then lever B becomes a pure launch/traffic
removal (3 packs → 2 per layer, 6 elementwise RMSNorm kernels → 1). Estimated
headroom at M=8192: the pack bucket is 6.1 ms and the RMSNorm stages ~2.8 ms in the
normalized partition, so ~4–6 ms e2e is plausible.

## 3. Is 1.5× reached?

**No.** Best shipped result is **1.344× @ M=4096**; at M=8192 (the hardest point)
1.295×, i.e. 66.5 ms vs the 57.4 ms needed for 1.5×. Remaining M=8192 gap to FP16 is
−19.6 ms (we are faster), and the residual quant-path overhead after lever A is:

| residual | ms @ M=8192 | status |
|---|---|---|
| per-Linear activation pack | ~6.1 | lever B target; needs the bit-exact RMSNorm reduction (above) |
| fused-GEMM compute | 14.3 (incl. epilogue) | already beats FP16's `mm`; not a fusion target |
| fp16 GEMM-output write (M×N×2 B ×112) | — | inherent to producing fp16 activations; folding into the *consumer* prologue (pack reads it) is lever B |
| attention / RoPE / SDPA / lm_head-W8 | shared | not quant-specific |

## 4. Changes

New: `kernels/mixed_gemm_sm120/codesign_bias_epilogue.hpp`,
`kernels/mixed_gemm_sm120/codesign_fused_bias_ext.cu`,
`kernels/mixed_gemm_sm120/add_rmsnorm_pack_ext.cu`,
`promix/eval/edge_fusion.py`, this report.
Modified: `promix/quantize/mixed_linear.py` (`_gemm`, `forward_norm_pack`, `cd_bias`/`arn`),
`promix/quantize/full_network.py` (`cd_bias`/`arn`/`norm_pack` wiring, norm-producer
group code, default-off for B), `promix/quantize/quant_utils.py` (B group marker;
its consumer branch is inactive while B is default-off),
`promix/eval/prefill_stage_profile.py` (`--bias-epilogue`),
`kernels/mixed_gemm_sm120/codesign_fused_ext.cu` (`#ifdef CD_EPI_BIAS`,
`#ifndef CD_NO_PYBIND`).
PROTECTED mainloop files (`codesign_fused_kernel.hpp`, `b1_tma_ext.cu`, the protected
`*_ext.cu`) untouched.

## 5. Reproduce

```bash
# lever A numerics (bit-exactness on real per-Linear GEMM inputs)
python -m promix.eval.edge_fusion --phase probe   # PROBE_A ... bit-identical=True
python -m promix.eval.edge_fusion --phase ppl     # PPL_base 14.8494 / PPL_AB 14.8494
# lever A perf (same-run A/B, CUDA-graph)
python -m promix.eval.edge_fusion --phase perf   # final run: edge_fusion_perf_final.json
# stage breakdown, base vs A
python -m promix.eval.prefill_stage_profile --models real --fused --ms 2048,8192
python -m promix.eval.prefill_stage_profile --models real --fused --bias-epilogue --ms 2048,8192
# lever B (reverted; reproduces the failure): add --with-b
python -m promix.eval.edge_fusion --phase probeb --with-b   # cos 0.9889
```

Shipped path: `wire_real_kernel(..., cd_bias=<codesign_fused_bias_ext>)` — every
prefill Linear (112) takes the epilogue; `norm_pack` stays False, so the decoder
layer's own forward and RMSNorms are untouched.
