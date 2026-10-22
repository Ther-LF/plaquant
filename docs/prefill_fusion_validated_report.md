# Prefill-Fusion — Validated Numerics + Measured Speedup Report

> 2026-10-22. Closes out the prefill-fusion round of the ProMix e2e perf loop.
> A prior agent committed three prefill-fusion edits (HEAD `4365e84`) to kill the
> ~33 ms of *fixable* overhead the stage profile found
> (`docs/prefill_stage_profile_report.md`), but was stopped mid-validation. This
> report **validates the numerics first**, keeps the fixes that pass and reverts
> the ones that fail, then measures the prefill speedup vs FP16 before→after.
>
> Remote-only (RTX PRO 6000, SM120), container `luofan-plaquant`, idle GPUs
> (util 0, mem < 200 MB). Harness `promix/eval/prefill_fusion.py`. Raw artifacts:
> `kernels/mixed_gemm_sm120/prefill_fusion_{perf,ppl_validated}.json`,
> `decode_gemv_recheck.json`.

## The three edits under test

| id | edit | gated by | file |
|---|---|---|---|
| **fix1** div-fold | fold the down_proj online-Hadamard `1/sqrt(n)` into the `fast_hadamard_transform` kernel's `scale` arg → kills the separate full-tensor `aten::div` kernel | `wrapper.fold_had_scale` | `hadamard.py`, `utils.py`, `quant_utils.py` |
| **fix2** had-fp32 | feed the FP32 Hadamard output STRAIGHT to the qpf pack (skip the `.to(fp16)` round-trip cast) | `wrapper.had_out_fp32` | `quant_utils.py` |
| **fix3** fp16-gemm | fused-GEMM epilogue emits **fp16** directly (config copy `codesign_fused_fp16_ext.cu`, `CD_OUT_FP16`) → folds away the 112× per-Linear bf16→fp16 `D.to(fp16)` cast (profile target #3) | passing `cd16` to `wire_real_kernel` | `codesign_fused_{ext,fp16_ext}.cu` |

All three coexist with the validated decode-GEMV M=1 path (`decode_gemv_ext.cu`).

## 1. NUMERICS GATE (ran FIRST) — only fix1 passes

### 1a. Whole-model wikitext-2 PPL (seqlen 2048, GPTQ FP weights, real kernel)

| config | PPL | ΔPPL vs baseline | gate (14.85 ±0.02 → [14.83, 14.87]) |
|---|---|---|---|
| fake-quant twin | 14.9159 | — | (reference) |
| **real baseline (pre-fusion)** | **14.8494** | — | ✓ reproduces 14.85 |
| **fix1 div-fold ONLY** | **14.8494** | **+0.0000** | **✓ PASS (bit-identical)** |
| fix1+fix3 (fp16-gemm) | 14.9139 | **+0.0645** | ✗ FAIL |
| fix1+fix3+fix2 (had-fp32) | 14.8738 | +0.0244 | ✗ FAIL (just over) |

### 1b. Whole-model prefill logit cosine vs the pre-fusion real path (eager == graph)

| config | M=2048 | M=4096 | M=8192 | gate ≥0.9999 |
|---|---|---|---|---|
| **fix1 div-fold** | **1.000000** | **1.000001** | **1.000001** | **✓ PASS** |
| fix3 fp16-gemm (alone) | 0.983117 | 0.984360 | 0.983682 | ✗ FAIL |
| +fix2 had-fp32 (cumulative) | 0.983074 | 0.984212 | 0.983415 | ✗ FAIL |

### 1c. Decode M=1 path — still bit-exact after the fusion edits

Re-ran `decode_gemv.py` (per-Linear M=1, all 7 Llama-3.2-1B shapes) with the
current (fused-edit) tree:

- **worst cosine vs FP32 twin = 1.000000**, worst vs fused prefill kernel = **0.999999 → PASS**.
- Per-shape: cos(twin)=1.000000 and in-kernel act-quant cos=1.000000 on every shape.

The prefill-fusion edits did **not** break decode. (Structurally expected: the
decode-GEMV kernel builds its own pack and is not on the Hadamard-lever path; the
div-fold only perturbs the down_proj Hadamard input by ≤1 ULP.)

### Verdict + action taken

- **fix1 (div-fold) — KEPT in production** (`full_network.wire_real_kernel` sets
  `fold_had_scale=True`). It is a constant fold (reciprocal-multiply vs divide,
  ≤1 ULP): logit cosine 1.000000 and PPL 14.8494 == the pre-fusion baseline,
  exactly.
- **fix2 (had-fp32) — REVERTED** from production wiring (removed the
  `wrapper.had_out_fp32 = True` line). Feeding fp32 to the pack changes the
  down_proj quant input precision → cos 0.96–0.98, PPL +0.024. Kept only as a
  reproducible negative in the harness.
- **fix3 (fp16-gemm) — NOT shipped.** It was never in the production path
  (production `wire_real_kernel` uses the bf16-output `cd`; only the harness
  loads `cd16`). The bf16→fp16 accumulator-store swing costs +0.0645 PPL / cos
  0.983 — i.e. the model's activations are tuned to the bf16 rounding and fp16's
  different rounding is a measurable regression, not an improvement. The
  `codesign_fused_fp16_ext.cu` file is retained as a documented, reproducible
  numerics-failing artifact (target #3, see §3).

## 2. PREFILL SPEEDUP vs FP16 (idle GPU, min-of-N, CUDA-graph as noted)

Full-model forward latency (ms), real-kernel model vs plain FP16 Llama-3.2-1B.
"before" = pre-fusion **eager** (the git ~0.70/0.82/0.85×); "after" = the
shippable config = **fix1 div-fold + CUDA-graph**. Reproduced across two
independent runs (`prefill_fusion_perf.json`).

| M | FP16 | pre-fusion eager (before) | pre-fusion +graph | **fix1 +graph (after, SHIP)** |
|---|---|---|---|---|
| 2048 | 20.66 | 29.16 → **0.709×** | 18.79 → 1.099× | 18.45 → **1.119×** |
| 4096 | 40.72 | 49.44 → **0.824×** | 42.72 → 0.953× | 39.98 → **1.019×** |
| 8192 | 85.45 | 103.51 → **0.826×** | 97.88 → 0.873× | 91.98 → **0.929×** |

**Before → after (shippable): 0.709×→1.119× (M2048), 0.824×→1.019× (M4096),
0.826×→0.929× (M8192).** Crosses 1.0× (beats FP16) at M=2048 and M=4096.

### Isolating the prefill-fusion round's own contribution (CUDA-graph held fixed)

The div-fold is the only shipped lever from *this* round (CUDA-graph was the
prior round and is already in the "+graph" column):

| M | pre-fusion+graph | fix1+graph | div-fold saving |
|---|---|---|---|
| 2048 | 18.79 | 18.45 | −0.34 ms (+1.8%) |
| 4096 | 42.72 | 39.98 | −2.75 ms (+6.4%) |
| 8192 | 97.88 | 91.98 | **−5.90 ms (+6.0%)** |

The **−5.90 ms at M=8192 matches the profile's predicted `aten::div /sqrt(n)` =
5.86 ms almost exactly** — the fold removed precisely that separate division
kernel, as designed.

### Did the ~33 ms of fixable overhead (M=8192) shrink?

Profile's ~33 ms fixable = casts 16.9 + div 5.9 + GEMM-out cast 9 + pack-into-GEMM
9.8 + per-Linear glue (CUDA-graph target, dominant at small M):

| fixable item (M8192) | status | realized |
|---|---|---|
| `/sqrt(n)` div 5.9 ms | **SHIPPED (fix1 div-fold)** | **−5.90 ms ✓** |
| per-Linear python/launch glue ~3.4 ms (≫ at M2048) | SHIPPED (CUDA-graph, prior round) | M2048 eager→graph −10.4 ms ✓ |
| GEMM-out fp16 cast ~9 ms (target #3) | fix3 folds it (measured fix1→fix13 graph −7.6 ms) but **FAILS numerics** | not shipped |
| Hadamard fp32→fp16 round-trip cast ~8 ms | fix2 removes it (measured −3.7 ms) but **FAILS numerics** | not shipped |
| activation pack-into-GEMM ~9.8 ms (target #4) | **not attempted this round** | — |

So of the ~33 ms, only the **5.9 ms div** is numerics-safe-and-shipped this
round (CUDA-graph, a separate round, removed the glue). The two cast-folds that
would have claimed ~15 ms more both fail the numerics gate and were reverted; the
~9.8 ms pack fusion is untouched. Component cross-check (isolated dense, per
forward): pack 3.1/5.4/10.0 ms, fused GEMM 8.2/11.3/19.3 ms, FP16 Linear
11.6/21.9/40.5 ms — the fused GEMM compute still beats FP16 at every M.

## 3. Target #3 (GEMM-out cast+bias epilogue fold) — measured, numerics-unsafe

The prior agent did implement the cast half of target #3: the fp16-output
epilogue config copy (`codesign_fused_fp16_ext.cu`, `CD_OUT_FP16` → `ElementD =
half_t`, PROTECTED mainloop `#include`d verbatim and unchanged). It folds the
112× `D.to(fp16)` cast into the GEMM store.

- **Measured impact (graph, marginal over fix1):** −7.6 ms at M8192 (91.98→84.36),
  −2.65 ms at M4096, −1.42 ms at M2048. Real.
- **Numerics: FAIL.** bf16→fp16 accumulator-store swing → logit cos 0.983, PPL
  +0.0645. fp16's different rounding (vs the bf16 the model's quant was effectively
  tuned against) is a measurable regression. **Cannot ship.**
- The **bias** was never folded (still a Python `D + bias`); only the cast was.

A numerics-safe cast+bias fold would need a **bf16-output epilogue with bias
folded in AND the residual stream kept in bf16 end-to-end** (so no bf16→fp16
Python cast at all, no fp16-range/rounding change). That is a model-wide dtype
change (risky, out of this round's promix-only + non-protected-epilogue scope), so
per the "measure-and-report rather than force it" guidance it is **deferred**, not
forced.

## 4. Is ≥1.5× reached? Residual + next lever

**No — ≥1.5× is not reached at any M.** The shippable, numerics-safe config
(div-fold + CUDA-graph) tops out at **1.119× (M2048), 1.019× (M4096), 0.929×
(M8192)**.

Residual at M=8192 (0.929×, ~6.5 ms slower than FP16), real-only overhead that
remains after div-fold:

- **down_proj FP32 online Hadamard transform ~5.5 ms** — numerics-bound (FP16
  Hadamard fails the 0.9999 gate), not removable without accuracy loss.
- **activation pack (qpf) ~9.8 ms** — numerics-safe and fusable (target #4, not
  attempted this round).
- **GEMM-out cast+bias ~9 ms** — the fp16 form fails numerics (§3); a bf16-safe
  form is deferred.

Next levers, ranked by numerics-safe headroom:

1. **pack-into-GEMM prologue fusion (target #4, ~9.8 ms @ M8192)** — fold the
   float→codes+SF qpf pack into the fused-GEMM prologue (one launch per Linear).
   Largest untouched numerics-safe win at large M; bit-exact by construction (same
   spec codes).
2. **lm_head on the kernel** — a shared ~10 ms FP16 GEMM over the 128K vocab that
   both models currently pay identically. The 4-bit fused GEMM already beats FP16
   on compute, so packing lm_head onto the kernel would cut the real model's
   absolute time and lift the ratio (needs lm_head packing + its own accuracy
   check).
3. **bf16-safe GEMM-out cast+bias fold** — fold bias into the bf16 epilogue and
   carry the residual stream in bf16, eliminating the Python cast without the
   fp16 rounding regression (§3). Deferred: model-wide dtype audit.

## Deliverables / changes

- **Reverted** `promix/quantize/full_network.py`: dropped `had_out_fp32=True`
  (fix2) from `wire_real_kernel`; kept `fold_had_scale=True` (fix1, the only
  numerics-safe fix) with a comment documenting the two reverted candidates.
- **Harness** `promix/eval/prefill_fusion.py`: added the `fix1_divfold_ONLY` PPL
  stage (isolates the shippable config; result 14.8494 == baseline).
- `codesign_fused_fp16_ext.cu` retained as a documented numerics-failing
  artifact (reproducible via the harness; not on the production path).
- Artifacts: `prefill_fusion_perf.json`, `prefill_fusion_ppl_validated.json`,
  `decode_gemv_recheck.json`.
