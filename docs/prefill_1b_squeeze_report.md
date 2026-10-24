# 1B-Prefill Squeeze — Pack-Fusion + MXFP8 lm_head (validated)

> 2026-10-24. Next 1B-prefill round of the ProMix e2e perf loop. Floor coming in:
> div-fold shipped (−5.9 ms @ M8192, bit-exact); prefill 1.12×/1.02×/0.93× at
> M=2048/4096/8192 (`docs/prefill_fusion_validated_report.md`). CUDA-graph in place,
> PPL=14.85. This round attacks the two remaining numerics-safe levers the prior
> report ranked #1 (pack-into-GEMM) and #2 (lm_head on the kernel).
>
> Remote-only (RTX PRO 6000, SM120), container `luofan-plaquant`, idle GPU, CUDA-graph
> on, min-of-N. Numerics gated FIRST after every change. Artifacts:
> `kernels/mixed_gemm_sm120/{pack_fusion_iso,prefill_squeeze_perf,lmhead_probe,lmhead_real_ppl}.json`.

## TL;DR

| lever | what | ms saved (M8192, e2e graph) | numerics | shipped? |
|---|---|---|---|---|
| 1 pack-fusion | single-launch fused fp4+fp8 activation pack | −0.21 (−1.4 @ iso small-M) | **bit-exact** (torch.equal; logit cos 1.000000) | **yes** |
| 2 lm_head MXFP8 | W8A8 pure-MXFP8 vocab GEMM replacing FP16 | −1.16 | real-kernel PPL 14.9021 (+0.053 ≤14.95); real-vs-twin cos 0.999998 | **yes (W8)** |

**New 1B prefill speedup vs FP16 (within-run before→after, CUDA-graph):**

| M | base (div-fold ship) | **+pack-fusion +MXFP8 lm_head** | Δ |
|---|---|---|---|
| 2048 | 1.054× | **1.085×** | +0.031× |
| 4096 | 0.982× | **1.002×** (crosses 1.0×) | +0.020× |
| 8192 | 0.925× | **0.939×** | +0.014× |

Both levers hold numerics at every M; W4A4 lm_head does NOT (reported, not shipped).

## 1. Prior art (per RESEARCH_TOOLS.md)

- **KernelWiki**: flashinfer **PR-2233** *"Fused RMSNorm + FP4 Quantization Kernels in
  CuTe-DSL"* (sm120) and sglang **PR-17449** *"mxfp8 online quantization"*. Both fuse
  the activation quant into the **PRODUCER** (RMSNorm) epilogue, NOT into the GEMM
  consumer prologue. This is the decisive signal: the SM120 block-scaled GEMM mainloop
  consumes block-scaled operands via TMA (pre-quantized codes + SF in gmem); there is
  no HW path to quantize-in-prologue without abandoning the block-scaled TMA load.
- CUTLASS `fused_gemm_prepacked` / `pure_presf` confirm the GEMM reads pre-interleaved
  SF; the pack already writes the exact interleaved atom layout it consumes.

**Verdict on lever 1 scope:** a true pack-INTO-GEMM-prologue fusion is NOT feasible
within the protected-mainloop + promix-only scope (it would require rewriting the
block-scaled TMA pipeline). We land the task's named **cheaper intermediate**
(single-launch fused pack) and, separately, confirm the pack already writes zero-copy
into the GEMM-consumed layout. The producer-side fusion (fold pack into the RMSNorm /
Hadamard epilogue, flashinfer-style) is the next numerics-safe step and is called out
in §5 — it is a bigger, cross-op change deferred out of this round's scope.

## 2. Lever 1 — single-launch fused activation pack

`kernels/mixed_gemm_sm120/quant_pack_fused_ext.cu` (NEW, non-protected). One CUDA
launch (`grid.z` selects segment) produces BOTH the NVFP4 (low 7/8) and MXFP8 (high
1/8) packed codes + interleaved SFA, replacing the two separate
`quant_pack_float_fp4` / `quant_pack_float_fp8` launches (224→112 pack launches/forward).
The per-block absmax reduce / scale / per-element cast are the **identical device code**
as `quant_pack_float_ext.cu` (duplicated verbatim, not the protected mainloop).

**Numerics — bit-exact by construction.** `pack_fusion_iso.py`: `torch.equal` on all
four outputs (data_lo, sfa_lo, data_hi, sfa_hi) across the 7 Llama-3.2-1B shapes × M ∈
{2048,4096,8192}, fp16: **BITEXACT_ALL True**. Whole-model prefill logit cosine vs the
2-launch base: **1.000000 / 1.000001 / 1.000001**.

**Isolated latency (summed ×16 layers, min-of-N):**

| M | 2-launch | 1-launch | save |
|---|---|---|---|
| 2048 | 3.045 ms | 2.739 ms | −0.307 ms (+10.1%) |
| 4096 | 5.360 ms | 4.876 ms | −0.484 ms (+9.0%) |
| 8192 | 10.040 ms | 9.635 ms | −0.405 ms (+4.0%) |

The win is launch-overhead-bound: the six small Linears each drop ~2 µs/launch, while
the single big down_proj pack (K=8192, memory-bound) is flat (238→241 µs). Under
CUDA-graph the host-launch overhead is already gone, so the **e2e** graph gain is
smaller than the iso gain (−0.21 to −0.31 ms, from reduced GPU-side kernel-tail gaps),
but it is free (bit-exact) and strictly positive, so it ships.

## 3. Lever 2 — lm_head on the kernel (MXFP8 W8A8)

lm_head is a shared ~10 ms FP16 GEMM (vocab 128256 × hidden 2048) that FP16 and the
real model pay identically; moving it onto the block-scaled tensor cores cuts the real
model's absolute time and lifts the ratio. Its input (post final-RMSNorm hidden state)
is **NOT rotated/reordered**, so the ResQ high-variance-channel assumption does not hold
and 4-bit is risky.

**Accuracy gate FIRST (fake-quant twin, bit-faithful to the real kernel), wikitext-2
PPL, GPTQ decoder, real-kernel decoder + div-fold:**

| lm_head | PPL | Δ vs baseline | gate (≤14.95) |
|---|---|---|---|
| FP16 (baseline) | 14.8494 | — | — |
| **W4A4-mixed** (NVFP4 7/8 + MXFP8 1/8) | 15.1444 | **+0.2950** | **FAIL** |
| **W8 (MXFP8-only)** | **14.8976** | **+0.0482** | **PASS** |

W4A4 on the un-rotated head regresses +0.295 (reported, **not shipped**). **W8 MXFP8
passes** (+0.048). Real-kernel confirmation: `MxFp8Linear` (pure MXFP8 prepacked GEMM,
`lmhead_mxfp8_ext.cu`) vs the fake-W8 twin → whole-model logit cosine **0.999998** at
every M, and end-to-end **real-kernel** wikitext-2 PPL (`lmhead_real_ppl.json`):
decoder-real + FP16 head **14.8494**, decoder-real + **MXFP8 head 14.9021 (+0.0527,
PASS ≤14.95)**. lm_head IS quantizable safely — as W8, not W4.

`MxFp8Linear` (`promix/quantize/lmhead_linear.py`): per forward one
`quant_pack_float_fp8` pack (interleaved SFA) + one pure-MXFP8 prepacked GEMM
(`mxfp8_gemm_prepacked`, activation+weight SF pre-interleaved, no per-call re-interleave).
The sync is `cudaStreamIsCapturing`-guarded so it is CUDA-graph safe.

**e2e saving (graph):** −0.21 ms (M2048) / −0.59 ms (M4096) / −1.16 ms (M8192). Smaller
than the raw ~10 ms FP16 lm_head because (a) the MXFP8 GEMM is ~2× not ∞× faster and
(b) the bf16→fp16 output cast over the (M × 128256) logits is itself large — but net
positive and PPL-safe.

## 4. Full-model prefill vs FP16 — before → after (CUDA-graph, min-of-N)

Within one harness run (`prefill_squeeze_perf.py`), real wikitext ids. NOTE: the shared
box had co-tenant load during this run, so the **absolute** FP16 and base numbers differ
from the cleaner-GPU shipped report (base here 1.054/0.982/0.925× vs report
1.119/1.019/0.929×); the **within-run before→after deltas** are the rigorous metric.

| M | FP16 (ms) | base (div-fold) | +pack-fusion | +MXFP8 lm_head (ship) | cos(pf vs base) | cos(lm vs twin) |
|---|---|---|---|---|---|---|
| 2048 | 19.077 | 18.097 (1.054×) | 17.792 (1.072×) | **17.582 (1.085×)** | 1.000000 | 0.999999 |
| 4096 | 39.314 | 40.019 (0.982×) | 39.839 (0.987×) | **39.247 (1.002×)** | 1.000001 | 0.999999 |
| 8192 | 85.089 | 91.951 (0.925×) | 91.738 (0.928×) | **90.575 (0.939×)** | 1.000001 | 0.999998 |

**Both levers improved e2e at every M.** Combined base→ship: −0.515 ms (M2048,
1.054→1.085×), −0.772 ms (M4096, 0.982→**1.002×** — crosses FP16), −1.376 ms (M8192,
0.925→0.939×). Decode M=1 path and the shipped div-fold are untouched (pack-fusion is a
prefill-only MixedQuantLinear option; decode still uses the bit-exact decode-GEMV).

Per-lever e2e (graph): pack-fusion −0.31/−0.18/−0.21 ms; lm_head −0.21/−0.59/−1.16 ms.

## 5. Residual vs the ~1.9× Amdahl ceiling + what's left

Still short of 1.9× at every M. After this round the M=8192 real prefill (90.6 ms vs
FP16 85.1 ms, 0.94×) is dominated by residual that is NOT the quant GEMM (which beats
FP16 3×):

- **down_proj FP32 online Hadamard ~5.5 ms transform** — numerics-bound (FP16 fails the
  0.9999 gate), not removable without accuracy loss.
- **activation pack ~9.6 ms** — now single-launch, but the big down_proj pack is
  memory-bound; the remaining numerics-safe win is **producer-side fusion** (fold the
  pack into the RMSNorm / Hadamard epilogue, flashinfer PR-2233 pattern) to eliminate
  the fp16-activation gmem round-trip. Cross-op change, deferred.
- **GEMM-out cast+bias ~9 ms** — the fp16-epilogue form fails numerics (prior round); a
  bf16-safe residual-stream form needs a model-wide dtype audit (deferred).
- **shared FP16 floor** (attention flash ~14 ms, RMSNorm/RoPE/SiLU/residuals) — paid by
  FP16 too; this is the Amdahl wall. lm_head is no longer fully in this floor (now MXFP8).

**Ranked next levers (numerics-safe headroom):** (1) producer-side pack fusion
(RMSNorm→pack), (2) bf16-safe GEMM-out cast+bias fold, (3) the down_proj Hadamard
transform is the irreducible ~5.5 ms core.

## 6. Deliverables / changes

- NEW `kernels/mixed_gemm_sm120/quant_pack_fused_ext.cu` — single-launch fused pack (lever 1).
- NEW `kernels/mixed_gemm_sm120/lmhead_mxfp8_ext.cu` — pure-MXFP8 prepacked GEMM (lever 2),
  CUDA-graph-safe sync guard.
- NEW `promix/quantize/lmhead_linear.py` — `MxFp8Linear` (W8A8 lm_head).
- `promix/quantize/mixed_linear.py` — optional `qpfm` (fused pack) path in `MixedQuantLinear`.
- `promix/quantize/full_network.py` — `wire_real_kernel(..., qpfm=)` + `wire_lmhead_mxfp8`.
- NEW harnesses: `promix/eval/lmhead_probe.py` (PPL gate), `kernels/.../pack_fusion_iso.py`
  (bit-exact+iso), `promix/eval/prefill_squeeze_perf.py` (e2e), `promix/eval/lmhead_real_ppl.py`.
- Artifacts: `pack_fusion_iso.json`, `prefill_squeeze_perf.json`, `lmhead_probe.json`,
  `lmhead_real_ppl.json`.
