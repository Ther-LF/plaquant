# RD Kernel-Fix — K_high==0 pure-NVFP4: eliminate the silent 1/8 fallback

**Branch:** `feat/rd-kernelfix` (off `origin/feat/rd-alloc`) · **HW:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, dedicated worktree
`/home/luofan/plaquant-kfix`, idle GPU 5 · **Date:** 2026-10-29

## TL;DR

A Linear configured with **K_high==0** (pure-NVFP4, 0 MXFP8 rings — produced by
rate-distortion allocation at low budgets) used to **crash** the two-phase fused
kernel and then **silently fall back to a uniform 1/8 split** (the perf sweep's
"r=0" row carried `avg_fp8_realized=0.125` — a served≠evaluated mismatch).

Fixed by routing K_high==0 to the **already-compiled single-phase NVFP4 path**
(`GemmSingle = CodesignFusedKernel<DualPhase=false>`), which **reuses the fused
kernel's tuned NVFP4 Phase-1 collective with Phase-2 (MXFP8) skipped at compile
time**. No silent fallback remains; the symmetric K_low==0 (pure-MXFP8) raises a
clear error.

| check | result |
|---|---|
| K_high==0 output vs fake-quant pure-NVFP4 ref | **cosine 0.99995** (qkv/gate/down), served as true pure-NVFP4 |
| zero-regression K_high=1/8 two-phase fused | cosine 0.99995 (unchanged) |
| K_low==0 pure-MXFP8 | raises a **clear ValueError** (no silent clamp) |
| **GEMM-level** pure-NVFP4 (skip-P2) vs 1/8 (summed over 1B net, M=2048) | **0.821× — 18% FASTER** |
| GEMM-level stock dedicated NVFP4 dispatcher vs 1/8 | 1.13× SLOWER (= 1.37× the skip-P2 path) |
| **full-model prefill TTFT** r=0 vs 1/8 (Llama-3.2-1B, M=2048) | **0.929× — 7.1% FASTER** (realized K_high=[0]) |

## Root cause

Two stacked defects for a configured K_high==0 Linear:

1. **Crash.** The two-phase fused NVFP4+MXFP8 path unconditionally builds the
   MXFP8 segment:
   - `weight_pack.pack_weight_mixed` → `cd.prepack_sf_fp8_b(SFB_hi, N, 0)` launches
     a 0-block grid → *"interleave_sfb launch: invalid argument"*;
   - the per-forward activation pack `quant_pack_float_fp8(..., K_high=0)` → *"qpf
     launch: invalid argument"*;
   - the fused GEMM asserts in CUTLASS TMA setup (`gmem_prob_shape[0] >= 1`) — the
     Phase-2 MXFP8 pack + TMA descriptor + SF layout all assume K_high>0.

2. **Silent fallback.** On that crash the perf harness (`rd_perf.measure_latency_for`)
   caught the exception and retried with `_aligned_uniform_chosen(..., n=max(1,…))`
   — i.e. forced ≥1 MXFP8 ring per Linear = the uniform **1/8** config. The timed
   model was therefore 1/8 (`avg_fp8_realized=0.125`, `bpw=4.969`) while labeled
   r=0: the headline "r=0 prefill ≈ 1/8" was the 1/8 latency **relabeled**, not a
   measurement. `full_network._khigh_for` was separately hardened earlier to honor
   a configured 0 (not collapse to the global 1/8), but the pack/dispatch still
   crashed below it, so the net effect was still the harness fallback.

## Diagnosis: why the "dedicated pure-NVFP4 is 24–65% slower" was an artifact

The perf-puzzle's only true K_high=0 path was the **stock** `nvfp4_gemm_ext`
dispatcher — a default-config CUTLASS block-scaled collective, NOT the
~15-round-tuned fused kernel. At our 1B–8B single-seq shapes the GEMM is not
compute-bound (TC 10–42%), so **kernel tuning dominates and the format's compute
saving is invisible** — a stock kernel with *less* math can still be *slower*. Our
GEMM sweep confirms it directly (summed over the 1B network, M=2048):

| path | summed GEMM (ms) | vs 1/8 |
|---|---|---|
| 1/8 dual fused (shipped) | 9.69 | 1.00× |
| **pure-NVFP4 skip-Phase-2 (this fix)** | **7.96** | **0.82× (18% faster)** |
| pure-NVFP4 stock dedicated dispatcher | 10.94 | 1.13× (slower) |

The stock dispatcher is **1.37× the skip-P2 path** — so its reported slowness is a
stock-kernel implementation artifact, not a datapath truth. Routing to the tuned
skip-Phase-2 path is what finally surfaces the format lever.

## The fix (skip-Phase-2 reuses the tuned kernel — a *few lines*, no mainloop surgery)

The fused kernel template `codesign::CodesignFusedKernel<..., bool DualPhase>` was
**already** designed to run single-format NVFP4 with Phase-2 elided at compile time
(`if constexpr (DualPhase)` guards the MX problem-shape / TMA descriptor / load /
mma / phase barriers; `GemmSingle = GemmUniversalAdapter<FusedKernel<false>>` was
already instantiated and even exposed as `single_nvfp4_gemm`). So the "skip-Phase-2
fast path" the coordinator asked about is **not a deep rewrite — it already exists**.
The only missing piece was a prepacked entry + the Python wiring.

Changes:

- **`kernels/mixed_gemm_sm120/codesign_fused_ext.cu`** — add
  `single_nvfp4_gemm_prepacked(A, physSFA, B, physSFB, M,N,K, alpha)`: the existing
  `GemmSingle` (Phase-2 skipped) consuming an **already-interleaved** activation SF
  (BSConfigNV atom) — byte-compatible with what `quant_pack_float_fp4` emits for the
  NVFP4 segment of the fused path, so no double interleave. `g_w` rides the epilogue
  alpha exactly as the dual path.
- **`promix/quantize/weight_pack.py`** — `pack_weight_mixed`: K_high==0 builds EMPTY
  MXFP8 buffers and skips `prepack_sf_fp8_b(...,0)` (the crash); W_deq == deq_lo.
  K_low==0 raises a clear `ValueError` (no single-phase MXFP8 kernel exists).
- **`promix/quantize/mixed_linear.py`** — `MixedQuantLinear.is_pure_nvfp4`: pack only
  the NVFP4 segment and dispatch `single_nvfp4_gemm_prepacked`; K_low==0 refused;
  decode (M==1) routed to the single-phase prefill GEMM (its decode-GEMV high buffers
  are empty). **Zero change for K_high>0** — the normal two-phase fused path is
  byte-identical.
- **`kernels/mixed_gemm_sm120/khigh0_probe.py`** — correctness + zero-regression +
  K_low==0-error probe.
- **`kernels/mixed_gemm_sm120/rd_kfix_latency.py`** — GEMM-level skip-P2 vs stock vs 1/8.
- **`promix/scripts/rd_kfix_e2e.py`** — full-model prefill r=0 vs 1/8 (prints realized
  K_high to prove r=0 is served, not faked).

## Before / after

- **Before:** K_high==0 → crash → harness serves 1/8, labeled r=0 (`avg_fp8=0.125`).
  "Is pure-NVFP4 faster than 1/8?" was unanswerable (r=0 was 1/8 by construction).
- **After:** K_high==0 runs true pure-NVFP4 (cosine 0.99995 vs fake-quant ref). The
  honest answer: pure-NVFP4 is **18% faster at the GEMM level** and **7.1% faster at
  full-model prefill (TTFT)** than 1/8 on Llama-3.2-1B at M=2048. This matches the
  perf-puzzle's predicted ~7% e2e ceiling — now actually realized, not a fallback.

## Scope / follow-ups

- Tier-1 (no silent fallback) and Tier-2 (correct + fast pure-NVFP4 routing) are both
  done. The skip-Phase-2 path was a few lines because the kernel already supported
  `DualPhase=false`.
- K_low==0 (pure-MXFP8) is a hard error by design (no single-phase MXFP8 kernel in the
  fused module); add `launch_best_mxfp8`-based routing if that config becomes needed.
- The `rd_perf` harness still has the uniform-at-b retry; with this fix it no longer
  triggers for pure-NVFP4 (attempt-0 now succeeds). The retry remains a flagged
  (`uniform_fallback=True`) last resort for genuinely unsupported splits — it should
  not be merged into an r=0 row without honoring that flag.
