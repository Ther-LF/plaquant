# Round-12 — PERFORMANCE axis: phase-adaptive DECODE (weight-only W4A16 GEMV)

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.11+cu130,
transformers 4.48.1. All perf numbers: idle GPU (one task/GPU), CUDA-graphed,
median over the generation, prompt 32 / gen 128. New code:
`kernels/mixed_gemm_sm120/decode_gemv_ext.cu` (`decode_gemv_w4a16` kernel),
`promix/quantize/mixed_linear.py` + `promix/eval/decode_merge.py`
(`PROMIX_DECODE_MODE` switch), `promix/quantize/full_network.py`
(`FakeMixedLinear.ACT_EXACT` for the PPL A/B), harness
`promix/eval/decode_w4a16.py`.

## The design mismatch (what the lead flagged)

At DECODE (M=1) the GEMV is **bandwidth/latency-bound, not compute-bound**, so
quantizing the activation (our shipped **W4A4** decode path) buys **no** tensor-core
throughput — it only adds a per-token activation-quantize pass and loses accuracy.
Weight-only methods (AWQ/GPTQ W4A16 via Marlin) skip it and beat us at decode
(b1 ours 0.85/0.96/1.23 vs Marlin 1.55/1.75/2.53 for 1B/3B/8B). Our activation
quantization only pays at **PREFILL** (compute-bound, tensor cores). Fix: a
**phase-adaptive** decode path — weight-only W4A16 at M=1, W4A4 kept at prefill.

## What was built

`decode_gemv_w4a16` — a weight-only M=1 GEMV. Body identical to the shipped W4A4
decode GEMV (`gemv_w4a4_v2`: warp-per-row, uint32 weight loads, shared activation
staging, smem opt-in for K>12k) **except** it stages the raw fp16/bf16/fp32
activation (converted to fp32 in-register) instead of quantizing it. It streams the
**same** packed 4-bit NVFP4 (low) + 8-bit MXFP8 (high) weights at HBM BW and
dequantizes them in-register (Marlin-style), dotting against the **exact** activation.
No activation quantize, no SFA. One launch.

Wired behind `PROMIX_DECODE_MODE=w4a16|w4a4` (default **w4a16**) in both
`MixedQuantLinear._decode_gemv` (per-proj) and `MergedQuantLinear.forward` (the
round-7 qkv/gate-up merge). **Prefill (M>1) is completely untouched** — it always
runs the W4A4 fused tensor-core GEMM where the activation quant pays.

## Numerics gates — ALL PASS

1. **Per-shape kernel faithfulness** (size-independent probe, every Linear shape incl.
   8B down_proj K=14336, the 57 KB smem-opt-in path): `decode_gemv_w4a16` vs the exact
   reference `x_fp16 @ W_deq.T` — **cos ≥ 0.9999998, rel ≤ 4e-7** (fp32-accumulation
   noise). The shipped W4A4 decode GEMV on the SAME weight is only cos ≈ 0.9975 vs the
   exact-activation truth, i.e. **W4A16 is bit-faithful; W4A4 carries real activation-
   quant error.**
2. **Graph == eager** (1B/3B): W4A16 graphed decode reproduces eager decode
   **argmax 100 %, min-cos 0.9999924–0.9999926** over 48 steps → CUDA-graph replay
   changes nothing.
3. **Full-model faithfulness vs the exact-activation reference** (teacher-forced,
   same `fed` tokens):

   | size | W4A16 decode vs EXACT-act twin | W4A4 decode vs its QUANT-act twin | W4A4 vs EXACT truth |
   |---|---|---|---|
   | 1B | cos 0.9781 / argmax 89.8 % | cos 0.8889 / 83.7 % | cos 0.9167 |
   | 3B | cos 0.9769 / argmax 91.8 % | cos 0.9547 / 87.8 % | cos 0.9352 |

   The absolute <1.0 is the KNOWN cross-path fp32-GEMV-vs-fp32-matmul accumulation
   drift over all layers (the shipped W4A4 decode's own analog is cos 0.889–0.955;
   decode_graph documented 0.896). **W4A16 is strictly MORE faithful to the exact-
   activation reference than the shipped W4A4 is to its own quantized reference** —
   because W4A16 removes the activation-quant error entirely.

## Accuracy — PPL (fake A/B, GPTQ-frozen-scale shipped wiring, wikitext-2)

Activation-quant is an M-independent property, so the W4A4→W4A16 PPL delta measured
on the fake twin (same W_deq, GPTQ frozen scales) is exactly what the W4A16 decode
kernel realizes. **W4A16 keeps the full mixed 4/8-bit weights (4.97 bits/weight,
1/8 MXFP8 high channels); only the activation quantization is removed.**

| size | W4A4 (act quant ON) | **W4A16 (act EXACT)** | delta | BF16 ref |
|---|---|---|---|---|
| 1B (GPTQ) | 14.577 | **13.659** | **−0.918** | 13.16 |
| 3B (GPTQ) | 8.286 | **7.968** | **−0.318** | 7.82 |
| 8B (GPTQ) | 6.595 | **6.311** | **−0.284** | 6.14 |

The 8B W4A4 6.595 reproduces the scoreboard 6.599. (A fast RTN cross-check gave the
identical delta: 6.848 → 6.571, −0.278 — the W4A4→W4A16 delta is GPTQ-independent since
only the activation quantizer toggles on the same `W_deq`.)

Removing activation quantization at decode recovers **−0.92 / −0.32 / −0.28** PPL for
free. The W4A16 decode PPL (13.66 / 7.97 / 6.31) is close to BF16 and beats the
pure-W4A16 baselines GPTQ (14.22 / 8.20 / 6.52) and AWQ (14.53 / ~8.35 / 6.59) — though
ours spends 4.97 bits/weight vs their 4.0, so this is not a matched-bit claim; the
point is the decode accuracy is now near-lossless on activations.

## Performance — decode b1/b8 (×FP16-graph, graphed, median)

Measured same-run, same FP16-graph baseline, merge ON (the round-7 qkv/gate-up merge).

### b1 (M=1 — the W4A16 target)

| size | W4A4 (shipped) | **W4A16 (new default)** | gain | Marlin (vLLM) |
|---|---|---|---|---|
| 1B | 0.847× | **1.094×** | +0.25 | 1.55× |
| 3B | 0.963× | **1.276×** | +0.31 | 1.75× |
| 8B | 1.168× | **1.450×** | +0.28 | 2.53× |

**W4A16 lifts decode b1 by +0.25–0.31× at every size and pushes all three sizes
above 1.0× FP16** (1B/3B were below before). The win is the removed per-token
activation-quant pass: the W4A16 kernel is the SAME single launch minus the
`quant_act_to_shared` compute (block-scale warp-reductions + E2M1/E4M3 RTN per
element), so the 1B merged decode drops 4.22 → 3.27 ms/tok (1.29× kernel speedup) at
**identical weight HBM traffic**.

### b8

| size | W4A4 | W4A16 |
|---|---|---|
| 1B | 1.387× | 1.385× |
| 3B | 0.980× | 0.976× |
| 8B | 1.170× | 1.164× |

**b8 is unchanged** — at batch 8 each decode step is M=8, which routes through the
prefill fused tensor-core GEMM (W4A4), not the M=1 GEMV. The W4A16 GEMV is M=1 only;
batched decode would need a W4A16 batched kernel (future work). Honest: W4A16 helps
b1, not b8, at this protocol.

## Wiring decision

- **Decode default flips to W4A16** (`PROMIX_DECODE_MODE=w4a16`, the global default in
  `mixed_linear.py`). **Prefill keeps W4A4** (the tensor cores pay there; untouched).
- **The qkv/gate-up merge is KEPT for decode.** Its original rationale (de-duplicate
  the activation quant, quant once vs 3×/2×) is gone under W4A16, but it still reduces
  launches 7→4/layer, and at M=1 (launch-bound) that still wins everywhere:

  | size | W4A16 unmerged b1 | W4A16 merged b1 |
  |---|---|---|
  | 1B | 1.024× | **1.094×** |
  | 3B | 1.172× | **1.276×** |
  | 8B | 1.376× | **1.450×** |

  So the merge's value shifted from quant-dedup to pure launch reduction, but it
  remains a net win — keep it.

## Remaining gap to Marlin (honest diagnosis)

After W4A16, b1 is 1.09 / 1.28 / 1.45× vs Marlin 1.55 / 1.75 / 2.53×. The residual:

1. **Bits/weight: ours 4.97 vs Marlin 4.0 (+24 %).** Decode is weight-bandwidth-bound,
   so the ResQ 1/8 high-precision MXFP8 channels (8-bit) cost ~24 % more weight HBM
   traffic per token than Marlin's pure 4-bit — a direct, structural bandwidth
   disadvantage inherent to our mixed-precision recipe (r=0 pure-NVFP4 would be 4.5
   bits, still +12 %, at an accuracy cost). This is the single biggest lever left.
2. **Launch/occupancy floor at M=1.** Even merged (4 launches/layer), the 1B W4A16
   decode (3.27 ms/tok) is still ~8× off its own 4-bit weight-BW floor — small-N
   q/k/v/o Linears at 1B/3B never leave the ~15 µs launch floor (round-8). Marlin runs
   a more aggressively fused single-graph decode; our 4 GEMVs + FP16 ops still pay
   per-launch latency. The gap narrows most on 8B (1.45×) where weights are large
   enough to be genuinely bandwidth-bound.
3. **FP16 attention + RMSNorm + RoPE + lm_head (~45 % of the per-token decode,
   round-8) is unaccelerated** — pure Amdahl drag. vLLM's Marlin stack has
   fused/optimized attention + norms + sampling; shrinking ours needs a FlashAttention/
   fused-norm decode path (helps regardless of quantization).
4. **Kernel layout.** Our decode GEMV is a hand-written warp-per-row uint32 kernel
   reaching 44–79 % HBM on large Linears; Marlin's interleaved-fp16 weight layout +
   GPTQ permutation gives near-perfect coalescing and mma-friendly in-register dequant.
   Closing this is a kernel-rewrite lever (adopt a Marlin-style packed layout for the
   decode GEMV), orthogonal to the W4A16 change landed here.

**Net:** W4A16 was the correct phase-adaptive fix — it removes a pass that bought
nothing at decode, recovers ~0.3–0.9 PPL, and lifts b1 above FP16 at all sizes. The
remaining Marlin gap is (1) our +24 % bits, (2) the M=1 launch floor, (3) FP16
attention/norm Amdahl, (4) a less-tuned GEMV layout — none a one-line fix, and (1)/(2)
are intrinsic to the mixed-precision W4A4-at-prefill design.

## Gates summary

- **Accuracy:** PPL W4A16 < W4A4 at every size (−0.92 / −0.32 / <8B>), as predicted.
  Per-shape kernel cos ≥ 0.9999998; graph==eager argmax 100 % / cos 0.99999;
  W4A16 strictly more faithful to the exact reference than W4A4. **PASS.**
- **Prefill untouched** (W4A4 path unchanged; the switch only affects M=1). PASS.
- **Perf:** idle GPU, graphed, median, same-run FP16 baseline. b1 reproducible
  +0.25–0.31× over W4A4 at all three sizes.

## Scoreboard deltas (axis 2, decode)

- Decode b1 ×FP16-graph, **W4A16 (new default, merged):** 1B/3B/8B
  **1.09 / 1.28 / 1.45×** (was W4A4 0.85 / 0.96 / 1.23). First time 1B/3B b1 > 1.0×.
- Decode b8 unchanged (M=8 uses the prefill GEMM): 1.39 / 0.98 / 1.16×.
- Accuracy (axis 1), decode path: activation quant OFF at decode recovers PPL to
  13.66 / 7.97 / 6.31 (W4A4 14.58 / 8.29 / 6.59, GPTQ).
- Still behind Marlin b1 (1.55/1.75/2.53): gap is +24 % bits + M=1 launch floor +
  FP16 attention Amdahl + GEMV layout.

## Artifacts (remote `/home/luofan/`)

- `r12_probe_1b.json` (per-shape kernel probe).
- `r12_num_1b.json`, `r12_num_3b.json` (full-model numerics, both gates PASS).
- `r12_np_1b.json`, `r12_np_3b.json`, `r12_np_8b.json` (perf matrix {w4a4,w4a16} ×
  {unmerged,merged} × {b1,b8}; 8B also carries the probe incl. K=14336).
- `r12_ppl_1b.json`, `r12_ppl_3b.json`, `r12_ppl_8b.json` (PPL fake A/B).
- Code: `decode_gemv_ext.cu` (`decode_gemv_w4a16`), `mixed_linear.py` +
  `decode_merge.py` (`PROMIX_DECODE_MODE`), `full_network.py`
  (`FakeMixedLinear.ACT_EXACT`), `promix/eval/decode_w4a16.py` (harness).
