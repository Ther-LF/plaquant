# Round-7 — PERFORMANCE axis (decode launch-merge) + 1B two-level accuracy closeout

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`, torch 2.11+cu130,
transformers 4.48.1. All perf numbers: idle GPU (one task per GPU), CUDA-graphed,
median. New code: `promix/eval/decode_merge.py` (decode QKV+gate/up merge harness),
`promix/eval/prefill_stage_profile.py` (always-wire + 8B hpk28 fused-producer path).

## TL;DR

1. **1B two-level weight scale: real-kernel PPL 14.760 → 14.563 (−0.198)**, real-vs-twin
   gate PASS (|Δ|=0.014). Far exceeds the expected −0.05..−0.1. **Flip 1B default ON.**
   The round-6 1B gate-OFF was only because the base model was offline; it is restored.
2. **Decode fix — QKV + gate/up GEMV MERGE (7→4 launches/layer, activation quantized
   once).** Decode M=1 merge is **BIT-EXACT** (max|Δ|=0.0). Measured **1.17–1.22×**
   decode speedup at every size/batch:

   | size | b1 OFF→ON (×FP16-graph) | b8 OFF→ON | merge gain b1/b8 |
   |---|---|---|---|
   | 1B | 0.724 → **0.848** | 0.653 → **0.794** | 1.17 / 1.22× |
   | 3B | 0.833 → **0.964** | 0.766 → **0.935** | 1.16 / 1.22× |
   | 8B | 1.033 → **1.233** | 0.910 → **1.066** | 1.19 / 1.17× |

   b1 target ≥1.0 reached on **8B** (1.233) and nearly on **3B** (0.964); 1B 0.848
   (its Linears are too small to leave the launch-floor even merged). b8 ≥1.1 reached
   on none but all improved; 8B b8 1.066.
3. **Prefill: the merge HURTS prefill (0.81/0.83/0.92×) — it is a decode-ONLY lever**
   (prefill is compute-bound, not launch-bound). The prefill stage profile shows the
   remaining NVFP4 gap is NOT cheaply addressable this round (Amdahl FP16 attention +
   RMSNorm + the 8B down-Hadamard occupancy wall). Prefill speedups unchanged.

---

## Task 1 — 1B two-level weight-scale PPL (accuracy closeout)

Harness `full_network_realkernel.py --phases ppl` (GPTQ frozen scales, wikitext-2),
the identical protocol/gate as the multisize PPL row. A/B on `PROMIX_WEIGHT_2LEVEL`:

| 1B (fp-perhead, r=1/8) | stock fake | fake twin | **real kernel** | real−twin gate |
|---|---|---|---|---|
| single-level (`=0`) | 14.7617 | 14.7647 | **14.7604** | 0.0043 ✓ |
| **two-level (`=1`)** | 14.5735 | 14.5770 | **14.5629** | 0.0140 ✓ |

- **Δ real-kernel = −0.1975**; Δ fake = −0.1882. Real-vs-twin gate PASS both sides
  (|Δ|=0.004 / 0.014 ≤ 0.1), so the two-level dequant ships faithfully through the
  packed kernel on 1B exactly as it did on 3B/8B in round-6.
- The single-level real PPL 14.7604 reproduces the shipped scoreboard 1B 14.76, so the
  shipped 1B number updates **14.76 → 14.563**. Default flips **ON** for 1B
  (`PROMIX_WEIGHT_2LEVEL=1`, already the global default), completing the accuracy table
  — all three sizes now run two-level.

## Task 2 — decode QKV + gate/up launch-merge

**Why.** The graphed decode breakdown (decode_cudagraph_report / multisize) is
GEMV-dominated: 1B graphed 4.75 ms/tok = GEMV 3.66 ms (77%) + attn/KV/norm/rope/lm_head
1.09 ms, and at M=1 each of the 112 per-token GEMVs is launch/occupancy-floor-bound
(11.3× off the weight-BW floor), not HBM-bound. So the lever is **fewer, larger
launches**, not a faster kernel. q/k/v read the SAME input (input_layernorm) and
gate/up the SAME input (post_attn_rmsnorm), with NO per-proj Hadamard/reorder/fake-quant
(`wire_real_kernel` disables the wrapper quantizer; only o_proj reorders, only down_proj
Hadamards). So per layer:

```
q,k,v,o,gate,up,down  (7 GEMV launches)  ->  qkv(1), o(1), gate_up(1), down(1)  (4)
```

**How.** `MergedQuantLinear` concatenates the already-packed per-proj buffers along the
output (row) dim N; one `decode_gemv_fused` (b1) / fused GEMM (b8) launch produces the
concatenated output, split back to q/k/v (or gate/up). The shared activation is
quantized **once** (vs 3×/2×). `LlamaAttention.forward`/`LlamaMLP.forward` are patched
to route through the merged modules; graph-capturable (buffers concatenated once at
setup, hot loop = one kernel + a slice).

**Numerics gate (first).**
- **Decode (M=1): BIT-EXACT.** Per-module merged-vs-unmerged max|Δ|=0.0 for qkv AND
  gate/up at every layer; full-model single-step logits max|Δ|=0.0, argmax match.
  (The round-6 two-level global `g` is pre-folded into the per-row FP32 decode scales
  `S_lo_dec/S_hi_dec`, so concatenation needs no shared epilogue scalar.) **GATE PASS**
  all sizes.
- **b8 / prefill (M>1):** the fused GEMM carries per-proj `g` as a post-GEMM per-row
  multiply (one epilogue scalar can't serve three different `g`); per-module cos
  0.999989, and an offline PPL A/B (1B, M>1 path) measured **+0.022 PPL** — negligible,
  the same order as the shipped real-vs-twin drift.

**Result (graphed, idle, median, prompt 32 / gen 128, ×FP16-graph):** see TL;DR table.
Merge gain is a robust **1.16–1.22×** across all sizes and both batches. The decode-only
nature is confirmed by the prefill measurement below.

**Which bucket was attacked:** the GEMV launch/occupancy floor (the dominant decode
bucket). Merge cuts launches 7→4/layer and de-duplicates the activation quant. The
remaining gap to ≥1.0× on 1B/3B is the residual per-layer launch floor of the 4 merged
GEMVs + FP16 attention/lm_head — on 1B the Linears are simply too small (q/k/v ~0.5–1.5
MB) to leave the launch floor; 8B's larger Linears convert the saving into 1.23× (b1).

## Task 3 — prefill stage profile @ M=8192 (3B, 8B)

`prefill_stage_profile.py` per-stage CUDA-event breakdown, eager (relative buckets;
graphed headline speedups are the scoreboard's). Numbers = ms/forward summed over all
layers, min-of-20 iteration.

**3B (real 211.6 ms vs FP16 229.9 ms, eager):**

| stage | ms | % | note |
|---|---|---|---|
| attn sdpa (QK·softmax·PV) | 48.7 | 23.0 | **FP16, same as baseline — Amdahl** |
| Linear GEMM (q/k/v/o/gate/up) | 39.7 | 18.8 | mixed GEMM |
| down_proj (fused producer+GEMM) | 32.7 | 15.4 | |
| RMSNorm ×2 | 28.7 | 13.6 | FP16, memory-bound, unaccelerated |
| activation pack | 15.0 | 7.1 | |
| RoPE 13.0, lm_head 16.1 | | 6+8 | FP16, unaccelerated |

Mixed GEMM on the 6 non-down Linears is **~1.36× faster than cuBLAS FP16**
(real Linear path ≈ 86 ms vs FP16 118 ms). The gap to NVFP4's 2.21× is the ~50 % of the
forward that is **unaccelerated FP16** (attention 23 % + RMSNorm 13.6 % + RoPE + lm_head),
identical in both models — pure Amdahl drag that quantization cannot touch.

**8B (real 402.4 ms vs FP16 468.9 ms, eager):**

| stage | ms | % | note |
|---|---|---|---|
| **down_proj (incl. FP32 Hadamard 87.1 ms)** | **121.1** | **30.1** | I=14336 online Hadamard |
| Linear GEMM (q/k/v/o/gate/up) | 95.9 | 23.8 | mixed GEMM |
| attn sdpa | 61.4 | 15.2 | FP16, Amdahl |
| activation pack | 29.3 | 7.3 | |
| RMSNorm ×2 25.7 / lm_head 18 / RoPE 17 | | | FP16 |

The dominant 8B bucket is the **down_proj FP32 online Hadamard (87 ms, 21.6 %)** on the
non-power-of-two width I=14336. I wired the already-built, bit-identical **hpk28 fused
SwiGLU+Hadamard+pack producer** (perf_round2 Task 1) into the profiler and measured it:
`lin.down_fused` = 121.5 ms, clean_min **403.2 ms ≈ the 402.4 ms reference** — **NO
prefill win.** This confirms perf_round2's honest limitation: the fp32 had28 matmul
forces a 56 KB fp32 SMEM tile → **1 CTA/SM**, so the fused producer is occupancy-bound
(~0.20 TB/s) and does not beat the reference chain at M=8192. The 8B Hadamard is a
genuine wall, not a quick win.

**Verdict (prefill):** none of the candidate levers is a free win this round:
- **pack** — the qkv/gate_up merge that wins at decode *hurts* prefill (0.81–0.92×:
  compute-bound GEMM gains nothing from merging + the per-row-g multiply is overhead);
  the norm→pack fusion is accuracy-gated-off (RMSNorm not bit-exact, round-26).
- **GEMM config** — round-19/24 dead; our mixed GEMM runs two sequential phases
  (7/8 NVFP4 + 1/8 MXFP8) and is structurally below a single pure-NVFP4 GEMM
  (measured 271 TFLOPS @3B / 292 @8B, multisize).
- **attention / RMSNorm / RoPE / lm_head** — FP16, unaccelerated, ~45–50 % of the
  forward (Amdahl); shrinking them (FlashAttention / fused norms) helps *both* stacks
  and is not quantization work.
- **8B down Hadamard** — occupancy-bound; the fused producer gives no net win (measured).

So **prefill speedups are unchanged** this round (scoreboard 1B/3B/8B graphed
1.27 / 1.35 / 1.22× @M8192 stand). The honest finding is that the NVFP4 prefill gap is
Amdahl (FP16 attention+norm) plus the structural mixed-GEMM-vs-pure-NVFP4 difference,
neither addressable without a new attention/norm kernel or a single-phase GEMM.

## Gates (round-7)

- **Accuracy:** 1B two-level real-vs-twin |Δ|=0.014 PASS; PPL two-level < single-level
  (−0.198). Decode merge M=1 **bit-exact** (max|Δ|=0.0). M>1 merge PPL +0.022 (tiny).
- **Perf:** idle GPU, graphed, median. Decode merge gain 1.16–1.22× reproducible across
  3 sizes × 2 batches.

## Scoreboard deltas

- Accuracy (axis 1): 1B shipped PPL **14.76 → 14.563** (two-level, now default-ON; all
  three sizes two-level).
- Perf (axis 2) decode ×FP16-graph, **with the QKV+gate/up merge**:
  b1 1B/3B/8B **0.85 / 0.96 / 1.23×** (was 0.71/0.86/0.81),
  b8 **0.79 / 0.94 / 1.07×** (was 0.73/0.86/1.01). Prefill unchanged.

## Artifacts (remote `/home/luofan/`)

- `r7_1b_ppl_2level_{off,on}.json` (Task 1).
- `r7_decode_merge_{1b,3b,8b}.json` (Task 2: numerics gate, decode b1/b8, prefill A/B).
- `r7_prefill_stage_{3b,8b,8b_fused}.json` + `.csv` (Task 3 profiles).
- Code: `promix/eval/decode_merge.py` (new), `promix/eval/prefill_stage_profile.py`
  (always-wire real kernel; 8B hpk28 fused-producer path under `--fused`).
