# Round 1 (ALGORITHM axis) — accuracy-per-bit ablations

Protocol: ProMix fake-quant STOCK path (`prepare_ptq_model`: fuse-norm → PCA
rotation (U+R1) → rearrange → actquant → online-Hadamard(down) → **GPTQ with
frozen FP block scales + Hessian error feedback** → activation quantizers → KV)
evaluated by the pipeline's own `evaluate_ppl` on wikitext-2. This is the exact
protocol behind `docs/multisize_report.md` (the fake-quant column) — **anchors
reproduce it to 4 dp**: 1B r=1/8 = 14.9195 (report 14.9195), 3B r=1/8 = 8.6470
(report 8.647). All numbers below are mutually comparable (same protocol).
Harness: `promix/scripts/ablation_acc_per_bit.py` (research-only accuracy path,
no kernel build). Raw JSON: `docs/round1_results/`.

bits/weight = WEIGHT storage only = (1−r_w)·4.5 + r_w·8.25 (NVFP4 block-16 +
E4M3 scale = 4.5 b/elem; MXFP8 block-32 + E8M0 = 8.25 b/elem). **Activation
precision r_a is FREE on this axis** (activations are not stored; r_a trades GEMM
compute).

Reference points on the SAME hardware/protocol family (docs): FP16 3B=7.818 /
8B=6.138 / 1B=13.16; vLLM **NVFP4 W4A4** 4.25b 3B=8.63 / 8B=6.87; GPTQ W4A16 4.0b
3B=8.20 / 8B=6.52; AWQ-asym W4A16 4.0b 8B=6.59. (1B is Instruct; per the
scoreboard the 1B absolute cannot be cross-compared to vLLM — 1B here is used
only for *relative* ablation verdicts, which are protocol-consistent.)

8B: model re-download (≈16 GB over a ~0.3 MB/s link) did not finish this round;
8B variant runs are pending. 8B r=1/8 fake-quant is already 6.821 (docs).

---

## 1. Fixed-r bit-vs-PPL curve (A4)

| r | bits/w | **3B PPL** | 1B PPL |
|---|-------|-----------|--------|
| 0 (pure NVFP4) | 4.500 | **9.644** | 16.220 |
| 1/16 | 4.734 | **8.682** | 15.475 |
| 1/8 (ship) | 4.969 | **8.647** | 14.919 |
| 1/4 | 5.438 | — | 14.426 |
| 1/2 | 6.375 | — | 14.027 |

Key fact (3B): going r=0→1/8 improves PPL by **−1.0** (9.644→8.647). The MXFP8
high segment is **not** "buying nothing" — it is load-bearing. The scoreboard's
"0.72 bit buys nothing" was a comparison *to vLLM NVFP4* (8.63), not to our own
r=0. Our **pure-NVFP4 base is weak** (3B r=0 = 9.644 ≫ vLLM NVFP4 8.63), and the
high segment is what drags us back up to ≈NVFP4.

## 2. Attribution — "why worse than GPTQ?" (answers coordinator Q1)

GPTQ coverage verified in code: `gptq_fwrd` quantizes all 7 Linears; the MXFP8
high segment **does** receive full Hessian error feedback (the per-column loop
routes `col ≥ high_dim` through `high_quantizer.quantize_column_with_frozen_scale`
and the error-feedback `W1[:, i:] -= …` spans all columns). Only the block
*scale* is frozen RTN (no MSE clip — see §3).

Decomposition at r=1/8 (3B), same protocol:

| config | PPL | Δ |
|---|---|---|
| FP16 | 7.818 | — |
| **weights-only** (4.97b NVFP4+MXFP8 wts, **acts FP16**) | **8.084** | weight quant +0.27 |
| full W4A4 (acts also quantized) | 8.647 | **activation quant +0.56** |
| GPTQ W4A16 (ref) | 8.20 | |

**Verdict: we are NOT worse than GPTQ on weights — we are better.** Our 4.97-bit
weights (8.084) beat GPTQ-W4A16 (8.20) by 0.12 (1B: 13.78 vs 14.22, by 0.44).
The entire W4A4 deficit vs GPTQ is **activation quantization** (W4A4 vs W4A16).
This is the attribution to use in the writeup.

Follow-up (3B): the +0.56 activation error is **broad, not tail-concentrated** —
raising the activation high-fraction r_a from 1/8→1/4 recovers only −0.04
(8.647→8.610). So more MXFP8 *activation* channels barely helps; the activation
error is NVFP4 rounding across all channels → the lever is a *better activation
NVFP4 quantizer*, not more high channels.

## 3. Levers tested at equal bits (3B unless noted)

| lever | result | verdict |
|---|---|---|
| **A2 block-scale MSE search** (deployable; scale stays on E4M3/E8M0 grid) | r=1/8 **8.647→8.545 (−0.10)**; r=1/16 8.682→8.666 (−0.02). 1B: r=1/16 −0.11, r=1/4 −0.08, r=1/8 +0.02 | **small free win**, best at r=1/8 on 3B. Fills the currently-empty `w_clip/mse` hook for FP. |
| **A1 weight-energy / combined channel selection** (contiguous reorder, not scatter) | 1B combined 15.04 vs 14.92 | **worse** — activation-variance tail (default) is right |
| **edriven variable-r per layer** (weight-error signal) | 3B 8.667, 1B 15.13 vs uniform 14.92 | **worse** |
| **edriven variable-r** (activation-variance signal) | 1B 14.995 vs 14.92 | **worse** (least-bad) |
| **rotation R1 off** (A3) | 3B r=1/8 8.677 vs 8.647 (+0.03); **3B r=0 13.56 vs 9.64 (+3.9!)**; 1B r=1/8 +0.34 | R1 is **critical at low r** (spreads concentrated variance), nearly irrelevant once the high segment exists |
| **two-level NVFP4 scale** (per-row FP32 global × per-16 E4M3 micro, vLLM-style) | **1B helps** (r=0 −0.28, r=1/16 −0.30, r=1/8 −0.11); **3B hurts** (r=0 +0.32, r=1/8 +0.10) | model-dependent; my per-row-global variant is not the universal fix |

**Unifying insight:** the error is *activation-dominated on 1B* but
*weight-high-dominated on 3B* (removing the weight high segment costs +0.90 on
3B; removing activation high costs +0.10). In all cases the **hardware-native
uniform [NVFP4 main | MXFP8 high] partition with activation-variance selection is
already near-optimal** — per-layer reallocation and weight-error selection do not
help. Chasing "smart allocation" is a dead end; the gains are in the *base
quantizer quality* (scale scheme, activation quantizer).

## 4. Weight/activation precision decoupling (Pareto probe)

At fixed weight bits r_w=1/8 (4.969b), vary activation r_a (free on bits/weight):
1B 16.67 (r_a=0) → 15.70 (1/16) → 14.92 (1/8) → 14.76 (1/4); 3B 8.647 (1/8) →
8.610 (1/4). Activation precision is a real lever on 1B, saturating fast on 3B.
Lowering weight bits r_w hurts a lot on 3B (rw=0,ra=1/8 = 9.544; the
concentrated-variance tail needs 8-bit *weights*). So the clean "cheap weights +
rich acts" Pareto play does **not** hold on 3B — the weight tail is the binding
constraint. (Deployability: r_a≠r_w needs a kernel that splits A and B
differently — not the current fused kernel; labelled research-only.)

## 5. Hardware-native partition (the selling point)

Every variant that helps keeps the structure **contiguous, block-aligned (NVFP4
16 / MXFP8 32), few segments** — exactly what the fused kernel consumes in one
launch with a shared accumulator. A1/edriven are implemented as *channel reorder
→ contiguous high tail*, never per-channel scatter. The packer already asserts
block alignment and the kernel takes K_low/K_high per-Linear, so variable
segment lengths are natively supported (per-layer cos 0.999999 in
`docs/multisize_report.md`). One line for the paper: *"the partition is
hardware-native — contiguous, block-aligned, 2 segments → one fused-kernel launch
with a shared accumulator; accuracy comes from choosing the boundary, not from
scattered per-channel precision."*

## 6. Pareto verdict vs NVFP4 / GPTQ / AWQ (3B, accuracy-per-weight-bit)

- Best ProMix point: 4.969b, **8.545** (r=1/8 + scale-opt). We **beat NVFP4's
  PPL** (8.63) but at **more bits** (4.97 vs 4.25) → neither dominates.
- GPTQ (4.0b, 8.20) and AWQ (4.0b, ≈8.35) **dominate us on this axis** (fewer
  bits, better PPL). **But** GPTQ/AWQ are W4A16 — activations are *not* quantized,
  so they get no prefill-compute speedup; we are W4A4. The accuracy-per-weight-bit
  axis structurally favors W4A16 because it does not pay for activation quant.
- **Honest conclusion:** ProMix does **not** currently have a strictly-better
  accuracy-per-weight-bit Pareto front than the W4A16 baselines. Its real
  advantage is W4A4 (activation quantized → prefill tensor-core speedup) at
  accuracy ≈ NVFP4, with a hardware-native partition. The story to push is
  *"≈NVFP4 accuracy + a cleaner, kernel-fused partition"* and the compute axis,
  not raw accuracy-per-bit.

## 7. Q2 — mixed-precision activations: prefill vs decode

- **Prefill (compute-bound):** plausible advantage over GPTQ/AWQ-W4A16, which run
  a dequant-to-FP16 GEMM (no math speedup; `docs/multisize_report` shows Marlin
  prefill ≈ neutral/negative). Our W4A4 uses FP4/FP8 tensor cores → real math
  throughput gain (NVFP4 W4A4 is the prefill winner on SM120). So "mixed-precision
  activations help in prefill" is **true in principle** — but today our pack
  overhead + unaccelerated attention cap it at ~1.3–1.4× vs FP16, still below
  vLLM-NVFP4's 1.85–2.2×.
- **Decode (bandwidth-bound):** we are weaker, as expected. Decode moves no
  activation FLOPs worth mentioning; it is gated by weight bytes/s, and Marlin
  W4A16 has a mature, HBM-saturating GEMV while our decode-GEMV under-saturates
  HBM and KV/attention stay FP16 (uncompressed). Quantizing activations buys
  nothing in decode; the mixed-precision activation path is a prefill lever only.

## 8. Recommendation (ship + next lever)

- **Ship now:** r=1/8 + **A2 block-scale MSE search** (deployable, free −0.10 on
  3B → 8.545 < NVFP4 8.63). Fill the FP branch of `WeightQuantizer.find_params`
  `mse` hook and add an optional per-block activation-scale search.
- **Biggest remaining accuracy-efficiency lever:** the **activation NVFP4
  quantizer** (the +0.56 on 3B is all activation, and it is broad → not fixable by
  more high channels). Try: per-token two-level activation scale, or a calibrated
  global×micro activation scale matching vLLM NVFP4's observer. This is *free* on
  the bits/weight axis and directly closes the gap to the W4A16 methods.
- **Do NOT** invest further in per-layer variable-r allocation or weight-error
  channel selection — both are ≤ uniform here.
- **Deployability of A2 for the current kernel:** scale search only changes the
  *values* of the frozen NVFP4/MXFP8 block scales (still on the E4M3/E8M0 grid),
  so the packer/kernel are unchanged — it needs **no Step-0/1 rerun**, only a
  re-pack. Variable-r and decoupled r_w/r_a would need kernel changes and a
  Step-0/1 reorder, and did not pay off.
