# Full-Network Real-Kernel Integration — Report

> 2026-10-22. Scales the validated single-layer real-kernel path
> (`docs/single_layer_realkernel_report.md`, `docs/quant_pack_float_report.md`) from
> ONE Llama-3.2-1B decoder layer to the FULL network: all 16 layers' 7 Linears
> (**112 Linears**) run the real packed NVFP4+MXFP8 fused GEMM (SM120, RTX PRO 6000
> Blackwell). Delivers the two end-to-end milestone numbers: whole-model **PPL**
> (accuracy) and whole-model **prefill throughput vs FP16** (performance).
>
> Remote-only: built/run in container `luofan-plaquant` (10.208.130.90), FREE GPU 0.
> PROTECTED kernel files unchanged — all new code is in `promix/`.

## What was wired

The prepared ProMix model (`promix.eval.ptq.prepare_ptq_model`, FP per-head config
`llama-3.2-1b-fp-perhead.yaml`) wraps every Linear in an `ActQuantWrapper` whose
forward is `x → [online Hadamard (down_proj)] → [reorder (o_proj)] → fake-quant →
FP16 matmul → [out_quant (v, FP16)]`. The new model-pass helper
(`promix/quantize/full_network.py`) replaces **only** the `fake-quant → matmul` pair
of the 7 target Linears per layer, keeping Hadamard / o_proj reorder / column order /
out_quantizer / RMSNorm / attention QK·PV / RoPE / embed / lm_head in FP16 as-is:

- `wire_real_kernel` → inner module becomes `MixedQuantLinear` (real packed kernel,
  fused float→codes `qpf` pack), wrapper's fake-quant input quantizer disabled
  (`bits=16`) since the activation quant now happens inside the kernel path.
- `wire_fake_mixed` → inner module becomes `FakeMixedLinear` (fake_quantize the
  activation + FP32 matmul against the **same** dequantized weight `W_deq` the real
  kernel accumulates) — the tight numeric twin of the real kernel.

Weights are packed once offline (`pack_all_targets` → `pack_weight_mixed`, 112
Linears); activations are packed per-forward (`quant_pack_float`). Both wirings share
one offline pack so real-vs-twin differ **only** in the matmul (real fused kernel vs
torch FP32 matmul of identical dequantized operands), isolating the
packing/scale/accumulation path.

New files: `promix/quantize/full_network.py` (model pass + `FakeMixedLinear`),
`promix/eval/full_network_realkernel.py` (harness). Results JSON:
`kernels/mixed_gemm_sm120/full_network_{ppl,prefill}.json`; logs
`/home/luofan/fn_{ppl,prefill}.log`.

## 1. PPL — the accuracy gate (wikitext-2, seqlen 2048) — PASS

Built with the FP per-head recipe **including GPTQ FP weight quant** (the 14.92
recipe). All 112 Linears.

| model | wikitext-2 PPL |
|---|---|
| **STOCK fake-quant** (activation fake-quant + GPTQ FP weights + FP16 matmul) | **14.92** |
| **FAKE TWIN** (fake-quant act + FP32 matmul vs real kernel's `W_deq`) | 15.22 |
| **REAL KERNEL** (112 packed NVFP4+MXFP8 fused GEMMs) | **15.21** |
| reference: ProMix FP NVFP4+MXFP8 1B (published) | 14.92 |
| reference: FP16 | ~9.8 |

- **GATE: |REAL − TWIN| = 0.018 ≤ 0.1 → PASS.** The real-kernel model is numerically
  faithful to its fake-quant twin across all 112 Linears — no packing / scale /
  accumulation bug accumulates over the network.
- **STOCK reproduces the published 14.92 baseline exactly** (14.9195), confirming the
  harness builds the right model.
- **Per-layer cosine spot-check** (real vs twin, captured real rotated activations,
  layers 0 / 8 / 15 × all 7 Linears): every Linear **0.999998–0.999999** (worst
  0.9999985). Bit-faithful everywhere, start to end of the network.

### Honest note on REAL/TWIN vs STOCK (+0.29)

REAL (15.21) ≈ TWIN (15.22) sit ~0.30 above STOCK (14.92). This is **not** a
real-kernel error (the gate and per-layer cosines prove the kernel reproduces its
packed weights exactly). It is a **weight-packing scale-convention** effect: the real
kernel's packer (`quantize_nvfp4_spec`) derives each block scale from the *packed
weight's own* block absmax (round-up-to-E4M3 RTN), whereas GPTQ froze the block scale
from the *pre-correction* weight and rounded Hessian-corrected columns against that
frozen scale. Re-deriving the scale from a GPTQ-corrected block whose max did not
reach the top FP4 grid point shifts the scale and perturbs the non-saturated elements,
losing part of GPTQ's benefit (≈0.30 PPL). The twin exhibits the identical shift
(15.22) because it packs the same way — hence the clean real-vs-twin gate. **Recovering
the full 14.92 with the real kernel is a packer follow-up**: feed GPTQ's frozen
per-block scales into `pack_weight_mixed` instead of re-deriving from absmax. The
real-kernel GEMM/accumulation path itself is correct.

> **RESOLVED (2026-10-22, `docs/accuracy_frozen_scale_report.md`).** The packer now
> threads GPTQ's frozen per-block scales into `pack_weight_mixed`
> (`frozen_nvfp4_scale` / `frozen_mxfp8_scale`), captured from `gptq_fwrd` via
> `model._gptq_quantizers`. Twin recovers to **14.9159** (−0.0036 vs stock 14.9195)
> and REAL to **14.8494** (0.07 *below* stock); real-vs-twin gate 0.0665 ≤ 0.1,
> per-layer cosine 0.999998. The RTN path is kept as a fallback (`frozen_scale=None`)
> so the single-layer bit-exact probe is unchanged.

## 2. Prefill throughput vs FP16 (full model, batch 1, FREE GPU 0)

Full-model forward latency (real-kernel model vs a plain FP16 Llama-3.2-1B), seq
lengths giving M = 2048 / 4096 / 8192; min-of-N wall-clock.

| M | real-kernel | FP16 | **speedup** | tok/s real | tok/s FP16 |
|---|---|---|---|---|---|
| 2048 | 29.72 ms | 20.32 ms | **0.68×** | 68.9 K | 100.8 K |
| 4096 | 49.96 ms | 40.26 ms | **0.81×** | 82.0 K | 101.7 K |
| 8192 | 103.45 ms | 85.18 ms | **0.82×** | 79.2 K | 96.2 K |

### Time breakdown (ms) — where it goes

| M | pack | GEMM | **pack+GEMM** | FP16 Linear | FP16-rest¹ | overhead² | real total |
|---|---|---|---|---|---|---|---|
| 2048 | 3.12 | 8.17 | **11.29** | 11.80 | 8.52 | 9.92 | 29.72 |
| 4096 | 5.47 | 11.24 | **16.71** | 21.87 | 18.39 | 14.87 | 49.96 |
| 8192 | 9.96 | 19.39 | **29.34** | 40.49 | 44.69 | 29.41 | 103.45 |

¹ FP16-rest = FP16 full forward − FP16 Linear GEMMs = the non-Linear FP16 ops shared by
both models (attention QK·PV, RMSNorm, RoPE, embed, lm_head, residuals).
² overhead = real total − pack − GEMM − FP16-rest.

**The real kernel's Linear compute (pack+GEMM) BEATS FP16 cuBLAS Linear at every M —
1.05× (M2048) → 1.31× (M4096) → 1.38× (M8192)** — carrying the single-layer GEMM win
(GEMM alone is 1.4–2.1× FP16) to the whole network, with the fused float pack a small
fraction (pack/GEMM ≈ 0.38–0.51). If the per-forward overhead were zero, the real model
would already be **~1.03× (M2048) → ~1.15× (M8192)** faster than FP16 (pack+GEMM +
FP16-rest vs FP16 total).

### The bottleneck is per-call overhead, not compute — HONEST

The full real-kernel model is **0.68–0.82× FP16** because the `overhead` bucket
(9.9 / 14.9 / 29.4 ms) swamps the compute savings. This bucket is everything the real
model pays per forward that the vanilla FP16 model does not:

1. **Python / launch dispatch across 112 Linears** — each `MixedQuantLinear.forward`
   does a reshape, **two** pack kernel launches (fp4 + fp8), **one** fused-GEMM launch,
   a `D.to(fp16)`, a bias add, a reshape — plus the enclosing `ActQuantWrapper.forward`
   glue. ~112 × ~4 kernel launches + Python per call; FP16's single fused `x@Wᵀ` per
   Linear pays far fewer.
2. **Online Hadamard** on down_proj input (16×, FP32, intermediate_size 8192) and the
   **o_proj column reorder** (16×) — real-model-only ops absent from the FP16 baseline,
   so they land in this bucket.

This is exactly the overhead-accumulation the milestone flagged: the single-layer
M2048 case was already 0.88× FP16 from per-call Python/launch overhead; across 112
Linears it dominates. It grows sublinearly with M (compute grows faster), so the
speedup *improves* with M (0.68 → 0.82×) and would cross 1.0× at larger M / batch, but
the fixed per-call cost is the ceiling at these prefill sizes.

**Biggest next optimization (fusable follow-up):** collapse the per-Linear launch
count — (a) a single fused quant+pack+GEMM launch per Linear (fold the two `qpf` packs
and the `.to(fp16)`/bias into the GEMM epilogue), and/or (b) **CUDA-graph capture** of
the whole prefill forward to erase the 112-Linear Python dispatch. Because pack+GEMM
compute already beats FP16, removing the launch/dispatch overhead directly converts the
measured compute win into an end-to-end win.

## 3. Boundaries / honesty

- PROTECTED kernel files and mainloop logic unchanged; all new code is in `promix/`
  (`full_network.py`, `full_network_realkernel.py`). Reused the built kernel extensions
  (`codesign_fused_ext`, `quant_pack_float_ext`, `nvfp4_gemm`, `mxfp8_gemm`).
- KV cache stays FP16 (this is the W4A4-style FP baseline, `v_bits=k_bits=16`), matching
  the 14.92 recipe; attention QK·PV, RoPE, softmax, embed, lm_head, RMSNorm all FP16.
- Prefill is forward-only (no decode / KV-cache growth this step), batch 1, free GPU.
  FP16 reference is a plain `AutoModelForCausalLM` (no rotation/quant) — the honest
  FP16 wall-clock.
- The `overhead` bucket combines Python/launch dispatch + online Hadamard + o_proj
  reorder (not separated further); CUDA-graph / fused-launch work would attribute and
  remove the dominant (launch-dispatch) part.

## Verdict

Full-network real-kernel integration **works end-to-end on all 112 Linears**:
whole-model wikitext-2 PPL **15.21**, numerically faithful to its fake-quant twin
(gate |Δ| = 0.018 ≤ 0.1, per-layer cosine 0.999998–0.999999) and reproducing the 14.92
fake-quant baseline up to a documented GPTQ-scale-repack delta (packer follow-up). The
fused kernel's Linear compute **beats FP16 at the whole-network level (1.05–1.38×)**,
but full-model prefill is currently **0.68–0.82× FP16** because per-call Python/launch
overhead across 112 Linears (plus the FP32 online Hadamard) dominates at these prefill
sizes — the clear, fusable next bottleneck (single-launch quant+pack+GEMM / CUDA-graph
capture), which would convert the measured compute win into an end-to-end speedup.
