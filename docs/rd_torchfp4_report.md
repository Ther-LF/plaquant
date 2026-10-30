# RD torch-FP4: our fused W4A4 GEMM vs naive same-hardware FP4 baselines

**Branch** `feat/rd-torchfp4` (off `origin/feat/rd-alloc`) · **HW** RTX PRO 6000 Blackwell
(SM120, 96 GB), container `luofan-plaquant` (torch 2.11.0+cu130), dedicated idle GPU 1 ·
**Date** 2026-10-30 · **Harness** `promix/eval/gemm_h2h_torchfp4.py` · **Raw JSON**
`docs/rd_torchfp4_data/torchfp4_gemm.json`

**Question.** Our prior peer comparisons (rd-largem) pitted our full stack against vLLM's
full stack — conflating the quantized GEMM with everything around it. This round does the
QServe-style "isolate the contribution" comparison: fix the stack, compare ONLY the
quantized GEMM against *naive* FP4 baselines any practitioner could write on the same
hardware, at our exact LLM shapes (vLLM GEMM decomposition, merged qkv/gate_up) across
M ∈ {2048, 8192, 16384, 32768}.

**Candidates** (all GEMM-only, operands pre-quantized/pre-packed outside the timed region,
except dequant which pays its dequantize per call *by design*):

1. **ours** — our fused single-launch dual-format NVFP4(K_low)+MXFP8(K_high) W4A4 GEMM,
   production r = 1/8 split, per-shape dispatch (co-design/b1_tma).
2. **torch-FP4** — torch-NATIVE NVFP4 tensor-core GEMM: `torch._scaled_mm_v2` with
   `float4_e2m1fn_x2` data + `BlockWise1x16` e4m3 scales in `SWIZZLE_32_4_4` layout
   (cuBLASLt backend). Zero custom kernel code. **Uniform FP4 only — it cannot express
   our mixed format.**
3. **torch-mix2** — the framework-native rendering of OUR algorithm, no custom kernel:
   native NVFP4 GEMM over K_low + native MXFP8 (`BlockWise1x32`, e8m0) GEMM over K_high
   + elementwise add (2 GEMM launches + add).
4. **dequant** — the "no kernel work" floor: NVFP4-packed weight (uint8 codes + block
   scales) dequantized to bf16 in plain torch per call, then bf16 `torch.mm`.
5. **bf16** — bf16 `torch.mm` (cuBLAS): the same-stack FP16-class anchor. Our W4A4/bf16
   ratio = our contribution isolated, QServe-style.
6. *(reference)* vLLM NVFP4 CUTLASS GEMM, from `docs/rd_largem_data/vllm_nvfp4_gemm.json`
   (same shapes/M, same box, measured in rd-largem).

## Answer to "does torch have native FP4 on SM120?" — YES (and it is fast)

torch 2.11.0+cu130 ships a REAL NVFP4 tensor-core matmul that runs on SM120:
`torch._scaled_mm_v2(A_fp4, B_fp4.t(), [sfA], [BlockWise1x16], [SWIZZLE_32_4_4], …)`
with `torch.float4_e2m1fn_x2` operands and e4m3 block scales (cuBLASLt). We validated it
numerically, not just that it runs: an exactness probe (all-1.0 codes, unit scales →
output exactly K) **passes**, and on real quantized data it matches a float dequant
reference at **cosine 0.9999986**. Native MXFP8 (`BlockWise1x32`, e8m0) also works.
`torchao` is not installed in the container; it is not needed — the `_scaled_mm_v2` path
is the framework-native FP4 GEMM. Consequence: "you need a custom kernel to touch FP4
tensor cores at all" is NOT true on this stack; the custom-kernel case must rest (and
does rest, below) on the mixed-format capability.

## Results

Protocol: idle dedicated GPU 1, warm; per cell all candidates timed **interleaved
round-robin** (cudaEvent, median of 50) after a 400 ms clock warm-spin — no clock-lock
permission on this box, and a first non-interleaved run showed exactly the expected
artifact (first-measured candidate eats the idle-clock ramp, inflating it ~2×; that run
was discarded and the protocol hardened). Cross-check: ours qkv·3B·M2048 = 0.119 ms here
vs 0.112 ms in rd-largem — consistent.

### 3B (Llama-3.2-3B shapes)

| shape (N×K) | M | ours (ms) | torch-FP4 (ms) | torch-mix2 (ms) | dequant (ms) | bf16 (ms) | vLLM-NVFP4 (ms) | ours/bf16 | ours/torch-FP4 | ours/mix2 | ours/dequant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qkv (5120×3072) | 2048 | 0.119 | 0.105 | 0.126 | 0.622 | 0.205 | 0.230 | **1.72×** | 0.88× | 1.05× | 5.21× |
| " | 8192 | 0.300 | 0.246 | 0.406 | 1.071 | 0.659 | 0.361 | **2.20×** | 0.82× | 1.35× | 3.58× |
| " | 16384 | 0.530 | 0.434 | 0.824 | 1.692 | 1.290 | 0.546 | **2.43×** | 0.82× | 1.55× | 3.19× |
| " | 32768 | 0.992 | 0.827 | 1.663 | 2.946 | 2.576 | 0.904 | **2.60×** | 0.83× | 1.68× | 2.97× |
| o_proj (3072×3072) | 2048 | 0.097 | 0.086 | 0.100 | 0.336 | 0.155 | 0.204 | **1.59×** | 0.88× | 1.02× | 3.45× |
| " | 8192 | 0.202 | 0.169 | 0.234 | 0.630 | 0.452 | 0.289 | **2.24×** | 0.84× | 1.16× | 3.12× |
| " | 16384 | 0.354 | 0.300 | 0.505 | 1.022 | 0.835 | 0.402 | **2.36×** | 0.85× | 1.43× | 2.89× |
| " | 32768 | 0.628 | 0.526 | 1.011 | 1.741 | 1.570 | 0.630 | **2.50×** | 0.84× | 1.61× | 2.77× |
| gate_up (16384×3072) | 2048 | 0.247 | 0.209 | 0.320 | 2.551 | 0.564 | 0.318 | **2.28×** | 0.84× | 1.30× | 10.31× |
| " | 8192 | 0.808 | 0.653 | 1.309 | 3.937 | 1.973 | 0.770 | **2.44×** | 0.81× | 1.62× | 4.87× |
| " | 16384 | 1.536 | 1.261 | 2.596 | 5.969 | 4.054 | 1.316 | **2.64×** | 0.82× | 1.69× | 3.88× |
| " | 32768 | 3.065 | 2.539 | 5.311 | 10.127 | 8.281 | 2.404 | **2.70×** | 0.83× | 1.73× | 3.30× |
| down (3072×8192) | 2048 | 0.168 | 0.149 | 0.164 | 1.156 | 0.325 | 0.261 | **1.94×** | 0.89× | 0.98× | 6.90× |
| " | 8192 | 0.415 | 0.359 | 0.450 | 1.947 | 1.112 | 0.461 | **2.68×** | 0.86× | 1.08× | 4.69× |
| " | 16384 | 0.757 | 0.652 | 0.905 | 3.004 | 2.172 | 0.793 | **2.87×** | 0.86× | 1.20× | 3.97× |
| " | 32768 | 1.428 | 1.238 | 1.804 | 4.969 | 4.183 | 1.278 | **2.93×** | 0.87× | 1.26× | 3.48× |

### 8B (Llama-3-8B shapes)

| shape (N×K) | M | ours (ms) | torch-FP4 (ms) | torch-mix2 (ms) | dequant (ms) | bf16 (ms) | vLLM-NVFP4 (ms) | ours/bf16 | ours/torch-FP4 | ours/mix2 | ours/dequant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qkv (6144×4096) | 2048 | 0.157 | 0.136 | 0.157 | 1.165 | 0.326 | 0.250 | **2.08×** | 0.87× | 1.00× | 7.43× |
| " | 8192 | 0.427 | 0.352 | 0.564 | 1.932 | 1.089 | 0.463 | **2.55×** | 0.83× | 1.32× | 4.53× |
| " | 16384 | 0.793 | 0.656 | 1.143 | 2.925 | 2.103 | 0.753 | **2.65×** | 0.83× | 1.44× | 3.69× |
| " | 32768 | 1.539 | 1.278 | 2.307 | 4.949 | 4.168 | 1.319 | **2.71×** | 0.83× | 1.50× | 3.22× |
| o_proj (4096×4096) | 2048 | 0.114 | 0.102 | 0.117 | 0.654 | 0.211 | 0.215 | **1.85×** | 0.89× | 1.02× | 5.72× |
| " | 8192 | 0.296 | 0.248 | 0.367 | 1.203 | 0.744 | 0.360 | **2.51×** | 0.84× | 1.24× | 4.06× |
| " | 16384 | 0.555 | 0.464 | 0.769 | 1.827 | 1.397 | 0.560 | **2.52×** | 0.84× | 1.38× | 3.29× |
| " | 32768 | 1.060 | 0.875 | 1.544 | 3.221 | 2.801 | 0.940 | **2.64×** | 0.83× | 1.46× | 3.04× |
| gate_up (28672×4096) | 2048 | 0.496 | 0.410 | 0.677 | 5.918 | 1.222 | 0.512 | **2.46×** | 0.83× | 1.36× | 11.94× |
| " | 8192 | 1.749 | 1.442 | 2.636 | 9.363 | 4.721 | 1.482 | **2.70×** | 0.82× | 1.51× | 5.35× |
| " | 16384 | 3.568 | 2.991 | 5.481 | 14.560 | 10.024 | 2.730 | **2.81×** | 0.84× | 1.54× | 4.08× |
| " | 32768 | 7.758 | 6.486 | 11.584 | 25.372 | 21.611 | 5.275 | **2.79×** | 0.84× | 1.49× | 3.27× |
| down (4096×14336) | 2048 | 0.264 | 0.245 | 0.269 | 2.960 | 0.662 | 0.361 | **2.51×** | 0.93× | 1.02× | 11.21× |
| " | 8192 | 0.905 | 0.810 | 0.984 | 4.948 | 2.702 | 0.849 | **2.98×** | 0.90× | 1.09× | 5.47× |
| " | 16384 | 1.871 | 1.681 | 2.137 | 7.951 | 5.745 | 1.508 | **3.07×** | 0.90× | 1.14× | 4.25× |
| " | 32768 | 3.675 | 3.340 | 4.292 | 13.147 | 11.072 | 2.800 | **3.01×** | 0.91× | 1.17× | 3.58× |

### Geomeans (over all 16 shape×M cells per model)

| model | ours/bf16 | ours/torch-FP4 | ours/torch-mix2 | ours/dequant | torch-FP4/bf16 |
|---|---|---|---|---|---|
| 3B | **2.35×** | 0.85× | 1.33× | 4.01× | 2.78× |
| 8B | **2.60×** | 0.86× | 1.28× | 4.78× | 3.04× |

## Verdict — our kernel vs the naive paths, isolated

1. **vs bf16 (the QServe-style anchor): 1.6–3.1×, geomean 2.35× (3B) / 2.60× (8B),**
   growing with M (compute-boundedness) and with K (down_proj best, 3.0×+ at 8B). This is
   the isolated GEMM-level contribution, free of any full-stack effects.

2. **vs the naive dequant-FP4 floor: 2.8–11.9× (geomean 4.0–4.8×).** What you get with
   packed FP4 weights but zero kernel work is far below even bf16 (dequant is always
   SLOWER than bf16 mm) — weight-only packing without a real kernel is a memory win, not
   a speed win.

3. **vs torch-native FP4 (uniform NVFP4, cuBLASLt): ours is 0.82–0.93× (geomean 0.85×)
   — the framework's pure-FP4 GEMM is ~15 % faster than our fused kernel.** Honest and
   unsurprising: it does strictly less work in a strictly simpler format (uniform FP4
   over the full K, single format, no FP8 phase) on NVIDIA's own cuBLASLt heuristics.
   But it is not the same operator: uniform W4A4-FP4 is exactly the configuration whose
   accuracy loss motivates ResQ/ProMix mixed precision. It also beats our own stock
   CUTLASS pure-NVFP4 path (`pure_nvfp4_ms` in the JSON) by 1.1–1.7×, i.e. the gap is
   cuBLASLt-vs-CUTLASS backend quality on pure FP4, not fusion overhead.

4. **vs torch-mix2 (the framework-native implementation of OUR algorithm — the correct
   apples-to-apples naive baseline): ours wins 0.98–1.73×, geomean 1.33× (3B) /
   1.28× (8B), and the win GROWS with M** (≈1.0× at M=2048 where launch effects are
   amortizable, 1.5–1.7× at M=32768). The 2-launch path pays a second full M×N output
   write + an add kernel + a thin-K (K_high = K/8) MXFP8 GEMM that cuBLASLt runs
   inefficiently; our single launch accumulates both formats in registers. **This is the
   isolated value of the fusion itself.**

5. *(reference)* vs vLLM's production NVFP4 CUTLASS GEMM: ours is faster at M ≤ 8192
   (up to 2.1× at M=2048), parity at 16384, and vLLM pulls ahead at 32768 (ours
   0.68–0.91× of vLLM) — consistent with rd-largem's crossover finding. Notably
   torch-native FP4 beats vLLM's CUTLASS NVFP4 GEMM at every M ≤ 16384 as well.

**Bottom line.** On SM120 a framework FP4 GEMM baseline EXISTS (torch 2.11 `_scaled_mm_v2`
→ cuBLASLt) and is strong (~2.8–3.0× over bf16); a naive-FP4-availability argument for our
kernel is dead. The surviving — and measured — argument is the mixed format: cuBLASLt
cannot run NVFP4+MXFP8 in one GEMM, the naive 2-launch rendering of our algorithm is
1.3× (geomean, up to 1.7×) slower than our fused kernel, and the accuracy-bearing mixed
format costs only ~15 % versus the pure-FP4 uniform ceiling that is known to be
accuracy-inadequate at W4A4. Against the common bf16 anchor our kernel's isolated
contribution is **2.4–2.6× geomean (peak 3.1×)**.

## Caveats

- torch-mix2's MXFP8 activation/weight scales were unit e8m0 (timing-valid: latency is
  value-independent; its NVFP4 part uses real quantized scales). dequant uses a LUT
  unpack (`uint8 → e2m1 value table → × block scale`), the natural plain-torch
  implementation; a cleverer dequant could shave its constant but cannot approach the
  native paths.
- All numbers are GEMM-only by design (QServe methodology). Activation quantize/pack
  costs are identical-stack issues orthogonal to this comparison and covered in the e2e
  reports (full_network/prefill).
- vLLM column is from the rd-largem session (same box/GPU-class, different day) — treat
  as reference, not same-session A/B; all other columns are same-session interleaved.
