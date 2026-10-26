# ProMix Multi-Size Scaling Report (1B / 3B / 8B)

Real-kernel ProMix (NVFP4 low 7/8 + MXFP8 high 1/8, per-head, frozen GPTQ
scales) on SM120 (RTX PRO 6000). The **same** shipping path — `qpfm`
single-launch pack + fused SwiGLU/Hadamard/pack down-proj producer (where the
intermediate width allows) + epilogue bias/cast fusion + decode-GEMV — run
across three Llama sizes. Shapes are derived from each model's config, not
hard-coded.

Harness: `promix/eval/multisize.py` (prefill + decode + weight memory) and
`promix/eval/full_network_realkernel.py --phases ppl` (PPL gate). Quiet GPUs
(one size per idle GPU), CUDA graphs, median timing.

## Headline table

| Size | PPL (fake-quant → real) | Gate | Prefill ×FP16 (graph) M=2048/4096/8192 | Decode ×FP16 (graph) b1 / b8 | Weight FP16→packed | Linear share @M2048 |
|------|-------------------------|------|----------------------------------------|------------------------------|--------------------|---------------------|
| **1B** | 14.919 → **14.849** (−0.070) | PASS | 1.215 / 1.302 / 1.271 | 0.712 / 0.731 | 1946 → 604 MB (3.22×) | 65.7% |
| **3B** | 8.647 → **8.615** (−0.032) | PASS | 1.385 / 1.437 / 1.354 | 0.859 / 0.860 | 5637 → 1751 MB (3.22×) | 71.7% |
| **8B** | 6.821 → **6.823** (+0.002) | PASS | 1.293 / 1.241 / 1.224 | 0.805 / 1.006 | 13959 → 4335 MB (3.22×) | 77.5% |

All real-kernel PPLs are within +0.1 of the size's fake-quant baseline (1B/3B
are actually *better*; 8B +0.002). Per-layer cosine (real vs fake twin) worst =
0.999999 for every size — the packed fused kernel is numerically identical to
the fake-quant recipe. Prefill graph-vs-eager cosine = 1.000000 at every M.

## PPL gate (wikitext-2, frozen GPTQ scales)

| Size | fake-quant baseline (STOCK) | fake twin | real kernel | \|real−twin\| gate | real − baseline |
|------|-----------------------------|-----------|-------------|--------------------|-----------------|
| 1B | 14.9195 | 14.9159 | 14.8494 | 0.0664 ✓ | −0.070 |
| 3B | 8.6470  | 8.6522  | 8.6151  | 0.0370 ✓ | −0.032 |
| 8B | 6.8214  | 6.8212  | 6.8233  | 0.0020 ✓ | +0.002 |

(1B row is the previously-archived `full_network_ppl_frozen.json`; 3B/8B were
produced this run with the identical harness. 8B PPL was evaluated at
`batch_size=1` — `bs=16` OOMs because the 8B logits tensor is ~17 GB/batch;
PPL is batch-size-independent so the number is unchanged.)

## Prefill (full-model, CUDA-graph real vs FP16, median)

Primary number is **vs FP16 graphed** (both models CUDA-graphed — the fair,
same-conditions comparison). `vs eager` is the FP16-eager baseline (what the
original 1B 1.29/1.35/1.30 figure used); it flatters the real kernel at small M
because eager FP16 pays launch overhead the graph removes.

| Size | M | real graph (ms) | ×FP16 graph | ×FP16 eager | real GEMM TFLOPS | Linear share of FP16 time | cos |
|------|------|-----------------|-------------|-------------|------------------|---------------------------|-----|
| 1B | 2048 | 15.95 | 1.215 | 1.292 | 250 | 65.7% | 1.000000 |
| 1B | 4096 | 30.16 | 1.302 | 1.349 | 264 | 59.1% | 1.000001 |
| 1B | 8192 | 66.23 | 1.271 | 1.286 | 241 | 50.2% | 1.000001 |
| 3B | 2048 | 36.64 | 1.385 | 1.426 | 315 | 71.7% | 1.000000 |
| 3B | 4096 | 72.30 | 1.437 | 1.460 | 319 | 65.1% | 1.000001 |
| 3B | 8192 | 170.16 | 1.354 | 1.360 | 271 | 53.7% | 1.000001 |
| 8B | 2048 | 78.90 | 1.293 | 1.303 | 362 | 77.5% | 1.000000 |
| 8B | 4096 | 170.81 | 1.241 | 1.243 | 335 | 70.9% | 1.000001 |
| 8B | 8192 | 391.94 | 1.224 | 1.219 | 292 | 62.4% | 1.000001 |

8B M=8192 ran without OOM (~392 ms, no reduction needed).

## Decode (static-KV CUDA graph, prompt 32 / gen 128, per-token median)

Primary number is **vs FP16 graphed** (fair). `vs eager` FP16 is also recorded
but is not a meaningful target (FP16 eager decode is dominated by Python/launch
overhead; the real path beats it 1.2–2.1× trivially).

| Size | batch | real graph ms/tok | tok/s (real) | ×FP16 graph | ×FP16 eager |
|------|-------|-------------------|--------------|-------------|-------------|
| 1B | 1 | 4.564 | 219.1 | 0.712 | 2.146 |
| 1B | 8 | 5.164 | 1549.2 | 0.731 | 1.946 |
| 3B | 1 | 8.526 | 117.3 | 0.859 | 1.939 |
| 3B | 8 | 9.376 | 853.2 | 0.860 | 1.778 |
| 8B | 1 | 16.691 | 59.9 | 0.805 | 1.170 |
| 8B | 8 | 14.874 | 537.8 | **1.006** | 1.346 |

Weight-bandwidth decode floor (packed bytes / measured HBM BW): 1B 0.419 ms/tok
(FP16 1.349), 3B 1.211 (3.898), 8B 3.010 (9.691) — a ~3.2× ceiling the current
decode-GEMV realizes only partially.

## Weight compression

Target-Linear weights only; identical recipe → identical ratio at every size.

| Size | FP16 MB | packed MB | compression | bits/weight |
|------|---------|-----------|-------------|-------------|
| 1B | 1946.2 | 604.4 | 3.220× | 4.969 |
| 3B | 5637.1 | 1750.6 | 3.220× | 4.969 |
| 8B | 13958.6 | 4334.8 | 3.220× | 4.969 |

4.969 bits/weight = 7/8·(4-bit NVFP4) + 1/8·(8-bit MXFP8) + the FP8 block scale
factors (NVFP4 group 16, MXFP8 group 32).

## Scaling analysis

**Prefill speedup peaks at 3B (1.35–1.44×), with 1B and 8B lower (~1.22–1.30×).**
Two forces pull against each other:

1. **Linear share rises monotonically with size** (66% → 72% → 78% at M=2048).
   A larger share of model time is spent in the accelerated Linears, which
   raises the Amdahl ceiling and favors bigger models.
2. **Realized per-Linear speedup `k` = (FP16 cuBLAS Linear) / (pack + mixed
   GEMM) peaks in the middle.** At 1B the shapes are small (K=2048): both cuBLAS
   and the mixed kernel underfill the SMs (mixed GEMM only ~250 TFLOPS), so `k`
   is modest. At 8B the pack cost grows O(M·K) with the larger K (4096, and
   down_proj K=14336), while cuBLAS FP16 becomes very efficient on the big
   tiles, shrinking the mixed kernel's edge; the mixed GEMM still peaks at
   362 TFLOPS but pack overhead eats the gain.

The product of (1) and (2) is maximized at 3B. Within each size the speedup also
rises from M=2048→4096 then dips at 8192: larger M amortizes pack/dispatch, but
at M=8192 the O(M²) attention/softmax share grows and the Linear share drops
(8B: 77.5% → 62.4%), diluting the win.

**Decode improves with both size and batch, crossing break-even only at 8B/b8
(1.006×).** Decode is memory-bound, so the 3.22× weight compression is the
lever — but it only pays when weight-BW actually dominates:

- At **batch 1** every size is below parity (0.71–0.86×). The FP16 CUDA-graph
  baseline is already tiny (3.2 ms/tok at 1B), and at M=1 the decode-GEMV kernel
  is latency-bound — it does not saturate HBM, so the 3.2× byte reduction does
  not convert to a 3.2× time reduction, while the per-token dequant/index work
  adds cost.
- **Larger models** shift the balance toward weight-BW (8B weight floor 3.0 ms
  vs FP16 9.7 ms), so the compression starts to pay: 1B 0.71× → 3B 0.86× → 8B
  0.81× at b1, and 8B reaches **1.006×** at b8.
- **Batch 8** amortizes per-step kernel-launch / indexing overhead across 8
  sequences, lifting every size over its b1 number.

Decode never approaches the 3.2× BW ceiling because (a) the KV cache and
attention run in FP16 and are *not* compressed, (b) the decode-GEMV kernel is
memory-latency bound at tiny M and under-saturates HBM, and (c) graph-replay and
sampling overhead. The honest decode story today is "at parity, best case."

## Honest gaps

- **Decode is the weak phase.** b1 is below FP16 for all three sizes; only 8B/b8
  breaks even. The 3.2× weight compression does not yet translate into decode
  speedup — KV/attention are uncompressed and the GEMV under-saturates HBM. This
  is the primary remaining gap.
- **8B down_proj producer is not fused.** The fused SwiGLU+Hadamard+pack
  producer is instantiated only for intermediate width 8192 (1B, 3B). 8B's
  width is 14336 = 7·2048 (not a power of two), so it silently falls back to the
  reference chain (silu·up → FP32 Hadamard → FP16 → pack). This is arithmetically
  bit-identical (GEMM is still the real kernel) but the producer is slower,
  shaving some of 8B's prefill/decode advantage. Recorded, not hidden.
- **Prefill gain is modest vs the ~3× arithmetic-intensity reduction** — bounded
  by pack overhead, unaccelerated attention (Amdahl), and a strong cuBLAS FP16
  baseline.
- **8B PPL at bs=1** (bs=16 OOMs on the logits tensor). Numerically identical,
  just slower to evaluate.

## Artifacts (remote, `kernels/mixed_gemm_sm120/`)

`multisize_1b.json` (prefill+decode+weight), `multisize_3b_prefill.json`,
`multisize_3b_decode.json` (+weight), `multisize_8b_prefill.json`,
`multisize_8b_decode.json` (+weight), `multisize_3b_ppl.json`,
`multisize_8b_ppl.json`, and the archived 1B `full_network_ppl_frozen.json`.
