# RD Perf-Puzzle — where the NVFP4-vs-MXFP8 speedup is lost end-to-end

**Branch:** `feat/rd-perfpuzzle` (off `origin/feat/rd-alloc`) · **HW:** RTX PRO 6000
Blackwell (SM120), docker `luofan-plaquant`, idle GPUs · **Date:** 2026-10-29

**Puzzle.** In the full-model Pareto sweep (`docs/rd_pareto_sweep.csv`), PURE-NVFP4
(r=0) prefill latency is essentially the SAME as the 1/8 mixed config — even
~1% *slower* (8B r=0 ×1.009, 3B ×0.9965, 1B ×0.9948). The paper-peak ideal
`T(r)∝(1−r)/1469+r/807` (FP4 1469 / FP8 807 TFLOPS = 1.82×) predicts dropping the
whole MXFP8 segment should buy ~10%. Where does that go?

**One-line answer.** The ~10% is lost in **four stacked places, and never actually
measured at r=0 in the first place**: (1) at our 1B–8B single-sequence Linear
shapes the FP4/FP8 GEMM ratio is only ~**1.3×**, not 1.82×, because the GEMMs run
at **10–42 % tensor-core utilization** (not compute-bound); (2) the *production
fused kernel* exposes **no r=0 lever at all** — its GEMM latency is **flat in
K_high** (tile-quantized Phase-2 + the removed FP8 columns are re-added to the
growing FP4 Phase-1), and the only true K_high=0 path (the dedicated pure-NVFP4
dispatcher) is **24–65 % *slower*** than the fused mixed kernel; (3) the
format-sensitive GEMM is only **~26 % (3B) / ~33 % (8B) of prefill wall-clock**, so
any GEMM win is diluted 3–4×; and (4) the sweep's "r=0" prefill row is a **harness
fallback to the 1/8 config** — the fused kernel *cannot build* a K_high=0 Linear,
so r=0 ≈ 1/8 **by construction**. The format ratio only recovers toward 1.82× at
**M≥8192 / K=N≥8192** (it scales with batch + model size), so the lever is real for
serving/large models, not for 1B–8B single-seq prefill.

Harness: `promix/scripts/rd_perfpuzzle.py` (full-model prefill decomposition,
cudaEvent + torch.profiler), `kernels/mixed_gemm_sm120/pp_ratio.py` (pure-format
GEMM ratio sweep), `kernels/mixed_gemm_sm120/rd_ratio.py` (fused-kernel K_high
sweep, clean 40-iter median). Data: `docs/rd_perfpuzzle_ratio.csv`,
`docs/rd_perfpuzzle_fused_khigh.csv`.

---

## The waterfall: 1.82× hardware ratio → ~1.00× end-to-end

| stage | what | measured ratio | why it shrinks |
|---|---|---|---|
| **1. HW peak** | pure-FP4 vs pure-FP8, 16384³ square | **2.05×** (8192³ 1.59×, 4096³ 1.41×) | TC saturates: FP4 67 % / FP8 60 % of peak |
| **2. LLM shapes** | pure-FP4 vs pure-FP8, real Linear shapes @ M=2048 | **~1.3×** (1.10–1.55) | TC util only **10–42 %** — prologue/epilogue/mem-bound, not compute-bound |
| **3. Production K_high lever** | fused kernel, 1/8 → near-pure (K_high→64) | **~1.00× (±7 %)** | GEMM latency **flat in K_high**; no K_high=0 path (dedicated pure-NVFP4 is 24–65 % slower) |
| **4. e2e dilution** | GEMM share of full prefill wall-clock | GEMM = **26 % (3B) / 33 % (8B)** | ÷3–4; rest is r-independent FP16 (attn, lm_head, norm, rope, hadamard, casts) |
| **(5. harness)** | sweep "r=0" latency row | = 1/8 config (`avg_fp8_realized=0.125`) | kernel can't build K_high=0 → **uniform fallback** |

**Net:** `1 / (0.74 + 0.26/1.3) ≈ 1.07×` is the *absolute* e2e ceiling if the GEMM
realized even the 1.3× shape ratio; the production fused kernel realizes ~0 of it
at the K_high knob, so e2e lands at **~1.00×** — exactly the sweep's ~1 %.

---

## Stage 1 — hardware peak ratio (confirmed ≈1.8–2.0×)

Pure-NVFP4 vs pure-MXFP8, dedicated SM120 CUTLASS dispatchers, large square GEMM
(`pp_ratio.py`, 40-iter median, idle GPU). Peaks per `docs/hardware_format_peak.md`
(FP4 1469 / FP8 807 TFLOPS = 1.82×).

| M=N=K | t_fp4 (ms) | t_fp8 (ms) | **ratio** | FP4 %peak | FP8 %peak |
|---|---|---|---|---|---|
| 4096 | 0.248 | 0.348 | 1.41× | 38 % | 49 % |
| 8192 | 1.185 | 1.887 | 1.59× | 63 % | 72 % |
| 16384 | 8.934 | 18.327 | **2.05×** | 67 % | 60 % |

The hardware delivers the ~1.8–2.0× format ratio **once the GEMM is big enough to
saturate the tensor cores**. (FP8 falls off faster at 16384³, pushing the ratio
past 1.82.)

## Stage 2 — our LLM Linear shapes collapse to ~1.3× (the main loss)

Same pure dispatchers, the actual 3B/8B decoder-Linear shapes at M=2048 (prefill):

| shape (M=2048) | N | K | t_fp4 | t_fp8 | **ratio** | FP4 %peak | FP8 %peak |
|---|---|---|---|---|---|---|---|
| 3b q/o_proj | 3072 | 3072 | 0.132 | 0.162 | 1.22× | 20 % | 30 % |
| 3b k/v_proj | 1024 | 3072 | 0.089 | 0.098 | 1.10× | 10 % | 16 % |
| 3b gate/up | 8192 | 3072 | 0.203 | 0.284 | 1.40× | 35 % | 45 % |
| 3b down | 3072 | 8192 | 0.273 | 0.357 | 1.31× | 26 % | 36 % |
| 8b q/o_proj | 4096 | 4096 | 0.173 | 0.229 | 1.33× | 27 % | 37 % |
| 8b gate/up | 14336 | 4096 | 0.386 | 0.568 | 1.47× | 42 % | 52 % |
| 8b down | 4096 | 14336 | 0.482 | 0.746 | 1.55× | 34 % | 40 % |

**Tensor-core utilization is only 10–42 % at these shapes** → the kernel is
prologue/epilogue/memory-diluted, so it does **not** realize the 1.82× peak-TFLOPS
gap between formats. The ratio is highest where the shape is biggest (8b
gate/down 1.47–1.55) and lowest for the skinny k/v projections (1.10). Weighted
over the network it is ~1.3×.

### Scaling projection — the ratio recovers toward 1.8× with M and K/N

| sweep | points | ratio (fp8/fp4) | FP4 %peak |
|---|---|---|---|
| **M** @ sq N=K=4096 | 512 / 1024 / 2048 / 4096 / 8192 / 16384 | 1.15 → 1.26 → 1.33 → 1.41 → 1.49 → **1.51** | 10 → 55 % |
| **K=N** @ M=2048 | 2048 / 4096 / 8192 / 16384 | 1.21 → 1.33 → 1.64 → **1.65** | 15 → 46 % |

The format lever **scales with model + batch size**: at our 1B–8B single-sequence
shapes (M=2048, K=3072–4096, TC 10–40 %) it is ~1.3×; it climbs back toward 1.8×
as M≥8192 (serving batch) or K=N≥8192 (bigger models) push TC utilization up.

## Stage 3 — the production fused kernel has no K_high lever

The deployed path is the **fused NVFP4+MXFP8 kernel**, not the pure dispatchers.
Its GEMM latency vs K_high (`rd_ratio.py`, clean 40-iter median, `launch_best_codesign`):

| shape M=2048 | K_high=0 (pure-NVFP4 disp.) | 64 | 128 | 384 | 1/8 | note |
|---|---|---|---|---|---|---|
| qkv (K=3072) | 0.1325 | 0.1064 | 0.1011 | **0.1068** (=1/8) | — | flat 64↔384 (0.3 %); pure 0 is **+24 %** |
| down (K=8192) | 0.2718 | 0.1679 | 0.1630 | 0.1680 | 0.1760 (=1024) | 64 ≈ 1/8; pure 0 is **+65 %** |

Two facts kill the r=0 lever:

1. **Flat in K_high.** The FP8 Phase-2 is tile-quantized to TK=128 (K_high 64 and
   128 both = 1 tile), and every FP8 column removed is *re-added* to the FP4
   Phase-1 (K_low grows to hold K_total fixed). Net work barely moves — the fused
   curve is flat (qkv) to only mildly decreasing (down). Shrinking the MXFP8
   fraction within the fused kernel buys ≈0 %.
2. **No pure-NVFP4 path exists.** Building a K_high=0 Linear fails twice:
   `weight_pack.prepack_sf_fp8_b(..., K_high=0)` → *"interleave_sfb launch:
   invalid argument"*, and the activation pack `quant_pack_float_fp8(..., 0)` →
   *"qpf launch: invalid argument"*, and the fused GEMM itself asserts in CUTLASS
   TMA setup (`gmem_prob_shape[0] >= 1`) — the Phase-2 MXFP8 pack + TMA descriptor
   are **unconditionally constructed**. No K_high==0 guard exists anywhere in the
   codebase (checked both the base and the current `feat/rd-alloc` HEAD). The only
   r=0 path, the *dedicated* pure-NVFP4 dispatcher, is **24–65 % slower** than the
   fused mixed kernel (unoptimized separate kernel; TC-diluted at these shapes).

## Stage 4 — e2e dilution: the GEMM is ¼–⅓ of prefill

Full-model prefill (M=2048) on the real packed fused-kernel model, decomposed by
torch.profiler (per-iter GPU time, uniform-1/8):

| component | 3B (ms) | 3B %wall | 8B (ms) | 8B %wall |
|---|---|---|---|---|
| **fused quantized-Linear GEMM** (7×layers) | **13.80** | **25.7 %** | **27.03** | **32.6 %** |
| activation quant+pack (NVFP4 qpf4 + MXFP8 qpf8) | 7.05 | 13.2 % | 10.68 | 12.9 % |
| FlashAttention (SDPA, FP16) | 3.88 | 7.2 % | 5.15 | 6.2 % |
| **lm_head (FP16 GEMM, r-independent)** | 3.96 | 7.4 % | 5.11 | 6.2 % |
| RMSNorm + RoPE + Hadamard + casts + cat + residual (FP16) | ~24.9 | ~46 % | ~35.0 | ~42 % |
| clean prefill wall-clock | **53.60** | 100 % | **83.00** | 100 % |

The **format-sensitive GEMM is only ~26 % (3B) / ~33 % (8B)** of prefill wall-clock
(and ~33–39 % of pure GPU-kernel time). Everything else — attention, the FP16
lm_head GEMM, RMSNorm, RoPE, the down_proj online Hadamard, casts, concats — is
FP16 and completely r-independent. So a GEMM win is diluted ÷3–4 before it reaches
the wall clock.

**Direct confirmation of the near-pure build (K_high=64, the leanest the kernel
supports):** total GPU-kernel time *drops* ~9 % (3B 41.2→37.5 ms, 8B 70.0→64.7 ms)
and the fused GEMM drops ~6–7 % (3B 13.80→12.96, 8B 27.03→25.16) — a genuine but
tiny GEMM saving — yet clean prefill wall-clock does **not** improve (3B 53.6→60.0,
8B 83.0→87.1; the wall-clock delta is CPU-dispatch/clock variance, not GPU work).
The GEMM saving is real, small, and diluted into e2e noise — i.e. textbook
dilution.

## Stage 5 — the sweep's "r=0" number is a fallback, not a measurement

The fused kernel cannot build K_high=0 (Stage 3). So in the Pareto sweep the r=0
latency build failed and the harness fell back: `RD_SWEEP_3b_lat.json`
`measured@0.00000` carries **`avg_fp8_realized=0.125`, `bits_per_weight=4.969`** —
i.e. the *timed model was the uniform-1/8 config*. (The CSV's r=0 accuracy columns
— `avg_fp8 0.0144`, `ppl 10.91` — come from the *separate fake-quant* PPL build,
which can do K_high=0; the merge stitched 1/8 latency onto r=0 PPL.) So the
headline "r=0 prefill ≈ 1/8" is literally the 1/8 latency relabeled, plus the
~±1 % run-to-run noise that makes 8B look 1 % *slower*.

---

## Verdict & how to surface the FP4 advantage e2e

**Verdict — all four causes, with numbers:**
- **Not compute-bound (primary):** LLM-shape FP4/FP8 ratio is ~1.3× (TC 10–42 %),
  not the 1.82× peak.
- **No production r=0 lever:** fused GEMM is flat in K_high; the only pure-NVFP4
  path is 24–65 % *slower*; K_high=0 doesn't even build.
- **Dilution:** GEMM = 26–33 % of prefill wall-clock.
- **Harness artifact:** the sweep's r=0 latency is the 1/8-fallback value.

**To make the FP4 advantage show e2e:**
1. **Grow the format-sensitive share** — quantize more of the model to FP4:
   the FP16 lm_head GEMM (~6–7 %) and the attention projections' companions; a
   larger quantized fraction multiplies any GEMM win. Shrinking the *MXFP8*
   fraction is **not** a lever (Stage 3 flat).
2. **A real lean pure-NVFP4 GEMM path** (compile-time skip the Phase-2 MXFP8 pack +
   TMA when K_high==0) — but only pays once that kernel reaches the fused kernel's
   efficiency; today the dedicated pure-NVFP4 dispatcher is *slower* than
   fused-1/8 at our shapes, so this is a kernel-optimization project, not a free
   win.
3. **Serving batch / larger models** — the scaling curve shows the ratio recovers
   to ~1.6–1.8× at M≥8192 and K=N≥8192 (TC util 55–72 %). The format lever is a
   **batch-and-size lever**, strongest exactly where single-seq 1B–8B prefill is
   weakest.
