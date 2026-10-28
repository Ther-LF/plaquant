# RD Perf Diagnostic — why shrinking the MXFP8 fraction only buys ~1%

**Question.** Shrinking the MXFP8 fraction `r = K_high / K_total` (shifting K from the
FP8 segment to the FP4 segment, total K fixed) should make the fused NVFP4+MXFP8 SM120
co-design GEMM faster. The *paper-peak* ideal, from `T(r) ∝ (1−r)/1469 + r/807`
(FP4 1469 / FP8 807 TFLOPS), predicts **1/8→1/16 ≈ +4.9%**, **1/8→1/24 ≈ +6.6%**,
**1/8→pure-FP4 ≈ +10.3%**. But a prior full-model sweep measured only **~1%** (prefill
×1.008 at 1/16). Why, and is the gap fixable?

**Answer (verdict first).** The +5–10% figure is **not realizable** on this kernel, and
the realized ~1% is **honest, not a bug**. The paper-peak ideal assumes the kernel
separates FP4 and FP8 by the 1.82× peak-TFLOPS ratio; at the relevant problem sizes this
kernel separates them by only **~1.2–1.5×**, so the *true* headroom for a 1/8→1/16 shift
is **~1.2–3%**, not 4.9%. Of that, the production dispatcher realizes **~0% on the K=3072
projections (q/k/v/o/gate/up) and ~3.5–4% on the K=8192 down_proj** — and because the
model is dominated by K=3072 layers, the weighted result is ~1%. The config dispatcher is
**not** at fault (it already picks the fastest tile at every K_high). The only lever that
pays 5–10% is dropping the MXFP8 phase **entirely** (pure-FP4), which is an accuracy
decision, not a free kernel win.

Hardware: RTX PRO 6000 Blackwell (SM120), GPU 4, idle. Kernel: `launch_best_codesign`
(`config_dispatch.hpp`, the production path used by `promix/quantize/mixed_linear.py`).
Harness: `kernels/mixed_gemm_sm120/rd_ratio.py` (+ `rd_cd_probe.cu` to force both tile
configs, `rd_ncu_point.py` for ncu). Raw data: `rd_ratio.csv`, `rd_ncu_summary.csv`.
Timing: cudaEvent, 12 warmup + 40 median iters, M = 2048 / 4096 (compute-bound).

---

## 1. Measured K_high → latency curve vs ideal (production `launch_best_codesign`)

For each 3B shape we sweep K_high in 64-wide ring steps (dense in the low-r region of
interest). Chosen config is **`codesign:128x128_nv4mx2` at every K_high** for M≥2048
(the dispatcher ignores K_high — see §2). Key operating points:

| shape (M=2048) | K_total | t_fp4 / t_fp8 (pure, ms) | **meas. FP4/FP8 ratio** | fused r≈0 | fused 1/16 | fused 1/8 | **realized 1/8→1/16** | ideal 1/8→1/16 (meas. EP) | ideal (peak) |
|---|---|---|---|---|---|---|---|---|---|
| qkv_3b  | 3072 | 0.1325 / 0.1586 | 1.20× | 0.1011 | 0.1068 | 0.1068 | **+0.0%** | +1.2% | +4.9% |
| gate_3b | 3072 | 0.2001 / 0.2798 | 1.40× | 0.1489 | 0.1595 | 0.1596 | **+0.0%** | +2.4% | +4.9% |
| down_3b | 8192 | 0.2716 / 0.3370 | 1.24× | 0.1608 | 0.1676 | 0.1735 | **+3.5%** | +1.5% | +4.9% |

| shape (M=4096) | K_total | FP4/FP8 ratio | realized 1/8→1/16 | ideal (meas. EP) | ideal (peak) |
|---|---|---|---|---|---|
| qkv_3b  | 3072 | 1.35× | ~+0.0%* | +2.1% | +4.9% |
| gate_3b | 3072 | 1.49× | −0.3% | +3.0% | +4.9% |
| down_3b | 8192 | 1.47× | +4.1% | +2.9% | +4.9% |

\* one noisy single-sample spike at qkv M4096 Kh=192 (`cos=nan`); the forced-128x128 probe
column gives ~+0.0% there.

Shape of the curve (see `rd_ratio.csv`): **smooth and monotonic in K_high, not stepped**,
and always on the same tile. There is no tile-band plateau *within the auto curve* — but
the slope in the low-r region is very shallow for K=3072 (0.1068 at both 1/16 and 1/8) and
only moderate for K=8192.

**Two things jump out:**

1. **The measured FP4/FP8 endpoint ratio is only ~1.2–1.5×, far below the 1.82× peak.**
   Re-anchoring the ideal interpolation to the *measured* endpoints (as instructed) drops
   the 1/8→1/16 ideal from +4.9% to **+1.2–3%**. The paper-peak number was never the right
   yardstick for these shapes: at M=2048–4096 the kernel is not fully tensor-bound, so it
   does not realize the peak-TFLOPS gap between the two formats. The ratio climbs toward
   peak with M (1.20→1.35 for qkv, 1.24→1.47 for down) but never reaches 1.82× here.

2. **The realized shift-payoff is ~0% for K=3072 and ~3.5–4% for K=8192.** The down_proj
   (K-heavy, large absolute K_high) actually *exceeds* its measured-EP ideal; the square /
   N-heavy projections (K=3072, small absolute K_high) realize essentially nothing.

(The `t_fp4`/`t_fp8` pure endpoints are measured through `launch_best_nvfp4/mxfp8`, which
re-interleave the weight SF inside the timed call and pick a different tile; they run
~20–25% slower than the fused kernel doing equivalent work, so the absolute `excess_pct`
column in `rd_ratio.csv` is anchored to an inflated line and is *not* the headline — the
realized-vs-ideal table above, and the fused kernel's own endpoints, are.)

---

## 2. Per-K_high config: the dispatcher is correct, not the problem

Leading hypothesis was tile-quantization in `select_codesign` — that it picks the SAME
tile across K_high so small K_high changes don't change the work. **Confirmed that it picks
the same tile, and confirmed that this is the right choice:**

- `select_codesign(M,N,K,K_high)` in `config_dispatch.hpp` takes `K_high` as an **ignored
  parameter** (`int /*K_high*/`) and returns `128x128_nv4mx2` for all M≥512 non-gate /
  M≥2048. The in-code comment already states the config optimum is *"essentially
  r-INDEPENDENT (round-19/24 verified across r{0,1/8,1/2})"*.
- We forced the other legal tile (`128x64_nv4mx2`) explicitly via `rd_cd_probe.cu` at every
  K_high. It is **always slower** (e.g. down_3b M2048 1/8: 128x64 = 0.184 ms vs 128x128 =
  0.172 ms; qkv similar). There is no better config being left on the table.

So hypothesis (a) *"fixable by finer per-K_high config dispatch"* is **refuted**. Keeping
one tile across K_high is optimal; the shallow curve is intrinsic to the kernel + problem,
not a dispatch mistake. (SM120 only admits 128-tall cooperative tiles with TK=128 here; a
hypothetical TK=64 tile to reduce K-granularity would be a kernel change, and §1 shows it
would still be capped by the ~1.2–1.5× format gap.)

---

## 3. ncu — does the MXFP8-phase work scale with K_high? (fixed total K, shift 1/16↔1/8)

`luofan-ncu`, single profiled `CodesignFusedKernel` launch, demangled-name filtered
(`rd_ncu_summary.csv`):

| point | K_total | K_high | tensor inst (`sm__inst_executed_pipe_tensor.sum`) | tensor-pipe active cycles | kernel dur (ns) |
|---|---|---|---|---|---|
| qkv  1/16 | 3072 | 192 | 2,654,208 | 225,890 | 75,808 |
| qkv  1/8  | 3072 | 384 | **2,654,208** | **225,890** | 75,072 |
| down 1/16 | 8192 | 512 | 6,684,672 | 568,908 | 154,688 |
| down 1/8  | 8192 | 1024 | 7,077,888 (+5.9%) | 602,373 (+5.9%) | 162,208 (+4.9%) |

- **K=3072 (qkv): the tensor work is byte-identical** between K_high=192 and 384. Shifting
  192 columns from FP8 to FP4 removes **zero** tensor-core work and leaves the kernel
  duration flat. The MXFP8 cost in this regime is tile-quantized (K_high rounds up to
  TK=128 multiples: 192→2 tiles, 384→3 tiles) and swamped by the fixed FP4 mainloop over
  the ~2.7–2.9k-wide K_low, which grows to keep K_total fixed. **The FP8 cost is *not*
  K_high-sensitive here.**
- **K=8192 (down): the tensor work *does* scale (+5.9%)**, because K_high is large enough
  (512↔1024 = 4↔8 TK=128 tiles) that the MMA-instruction-density difference bites. The FP4
  atom is `SM120_16x8x64` (64 K/inst); the FP8 atom is `SM120_16x8x32` (32 K/inst), so FP8
  issues ~2× the MMAs per K-column. Trading FP8 K for FP4 K therefore *does* cut tensor
  instructions — but only once K_high is a non-trivial multiple of the tile.

This is the mechanistic root cause: the fused kernel's FP8 cost scales with K_high **only
when K_high is a large multiple of TK=128**. For the dominant K=3072 projections, K_high at
1/8 is just 3 tiles (and 1/16 rounds up to 2), so the realized FP8 work barely moves.

---

## 4. Gap attribution

| cause | verdict | weight |
|---|---|---|
| **(c) range too narrow / small ceiling** — measured FP4/FP8 separation is ~1.2–1.5×, not the 1.82× peak the ideal assumed; honest 1/8→1/16 ideal is only +1.2–3% | **primary** | dominant |
| **(b) kernel FP8 phase does not scale with K_high in the K=3072 regime** — ncu: identical tensor work at 1/16 vs 1/8 for qkv; tile-quantized to TK=128 and swamped by the fixed K_low=K_total−K_high FP4 mainloop | **confirmed, major** | large for K=3072 shapes |
| **(a) tile-quantization in config dispatch** (fixable by finer per-K_high config) | **refuted** — dispatcher already picks the fastest tile (128x128) at every K_high; the alternative 128x64 is always slower; no TK<128 tile exists | not a lever |
| **(d) measurement noise in prior full-model run** | minor — a few single-sample `cos=nan` spikes seen here; the 47-point cudaEvent curves are otherwise smooth and reproduce the ~1% cleanly | small |

The full-model ~1% = weighted average of ~0% (K=3072: q/k/v/o/gate/up, the majority of
GEMM time) and ~3.5–4% (K=8192: down_proj). It is consistent with the GEMM microbench.

---

## 5. Verdict

- **Is the 5–10% payoff realizable on this kernel? No.** It rested on the 1469/807 ≈ 1.82×
  peak-TFLOPS ratio, which the kernel does not achieve at 3B shapes (it gets ~1.2–1.5×).
  Against *measured* endpoints the 1/8→1/16 ideal is only +1.2–3%, and the realized ~1%
  (full model) / +0% (K=3072) / +3.5–4% (K=8192, down) is close to that honest ceiling.
- **Realized incremental ceiling (shrinking the FP8 fraction): ~1–2% full-model, ~3–4% on
  down_proj only.** This is a genuine small ceiling, not a tuning miss.
- **The one lever that pays 5–10%+ is eliminating the MXFP8 phase entirely (pure NVFP4):**
  that removes the whole fixed FP8-phase cost and gives a discrete drop (fused r≈0 is
  5–10%+ below the 1/8 point here, and more once Phase 2 is skipped outright). That is an
  **accuracy trade**, not a kernel/dispatch fix.
- **What would be needed to approach even the honest ideal:** (i) a higher-MFU fused kernel
  that actually separates FP4/FP8 nearer the 2× instruction-density gap at M=2048–4096
  (big kernel lift; today it is pipeline/epilogue/memory-diluted), and/or (ii) finer K
  granularity (TK=64 / ragged-K) to de-quantize small K_high — but §1 shows this is still
  capped at ~2–3% for K=3072. **Not worth it**; the allocation effort is better spent on
  accuracy-driven pure-FP4 decisions where the real GEMM payoff lives.

Corroboration: the pre-existing 1B `ratio_curve.csv` (hand-rolled `b1` kernel, different
implementation) shows the same signature — a ~10% discrete jump the moment Phase 2 turns on
(down_class r=0: 0.138 ms → Kh=128: 0.152 ms) and a nearly flat band thereafter — i.e. a
fixed FP8-phase cost plus a shallow K_high slope, independent of kernel implementation.
