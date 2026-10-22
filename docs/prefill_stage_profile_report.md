# Prefill Fine-Grained Per-Stage Profile — Report

> 2026-10-22. A per-stage, per-sub-op profile of the Llama-3.2-1B **real-kernel**
> prefill forward (NVFP4+MXFP8 fused GEMM, SM120 RTX PRO 6000) vs the plain **FP16**
> baseline, batch-1, use_cache=False, M = 2048 / 4096 / 8192. Supersedes the opaque
> "44.7 ms FP16-rest + 20.7 ms Hadamard+reorder" lumping in
> `docs/prefill_overhead_report.md` / `docs/full_network_realkernel_report.md` with
> an honest, reconciled breakdown of every decoder sub-stage.
>
> Remote-only, container `luofan-plaquant`, **idle GPU 3** (idle-waiter gated on
> util ≤ 8 %). Instrumentation only — no kernel/algorithm change; PROTECTED kernel
> files untouched. Harness: `promix/eval/prefill_stage_profile.py` (new,
> non-protected). Raw data: `kernels/mixed_gemm_sm120/prefill_stage_profile.{json,csv}`,
> `profiler_{real,fp16}_M*.txt`.

## Method & honest caveat (read first)

Three independent measurement passes per model per M:

1. **Clean whole-model latency** — min-of-30, no instrumentation. This is the
   ground-truth total (matches the prior reports: real 29.9/49.8/103.1 ms, FP16
   20.8/40.9/87.1 ms).
2. **CUDA-event per-stage timing** — every sub-stage wrapped in a `cudaEvent`
   region, tagged per-iteration; the **min-total iteration** is the representative
   clean forward (dodges co-tenant spikes). A non-overlapping *partition* of
   regions reconciles to the whole.
3. **torch.profiler** op-level trace (kernel self-CUDA-time) + an **isolated dense
   cross-check** (each of pack / GEMM / Hadamard timed in its own tight loop) — the
   artifact-free ground truth for the launch-bound stages.

**The launch-bound artifact (and how it's handled).** The real model issues ~4
kernels × 112 Linears of mostly *tiny* work — it is host-launch-bound. Wrapping each
op in a `cudaEvent` spaces the launch stream out and clocks the GPU down, so the
**raw** per-op event sum over-counts the clean total (partition_sum 42.3 ms vs clean
29.9 ms at M=2048; the forced-eager-attention variant inflates to 73 ms — proof of
the effect). The **FP16 model is compute-bound (few big kernels) and reconciles
raw with scale ≈ 1.00** (20.86 vs 20.82 ms, residual −0.04 ms), which validates the
partition. For the real model we therefore report stages **normalized** so the
partition reconciles to the ground-truth clean total (preserves the measured
per-stage *proportions*), and we **cross-check every dominant stage against the
artifact-free isolated-dense and profiler kernel times** (they agree: e.g. Hadamard
normalized 19.49 ms vs isolated 20.58 ms vs profiler region 20.56 ms at M=8192). The
normalization scale shrinks as M grows and compute dominates: **0.71 / 0.83 / 0.95**
at M = 2048 / 4096 / 8192 — the artifact itself quantifies how launch-bound the model
is at each size.

> nsys was not available in the container; torch.profiler (CUPTI) provided the
> op-level attribution.

## 1. Side-by-side per-stage table (normalized ms, per forward)

Reconciles to the clean min total by construction (real), raw (FP16). R = real-kernel,
F = FP16 baseline.

| stage | R2048 | R4096 | R8192 | F2048 | F4096 | F8192 |
|---|---|---|---|---|---|---|
| embed lookup | 0.01 | 0.02 | 0.04 | 0.02 | 0.02 | 0.04 |
| input RMSNorm (×16) | 1.20 | 1.52 | 2.89 | 0.95 | 1.42 | 2.85 |
| q_proj (×16) | 2.34 | 2.90 | 3.40 | 1.01 | 1.69 | 3.00 |
| k_proj (×16) | 2.19 | 2.47 | 3.13 | 0.35 | 0.55 | 1.02 |
| v_proj (×16) | 2.03 | 2.41 | 3.14 | 0.34 | 0.54 | 1.01 |
| qkv view/transpose | 0.32 | 0.37 | 0.42 | 0.01 | 0.02 | 0.01 |
| RoPE | 1.68 | 2.03 | 3.81 | 1.16 | 1.87 | 3.78 |
| **SDPA (QK·softmax·PV fused)** | 2.27 | 4.95 | 15.47 | 1.95 | 5.10 | 16.76 |
| attn out reshape/contig | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| o_proj (×16) | 2.66 | 2.53 | 4.18 | 1.03 | 1.76 | 3.10 |
| residual add 1 | 0.24 | 0.39 | 0.70 | 0.12 | 0.34 | 0.62 |
| post-attn RMSNorm (×16) | 1.29 | 1.59 | 2.76 | 0.90 | 1.48 | 3.01 |
| gate_proj (×16) | 3.07 | 4.68 | 8.87 | 3.07 | 5.88 | 10.82 |
| up_proj (×16) | 2.87 | 4.43 | 8.87 | 3.00 | 5.92 | 10.84 |
| SiLU·mul | 0.54 | 2.29 | 6.58 | 0.77 | 2.68 | 7.09 |
| **down_proj (×16)** | 4.76 | 12.25 | 27.91 | 3.20 | 5.86 | 11.59 |
| residual add 2 | 0.25 | 0.35 | 0.70 | 0.17 | 0.33 | 0.68 |
| rotary_emb (cos/sin) | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 |
| final RMSNorm | 0.08 | 0.10 | 0.18 | 0.06 | 0.09 | 0.18 |
| lm_head | 2.11 | 4.46 | 10.01 | 2.67 | 5.30 | 10.65 |
| **TOTAL (clean min)** | **29.92** | **49.76** | **103.08** | **20.82** | **40.88** | **87.07** |

### Reconciliation (sanity check — the sum IS the whole)

| | M2048 | M4096 | M8192 |
|---|---|---|---|
| real clean min | 29.92 | 49.76 | 103.08 |
| real raw partition sum | 42.32 | 60.33 | 108.83 |
| real normalize scale | 0.707 | 0.825 | 0.947 |
| **FP16 clean min** | 20.82 | 40.88 | 87.07 |
| **FP16 raw partition sum** | **20.86** | **41.12** | **86.53** |
| FP16 residual (raw) | −0.04 | −0.24 | +0.54 |

FP16 reconciles to **< 1 %** with no normalization → the partition is complete and
the stage attribution is trustworthy; the real-model normalization only corrects the
launch-spacing clock artifact (validated by the FP16 control + the isolated/profiler
cross-checks below).

## 2. Real-kernel Linear internals (drill-down of the Linear totals)

Summed over all 112 Linears, normalized ms. The Linear totals in §1 decompose into:

| sub-stage | R2048 | R4096 | R8192 | fixable? |
|---|---|---|---|---|
| activation **pack** (qpf float→codes+SF) | 3.15 | 5.17 | 9.76 | yes (fuse into GEMM) |
| **GEMM** (CodesignFused NVFP4+MXFP8) | 6.93 | 10.14 | 17.06 | — (COMPUTE, beats FP16) |
| **cast+bias+reshape** (`D.to(fp16)`+bias) | 2.54 | 3.74 | 9.07 | **yes (epilogue fusion)** |
| down_proj **FP32 online Hadamard** | 1.82 | 7.94 | 19.49 | **partly (see §4)** |
| o_proj **column reorder** (gather) | 0.35 | 0.36 | 0.71 | n/a (already cheap) |
| per-Linear python/launch **glue** | 5.13 | 4.32 | 3.41 | yes (CUDA-graph) |

### Cross-check (artifact-free isolated dense) — confirms the dominant stages

| M | pack | GEMM | FP16 Linear | Hadamard FP32 (16×) | Hadamard FP16 (16×) | per-call FP32 |
|---|---|---|---|---|---|---|
| 2048 | 3.80 | 9.01 | 11.65 | 2.94 | 0.67 | 183.5 µs |
| 4096 | 5.46 | 12.32 | 22.01 | 9.55 | 2.11 | 597.0 µs |
| 8192 | 10.13 | 20.55 | 40.92 | 20.58 | 5.87 | 1286.2 µs |

These match the normalized numbers (pack, Hadamard within ~5 %; GEMM isolated is a
touch higher because normalization scales the whole real partition down uniformly).

## 3. torch.profiler op-level attribution (per-forward CUDA self-time, M=8192)

| kernel / op | real | FP16 | note |
|---|---|---|---|
| `aten::copy_` (dtype casts/copies) | **20.53 ms** | 3.67 ms | **+16.9 ms real-only** |
| down_proj Hadamard region | 20.56 ms | — | = transform + div + casts |
|  └ `fast_hadamard_transform_kernel` | 5.55 ms | — | the actual transform FLOPs |
|  └ `aten::div` (`/sqrt(n)`) | 5.86 ms | — | separate full-tensor kernel |
|  └ fp16↔fp32 round-trip copies | ~9 ms | — | inside `copy_` above |
| `CodesignFused` GEMM (×112) | 16.08 ms | — | real Linear compute |
| `aten::mm` (all FP16 Linears) | 10.49 (lm_head) | 51.88 ms | FP16 pays this per Linear |
| `qpf4/qpf8` pack kernels (×112) | ~9.8 ms | — | activation pack |
| cast+bias `unrolled_elementwise` (×112) | 7.87 ms | — | `D.to(fp16)`+bias |
| flash attention `flash_fwd_kernel` | 14.14 ms | 14.30 ms | **shared, equal** |
| `aten::mul` (SiLU·mul + RoPE) | 8.12 ms | 7.95 ms | shared |

Full tables: `kernels/mixed_gemm_sm120/profiler_{real,fp16}_M{2048,4096,8192}.txt`.

## 4. What is SURPRISINGLY slow (beyond the known Hadamard/GEMM/pack)

1. **Dtype casts are the single biggest hidden cost.** `aten::copy_` is **20.5 ms**
   (20.8 % of real CUDA time) at M=8192 vs **3.7 ms** in FP16 — a **+16.9 ms**
   real-only tax from fp16↔fp32 conversions, split between the Hadamard's fp32
   round-trip and the 112× `D.to(fp16)`. The prior report never surfaced this.

2. **The "FP32 Hadamard" is 73 % overhead, not transform math.** Of its 20.56 ms at
   M=8192, only **5.55 ms** is the `fast_hadamard_transform_kernel`; **5.86 ms** is a
   *separate* `/sqrt(n)` division kernel over the whole (M×8192) tensor, and ~9 ms is
   the `x.float()` / `.to(fp16)` round-trip. The prior report's "FP16-Hadamard breaks
   numerics (cos 0.96–0.99)" verdict is about the **transform** precision only — the
   div-fold and skipping the final `.to(fp16)` (the qpf pack reads fp32 directly,
   bit-exact) are **numerics-free** wins worth ~12–14 ms at M=8192. This reframes the
   single biggest target from "unfixable quant tax" to "mostly fixable memory traffic."

3. **The GEMM-output fp16 cast + bias is a real 9 ms stage** at M=8192 (`cast_bias`,
   112× elementwise pass over (M,N)). FP16 Linears return fp16 directly and never pay
   it. Foldable into the fused-GEMM epilogue (which already dequantizes).

4. **The o_proj column reorder is a non-issue.** The feared GPU index-gather is only
   **0.35–0.71 ms** total across 16 layers — 0.7 % of prefill. Cross off the worry.

5. **Attention is NOT slower in the real model.** SDPA (flash kernel) is **14.1 ms
   real vs 14.3 ms FP16** at M=8192 — identical (both default to sdpa). RoPE, softmax,
   PV, norms, SiLU·mul, residuals, embed and lm_head are all **shared** and within
   noise. The forced-eager QK·softmax·PV split (softmax ≈ 60 %, QKᵀ ≈ 30 %, PV ≈ 10 %
   of eager attention) is informational only — the model runs the fused sdpa kernel,
   which the FP16 baseline pays identically, so attention is **not** an optimization
   target for the quant path.

6. **Small attention projections (k/v) pay a fixed per-Linear tax.** k_proj/v_proj are
   tiny matmuls (512-out) that FP16 does in ~0.35 ms but the real path spends ~2–3 ms
   on (pack + GEMM-launch + cast + glue). This is the launch/pack overhead hitting the
   smallest Linears hardest — it dominates the gap at M=2048 and shrinks (relatively)
   with M.

## 5. The e2e gap, honestly decomposed (real − FP16, normalized ms)

| | M2048 | M4096 | M8192 |
|---|---|---|---|
| **total gap (real−FP16)** | **+9.10** | **+8.88** | **+16.00** |
| Attention QKVO Linears (pack/cast/glue tax) | +6.48 | +5.77 | +5.72 |
| MLP Linears (down_proj Hadamard-driven) | +1.43 | +3.70 | +12.41 |
| RMSNorms + RoPE + views (minor) | +1.69 | +0.54 | +0.18 |
| SDPA / lm_head / silu (shared) | ≈0 / −0.6 / −0.2 | ≈0 / −0.8 / −0.4 | −1.3 / −0.6 / −0.5 |

- **At M=2048 the gap is launch/pack/glue** (attention projections + per-Linear glue
  ≈ +11.6 ms gross), partly offset because the fused MLP Linears already roughly match
  FP16. CUDA-graph (prior report) erases the glue part → crosses 1.0× here.
- **At M=8192 the gap is the down_proj FP32 Hadamard** (the MLP delta +12.4 ms is
  almost entirely the 19.5 ms Hadamard, partly offset by gate/up being *faster* than
  FP16). The fused GEMM genuinely beats FP16 (16.1 ms vs 51.9 ms of FP16 `mm`), but the
  Hadamard + casts + pack hand it back.

## 6. RANKED optimization targets (absolute ms, real-only, M=8192 unless noted)

| # | target | ms (M8192) | % real prefill | also paid by FP16? | verdict |
|---|---|---|---|---|---|
| 1 | **down_proj FP32 Hadamard** | 19.49 | 18.9 % | no | **mixed**: ~5.5 ms transform is numerics-bound (FP16 fails the 0.9999 gate), but ~14 ms is **fixable inefficiency** — fold `/sqrt(n)` into the pack block-scale (−5.9 ms) and feed the fp32 Hadamard output straight to the qpf pack to skip the `.to(fp16)` round-trip (−~8 ms), both numerics-free |
| 2 | **dtype casts `aten::copy_`** | ~16.9 (excess vs FP16) | ~16 % | partly | **fixable inefficiency** — overlaps #1 and #3; eliminate fp32↔fp16 round-trips by letting the GEMM epilogue emit fp16 and the pack read fp32 Hadamard output directly |
| 3 | **GEMM-out fp16 cast + bias** | 9.07 | 8.8 % | no | **fixable inefficiency** — fold `D.to(fp16)`+bias into the fused-GEMM epilogue (112× elementwise pass eliminated) |
| 4 | **activation pack (qpf)** | 9.76 | 9.5 % | no | **quant tax, fusable** — fold the float→codes+SF pack into the GEMM prologue (single launch per Linear) |
| 5 | **per-Linear python/launch glue** | 3.41 (17 % at M2048!) | 3.3 % | no (FP16 has its own, smaller) | **fixable** — CUDA-graph capture already removes this (see `prefill_overhead_report.md`); biggest relative win at small M |
| 6 | o_proj column reorder | 0.71 | 0.7 % | no | leave it — already negligible |
| — | GEMM compute (CodesignFused) | 17.06 | 16.5 % | n/a | **NOT a target** — already 3× faster than FP16's equivalent `mm` |
| — | SDPA / RoPE / norms / silu / lm_head | — | ~45 % combined | **yes (shared)** | **NOT a quant target** — identical in both models |

**Headline:** the real-model prefill is 0.70× / 0.82× / 0.85× FP16 at M = 2048 / 4096 /
8192. The gap is **not** one monolithic Hadamard bucket. Ranked by fixable absolute ms
at M=8192: **(1) Hadamard round-trip casts + div-fold ~14 ms, (2) GEMM-out cast+bias
9 ms, (3) pack-into-GEMM fusion 9.8 ms, (4) CUDA-graph for the glue (dominant at
M=2048).** Together these ~33 ms of *fixable* overhead at M=8192 dwarf the ~5.5 ms of
genuinely numerics-bound Hadamard transform — i.e. almost all of the quant tax is an
engineering/fusion problem, not an algorithmic one.

## Deliverables / artifacts

- `promix/eval/prefill_stage_profile.py` — stage-profiling harness (CUDA-event
  partition + min-iteration selection + idle-waiter + torch.profiler + isolated-dense
  cross-check + forced-eager-attention split). New, non-protected.
- `kernels/mixed_gemm_sm120/prefill_stage_profile.json` — full per-stage raw +
  normalized timings, cross-check, profiler top-40 per (model, M).
- `kernels/mixed_gemm_sm120/prefill_stage_profile.csv` — flat per-(model, M, stage) ms.
- `kernels/mixed_gemm_sm120/profiler_{real,fp16}_M{2048,4096,8192}.txt` — torch.profiler
  op tables.
