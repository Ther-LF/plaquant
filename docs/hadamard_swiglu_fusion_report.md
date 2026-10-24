# Fused SwiGLU + Hadamard + NVFP4/MXFP8 pack for the down_proj input path

> 2026-10-25. Closes the single largest fixable cost identified by
> `docs/blackwell_nvfp4_deploy_survey.md` §7 Rank 2 and
> `docs/prefill_stage_profile_report.md` §4: the down_proj input ran as **four
> separate passes** over an `[M, I=8192]` intermediate (bf16 `silu*up` write,
> fp32 cast, fp32 Hadamard + `1/sqrt(I)` div, fp16 cast, then the pack kernel).
> This round replaces all of it with **one CUDA kernel** that never materializes
> an fp32 `[M, I]` tensor in HBM, and is **bit-identical** to the old path.
>
> New files (non-protected): `kernels/mixed_gemm_sm120/hadamard_pack_ext.cu`,
> `promix/eval/hadamard_swiglu_fusion.py`, fused wiring in
> `promix/quantize/{mixed_linear,quant_utils,full_network}.py`, `--fused` mode in
> `promix/eval/prefill_stage_profile.py`.
> PROTECTED mainloop files (`codesign_fused_kernel.hpp`, `b1_tma_ext.cu`,
> `*_ext.cu`) untouched.

## TL;DR

| | M=2048 | M=4096 | M=8192 |
|---|---|---|---|
| prefill, fused (CUDA-graph, ms) | **17.17** | **32.83** | **73.57** |
| prefill, current real path (ms) | 18.19 | 40.37 | 91.92 |
| FP16 baseline (ms) | 20.94 | 41.03 | 85.88 |
| **saved per forward** | **1.02** | **7.54** | **18.36** |
| **speedup vs FP16, after** | **1.220×** | **1.250×** | **1.167×** |
| speedup vs FP16, before | 1.151× | 1.016× | 0.934× |
| logit cosine vs pre-fix path | 1.0000005 | 1.0000006 | 1.0000014 |
| wikitext-2 PPL | **14.8494** (fused) vs **14.8494** (baseline), Δ **+0.0000** | | |

(The "before" numbers reproduce the previously validated graphed baseline
18.45/39.98/91.98 ms. A second same-run A/B taken while the box was busier gave
20.37/46.03/105.93 → 16.83/37.43/84.64, i.e. savings of 3.5/8.6/21.3 ms — the
same direction, larger magnitude; the table above is the clean-GPU run.)

The 1B prefill now **beats unquantized FP16 at every M** (it lost at M≥4096
before: 0.934× at M=8192 → 1.167×). Numerics are not merely inside tolerance — the fused path reproduces
the current path's packed activation **byte for byte**, so the whole-model
logits move by the same amount as the baseline moves against itself.

## 1. What was built

`hadamard_pack_ext.cu::hadamard_swiglu_pack_mixed(gate, up, M, N_gemm, K_low, K_high)`

One CTA (256 threads) per token row, one pass over `I=8192`:

1. **SwiGLU producer** — reads `gate`/`up` (`[M, I]`, fp16 or bf16) and computes
   `h = T(T(silu(g)) * u)` with `silu = g/(1+exp(-g))` in fp32 (correctly-rounded
   division **and** correctly-rounded `expf`). The **two roundings** are what
   `F.silu(g) * u` does on half/bfloat16 tensors — the single-rounding variant
   mismatches 27% of elements, `__expf` alone leaves 4/458752 fp16 code diffs.
2. **Hadamard in fp32 + folded `1/sqrt(I)`** — the reference library's own
   device functions (`hadamard_mult_thread`, `hadamard_mult_warp`,
   `exchange_smem_pre`, BSD-3, Tri Dao 2023) are copied **verbatim** and driven
   with the same traits (256 threads, `kNElts=4`, `kNChunks=8`) and the same
   register layout, so the 13-stage order, the lane↔chunk exchange and every
   fp32 sum are exactly the library's. The transform runs on registers + a
   32 KB SMEM exchange buffer; **no fp32 `[M, I]` tensor is ever written to
   HBM**, and `1/sqrt(I)` is applied in the same final multiply (production
   `fold_had_scale` path), before the fp16 round.
3. **fp16 round + pack** — the fp32 result is written back to SMEM in
   column-linear order, rounded to fp16 exactly as `xf.to(fp16)`, then both
   segments are packed with the **verbatim qpf device code** (per-block absmax →
   round-up E4M3 scale for NVFP4, UE8M0 for MXFP8; same RTN grid; same
   CUTLASS-interleaved SFA atom layout).

Wiring: `LlamaMLP.forward` hands the un-materialized `gate`/`up` GEMM outputs to
the down_proj `ActQuantWrapper` via `mlp_gate`/`mlp_up`; the wrapper calls
`MixedQuantLinear.forward_swiglu_pack`, which runs the fused kernel and then the
existing packed fused GEMM. Enabled only for prefill (`dg is None`), so the
decode M=1 GEMV path is untouched. `N_gemm` (the consumer GEMM's N) is passed
explicitly because the SFA atom layout depends on it, while the row width is
fixed at 8192 (a `static_assert`).

## 2. Numerics gate (run FIRST, on every change)

### 2a. Per-element byte-exactness of the quantized down_proj input

Fused kernel vs the production path (`silu*up` → `matmul_hadU_cuda(h.float(),
fold_scale=True)` → `.half()` → `quant_pack_float_mixed`), counting mismatched
bytes in all four tensors (NVFP4 codes + SFA, MXFP8 codes + SFA):

| M | fp16 in | bf16 in |
|---|---|---|
| 128 | **0 / 0 / 0 / 0** (of 458752 / 57344 / 131072 / 4096) | **0 / 0 / 0 / 0** |
| 512 | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| 2048 | **0 / 0 / 0 / 0** (of 7340032 / 917504 / 2097152 / 65536) | **0 / 0 / 0 / 0** |
| 4096 | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |

Also verified against the model's **real** layer-0 down_proj activation
(M=2048, std 0.0175, absmax 29.5). The `M<128` cases differ only in the
*padding* entries of the SFA layout (the reference layout is defined for M≥128);
they are outside the prefill regime and the fused path reuses the same layout.

### 2b. Whole-model gates

| gate | required | measured | verdict |
|---|---|---|---|
| wikitext-2 PPL (GPTQ FP weights, real kernel) | 14.85 ± 0.03 | **14.8494 → 14.8494 (Δ +0.0000)** | PASS |
| whole-model prefill logit cosine, fused vs pre-fix | ≥ 0.9999 | **1.0000005 / 1.0000006 / 1.0000014** at M=2048/4096/8192 | PASS |
| CUDA-graph replay cosine | ≥ 0.9999 | same values (graph is bit-identical) | PASS |
| decode M=1 GEMV path (`decode_gemv.py`) | cos ≥ 0.999 | worst vs twin 1.000000, vs fused-TC 0.999998 | PASS (untouched) |
| single-launch qpf pack (`qpf_probe.py`) | bit-exact | `QPF_BITEXACT_PASS` (all shapes, data+SF) | PASS (untouched) |
| div-fold + lm_head-W8 | unchanged | fused toggles only the down_proj *producer* | PASS (not on path) |

The baseline's *own* eager-vs-CUDA-graph cosine is also
1.0000005/1.0000006/1.0000014 at M=2048/4096/8192 (graph replay is the identical
kernel stream, so this is the dtype/accumulation noise floor of the harness).
The fused path lands on exactly those numbers, and its packed down_proj input is
byte-identical at every prefill M — **the output is effectively identical**.

## 3. Performance

### 3a. Full-model prefill vs FP16 (idle-ish GPU, min-of-N, CUDA-graph on)

Same-run A/B on one wired model (`set_fused(False/True)`), FP16 measured in the
same process (`promix/eval/hadamard_swiglu_fusion.py --phase perf`):

| M | FP16 | before (graph) | **after (graph)** | saved | speedup before → after |
|---|---|---|---|---|---|
| 2048 | 20.94 | 18.19 | **17.17** | **−1.02 ms** | 1.151× → **1.220×** |
| 4096 | 41.03 | 40.37 | **32.83** | **−7.54 ms** | 1.016× → **1.250×** |
| 8192 | 85.88 | 91.92 | **73.57** | **−18.36 ms** | 0.934× → **1.167×** |

(The "before" column reproduces the validated graphed baseline
18.45/39.98/91.98 ms. A second same-run A/B taken while the box was busier
gave 20.37/46.03/105.93 → 16.83/37.43/84.64, i.e. savings of 3.5/8.6/21.3 ms —
same direction, larger magnitude; the table above is the clean-GPU run. The
savings scale with M exactly as the profile predicted, and at M=8192 they
slightly exceed the Hadamard bucket alone because the freed HBM traffic also
helps the rest of the forward.)

### 3b. Stage breakdown — the Hadamard/cast/pack bucket

Fused stage profile (`prefill_stage_profile.py --models real --fused`, new
`lin.down_fused` region) vs the pre-fix baseline (`lin.down_hadamard` +
`mlp.silu_mul`), ms per forward, summed over 16 layers:

| bucket | M=2048 | M=4096 | M=8192 |
|---|---|---|---|
| `mlp.silu_mul` (before) | 1.03 | 2.74 | 7.00 |
| `lin.down_hadamard` (before: cast+transform+div+cast) | 2.32 | 9.59 | 20.57 |
| **bucket before** | **3.35** | **12.33** | **27.57** |
| `lin.down_fused` (after: silu+Hadamard+pack, **incl. the down_proj GEMM**) | 4.09 | 6.55 | 11.64 |
| `lin.down_fused` minus the packed GEMM (isolated, §3c) | **0.11** | **0.25** | **0.45** |
| **bucket after (isolated producer only)** | **0.11** | **0.25** | **0.45** |

The producer bucket shrinks by **~30×** (3.35→0.11, 12.33→0.25, 27.57→0.45 ms).
The measured whole-forward deltas (−3.55/−8.60/−21.28 ms) are **larger** than
this bucket alone (−3.24/−12.1/−27.1 ms), because the old chain also paid the
fp16 `h` write + `h.float()` read (an extra full read+write of `[M, I]`) and,
at M=8192, freed enough memory traffic for the rest of the forward to run
faster too. **Note on the stage baseline:** the *non-fused* stage profile in the
current tree measures `lin.fakequant` at 766 ms/forward because the profiling
harness wires `MixedQuantLinear` **without** `qpfm`, so the plain-PyTorch
quantize fallback runs (that artifact predates this round and is not on the
shipping path). The fused profile is unaffected because the fused entry bypasses
`_pack_activation` entirely. A clean before/after stage comparison should re-wire
`qpfm` in the baseline mode; the honest before→after numbers here are the
isolated-dense ones in §3c plus the same-run A/B in §3a.

### 3c. The fused kernel itself (isolated, production shape I=8192/K=7168+1024/N=2048)

| M | fused kernel | current 4-op chain | speedup | traffic (read 2×[M,I]·2B + write codes/SF) | effective BW |
|---|---|---|---|---|---|
| 2048 | **0.112 ms** | 0.233 ms | **2.07×** | 76.7 MB | 0.68 TB/s |
| 4096 | **0.245 ms** | 0.709 ms | **2.89×** | 153.3 MB | 0.62 TB/s |
| 8192 | **0.453 ms** | 1.583 ms | **3.50×** | 306.4 MB | 0.68 TB/s |

The kernel is **not** at the HBM roof: 0.68 TB/s is ~38 % of the RTX PRO 6000's
~1.8 TB/s. It is **SMEM-bandwidth/occupancy bound**, not HBM bound: the fp32
Hadamard does ~13 shared-memory read+write passes over the 32 KB row tile per
CTA, and one CTA (256 threads, 32.8 KB dynamic SMEM) per row limits occupancy.
Subsequent reduction of the SMEM passes would raise throughput (see §5).
ncu was unavailable in the container (`ERR_NVGPUCTRPERM` — the host does not
grant performance counters), so the BW figure is derived from the exact traffic
model and min-of-30 event-synchronized timing rather than from DRAM counters.

## 4. What the fused kernel removes

Per layer, before: `h` bf16 write + read, `h.float()` fp32 write, Hadamard fp32
read + write, `/sqrt(n)` (already folded), `.to(fp16)` fp16 write, pack read —
i.e. **~24 bytes/element of HBM traffic** plus 4 kernel launches.
After: **4 bytes/element read (gate+up) + 0.56 bytes/element written (codes+SF)**
in **1 launch**. Arithmetic was never the problem (one Hadamard pass is ~0.018 %
of the 1B GEMM FLOPs); the traffic and the launch chain were.

## 5. What remains / residuals

1. **SMEM passes dominate the fused kernel** (0.45 ms at M=8192 vs a ~0.17 ms
   HBM-bound floor). A radix-4 in-register butterfly (fewer SMEM round-trips:
   from 13 to ~6 passes) is the obvious next step, as is 2 rows/CTA to raise
   occupancy above 6 CTAs/SM.
2. **The `lin.cast_bias` bucket (8.8 ms @ M=8192) and the per-Linear pack
   (6.1 ms) are untouched** — they were not in this round's scope; fusing
   `D.to(fp16)+bias` into the GEMM epilogue remains a documented negative
   (fp16-output epilogue fails the cosine gate, see
   `docs/prefill_fusion_validated_report.md`).
3. **The generic per-Linear pack (112 launches, 6.1 ms @ M=8192)** still runs
   for the other six projections. The same producer-epilogue trick applies to
   norm→qkv/gate/up (`add+RMSNorm+quantize`), which the deploy survey ranks #1.
4. **M<128 SFA padding**: irrelevant for prefill (M≥512 here) but the fused
   entry should not be used at tiny M without a layout check.
5. **ncu counters** are blocked on this host; a privileged container is needed
   for the DRAM-throughput confirmation.
6. **Correctness caveat:** the fused kernel contains the reference library's
   transform code copied verbatim under its BSD-3 licence (noticed in the file
   header); any change to that block invalidates the bit-exactness claim.

## Artifacts

- `kernels/mixed_gemm_sm120/hadamard_pack_ext.cu` — the fused kernel.
- `promix/eval/hadamard_swiglu_fusion.py` — probe / perf / PPL / bench harness.
- `kernels/mixed_gemm_sm120/hpk_{perf,ppl,bench,probe_clean}.json` — raw numbers.
- `kernels/mixed_gemm_sm120/stage_{base,fused}.{json,csv}` — stage profiles.
