# C2 Rung B report — toward the true single-kernel fused mixed-precision GEMM (SM120)

Status: **DONE_WITH_CONCERNS** — **B0 survey complete + a validated intermediate
fallback shipped** (`mixed_gemm_fused_b`, cosine ≥ 0.999997, a modest decode win over
Rung A). The **true single-CUDA-kernel dual-phase mainloop (full B1/B2) was NOT reached**:
B0 established that it is a large multi-cycle effort against SM120's warp-specialized TMA
block-scaled collective, not reliably completable in one remote-only session. Evidence
(torch-profiler, ncu blocked by host perms) pinpoints the real bottleneck and the highest
-impact next step. Honest rung reached: **B0 (complete) + intermediate fallback (validated)**.

Hardware: RTX Pro 6000 Blackwell, **SM120 (sm_120a)** — NOT B200. No tcgen05 / TMEM /
2-SM cooperative / cluster multicast. Block-scaled `mma.sync` atoms + register accumulator,
~99 KB SMEM, GDDR7 ~1.8 TB/s (so avoiding gmem round-trips matters more than on B200).

---

## B0 survey — can CUTLASS 3.x ease the dual-atom single-accumulator kernel?

Read the SM120 block-scaled stack on remote (`third_party/cutlass`):
`cute/atom/mma_traits_sm120.hpp`, `gemm/collective/sm120_blockscaled_mma_tma.hpp`,
`gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`.

Findings:

1. **The accumulation itself is exactly the C0.4-predicted shared-accumulator pattern.**
   The collective mainloop's inner op is literally:
   ```cpp
   cute::gemm(tiled_mma,
              make_zip_tensor(tCrA(_,_,k), tCrSFA(_,_,k)),
              make_zip_tensor(tCrB(_,_,k), tCrSFB(_,_,k)),
              accum);                       // accum = FP32[4]/thread, CLayout SM80_16x8_Row
   ```
   Both atoms (`SM120_16x8x64_TN_VS` NVFP4, `SM120_16x8x32_TN_VS` MXFP8) resolve the
   **same** `CLayout = SM80_16x8_Row` → a single kernel *can* run Phase-1 NVFP4 then
   Phase-2 MXFP8 into one `accum` register tensor. The math is unblocked, as C0.4 said.
   `fp4_shift_A/B` (same header) handle the FP4 `<<2` placement; they are applied per
   k-block inside `copy_kblock` before the gemm.

2. **But the surrounding machinery is a heavy warp-specialized TMA collective, not an
   SM80-style cp.async loop.** `MainloopSm120TmaWarpSpecializedBlockScaled` +
   `PipelineTmaAsync`; the kernel layer has separate **MK and NK load pipelines**, a
   scheduler pipeline + a throttle pipeline, a cooperative TMA-store epilogue, and named
   barriers (`Sm120MainloopBarrier`). SF is TMA-loaded as `uint16_t` into swizzled SMEM,
   then partitioned to registers via ~150 lines of generic helpers (`thrfrg_SFA/SFB`,
   `partition_fragment_SFA/SFB`, `get_layoutSFA_TV/SFB_TV`, `SmemCopyAtomSFA/SFB`).

3. **No prebuilt primitive runs two different element types into one accumulator.**
   Grouped/array GEMM requires one element type per group. Driving two `CollectiveMma`
   instances in one custom kernel is possible in principle (the kernel owns `accum`), but
   `mma()` hard-codes `clear(accum)` at entry and is welded to its own TMA pipeline +
   warp roles + per-collective `SharedStorage`; a correct SMEM-union dual-drive requires
   copying and surgically editing the collective. Hand-rolling from raw atoms needs
   faithful replication of the FP4 `ldmatrix` (b4x16_p64), the swizzled A/B/SF SMEM
   layouts, and the SF partition — all debuggable only via ~2-min remote builds.

**B0 verdict:** B1 (true single kernel) is viable but is a from-scratch,
multi-build-cycle CuTe effort. In a remote-only session the risk of silent SF-layout
bugs is high. The documented **intermediate fallback** was chosen as the correct,
validated deliverable, with the full B1 recipe captured above for the next session.

---

## Intermediate fallback — `mixed_gemm_fused_b`  (file: `fused_gemm_ext.cu`)

Reuses the proven `nvcfg`/`mxcfg` CUTLASS configs unchanged. Reduces Rung A's **7**
device launches (4 SF-fill + 2 GEMM + 1 combine) to **4**:

1. **One merged SF-fill** (`fill_sf_merged`) — flat cumulative index space over all 4
   interleaved SF buffers (NVFP4 A/B + MXFP8 A/B), exactly `w0+w1+w2+w3` threads, no idle
   oversizing. Replaces 4 fill launches with 1.
2. **GEMM1** NVFP4 (K_low) → bf16 `D_low` (unchanged config).
3. **GEMM2** MXFP8 (K_high) with **C = `D_low`, beta = 1** → bf16 `D_comb` = acc_high +
   D_low. The combine-add is folded into GEMM2's epilogue — **no separate combine pass,
   no second bf16 round-trip through a combine kernel**.
4. **`scale_only_kernel`**: `out[m,n] = D_comb[m,n]·s_x[m]·s_w[n] → fp16`.

All on one stream, single `cudaStreamSynchronize`. (Note: an early version sized the
merged fill as `gridDim=(wmax,4)` and was 1.1–2.0× **slower** than Rung A from idle-block
overhead; the flat-index fix below is what makes it competitive — a cautionary data point
kept in the git history.)

### Correctness (cosine vs mixed oracle, GPU 0, RTX Pro 6000 SM120)

PASS at gate ≥ 0.999 on every shape, M ∈ {1,16,256,2048}:

```
  q_proj_1B    Klow=1792 Khigh=256 : rungB_vs_oracle = 0.999997   (rel ~2.28e-3)
  down_proj_1B Klow=7168 Khigh=1024: rungB_vs_oracle = 0.999997–0.999998
  q_proj_3B    Klow=2688 Khigh=384 : rungB_vs_oracle = 0.999997–0.999998
```
(rel_err ~2.3e-3 = bf16 product + bf16-C accumulate rounding, on par with Rung A's 1.7e-3.)

### Benchmark: Rung B vs Rung A vs 2-launch vs fp16 cuBLAS (ms; full `benchmark.csv`)

| shape        |    M | rungB  | rungA  | 2launch | fp16   | B/A       | B/fp16 |
|--------------|-----:|-------:|-------:|--------:|-------:|----------:|-------:|
| v_proj_1B    |    1 | 0.0530 | 0.0634 | 0.1237  | 0.0123 | **0.83x** | 4.31x  |
| v_proj_1B    |  256 | 0.0533 | 0.0612 | 0.1319  | 0.0232 | **0.87x** | 2.30x  |
| k_proj_1B    |    1 | 0.0540 | 0.0607 | 0.1320  | 0.0123 | **0.89x** | 4.40x  |
| q_proj_3B    |    1 | 0.0860 | 0.0930 | 0.1367  | 0.0123 | **0.92x** | 6.99x  |
| q_proj_1B    |    1 | 0.0575 | 0.0605 | 0.1260  | 0.0140 | **0.95x** | 4.12x  |
| q_proj_1B    |   16 | 0.0605 | 0.0659 | 0.1265  | 0.0126 | **0.92x** | 4.80x  |
| o_proj_1B    |   16 | 0.0574 | 0.0601 | 0.1254  | 0.0126 | **0.96x** | 4.56x  |
| up_proj_1B   |   16 | 0.1100 | 0.1229 | 0.1809  | 0.0128 | **0.90x** | 8.60x  |
| gate_proj_1B | 2048 | 0.2164 | 0.2119 | 0.5031  | 0.1939 | 1.02x     | 1.12x  |
| down_proj_1B | 2048 | 0.2516 | 0.2368 | 0.3124  | 0.2069 | 1.06x     | 1.22x  |
| down_proj_3B | 2048 | 0.2786 | 0.2878 | 0.3778  | 0.3070 | **0.97x** | 0.91x  |

**Summary:** Rung B is a **modest win in the decode regime** (small M and/or small N:
B/A ≈ 0.83–0.97x, i.e. up to ~17% faster — exactly where the task said the fused path
should help), and roughly tied (0.97–1.06x, within noise) at large prefill shapes. It
beats the 2-launch baseline everywhere. It does **not** uniformly beat Rung A, so it is a
**valid intermediate, not a full promotion over Rung A**; both still trail fp16 cuBLAS
except down_proj_3B M=2048.

---

## Profiling evidence (torch.profiler; ncu blocked)

`ncu --set full` fails in the container with **`ERR_NVGPUCTRPERM`** (GPU performance-counter
access is disabled at the driver/host level; not fixable from inside the unprivileged
container). Fell back to `torch.profiler` CUDA-activity timing (CUPTI kernel durations, no
perf-counter permission needed) for the per-kernel breakdown.

Per-iter device time, Rung B:

| kernel            | decode M=16 N=2048 | prefill M=2048 N=8192 | smallN M=16 N=512 |
|-------------------|-------------------:|----------------------:|------------------:|
| `fill_sf_merged`  | **14.1 us (42%)**  | **65.6 us (33%)**     | 4.5 us (19%)      |
| NVFP4 GEMM        | 11.0 us (33%)      | 58.7 us (30%)         | 10.9 us (46%)     |
| MXFP8 GEMM        | 7.2 us (21%)       | 34.3 us (17%)         | 7.1 us (30%)      |
| `scale_only`      | 1.1 us (3%)        | 38.1 us (19%)         | 1.1 us (5%)       |
| **total**         | 33.4 us            | 196.8 us              | 23.6 us           |

**Diagnosis (memory dimension dominates):** the **SF-fill is the single largest cost
whenever N·K is large** (42% at decode-N2048, 33% at prefill). It is a scattered byte-write
pass sized by `N·Klow/16 + N·Khigh/32` — i.e. dominated by the **weight-side** SFB
interleave, on GDDR7 where scattered writes are slow. Both Rung A and Rung B pay this
(Rung A in 4 launches). The `scale_only` kernel also becomes significant at large M·N
(19% at prefill: 16.7M bf16 reads + fp16 writes).

---

## Ranked next steps (evidence-backed)

1. **Hoist the weight SF-interleave out of the hot path (highest impact).** Weight scale
   factors are static across calls; pre-interleave SFB_low/SFB_high **once at model load**
   and pass the physical buffers in. At decode M=16/N=2048 this removes ~90% of the 14 us
   fill (SFB = 229k of the 231k scatter writes) → fill ~1 us, Rung B total 33→~20 us.
   Helps Rung A identically. Requires an API/data-contract change (pass pre-interleaved
   weight SF), hence deferred from this session.
2. **Small-tile-N GEMM for decode (KernelWiki cutlass PR-3176 / flashinfer PR-3152).**
   SM120 block-scaled GEMM supports **TileN = 32/64** for `Out^T = W^T·Act^T` decode; the
   default 128-N tile wastes 3/4 of N at N=512 (k/v_proj) and all small-M cases. PR-3152
   measured ~1.2x at concurrency 1 on SM121. (Note: sglang PR-10101's ClusterShape/TMA-
   multicast lever is **B200-only** — SM120 has no cluster multicast, ignore it.)
3. **Fold `scale_only` into GEMM2's epilogue via EVT** (per-row s_x × per-col s_w ×
   (acc + C) → fp16). Removes the 4th launch and the 16.7M-elem scale pass (−19% at
   prefill); gets Rung B to 2 launches. Non-trivial SM120 EVT wiring.
4. **The true B1/B2 single kernel** (recipe in B0 above). Removes the bf16 `D_low` gmem
   round-trip entirely (most valuable on GDDR7) and the SF double-handling. This is the
   real prize and the correct focus for a dedicated multi-build session; guard against
   register spilling (KernelWiki sglang PR-8127) given the dual-phase register pressure.

## Build / run (unchanged from C1/C2)

`load()` JIT, `TORCH_CUDA_ARCH_LIST=12.0a` → `-gencode compute_120a,sm_120a`, flags
`-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3`, CUTLASS includes.
Run: `CUDA_VISIBLE_DEVICES=<free> python mixed_gemm.py` in `luofan-plaquant`.
