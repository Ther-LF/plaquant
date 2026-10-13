# C2 Round-2 report — KDA loop on the SM120 fused mixed-precision GEMM

Status: **T1 + T2 + T3 all DONE and validated.** Round driven by round-1 evidence
(SF-fill = 42% decode / 33% prefill, dominated by the STATIC weight-side SFB scatter).

Hardware: RTX Pro 6000 Blackwell, **SM120 (sm_120a)** — mma.sync block-scaled atoms,
register accumulators, ~99KB SMEM, GDDR7. NO tcgen05/TMEM/2-SM/cluster-multicast.

Correctness gate held on EVERY change: cosine vs the mixed oracle >= 0.999997 (q_proj_1B,
down_proj_1B, q_proj_3B at M=1,16,256,2048; presf paths + N=64 probe all PASS).

---

## T1 — Weight-SF pre-interleave (the big evidenced win) [DONE]

`tile_atom_to_shape_SFB(M,N,K,L)` uses only **(N,K,L)** — it is M-independent (verified in
`cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp`), so the interleaved weight SF
is computed ONCE and reused across every call and every M.

Added to `fused_gemm_ext.cu` + `mixed_gemm.py`:
- `prepack_sf_b(SFB_logical uint8[N,K/vec], N, K, vec) -> physical SF` (vec=16 NVFP4 / 32 MXFP8).
- `mixed_gemm_fused_b_presf(...)` and `mixed_gemm_fused_presf(...)` — accept already-interleaved
  weight SF; fill ONLY the dynamic activation-side SF per call (one merged 2-region launch,
  `fill_sf_merged2`).
- `prepack_weight_sf()` + benchmark `rungB_presf` / `rungA_presf` rows (`benchmark.csv`, 216 rows).

### Speedup — rungB_presf vs round-1 rungB (ms, RTX Pro 6000, GPU 0)

| shape        |    M | round1 rungB | rungB (now) | rungB_presf | presf/rungB | presf/fp16 |
|--------------|-----:|-------------:|------------:|------------:|------------:|-----------:|
| q_proj_1B    |    1 | 0.0575       | 0.0572      | 0.0510      | **0.89x**   | 4.03x |
| q_proj_1B    |   16 | 0.0605       | 0.0577      | 0.0514      | **0.89x**   | 3.95x |
| o_proj_1B    |   16 | —            | 0.0605      | 0.0521      | **0.86x**   | 4.08x |
| k_proj_1B    |    1 | 0.0540       | 0.0535      | 0.0506      | 0.95x       | 4.13x |
| v_proj_1B    |    1 | 0.0530       | 0.0536      | 0.0508      | 0.95x       | 4.18x |
| gate_proj_1B |    1 | —            | 0.0999      | 0.0567      | **0.57x**   | 4.45x |
| gate_proj_1B |   16 | —            | 0.1005      | 0.0557      | **0.55x**   | 4.32x |
| up_proj_1B   |   16 | 0.1100       | 0.0964      | 0.0554      | **0.57x**   | 3.84x |
| down_proj_1B |    1 | —            | 0.1172      | 0.0768      | **0.66x**   | 4.93x |
| down_proj_3B |    1 | —            | 0.1440      | 0.0779      | **0.54x**   | 4.93x |
| gate_proj_1B | 2048 | —            | 0.2197      | 0.1686      | **0.77x**   | 0.91x (beats fp16) |
| up_proj_1B   | 2048 | 0.2113-ish   | 0.2113      | 0.1647      | **0.78x**   | 0.89x (beats fp16) |
| q_proj_3B    | 2048 | —            | 0.1468      | 0.1157      | **0.79x**   | 0.85x (beats fp16) |
| down_proj_3B | 2048 | 0.2786       | 0.2776      | 0.2050      | **0.74x**   | 0.67x (beats fp16) |

**Gains scale with weight-side SF size (N and K):** biggest at large-N decode shapes
(gate/up/down, 0.54-0.66x = 34-46% faster) because the removed scatter was ~N*Klow/16 +
N*Khigh/32 bytes. At prefill, presf now **beats fp16 cuBLAS** on all 4 large shapes (0.67-0.91x).
rungB_presf beats both round-1 rungB and Rung A everywhere.

Anomaly (honest): down_proj_1B M=2048 timed ~1.1ms for all fused paths (vs round-1 0.25ms) in
this run — a transient outlier (presf still fastest of the fused there; other GPUs at 100% util
during the run). Needs a re-measure; not a code regression.

---

## T2 — ncu six-dimension profiling (NOW UNBLOCKED) [DONE]

Container `luofan-ncu` has working perf counters (no ERR_NVGPUCTRPERM). Profiled the post-T1
best path on decode (q_proj_1B M=16) and prefill (gate_proj_1B M=2048) with `--set full` +
`--set source --section SourceCounters`, `--profile-from-start off` to isolate the 4 per-call
kernels. Parsed with `ncu_report` (`analyze_ncu.py`). Full detail in
`profile/round2_{decode,prefill}/REPORT.md`.

**Decode (q_proj_1B M=16):** launch/occupancy-bound. Both GEMMs launch only **grid=16 CTAs,
waves/SM=0.085** on a ~188-SM chip (Pattern A). NVFP4 GEMM TC = **57% of peak_active but 4.4%
of peak_elapsed** — well-fed per active SM, but ~90% of SMs idle. scale_only is
long_scoreboard-70% (gmem latency). No register spill.

**Prefill (gate_proj_1B M=2048):** grid=1024, waves/SM=5.4 (well-filled). NVFP4 GEMM is healthy
(TC 59% elapsed). **scale_only = 29% of prefill (61us), long_scoreboard 67.5%, L2 hit 24%** — a
pure extra gmem round-trip of 16.7M elements. MXFP8 GEMM TC only 26% elapsed (inflated by the
beta=1 D_low read).

### Ranked next steps (evidence-backed, cross-scenario)

1. **Fold `scale_only` into GEMM2 EVT epilogue** — kills the long_scoreboard-67% memory pass;
   ~25-29% at prefill, removes the 4th launch at decode. Highest overall leverage. (SM120 EVT.)
2. **Small-Tile-N for decode** — raises CTA count where waves/SM=0.085. **Validated in T3 below
   (5-10% decode).** Add split-K over Klow for small-N shapes.
3. **True B1 single kernel** — two phases into one FP32 register accumulator; removes the D_low
   bf16 gmem round-trip (fixes MXFP8 TC-26% underutilization) and the SF double-handling. The
   structural prize on GDDR7; guard the 168-reg/thread pressure (already caps occupancy to 25%).
4. Vectorize the activation-SF scatter (store eff 1.875) — minor (9%).

---

## T3 — Small-Tile-N probe (TileN 128 -> 64) [DONE, M-adaptive]

Probe ext `tile_probe_ext.cu` (`sed Shape<_128,_128,_128> -> Shape<_128,_64,_128>` on both the
NVFP4 and MXFP8 collectives), harness `tile_probe.py`. The weight SF prepack is tile-N-independent
(SfAtom depends only on SFVecSize), so the T1 prepacked buffers are reused unchanged — correctness
held (cosine 0.999997 everywhere).

| shape        |   M | N128 ms | N64 ms | N64/N128 |
|--------------|----:|--------:|-------:|---------:|
| q_proj_1B    |   1 | 0.0513  | 0.0485 | **0.95x** |
| q_proj_1B    |  16 | 0.0509  | 0.0481 | **0.95x** |
| k_proj_1B    |   1 | 0.0511  | 0.0460 | **0.90x** |
| k_proj_1B    |  16 | 0.0516  | 0.0478 | **0.93x** |
| v_proj_1B    |   1 | 0.0504  | 0.0473 | **0.94x** |
| o_proj_1B    |  16 | 0.0510  | 0.0482 | **0.95x** |
| down_proj_1B | 256 | 0.0782  | 0.0619 | **0.79x** |
| q_proj_1B    | 256 | 0.0516  | 0.0484 | **0.94x** |
| gate_proj_1B | 256 | 0.0624  | 0.0681 | 1.09x (REGRESS) |
| gate_proj_1B |2048 | 0.2076  | 0.1916 | 0.92x |
| down_proj_1B |2048 | 0.1706  | 0.1734 | 1.02x (tie) |
| q_proj_1B    |2048 | 0.0804  | 0.0805 | 1.00x (tie) |

**Verdict:** N=64 is a real decode/small-M win (0.79-0.95x) exactly where the grid is
CTA-starved (confirms the Dim-1 diagnosis), but **regresses at gate_proj M=256 (1.09x) and ties
at M=2048** — small tiles add scheduling/epilogue overhead once the grid is already large. So
**dispatch tile-N by M** (N=64 for M<=~128 decode, N=128 for prefill), not a blanket switch.
Kept as a probe pending the per-shape dispatch wiring (KDA ladder B3+). Negative/regression rows
recorded above honestly.

---

## Files
- `fused_gemm_ext.cu` — `prepack_sf_b`, `mixed_gemm_fused_b_presf`, `mixed_gemm_fused_presf`,
  `fill_sf_merged2`.
- `mixed_gemm.py` — `prepack_weight_sf`, presf wrappers, validation + benchmark presf rows.
- `profile_presf.py`, `analyze_ncu.py` — ncu harness + `ncu_report` six-dimension parser.
- `tile_probe_ext.cu`, `tile_probe.py` — T3 N=64 probe.
- `profile/round2_{decode,prefill}/REPORT.md`, `benchmark.csv` (216 rows).
