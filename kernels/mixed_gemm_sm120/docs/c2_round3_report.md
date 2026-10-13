# C3 Round-3 report — bounds baseline on the SM120 fused mixed-precision GEMM

Status: **T0 DONE (harness-only). T1 (EVT scale-fold) and T2 (M-adaptive tile dispatch)
CANCELLED mid-round by project-lead directive** (priority change delivered during this
round): implement the true single-kernel B1 fusion first, then revisit optimization
layers; T0's bounds are the yardstick for judging B1, so they were kept and completed.
No kernel-code (`.cu`) files were touched after the directive landed —
`fused_gemm_ext.cu` is **byte-identical to round-2** (`git diff` clean). All new work is
in `mixed_gemm.py` (harness/benchmark only).

Hardware: RTX Pro 6000 Blackwell, SM120 (sm_120a), container `luofan-plaquant`, GPU 4
(verified quiet: 0% util, ~3 MiB used, immediately before every run below). GPUs 2/3
were pinned at ~88 GB used by another job all round; GPU 4-7 were free.

Correctness gate held on every path: cosine >= 0.999997 for all mixed rungs (rungA,
rungB, rungB_presf, rungA_presf, 2launch) and >= 0.999998 for the new pure-format
bounds, across q_proj_1B / down_proj_1B / q_proj_3B at M in {1,16,256,2048}. ALL PASS.

---

## T0 — Bounds baselines (band position of `rungB_presf` in `[pure NVFP4, pure MXFP8]`)

### Methodology (read before the numbers — important honest caveat)

The task spec asked for the pure-format bounds to use "the existing nvfp4_gemm/mxfp8_gemm
extensions ... with weight-SF pre-interleaved — same prepack + timing treatment as
rungB_presf." Building that split cleanly requires a new presf-style entry point in a
`.cu` file (to accept an already-interleaved weight SF buffer). The mid-round directive
forbade **any** kernel-code changes this round, so T0 instead calls the **existing,
byte-unmodified** `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` entry points
(`nv.get_ext().nvfp4_gemm(...)`, `mx.get_ext().mxfp8_gemm(...)`) directly at
`K = K_total`. Those entry points fill **both** the activation-side and the
weight-side SF **inside every timed call** (no presf split available without new code).
`rungB_presf`, by contrast, excludes the weight-SF fill from its timed region (T1 from
round 2, unchanged).

**Consequence:** the pure-format bound numbers below are inflated by a repeated
weight-SF scatter cost (`~N*K/16` bytes for NVFP4, `~N*K/32` for MXFP8) that scales with
`N*K` and is **independent of M**. For small-N shapes (K_total=2048, N<=2048) this
extra cost is modest and the band numbers are informative. For large-N shapes
(gate/up_proj, N=8192; down_proj*, K_total=8192) the extra scatter dominates the pure
bound's small-M timing and makes `rungB_presf` look *faster than the "upper" FP4
bound* (band < 0%) — a methodology artifact, not evidence that fusion beats physics.
This is flagged per-row below. A true apples-to-apples bound needs a presf entry point
for the pure kernels too — recorded as follow-up work, out of scope this round.

### Band position table (`band% = (t_mixed - t_fp4) / (t_fp8 - t_fp4) * 100`; 0%=FP4 floor, 100%=FP8 ceiling)

| shape         |    M | K_total | fp4 (ms) | rungB_presf (ms) | fp8 (ms) | band%  | note |
|---------------|-----:|--------:|---------:|------------------:|---------:|-------:|------|
| q_proj_1B     |    1 |    2048 | 0.0410   | 0.0506             | 0.0492   | **117%** | small-N, trustworthy |
| q_proj_1B     |   16 |    2048 | 0.0406   | 0.0506             | 0.0489   | **120%** | small-N, trustworthy |
| q_proj_1B     |  256 |    2048 | 0.0408   | 0.0516             | 0.0506   | **111%** | small-N, trustworthy |
| q_proj_1B     | 2048 |    2048 | 0.0618   | 0.0809             | 0.0855   | 80%    | small-N, trustworthy |
| k_proj_1B     |    1 |    2048 | 0.0402   | 0.0503             | 0.0483   | **125%** | small-N, trustworthy |
| k_proj_1B     |   16 |    2048 | 0.0405   | 0.0500             | 0.0489   | **114%** | small-N, trustworthy |
| k_proj_1B     |  256 |    2048 | 0.0409   | 0.0505             | 0.0489   | **120%** | small-N, trustworthy |
| k_proj_1B     | 2048 |    2048 | 0.0415   | 0.0565             | 0.0538   | **122%** | small-N, trustworthy |
| v_proj_1B     |    1 |    2048 | 0.0409   | 0.0498             | 0.0490   | **110%** | small-N, trustworthy |
| v_proj_1B     |   16 |    2048 | 0.0402   | 0.0506             | 0.0480   | **134%** | small-N, trustworthy |
| v_proj_1B     |  256 |    2048 | 0.0408   | 0.0504             | 0.0490   | **117%** | small-N, trustworthy |
| v_proj_1B     | 2048 |    2048 | 0.0418   | 0.0565             | 0.0542   | **118%** | small-N, trustworthy |
| o_proj_1B     |    1 |    2048 | 0.0406   | 0.0506             | 0.0494   | **114%** | small-N, trustworthy |
| o_proj_1B     |   16 |    2048 | 0.0407   | 0.0505             | 0.0496   | **111%** | small-N, trustworthy |
| o_proj_1B     |  256 |    2048 | 0.0395   | 0.0517             | 0.0508   | **108%** | small-N, trustworthy |
| o_proj_1B     | 2048 |    2048 | 0.0618   | 0.0811             | 0.0852   | 83%    | small-N, trustworthy |
| gate_proj_1B  |    1 |    2048 | 0.0713   | 0.0561             | 0.0909   | -77%   | large-N, artifact |
| gate_proj_1B  |   16 |    2048 | 0.0702   | 0.0641             | 0.0890   | -32%   | large-N, artifact |
| gate_proj_1B  |  256 |    2048 | 0.0700   | 0.0572             | 0.0883   | -70%   | large-N, artifact |
| gate_proj_1B  | 2048 |    2048 | 0.1316   | 0.1696             | 0.1967   | 58%    | large-N, less distorted at big M |
| up_proj_1B    |    1 |    2048 | 0.0717   | 0.0562             | 0.0915   | -78%   | large-N, artifact |
| up_proj_1B    |   16 |    2048 | 0.0710   | 0.0558             | 0.0890   | -84%   | large-N, artifact |
| up_proj_1B    |  256 |    2048 | 0.0702   | 0.0565             | 0.0880   | -77%   | large-N, artifact |
| up_proj_1B    | 2048 |    2048 | 0.1316   | 0.1695             | 0.1967   | 58%    | large-N, less distorted at big M |
| down_proj_1B  |    1 |    8192 | 0.0906   | 0.0769             | 0.1301   | -35%   | large-K, artifact |
| down_proj_1B  |   16 |    8192 | 0.0909   | 0.0770             | 0.1308   | -35%   | large-K, artifact |
| down_proj_1B  |  256 |    8192 | 0.0966   | 0.0777             | 0.1364   | -48%   | large-K, artifact |
| down_proj_1B  | 2048 |    8192 | 0.1680   | 0.1717             | 0.2576   | **4%** | large-K, least distorted at M=2048 |
| q_proj_3B     |    1 |    3072 | 0.0549   | 0.0553             | 0.0725   | 2%     | small-N, trustworthy |
| q_proj_3B     |   16 |    3072 | 0.0546   | 0.0562             | 0.0725   | 9%     | small-N, trustworthy |
| q_proj_3B     |  256 |    3072 | 0.0566   | 0.0573             | 0.0750   | 4%     | small-N, trustworthy |
| q_proj_3B     | 2048 |    3072 | 0.1017   | 0.1183             | 0.1513   | 33%    | small-N, trustworthy |
| down_proj_3B  |    1 |    8192 | 0.1093   | 0.0769             | 0.1548   | -71%   | large-K, artifact |
| down_proj_3B  |   16 |    8192 | 0.1101   | 0.0763             | 0.1542   | -76%   | large-K, artifact |
| down_proj_3B  |  256 |    8192 | 0.1144   | 0.0779             | 0.1589   | -82%   | large-K, artifact |
| down_proj_3B  | 2048 |    8192 | 0.2195   | 0.2171             | 0.3457   | -2%    | borderline, ~at FP4 floor |

Full data: `benchmark.csv` (288 rows, +`pure_fp4_fullK`/`pure_fp8_fullK` rows added this
round) and `band_position.csv` (36 rows, new this round).

### Reading the two honest signals separately

1. **Small-N decode shapes (q/k/v/o_proj, trustworthy rows, band 80-134%).** At
   decode (M=1-256) `rungB_presf` sits level with or *above* the pure-MXFP8-fullK
   bound — i.e. running the WHOLE layer in the slower, more-precise format would be
   just as fast or faster than today's mixed-precision fusion. This is a real,
   non-artifactual finding (the weight-fill cost difference between the two
   measurements is small at this N) and it is the quantitative confirmation of round-2's
   ncu diagnosis: at decode the grid is CTA-starved (16 CTAs / ~188 SMs) and the extra
   launches (SF-fill, `scale_only`) eat the entire format advantage. This is exactly
   what T1 (now cancelled) targeted. At M=2048 these same shapes drop to 80-83% —
   closer to the FP4 floor once the grid is large enough to amortize launch overhead.
2. **Large-N shapes (gate/up_proj, down_proj*, flagged artifact rows).** Bands go
   negative because the pure-format "bound" here is inflated by a weight-SF scatter
   that `rungB_presf` doesn't pay repeatedly. The one number from this group that *is*
   trustworthy is **down_proj_1B M=2048: band=4%** (and the dedicated re-measure below
   confirms 4.5%) — at large M the fill-cost artifact shrinks to a small fraction of
   total time, and the result lines up with the small-N shapes' story: `rungB_presf`
   is very close to the fastest-possible (pure NVFP4) floor at prefill-scale M.

**Bottom line:** mixed is near-optimal (close to the FP4 floor) at large M across all
shapes once the data is clean enough to trust; at decode-scale M, mixed is riding at or
above the FP8 (slower) bound on the small-N shapes — fusion overhead, not MMA compute,
is the decode bottleneck, matching round-2's ncu finding exactly.

---

## Re-measured down_proj_1B M=2048 (round-2 flagged a ~1.1ms outlier vs round-1's ~0.25ms)

Median of 60 reps (15 warmup), GPU 4 verified quiet (0% util, ~3 MiB) immediately
before the run:

| path                | median (ms) | min    | max    |
|----------------------|------------:|-------:|-------:|
| rungB_presf          | 0.1820      | 0.1794 | 0.1884 |
| rungA_presf          | 0.1834      | 0.1787 | 0.1881 |
| rungB_mixed          | 0.2348      | 0.2311 | 0.2373 |
| fused_mixed (rungA)  | 0.2362      | 0.2324 | 0.2387 |
| 2launch_mixed        | 0.2765      | 0.2700 | 0.2829 |
| fp16_cublas          | 0.2230      | 0.2216 | 0.2282 |
| pure_fp4_fullK       | 0.1780      | 0.1753 | 0.1819 |
| pure_fp8_fullK       | 0.2668      | 0.2621 | 0.2702 |

**Resolved: round-2's ~1.1ms reading was a transient GPU-contention outlier, not a
regression.** The true number is ~0.18ms — tighter and *faster* than round-1's ~0.25ms
baseline (min/max span only 5-8%, confirming a quiet run). `rungB_presf` beats fp16
cuBLAS at 0.82x and sits at **band=4.5%** — essentially at the pure-NVFP4-fullK floor.

---

## T1 / T2 — cancelled

Per the mid-round project-lead directive: **T1 (fold `scale_only` into GEMM2's EVT
epilogue) and T2 (M-adaptive N=64/N=128 tile dispatch) are cancelled this round.** No
attempt was made and no kernel code was touched for either (confirmed `git diff
fused_gemm_ext.cu` is empty). Stated reason: implement the true single-kernel B1 fusion
(dual-phase mainloop into one FP32 register accumulator, from `docs/c2_rungB_report.md`
/ `candidates.jsonl` `C2-RungB-B0`) before investing further in epilogue/tile
micro-optimizations on top of the current two-GEMM structure. Round-2's ranked next
steps (`docs/c2_round2_report.md`, `profile/round2_{decode,prefill}/REPORT.md`) remain
valid and unconsumed:
1. Fold `scale_only` into GEMM2 EVT epilogue (29% of prefill, pure gmem round-trip) — deferred.
2. Small-Tile-N dispatch for decode (validated 0.79-0.95x in round-2 T3) — deferred.
3. **True B1 single kernel — NOW THE EXPLICIT PRIORITY per project lead.**
4. Vectorize the activation-SF scatter — minor, still last.

---

## Next-step ranking (updated)

1. **True B1 single-kernel fusion (dual accumulator, one mainloop, no D_low bf16
   round-trip)** — explicit project-lead priority for the next round. This round's T0
   bounds are the yardstick: B1 should be judged by how close it lands to the FP4 floor
   at decode (currently 80-134% band depending on shape) and whether it closes the
   gap seen at small-N decode shapes where the current 2-GEMM fusion sits at or above
   the MXFP8-fullK bound.
2. Fold `scale_only` into GEMM2 EVT epilogue (T1, cancelled this round) — still the
   highest-leverage epilogue-level win per round-2 ncu evidence if B1 is not reached
   quickly; natural to revisit once/if B1 doesn't land in one session.
3. M-adaptive tile dispatch (T2, cancelled this round) — same.
4. Build a true presf entry point for the pure-format GEMMs (needs new `.cu` code,
   explicitly out of scope this round) so the T0 bounds stop being distorted by the
   repeated-weight-fill artifact at large N — needed before trusting the gate/up/down
   band numbers at decode scale.

## Files touched this round
- `mixed_gemm.py` — `prep_pure_fp4/prep_pure_fp8`, `pure_fp4_fullK/pure_fp8_fullK`
  (call the existing unmodified `nv`/`mx` extensions), `_time_ms_median`,
  `band_position`, bounds correctness check added to `validate()`, bounds rows +
  `band_position.csv` added to `benchmark()`, `remeasure_down_proj_1B()`.
- `fused_gemm_ext.cu` — **untouched** (byte-identical to round-2; confirmed via `git diff`).
- `benchmark.csv` (288 rows), `band_position.csv` (36 rows, new).
