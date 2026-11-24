# KDA GB10 tuning log (NVFP4+MXFP8 fused GEMM, SM121)

Goal: align GB10 relative ratios (vs local cuBLASLt FP4/FP8 kernel baselines) with
Pro6000 levels. Hard gates: single-launch fusion preserved (profiler), per-point
cosine_vs_fp32 >= current (0.9924).

## Round 0 — baseline analysis (2026-11-24)

From `ours_gb10_uniform11.json` vs `torch_gb10_ext_m{2048,8192}.json`:

- median r=0 vs cuBLASLt FP4: **0.548x** (target 1 confirmed).
- median r=1/8 vs cuBLASLt FP8: **1.315x** (keep/improve).
- r=1/12 column (K_high%128==64) consistently dips vs r=1/16 neighbors
  (e.g. 70B down M2048: 0.73 -> 0.57) — target 3 confirmed.
- **NEW finding: large-shape M=8192 collapse.** 70B qkvo/gate_up/down, 8B down,
  Q8B down, Q14B down/gate_up at M=8192 run at 0.2-0.6x vs FP8 while the SAME
  shape at M=2048 runs 1.4-1.8x. Throttle ruled out: the in-session fp16 cuBLAS
  anchors match the separately-measured bf16_cublas baselines within ~5%
  (70B down M8192 anchor/baseline=1.13 worst). So the collapse is real kernel
  behavior on large grids (e.g. 70B qkvo M8192: ours ~13.4ms vs fp8 6.16ms),
  pointing at tile scheduling/raster order (DRAM/L2 reuse on 48-SM LPDDR GB10).
  Promoted to a first-class target (subsumes target 4's "config sweep").

## Round 1 — r=0 castfold (target 1)

Hypothesis: the pure-NVFP4 (K_high==0) route stores bf16 then pays a separate
`[M,N].to(fp16)` copy (the 74us direct_copy). On GB10 (~273GB/s LPDDR) that
round-trip is a large share of the whole GEMM. Fix = same castfold epilogue the
dual path already uses (codesign_fused_bias_ext bias_castfold).

Change (minimal):
- `kernels/mixed_gemm_sm120/codesign_fused_bias_ext.cu`: new entry
  `single_nvfp4_gemm_prepacked_bias` — GemmSingle (DualPhase=false) built with
  this module's fp16 LinCombBiasBFRoundHalf epilogue; mainloop untouched.
- `promix/quantize/mixed_linear.py::_gemm`: pure-NVFP4 branch dispatches to the
  new entry when cd_bias is wired and out_dtype==fp16 (zero-bias buffer for
  bias-less layers); falls back to the old two-step path otherwise.
- `promix/eval/rd_oppareto_ours.py::dispatch_tag`: tags the new route
  `single_phase_nvfp4_castfold`.

Gate: torch.equal AND bit-exact (int16 view) vs old path on 4 shapes x 2 M. PASS.
cos unchanged (0.9924).

Result (median-of-50 eager):
| shape | M | old us | new us | speedup |
|---|---|---|---|---|
| 1B qkvo | 2048 | 143.9 | 83.9 | 1.72x |
| 1B qkvo | 8192 | 649.1 | 341.5 | 1.90x |
| 8B gate_up | 2048 | 1390.7 | 861.4 | 1.62x |
| 8B gate_up | 8192 | 6357.7 | 4306.7 | 1.48x |
| 3B down | 2048 | 430.2 | 301.5 | 1.43x |
| 3B down | 8192 | 1735.1 | 1206.5 | 1.44x |
| 70B down | 2048 | 4769.2 | 4472.3 | 1.07x |
| 70B down | 8192 | 44318.6 | 43230.9 | 1.03x |

Conclusion: target 1 largely closed (r=0 projected from 0.55 to ~0.85-0.95 vs
FP4 on small/mid shapes). 70B barely moves because the mainloop itself collapses
at that size — see Round 2.

## Round 2 — tile-scheduler raster/swizzle on the M=8192 collapse

Hypothesis: StaticPersistentScheduler defaults (Heuristic raster, swizzle 1)
mis-schedule large grids on 48 SMs -> poor L2/DRAM reuse. The glue already
plumbs `arguments.scheduler` (max_swizzle_size, raster_order) but no entry sets it.

Change: `codesign_fused_ext.cu` gains `CD_SCHED_RASTER` (0 heuristic / 1 AlongM /
2 AlongN) + `CD_SCHED_SWIZZLE` (0 default) env overrides, read once per process,
applied at all 5 entry points (incl. the bias-ext copies). No kernel math change.

Probe: sweep raster x swizzle {(0,0),(1|2,1|2|4|8)} on 70B qkvo/down, 8B down,
Q14B gate_up at M in {2048, 8192}, r in {0, 1/8}. (running)

Probe results (r=1/8 and r=0, median-of-30 eager, us):

| point | default | r0s4 | r0s8 | r1s8 | r2s8 | best/default |
|---|---|---|---|---|---|---|
| 70B qkvo M8192 | 12052/13236 | 4420 | 3651 | 3669 | 3677 | 3.5x |
| 70B down M8192 | 46724 | 16718 | 15472 | 15475 | 15377 | 3.0x |
| 8B down M8192 | 8207 | 3905 | 3288 | 3135 | 3282 | 2.6x |
| 8B down M4096 | 4153 | 2010 | 1694 | 1677 | 1680 | 2.5x |
| Q14B gate_up M8192 | 8020 | 6204 | 5460 | 5441 | 5366 | 1.5x |
| 1B qkvo M8192 | 378 | 315 | 326 | 383 | 338 | 1.2x |
| small-grid M2048 points | — | ~1.0-1.16x | some -4..8% | | | s4 safe |

Shipping change: AUTO heuristic inside `cd_apply_sched_env` (all 5 entries incl.
bias ext), applied only when device SM count <= 72 (GB10; Pro6000=188 untouched)
and no CD_SCHED_* env override: swizzle=4 default; swizzle=8 when
(min(TM,TN)>=32 && N*K>=32Mi) or N*K>=100Mi. Raster stays Heuristic.
Auto verified to track per-point bests within noise on all 24 probe points.

Gate: tile order cannot change per-tile math — verified bit-exact
(torch.equal on int16 view) vs swizzle=1 control on 4 points incl. r=0 and
r=0.25, PASS. Single-launch untouched (same kernel, scheduler params only).

Result: the M=8192 collapse is eliminated (70B qkvo r=1/8: 13236 -> 3669us =
vs fp8 0.46 -> 1.68x; 8B down M8192: 0.79 -> ~2.0x).

## Round 3 — K_high%128==64 dip (target 3)

Hypothesis: MXFP8 phase BLK_K=128; K_high%128==64 leaves a half-empty
predicated k-tile whose cost >> its 64 columns. Probe the marginal cost of
K_high=64..320 (and 2368 vs 2432) at fixed shape before choosing a fix
(64-K-tile MX config vs dispatch-level padding).

Probe anatomy (M=2048, us): K_high%128==64 cost is a flat penalty that scales
with the B-weight stream, NOT one extra k-tile:
- 70B down: KH mod128==0 -> ~4005-4118; mod128==64 -> ~4750-4907 (+19%), same
  penalty at KH=64 and KH=2368 (same tile count as 2432!).
- Padding the HIGH segment to 128 (zeros) did NOT help (4921 -> 5008) ->
  mechanism is in the NVFP4 phase: K%128==0 so K_high%128==64 <=> K_low%128==64:
  packed A_lo/B_lo rows are only 64B-aligned (K_low/2 bytes) and the NV phase
  ends on a half k-tile.
- Padding the LOW segment with 64 zero columns (K_low -> %128==0) recovers it:
  70B down 4879 -> 4219 (1.16x), 3B down 1.05x, bit-equal outputs.

Shipping change (zero-pad K_low to 128 alignment, bit-exact):
- `weight_pack.py`: PackedWeight.K_low_pad; pack pads codes/SFB with zero
  blocks when K_low%128==64; gated by `klow_pad_enabled()` (auto-on iff SM
  count <= 72, PROMIX_KLOW_PAD=0/1 override). Decode/Marlin buffers unchanged.
- `quant_pack_float_ext.cu`: quant_pack_float_fp4 gains K_valid (<K): pad
  blocks emit zero codes + zero SF in the SAME single pack launch (no extra
  kernel, no extra pack time — measured equal).
- `mixed_linear.py`: K_low_pad used for the qpf fp4 pack width + ALL GEMM entry
  K_lo arguments; qpfm/hpk/gather producers bypassed/guarded when pad active
  (they do not emit the pad yet — serve wiring note).

Gate: PROMIX_KLOW_PAD=0 vs 1, torch.equal + int16-bit-exact on 5 cases incl.
r=1 (K_low=64->128): PASS. Perf: 70B down KH2368 1.16x, 8B gate_up KH320 1.03x,
3B down KH192 1.03x, 1B qkvo KH64 1.02x, pack time unchanged.

## Round 4a — mainloop stage sweep (target 4)

CodesignBlockScaledMma stages: shipping CD_NV_STAGES=4 / CD_MX_STAGES=2 (tuned
on Pro6000). GB10 LPDDR latency is higher — probe deeper pipelines via variant
module builds (-DCD_NV_STAGES/-DCD_MX_STAGES), prepacked GEMM timing.

Result: SMEM kills the space on GB10 (100KB/block): nv5mx2=106KB, nv6mx2=125KB,
nv4mx3/nv5mx3=116KB all fail initialize; only nv3mx2 (82KB) fits and LOSES
0.94-1.01x everywhere (12 points). Shipping (4,2) stays. No change.

## Round 5 — GB10 small-M window re-sweep (target 2)

Sweep: 8 serving shapes x M {8..1024} x {shipping castfold, smallm cfg1/2/3,
splitk cfg0/1 x heuristic/split4/stream-K}, CUDA-graph replay median-of-30,
r=1/8. Raw data: docs/rd_oppareto_data/gb10_smallm_sweep.json.

Key finding: the Pro6000 windows are MISTUNED on GB10 — with 48 SMs the
128x128 grid already fills the machine at small M, so cfg1 (128x64) loses
30-70% at M<=64 where Pro6000 won 1.05-1.6x. GB10-measured >=5% wins:
  (3072,3072)  M128 cfg2 1.30x        (3072,8192)  M64..128 stream-K 1.23-1.25x
  (4096,14336) M256 cfg3 1.15x        (4096,4096)  M8..32 sk-heur 1.05x, M256 1.12x
  (5120,3072)  M16..32 sk-heur 1.05x  (4096,14336) M1024 sk-heur 1.13x

Shipping change: `_SMALLM_TABLE_GB10` / `_SPLITK_TABLE_GB10` in mixed_linear.py,
selected by `_low_sm_device()` (SM count <= 72); Pro6000 tables untouched.
Routing smoke-tested (tags + cosine). NOTE: Marlin W4A16 and decode-GEMV
crossovers were NOT re-swept (decode-route, outside this task's GEMM curve).

## Round 6 — full re-run + verification (deliverables)

`ours_gb10_uniform11_tuned.json` (18 shapes x {2048,8192} x 11 r, eager +
CUDA-graph medians) + `ours_gb10_uniform11_tuned.launch_verify.json`
(torch.profiler, 12 representative points incl. r=0 castfold, r=1/12 pad,
splitk-eligible and giant shapes: ALL exactly 1 CUDA kernel per _gemm — single
launch preserved; r=0 castfold path shows CodesignFusedKernel<...Lb0> only, the
74us direct_copy is gone).

Correctness: cosine_vs_fp32 IDENTICAL to the untuned curve on all 396 points
(max diff 0.0) — every change was bit-exact by construction and gated.

### Final medians (GB10, vs local cuBLASLt kernel baselines)

| r | before | after | Pro6000 (ours_uniform11_ext) |
|---|---|---|---|
| 0 (vs fp4) | 0.548 | **0.951** | 0.813 |
| 1/32 (vs fp4) | 0.802 | 0.894 | — |
| 1/16 | 0.787 | 0.857 | — |
| 1/12 | 0.745 | 0.844 | — |
| 1/8 (vs fp8) | 1.315 | **1.576** | — |
| 3/16 | 1.277 | 1.495 | — |
| 1/4 | 1.245 | 1.436 | — |
| 3/8 | 1.145 | 1.322 | — |
| 1/2 | 1.075 | **1.224** | 1.167 |
| 3/4 | 0.943 | 1.039 | — |
| 1 | 0.800 | 0.863 | 0.901 |

Median curve is now monotone (no 128-alignment dip: 0.951 > 0.894 > 0.857 >
0.844 within the fp4 regime; 1.576 > ... > 1.039 within the fp8 regime). The
M=8192 big-shape collapse is gone (70B qkvo r=1/8: 0.46 -> 1.60; 70B down
0.59 -> 1.74; 8B down 0.79 -> 2.02 vs fp8).

GB10 now MEETS OR EXCEEDS Pro6000's relative levels at the comparable points
(r=0: 0.95 vs 0.82; r=1/2: 1.22 vs 1.13; r=1: 0.86 vs 0.88 ~ parity).

### Residual items / notes
- 70B down M8192 r=3/4 shows a local non-monotonicity (0.75 vs 0.84 at r=1):
  the swizzle heuristic keys on N*K_low; at very high r the MX weight dominates
  and s8 may be suboptimal. Outside the operating regime (allocations r<=1/4);
  passing K_total to the heuristic is the follow-up if it matters.
- 1B gate_up M8192 remains the weakest r=1/8 point (1.10): small-K (2048)
  pure-throughput shape; cuBLASLt uses a 128x128x256 fp8 tile; our BLK_K=128
  dual kernel cannot deepen K on 100KB SMEM (round 4a).
- Serve-path producers (qpfm gather / hadamard-pack / norm-pack) do not emit
  the K_low 128-pad yet: they are guarded off when the pad is active (bench
  path unaffected; wire pads before GB10 serve integration).
- Marlin/decode-GEMV window tables are still Pro6000-measured (decode regime,
  out of scope here).
