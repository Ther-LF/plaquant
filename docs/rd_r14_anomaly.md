# rd-r14: the 3B r=1/4 decode anomaly — root cause and corrected point

> Resolves docs/vllm_final_fiveway.md §4 finding 4 (the campaign's one open
> item): the 3B r=1/4 sweep point showed decode TPOT b256 22.88 ms vs
> 18.1-19.1 ms for every r <= 3/16 — the only swept point whose decode fell
> below BF16. 2026-11-03, branch feat/rd-r14probe, dir docs/rd_r14_{data,scripts}.

## VERDICT

**(a) Measurement artifact — GPU-6 cross-tenant interference. NOT a
route/window bug, NOT a kernel regime change.** The anomaly does not
reproduce on a clean GPU: two independent same-GPU-anchored sessions put the
r=1/4 decode point within **0.2-1.8 %** of r=1/8. The corrected r-sweep row
(now in the fiveway doc):

| r | bits/wt | PPL | decode TPOT ms b128/b256 | gen tok/s b256 | QServe c256 | prefill L2048 b128 |
|---|---|---|---|---|---|---|
| 0.25 (was) | 5.44 | 8.283 | 14.12/22.88 | 10942 | 5623 | 3833 ms |
| **0.25 (corrected)** | 5.44 | 8.283 | **11.45/18.42** | **13617** | **6865** | **3305 ms** |

Curve consequence: the serving cost of the accuracy knob is near-flat over
the ENTIRE swept range r=1/32..1/4 (3B decode b256 18.1-19.1 ms), not just
r <= 3/16. The r=1/4 point no longer falls below BF16 anywhere — it beats
BF16 decode b256 by 1.07x like its neighbors.

## 1. Why the campaign number was wrong (evidence trail)

The published r-sweep table only surfaced the 128:1024 decode cells. The raw
session data (`docs/vllm_final_data/contaminated_gpu6_r0.25/`, 2026-11-01
10:49-11:00, GPU 6) shows a time-correlated — not shape-correlated — failure:

1. **Within one engine session, early cells clean, later cells wrecked**:
   1024:128 TPOT b16/b64/b128 = 5.89/10.51/16.15 ms (equal to or FASTER than
   the r=1/8 control) but b256 = **52.08 ms (+78 %)** — a number that never
   made the table; then every subsequent 128:1024 cell elevated +25 %, small
   prefill TTFTs +39..107 %, prefill L2048 +21 %, and the chained qserve
   engine -19 %. A per-kernel regime change cannot be +78 % in one case and
   +26 % in the other at the SAME decode M=256 while M=128 decode of the
   first case is simultaneously clean.
2. **Run-to-run spread inside the anomalous cells is up to 42 %** (128:1024
   b128 full runs 17.08/14.75/11.99 s) **with the fastest run matching the
   r=1/8 control** (11.99 vs 11.74 s). Every other sweep point's runs are
   steady to 0.4 %. Bursty external competitor signature.
3. **The engine mis-attributed ~12 GiB of external memory to itself**: the
   GPU-6 session reported "18.45 GiB consumed (weights + non-torch)" vs
   4.67-6.6 GiB for every other sweep point (checkpoint on disk 3.3G vs
   3.2G). On a clean GPU the same checkpoint loads at **6.2 GiB** and gets
   the full 77.9 GiB KV budget. vLLM's consumed-memory accounting absorbs
   any memory another process allocates during engine init — direct evidence
   a co-tenant was active on GPU 6. (No preemption was involved: preempt
   count 0; max live tokens 294,912 << even the shrunken 617,872 budget.)
4. **Lane history**: the r=1/4 3B point was the ONLY sweep point on GPU 6,
   launched at 10:49 as a `pm3b_r0.25_fix` re-run immediately after the 8B
   r=0.25 chain freed that GPU (10:48:29); the box is shared with other
   tenants and GPU 6 demonstrably had external activity through that window.

## 2. Clean re-measure (the correction)

`docs/rd_r14_scripts/run_session.sh` — one session = three engines
sequentially on ONE idle GPU, identical bench_serve protocol/stack as the
campaign (same venv, same checkpoints, all R1-R5 levers default-ON), with an
nvidia-smi monitor running throughout:

- Session A: GPU 4, order BF16 -> r=1/8 -> r=1/4
- Session B: GPU 5, order BF16 -> r=1/4 -> r=1/8 (order control)

Anchors prove lane comparability: BF16 matches the campaign table within
+-3 % on every cell (e.g. decode b256 19.69/19.63 vs 19.70); the r=1/8
control matches the campaign's GPU-7 numbers within 1.2 % (18.24/18.19 vs
18.14). Sessions agree with each other to <= 0.7 % everywhere.

| cell | campaign r=1/4 (gpu6) | sA (gpu4) | sB (gpu5) | r=1/8 ctl (sA/sB) |
|---|---|---|---|---|
| 128:1024 TPOT b128 | 14.12 | 11.45 | 11.53 | 11.43 / 11.33 |
| 128:1024 TPOT b256 | 22.88 | 18.42 | 18.49 | 18.24 / 18.19 |
| 1024:128 TPOT b256 | 52.08 | 29.27 | 29.27 | 29.13 / 29.03 |
| qserve c256 tok/s | 5623 | 6865 | 6863 | 6979 / 6965 |
| prefill L2048 b128 ms | 3833 | 3305 | 3368 | 3133 / 3125 |

Run spread in the re-measure: <= 0.4 %
(e.g. sA r25 128:1024 b256 full runs 19.259/19.252/19.251 s).

**The honest real cost of r=1/4 vs r=1/8** (reproduced in both sessions):
decode +0.2-1.8 %, qserve -1.6 %, prefill L2048 +5.5-7.8 %. A smooth
continuation of the curve, not a step.

## 3. Bisection anyway: route-table audit + per-shape GEMM probe

Even though the serve anomaly is an artifact, hypothesis (b) — the
(N,K)-keyed route tables (`_SMALLM_TABLE`/`_SPLITK_TABLE` in
promix/quantize/mixed_linear.py) were measured at r=1/8 and are r-blind —
deserved a kernel-level check. `docs/rd_r14_scripts/r14_gemm_probe.py`
(graph-replay timing, same bench_graph as the r2/r3 probes, GPU 4) at the 3B
shapes x M in {64,128,192,256,384} x Kh in {r=1/8, r=1/4}:

1. **No mis-window**: at every M probed, the config the r-blind tables pick
   is ALSO the per-M best config at the r=1/4 splits (qkv/o/down cfg1 and
   down's M<=128 split-K stay winners; gate_up cfg0 stays winner; the qkv
   window end at M=256 stays correct — cfg1 loses at M=384 at both r).
   No re-window is needed; no code change ships from this probe.
2. **The kernel-level r=1/4 cost is small and smooth**: route-pick GEMM time
   at M=256: qkv 16.6->17.5 us, o 13.9->14.9, gate_up 40.8->44.6,
   down 27.1->29.5 (+8.1 us total per layer-set); pack +~1.4 us. Over 28
   layers ~= +0.27 ms per decode step = +1.5 % of an 18.2 ms b256 TPOT —
   which is exactly the measured serve delta. Hypothesis (c) (a fall-off-the-
   fast-path step in the MXFP8 phase at Kh=768/2048) is refuted: cfg0/cfg1/
   split-K all shift by single-digit percent, monotonically, no cliff.

## 4. What changed in the repo

- `docs/vllm_final_fiveway.md`: 3B r-sweep r=0.25 row corrected (regenerated
  via harvest_rsweep from the substituted clean JSONs); finding 1 extended
  (flat through r=1/4); finding 4 rewritten as RESOLVED; lane caveat updated.
- `docs/vllm_final_data/pm3b_r0.25_{main,qserve}.json`: replaced with the
  clean session-A measurements (meta tags r14_sA_r25_*; batches 16/128/256 —
  all cells the harvest reads are present). Contaminated originals preserved
  under `docs/vllm_final_data/contaminated_gpu6_r0.25/`.
- `docs/rd_r14_data/`: both sessions' raw JSONs/logs, gpumon CSVs, GEMM-probe
  JSON/log, prebuild log. `docs/rd_r14_scripts/`: run_session.sh,
  r14_gemm_probe.py, r14_compare.py, r14_mem_audit.py (written for the
  memory-bloat side-quest; not needed — the clean-GPU engine log settles it),
  prebuild.py.
- **No serving-stack code changes** — therefore no gate re-runs are required
  (the stack is byte-identical to the gated r5/final configuration; the
  re-measure itself ran the identical plugin path, and its r=1/8 control
  reproduces the gated campaign numbers).

## 5. Secondary finding (resolved alongside)

The "18.45 GiB consumed" / shrunken-KV outlier of the campaign r=0.25 engine
is the SAME artifact (external allocation during init), not a load-path bug:
on clean GPUs the r=0.25 checkpoint consumes 6.2 GiB — in line with every
other sweep point — and gets KV 77.86 GiB. The r=1/4 splits (Kh=768/2048)
are 128-aligned, pad-free, and marlin low+high prepack succeed normally.

## 6. Caveat for the paper figure

The corrected r=1/4 serve numbers come from the 2026-11-03 sessions while
the other five sweep points keep their 2026-11-01 campaign lanes; the shared
r=1/8 control bridging both days agrees within 1.2 % on every cell, so the
curve remains cross-comparable at the stated +-4 % lane-noise bar (the
corrected point is additionally the best-attested one: two GPUs, two orders,
anchored).
