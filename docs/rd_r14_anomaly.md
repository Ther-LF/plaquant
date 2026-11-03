# rd-r14: the 3B r=1/4 decode anomaly (final-campaign open item)

> Investigation of docs/vllm_final_fiveway.md §4 finding 4: the 3B r=1/4
> sweep point showed decode TPOT b256 22.88 ms (vs 18.1-19.1 ms for every
> r <= 3/16) — the only swept point whose decode falls below BF16.
> Status: IN PROGRESS.

## 0. What the raw campaign data actually shows (worse than the table)

The published r-sweep table only surfaced the 128:1024 decode cells. The raw
JSON (`docs/vllm_final_data/pm3b_r0.25_main.json`) shows, within the ONE
GPU-6 session (2026-11-01 10:49-10:56):

| cell | r=1/4 | r=1/8 (gpu7) | delta |
|---|---|---|---|
| 1024:128 TPOT b16/b64/b128 | 5.89 / 10.51 / 16.15 | 6.09 / 10.74 / 16.30 | **-2 to -3 % (CLEAN, slightly faster)** |
| 1024:128 TPOT b256 | **52.08** | 29.27 | **+78 %** (never made the table) |
| 128:1024 TPOT b128/b256 | 14.12 / 22.88 | 11.29 / 18.14 | +25 / +26 % |
| 128:1024 TTFT b64/b128/b256 | 0.220 / 0.302 / 0.549 s | 0.106 / 0.198 / 0.396 s | +39 to +107 % |
| prefill L2048 b128 | 3833 ms | 3159 | +21 % |
| qserve c256 (separate engine, 10:56) | 5623 tok/s | 6931 | -19 % |

Within one session the early cells are clean (even fast) and everything from
1024:128 b256 onward is heavily elevated — a time-correlated signature, not a
per-kernel-regime signature (the same M=256 decode GEMMs run in both cases;
a kernel regime change cannot be +78 % in one case and +26 % in the other,
while M=128 decode in 1024:128 is simultaneously CLEAN at 16.15 ms).

## 1. Context facts gathered

- The r=0.25 point is the ONLY sweep point that ran on GPU 6; all others ran
  on GPUs 1/3/7. GPU 6 hosted the whole 8B chain (fp88b, gptq8b, pm8b_r0.0625
  qserve, pm8b_r0.25 main+qserve) until 10:48:29, and the published 3B r0.25
  numbers come from a `pm3b_r0.25_fix` RE-RUN (10:49-11:00) launched right at
  that boundary (the first 10:22 run predates the pad-aware SF alloc fix).
- Engine-level outlier: the r=0.25 engine reports **18.45 GiB consumed
  (weights + non-torch)** vs 4.67-6.6 GiB for every other sweep point
  (checkpoint on disk: 3.3G vs 3.2G) -> KV budget 617,872 vs 724,960 tokens.
  No preemption at any anomalous cell (grep preempt = 0; max live tokens
  294,912 << budget). Load-path memory bloat at r=1/4 is a REAL secondary
  finding (suspect: marlin/GEMV companion prepack at Kh=768/2048), but by
  itself does not explain decode TPOT.
- Route tables (_SMALLM_TABLE / _SPLITK_TABLE / _MARLIN_TABLE in
  promix/quantize/mixed_linear.py) are keyed by (N, K) only — windows were
  MEASURED at r=1/8 splits and apply unchanged at r=1/4. At decode M=128/256,
  3B qkv/o/down route through cfg1 (128x64) / split-K windows. A mis-window at
  Kh=768/2048 is hypothesis (b); it predicts an equal step in BOTH decode
  cases at the same M — which the raw data contradicts.

## 2. Plan

1. Re-measure cleanly (docs/rd_r14_scripts/run_session.sh): 2 sessions
   (GPU 4 fwd order, GPU 5 rev order), each self-anchored: BF16 anchor +
   r=1/8 control + r=1/4, same protocol cells, nvidia-smi interference
   monitor running throughout.
2. If reproduced: per-shape GEMM probe at r=1/4 splits vs r=1/8 splits under
   graph replay (gemm_probe --khigh), route-table audit at decode M, and a
   levers-off control serve (PROMIX_SMALLM=0 PROMIX_SPLITK=0).
3. Fix (measured re-window) or document the real regime cost; correct the
   r-sweep point either way. Also root-cause the 18.45 GiB load bloat.

(Results sections to follow.)
