# rd-buildup: E5+E6 — cumulative component build-up ablation + error bars

**Branch** `feat/rd-buildup` (off `origin/feat/rd-alloc`) · **HW** RTX PRO 6000
Blackwell (SM120), container `luofan-plaquant`, remote dir
`/home/luofan/plaquant-buildup`, GPU lane 5 (accuracy) / 7 (performance) ·
**Date** 2026-11-03 · **Raw JSONs** `docs/rd_buildup_data/` (synced from
`/home/luofan/buildup/`) · **Drivers** `promix/scripts/rd_buildup.py`
(ladder), `rd_buildup_queue.sh` (GPU-5 queue),
`promix/vllm_integ/run_buildup_perf.sh` (GPU-7 ladder),
`rd_buildup_harvest.py` (tables), `rd_acctune.py --calib_seed` /
`rd_validity.py --calib_seed` (E6).

This is the paper-spec E5 ("系统把几项工作拆开，各自收益多少") plus E6
(3-seed error bars): ONE clean cumulative ladder re-run end-to-end on a
single substrate/protocol — not stitched from prior campaigns. Protocol
gates confirm the ladder's endpoint reproduces the shipped recipe
**digit-exactly** on all three models (3B 8.2180/11.2549 · 1B
14.1694/19.7110 · 8B 6.5642/9.5639 == `docs/rd_accuracy_tuning.md`).

---

## 1. E5a — accuracy build-up (Llama-3.2-3B, wiki / C4 PPL)

Each row = previous row + ONE component. Rows 1–3 are pure NVFP4 (K_high=0
everywhere, incl. o_proj); rows 4–8 at the matched avg FP8 = 1/8 budget.
bf16 bridge wiki = 7.8148 (digit-exact per acctune protocol).
One row per process (env knobs are import-time), env verified by the driver.

| # | cumulative config | wiki | Δ wiki | C4 | Δ C4 |
|---|---|---|---|---|---|
| 1 | RTN pure-NVFP4, **no rotation** | 8.9896 | — | 12.1332 | — |
| 2 | + PCA U rotation (no R1; full offline substrate: norm fusion, sorted coords, per-head o_proj PCA, down_proj online Hadamard) | 13.5351 | **+4.546** | 17.7806 | +5.647 |
| 3 | + ring512 R1 (T = U·ring512_R1) | 13.4905 | −0.045 | 17.5308 | −0.250 |
| 4 | + uniform-1/8 mixed precision (positional tail) | 9.9651 | **−3.525** | 13.4859 | **−4.045** |
| 5 | + GPTQ (n=512, percdamp .025) | 8.8960 | −1.069 | 12.3437 | −1.142 |
| 6 | + two-level weight scale (micro-MSE d7) | 8.5964 | −0.300 | 11.8524 | −0.491 |
| 7 | + act-MSE scale search (d3u2) | 8.4221 | −0.174 | 11.5420 | −0.310 |
| 8 | + measured-ΔD allocation (replaces uniform) | **8.2180** | **−0.204** | **11.2549** | **−0.287** |
| 9 | + decode W4A16 (decode-route PPL: same weights, act exact) | 7.9527 | −0.265 | 10.7997 | −0.455 |

Row 8 == shipped anchor **digit-exact** (acctune `3b_stack_full_g512_d025`).
Row 9 is not a serving config change on the prefill route — it reports the
PPL the decode-route W4A16 kernels (marlin/GEMV m≤16 windows) realize when
activations stay exact; its *gain* (−0.27) is the accuracy headroom the
decode route enjoys over the W4A4 number we report (we report the W4A4 row-8
number as the model's PPL; decode quality is strictly better).

### Final-row replication (full recipe, other sizes)

| model | wiki | C4 | decode-route W4A16 wiki / C4 |
|---|---|---|---|
| 1B | 14.1694 | 19.7110 | 13.4540 / 18.9064 |
| 3B | 8.2180 | 11.2549 | 7.9527 / 10.7997 |
| 8B | 6.5642 | 9.5639 | 6.3023 / 9.1649 |

All three digit-exact vs the shipped acctune anchors → the ladder's endpoint
is the production recipe, not a re-tuned variant.

### Step-order sensitivity (the honest part) + no-rotation controls

The ladder is NOT monotone, and that is the finding:

* **Rotation alone HURTS pure NVFP4** (rows 2–3: 8.99 → 13.49). NVFP4's
  per-16-channel E4M3 block scales already contain unrotated outliers
  locally; PCA + ring rotation *concentrates* outlier energy into a few
  sorted coordinates, so without the FP8 partition those coordinates blow up
  their blocks. Rotation's value is only realized by the partition it
  enables (row 4 recovers −3.53 at once) — and even then the
  rotated+partitioned RTN stack (9.97) is still +0.98 over plain RTN: at the
  RTN stage the substrate has not yet paid for itself. It pays through its
  interactions — GPTQ (row 5, 8.90, first crossing below row 1), scale
  search, and allocation — ending at 8.22 vs the best achievable
  no-rotation stack 14.88 (C1) / 8.99 (plain RTN, no compensation possible
  on top, see controls).
* **GPTQ's value is conditional on the substrate.** On the ladder it buys
  −1.07 (row 5). On the UNROTATED model it is catastrophic:

| control (3B, pure NVFP4, no rotation) | wiki | C4 |
|---|---|---|
| C3: RTN + GPTQ only (plain scales) | 35.3901 | 37.3272 |
| C2: + two-level weight scale (no act-MSE) | 27.0719 | 27.8736 |
| C1: + act-MSE d3u2 (full weight/act stack) | 14.8847 | 19.7747 |
| reference: plain RTN (ladder row 1) | 8.9896 | 12.1332 |

  GPTQ calibrates against exact activations; on the unrotated model the
  eval-time NVFP4 activation error on outlier channels interacts with
  GPTQ's error redistribution and *destroys* the model (8.99 → 35.39).
  The entire weight/activation compensation stack only becomes a
  net-positive on the rotated+partitioned substrate — so the simple
  "per-component gain" question is ill-posed without fixing the order; this
  cumulative ladder (substrate first, compensation second, allocation last)
  is the order in which every component is individually positive from row 4
  on.
* Component order within the compensation block (rows 5–7) matches the
  acctune per-lever signs; allocation last (row 8) replaces uniform at the
  same budget (avg_fp8 0.125 both, `avg_fp8` field in the JSONs).

---

## 2. E5b — performance build-up (vLLM serving, ring512 ckpts, same session/GPU)

Protocol: vLLM 0.32, graphs+compile ON, ckpts
`/home/luofan/vllm_ring512/promix-{3b,8b}-ring512`, one engine per step,
decode TPOT = case ISL128/OSL1024, prefill = TTFT L2048, GPU 7. The
kill-switch env is read by the plugin at runtime (identical compiled graph,
different routing), so each engine realizes exactly its step. Step 6 (all
levers = shipped default) reproduces the committed ring512 serving numbers
(3B b128: 11.72/3193 vs committed 11.64/3198).

Step 1 (naive 2-launch) is measured at the GEMM level — torch-native
`torchmixed2` = native NVFP4 `_scaled_mm_v2` over K_low + native MXFP8 over
K_high + add, the framework-native 2-launch rendering of our algorithm, from
`promix/eval/gemm_h2h_torchfp4.py` at the exact vLLM decomposition shapes —
and converted to a serving estimate analytically (full-serving naive is
impractical: the plugin has no 2-launch path). **Method**: naive_serving =
step-2 serving + n_layers × n_steps × Σ_shapes(mix2 − fused); decode = 1
step/token at M=batch; prefill b128·L2048 = 262144 tokens / 16384-token
chunks (`max_num_batched_tokens=16384`, log-verified) = 16 chunked-prefill
steps at M=16384. Per-layer-set deltas: 3B +74.2 µs (M128) / +71.8 µs (M16)
/ +1672.9 µs (M16384); 8B +35.6 / +33.8 / +2725.5 µs.

### Llama-3.2-3B (28 layers)

| # | cumulative config | decode b128 TPOT (ms) | decode b16 TPOT (ms) | prefill L2048 b128 (ms) |
|---|---|---|---|---|
| 1 | naive 2-launch (torch-native, analytic*) | ~14.5 | ~9.0 | ~4508 |
| 2 | single-launch fused GEMM (levers off) | 12.44 | 6.99 | 3758.8 |
| 3 | + per-shape dispatch / small-M configs | 12.10 | 6.73 | 3766.0 |
| 4 | + split-K | 11.98 | 6.73 | 3713.9 |
| 5 | + GEMV/marlin decode routes (+hi) | 12.18 | **5.27** | 3771.8 |
| 6 | + pack/gather/cast fusions (= shipped) | **11.72** | **5.05** | **3193.0** |

### Llama-3-8B (32 layers)

| # | cumulative config | decode b128 TPOT (ms) | decode b16 TPOT (ms) | prefill L2048 b128 (ms) |
|---|---|---|---|---|
| 1 | naive 2-launch (analytic*) | ~18.5 | ~11.9 | ~8966 |
| 2 | single-launch fused GEMM (levers off) | 17.34 | 10.85 | 7570.8 |
| 3 | + per-shape dispatch / small-M | 16.89 | 10.80 | 7601.6 |
| 4 | + split-K | 16.92 | 10.54 | 7586.6 |
| 5 | + GEMV/marlin routes (+hi) | 17.04 | **7.85** | 7578.9 |
| 6 | + pack/gather/cast fusions (= shipped) | **16.61** | **7.84** | **6637.8** |

\* step-1 = step-2 + analytic GEMM delta (method above); GEMM-only mix2/fused
ratios: 1.1–1.7× at prefill M, 1.3–1.5× on qkv/o_proj at decode M (gate_up/
down at M≤128 are launch-overhead-bound, ≈1.0×).

Readings (who buys what):

* **Fusion (2-launch → single launch)** is the largest single kernel lever
  everywhere: −14 % 3B / −6 % 8B decode b128, −17/−16 % prefill (analytic).
* **Dispatch + split-K** are small-but-real decode levers (−0.5 ms 3B b128
  combined; −0.3 ms 8B b16); within session noise on prefill.
* **GEMV/marlin routes only engage in their m≤16 windows**: invisible at
  b128 (±noise), the dominant decode lever at b16 (3B 6.73→5.27, −22 %;
  8B 10.54→7.85, −26 %). We report both batch points precisely to expose
  this routing structure; a b128-only table would wrongly call step 5 dead.
* **Pack/gather/cast fusions** are the big prefill lever (−15 % 3B, −12.5 %
  8B) plus −2..−4 % decode.
* Cumulative step-2→step-6: 3B decode b16 −28 %, prefill −15 %; 8B decode
  b16 −28 %, prefill −12 %. Against the naive step-1 estimate: 3B prefill
  −29 %, decode b16 −44 %; 8B prefill −26 %, decode b16 −34 %.

---

## 3. E6 — error bars (3 calibration seeds)

Seed enters calibration only (`--calib_seed`: transformers.set_seed, wiki
calib sampling for GPTQ + ΔD collect); eval sets pinned (wiki test set is
seed-free; C4 eval windows pinned to seed 0). Seed 0 = the ladder/replicate
runs above; seeds 1, 2 fresh.

| headline | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| 3B ProMix measured-1/8 wiki | 8.2180 | 8.2151 | 8.2142 | **8.2158 ± 0.0020** |
| 3B — C4 | 11.2549 | 11.2743 | 11.2599 | 11.2631 ± 0.0101 |
| 8B ProMix measured-1/8 wiki | 6.5642 | 6.5695 | 6.5666 | **6.5668 ± 0.0026** |
| 8B — C4 | 9.5639 | 9.5806 | 9.5733 | 9.5726 ± 0.0084 |

Three-way allocation gate (3B, rd_validity protocol: R1-off, matched 1/8
budget, bf16 bridge 7.8148 digit-exact every seed):

| seed | A uniform | B variance-proxy | C measured-ΔD | ordering |
|---|---|---|---|---|
| 0 | 8.3924 | 8.4711 | 8.3124 | C < A < B — PASS |
| 1 | 8.4115 | 8.5318 | 8.3368 | C < A < B — PASS |
| 2 | 8.4067 | 8.4753 | 8.3057 | C < A < B — PASS |

**Orderings are stable across all seeds**: measured-ΔD beats uniform beats
the variance proxy, every seed, with gaps (C→A ≥ 0.075, A→B ≥ 0.069) an
order of magnitude above the seed std (~0.002–0.010). Headline deltas
between methods in the paper (e.g. measured vs uniform −0.05 wiki at 3B)
are ~5–25× the calibration-seed std.

Seed-0 A/C values reproduce the committed campaign numbers digit-exactly
(A 8.3924, C 8.3124) — same-session re-runs, not copied.

---

## 4. Reproduce

```bash
# accuracy ladder + E6 (GPU 5, container, detached)
docker exec -d -w /home/luofan/plaquant-buildup -e CUDA_VISIBLE_DEVICES=5 \
  luofan-plaquant bash promix/scripts/rd_buildup_queue.sh all
# controls
docker exec -d ... bash promix/scripts/rd_buildup_queue.sh controls
# performance ladders (GPU 7): b128 pass then b16 pass
docker exec -d -w /home/luofan/plaquant-buildup -e CUDA_VISIBLE_DEVICES=7 \
  luofan-plaquant bash promix/vllm_integ/run_buildup_perf.sh all
docker exec -d ... bash promix/vllm_integ/run_buildup_perf.sh b16
# tables
python promix/scripts/rd_buildup_harvest.py
```
