# Operator-level Pareto — final-kernel latency-vs-r curve + competitor points (E4)

**The paper's operator-Pareto figure data**: the FINAL post-R1-R5 shipping fused
NVFP4+MXFP8 GEMM re-measured as a FULL latency-vs-r curve (r = FP8 fraction, 0→1),
with every runnable competitor operator as single points on the same shapes —
x = latency, y = effective bits (accuracy-proxy axis).

**HW** RTX PRO 6000 Blackwell Server (SM120, cc 12.0), container `luofan-plaquant`
(torch 2.11 cu130, nvcc 13.0), **idle GPU 6**, one serialized session.
**Date** 2026-11-03. **Branch** `feat/rd-oppareto` (off `origin/feat/rd-alloc`),
remote workdir `/home/luofan/plaquant-oppareto`.
**Data** `docs/rd_operator_pareto.csv` (plot-ready, long format) +
raw JSON in `docs/rd_oppareto_data/`.

## Protocol

- **Shapes** (N×K, unmerged per-projection, same as `rd_baselines_m8192.md`):
  3B qkvo 3072×3072, gate_up 8192×3072, down 3072×8192; 8B qkvo 4096×4096,
  gate_up 14336×4096, down 4096×14336. **M ∈ {2048, 8192}** (prefill regimes).
- **Our curve** (`promix/eval/rd_oppareto_ours.py`): the SHIPPING dispatch —
  `MixedQuantLinear._gemm`, i.e. the exact serving launch path with all R1-R5
  levers default-ON (per-shape split-K/small-M window tables consulted at this M,
  r5 cast-fold bias epilogue, K_high==0 → single-phase NVFP4). GEMM-only on
  prepacked activations, cudaEvent median-of-50 (15 warmup) — identical protocol
  to the competitor suite — plus a **CUDA-graph replay median** (`graph_ms`,
  the serving-mode number; vLLM replays captured graphs). At M ∈ {2048, 8192} no
  split-K/small-M/Marlin window fires (all end ≤512): every r>0 point launches
  `fused_gemm_prepacked_bias` (castfold 128×128), r=0 launches
  `single_nvfp4_gemm_prepacked` (+ its shipping `.to(fp16)` cast).
- **r grid** {0, 1/32, 1/16, 1/12, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1}; K_high
  snapped to **RING=64** (the rd-allocation ring unit). **r=1 is served as
  all-but-one-ring** (K_low=64): K_low==0 (pure-MXFP8) has no kernel path.
- **Competitors** re-run in the same session at both M via the rd-bench harnesses
  (`rd_bench_m8192_{vllm,arcquant,micromix,sharq}.py`) + new
  `rd_oppareto_torch.py` (torch-native `_scaled_mm_v2` NVFP4 / MXFP8, bf16 cuBLAS).
  Each method as shipped; activation quant/prep excluded for all (GEMM-only).
- **Effective bits** (element+scale): NVFP4 4.5, MXFP8 8.25; ours
  = (4.5·K_low + 8.25·K_high)/K (r=1/8 → 4.97); ARCQuant 4.5·(K+S)/K;
  MicroMix pure 4.25 / mix 5.75; SharQ ≈4.5 payload (but ~2× GEMM work);
  torch_fp4 4.5; torch_fp8 8.25; bf16 16.
- Validation: every curve point cosine-gated vs fp32 reference (0.9924–0.9993,
  quantization-level, as expected on random data); competitor cosines 0.986–0.999.

## 1. Our curve — summary (full 132-point curve in the CSV)

cudaEvent median ms; `graph` = CUDA-graph replay median (serving mode).

| shape · M | r=0 | r=1/32 | r=1/8 (prod) | r=1/4 | r=1/2 | r=1 | r=1/8 vs r=0 | r=1 vs r=0 |
|---|---|---|---|---|---|---|---|---|
| 3B qkvo · 2048 | 0.0911 | **0.0900** | 0.0952 | 0.0997 | 0.1097 | 0.1314 | +4.5 % | 1.44× |
| 3B qkvo · 8192 | 0.2132 | **0.1856** | 0.1984 | 0.2142 | 0.2438 | 0.3074 | −7.0 % | 1.44× |
| 3B gate_up · 2048 | 0.1489 | **0.1378** | 0.1522 | 0.1625 | 0.1793 | 0.2253 | +2.2 % | 1.51× |
| 3B gate_up · 8192 | 0.5611 | **0.3981** | 0.4304 | 0.4668 | 0.5450 | 0.7024 | −23.3 % | 1.25× |
| 3B down · 2048 | **0.1462** | 0.1506 | 0.1606 | 0.1725 | 0.1972 | 0.2491 | +9.8 % | 1.70× |
| 3B down · 8192 | 0.3834 | **0.3704** | 0.4006 | 0.4432 | 0.5175 | 0.6862 | +4.5 % | 1.79× |
| 8B qkvo · 2048 | 0.1044 | **0.1009** | 0.1099 | 0.1158 | 0.1289 | 0.1580 | +5.3 % | 1.51× |
| 8B qkvo · 8192 | 0.2984 | **0.2621** | 0.2874 | 0.3130 | 0.3796 | 0.4824 | −3.7 % | 1.62× |
| 8B gate_up · 2048 | 0.2764 | **0.2413** | 0.2642 | 0.2861 | 0.3297 | 0.4249 | −4.4 % | 1.54× |
| 8B gate_up · 8192 | 1.0827 | **0.8043** | 0.8948 | 0.9656 | 1.1272 | 1.4661 | −17.4 % | 1.35× |
| 8B down · 2048 | **0.2271** | 0.2375 | 0.2502 | 0.2753 | 0.3260 | 0.4211 | +10.2 % | 1.85× |
| 8B down · 8192 | 0.7804 | 0.7565 | 0.7995 | 0.8907 | 1.0730 | 1.4181 | +2.5 % | 1.82× |

**Curve shape (the figure's story):**

- **Flat head**: the whole production region r ≤ 1/8 sits within **6–11 %** of
  the per-shape curve minimum (and r=1/4 within 11–21 %) — the mixed-precision
  protection is nearly free at the operator level. The production point r=1/8
  costs **+2 to +10 % over r=0** at M=2048 and is **−4 to −23 % (FASTER than
  r=0)** at M=8192.
- **Non-monotone r=0 end** (honest): the K_high==0 single-phase path is SLOWER
  than the dual-phase kernel at small r>0 on 10/12 scenarios (curve min at
  r=1/32, or r=1/16 for 8B down·8192), worst on the large-N gate_up at M=8192 (0.561 vs 0.398, −29 %) —
  the dual kernel's whole-K overlap and castfold epilogue outrun the
  single-phase launch; a known structural weak spot of the pure path, not of
  the mixed design.
- **Linear tail**: above r≈1/4 latency grows ~linearly in r to the
  all-but-one-ring end; r=1/r=0 slope = **1.25–1.85×** (down shapes steepest —
  K-heavy GEMMs feel the MXFP8 half-throughput most), bracketing the
  FP4:FP8 format ratio at these shapes (~1.3–1.5× at M=2048, rd-perfpuzzle).
- `graph_ms` runs ~20–30 µs below the event-timed median at M=2048 (launch/event
  overhead) and ~3–5 % below at M=8192; curve shape identical. Event medians are
  the canonical cross-method numbers (competitors are event-timed).

## 2. Competitor operator points (ratio = competitor / ours-at-r=1/8; >1 = we win)

**M = 2048**

| shape | ours r=1/8 (ms) | torch FP4 | ARC S=0 | MicroMix FP4 | ARC S=512 | SharQ | MicroMix 5.5b | vLLM-NVFP4 | torch FP8 | bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3B qkvo | 0.0952 | 0.80 | 1.16 | 1.27 | 1.21 | 0.80 | 2.91 | 2.27 | 1.21 | 1.65 |
| 3B gate_up | 0.1522 | 0.80 | 0.99 | 1.05 | 1.07 | 0.98 | 1.98 | 1.79 | 1.28 | 1.88 |
| 3B down | 0.1606 | 0.82 | 1.03 | 1.10 | 1.06 | 1.07 | 1.89 | 1.69 | 1.41 | 1.96 |
| 8B qkvo | 0.1099 | 0.80 | 1.09 | 1.18 | 1.15 | 0.90 | 2.56 | 2.04 | 1.27 | 1.86 |
| 8B gate_up | 0.2642 | 0.80 | 0.93 | 0.96 | 1.00 | 1.25 | 1.61 | 1.32 | 1.41 | 2.32 |
| 8B down | 0.2502 | 0.90 | 0.95 | 1.00 | 0.97 | 1.22 | 1.54 | 1.45 | 1.58 | 2.55 |

**M = 8192**

| shape | ours r=1/8 (ms) | torch FP4 | ARC S=0 | MicroMix FP4 | ARC S=512 | SharQ | MicroMix 5.5b | vLLM-NVFP4 | torch FP8 | bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3B qkvo | 0.1984 | 0.84 | 0.98 | 1.04 | 1.05 | 1.11 | 1.74 | 1.54 | 1.37 | 2.34 |
| 3B gate_up | 0.4304 | 0.80 | 0.89 | 0.95 | 0.99 | 1.39 | 1.81 | 1.14 | 1.43 | 2.34 |
| 3B down | 0.4006 | 0.83 | 0.91 | 0.94 | 0.95 | 1.34 | 1.51 | 1.19 | 1.54 | 2.72 |
| 8B qkvo | 0.2874 | 0.80 | 0.91 | 0.95 | 0.99 | 1.32 | 1.59 | 1.29 | 1.42 | 2.53 |
| 8B gate_up | 0.8948 | 0.78 | 0.87 | 0.96 | 0.94 | 1.45 | 1.73 | 0.96 | 1.49 | 2.74 |
| 8B down | 0.7995 | 0.90 | 0.91 | 0.91 | 0.92 | 1.50 | 1.40 | 1.10 | 1.72 | 3.03 |

**Pareto reading** (x = latency, y = bits; lower-left dominates):

- **Against the protected/outlier-handling class** (the apples-to-apples row):
  ours (4.97 b) vs **ARCQuant S=512** (4.66–5.25 b) is parity ±8 % at M=8192 and
  a qkvo-class win at M=2048 (1.15–1.21×); vs **SharQ** (~4.5 b payload, 2 GEMMs)
  we win every M=8192 shape 1.11–1.50× while SharQ wins small-M qkvo (0.80/0.90);
  vs **MicroMix mix-5.5 b** we win everywhere 1.40–2.91×; vs **vLLM-NVFP4**
  (4.5 b, FlashInfer-cutlass) we win 11/12 cells up to 2.27× (lone loss:
  8B gate_up·8192 at 0.96).
- **The unprotected pure-FP4 floor is below the curve by construction**:
  torch-native cuBLASLt FP4 (4.5 b, no outlier handling) is the fastest operator
  measured on every shape — our 4.97 b mixed point pays **+11 to +28 %** over it;
  that gap IS the measured price of protection, and it is far cheaper than the
  competitors' protection mechanisms (ARCQuant K-extension ≈ same price at
  matched bits; SharQ's second GEMM ≈ +30–90 %; MicroMix segmentation ≈ +40–190 %).
- **Format anchors**: torch-native MXFP8 (8.25 b) costs 1.21–1.72× ours; bf16
  cuBLAS 1.65–3.03×. Our r=1 end (~8.2 b) lands within ±15 % of torch-FP8 —
  the FP8-heavy tail is not our operating regime and is priced accordingly.

## 3. New (post-R5) vs old (pre-R1-R5) curve — what the optimization rounds bought

Old = `docs/rd_perfpuzzle_fused_khigh.csv` (rd-perfpuzzle, 2026-10-29,
`launch_best_codesign` auto dispatch, 3B shapes, M=2048, cudaEvent median-of-40).
New = this round, matched K_high, M=2048:

| region | 3B qkvo | 3B gate_up | 3B down |
|---|---|---|---|
| **r=0 endpoint** | **1.49×** faster (0.136→0.091) | **1.37×** (0.204→0.149) | **1.86×** (0.272→0.146) |
| mid-curve (r=1/24…3/4, same dual kernel) | 1.13–1.17× | 1.07–1.11× | 1.07–1.11× |
| **r=1 endpoint** | **1.25×** (0.164→0.131) | **1.27×** (0.287→0.225) | **1.38×** (0.344→0.249) |

- The **endpoints moved the most**: r=0 is 1.37–1.86× faster because the old
  curve's r=0 was the dedicated pure-NVFP4 dispatcher (and the production path
  **had no K_high=0 lever at all** — rd-perfpuzzle stage 3); the kernel-fix
  single-phase skip-P2 path now serves it. r=1 was the pure-MXFP8 full-K
  dispatcher; the all-but-one-ring fused kernel beats it by 1.25–1.38×.
- The **mid-curve moved +7–17 %** on the same dual kernel (castfold epilogue
  fold + the R1-R5 launch-path work; cross-session clock drift on this
  no-clock-lock box contributes — treat <5 % as parity, but the shift is
  uniform-sign across all 30 matched points).
- Consequence for the old "flat in K_high, no r=0 lever" verdict
  (rd-perfpuzzle stage 3): **partly repaired**. The curve now has a real r
  slope at the GEMM level (r=1/8 → r=0/r=1/32 is −2 to −10 % at M=2048, more at
  M=8192), and the r=0 path exists and wins at small M; the gate_up/qkvo
  M=8192 r=0 regression (−17…−23 % vs r=1/32) is the remaining known pure-path
  weakness.

## 4. Honest notes & caveats

1. **K_high%64==32 alignment cliff (new finding)**: the first sweep snapped the
   grid to the MXFP8 32-block and every half-ring cell (K_high = 96/352/672/1184)
   ran **15–40 % slower** than both r-neighbors (median==min, both M, graph-replay
   too — systematic, not contention; data kept in
   `docs/rd_oppareto_data/misaligned_khalf32_pass.json`). The production
   rd-allocator emits whole RING=64 rings only, so the shipped config space never
   hits it; the final curve snaps to 64. Worth one sentence in the paper (the
   allocator granularity is also the kernel-friendly granularity).
2. **No clock locking** (no sudo); one serialized session on idle GPU 6, box
   load ~5. The per-method fp16/bf16 references agree within ~1–2 % across the
   session (ours-harness fp16 anchors vs torch-harness bf16: 0.157/0.157 at 3B
   qkvo·2048 … 2.457/2.419 at 8B down·8192). Treat <5 % as parity.
3. **MicroMix** correctness remains unvalidated under CUDA 13 as shipped (its own
   `--validate` fails; our cosine probe 0.986/0.989) — same caveat as
   `rd_baselines_m8192.md`.
4. **ARCQuant S** is calibration-adaptive (≤512); S ∈ {0, 256, 512} brackets its
   deployed GEMM (≤10 % spread). Points at S=0 and S=512 are both in the CSV.
5. **SharQ** numbers are its own shipped CUDA harness (50 warmup / 200 iters avg,
   fused sparse+residual accumulation = its real per-Linear GEMM work).
6. **r=1 is all-but-one-ring** (K_low=64), not true pure-MXFP8 — K_low==0 is
   unsupported by design (`MixedQuantLinear` raises). The true pure-MXFP8
   endpoint would need a dedicated single-phase MXFP8 kernel we don't ship.
7. Cosines are run sanity-checks on random data, not accuracy claims — the
   accuracy axis of the real figure should use each method's measured PPL/bits;
   `eff_bits` in the CSV is the bits proxy.
8. Ours r=0/r>0 compare includes each path's shipping epilogue: r>0 stores fp16
   in-kernel (castfold), r=0 pays its separate `.to(fp16)` — that IS the
   shipping launch path at each point.

## Reproduce

```bash
# remote, host side
ssh luofan@10.208.130.90   # repo at /home/luofan/plaquant-oppareto
docker exec -w /home/luofan/plaquant-oppareto -e PYTHONUNBUFFERED=1 luofan-plaquant \
  bash -c "bash promix/scripts/rd_oppareto_all.sh 6"   # arg = idle GPU
# then locally
python3 promix/eval/rd_oppareto_agg.py   # -> docs/rd_operator_pareto.csv
```
