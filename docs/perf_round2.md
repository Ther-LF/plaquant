# PERF round-2 report (performance axis)

RTX PRO 6000 Blackwell (SM120), container `luofan-plaquant`. All numbers measured
this round in standalone harnesses; where isolation prevents an end-to-end number
the component numbers are given with the reasoning. No shared-forward wiring was
changed (that is next round); the exact wiring patch for each result is listed.

New files (all non-protected / new standalone):
- `kernels/mixed_gemm_sm120/hadamard28_pack_ext.cu` — Task 1 kernel (8B I=14336 fused producer).
- `kernels/mixed_gemm_sm120/decode_gemv_ext.cu` — Task 3 added `gemv_w4a4_v4` (negative) + `decode_gemv_v5` / `launch_v5` (the win).
- `promix/eval/perf_r2_isolated.py` — Task 1 gate + perf harness.
- `promix/eval/perf_r2_decode_gemv.py` — Task 3 GEMV roof sweep.
- `promix/eval/perf_r2_kv4.py` — Task 2 KV4 memory + bandwidth + accuracy probe.
- Result JSONs: `perf_r2_hadamard28.json`, `perf_r2_decode_gemv.json`, `perf_r2_kv4.json`.

---

## Task 1 — 8B fused producer for non-power-of-two width (I = 14336)  ✅ BIT-IDENTICAL

**What was built.** `hadamard28_swiglu_pack_mixed` — one CUDA kernel that fuses
`silu(gate)*up → online Hadamard → *1/sqrt(14336) → fp16 → NVFP4(low)+MXFP8(high)
pack` for Llama-3.1-8B's down_proj input, replacing the slow 4-op reference chain
(`silu*up → matmul_hadU_cuda(h.float(), had28, 28, fold_scale=True) → .half() →
quant_pack_float_mixed`) that the shipping path falls back to because 14336 is not
a power of two.

**How bit-identity was achieved** (the hard gate, matched to the 1B fusion's
0-byte standard):
- 14336 = 28·512. ResQ's reference reshapes each row to [28, 512], runs the
  library's power-of-two Hadamard on the 512 sub-dim, then a torch fp32 `had28 @`
  (28×28 ±1 matmul), then fp16+pack.
- The library picks **kNThreads=32, kNElts=4, kNChunks=4, kNWarps=1 (no smem
  exchange)** for log_N=9 (dim 512). The kernel reproduces exactly that
  decomposition per 512-group inside one warp (verbatim `hadamard_mult_thread` /
  `hadamard_mult_warp` from fast-hadamard-transform, BSD-3), with load_input's
  index map `x_vals[c][i] ↔ col (c·128 + lane·4 + i)`.
- `1/sqrt(14336)` is applied at the transform store **before** the had28 sum
  (the reference fold order; fp32 rounding depends on it).
- **had28 accumulation order verified empirically**: torch `had28 @ x` is
  byte-identical to the ascending sequential sum g'=0..27 (probe: 0/917504
  mismatch ascending; 729888 descending). had28 entries are ±1 so `fma(±1,x,acc)
  == acc±x` (one rounding == mul-add) and only the order matters. The kernel sums
  ascending.
- pack = verbatim qpf device code (same NVFP4 per-16 / MXFP8 per-32 path already
  proven byte-exact for the 8192 fusion).

**Gate result (`perf_r2_isolated.py --phase gate`).** 0-byte mismatch on all four
output tensors (code_lo, sfa_lo, code_hi, sfa_hi), both bf16 and fp16, at
M=512/2048/4096 — e.g. bf16 M=2048 `[0,0,0,0]` of `[12845056,1605632,3670016,114688]`.
**PASS — bit-identical, matching the 1B 0-byte standard.**

**Isolated speed** (`--phase perf`, production 8B shape I=14336 K_low=12544
K_high=1792 N=4096, min-of-30, event-synced; fused vs the 4-op chain, both are
producer-only, no GEMM):

| M | fused (ms) | ref chain (ms) | speedup | fused eff. BW |
|---|-----------|----------------|---------|---------------|
| 512  | 0.212 | 0.134 | 0.63× | 0.16 TB/s |
| 2048 | 0.721 | 0.745 | 1.03× | 0.19 TB/s |
| 4096 | 1.374 | 1.717 | 1.25× | 0.20 TB/s |
| 8192 | 2.654 | 3.445 | 1.30× | 0.20 TB/s |

The fused producer wins from M=2048 up (the prefill regime) and the win grows with
M; it loses at M=512 (below the prefill range).

**Honest limitation.** The kernel is SMEM/occupancy-bound, not HBM-bound (0.20 vs
~1.46 TB/s). The had28 matmul must read all 28 input groups in **fp32** to match
torch's fp32 accumulation, forcing a 56 KB fp32 SMEM tile per CTA → the device cap
(99 KB/block, 100 KB/SM) allows **only 1 CTA/SM**. A 512-thread + coalesced-had28
+ fp16-reshuffle variant was tried and was ~2× **slower** (extra sync + second pass
hurt ILP at the forced 1 CTA/SM), so the shipped kernel is the 256-thread, direct
per-block-had28 version. This is why the 14336 fusion does not reach the near-free
efficiency the power-of-two 1B fusion did (that one had no had28 matmul, all the
reduction stayed in the register butterfly).

**Expected e2e impact (8B prefill).** Producer saving per forward = 32 layers ×
(chain − fused): +0.8 ms @M2048, +11.0 ms @M4096, +25.3 ms @M8192. Applied to the
multisize 8B real-graph prefill (slow-producer baseline 78.90 / 170.81 / 391.94 ms,
= 1.293 / 1.241 / 1.224 ×FP16):

| M | 8B prefill ×FP16 before | after (fused producer) |
|---|-------------------------|------------------------|
| 2048 | 1.293 | ~1.306 |
| 4096 | 1.241 | ~1.327 |
| 8192 | 1.224 | ~1.309 |

So fusing the 8B producer lifts 8B prefill by ~+0.07–0.09× at M≥4096 (best at
M=8192: 1.22 → 1.31×) — a real step toward the 1B fusion gain, bounded by the
1-CTA/SM SMEM limit above (a fully HBM-bound producer would save ~3× more; the fp32
had28 is the blocker). It does **not** by itself close the gap to vLLM NVFP4's
2.21×; it is one of the levers.

**Wiring patch for next round (not applied).** In `promix/quantize/mixed_linear.py`
`MixedQuantLinear.forward_swiglu_pack` (and the `hpk` loader in
`promix/quantize/full_network.py` / `promix/eval/full_network_realkernel.py`): when
the intermediate width I == 14336 (8B), dispatch to the new extension
`hadamard28_pack_ext.hadamard28_swiglu_pack_mixed(gate, up, had28_cpu, M, N_gemm,
K_low, K_high)` instead of the power-of-two `hadamard_swiglu_pack_mixed` (which
`static_assert`s I==8192). `had28_cpu` is `get_hadK(14336)[0]` (the 28×28 ±1 matrix
from `promix/quantize/hadamard_data.json`). K_low=12544, K_high=1792, N_gemm=4096.
Keep the existing power-of-two path for 1B/3B (I=8192). Gate before ship: re-run
`perf_r2_isolated.py --phase gate` (must stay 0-byte).

---

## Task 3 — decode GEMV toward the HBM roof  ✅ (large Linears now 73–79 % HBM)

**What was built.** Two new variants in `decode_gemv_ext.cu`:
- `gemv_w4a4_v4` — warp-per-row, **uint4/128-bit** weight loads, **no** shared
  activation staging (reads xq from L2). Hypothesis: freeing SMEM raises occupancy.
- `decode_gemv_v5` / `launch_v5` — the **winning** design = the v2 shared-staged
  kernel + (a) a **SMEM opt-in** so it runs when K>12k (v2 crashed at 8B down_proj
  K=14336), and (b) **tunable rows/block (warps ∈ {4,8,16,24,32})** so the roof can
  be swept per shape.

**Result (`perf_r2_decode_gemv.py`, M=1, cos vs fp32 twin = 1.000000 everywhere):**

- **v4 LOST on every shape (2–3× slower).** Clean negative result: shared-staging
  of the reused activation beats re-reading it from L2; uint4 did not help. Kept in
  the file, documented, not recommended.
- **v5 wins vs v2** (shipped fixed-8-warp): per-layer sum **1B 1.09×, 3B 1.10×,
  8B 1.23×**, worst-case cosine 1.000000.
- **HBM% roof reached on the large Linears:** 8B gate/up **77–79 %**, 8B down_proj
  **72.9 %** (v5 w24, vs v2 which *crashed* at K=14336 — only v5-w8-fallback 38.8 %
  ran → **1.88×** on the biggest decode Linear), 3B gate/up 53 %, 3B down 45 %. The
  optimal warp count is 16–24 for large N (vs v2's fixed 8).
- **Small-N shapes stay ~4–13 % HBM (k/v/1B-q/o).** These are **launch-latency
  bound at M=1**, not bandwidth-bound (q_proj's whole 4-bit weight is ~1 MB → <1 µs
  at HBM vs ~15 µs fixed launch cost). A faster kernel cannot fix this — the lever
  is **CUDA-graph decode capture + batch-8** (amortize the fixed per-launch cost),
  which is the algorithm/graph path, not the GEMV kernel.

| size | shape highlights (v2 → v5) |
|------|----------------------------|
| 8B | down_proj 38.8 %→**72.9 %** (1.88×), gate/up 74 %→**79 %**, q/o 38 %→44 % |
| 3B | gate/up 48–50 %→53 %, down 39 %→45 %, q/o 23–24 %→27 % |
| 1B | gate/up 39 %→44 %, down 32 %→34 %, q/o 13–14 %→15 % |

**Expected e2e impact (decode).** The large-Linear time drops ~1.1–1.9×; weighted
by the per-layer Linear mix the GEMM-compute part of decode shrinks ~1.1× (1B/3B)
to ~1.23× (8B). But decode at M=1 is dominated by per-token host/launch + attention
(see `docs/decode_throughput_report.md`); the GEMV is only ~3.7 ms of ~15 ms/token.
So v5 moves the GEMV bucket toward its roof (the stated Task-3 goal) but the e2e
decode win still needs graph capture + KV4 (Task 2) + batch to convert HBM% into
tokens/s. **8B down_proj is the standout**: the smem opt-in fixes a crash and gives
1.88× on the single largest decode-GEMV shape.

**Wiring patch for next round (not applied).** In `promix/quantize/mixed_linear.py`
the M=1 decode path currently calls `decode_gemv_fused` (v3, shared-staged, 8 warps,
uint32). Two changes: (1) switch the weight-stream GEMV to the v5 launcher with a
per-shape warp pick — large N (≥4096) → 16–24 warps, small N → 8; (2) ensure the
smem opt-in path is taken for K≥12k (8B down_proj) — v3 already opts in, but if any
call still routes through `decode_gemv_v2` it must move to `decode_gemv_v5`. The
single-launch quant+GEMV (v3) keeps its in-kernel `quant_act_to_shared`; only the
weight-streaming inner loop adopts v5's tunable-warps launch. cos gate unchanged
(1.000000).

---

## Task 2 — KV4 (NVFP4 / mixed KV cache)  ⚠️ big perf prize, accuracy-gated (naive KV4 FAILS)

**What was built.** `perf_r2_kv4.py` — standalone KV4 probe: (a) exact KV-cache byte
model FP16 vs NVFP4, (b) decode KV-read bandwidth (attention streams the full cache
each step), (c) wikitext-2 PPL with K and V fake-quantized (SDPA patched; attention
math stays FP16, only the cache is quantized). No KV kernel was wired into the
forward (the K/V quant accuracy path `promix/quantize/kv_quant.py` is the algorithm
agent's; this is the perf-axis measurement + accuracy screen).

**(a) Memory — NVFP4 KV = 3.56× smaller** (4-bit code 0.5 B/elem + 1 E4M3 scale per
16 = 0.0625 → 0.5625 B/elem vs 2 B). Identical ratio at every size/context:

| | per-token KV (fp16) | ctx 32k, b8: fp16 → NVFP4 |
|---|---|---|
| 1B | 32.8 KB | 8.59 → 2.42 GB |
| 3B | 114.7 KB | 30.06 → 8.46 GB |
| 8B | 131.1 KB | 34.36 → 9.66 GB |

**(b) Bandwidth — 3.56× less KV traffic/step** (HBM ceiling measured 1.46 TB/s;
read-time = bytes/HBM):

| shape | fp16 KV read | NVFP4 KV read | saved/step |
|-------|-------------|---------------|------------|
| 1B ctx8k b1 | 0.184 ms | 0.052 ms | 0.13 ms (3.56×) |
| 1B ctx8k b8 | 1.472 ms | 0.414 ms | 1.06 ms |
| 8B ctx8k b1 | 0.736 ms | 0.207 ms | 0.53 ms |
| 8B ctx8k b8 | 5.887 ms | 1.656 ms | **4.23 ms** |
| 8B ctx32k b1 | 2.944 ms | 0.828 ms | 2.12 ms |

At decode this is pure attention-read savings — large at long context / batch (8B
b8 ctx8k: −4.2 ms/step; the whole FP16 decode is ~15 ms/token, so KV4 is a first-
order decode lever once context grows). It is also what frees the HBM the Task-3
GEMV competes for.

**(c) Accuracy — the gate naive KV4 does NOT clear (Llama-3.2-1B, 40×2048 tok):**

| KV scheme (both K and V) | PPL | Δ vs fp16-KV (12.6343) | gate <0.05 |
|--------------------------|-----|------------------------|-----------|
| NVFP4 symmetric per-16 | 13.3942 | +0.76 | FAIL |
| Hadamard(head_dim)+NVFP4 | 13.5085 | +0.87 | FAIL |
| INT4 asym group-128 | 13.8860 | +1.25 | FAIL |
| Hadamard + mixed MXFP8(1/8)+NVFP4 | n/a | n/a | — (blocked: 1B head_dim=64 → 1/8 high = 8 ch < MXFP8 block-32) |

Uniform 4-bit KV — in every generic recipe tried — blows the 0.05 gate by a wide
margin (+0.76 to +1.25). A generic Hadamard did **not** help (it was slightly
worse), confirming that KV4 accuracy needs the **ResQ-calibrated** per-layer
rotation (U_key_pos @ R2, trained) + mixed-precision high channels — i.e. the
algorithm agent's `kv_quant.py` path with the kv4 configs — not a drop-in KV4.
The ProMix mixed-precision KV recipe also can't be screened on 1B's head_dim=64
(1/8 high = 8 channels is below the MXFP8 block of 32); it needs head_dim≥256 or a
different high-block size, another reason the recipe is non-trivial.

**Verdict / wiring.** KV4 is the single biggest *decode* perf lever (3.56× memory
AND 3.56× per-step KV-read bandwidth, growing with context/batch), but the win is
**contingent on the accuracy path clearing <0.05**, which the naive recipes do not.
The perf-axis recommendation: build the quantized-KV **storage + attention-read
kernel** (store NVFP4 codes + per-16 E4M3 scales; dequant in the attention epilogue)
and gate it against the algorithm agent's rotated+mixed KV recipe, not a uniform
4-bit cache. No forward wiring this round.

---

## Summary — which lever gives the biggest win

- **Biggest prefill win:** Task 1. The 8B fused producer is **bit-identical** (0-byte)
  and 1.03–1.30× faster than the slow chain at M=2048–8192, lifting 8B prefill
  ~1.22→1.31× at M=8192. Ship it (wiring above). Residual: 1-CTA/SM SMEM bound
  caps the gain — a fully HBM-bound producer would save ~3× more.
- **Biggest decode-GEMV win:** Task 3 `decode_gemv_v5` — 8B down_proj **1.88×**
  (fixes a crash) and the large Linears now hit **73–79 % HBM** (the ≥70–80 % goal).
  Small-N shapes remain launch-bound (graph+batch lever, not kernel). Negative
  result: v4 (no-shared/uint4) lost everywhere.
- **Biggest decode win overall (potential):** Task 2 KV4 — **3.56×** KV memory and
  per-step KV-read bandwidth, large at long-context/batch-8 — but **accuracy-gated**
  and naive KV4 fails (+0.76 PPL min). It only pays if the ResQ rotated+mixed KV
  path clears <0.05.
