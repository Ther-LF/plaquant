# B1 report — TRUE single-CUDA-kernel fused mixed-precision GEMM (SM120)

Status: **DONE (correctness gate PASS)** — `b1_fused` is a single CUDA kernel that runs
Phase 1 (NVFP4 block-scaled MMA over K_low) and Phase 2 (MXFP8 block-scaled MMA over
K_high) into ONE FP32 register accumulator, applies `s_x[m]*s_w[n]`, and stores fp16.
**cosine = 1.000000 (rel ≈ 2.1e-4, pure fp16 output rounding) on every gate shape and M.**
This round was correctness-only; the kernel is deliberately naive (one warp per 16x8
output tile, no TMA / ldmatrix / pipelining) and is 1.0–13x slower than `rungB_presf`
(honest numbers below).

Hardware: RTX Pro 6000 Blackwell (SM120, `sm_120a`), GPU 6, container `luofan-plaquant`.
Files: `b1_fused_ext.cu` (kernel + `b1_nvfp4_only` isolation entry), `b1_fused.py`
(loader, step-1 validator, fused validator, benchmark appender).

---

## Root cause of the inherited cosine≈0 bug (with isolation evidence)

**Bug:** `copy(thr_mma.partition_A(sA), tCrA)` silently did an elementwise **VALUE
conversion**, not a bit copy. `MMA_Traits<SM120_16x8x64_TN_VS>::ValTypeA/B = uint4_t`
("the MMA accepts 4-bit inputs regardless of the types for A and B"), so
`partition_fragment_A/B` creates **uint4_t** fragments — but the smem views had been
recast to `float_e2m1_t`. `cute::copy` between `float_e2m1_t` and `uint4_t` converts
each element float→unsigned-int: **truncation toward zero + clamp of negatives to 0**.
Every nibble was corrupted: e2m1 code c was replaced by the bit pattern
`uint4(trunc(e2m1_value(c)))`, then decoded by the MMA as e2m1 again.

**Fix** (one line + comment): view A/B smem as the trait's value type so the copy is
bit-identical — `using ElemFP4 = cute::uint4_t;` (and `uint8_t` for the MXFP8 phase).
SF views were already `float_ue4m3_t`/`float_ue8m0_t` (== ValTypeSF), hence exact.

### Isolation trail (each step → conclusion)

1. Reproduced: cosine 0.017–0.049 on all shapes; `|D|/|ref| ≈ 1.77·sqrt(K/1792)` —
   D grows ∝K (coherent positive bias) vs ref ∝sqrt(K). → some per-product positive
   bias (later explained: negative codes clamped to 0 kill all negative products).
2. Single-atom one-hot probes (M=16,N=8,K=64, SF=1.0=0x38): `A[m0,k0]=1.0, B[:,k0]=1.0`
   → output lands at EXACTLY the right (row, col) for k0 ∈ {0,1,17,40,63} and m0 ∈
   {0,1,3,5,9}, but value = **0.25** instead of 1.0. → A/B (m,k) mapping, SF k-blocking
   and C-store (SM80_16x8_Row: thread t → rows {t/4, t/4+8}, cols {2(t%4), 2(t%4)+1},
   vals (v%2 → col, v/2 → row+8)) are ALL CORRECT; pure value distortion.
3. SF probes: SFA=0x40 (2.0) → 0.5 = 2.0×0.25; SFA=0x44 (3.0) → 0.75 (mantissa
   survives); SFA=SFB=0x30 (0.5) → 0.0625 = 0.5·0.5·0.25. SF path exactly linear →
   **SF handling verbatim-correct**, distortion is in the data nibbles.
4. Full decode tables (A-side and B-side, k0 even/odd/reg-boundary): observed
   d(c)·d(2) for c=0..15 = [0, 0, .25, .25, .5, .75, 1.0, 2.0, 0×8]. Unique exact fit:
   **d(c) = e2m1_decode(uint4(trunc(e2m1_value(c))))** — e.g. c=2 (1.0)→bits 1→0.5,
   c=5 (3.0)→bits 3→1.5, c=7 (6.0)→bits 6→4.0, c=1 (0.5)→0, c≥8 (negative)→0.
   Identical both sides, k-position independent → float→uint conversion in the copy.
5. Post-fix step 1 (NVFP4-only vs fp32 oracle): **rel = 0.000e+00 (bit-exact)** on all
   6 shapes including M=1 (padding) and K=7168.

### Failed hypotheses log (ruled out, with evidence)

- SF fragment ordering / SFALayout vs SFBLayout mix-up — ruled out by probe 3 (SF
  exactly linear, correct per-side). The b1_* SF helpers are verbatim-identical to
  `sm120_blockscaled_mma_tma.hpp` and degenerate correctly for the raw-atom TiledMMA.
- C-store (accumulator→gmem) mapping — ruled out by probe 2 (one-hots land exactly
  right; decoded SM80_16x8_Row by hand and it matches the store loop).
- fp4_shift_A/B needed for the mxf4nvf4 atom — no: packed-nibble kind; generic no-op
  overload is correct (CUTLASS shifts only for e2m1 inside the F8F6F4 8-bit container).
- ue4m3 bias mismatch (bias 8 vs e4m3fn bias 7) — ruled out: working nvfp4_gemm_ext
  passes e4m3fn bytes with small rel_err; probe 3 confirms byte 0x38 scales by exactly 1.
- K-index permutation between A and B fragments — ruled out: a π≠σ mismatch would zero
  the one-hot product; all one-hots were nonzero at the right cell.

---

## Full B1 kernel (what landed)

- `b1_run_phase<ElemData, ElemSF, KATOM, VEC>(tiled_mma, …, accum)` — templated phase
  runner: stages one KATOM-wide chunk (32 bytes/row for both formats) through
  non-swizzled smem (zero-padded for M<16), partitions with the raw atom TiledMMA,
  plain bit-copies smem→reg, `cute::gemm(tiled_mma, zip(A,SFA), zip(B,SFB), accum)`.
- `b1_fused_kernel`: `clear(accum)` → phase NVFP4 (K_NV=64, SFVec16, ue4m3) → phase
  MXFP8 (K_MX=32, SFVec32, ue8m0) into the SAME `accum` → `*s_x[m]*s_w[n]` → fp16.
  `static_assert(is_same<CLayout_NV, CLayout_MX>)` guards the shared accumulator.
  Grid (ceil(M/16), ceil(N/8)), 32 threads/block; smem < 1 KB.
- Per-phase dual-TiledMMA C partitions are structurally identical by the static_assert
  (both SM80_16x8_Row, FP32[4]/thread), and the fused result validates against the
  mixed oracle (which exercises both phases plus the outer scales) at rel ≈ 2e-4.

## Correctness (gate: cosine ≥ 0.999 vs mixed oracle, GPU 6)

```
q_proj_1B    N=2048 Klow=1792 Khigh=256 : M=1/16/256/2048  cosine=1.000000  rel≈2.1e-4
down_proj_1B N=2048 Klow=7168 Khigh=1024: M=1/16/256/2048  cosine=1.000000  rel≈2.1e-4
q_proj_3B    N=3072 Klow=2688 Khigh=384 : M=1/16/256/2048  cosine=1.000000  rel≈2.1e-4
```
ALL PASS. rel ≈ 2.1e-4 is fp16 output rounding only — an order of magnitude tighter
than rungB's 2.3e-3 (no bf16 D_low round-trip, single fp32 accumulation chain).
Step-1 NVFP4-only entry is bit-exact (rel = 0) vs the fp32 dequant oracle.

## Honest benchmark (b1_single rows appended to benchmark.csv; 36 rows)

Same `_time_ms` harness as every other benchmark.csv path. Same-session, same-GPU
head-to-head (fresh rungB_presf/2launch timings, not the round-2 numbers):

| shape        |    M | b1_single | rungB_presf | 2launch | b1/rungB |
|--------------|-----:|----------:|------------:|--------:|---------:|
| q_proj_1B    |    1 |   0.0575  |     0.0509  | 0.1303  |   1.13x  |
| q_proj_1B    |   16 |   0.0742  |     0.0505  | 0.1275  |   1.47x  |
| q_proj_1B    | 2048 |   0.5951  |     0.0772  | 0.1479  |   7.71x  |
| k_proj_1B    |    1 |   0.0549  |     0.0491  | 0.1247  |   1.12x  |
| k_proj_1B    | 2048 |   0.1712  |     0.0572  | 0.1270  |   2.99x  |
| down_proj_1B |    1 |   0.1806  |     0.0760  | 0.1686  |   2.38x  |
| down_proj_1B | 2048 |   2.4642  |     0.1886  | 0.2850  |  13.07x  |
| q_proj_3B    |    1 |   0.0882  |     0.0688  | 0.1654  |   1.28x  |
| gate_proj_1B |    1 |   0.0558  |     0.0553  | 0.1489  |   1.01x  |
| gate_proj_1B | 2048 |   2.0923  |     0.1583  | 0.4849  |  13.22x  |

**Naive B1 is slower than rungB_presf everywhere (1.0–13x), as expected** for a
one-warp-per-tile kernel with no copy pipelining. Notable: at decode M=1 it is within
1.0–1.3x of rungB_presf on small-K shapes (gate_proj M=1 ties at 1.01x) and already
beats the 2-launch baseline at M=1 on q/k/v/o/gate/up/q3B — the single-launch,
no-gmem-round-trip structure shows through exactly where launch overhead dominates.
Cost grows ∝ M·N tiles (each 16x8 tile re-reads its full B rows): worst at prefill.

## First optimization targets (next round)

1. **CTA tiling + warp count**: 128x128 CTA tile (8 warps, each 2x4 atom tiles via
   `make_tiled_mma` AtomLayout / permutations), B-tile reuse across the M dimension —
   kills the dominant redundant-B-traffic term (M·N/128 fewer B reads).
2. **cp.async double-buffered smem staging** (SM80-style) for A/B/SF; the 99 KB SMEM
   budget allows 3+ stages at 128x128x64.
3. **ldmatrix / vectorized smem→reg**: replace the per-element generic copies (the
   current copies are subbyte scalar loops — likely >50% of inner-loop time).
4. **Interleave the two phases' K loops** (NV then MX chunks back-to-back per CTA) is
   already sequential-in-registers; once pipelined, overlap phase-2 loads with phase-1
   tail MMAs.
5. Only then: epilogue vectorization (half2 stores) and tile-N dispatch by M.

## Repro

```
docker exec -w /home/luofan/plaquant/kernels/mixed_gemm_sm120 \
  -e PYTHONPATH=/home/luofan/plaquant -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e CUDA_VISIBLE_DEVICES=<free> luofan-plaquant python b1_fused.py
```
Build flags unchanged (`-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3`,
CUTLASS includes, `compute_120a/sm_120a`).
