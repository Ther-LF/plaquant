# C1 baseline report — SM120 MXFP8 GEMM (C1.2) + 2-launch mixed-precision baseline (C1.3)

Status: **DONE**. MXFP8×MXFP8 → bf16 GEMM validated at **cosine = 0.999999**; the
NVFP4+MXFP8 2-launch mixed-precision path validated at **cosine = 0.999999** on all
real LLM shapes; benchmark (2-launch vs fp16 cuBLAS) written to `benchmark.csv`.

Built on C1.1's working NVFP4 extension — same SF-fill machinery
(`Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB` + device scatter kernel), same
build flags, same oracle style.

## Files (all under `kernels/mixed_gemm_sm120/`)

- `mxfp8_gemm_ext.cu` — MXFP8×MXFP8 block-scaled GEMM extension. GEMM type from CUTLASS
  example 79c (MXFP8 mainloop), with **B switched from mxfp6 to mxfp8** so both operands
  are symmetric `cutlass::mx_float8_t<cutlass::float_e4m3_t>`. Differences vs the NVFP4
  ext: 1 byte/element (not 2-packed), AlignmentA/B = 16, SF vector size = 32 (not 16),
  scale element type `float_ue8m0_t`. The SF-fill kernel is the same scatter, parameterized
  by `VEC=32`.
- `mxfp8_gemm.py` — JIT loader + MXFP8 data gen + oracle (mirrors
  `promix.quantize.quant_utils.fake_quantize_mxfp8`: E4M3 grid via `torch.float8_e4m3fn`,
  E8M0 power-of-2 block scale via `_e8m0_round_up`).
- `mixed_gemm.py` — C1.3. Loads BOTH extensions, defines `mixed_gemm_2launch()` (NVFP4
  low-segment GEMM + MXFP8 high-segment GEMM + outer `s_x`/`s_w` scale → fp16), the full
  mixed-precision oracle, correctness validation, and the 2-launch-vs-cuBLAS benchmark.
- `benchmark.csv` — 72 rows (`shape, M, path, latency_ms`).

## MXFP8 standalone (C1.2) cosine

```
  M=1024 N=1024 K=1024: cosine=0.999999  rel_err=1.66e-03
  M=2048 N=2048 K=256 : cosine=0.999999  rel_err=1.66e-03   (LLM high-segment shape)
  M=512  N=2048 K=384 : cosine=0.999999  rel_err=1.66e-03   (3B high-segment shape)
ALL PASS (gate: cosine >= 0.999)
```

## 2-launch mixed (C1.3) cosine

```
  q_proj_1B    M=16   N=2048 Klow=1792 Khigh=256 : cosine=0.999999
  q_proj_1B    M=2048 N=2048 Klow=1792 Khigh=256 : cosine=0.999999
  down_proj_1B M=16   N=2048 Klow=7168 Khigh=1024: cosine=0.999999
  down_proj_1B M=2048 N=2048 Klow=7168 Khigh=1024: cosine=0.999999
  q_proj_3B    M=16   N=3072 Klow=2688 Khigh=384 : cosine=0.999999
  q_proj_3B    M=2048 N=3072 Klow=2688 Khigh=384 : cosine=0.999999
ALL PASS
```

The outer scales are applied as `out[m,n] = (D_low + D_high)[m,n] * s_x[m] * s_w[n]`,
with `s_x` per-token (M,) and `s_w` per-channel (N,). The oracle dequantizes both
segments (NVFP4 for K_low, MXFP8 for K_high), sums, and applies the same outer scales.
~1.66e-3 rel-err is bf16 output rounding; cosine is effectively 1.0.

## Benchmark: 2-launch vs fp16 cuBLAS (GPU 0, RTX Pro 6000, SM120)

Full table in `benchmark.csv`. Representative rows (latency ms):

| shape        |    M | 2launch_mixed | fp16_cublas | ratio |
|--------------|-----:|--------------:|------------:|------:|
| q_proj_1B    |    1 | 0.1282        | 0.0129      | 9.96x |
| q_proj_1B    | 2048 | 0.1544        | 0.0639      | 2.42x |
| gate_proj_1B | 2048 | 0.5044        | 0.1937      | 2.60x |
| up_proj_1B   | 2048 | 0.5117        | 0.1941      | 2.64x |
| down_proj_1B | 2048 | 0.2657        | 0.2008      | 1.32x |
| q_proj_3B    | 2048 | 0.2384        | 0.1428      | 1.67x |
| down_proj_3B | 2048 | 0.3763        | 0.3229      | 1.17x |

### Reading the baseline

The 2-launch path is currently **slower** than fp16 cuBLAS everywhere. This is the
honest baseline of this wrapper, but the gap is dominated by **wrapper/launch overhead,
not MMA compute**:

- At small M the 2-launch latency is pinned to a ~0.125 ms floor independent of M — this
  is fixed per-call overhead: two separate GEMM launches, each doing its own SF-buffer
  `torch.empty` + **two SF-fill kernels** + workspace alloc + **its own
  `cudaStreamSynchronize`**, plus the Python-level `.float()`/add/scale.
- The overhead-per-call is the thing the fused single-launch kernel (Candidate 2) exists
  to remove: one launch, one accumulator, scales folded into the epilogue, no interleave
  double-fill, no intermediate bf16 round-trip. The ratio closes toward 1.0 as M and the
  total FLOPs grow (down_proj M=2048: 1.17–1.32x), confirming the gap is fixed cost rather
  than arithmetic.

So this baseline establishes: (1) correctness anchor at cosine 0.999999 for the exact
mixed-precision math the model needs, and (2) a latency table Candidate 2 must beat. The
beat target is not just "< this 2-launch number" but ideally approaching fp16 cuBLAS,
since removing the double-sync/double-SF-fill is low-hanging.

## Build (same as C1.1)

JIT `load()`, `TORCH_CUDA_ARCH_LIST=12.0a` → `-gencode compute_120a,sm_120a`, nvcc flags
`-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3`, CUTLASS includes
`include`, `tools/util/include`, `examples/common`.

## Gotchas solved (new vs C1.1)

1. **MXFP8 element = `float_e4m3_t`, 1 byte, not packed.** `A` is uint8[M,K] of raw E4M3
   bytes (vs NVFP4's uint8[M,K/2]). CUTLASS `float_e4m3_t` decodes bit-identically to
   `torch.float8_e4m3fn` (both OCP E4M3-FN), so Python quantizes with the torch dtype and
   `.view(torch.uint8)`.
2. **E8M0 block scale.** SF element is `float_ue8m0_t` (8-bit biased exponent, value =
   `2**(byte-127)`, no mantissa/sign). The block scale from `fake_quantize_mxfp8` is already
   a power of two (`_e8m0_round_up`), so the SF byte is just `round(log2(scale)) + 127`,
   clamped to [0,254]. This is exact — oracle and kernel decode to identical values.
3. **SF vector size = 32** (vs 16 for NVFP4). The CUTLASS `Sm1xxBlkScaledConfig` picks the
   vec size from the element type automatically; the only wrapper change is `VEC=32` in the
   scatter fill (logical SFA is (M,K/32)).
4. **B as mx_float8** — example 79c ships mxfp8×**mxfp6**; set both operands to
   `mx_float8_t<float_e4m3_t>` (AlignmentB 128→16) for a symmetric MXFP8 GEMM.

## Notes / remaining

- The 2-launch combine (NVFP4 + MXFP8 + outer scale) is done at the Python level across the
  two separate extensions — this *is* the "two launches + add" baseline by construction.
- Benchmark includes the wrapper's SF-fill + per-call sync overhead; a production 2-launch
  path would precompute interleaved SF once and use a single sync. The fused kernel removes
  all of it. Flagged in the table reading above.
- M=1 works through the 128×128×128 tile (padded); no can_implement failures on any shape.
