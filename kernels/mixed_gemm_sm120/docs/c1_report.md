# C1.1 report — SM120 NVFP4 block-scaled GEMM torch extension

Status: **DONE**. NVFP4×NVFP4 → bf16 GEMM wrapping CUTLASS ex79a/79b config, validated
against a PyTorch dequant-matmul oracle at **cosine = 0.999999** on all tested shapes.

## What was built

A PyTorch-callable NVFP4 block-scaled GEMM for SM120 (RTX Pro 6000, Blackwell):

```
nvfp4_gemm(A_packed uint8[M,K/2], SFA uint8[M,K/16],
           B_packed uint8[N,K/2], SFB uint8[N,K/16], M, N, K) -> bf16[M,N]
```

- A: NVFP4 RowMajor (M,K), 2 FP4/byte. B: NVFP4 ColumnMajor (N,K) — our (N,K) weight
  convention. Both AlignmentA/B = 32.
- Block scale: InputSFVectorSize = 16, element type `float_ue4m3_t`.
- Output: bf16 (M,N) RowMajor via a plain `LinearCombination` epilogue (alpha=1,beta=0).

## Files (all under `kernels/mixed_gemm_sm120/`)

- `nvfp4_gemm_ext.cu` — the CUDA/CUTLASS extension. GEMM type copied from example 79a
  (NVFP4×NVFP4 mainloop, `KernelScheduleAuto` cooperative, TileShape 128×128×128,
  ClusterShape 1×1×1), but with a bf16 `CollectiveBuilder` epilogue (no output SF, unlike
  79b which emits FP4+SFD). Accepts external device pointers instead of generating random
  data. Includes a templated device kernel `fill_sf_kernel` that scatters the logical
  [rows,K/16] ue4m3 bytes into the CUTLASS interleaved SF buffer.
- `nvfp4_gemm.py` — `torch.utils.cpp_extension.load()` JIT loader + data generation + oracle
  validation. FP4 E2M1 encoder, 2-per-byte packer, ue4m3 scale encoder, and the
  dequant-matmul oracle.

## Validation output (GPU 0, container luofan-plaquant)

```
  shape M=1024 N=1024 K=1024: cosine=0.999999  rel_err=1.6643e-03
  shape M=2048 N=2048 K=1792: cosine=0.999999  rel_err=1.6584e-03   (LLM q/k/v/o/gate/up K_main)
  shape M=512  N=2048 K=2048: cosine=0.999999  rel_err=1.6562e-03
ALL PASS  (gate: cosine >= 0.999)
```

The ~1.6e-3 relative error is bf16 output rounding (oracle accumulates/stores fp32); cosine
is effectively 1.0.

## Build

JIT `load()` with:
- `extra_include_paths`: cutlass `include`, `tools/util/include`, `examples/common`
- `extra_cuda_cflags`: `-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3`
- `TORCH_CUDA_ARCH_LIST=12.0a` → torch emits `-gencode=arch=compute_120a,code=sm_120a`
  (confirmed in the build log). This is the key to getting the SM120a blockscaled MMA; a
  plain `12.0` would miss the `a` variant.

First build ~2–3 min; subsequent runs use the cached `.so`.

## CUTLASS layout gotchas solved

1. **SF element type is `float_ue4m3_t`** (unsigned E4M3: 4 exp / 3 mantissa / bias 7,
   range [0,448], top bit unused). For a positive value its byte is bit-identical to
   `torch.float8_e4m3fn` with sign=0. So the Python side computes the block scale, casts to
   `torch.float8_e4m3fn`, and `.view(torch.uint8)` — those raw bytes are exactly what the
   kernel decodes. Block scales are always positive (max_abs/6), so this is lossless.

2. **SF interleaved layout — never hand-derived.** Used
   `Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`
   on `make_shape(M,N,K,1)`. Physical buffer size = `size(filter_zeros(layout_SFA))`. The
   layout's coordinate domain is (M,K,L) with the 16-element vectorization carried as a
   stride-0 mode (confirmed by how the host reference `GettBlockScaling` reads `SfA(m,k,l)`
   over the *full* K). Fill is done by a device kernel: one thread per logical (row,k),
   writing `phys[layout(row,k,0)] = logical[row, k/16]`. Writing every k (redundant within a
   block) is layout-agnostic and robust to any internal reordering — correctness confirmed
   by the oracle.

3. **FP4 E2M1 packing** — even k → low nibble, odd k → high nibble; code table
   {0:+0, 1:+.5, 2:+1, 3:+1.5, 4:+2, 5:+3, 6:+4, 7:+6, sign bit = 0x8}. Matched CUTLASS on
   first try (cosine 0.999999), confirming nibble order and code assignment.

4. **bf16 output epilogue** instead of 79b's `LinCombBlockScaleFactor` (which outputs FP4 +
   an output scale factor). Dropping the fusion op and switching ElementD/C to bfloat16_t
   (as in 79a) gives a directly-consumable float output and a simpler oracle comparison.

## Remaining issues / notes

- None blocking. Output is bf16; an fp32 output variant is a one-line ElementD change if a
  downstream wants it.
- The GEMM uses `cutlass::make_cute_packed_stride` for A/B/C/D strides; inputs are assumed
  contiguous (enforced via `.contiguous()`).
- This is the single-precision NVFP4 anchor for Candidate 1/2. MXFP8 and the 2-launch /
  fused mixed-precision paths build on this same SF-fill + packing machinery.
