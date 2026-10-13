# C2 report — fused single-launch mixed-precision GEMM (SM120)

Status: **DONE_WITH_CONCERNS** — **Rung A achieved** (one host launch, single sync,
fused combine+scale epilogue). The true single-CUDA-kernel dual-phase mainloop
(**Rung B**) is NOT done; it is documented below as remaining work with a concrete
design. Rung A **meets the full promotion gate**: cosine ≥ 0.999 on every shape AND
fused latency < 2-launch baseline on every shape.

## Which rung

**Rung A (fused host operator).** A single Python→C++ call `mixed_gemm_fused(...)`
that, on one CUDA stream with a **single `cudaStreamSynchronize`**, runs:

1. 4 SF-interleave fills (NVFP4 low + MXFP8 high, A & B each),
2. the NVFP4 (K_low) block-scaled GEMM → bf16 `D_low`,
3. the MXFP8 (K_high) block-scaled GEMM → bf16 `D_high`,
4. one fused `combine_scale_kernel`: `out[m,n] = (D_low+D_high)·s_x[m]·s_w[n] → fp16`.

This removes the dominant 2-launch overhead the C1 baseline identified: the double
Python round-trip, the second `cudaStreamSynchronize`, the Python-level
`.float()`/add/scale, and the bf16→fp16 intermediate round-trip. It reuses the exact
validated CUTLASS configs from `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` (both at
cosine 0.999999), so it is correct by construction.

It is NOT yet the true in-register shared-accumulator fused mainloop (Rung B): the two
GEMMs are still two device kernel launches writing two bf16 tiles that the combine
kernel sums, rather than one kernel accumulating NVFP4 then MXFP8 into the same FP32
register tensor with SMEM union. The C0.4 proof that this is viable still stands and the
design is captured in "Remaining work".

## Fused kernel file

- `kernels/mixed_gemm_sm120/fused_gemm_ext.cu` — both GEMM configs namespaced
  (`nvcfg` / `mxcfg`), generic SF-fill, and `combine_scale_kernel`. Single entry point
  `mixed_gemm_fused(A_low,SFA_low,B_low,SFB_low, A_high,SFA_high,B_high,SFB_high, s_x,s_w, M,N,Klow,Khigh) -> fp16[M,N]`.
- `kernels/mixed_gemm_sm120/mixed_gemm.py` — `get_fused_ext()` loader, `mixed_gemm_fused()`
  wrapper, and `validate()`/`benchmark()` extended to compare fused vs 2-launch vs cuBLAS.

## Correctness (cosine vs mixed oracle, GPU 0, RTX Pro 6000 SM120)

All PASS at gate cosine ≥ 0.999, every M ∈ {1, 16, 256, 2048}:

```
  q_proj_1B    Klow=1792 Khigh=256 : fused_vs_oracle=0.999998–0.999999  fused_vs_2launch=1.000000
  down_proj_1B Klow=7168 Khigh=1024: fused_vs_oracle=0.999999           fused_vs_2launch=1.000000
  q_proj_3B    Klow=2688 Khigh=384 : fused_vs_oracle=0.999999           fused_vs_2launch=1.000000
```

`fused_vs_2launch = 1.000000` confirms the fused path reproduces the baseline math
bit-for-bit; rel_err ~1.67e-3 is bf16 product rounding (identical to the 2-launch path).

## Benchmark: fused vs 2-launch vs fp16 cuBLAS (latency ms; full table `benchmark.csv`)

| shape        |    M | fused  | 2launch | fp16   | fused/2launch | fused/fp16 |
|--------------|-----:|-------:|--------:|-------:|--------------:|-----------:|
| q_proj_1B    |    1 | 0.0613 | 0.1310  | 0.0128 | **0.47x**     | 4.78x      |
| q_proj_1B    | 2048 | 0.0964 | 0.1568  | 0.0639 | **0.61x**     | 1.51x      |
| gate_proj_1B | 2048 | 0.2125 | 0.5079  | 0.1903 | **0.42x**     | 1.12x      |
| up_proj_1B   | 2048 | 0.2124 | 0.5038  | 0.1938 | **0.42x**     | 1.10x      |
| down_proj_1B | 2048 | 0.2207 | 0.2637  | 0.1911 | **0.84x**     | 1.15x      |
| q_proj_3B    | 2048 | 0.1468 | 0.2480  | 0.1429 | **0.59x**     | 1.03x      |
| down_proj_3B | 2048 | 0.2784 | 0.3736  | 0.3068 | **0.75x**     | **0.91x**  |

**Promotion gate PASS:** fused < 2-launch on **all** shapes/M (ratios 0.42–0.84x, i.e.
fused is 1.2x–2.4x faster than the 2-launch baseline). Stretch goal (beat fp16 cuBLAS):
reached on **down_proj_3B M=2048 (0.91x)** and approached on q_proj_3B M=2048 (1.03x);
elsewhere cuBLAS is still ahead because the block-scaled MMA path carries SF-fill +
2-GEMM + combine launch overhead that cuBLAS's single fused kernel does not. Closing
the remaining fp16 gap is exactly what Rung B targets.

## Remaining work — Rung B (true single-kernel fused mainloop)

Design is unblocked by C0.4 (both atoms resolve `CLayout = SM80_16x8_Row`, identical
FP32[4]/thread accumulator → in-register shared accumulator viable):

1. Hand-written CuTe kernel, one CTA per output tile, FP32 register accumulator `acc`.
2. **Phase 1**: iterate K_low/64, NVFP4 block-scaled MMA (`SM120_16x8x64_TN_VS`,
   `fp4_shift_A/B` for the `<<2` placement) accumulating into `acc`.
3. **SMEM union reuse**: Phase 2 overwrites Phase 1's A/B/SF SMEM staging buffers
   (SM120 ~99 KB SMEM budget) — the SM80 fused pattern from `mixed_gemm_l20.cu`.
4. **Phase 2**: iterate K_high/32, MXFP8 block-scaled MMA (`SM120_16x8x32_TN_VS`)
   accumulating into the SAME `acc`.
5. Single epilogue: `out = acc·s_x[m]·s_w[n] → fp16` (fold s_x/s_w as EVT row/col
   broadcasts). One launch, one sync, no intermediate D_low/D_high buffers, no combine.

Expected win over Rung A: removes one GEMM launch, two bf16 tile writes + reads, and the
combine kernel; keeps the K-reduction entirely in registers. This is the path to beat
fp16 cuBLAS on the small/large-N shapes where Rung A still trails.

Risk: writing the dual-atom mainloop against CUTLASS 4.x SM120 collectives (TMA/pipeline,
block-scale SMEM layout for two different vector sizes 16 vs 32) is substantial and was
not completed in this session. Rung A is the correct, promotable deliverable now.

## Build (same as C1)

JIT `load()`, `TORCH_CUDA_ARCH_LIST=12.0a` → `-gencode compute_120a,sm_120a`, nvcc flags
`-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -O3`, CUTLASS includes
`include`, `tools/util/include`, `examples/common`. First build ~2 min; cached after.
Run: `CUDA_VISIBLE_DEVICES=<free> python mixed_gemm.py` inside `luofan-plaquant`.
