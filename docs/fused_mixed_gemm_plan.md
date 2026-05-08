# Fused Mixed-Precision GEMM Kernel — Implementation Plan (方案C)

## Goal
Single kernel: INT4(main) + INT8(high) → shared INT32 accumulator → FP16 output

## Architecture
```
Single kernel launch:
┌─ TMA load X_main (INT4 packed) → SMEM_A_main
├─ TMA load X_high (UINT8)       → SMEM_A_high
├─ TMA load W_main (INT4 packed) → SMEM_B_main
├─ TMA load W_high (INT8)        → SMEM_B_high
│
├─ Mainloop (K_main tiles):
│   ├─ SMEM → RF: load INT4 packed data
│   ├─ INT4 → INT8 convert (prmt LUT, 3 instructions per 8 values)
│   └─ WGMMA S8S8: acc_int32 += A_int8 @ B_int8
│
├─ Mainloop (K_high tiles):
│   ├─ SMEM → RF: load UINT8 data (no conversion needed)
│   └─ WGMMA U8S8: acc_int32 += A_uint8 @ B_int8  (SAME accumulator!)
│
└─ Epilogue (EVT):
    D = s_x[m] * s_w[n] * (float(acc) - zero_m * colsum_main - zero_h * colsum_high)
```

## Key Innovation vs Baseline
- **Baseline**: 2 separate GEMMs (2 kernel launches) + 1 add = 3 launches, 3x global memory traffic
- **Fused**: 1 kernel launch, 1x write to global memory, shared accumulator registers

## Performance Advantage Source
1. **Bandwidth**: INT4 packed storage (half the bytes vs INT8 expanded)
2. **Compute**: Both portions accumulate into same INT32 registers (no intermediate write-back)
3. **Latency**: Single kernel launch (save ~5μs launch overhead)
4. **Memory**: No intermediate (M,N) FP16 buffer for Y_main

## Implementation Steps

### Step 1: Fork CUTLASS mixed-input mainloop
- Base: `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- Modify to support **dual narrow-type**: both A and B are INT4, both upcast to INT8
- The standard CUTLASS assumes one side is narrow (register path), one side is wide (SMEM path)
- We need: both sides load INT4 from SMEM, both upcast to INT8 in registers before WGMMA

### Step 2: Two-phase K-loop
- Phase 1: process K_main tiles (INT4 activation × INT4 weight → INT32 accumulator)
- Phase 2: process K_high tiles (UINT8 activation × INT8 weight → same INT32 accumulator)
- Both phases share the same `acc_int32` register file

### Step 3: INT4→INT8 conversion using prmt LUT
- Use vLLM Machete's `lut_4bit_to_8bit_convert` pattern
- 3 `prmt.b32` instructions convert 4 INT4 values to 4 INT8 values
- Applied to both A and B after loading from SMEM to registers

### Step 4: Weight prepacking
- Offline: reorder INT4 weight into interleaved layout (like Machete)
- Ensures 128-bit aligned vectorized SMEM loads
- Store alongside scale/colsum metadata

### Step 5: Epilogue with combined bias correction
- Modified EVT: accounts for both main and high zero points
- `D = s_x_m * s_w_m * (acc_main_part) + s_x_h * s_w_h * (acc_high_part)`
- But since they share one accumulator, need separate colsum corrections:
  `D = combined_scale * (acc - zero_m_correction - zero_h_correction)`

### Step 6: TMA descriptor setup
- Need 4 TMA descriptors: A_main, B_main, A_high, B_high
- Tile K dimension differently for main (INT4) vs high (INT8)

## Reference Code
- vLLM Machete: `third_party/vllm/csrc/quantization/machete/`
- CUTLASS mixed-input: `third_party/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- INT4→INT8 converter: `third_party/vllm/csrc/cutlass_extensions/vllm_numeric_conversion.cuh`

## Testing
- Compare against baseline (2 separate GEMMs + add)
- Correctness: match output_real_quant ground truth
- Performance target: >50% H20 INT8 TOPS, with 2x bandwidth advantage over baseline

## Timeline Estimate
- Step 1-3: Core mainloop modification (~2 weeks)
- Step 4: Prepacking (~1 week)
- Step 5-6: Epilogue + TMA setup (~1 week)
- Testing + tuning: (~1 week)
