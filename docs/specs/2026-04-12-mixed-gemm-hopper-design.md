# Design: ResQ Mixed-Precision GEMM Optimization for Hopper

**Date**: 2026-04-12
**Goal**: Fuse ResQ's dual INT4/INT8 GEMM calls into a single mixed-precision GEMM kernel on Hopper (SM90), then integrate into ResQ inference pipeline.

---

## Problem Statement

ResQ currently performs two separate GEMM calls for each linear layer:
1. `Q_4bit(X·U_l) × Q_4bit(U_l^T·W)` — INT4×INT4 GEMM (7/8 of K dimension)
2. `Q_8bit(X·U_h) × Q_8bit(U_h^T·W)` — INT8×INT8 GEMM (1/8 of K dimension)

This is inefficient due to double kernel launch overhead, double epilogue memory access, and inability to share accumulators.

## Solution

A single CUDA kernel that executes both INT4 and INT8 Tensor Core MMA instructions, accumulating results to the same register file. Exploit the fact that INT4 `m16n8k64` and INT8 `m16n8k32` MMA instructions produce identically-laid-out 8×8 FP32/INT32 accumulator tiles.

---

## Implementation Phases

### Phase 1: Test Infrastructure

**Goal**: Collect all mixed-precision GEMM call sites from ResQ, extract their shapes, and build a comprehensive test suite.

**Tasks**:
1. Instrument ResQ's `ActQuantWrapper.forward()` and `KernelMixedGemm` equivalent paths to log:
   - Input shapes (M, N, K_high, K_low)
   - Data types (INT4, INT8)
   - Scale factors
   - Reference outputs (for correctness verification)
2. Create test shapes for Llama-3.2-1B-Instruct (hidden=2048, ffn=8192, heads=32, head_dim=64):
   - q/k/v_proj: M=batch×seqlen, N=2048, K_INT8=256, K_INT4=1792
   - o_proj: M=batch×seqlen, N=2048, K_INT8=256, K_INT4=1792
   - gate/up_proj: M=batch×seqlen, N=8192, K_INT8=256, K_INT4=1792
   - down_proj: M=batch×seqlen, N=2048, K_INT8=1024, K_INT4=7168
3. Build Python test script with:
   - Correctness test: mixed GEMM output vs. two separate GEMM + add (bitwise comparison)
   - Performance test: CUDA event timing, comparison against baseline (two GEMMs)
   - Batch sizes: 1, 4, 16, 64, 128, 256, 512, 1024, 2048

### Phase 2: CUTLASS Hopper GEMM Study

**Goal**: Understand CUTLASS 3.x SM90 GEMM architecture (CuTe, TMA, wgmma) to identify modification points.

**Tasks**:
1. Read CUTLASS 3.x SM90 integer GEMM examples
2. Understand the producer-consumer mainloop (TMA producer → wgmma consumer)
3. Identify:
   - How to switch TMA descriptors between INT4 and INT8 data
   - wgmma instruction availability for INT4 operands on SM90
   - Shared memory swizzle patterns for different precisions
   - Accumulator register layout compatibility between INT4 and INT8 wgmma

### Phase 3: Hopper Mixed-Precision GEMM Kernel

**Goal**: Implement the fused kernel based on CUTLASS 3.x SM90 templates.

**Tasks**:
1. Define MixedGemmKernel for SM90 with two MMA configurations
2. Implement TMA-based data loading for both precision streams
3. Implement wgmma mainloop that alternates between INT4 and INT8 tiles
4. Port the dynamic tile scheduling algorithm (`get_next_low_tile`) to SM90
5. Handle epilogue (shared accumulator → output)
6. Validate against Phase 1 test suite: **exact numerical match required**
7. Performance benchmarking against two-GEMM baseline

### Phase 4: ResQ Integration

**Goal**: Replace ResQ's two separate GEMM calls with the fused kernel.

**Tasks**:
1. Create Python/C++ binding for the mixed GEMM kernel
2. Modify `ActQuantWrapper.forward()` to call fused kernel
3. Modify KV cache quantization path if applicable
4. End-to-end validation: WikiText PPL must match original ResQ
5. End-to-end performance: measure token throughput improvement

### Phase 5: Pure CUDA Optimization (Future)

**Goal**: Maximum performance via hand-tuned CUDA.

**Tasks**:
1. Compile CUTLASS kernel, inspect .i (preprocessed) output
2. Rewrite critical path in raw CUDA/PTX
3. Use ncu to identify bottlenecks (memory bandwidth, compute utilization, occupancy)
4. Optimize register allocation, shared memory usage, warp scheduling

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Correctness | Bitwise match with two-GEMM baseline |
| WikiText PPL | Identical to original ResQ |
| Kernel speedup (decode, batch=1) | >1.3x over two separate GEMMs |
| Kernel speedup (prefill, batch=256) | >1.1x over two separate GEMMs |
| Integration overhead | <5% vs. kernel-only speedup |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| INT4 wgmma not supported on SM90 | Fall back to mma.sync for INT4 portion, use wgmma only for INT8 |
| TMA descriptor switching overhead | Pre-create both descriptors, measure overhead, consider single-descriptor approach |
| Numerical mismatch from accumulation order | Test with sorted tile order matching baseline |
| H20 GPU (Hopper variant) specific issues | Test on both H20 and H100 if available |
