# Mixed-Precision FlashAttention Kernel Design Spec

> Date: 2026-04-20
> Target: Hopper SM90 (H20/H100), CUTLASS 3.x

## 1. Overview

A fused FlashAttention-3-style kernel where **all GEMMs use INT8/INT4 Tensor Cores**, with in-register accumulation across precision types.

**Baseline (ResQ current):** Separate kernel launches for each GEMM:
- Q·K^T: INT8 GEMM + INT4 GEMM → dequant → FP16 S (2 kernel launches + elementwise)
- Softmax: FP32 (separate kernel)
- P·V: FP16 GEMM (1 kernel launch)
- o_proj: grouped INT8 GEMM + grouped INT4 GEMM → dequant (2 kernel launches)

**Our kernel:** All fused into one persistent kernel, eliminating HBM round trips for S, P, and intermediate O.

## 2. Data Layout

Each head's d_head is split along K dimension into two segments:

```
[0 : k_high]      → INT8  (high-variance PCA components)
[k_high : d_head] → INT4  (low-variance PCA components, packed 2/byte)
```

**Configurable parameters:** `k_high`, `k_low` (not hardcoded — adjusted per model/layer).

**Padding:** Both segments padded to TC tile alignment:
- `pad_k_high = align_up(k_high, K_TILE_INT8)` (typically 128)
- `pad_k_low = align_up(k_low, K_TILE_INT4)` (typically 128)

**Benchmark default:** `k_high=128, k_low=128` (50/50 split, full TC utilization).

**Tensor shapes (per head, per batch):**
```
Q_int8:  (Lq, pad_k_high)           INT8
Q_int4:  (Lq, pad_k_low/2)          packed INT4 (2/byte)
K_int8:  (Lkv, pad_k_high)          INT8
K_int4:  (Lkv, pad_k_low/2)         packed INT4
V_fp16:  (Lkv, d_head)              FP16 (dequant from KV4 cache)
W_o_int8:(d_head, pad_k_high)      INT8 (grouped per head)
W_o_int4:(d_head, pad_k_low/2)     packed INT4
```

**Scales (per-tensor or per-block):**
```
scale_q8, scale_q4, scale_k8, scale_k4
scale_v8, scale_v4 (for V dequant, applied before kernel)
scale_wo8, scale_wo4
```

## 3. GEMM Precision Assignment

| GEMM | A | B | Accumulator | TC Type | WGMMA? |
|------|---|---|-------------|---------|--------|
| **Q·K^T** (low) | Q_int4 | K_int4^T | INT32 | INT4 | Yes (`wgmma.mma_async.s4`) |
| **Q·K^T** (high) | Q_int8 | K_int8^T | INT32 (accum) | INT8 | Yes (`wgmma.mma_async.s8`) |
| **P·V** | P_fp16 | V_fp16 | FP32 | FP16 | Yes |
| **O·W_o** (low) | O_int4 | W_o_int4^T | INT32 | INT4 | Yes (grouped) |
| **O·W_o** (high) | O_int8 | W_o_int8^T | INT32 (accum) | INT8 | Yes (grouped) |

P·V stays FP16 (aligned with ResQ) — softmax output P is FP16 in [0,1], re-quantizing would lose precision.

## 4. Key Insight: In-Register Accumulation

INT4 and INT8 WGMMA on SM90 produce accumulator fragments with **identical thread-to-element mapping for the same M×N tile** (the K dimension differs but doesn't affect fragment layout).

This means:
```cpp
FragmentC_i32 S_acc;  // Same layout for both precisions
INT4_WGMMA(S_acc, Q_int4, K_int4);   // S_acc = Q4 @ K4^T
INT8_WGMMA(S_acc, Q_int8, K_int8);   // S_acc += Q8 @ K8^T  (in-place!)
// One dequant: S_fp32 = S_acc * scale
```

No data shuffling, no shared memory, zero overhead.

## 5. Kernel Architecture (FA3-Style)

### 5.1 Warp Specialization

```
CTA = 8 warps (256 threads)
  Producer (Warp 0-1):     TMA loads, setmaxnreg ~32 regs
  Consumer WG1 (Warp 2-4): WGMMA + softmax, setmaxnreg ~232 regs
  Consumer WG2 (Warp 5-7): WGMMA + softmax, setmaxnreg ~232 regs
```

### 5.2 Shared Memory Layout (Circular Buffer, s=2)

```
Q persistent (loaded once per Q tile):
  Q_int8:    Br × pad_k_high        = 128×128 = 16KB
  Q_int4:    Br × pad_k_low/2       = 128×64  =  8KB
  Total Q:   ~24KB

Per stage (×2):
  K_int8:    Bc × pad_k_high        = 128×128 = 16KB
  K_int4:    Bc × pad_k_low/2       = 128×64  =  8KB
  V_fp16:    Bc × d_head            = 128×128 = 32KB
  Per stage: ~56KB

Total SMEM: 24 + 2×56 = ~136KB (fits H100 228KB)
```

### 5.3 Two-Warpgroup Pingpong

```
Timeline:
  WG1: [QK^T(j)] [softmax(j)+PV(j)] [QK^T(j+2)] [softmax(j+2)+PV(j+2)] ...
  WG2:           [QK^T(j+1)]         [softmax(j+1)+PV(j+1)]             ...
```

`bar.sync` ensures WG1's PV and QK^T are scheduled before WG2's GEMMs, allowing WG2's softmax to overlap with WG1's GEMMs.

### 5.4 Intra-Warpgroup 2-Stage Pipeline

Register buffers: `S_cur`, `S_next`, `P_cur`, `P_next`

```
Prologue (j=0):
  1. bar.sync → K_int4[0], K_int8[0] ready
  2. INT4_WGMMA.commit(S_cur, Q_int4, K0_int4)       // async
  3. INT8_WGMMA.commit(S_cur, Q_int8, K0_int8)       // async, in-place accumulate
  4. WGMMA.wait(S_cur)
  5. S_fp32 = dequant(S_cur, scales)
  6. (m, P_cur, l) = online_softmax(S_fp32)
  7. rescale O

Mainloop (1 ≤ j < Tc-1):
  1. bar.sync → K_int4[j], K_int8[j] ready
  2. INT4_WGMMA.commit(S_next, Q_int4, Kj_int4)     // async
  3. INT8_WGMMA.commit(S_next, Q_int8, Kj_int8)     // async, accumulate
  4. bar.sync → V_fp16[j-1] ready
  5. FP16_WGMMA.commit(O, P_cur, V_{j-1})            // async PV GEMM
  6. WGMMA.wait(S_next)
  7. S_fp32 = dequant(S_next, scales)
  8. (m, P_next, l) = online_softmax(S_fp32)          // OVERLAPPED with PV GEMM
  9. WGMMA.wait(O)                                    // wait PV GEMM
  10. rescale O with (m_old, m_new)
  11. Release K_j, V_{j-1} stages
  12. swap(P_cur = P_next)

Epilogue (j = Tc-1):
  1. bar.sync → V_fp16[Tc-1] ready
  2. FP16_WGMMA.commit(O, P_cur, V_{Tc-1})
  3. WGMMA.wait(O), final rescale
```

### 5.5 Performance: Why Mixed-Precision Overhead is Hidden

- INT4 + INT8 WGMMAs run back-to-back (serial on Tensor Cores)
- Extra latency (~1 WGMMA) is **hidden by PV GEMM and softmax**
- Softmax exp is 256× slower than WGMMA → dominant pipeline stage
- Net overhead vs single-precision FA3: **~10-15%** (from extra SMEM pressure, not compute)

## 6. Fused o_proj (Optional Phase)

After attention, O_i (Br, d_head) is still in registers/SMEM.

```
O_int4 = quantize_i4(O_i[:, k_high:])     // quantize low-variance part
O_int8 = quantize_i8(O_i[:, :k_high])     // quantize high-variance part

// Grouped GEMM (one kernel launch for all heads)
INT4_WGMMA(Out, O_int4, W_o_int4[h])      // async
INT8_WGMMA(Out, O_int8, W_o_int8[h])      // async, in-place accumulate
Out_fp16 = dequant(Out, scales)            // write to HBM
```

## 7. Tile Sizes

| Parameter | Value | Notes |
|-----------|-------|-------|
| Br (Q tile) | 128 | Along sequence length |
| Bc (KV tile) | 128 | Along sequence length |
| INT8 WGMMA | m64×n128×k32 | SM90 native |
| INT4 WGMMA | m64×n128×k128 | SM90 native |
| FP16 WGMMA (PV) | m64×n128×k64 | SM90 native |

## 8. Benchmark Design

### Test Configurations

```python
configs = [
    # (B, H, Lq, Lkv, d_head, k_high, k_low) — name
    (1, 32,  1024, 1024, 128, 128, 128),  # small prefill
    (1, 32,  2048, 2048, 128, 128, 128),  # medium prefill
    (1, 32,  4096, 4096, 128, 128, 128),  # large prefill
    (1, 32,  8192, 8192, 128, 128, 128),  # very long context
    (1, 1,   1,    4096, 128, 128, 128),  # decode (single query)
]
```

### Baselines

| Baseline | Description |
|----------|-------------|
| `fp32_ref` | PyTorch FP32 attention (accuracy reference) |
| `fa3_fp16` | FlashAttention-3 FP16 (performance upper bound) |
| `resq_baseline` | ResQ current: separate INT8+INT4 GEMMs for Q·K^T + softmax + FP16 P·V + grouped o_proj |
| `mixed_fa` | Our fused mixed-precision kernel |

### Metrics

- **Accuracy vs fp32_ref:** cosine similarity, max absolute error, RMSE, SNR(dB)
- **Performance:** latency (ms), TFLOPS, speedup vs resq_baseline and fa3_fp16
- **Memory:** KV cache bandwidth savings vs FP16

## 9. Implementation Plan

### Phase 1: Benchmark Infrastructure
- `bench_flash_attn.py` — generate test tensors, run baselines, collect metrics
- Reference FP32 implementation for accuracy comparison
- ResQ baseline using existing `resq_gemm_v2.gemm_s8s8` + PyTorch softmax + FP16 matmul

### Phase 2: Core Kernel — Q·K^T Mixed Precision
- TMA loads for Q, K (INT8 + INT4)
- INT4 WGMMA + INT8 WGMMA with in-register accumulation
- Online softmax
- FP16 P·V WGMMA
- Basic correctness at this stage (no warp specialization, no pipeline)

### Phase 3: FA3 Pipeline Integration
- Warp specialization (producer/consumer split)
- Circular SMEM buffer (2 stages)
- Two-warpgroup pingpong
- Intra-warpgroup 2-stage pipeline (S_cur/S_next/P_cur/P_next)
- `setmaxnreg` register allocation

### Phase 4: Fused o_proj
- Per-head grouped mixed-precision GEMM
- Quantize O → INT4/INT8 → WGMMA → dequant

### Phase 5: Optimization & Tuning
- Autotuning tile sizes (Br, Bc, stages)
- INT4/INT8 WGMMA interleaving optimization
- SMEM pressure analysis for different k_high/k_low ratios

## 10. File Structure

```
mixed_tensor/
  flash_attn/
    mixed_flash_attn.cu           # Main kernel (CUTLASS 3.x)
    mixed_flash_attn_binding.cpp  # PyTorch binding
    setup.py                      # Build
    bench_flash_attn.py            # Benchmark script
    test_correctness.py            # Correctness unit tests
```
