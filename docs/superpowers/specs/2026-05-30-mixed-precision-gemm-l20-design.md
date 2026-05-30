# Mixed-Precision Fused GEMM on L20 — Design Spec

**Author**: 罗凡  
**Date**: 2026-05-30  
**Status**: Approved

## Goal

在 L20 (SM89) 上实现单 kernel launch 的混合精度 GEMM：
- 输入：4 个 INT8 矩阵（A_low, B_low, A_high, B_high）
- 计算：`D = (A_low × B_low) + (A_high × B_high)`
- 输出：FP16 矩阵 D

对比 naive baseline（2 次 CUTLASS GEMM + 1 次 elementwise add）。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| API 层级 | CUTLASS 2.x 风格（直接操作 MMA/Iterator） | 快速出原型，参考 mixed_tensor 现成结构 |
| Mainloop 策略 | 顺序执行（先 low 后 high） | 最简单，第一版验证正确性和性能 |
| 维度 | 固定 M=128, N=2048, K_high=256, K_low=1792 | PLAQuant Llama-3.2-1B 的实际参数 |
| Epilogue | 简单 cast INT32→FP16（无 dequant） | 先验证 mainloop 性能 |
| 目标架构 | SM89 (L20), mma.sync 指令 | 本地开发验证 |

## Architecture

```
KernelMixedGemmSimple<MmaLow, MmaHigh, Epilogue, Swizzle>
├── Params: A_low, B_low, A_high, B_high, D 指针 + 维度
├── SharedStorage: union { MmaLow::SharedStorage, MmaHigh::SharedStorage, Epilogue::SharedStorage }
└── operator():
    1. 计算 CTA tile offset
    2. 构造 A_low/B_low iterators
    3. MmaLow(k_iterations_low, acc, iter_A_low, iter_B_low)  ← acc = A_low × B_low
    4. 构造 A_high/B_high iterators
    5. MmaHigh(k_iterations_high, acc, iter_A_high, iter_B_high)  ← acc += A_high × B_high
    6. Epilogue: D = FP16(acc)
```

### Tile Configuration

```
TileShape: GemmShape<128, 128, 32>  (M_tile, N_tile, K_tile)
WarpShape: GemmShape<64, 64, 32>
InstructionShape: GemmShape<16, 8, 32>  (INT8 mma.sync on SM80+)
Stages: 4 (cp.async multi-stage pipeline)

K iterations:
  low:  K_low / K_tile = 1792 / 32 = 56 iterations
  high: K_high / K_tile = 256 / 32 = 8 iterations
```

### Data Layout

```
A_low:  (M, K_low)  = (128, 1792), RowMajor, INT8
B_low:  (K_low, N)  = (1792, 2048), ColumnMajor, INT8
A_high: (M, K_high) = (128, 256), RowMajor, INT8
B_high: (K_high, N) = (256, 2048), ColumnMajor, INT8
D:      (M, N)      = (128, 2048), RowMajor, FP16
```

### Grid Configuration

```
grid = (M / 128, N / 128) = (1, 16) for M=128
block = 128 threads (4 warps)
```

## Benchmark Plan

对比三个配置：
1. **Naive baseline**: 2× CUTLASS INT8 GEMM + 1× elementwise add
2. **Fused kernel**: 本实现（单 launch）
3. **cuBLAS reference**: cublasSgemm 等效（如果有 INT8 support）

测量指标：
- Latency (μs)
- Throughput (TOPS)
- Correctness vs naive (cosine similarity)

## File Structure

```
kernels/mixed_gemm_l20/
├── mixed_gemm_l20.cu        # kernel 实现
├── mixed_gemm_l20.h         # host-side wrapper
├── setup.py                  # PyTorch CUDAExtension
├── benchmark.py              # performance benchmark
└── test_correctness.py       # correctness test
```

## Out of Scope (后续)

- Dequant EVT epilogue
- 可变 M (batch size 1-128)
- INT4 packed load（当前 INT4 预先转为 INT8）
- SM90/TMA 版本
- 交织执行策略
