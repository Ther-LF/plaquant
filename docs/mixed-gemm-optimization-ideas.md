---
title: "Mixed-Precision Fused GEMM — 未实施的优化方向"
project: "PLAQuant/ProMix"
version: ""
commit: ""
description: "两个待实施的优化方向：(1) 跨 Phase Pipeline 衔接消除 drain+prologue 空档；(2) A/B 矩阵 K 维拼接改善 L2 局部性"
tags: [mixed-precision, gemm, pipeline, l2-cache, optimization, future-work]
related_kps: []
related_papers: []
created: 2026-06-03
updated: 2026-06-03
author: 罗凡
---

# Mixed-Precision Fused GEMM — 未实施的优化方向

## 背景

当前实现：顺序执行 MmaLow（INT4, K=1792, 14 次 K 迭代）和 MmaHigh（INT8, K=256, 4 次 K 迭代），两者共享 accumulator，SMEM 用 union 复用。性能 60μs fused vs 68μs baseline = 1.13× speedup。

代码：`/data/home/spanaluo/plaquant/kernels/mixed_gemm_l20/mixed_gemm_l20.cu`

---

## 优化方向 1: 跨 Phase Pipeline 衔接

### 问题

MmaLow 的 `gemm_iters()` 结尾有一段 drain：

```cpp
// mma_multistage.h:L663-666
cutlass::arch::cp_async_fence();
cutlass::arch::cp_async_wait<0>();   // 等所有 cp.async 完成（~几 μs stall）
__syncthreads();                      // 全 block 同步
```

然后 MmaHigh 的 `operator()` 又从头开始：

```cpp
// mma_multistage.h:L720-724
prologue(iterator_A, iterator_B, gemm_k_iterations);  // 填充 kStages-1 次 cp.async
gmem_wait();                                            // 再等一次
```

这段空档里 **tensor core 完全空闲**（没有 MMA 在飞），只在做 memory fence + 新数据搬运。

### 时间估算

- `cp_async_wait<0>` + `__syncthreads`: ~2-3μs
- prologue 的 3 次 cp.async + gmem_wait: ~3-5μs
- 总空档: ~5-8μs（占 60μs 总延迟的 8-13%）

### 优化方案

**方案 A: 分开 SMEM + Cross-Phase Prefetch**

不用 union，MmaLow 和 MmaHigh 各自一块 SMEM。在 MmaLow 最后几个 K 迭代中：
- Tensor Core 在算 MmaLow 的最后几个 tile
- 同时 cp.async 开始从 A_high/B_high 搬数据到 MmaHigh 的 SMEM slot

```
MmaLow steady-state:  [cp.async A_low + MMA] [cp.async A_low + MMA] [cp.async A_high!!! + MMA]
MmaHigh:              直接开始 MMA（prefetch 已完成）→ 无 prologue 空档
```

Trade-off:
- ✓ 消除 ~5-8μs 空档 → 预期 8-13% 加速
- ✗ SMEM 不能用 union → 两块 SMEM 共存 → SMEM 用量翻倍
- ✗ 需要自定义 MMA mainloop（CUTLASS 标准 Mma 不支持跨 phase prefetch）
- ✗ 实现复杂度高：需要在 MmaLow 的 mainloop 末尾注入 MmaHigh 的 cp.async

可行性条件：SMEM 总量 ≤ 228KB（H20 上限）。当前 64×64 tile + 4/5 stages 的 SMEM 约 30-40KB 每路，两路约 60-80KB，加上 epilogue 不超过 100KB → 可行。

**方案 B: 保持 union，缩短 drain**

用 `cp_async_wait<kStages-2>` 替代 `wait<0>` — 不完全 drain。但风险是 Phase 2 开始时旧 SMEM slot 可能还没被 Phase 1 消费完。

Trade-off:
- ✓ 不增加 SMEM
- ✗ 很难验证正确性（潜在 race condition）
- ✗ 收益有限（只省了 wait<0> 和 wait<N> 的差值）

### 结论

方案 A 更有价值，但需要自定义 MMA mainloop。建议在 SM90 版本（Hopper）中实现——那里 TMA + WGMMA 的 pipeline 更灵活，可以直接在 GemmKernel::operator() 中手写两个 phase 的 load 交织。

---

## 优化方向 2: A/B 矩阵 K 维拼接改善 L2 局部性

### 问题

当前 A_low(M, K_low) 和 A_high(M, K_high) 是两个**独立的 tensor**。同一个 CTA：
1. Phase 1: 读 A_low 的第 m 行（物理地址区域 A）
2. Phase 2: 读 A_high 的第 m 行（物理地址区域 B，可能远离 A）

两者在 HBM 中地址不连续 → Phase 2 读 A_high 时，L2 中 A_low 的数据可能已被 evict（两 phase 间隔 ~50μs）。

### 优化方案

**K 维拼接**：将 A_low 和 A_high 在 K 维度拼接为一个连续 tensor：

```
A_concat[M, K_low + K_high] = [A_low(M, K_low) | A_high(M, K_high)]
B_concat[N, K_low + K_high] = [B_low(N, K_low) | B_high(N, K_high)]
```

每行的 A_low 和 A_high 数据在物理内存中**相邻**。

### 收益分析

对于 M=128, K_low=1792(INT4 packed=896B), K_high=256(INT8=256B)：
- 每行 A_concat: 896 + 256 = 1152 bytes = 9 个 L2 cache line (128B)
- Phase 1 读 A_low（前 7 个 cache line）
- Phase 2 读 A_high（后 2 个 cache line）

**乐观情况**：Phase 1 最后一次 cp.async 的 cache line 顺带 prefetch 了 A_high 的开头 → Phase 2 直接 L2 hit。

**现实情况**：Phase 1 和 Phase 2 之间有 ~50μs 的计算。H20 的 L2 = 50MB，如果数据总量 < 50MB 则不会被 evict。对 M=128: A_concat 总量 = 128 × 1152B = 144KB << 50MB → **数据完全在 L2 中**！

### 结论

对 **small M**（M ≤ 256）：数据总量远小于 L2 容量，不管拼不拼接都能命中 L2。**收益可忽略**。

对 **large M**（M ≥ 4096）：A_concat 总量 = 4096 × 1152B = 4.5MB。仍然 < 50MB L2。只有当多 CTA 竞争时才有 L2 pressure。

**真正有用的场景**：极大 M（如 M=32768+）或 N 很大（如 N=32768）时，数据总量超过 L2 容量。此时拼接让同行数据连续 → 减少 L2 eviction + refetch。

### 实现成本

- 零 kernel 修改：只改 host 端的 tensor 构造（拼接后传不同的 offset pointer 和 stride）
- 一次性内存拷贝（推理时可以预处理好，不影响 latency）

### 额外考虑：Swizzle 与混合精度

当前 swizzle 对两个 phase 的效果相同——因为两个 phase 的 CTA grid 一样（同一个 CTA 处理同一个 (m, n) tile 的两个 K 段）。不需要针对混合精度做特殊 swizzle。

---

## 优先级总结

| 优化方向 | 预期收益 | 实现成本 | 优先级 |
|----------|---------|---------|--------|
| Pipeline 衔接 (方案 A) | 8-13% | 高（自定义 MMA） | 中（SM90 版本时做） |
| L2 矩阵拼接 | < 5%（small M 无感） | 低（仅 host 端） | 低（large M 场景再做） |
| 上 SM90 WGMMA | 3-5× | 高 | **高**（已有前期工作） |
