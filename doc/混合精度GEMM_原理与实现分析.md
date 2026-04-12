# 混合精度 GEMM Kernel：原理与实现分析

## 一、问题背景

### 1.1 ResQ 的混合精度矩阵乘问题

ResQ 将 K 维度（即 hidden_dim）分成两部分：
- **高精度部分**（1/8）：用 INT8 量化
- **低精度部分**（7/8）：用 INT4 量化

ResQ 原始实现的计算方式（见 `ptq.py` 公式 4）：

```
Output = Q_4bit(X·U_l) · Q_4bit(U_l^T·W) + Q_8bit(X·U_h) · Q_8bit(U_h^T·W)
```

这意味着需要**两次独立的 GEMM 调用**：
1. INT4 × INT4 GEMM（处理 7/8 K 维度）
2. INT8 × INT8 GEMM（处理 1/8 K 维度）
3. 两个结果相加

**两次 GEMM 的低效之处**：
- 两次 kernel launch 的开销
- 两次读取输出矩阵 C 的带宽浪费
- 不能共享 shared memory 和 accumulator 寄存器
- 对于 memory-bound 的小 batch 场景（LLM decode），launch overhead 占比很大

### 1.2 我们的解决方案

**在一个 CUDA kernel 内部，同时执行 INT4 和 INT8 两种精度的 Tensor Core 矩阵乘法**，将结果累加到同一组 accumulator 寄存器中。

---

## 二、核心原理：8×8 Accumulator Tile 对齐

### 2.1 Tensor Core MMA 指令回顾

在 Ampere (SM80) 架构上，整数 Tensor Core 提供以下 MMA 指令：

| 精度 | MMA 指令形状 | 输入 A | 输入 B | Accumulator |
|------|-------------|--------|--------|-------------|
| INT8 | `m16n8k32` | 16×32 INT8 | 32×8 INT8 | 16×8 INT32 |
| INT4 | `m16n8k64` | 16×64 INT4 | 64×8 INT4 | 16×8 INT32 |

**关键观察**：

1. **两种精度的 MMA 产生的 accumulator 形状完全一致**：都是 16×8 的 INT32 矩阵
2. 16×8 的 accumulator 在 warp 内部以 **8×8 tile** 为单位分布在各线程的寄存器中
3. 每个线程持有的 accumulator fragment 寄存器布局，在 INT4 和 INT8 的 MMA 指令中**完全相同**

### 2.2 为什么可以在同一个 kernel 中混合使用

```
                    K 维度 (hidden_dim = 2048)
    ├──────── 7/8 (1792) ────────┤── 1/8 (256) ──┤
    │     INT4 tiles (低精度)     │  INT8 tiles   │
    │                             │  (高精度)      │
    └─────────────────────────────┴───────────────┘
                    │                      │
                    ▼                      ▼
              m16n8k64 MMA            m16n8k32 MMA
                    │                      │
                    ▼                      ▼
              16×8 INT32 acc         16×8 INT32 acc
              (8×8 tile ×2)          (8×8 tile ×2)
                    │                      │
                    └──────────┬───────────┘
                               ▼
                        直接累加到同一组
                        accumulator 寄存器
                        （寄存器布局自然对齐）
```

**这就是核心 insight**：

> 尽管 INT4 的 `m16n8k64` 和 INT8 的 `m16n8k32` 在输入维度（K=64 vs K=32）上不同，
> 但它们产生的 accumulator fragment **在每个线程中的寄存器分布是完全一致的**，
> 都遵循 8×8 tile 的 warp-level 分布模式。
>
> 因此，一个 warp 可以先执行若干次 INT4 MMA 累加部分结果，
> 再切换到 INT8 MMA 继续累加到**同一组** accumulator 寄存器上，
> 无需任何 shuffle 或格式转换。

### 2.3 Accumulator 寄存器布局详解

对于 `m16n8k*` 系列指令，每个线程持有的 accumulator 为 4 个 INT32 值：

```
一个 warp (32 线程) 的 16×8 accumulator 分布:

行维度 (M=16):  线程 0-3 负责 row 0-7（每线程2行），线程 4-7 负责同样的行但不同列
               线程 16-19 负责 row 8-15
列维度 (N=8):   由线程的 lane_id 决定

每个线程持有:
  C[0], C[1] → 属于 8×8 tile 0 (M 的前8行)
  C[2], C[3] → 属于 8×8 tile 1 (M 的后8行)
```

**INT4 MMA 和 INT8 MMA 的 accumulator 都遵循完全相同的布局**——这是硬件设计的结果，因为 accumulator 的分布只取决于 M×N 的 shape（都是 16×8），与 K 维度无关。

---

## 三、实现架构分析

### 3.1 整体架构

代码基于 CUTLASS 2.x 的多阶段（multistage）流水线 GEMM 框架修改。核心文件：

```
mixed_gemm_include/
├── mixed_gemm_config.h      # 31 种预定义的 tile 配置（高/低精度各一套）
├── kernel_mixed_gemm.h      # Kernel 入口：先跑 Low GEMM，再跑 High GEMM，共享 accumulator
├── mixed_mma_multistage.h   # 核心：多阶段流水线 MMA mainloop + 动态 tile 调度
├── mixed_mma_base.h         # MMA 基类（shared memory 布局定义）
├── mixed_gemm_mma.h         # CUTLASS MmaCore 配置 → MixedMmaMultistage
├── mixed_gemm_op.h          # Device-level GEMM operator 封装
├── device_mixed_gemm.h      # Host API 入口
└── configs.h                # 运行时配置选择
```

### 3.2 Kernel 执行流程

```
KernelMixedGemm::operator()
│
├── 1. 初始化 accumulator (清零)
│
├── 2. 低精度阶段 (INT4)
│   ├── 创建 MmaLow (INT4 MMA)
│   ├── 遍历 K 维度中属于低精度的 tiles
│   │   └── 使用动态 tile 调度 (get_next_low_tile)
│   ├── 累加到 accumulators_low
│   └── 转换到高精度 accumulator:
│       accumulators[i] = scale * accumulators_low[i]
│
├── 3. 高精度阶段 (INT8)
│   ├── 创建 Mma (INT8 MMA)
│   ├── 遍历 K 维度中属于高精度的 tiles
│   │   └── 跳过已被低精度处理的 tiles
│   └── 累加到同一个 accumulators
│
└── 4. Epilogue：写回结果
```

### 3.3 高低精度配置的关系

从 `mixed_gemm_config.h` 可以看到关键设计：

```
高精度 (INT8):  threadblock_k = 16, stage = 3
低精度 (INT4):  threadblock_k = 32 或 64, stage = 3-10
```

- **M 和 N 维度完全相同**（如都是 128×128）→ 保证 accumulator 形状一致
- **K 维度不同**：INT4 每次处理 32 或 64 个元素（因为 4-bit 更紧凑），INT8 处理 16 个
- `kMultipleCount = MmaLow::Shape::kK / Mma::Shape::kK`：低精度 K tile 是高精度的 2-4 倍

### 3.4 Shared Memory 的复用

```cpp
union SharedStorage {
    typename Mma::SharedStorage main_loop;        // INT8 用的 smem
    typename MmaLow::SharedStorage main_loop_low; // INT4 用的 smem
    typename Epilogue::SharedStorage epilogue;
};
```

**关键**：使用 `union` 让高低精度阶段复用同一块 shared memory，因为两个阶段不会同时执行。这最大化了 shared memory 利用率。

### 3.5 动态 Tile 调度算法

`get_next_low_tile()` 是核心调度算法。它解决的问题是：

> K 维度被分成 `total_tile_cnt` 个 tile（以低精度粒度计），
> 其中 `low_tile_cnt` 个是低精度，其余是高精度。
> 如何在 K 维度上**均匀交错**分布低精度 tiles？

算法使用**带残差的均匀步进**策略：

```
normal_step = low_tile_cnt / total_tile_cnt    // 期望的均匀间隔
residual 处理: 当整除不完美时，动态插入额外的 tile 来保持均匀分布
```

这确保了低精度 tiles 在 K 维度上均匀分布，而不是聚集在一端——避免了数值精度的局部恶化。

---

## 四、与标准 CUTLASS GEMM 的关键差异

| 方面 | 标准 CUTLASS GEMM | Mixed GEMM |
|------|------------------|------------|
| MMA 指令 | 单一精度 | 两种精度交替使用 |
| K 维度遍历 | 连续遍历所有 tiles | 分两阶段：先低精度tiles，后高精度tiles |
| Accumulator | 单个 | 共享（低精度累加 → 转换 → 高精度继续累加） |
| Shared Memory | 单精度 layout | union 复用高低精度 layout |
| Tile Iterator | 连续步进 | 动态跳跃（get_next_low_tile 决定下一个 tile 位置） |
| Global Memory | 单份 A, B | 两份：A_high/B_high + A_low/B_low |
| 配置空间 | 单套 M×N×K×stage | 两套独立配置（31种预定义组合） |

---

## 五、Hopper (SM90) 优化方向

### 5.1 当前实现的架构局限

当前代码基于 **Ampere (SM80)** 的编程模型：
- 使用 `cp.async` 做 Global → Shared Memory 的异步拷贝
- 使用 `mma.sync` 系列指令做 Tensor Core 计算
- 手动管理多阶段 (multistage) 流水线

### 5.2 Hopper 的关键新特性

| 特性 | 说明 | 对混合精度 GEMM 的意义 |
|------|------|----------------------|
| **TMA (Tensor Memory Accelerator)** | 硬件加速的多维张量搬运，替代 cp.async | 大幅减少数据搬运的指令开销，特别是对不同精度数据的 layout 转换 |
| **wgmma (Warp Group MMA)** | 新的 MMA 指令，4 个 warp 协作执行更大的矩阵乘 | 更大的 tile size，更高的计算吞吐 |
| **异步 wgmma** | wgmma 可以异步执行，与 TMA 重叠 | 更好的计算/搬运重叠 |

### 5.3 TMA + wgmma 的优化策略

1. **TMA 替代 cp.async**：
   - 用 TMA 描述符一次性搬运整个 tile，无需逐元素的迭代器
   - 对于混合精度场景，可以为 INT4 和 INT8 数据分别设置 TMA 描述符
   - TMA 天然支持不同的 data type 和 swizzle pattern

2. **wgmma 替代 mma.sync**：
   - wgmma 直接从 shared memory 读取操作数（无需 register → register 的 warp tile iterator）
   - 支持 INT8 的 `wgmma.mma_async.sync.aligned.m64n{8-256}k32`
   - 需要验证：INT4 wgmma 的支持情况和指令形状

3. **流水线模型变化**：
   - Ampere: N-stage software pipeline with cp.async
   - Hopper: Producer-consumer model with TMA + wgmma
   - 混合精度调度需要适配新的 producer-consumer 模型

### 5.4 关键挑战

1. **INT4 wgmma 支持**：需要确认 Hopper 的 wgmma 是否原生支持 INT4 操作数（CUTLASS 3.x 的 SM90 kernel 列表需要检查）
2. **TMA 描述符切换**：在 K 维度遍历中动态切换 INT4/INT8 的 TMA 描述符，可能引入额外延迟
3. **Shared Memory Swizzle**：不同精度的数据在 shared memory 中的 swizzle pattern 不同，需要确保 wgmma 能正确读取
4. **Register Pressure**：wgmma 使用更大的 tile，accumulator 占用更多寄存器

---

## 六、对 ResQ 优化的应用

### 6.1 ResQ 的矩阵乘 Shape 分析

对 Llama-3.2-1B (hidden_dim=2048, intermediate_size=8192)：

| 层 | M | N | K_total | K_INT8 (1/8) | K_INT4 (7/8) |
|----|---|---|---------|-------------|-------------|
| q/k/v_proj | batch×seqlen | 2048 | 2048 | 256 | 1792 |
| o_proj | batch×seqlen | 2048 | 2048 | 256 | 1792 |
| gate/up_proj | batch×seqlen | 8192 | 2048 | 256 | 1792 |
| down_proj | batch×seqlen | 2048 | 8192 | 1024 | 7168 |

### 6.2 预期收益

将 ResQ 的两次独立 GEMM 替换为我们的单 kernel 混合 GEMM：

1. **消除一次 kernel launch**：对 decode（batch=1, seqlen=1）来说，launch overhead 占比可达 30-50%
2. **减少一次 C 矩阵的 global memory 读写**：省一次 epilogue
3. **更好的寄存器利用**：accumulator 复用，不需要两次独立分配
4. **更灵活的 K 维度调度**：动态 tile 调度可以在 K 维度上交错高低精度，改善数值精度

### 6.3 与 ResQ 正交性的利用

ResQ 的 PCA 投影保证了低精度和高精度分量之间**不存在交叉乘法**（正交性）。这恰好符合我们 kernel 的设计——在 K 维度上，高精度 tiles 和低精度 tiles 完全独立，可以分开处理后累加。
