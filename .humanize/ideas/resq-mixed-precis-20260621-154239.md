# PLAQuant-SM100: Single-Kernel FP8+NVFP4 Mixed-Precision via tcgen05.mma

## Goal

让混合精度 fused GEMM kernel（PLAQuant 系统设计）真正用进 LLM 推理，**在 1B/3B/8B 三个模型尺寸上同时**达成三条硬指标，每条都要在两个层级（kernel-level micro benchmark + end-to-end real inference）上分别实测：

1. **精度不退化**——wikitext PPL 不退化于现有 INT4+INT8 W4A4 baseline（quant-vs-quant 对比，不是 vs FP16；FP16 太严格不现实）
2. **快于 FP16 cuBLAS**——kernel 在 LLM 真实命中 shape sweep 上 + end-to-end real inference 都比 torch.matmul FP16（5th-gen TC cuBLAS）快
3. **快于 2-launch FP**——单 kernel fused launch 比"两次单独 launch（独立 FP8 GEMM 跑高路径 + 独立 NVFP4 GEMM 跑低路径，FP32 顺序累加）"快——同样在 kernel micro 和 end-to-end 两个层级都要赢；这是 PLAQuant single-kernel 系统贡献的核心 claim

为什么两个层级都测：**只跑 end-to-end 不知道性能收益从哪儿来**（哪个 layer / 哪个 shape 在贡献？），**只跑 micro 不知道是否被 Python 侧 quant pack / dequant 开销吃光**。两层都过才算项目完成。

算法侧用 ResQ 的 PCA + variance-split + learnable rotation 作为生成"高/低精度通道"数据布局的方式。

## Final Acceptance Criteria

硬指标分两层——**kernel 层（micro benchmark）** 和 **端到端层（real inference）**——每层都要在 1B/3B/8B 三个模型尺寸 × W4A4/W4A4KV4 两个量化配置上实测，**任何一条任何一个组合 fail 都算项目未完成**：

### 端到端层（real inference）

| 维度 | 阈值 | 1B | 3B | 8B |
|---|---|---|---|---|
| **精度（PPL 不退化）— W4A4** | wikitext PPL ≤ 现有 INT4+INT8 W4A4 baseline | ≤ 11.70 | ≤ 8.61 | ≤ 6.99 |
| **精度（PPL 不退化）— W4A4KV4** | 同上但 KV cache 也 4-bit | ≤ 11.96 | ≤ 8.70 | ≤ 7.04 |
| **性能 vs FP16 cuBLAS** | 端到端 latency speedup > 1.0×（目标 1.5-2×） | ≥ 1.0× | ≥ 1.0× | ≥ 1.0× |
| **性能 vs 2-launch FP** | 单 kernel fused vs "FP8 GEMM + NVFP4 GEMM 顺序两次 launch + Python 累加" | > 1.0× | > 1.0× | > 1.0× |

### Kernel 层（micro benchmark，shape sweep）

只测 fused kernel 自己，独立于模型——目的是知道**在哪些 (M, N, K_high, K_low) 形状下 fused 比 2-launch / FP16 快，哪些形状下输**。如果只看 end-to-end 数字，赢了不知道是哪个 shape 在贡献、输了不知道是哪一个层拖累，**没法 attribute 性能收益**。

| 维度 | 阈值 | 测法 |
|---|---|---|
| **正确性** | 每个 sweep shape cosine ≥ 0.9999 vs FP32 reference | 整 sweep 不允许任何形状失真 |
| **vs FP16 cuBLAS（per-shape）** | LLM 真实命中 shape 子集（1B/3B/8B 各层 × batch ∈ {1, 16, 64, 128}）上 fused/FP16 ≥ 1.0×；目标 batch=128 时 ≥ 1.5×、batch=2048 时 ≥ 2× | per-shape 加速比表 + heatmap |
| **vs 2-launch FP（per-shape）** | LLM 真实命中 shape 子集上 fused/2-launch > 1.0×（目标 1.05-1.20×，对照 SM80 INT 路径同量级） | per-shape 加速比表 + heatmap |
| **Characterization report** | `kernels/mixed_gemm_sm100/RESULTS.md` 列出：fused 赢的 shape regime、输的 shape regime、per-stage profile (TMA / WGMMA / accumulate / store)、归因 | β-2 deliverable，没这份报告 β-2 不算过 |

阈值定义：
- **精度不退化** ≠ "略好"或"接近"——任何一项 PPL 退化于 baseline 即未达标。**float 噪声允许 ±0.01 buffer**，但不接受 +0.05 以上的退化。
- **性能 vs FP16** 是和真实部署基线（torch.matmul FP16，走 5th-gen TC cuBLAS）比；最低门槛 1.0×，目标 1.5-2×。
- **性能 vs 2-launch FP** 是 PLAQuant 论文的核心 claim："single-kernel multi-precision 比 sequential 两次 launch 快"。**这一条不依赖 FP16 比较，是对 PLAQuant 论文卖点的硬证明**。即使 vs FP16 已经赢，没赢 2-launch FP 等于系统层贡献没立住。
- **kernel 层 vs 端到端层不可互相替代**——只跑 end-to-end 但没有 shape sweep，意味着不知道性能收益从哪儿来；只跑 micro 但没 end-to-end，意味着不知道是否 Python 侧 dequant / quant pack 把 kernel 收益吃光。两层都过才算项目完成。
- **覆盖** 不是"1B 验证后推断 3B/8B 同样有效"，是**三个尺寸都要实测**——尤其 8B 上 cross-head 统计差异更可能暴露 global PCA 与 per-head 的细节差距。

每个 phase 验收点（β-0..β-3）都要能直接对回到上面两张表的某一格；任何阶段无对应 acceptance row 的 verify 不算 verify。

## Verified Current State (实测 + 源码核查)

**算法侧（已完成）：**
- ResQ PTQ 流程在 1B/3B/8B 上复现了论文 W4A4 / W4A4KV4 PPL（fake quant 路径，走 quant→dequant→FP16 GEMM 模拟）
- Step 0 PCA basis 计算 / Step 1 rotation 优化 / Step 2 PTQ eval 三步在 promix/ 下都跑通了

**Kernel 层（B20Z micro benchmark + CUTLASS 4.5 源码核查）：**

| 比较 | 结果 |
|---|---|
| 当前 SM80 fused INT4+INT8 vs CUTLASS 2-launch INT (两边都 SM80 mma.sync) | 0.98-1.15× |
| 当前 SM80 fused INT4+INT8 vs FP16 cuBLAS (SM100 5th-gen TC) | **0.04-0.32×**（fused 慢 3-25×） |
| 当前 fused 峰值 TFLOPS | 58.5 |
| FP16 cuBLAS 峰值 TFLOPS | 1434.6 |

也就是说：以前 DEVLOG 写的"1.0-1.15×加速"是和 INT 自己的 2-launch baseline 比，跟最终目标（vs FP16 cuBLAS）是两回事；**真正要赢的对手 FP16 cuBLAS 在 SM100 上比当前 kernel 快 3-25 倍**。

**SM100 (Blackwell) native MMA 类型支持（`include/cute/atom/mma_traits_sm100.hpp` 完整目录）：**

| 类型 | instruction kind | 支持 | atom family |
|---|---|---|---|
| FP16 / BF16 | `kind::f16` | ✓ | SM100_MMA_F16BF16_* |
| TF32 | `kind::tf32` | ✓ | SM100_MMA_TF32_* |
| INT8 | `kind::i8` | ✓ | SM100_MMA_S8_* |
| FP8 / FP6 / FP4 (dense, no microscaling) | `kind::f8f6f4` | ✓ | SM100_MMA_F8F6F4_* |
| **MXFP8 / MXFP6 / MXFP4 微缩放** (block=32, FP8 E8M0 scale) | `kind::mxf8f6f4.block_scale` | ✓ | **SM100_MMA_MXF8F6F4_*** |
| MXFP4 (block=32, FP8 E8M0 scale) | `kind::mxf4nvf4` (VS=32) | ✓ | SM100_MMA_MXF4_* |
| **NVFP4 微缩放** (block=16, FP8 E4M3 scale) | `kind::mxf4nvf4` (VS=16) | ✓ | **SM100_MMA_MXF4NVF4_*** |
| **INT4** | — | ✗ | **不存在**——Blackwell 砍掉了 native INT4 |

PRIMARY 选用的两个 atom 加粗：高路径 `SM100_MMA_MXF8F6F4_*`，低路径 `SM100_MMA_MXF4NVF4_*`。

**端到端（已观察）：**
- promix/inference/ 已接上 real INT4+INT8 GEMM forward 路径，1B pipeline 不崩
- 1B real-quant 端到端比 FP16 慢 ~9×（latency 测过）
- 但 real-quant 路径的 PPL **从未测过**，所以"换 kernel 后 PPL 与 FP16 一致"这条没数据

## What Will Change (后续要改的地方，由 PRIMARY 接手)

按照 β 路径决策，下面这些是**不再保持现状**的：

- promix/inference/ 当前的 INT4+INT8 real path **整条换成 MXFP8 + NVFP4 microscaled native path**（统一到一种数据模型，**不保留 dense FP fallback**——理由见 PRIMARY 中 "MXFP/NVFP vs Dense FP 对照"）
- ResQ 量化器从 INT4/INT8 fake quant **改成 MXFP8 (E4M3 + 32-element FP8 block scale) + NVFP4 (E2M1 + 16-element FP8 E4M3 block scale) fake quant**；PCA basis / 高低 split / 旋转架构都不变，但 **R 必须重训**（per-block scale 模型与原 per-channel 模型不同）
- `kernels/mixed_gemm_l20/` 保留为 SM80 历史代码 + 正确性 oracle，**不再是真实推理路径**；新建 `kernels/mixed_gemm_sm100/` 走 **`tcgen05.mma.kind::mxf8f6f4` + `kind::mxf4nvf4`**（**全部 microscaled**，不混 dense `f8f6f4`），共享 TMEM FP32 accumulator
- 当前 5th-gen Tensor Core 没用上 → 新 kernel 走 native microscaled 通道，目标拿到 ~2-4× FP16 加速（MXFP8 ~3 PFLOPS、NVFP4 ~6 PFLOPS 的理论上限）
- **o_proj 的 per-group 量化（groupsize=64）整条结构被替换掉**：算法侧把 o_proj 输入 PCA 从 per-head 64×64 升级到 hidden_dim 2048×2048 global PCA（详见 PRIMARY "配套算法侧改动" 段），global rotation 之后 o_proj 输入按全局 variance 排序，直接切前 256 hi / 后 1792 lo，**不再有 per-head group 概念**——per-group scale 不是被 block scale 吸收，而是从模型里彻底消失。系统侧 o_proj 跟 q/k/v/gate/up/down 走同一条共同 path：单 scale-per-row + 单 scale-per-channel + per-block scale，**不再 skip、不再特殊处理**
- Python 侧 dequant 的 9× 端到端慢 → 在 β-2 末尾通过 EVT epilogue 把 dequant fuse 进 kernel 解决（FP path 的 dequant 公式比 INT path 简化：只有 `out = s_x_token · s_w_channel · acc`，没有 INT 的 shift/zero/colsum 项）
- 现有 1B/3B/8B 的 PTQ rotation 文件**按存盘内容拆解**：
  - **`rotation/U-*.bin` 部分重算** —— q/k/v/gate/up 的 hidden_dim 2048×2048 PCA 复用（与量化格式无关），**但 o_proj 输入 PCA 从 per-head 64×64 升级到 hidden_dim 2048×2048 global**，需要重新跑 basis.py 生成新 U（cost: 1B ~20min, 3B ~30min, 8B ~45min；与原始 basis 计算同量级）。理由详见下方"配套算法侧改动：o_proj input PCA 升级"段
  - **`rotation/E-*.bin` 部分重算** —— 特征值跟着 U 一起重算
  - **`rotation/R-*.bin` 作废，必须重训** —— `optimize_rotation.py` 的训练 loss 是量化噪声，原 R 是按 INT4/INT8 noise 模型 + per-head U 训的；切到 MXFP8/NVFP4 + global U 之后两个假设都变了（cost: 1B ~1min, 3B ~5min, 8B ~10min，每个 model 重训成本可控）
  - **运行时合成逻辑不变** —— Step 2 的 `rotation.py` line 194 在内存里现算 `U_combined = U_pca @ R_new` 然后应用到 weight，这段代码 PRIMARY 不需要改；只是被它组合的两个输入张量来源不同（新 U + 新 R）。注意"weight 实际生效的 rotation"是 U×R 合成结果，不是 U 文件本身——文件层面的拆分不代表数学层面的拆分

## Primary Direction: PLAQuant-SM100 Single-Kernel FP8+NVFP4 Pipeline

### Rationale

我们前期工作（PLAQuant）的核心贡献是"不同精度的 GEMM 可以在同一个 kernel 里 single-launch 累加"。它在 SM80 上用 INT4×INT4 + INT8×INT8 两个 mma.sync atom 共享 register accumulator 做出来。ResQ 算法刚好天然产生符合这个架构的数据布局（按 channel 方差切高/低段，1/8 高 + 7/8 低），所以系统 + 算法天然对齐。但搬到 B20Z (SM100 Blackwell) 后两件事破了：(1) INT4 native 在 Blackwell 上不存在了，连 CUTLASS atom 都没有；(2) FP16 cuBLAS 已经走 5th-gen Tensor Cores 跑到 ~1.4 PFLOPS，我们 SM80 路径 ~58 TFLOPS，差 25×。

正确的应对是**把 PLAQuant 的 single-kernel 架构思想从 INT4+INT8 改写到 SM100 native 的 FP8+NVFP4**：
- 系统层：完整保留 PLAQuant 的 single-kernel two-phase shared-accumulator 拓扑，从 register accumulator 升级到 TMEM accumulator
- 算法层：ResQ → ResQ-FP，PCA basis / 高低 split / 旋转 R 全部沿用，只把量化器从 INT 换 FP

这条路保留 PLAQuant 论文卖点（系统层 single-kernel 多精度）+ 升级到 SM100 真正 native 通道（理论 ~3-4× FP16 加速 ceiling）+ 落到一个新颖的 GEMM 形态（FP8+FP4 mixed PLAQuant），**比单纯做 SM100 INT8 退化版（路径 α）有真正算法故事**，比单纯做 SM90 INT4 验证（路径 γ）有真正硬件代际优势。

### Approach Summary

**核心拓扑（不变 vs SM80 PLAQuant）：**

```
                        ┌──────────────────────────────────────┐
                        │   TMEM FP32 accumulator (M × N)      │
                        │   M ∈ {64, 128}, N ∈ [8, 256]        │
                        └──────────┬─────────┬─────────────────┘
                                   │ accumulate(scaleC=true)
       ┌───────────────────────────┘         └──────────────────────┐
       │ Phase 1: MXFP8 high path            │ Phase 2: NVFP4 low path
       │   tcgen05.mma.kind::mxf8f6f4        │   tcgen05.mma.kind::mxf4nvf4
       │   K_iter = K_high / 32              │   K_iter = K_low / 64
       │   A_high (FP8 E4M3, M×K_high)       │   A_low (NVFP4, M×K_low)
       │     + block_scale_A (FP8, 1/32)     │     + block_scale_A (FP8, 1/16)
       │   B_high (FP8 E4M3, N×K_high)       │   B_low (NVFP4, N×K_low)
       │     + block_scale_B (FP8, 1/32)     │     + block_scale_B (FP8, 1/16)
       │   TMA pull A/B + 2 scale tiles      │   TMA pull A/B + 2 scale tiles
       └─────────────────────────────────────┴──────────────────────┘
                                   │
                            ┌──────▼──────┐
                            │EVT epilogue │  s_x·s_w·acc → FP16
                            │ (β-2 内做)  │  （FP path 无 shift/zero/colsum 项）
                            └──────┬──────┘
                                   ▼
                            Output (M×N) FP16
```

CUTLASS atom 约束（来源 `include/cute/atom/mma_traits_sm100.hpp`）：
- `SM100_MMA_MXF8F6F4_SS`（高路径）：M ∈ {64, 128}，N ∈ [8, 256]，K = 256/sizeof_bits<A> = **32 elements** for FP8，**每 32 element 带一个 FP8 (E4M3) block scale**
- `SM100_MMA_MXF4NVF4_SS`（低路径）：M ∈ {64, 128}，N ∈ [8, 256]，K = 256/sizeof_bits<A> = **64 elements** for FP4，block scale vector size **VS=16 → NVFP4**（**不是 VS=32 的 MXFP4**），block scale 是 FP8 E4M3（带 1-bit mantissa，比 MXFP4 的 E8M0 power-of-2 scale 精度高约 1 bit）。理由：低路径吃 7/8 channel 是量化主力，精度优先；NVFP4 跨厂商不兼容（仅 NVIDIA 自家），但项目硬件 target 已锁定 SM100，这条不重要
- **两条 phase 都用 microscaled atom**，是 PRIMARY 的关键设计（详见下面"o_proj 不再是特例"段）
- **C (accumulator) layout 不依赖 K**——只依赖 M 和 N。所以两 phase 选相同 M, N → 共享同一块 TMEM C 区域 → 直接 FP32 累加。这是 SM80 share register accumulator 思路在 TMEM 上的自然延伸。

**背景：MXFP / NVFP vs Dense FP 对照**

普通 FP（FP8 E4M3 / FP4 E2M1）是单元素自带 sign + exp + mantissa，scale 是 per-tensor 或 per-channel 共享。Microscaled FP（MXFP / NVFP）多了一层结构：**每 N 个连续 element 共享一个 FP8 block-level scale，element 的 FP 表达只承担"这一小段内的相对位置"**。具体格式：

| 格式 | 元素 | block_size | block scale | 效用 bits/element | 标准 |
|---|---|---|---|---|---|
| FP8 (dense) | E4M3 / E5M2 | — (per-channel) | FP32 (per channel, 摊薄) | 8 | NVIDIA 早期 |
| **MXFP8** | E4M3 / E5M2 | 32 | FP8 E8M0 (power-of-2) | **8.25** | OCP MX |
| FP4 (dense) | E2M1 (16 个值) | — | 实际不可用 | 4 | 仅理论 |
| **MXFP4** | E2M1 | 32 | FP8 E8M0 | **4.25** | OCP MX |
| **NVFP4** | E2M1 | **16** | FP8 **E4M3** (带 mantissa) | **4.5** | NVIDIA 变种 |

各维度比较：

| 维度 | Dense FP 占优 | MXFP/NVFP 占优 | 说明 |
|---|---|---|---|
| 精度 | — | ✓（几乎所有 LLM 工况） | LLM weight/activation 天然有 channel-wise outlier，per-block scale 局部适应远优于 per-channel；分布完全均匀的合成数据上两者等价 |
| 吞吐 (SM100 native) | — | ≈ 持平 | 都是单条 tcgen05.mma instruction；microscaled 多读一份 block scale tile (SMEM)，带宽开销 < 1% |
| 存储 | ✓ | — | 多 0.25-0.5 bit/element 的 block scale 元数据；vs FP16 → FP4 省的 12 bit，可忽略 |
| 硬件支持 | ✓ (SM89 / SM90) | — (SM100 only) | SM89 完全不支持 MXFP；SM90 部分支持 mxfp8 但 mxf4nvf4 仍是 SM100 起步；老卡上 MXFP 走软件 emulation 反而极慢 |
| 软件生态 | ✓ | — (落后 ~2 年) | dense FP8 自 2022 起在 PyTorch / vLLM / TRT-LLM 都是默认路径；MXFP/NVFP 序列化、模型 hub 兼容性仍在补 |
| 标准 fragmentation | 中性 | 略劣 | NVFP4 (NVIDIA, block=16) vs MXFP4 (OCP, block=32) 不互通；跨厂商部署需选 MX 标准 |

**对本项目（LLM PTQ + SM100 + outlier-dominated weights/activations）的取舍**：所有重要维度上 microscaled 都至少打平 dense FP、且多数维度上更好；唯一的代价 (0.25-0.5 bit/element + 软件成熟度) 在我们工作负载下都可忽略。**所以 PRIMARY 选 microscaled，没有保留 dense f8f6f4 fallback 的理由**。这是把整个项目限定到 SM100 之后的自然推论——一旦放弃 SM89/SM90 硬件兼容，也就同时放弃了"必须选 dense FP"的最后一条约束。

唯一 β-1 必须显式处理的副作用：**ResQ 当前的旋转 R 是在 INT per-channel scale 模型下训出来的**；切到 per-block scale 后量化噪声分布变化，R 必须重训（不能直接 carry over）。理论上 per-block 是 per-channel 的细化（信息更多），最优 R 至少不比老 R 差，但 R 重训这件事不是 free，要进 β-1 工作量。

**为什么两条 phase 都选 microscaled atom（关键设计决策）：**

ResQ 算法在 o_proj 上用 per-group 量化（groupsize=head_dim=64），每个 (row, group) 有独立 scale，每个 (column, group) 也有独立 scale。这个特征在 INT 路径下是死结——单 scale-per-row 的 epilogue 不能表达 per-group 累加，所以原 ResQ 实现里 o_proj 一直被 weight_packer skip 掉，留 fake quant。

如果我们高路径选 `kind::f8f6f4` (dense FP8，no microscale)，o_proj 在高路径上仍然有同样问题：FP8 也是单 scale-per-row × 单 scale-per-column。

**关键洞察：FP microscaled 比 ResQ per-group 是更细的量化粒度，不是要"吸收"per-group，而是 per-group 在 microscaled 下变得多余。**

| 量化粒度（K 方向） | INT path | FP microscaled path |
|---|---|---|
| 通用层 (q/k/v/gate/up/down) | per-channel weight + per-token activation | per-block (MXFP8=32, NVFP4=16) |
| o_proj 特例 | **per-group=64**（捕捉 per-head 统计差异） | **per-block (MXFP8=32, NVFP4=16)** —— 比 per-head 还细 |

NVFP4 的 block=16 比 ResQ groupsize=64 还细 4 倍，**microscaling block 提供的 scale granularity 严格优于 per-group**。换言之 per-group 是 microscaling 的更粗版本，存在的意义在 INT 路径下是因为没法做 per-block；FP path 直接走更细的 per-block，不需要 per-group。

更重要的是：保留 per-group + 1/8 ratio 在 alignment 上**根本走不通**。groupsize=64、high_fraction=1/8 意味着每个 head 内 8 channels 高 + 56 channels 低，但 MXFP8 instruction K extent = 32，单 head 的 8-channel 高段**凑不齐一条 instruction**，56-channel 低段也不是 16/64 的整数倍。

**配套算法侧改动：o_proj input PCA 从 per-head 升级到 full hidden_dim global**。ResQ 现状下 o_proj 输入做 per-head 64×64 PCA（block-diagonal U），是 q/k/v/gate/up 全用 hidden_dim 2048×2048 global PCA 时唯一的特例——选 per-head 当年是为了和 R2 (head_dim per-head 旋转，attention 内部约束) 结构对称，**但 R2 作用在 V 上、U 作用在 o_proj 输入上，是数据流不同位置的两个独立旋转，可以独立选**。改 global 后三件事自然成立：

- **没有 head 结构需要重排**——variance 全局排序后直接切前 256 high / 后 1792 low，K_high=256 整除 32 ✓，K_low=1792 整除 16 ✓
- **MXFP8 block-of-32 内 magnitude 同质**——全局 top 256 是单调递减 variance 序列，相邻 32 个 channel 数量级几乎一致，block scale 利用率拉满；不再有 per-head 切法下"4 个 head 的 hi 凑一个 block，magnitude 跨多倍"的潜在问题
- **数学上严格不弱于 per-head**——per-head 是 O(2048) 中的 block-diagonal subset，global 是 full manifold；top-K 方差捕获 global ≥ per-head（cross-head 无相关时打平，有相关时严格更优），所以 INT 路径 PPL 期望持平或微好（**不应该退化**）

R2 保持 per-head 不变，attention 内部计算流程不动。代价：basis.py 增加 `o_proj_pca: full_global` 模式，PCA 矩阵从 32×(64×64) 变成 1×(2048×2048)，PTQ 时一次性算（秒级）；fuse 进 o_proj weight 后零推理开销。R 必须按新 U 重训（原 R 是 per-head U 上的最优解）。

风险与回退路径：尽管数学上 global ≥ per-head，PPL 是否真的持平/略好需要 β-1 INT 路径冒烟测试（先把 PCA mode 切掉，量化器仍走 INT，验证 INT W4A4 PPL）；预期 ±0.1 之内。如果实测退化 > 0.3，**先怀疑实现 bug 而不是方案选错**；屡 debug 不通才触发 Alt-4 fallback（回退 per-head PCA + cross-head rearrange 这条原 ResQ 设计思路）。

**算法侧（ResQ → ResQ-FP）改动文件级 map：**

| 文件 | 改动 | 性质 |
|---|---|---|
| `promix/quantize/basis.py` | 加 `o_proj_pca: full_global` 模式（默认 per_head 保持兼容）；hidden_dim PCA 路径不动 | 改一段 |
| `promix/quantize/rotation.py` | **不改** —— 旋转融合格式无关 | 复用 |
| `promix/quantize/fuse_norm.py` | **不改** —— RMSNorm 融合格式无关 | 复用 |
| `promix/quantize/hadamard.py` | **不改** | 复用 |
| `promix/models/loader.py` | **不改** | 复用 |
| `promix/quantize/quant_utils.py` | `ActQuantizer` 加 `bits=mxfp8`、`bits=nvfp4` 两路；保留旧 INT 路（共存） | 扩展 |
| `promix/quantize/optimize_rotation.py` | quant noise 模拟从 INT round 改 FP cast (mxfp8 / nvfp4) | 改一段 |
| `promix/quantize/gptq.py` | round-to-nearest 改 FP nearest representable（hessian 项不变） | 改一段 |
| `promix/inference/quant_ops.py` | 新增 `quantize_activation_mxfp8_per_token`、`quantize_activation_nvfp4_per_token`（输出包含 block_scale 张量）；旧 INT pack/shift 函数保留 | 加新函数 |
| `promix/inference/weight_packer.py` | 新增 MXFP8 + NVFP4 packing 路径，**o_proj 跟 q/k/v 走相同代码路径**（global PCA 后没有 per-group 结构）；删除现有 `groupsize > 0: continue` skip；旧 INT packer 保留 | 加新分支 |
| `promix/inference/real_forward.py` | 新增 FP 分支，dequant 公式简化为 `out = s_x_token · s_w_channel · acc`（FP path 无 shift/zero/colsum 项；o_proj 没有 group 概念，跟其他层共用 dequant）；**o_proj 不再 skip**，跟其他层走同一路径 | 加新分支 |
| `kernels/mixed_gemm_l20/` | **保留作正确性 oracle**，**不再是真实推理路径** | 冻结 |
| `kernels/mixed_gemm_sm100/` (新建) | dual-phase `mxf8f6f4 + mxf4nvf4`，TMEM 共享 accumulator + EVT epilogue | 新写 |
| `promix/configs/llama-3.2-{1b,3b,8b}-mxfp8-nvfp4.yaml` (新建) | high_bits=mxfp8、low_bits=nvfp4 标识；可与 β-4 per-op variable ratio 配合 | 新写 |

### Phase Plan

**β-0 调研 + minimal PoC kernel（~半天到一天）**
- 读 CUTLASS 4.5 SM100 examples（特别是 `examples/61_hopper_gemm_with_topk_and_softmax/`、SM100 集合）
- 确认 NVFP4 block scale 内存布局（tile 内交错 vs 单独 buffer）
- 写 ~50-80 行 minimal kernel：M=128, N=128, 单 phase FP8 GEMM；只为编出来 + 跑通；**不接 ResQ 数据，不接 dequant**
- 同上做 minimal NVFP4 单 phase kernel
- **关键：用这两个 minimal kernel 直接拼一个 2-launch FP baseline**（Python 串调 mxfp8_gemm() + nvfp4_gemm() + FP32 累加），跑单点 (M=128, N=2048, K_h=256, K_l=1792) 测 latency
- 验收 (a)：两个单 phase kernel 都能编出来 + cosine ≥ 0.9999
- 验收 (b)：两个单 phase kernel 各自 vs FP16 cuBLAS 跑出 ≥ 1.5× 单点加速
- 验收 (c)：**2-launch FP baseline latency 记下来作为 β-2 fused 必须打败的对照线**——这一步早测的目的是：如果 2-launch 已经接近 FP16 cuBLAS 速度上限，β-2 fused 即使写出来也很难拉开有意义的差距（PLAQuant single-kernel claim 风险，详见 Known Risks）。该数据决定 β-2 是否需要 tile shape 等级的 aggressive tuning，还是写完简洁版即可

**β-1 算法 sanity check（~1-2 天）**
- 在 `promix/quantize/quant_utils.py` 加 MXFP8 (E4M3 + 32-element FP8 block scale) + NVFP4 (E2M1 + 16-element FP8 block scale) fake quantizer（PyTorch 实现，不需要 kernel）
- 在 `promix/quantize/basis.py` 加 `o_proj_pca: full_global` 模式（详见 PRIMARY 中"配套算法侧改动"段）
- 在 `promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml` 创建新 config（high_bits=mxfp8、low_bits=nvfp4 标识、o_proj_pca=full_global）
- **重跑 1B Step 0** 生成新 U/E bin（global PCA on o_proj 输入）；q/k/v/gate/up 的 hidden_dim PCA 部分跟旧文件一致
- Step 1：跑 rotation 优化（quant noise 用 MXFP8/NVFP4 模拟）
- Step 2：fake quant PPL 评估，**o_proj 走和其他层一样的 microscaled path（不再 skip）**
- **验收点 (a-pre)：global PCA on o_proj 在 INT 路径下的冒烟测试** —— 先不动量化器，只把 basis.py 的 o_proj PCA 从 per-head 切成 full_global，跑 INT W4A4 PPL，对比 ResQ baseline 11.70（1B），**预期 ±0.1**。这一步把"PCA mode 变化"和"FP 量化变化"两个改动隔开，叠加出问题没法 bisect。失败 (PPL > baseline + 0.3) **一律先怀疑实现 bug 而不是方案选错**（数学上 global PCA 是 per-head 的 strict superset，不应退化）；多次 debug 不通才触发 Alt-4 回退 per-head + cross-head rearrange
- 验收点 (a)：**所有层（含 o_proj）都过 microscaled fake quant**，PPL 不退化于现有 INT4+INT8 W4A4 baseline——**1B ≤ 11.70**、**3B ≤ 8.61**、**8B ≤ 6.99**（KV4 配置：1B ≤ 11.96 / 3B ≤ 8.70 / 8B ≤ 7.04）。注意：阈值是"不退化"而不是"+0.5 buffer"——浮点噪声允许 ±0.01，超过 +0.05 视为退化
- 验收点 (b)：**o_proj 单独切回 fake-int per-group**，对比 microscaled o_proj 的 layer-wise 输出 cosine ≥ 0.99，证明 NVFP4 microscaling 没有显著比 INT per-group 差
- 验收点 (c)：3B / 8B 顺次重复验证 (a-pre)(a)(b)——**三个模型都过才算 β-1 通过**，不接受"1B 过了 3B 没测"

**β-2 SM100 fused kernel（~1-2 周）**
- 新建 `kernels/mixed_gemm_sm100/`，目录结构对照 `kernels/mixed_gemm_l20/`
- 用 CUTLASS 4.5 CollectiveBuilder 写 dual-phase kernel：
  - Phase 1：`SM100_MMA_F8F6F4_SS`，FP8 E4M3 × FP8 E4M3 → FP32 (TMEM C)
  - Phase 2：`SM100_MMA_MXF4NVF4_SS`，NVFP4 × NVFP4 → FP32 (同一 TMEM C，scaleC=true 累加)
  - 共享 KernelTmaWarpSpecialized schedule + TMA descriptors
- TileShape 起点：128×128×{32,64}（high K = 32, low K = 64，单 instruction K extent）
- ThreadblockShape：128×128（M=128, N=128 占 1-CTA cluster 上限）
- pybind11 暴露 `fused_fp8_nvf4_gemm(A_high_fp8, B_high_fp8, A_low_nvf4, B_low_nvf4, scales_low)`
- **同时实现 2-launch FP baseline 作为对照** —— 一个独立的 `mxfp8_gemm()` + 一个独立的 `nvfp4_gemm()`，Python 层串联调用 + FP32 累加；**这是 PLAQuant single-kernel 论文卖点的硬对照基线，必须自己写**（参考 `kernels/mixed_gemm_l20/mixed_gemm_l20.cu` 的 `baseline_cutlass_mixed_gemm` 在 INT 路径上扮演同样角色）
- 写 `kernels/mixed_gemm_sm100/benchmark.py`，方法对照 `kernels/mixed_gemm_l20/benchmark.py`：每行三列输出（fused / 2-launch FP / FP16 cuBLAS），扫描 (M, N, K_high, K_low) 矩阵覆盖：
  - **M 扫描**：1, 16, 64, 128, 256, 512, 1024, 2048, 4096（batch / context length 全谱）
  - **N 扫描**：2048, 4096, 8192（q/k/v vs gate/up_proj 输出维度）
  - **K_high/K_low 比例扫描**：高比例 1/16, 1/8（默认）, 1/4, 1/2（对应 β-4 per-op variable ratio 的搜索域）
  - **down_proj 形状**：(M, 2048, 1024, 7168)，因为 K_total 大、敏感度不同
  
- 验证 bit-exact (cosine ≥ 0.9999) vs reference FP32 dual-phase 累加
- 验收 (a)：**正确性** —— 整个 shape sweep 上每一行 cosine ≥ 0.9999
- 验收 (b)：**vs FP16 cuBLAS shape sweep characterization** —— 输出 per-shape 加速比表 + heatmap；要求在"LLM inference 真实命中 shape"子集上（即 1B/3B/8B 各层的 (M, N, K_high, K_low)，常见 batch=1/16/64）M=128 上 ≥ 1.5×、M=2048 上 ≥ 2×；其他 shape 退化是允许的，但必须**明确记录在 characterization report 里**（哪些形状下 fused 慢、原因猜测、是否值得后续 tune）
- 验收 (c)：**vs 2-launch FP baseline shape sweep characterization** —— 同样输出 per-shape 加速比表；**这一条最核心**——比 (b) 更接近 PLAQuant 论文卖点，因为它直接量化 single-kernel 拓扑相对 sequential 2-launch 的收益，**不依赖任何 FP16 比较**。要求 LLM inference 真实命中 shape 上 > 1.0×（目标 1.05-1.20×）；如果 fused 在所有 LLM 形状上都 ≤ 2-launch，PLAQuant single-kernel 卖点失败，需要回到 kernel design 反思（tile shape、SMEM 分配、warp specialization 哪个环节没拿到收益）
- **deliverable**：`kernels/mixed_gemm_sm100/RESULTS.md`，包含：(1) shape sweep 三列 latency 表；(2) 加速比 heatmap（fused/FP16, fused/2-launch）；(3) 哪些 shape regime fused 赢、哪些输；(4) per-stage profile（TMA load / WGMMA compute / accumulate / store 各占多少），用来归因性能

**β-3 端到端集成 + 速度验证（~3-5 天）**
- `promix/inference/weight_packer.py` 加 FP8/NVFP4 weight packing 路径
- `promix/inference/quant_ops.py` 加 FP8/NVFP4 activation per-token quant（FP8 不需 shift/zero，NVFP4 需 block scale 计算）
- `promix/inference/real_forward.py` 新增 FP path 分支
- EVT epilogue 把 dequant 折进 kernel（沿用 SM90 EVT 范式，参考 `kernels/mixed_gemm/mixed_gemm.cu` lines 49-92）
- 跑 1B real-FP 端到端 PPL eval（与 fake-FP PPL 对齐，且 ≤ 11.70）
- 跑 1B real-FP 端到端 latency 三组对照：vs FP16 cuBLAS、vs 2-launch FP path（用 β-2 写的独立 mxfp8_gemm + nvfp4_gemm Python 串联）、vs 现有 fake quant baseline；输出三组数据
- **验收 (1B)**：PPL ≤ 11.70 ✓，latency vs FP16 ≥ 1.0× ✓，latency vs 2-launch FP > 1.0× ✓
- **3B、8B 顺次完整重跑**：每个模型同样三组数据，三个模型都过才算 β-3 通过；只过 1B 不算
- 如果 end-to-end 性能不达标但 β-2 micro shape sweep 显示 kernel 自己赢，**触发归因调查**：是 Python 侧 quant pack 开销吃掉 kernel 收益（→ β-3 末尾 EVT epilogue fuse），还是 layer 形状跟 micro sweep 的 LLM-shape subset 不匹配（→ 重跑 sweep 加 missing shape）；定位之前不算项目完成

**β-4 算法侧扩展：per-operator variable high/low ratio（β-3 之后，可独立推进）**

ResQ 论文当前对每个算子一律用 `high_fraction = 1/8`（即 K_high : K_low = 1 : 7）。这个比例是为简化设计选的固定值，不是每个算子的最优解——同一模型里 attention 的 q/k/v 与 MLP 的 gate/up/down 对量化噪声的敏感度天然不同，down_proj 由于 intermediate_size 大、累加路径长，往往比 attention 投影更敏感。固定 1/8 在敏感算子上精度保守、在不敏感算子上精度浪费。

β-4 把 high_fraction 从模型级常量解放成**每个 Linear 的独立旋钮**，搜索空间被两条硬件约束界定：
- K_high 必须是 **32 的倍数**（FP8 `kind::f8f6f4` 单条 instruction K extent = 32 elements）
- K_low 必须是 **64 的倍数**（NVFP4 `kind::mxf4nvf4` 单条 instruction K extent = 64 elements；同时 ≥ block scale vector size = 16，已被 64 倍数覆盖）
- K_high + K_low = layer.K_total（fixed）

举例 Llama-3.2-1B 的 q_proj（K_total = 2048）的可行 (K_high, K_low) 配对：
- (32, 1984)、(64, 1984)、(64, 1920)、(96, 1920)、(96, 1856) ... 一直到 (2048, 0)
- 即 K_high ∈ {32, 64, 96, 128, ..., 2048} ∩ {x : (2048 - x) % 64 == 0}

工作内容：
- **重构 config schema**：把 yaml 的 `quantize.high_fraction: 0.125`（标量）允许变成 `quantize.high_fraction_per_layer: {q_proj: 0.10, k_proj: 0.10, v_proj: 0.10, o_proj: 0.20, gate_proj: 0.05, up_proj: 0.05, down_proj: 0.15}` 这种 dict（向后兼容标量 broadcast）
- **per-layer 量化器与 weight packer 适配**：现有 ActQuantWrapper 已经按 layer 实例化，传入不同 high_fraction 不需要改架构
- **搜索策略（候选）**：
  - 静态启发式：按算子类型固定（attention vs MLP，down_proj 高、其他低）
  - 方差累积阈值：在 PCA basis 的 eigenvalue spectrum 上选累积方差 ≥ τ 的前 k 个 channel 当 high，对每层独立选 k（受硬件对齐 round-up）
  - learnable / 网格搜索：固定总 bit budget，搜每层 high_fraction 的 PPL 最低组合（贵但精确）
- **kernel 不需要改**：β-2 写出来的 SM100 kernel 已经按 (M, N, K_high, K_low) 参数化（参考 `kernels/mixed_gemm_l20/benchmark.py` 中的 test_cases，本来就枚举不同 K_high/K_low 组合），β-4 只是给不同 layer 实例传不同的 K 配置
- **验收**：在 1B/3B/8B 上证明 per-op variable ratio 比固定 1/8 至少 PPL 不退化、最好略改善（< -0.2 PPL）；同时 average bit-width 不上升（即不是靠堆 high 换 PPL，而是把 high 配额从不敏感层挪到敏感层）

这一步把项目从"PLAQuant 系统层 + ResQ 固定算法"扩展成"PLAQuant 系统层 + ResQ-Adaptive 算法"，是一个独立可发表的算法贡献，且不依赖 SM100 native（在 SM80 INT4+INT8 kernel 上同样适用——只是 K 对齐约束变成 INT4 K%64=0 / INT8 K%32=0）。

### Objective Evidence

- `kernels/mixed_gemm_l20/mixed_gemm_l20.cu`（571 LOC，SM80 PLAQuant kernel）：完整保留架构作为 SM100 移植的拓扑模板。Phase split / shared accumulator 的设计直接迁移。
- `kernels/mixed_gemm/mixed_gemm.cu`（410 LOC，SM90 CollectiveBuilder + WGMMA + EVT）：CUTLASS 3.x 范式参考，SM100 collective 写法在此基础上替换 atom 即可。
- `include/cute/atom/mma_traits_sm100.hpp`：SM100 atom 完整目录（21 个 atom family），其中 `SM100_MMA_F8F6F4_*` 和 `SM100_MMA_MXF4NVF4_*` 是 PLAQuant-SM100 双 phase 直接使用的两个。
- `include/cute/arch/mma_sm100_umma.hpp` lines 993-1197：`tcgen05.mma.kind::i8 / f16 / tf32 / f8f6f4 / mxf8f6f4 / mxf4nvf4` 全部 PTX 字面量；确认 INT4 不存在，FP8/NVFP4 native 存在。
- atom 形状约束（直接来自 `mma_traits_sm100.hpp`）：M ∈ {64, 128}，N 步长 8 (B K-major) 或 16 (B MN-major) 介于 8 到 256，K 由 256/sizeof_bits<ValTypeA> 推导（FP8 K=32，FP4 K=64）。
- B20Z 实测 micro：FP16 cuBLAS 峰值 1434 TFLOPS（=5th-gen TC 工作中）；当前 SM80 fused 58 TFLOPS（差 25×）；理论 SM100 FP8 ~3 PFLOPS、NVFP4 ~6 PFLOPS。
- `promix/quantize/basis.py`：PCA basis 计算与数据格式无关，q/k/v/gate/up 的 hidden_dim PCA 部分可直接被 ResQ-FP 复用；o_proj 部分要切到 global mode 重算（详见 PRIMARY 中"配套算法侧改动：o_proj input PCA 升级"段，~20-45 min/model 一次性 cost）。
- `promix/configs/llama-3.2-1b-resq.yaml`、`llama-3.2-1b-w4a4kv4.yaml` 等：现有 yaml 定义了 `high_bits` / `low_bits` / `high_fraction` 等参数；ResQ-FP 只需扩展枚举允许 FP8 / NVFP4 标识，不改 schema。
- `kernels/mixed_gemm_l20/setup.py` 已含 `-gencode arch=compute_100,code=sm_100`（commit f9afacd）；新 `kernels/mixed_gemm_sm100/setup.py` 同模板即可。
- `tests/test_mixed_gemm.py`：现有 INT 路径正确性测试；为 ResQ-FP 写对应 FP 测试是机械工作。

### Known Risks

- **NVFP4 quantization PPL 风险**：NVFP4 比 INT4 数值范围大但 mantissa 更少（FP4 E2M1 只有 1 bit mantissa）；ResQ 的 PCA + 旋转 + variance-based split 是为 INT 设计的，FP4 上是否同样有效需要 β-1 验证。如果 1B fake-FP PPL 退化 > 1，需要回到设计层面调（提高 high_fraction？换 FP6+FP4？）
- **o_proj global PCA 实现风险**：数学上 global PCA 严格不弱于 per-head（block-diagonal subset of full O(2048)），但实现上要正确处理：协方差矩阵从 64×64 变 2048×2048（内存与数值稳定性）、特征向量按方差排序、跟 R2 (per-head, attention 内部) 的组合关系正确。若 β-1 (a-pre) 实测 INT 路径 global PCA PPL 比 per-head 退化 > 0.3，**绝大概率是 bug 不是方案错**——sanity 工具：layer-wise 输出 cosine 对比 per-head vs global、per-channel variance scatter plot、检查 U×R 合成 weight 的旋转 unitarity。屡 debug 不通才触发 Alt-4 回退 per-head + cross-head rearrange。
- ~~MXFP8 E8M0 scale 精度风险~~：先前担心 E8M0 (power-of-2 only) scale 在 outlier-heavy block 上精度不够，但 **ResQ 的 PCA + Stiefel R + (down_proj) Hadamard 三层旋转专门把 outlier 从 single-channel 模式打散成 per-block 平滑的 Gaussian-like 分布**，进 GEMM 时不存在 "block 内 31 个小值 + 1 个 outlier 100" 这种 worst case。E8M0 在 ResQ-rotated 数据上是足够的（element FP8 mantissa 兜底 ±√2 范围细调；block 间最优 scale 也接近 power-of-2）。**ResQ 的旋转 pre-conditioning + microscaling 的 per-block scale 自适应是天作之合**——风险撤销。残余 caveat：β-1 整体 PPL 不达标时仍需诊断，但 E8M0 不再是首要怀疑对象
- **TMEM 学习曲线**：tcgen05 的 TMEM 编程模型（descriptor 寻址、warp specialization、persistent kernel）和 SM80 的 register fragment 完全不同；CUTLASS 4.5 抽象掉一部分但底层细节多；β-0 minimal PoC 是为了把这段学习成本前置。
- **CUTLASS 4.5 SM100 文档稀疏**：CUTLASS GitHub 上 SM100 examples 数量远少于 SM80/SM90；需要直接读 traits / collective 源码理解 API。但 mma_sm100.hpp / mma_traits_sm100.hpp 注释相对完整。
- **Mixed FP8+NVFP4 single-kernel 公开案例几乎为零**：CUTLASS samples 里能找到单独 FP8 GEMM、单独 NVFP4 GEMM，但**两个 phase 共享 TMEM accumulator 的混合 kernel 没有现成参考**——这正是 PLAQuant 论文卖点的核心；意味着开发上要自己拼。
- **NVFP4 block scale 的内存带宽与 layout**：每 16 个 FP4 element 要带一个 FP8 scale，weight packing 时 scale 张量与数据张量怎么布局（interleaved vs separate）影响 TMA descriptor 设计；β-0 PoC 必须解决这个。
- **算法侧重训成本**：rotation R 在 INT loss 上训过，FP loss 上要重新拟合；1B 上单次 ~1 min，3B ~5 min，8B ~10 min；多次实验代价不大但要有 budget。
- **2-launch FP 可能已经接近 single-kernel 上限（PLAQuant 论文 claim 风险）**：SM100 的 TMA + warp specialization + 大 SMEM 让独立的两次 FP launch 之间 overhead 比 SM80 时代低很多。如果 micro sweep 上 fused 比 2-launch 只快 1.01-1.04×，**PLAQuant single-kernel 系统贡献的量化证据就很弱**——能写论文但 claim 不硬。Mitigation：β-0 minimal PoC 时**先把 2-launch FP baseline 也写出来**（不只是 fused），即早测 fused vs 2-launch 单点比较；如果该比较已经 < 1.05×，提前调整 fused kernel 的 tile shape / SMEM 分配 / warp specialization 策略，争取拉开差距。如果调到极限仍 < 1.05×，触发 Alt-1 (path α) 退路，至少保 deliverable。
- **B20Z idle reaper**：cluster 不看 GPU 活动而是看 portal 网页活动（已踩过坑）；β-2 kernel 编译 + benchmark 跨多次 session，需要 portal 保持开 + bring_up_remote.sh 恢复脚本。

## Alternative Directions Considered

### Alt-1: Path α — Pure W8A8 SM100 Native (INT4 sign-extend → INT8)

- Gist: 不改算法，把现有 ResQ INT4+INT8 在 SM100 上跑 native—— INT8 path 用 `tcgen05.mma.kind::i8` 直接，INT4 path 把 4-bit 数据 sign-extend 到 INT8 再走 `kind::i8`（与 SM90 native 缺 INT4 时同思路）。结果是 effective W8A8（INT4 的 2× 压缩消失），但完全不用动 PTQ pipeline，可能 1-2 周拿到 ~2× FP16 加速。算作"PRIMARY 失败时的 fallback"或"短期内拿数据"路径。
- Objective Evidence:
  - SM90 git 历史 commit `987d3c9`、`5992d48`：明确 SM90 同样无 INT4 native，sign-extend 到 INT8 是已知工程做法。
  - `SM100_MMA_S8_*` atom family 完整可用（4 个 sub-variant）。
  - 现有 PTQ pipeline 完全不动；现有 1B/3B/8B PPL 数据直接 carry over。
  - 缺点：PLAQuant 的"single-kernel multi-precision"在 α 下退化为"single-kernel single-precision"，论文层面没有 system-level 贡献，只是用了 SM100 native。
- Why not primary: 可以 fallback 但不是值得做的目标——保留 ResQ 算法的 INT8/INT4 split 但 hardware 上没有 INT4 通道，等于 split 完了又拼回去；既不是新 system 也不是新 algorithm。

### Alt-2: Profile-Driven End-to-End Bottleneck Map

- Gist: 给 `promix/inference/real_forward.py` 每个阶段插 `torch.cuda.Event`，输出 per-layer + per-stage 的 latency 占比 JSON 报告。在 PRIMARY 下，这是阶段验收基础设施——β-1 fake quant 后看 PPL 哪个 layer 退化最多、β-3 端到端后看剩余 gap 在哪儿。本身不产生加速。
- Objective Evidence:
  - `promix/inference/benchmark.py` lines 17-37：现成的 `measure_latency` helper 用 CUDA events，但只测端到端总时长。
  - `tests/bench_h20_peak.py` lines 24-77：cold/hot CUDA event 模式。
  - `promix/inference/real_forward.py` lines 57-143：10 个阶段无 timing checkpoint。
  - `promix/inference/quant_ops.py` lines 6-38, 51-64：4096 次/forward 调用，零 instrumentation。
- Why not primary: 不解决"kernel 自己跑不过 FP16"这个核心问题。在 PRIMARY 内部当 β 阶段验收工具用即可。

### Alt-3: Path γ — Defer SM100, Validate PLAQuant on H100/L40 with Native INT4

- Gist: 不动现有 INT4+INT8 kernel，找还有 INT4 native MMA 的硬件（A100 SM80、L40 SM89、H100 SM90 都有 `mma.sync.m16n8k64.s4s4.s32`）跑 micro + 端到端，证明 PLAQuant 设计在合适硬件上确实拿到加速比。把 B20Z 留给 β 当下一代 prototype。短期能在 paper 里 claim "PLAQuant 在 Hopper/Ada 上拿到 X× FP16 加速"，长期再补 Blackwell。
- Objective Evidence:
  - `kernels/mixed_gemm_l20/mixed_gemm_l20.cu`：现有 SM80 kernel 已经支持。
  - DEVLOG 历史 1B PPL=14.72 + benchmark 1.12-1.19× speedup 是在 H20 (SM90a) 上的数据，未在 H100/L40 上重测。
  - 远程容器从 H20 换到 B20Z 是基础设施约束，但本地 8× L20 (SM89) 没 nvcc 不能编 kernel；要走 γ 需要换/借 H100/L40 算力。
- Why not primary: 算是科研策略，不是技术方向；不解决 Blackwell 上跑不动这个长期问题；只是把锅往后推。

### Alt-4: Fallbacks If o_proj 设计的两层假设 Fail

- Gist: PRIMARY 在 o_proj 上做了两层"假设 fail 才用 alt"的依赖：(1) input PCA 从 per-head 升级到 global，假设 INT 路径 PPL 持平或微好；(2) NVFP4 microscaled (block=16) 替换 ResQ per-group=64，假设 ≥ per-group。每个假设各自有 fallback：
  - **β-1 (a-pre) 失败**（global PCA 在 INT 路径退化 > 0.3，且 debug 后排除 bug）：回退 **per-head PCA + cross-head rearrange** 这条原 ResQ 思路——每 head 内 8 hi + 56 lo，concat 成 [所有 head hi (256) | 所有 head lo (1792)]，K 对齐 ✓；代价是 MXFP8 block-of-32 跨 4 head 共享 scale、NVFP4 block=16 大部分 within-head 但每 7 个 block 跨边界一次。R1+R2 旋转 smooth head 间统计差异是这个方案能站住的前提。
  - **β-1 (b) 失败**（NVFP4 microscaled o_proj 在 layer-wise cosine 上偏离 INT per-group > 0.01）：fallback 选项 (a) 提高 o_proj 的 high_fraction（β-4 per-op variable ratio 提供旋钮，从 1/8 调到 1/4 或 1/2）、(b) 在 epilogue 里加 per-group scale 索引做 group-aware dequant、(c) 接受 o_proj 退回 fake quant 留作未来工作。
  - 这两个 fallback 互不依赖，可以独立启用；同时 fail 的概率极低，但流程上要分别 handle。
- Objective Evidence:
  - `promix/quantize/basis.py`：当前 o_proj 走 per-head 64×64 PCA，q/k/v/gate/up 走 hidden_dim 2048×2048 global PCA；改 o_proj 到 global 仅需新增分支，hidden_dim 路径直接复用
  - `promix/inference/weight_packer.py` lines 34-36：`if wrapper.quantizer.groupsize > 0: continue` 显式 skip — 旧 INT 路径的特殊处理代码，PRIMARY 完成后这条 skip 应该删除
  - `promix/inference/real_forward.py` line 149：`"Skips layers with per-group quantization (o_proj) — they keep fake quant."` 注释 — 同上，PRIMARY 后失效
  - `inference/forward_pass.py` lines 78-86, 215-226（旧路径）：ResQ per-group activation quant 实现，groupsize=head_dim=64；fallback (a-pre) 启用时这段逻辑可作为 cross-head rearrange 设计的参考
  - `tests/test_mixed_gemm.py` 的 `o_proj_dir()` fixture：测试数据齐全，可用于 β-1 验收点 (b) 的 cosine 对比
- Why not primary: PRIMARY 在 o_proj 上做了"global PCA + microscaled subsumption"两个假设以最小化复杂度（不需要 cross-head rearrange，不需要 group-aware kernel）；这两个 alt 都是 contingency，**只有 β-1 对应验收失败才启用**。

### Alt-5: Custom Op + FX Graph Rewriter Integration

- Gist: 把 kernel 注册成 `torch.library.custom_op` + FX 图重写自动替换 `nn.Linear`，干掉 `install_real_forward()` lambda monkey-patching。架构 / 可维护性提升，让任何 HF 模型自动受益。在 PRIMARY 下作为 β-3 之后的代码质量收尾，不是关键路径。
- Objective Evidence:
  - `promix/inference/real_forward.py` lines 146-164：当前 lambda 模式。
  - 仅 2 个 call site（`real_forward.py:159` 与 `benchmark.py:181`）。
  - `kernels/mixed_gemm_l20/setup.py` lines 9-31：pybind11 已暴露 `torch::Tensor` 签名，与 `torch.library` 完全兼容。
  - 全 repo 无 `torch.library` / `torch.fx` 既有用法——全新 infra。
- Why not primary: 不解决 perf 问题。在慢 kernel 外面套漂亮 op API 没意义；等 PRIMARY 拿到加速再加 wrapper。

## Synthesis Notes

补做 vs FP16 cuBLAS 的 micro benchmark + 核查 CUTLASS 4.5 SM100 atom 目录之后，问题被重新定位了两次：第一次从"Python overhead 主导"重定位到"kernel 自己慢 25×"，第二次从"SM100 INT4+INT8 native 升级"重定位到"SM100 INT4 不存在，必须迁 FP8+NVFP4"。最终的 PRIMARY（PLAQuant-SM100 FP8+NVFP4 single-kernel）保留了原项目两个核心贡献——系统层的 single-kernel multi-precision 拓扑，算法层的 PCA + variance-split + 可学习旋转——只把"INT4+INT8 atom"这个具体实现层换成 SM100 上等价的"FP8+NVFP4 atom"。M, N 对齐（PLAQuant SM80 上的 8×8 tile 共享原则）在 SM100 上自然继承到 TMEM accumulator 上。

阶段 β-0 → β-1 → β-2 → β-3 是有先后依赖的硬流水：算法不通（β-1 PPL 退化）就不该花时间写 kernel（β-2）；kernel 不通（β-2 vs FP16 < 1×）就不该花时间集成（β-3）。每个阶段都有可量化的 go/no-go 验收点，避免把整个赌注押到最后才发现路走错。

Alt-1（α）是 PRIMARY 真撞墙时的安全网（保 deliverable，认怂论文卖点）。Alt-2（profile）是 PRIMARY 阶段验收工具（可以放在 β-1 末尾接进来，量化 PPL 退化的 layer-级归因，引导调参）。Alt-3（γ）是 PRIMARY 同期可以并行的科研策略（在 H100/L40 上跑现有 INT 版本拿 paper 数据，不影响 Blackwell 主线）。Alt-4（o_proj）和 Alt-5（FX）是 PRIMARY 跑通后的覆盖率与代码质量收尾，不在关键路径上。

如果让我此刻只能选一件事开工，是 β-0：用 ~半天写两个 minimal SM100 单 phase kernel（FP8 一个、NVFP4 一个），跑 vs FP16 cuBLAS 的 micro，把"SM100 native 路径理论 ~2-4× FP16 加速"从理论变成实测。这一步成本极低（不接 ResQ、不接 dequant、~80 LOC 一个 kernel）但决定后面所有阶段值不值得做——如果 minimal FP8 kernel 都跑不出 1.5× FP16，整个 β 路径都得重新评估。
