# ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

**论文信息**
- **标题**: ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals
- **作者**: Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, Xin Wang (Purdue University + d-Matrix)
- **发表**: ICML 2025
- **arXiv**: 2412.14363v2
- **代码**: https://github.com/utkarsh-dmx/project-resq

---

## 一、背景知识

### 1.1 LLM 推理的两个阶段

LLM 推理分为两个阶段：
- **Prefilling（预填充）**: 处理输入 prompt，填充 KV cache。此阶段是 **compute-bound**，需要大量浮点运算。
- **Generation（生成）**: 逐 token 自回归生成。此阶段是 **memory-bound**，瓶颈在于反复读取和更新 KV cache。

现代 LLM 参数量可超 400B，加上长上下文需要的大 KV cache，推理成本极高。量化是降低推理成本的核心手段。

### 1.2 量化基础

N-bit 整数量化和反量化过程：

```
Q(X) = round_clip((X - z_X) / s_X) * s_X + z_X
```

其中 `s_X` 是 scale，`z_X` 是 zero-point。

两种模式：
- **对称量化**: `z_X = 0`, `s_X = max(|X|) / (2^(N-1) - 1)`
- **非对称量化**: `z_X = min(X)`, `s_X = (max(X) - min(X)) / (2^N - 1)`

### 1.3 量化的两大类别

| 类别 | 说明 | 代表方法 |
|------|------|----------|
| **均匀精度量化 (UPQ)** | 所有层使用相同 bit-width | GPTQ, AWQ, QuaRot, SpinQuant |
| **混合精度量化 (MPQ)** | 根据层/通道敏感度自适应 bit-width | LLM.int8(), QUIK, Atom, ResQ |

### 1.4 激活量化的核心挑战：Outlier 问题

LLM 激活中存在极端 outlier（可达其他值的 ~20 倍），导致低精度量化误差极大。

目前有两条主要技术路线应对：

1. **Outlier 检测 + 混合精度**: 识别 outlier channel，保留高精度（如 LLM.int8(), QUIK）
2. **随机旋转 + 均匀精度**: 用正交旋转矩阵压制 outlier，使分布更均匀（如 QuaRot, SpinQuant）

**ResQ 的核心 insight**: 将这两条路线统一——先用 PCA 做最优"分流"，再在子空间内施加随机旋转做"平滑"。

### 1.5 PCA 原理详解

PCA（主成分分析）是 ResQ 的数学基础。其本质是**对协方差矩阵做特征分解，找到数据方差最大的正交方向，把原来相互纠缠的维度旋转成互不相关且按重要性排序的新维度**。

#### 1.5.1 数学推导

给定 n 个 d 维数据 `X ∈ R^{n×d}`（每行一个样本）：

**Step 1: 协方差矩阵**

```
C = (1/n) X̃ᵀ X̃    ∈ R^{d×d}
```

- `C[i][i]` = 第 i 维的方差（自身变化程度）
- `C[i][j]` = 第 i 维和第 j 维的协方差（两个维度的联动关系）

**Step 2: 特征分解**

```
C = V Λ Vᵀ
```

- `V = [v₁, v₂, ..., v_d]` — 特征向量（正交的），每个是一个方向
- `Λ = diag(λ₁, λ₂, ..., λ_d)` — 特征值，λᵢ 表示数据在 vᵢ 方向上的方差

**特征值越大 → 数据在那个方向上越"散" → 包含的信息越多。**

**Step 3: 投影降维**

```
Z = X̃ · V_k    ∈ R^{n×k}    (投影到前 k 个主成分)
```

#### 1.5.2 为什么特征向量是"最优方向"

PCA 等价于解优化问题：找单位向量 v，使数据投影后方差最大。

```
max_v  vᵀCv
s.t.   vᵀv = 1
```

用拉格朗日乘子法：`∂L/∂v = 2Cv - 2λv = 0 → Cv = λv`，这正是**特征值方程**。

所以：
- **第一主成分** v₁ = 最大特征值 λ₁ 对应的特征向量（方差最大方向）
- **第二主成分** v₂ = 第二大特征值对应的特征向量（与 v₁ 正交、方差第二大）
- 以此类推...

#### 1.5.3 PCA 在 ResQ 中的应用

对 LLM 某层的激活 `X ∈ R^{n×d}`（n 个 token，d 维隐藏状态）：

```
协方差矩阵:  C = XᵀX ∈ R^{d×d}
特征分解:    C = V Λ Vᵀ

按特征值从小到大排列:
  v₁, v₂, ..., v_{d-r}     → 方差小的方向 → P_l（低精度 4-bit）
  v_{d-r+1}, ..., v_d       → 方差大的方向 → P_h（高精度 8-bit）
```

PCA 只是"旋转了坐标系"，没有丢失任何信息，只是换了一种看数据的角度，使得重要性一目了然。

### 1.6 为什么混合精度 4-bit/8-bit 而不是全 4-bit？

#### 量化误差与 bit 数的关系

量化误差的关键在于 scale factor: `s = max(|X|) / (2^N - 1)`

假设数据范围是 [-100, 100]：
- **4-bit**: 16 个量化 level，步长 = 200/15 ≈ **13.3**（最大误差 6.7）
- **8-bit**: 256 个量化 level，步长 = 200/255 ≈ **0.78**（最大误差 0.39）

**8-bit 的精度是 4-bit 的 17 倍。**

#### 混合精度的性价比

| 方案 | 平均 bit 数 | 效果 |
|------|------------|------|
| 全 4-bit | 4.0 | 高方差分量误差极大，模型精度崩溃 |
| 全 8-bit | 8.0 | 精度好，但没有加速意义 |
| ResQ: 7/8 用 4-bit + 1/8 用 8-bit | **4.5** | 用 0.5 bit 的额外预算保护最关键分量 |

只多花了 **12.5% 的存储**（4.5 vs 4.0），就把最危险的 1/8 分量的量化误差降低了 17 倍。

#### 为什么不用 16-bit 保护高方差分量？

硬件效率问题：
- **4-bit × 4-bit** 和 **8-bit × 8-bit** 的 GEMM kernel 在 GPU 上有高效原生实现
- **4-bit × 16-bit** 没有原生支持，需要类型转换，反而更慢

ResQ 通过正交性保证低精度和高精度分量不会交叉相乘，只需两种同精度 GEMM。

### 1.7 ResQ 为什么优于之前的方法？——逐一对比

理解 ResQ 的价值，需要搞清楚之前三类方法各自的缺陷。

#### 方法 A: 只做 PCA 分解（SliceGPT、ESPACE）

这类方法用 PCA 降维——直接**丢掉**低方差维度：

```
SliceGPT:  [保留前 3072 维] [丢弃后 1024 维]    ← 信息永久丢失
ResQ:      [前 7/8 维 → 4-bit] [后 1/8 维 → 8-bit]  ← 信息都在，精度不同
```

**问题**: 丢弃的维度信息永远回不来。即使方差小，累积影响也不可忽略。ResQ 不丢弃任何维度。

#### 方法 B: 只做 Hadamard 旋转（QuaRot）

用 Hadamard 矩阵旋转激活，把 outlier **均匀摊开**到所有维度，然后统一 4-bit：

```
旋转前:  [1, 1, 1, 100, 1, 1]    ← outlier 集中在一个维度
旋转后:  [42, 41, 43, 40, 42, 41] ← 摊平后每个维度差不多大
```

**隐藏问题**: 摊开 outlier 的代价是**所有维度的量化范围都被拉大了**。原来大部分维度范围是 [-1, 1]，现在变成 [-42, 42]。对于那些原本值很小的维度：
- 旋转前步长: 2/15 ≈ **0.13**
- 旋转后步长: 84/15 ≈ **5.6**

**误差反而变大了 40 多倍。** 旋转只是"均匀化"了误差，没有"消除"误差。这就是为什么 QuaRot 在 4-bit 下仍然比 16-bit 差 ~20% perplexity。

#### 方法 C: ℓ∞ norm 选 outlier + 旋转（QUIK 的思路）

QUIK 的做法：按每个 channel 的 `|max|`（ℓ∞ norm）找 outlier → 分成高低精度组 → 组内做旋转。

**注意: QUIK 没有使用 PCA，它的 outlier 选择是纯启发式的。**

**问题 1: ℓ∞ norm 会被单个极端值误导**

```
Channel A: [0.1, 0.1, 0.1, ..., 100]   ← ℓ∞ = 100，被选为 outlier
Channel B: [30, -25, 35, -28, 32, ...]  ← ℓ∞ = 35，没被选

但 Channel B 的方差远大于 Channel A（A 只是偶尔有一个尖峰）
量化 Channel B 的误差其实远大于 Channel A
```

ℓ∞ norm 被**单个极端值**误导了，而 PCA 看的是**整体分布的方差结构**。

**问题 2: 全局旋转可能破坏分组结构**

如果在分组后对整个空间做旋转，会把已经分好的高低精度组重新搅混，分组边界模糊化。

#### ResQ 的方案: 先分、后转、子空间内转

```
                    原始激活空间 (d 维)
                         │
                    PCA 投影 (P)        ← 按全局方差结构分流
                    ╱           ╲
          低方差子空间            高方差子空间
          (d-r 维)               (r 维)
              │                      │
         随机旋转 R_l            随机旋转 R_h    ← 子空间内部平滑
              │                      │
        子空间内 outlier         子空间内 outlier
        被压制                   被压制
              │                      │
          4-bit 量化              8-bit 量化
```

**解决了所有前述问题**:

| 前述方案的问题 | ResQ 如何解决 |
|--------------|-------------|
| 只做 PCA 会丢维度 | 不丢维度，只是分精度 |
| 全局旋转会拉高所有维度的范围 | 只在子空间内旋转，不跨组污染 |
| 全局旋转会破坏分组结构 | 先分组再旋转，旋转不跨组 |
| ℓ∞ norm 选 outlier 被极端值误导 | PCA 按全局方差结构分组，理论更优 |

**一句话总结**: PCA 负责"分清主次"，随机旋转负责"内部平滑"，两者在各自的职责范围内工作，互不干扰。之前的方法要么只做一件事（效果有限），要么两件事做了但顺序或范围搞错了（互相破坏）。

### 1.8 值得进一步思考的问题

虽然 ResQ 的设计逻辑通顺，但有几点仍值得质疑：

1. **低方差 ≠ 不重要**: PCA 按方差排序，但方差小的方向对模型输出可能有高灵敏度（如某些 attention pattern）。更好的标准可能是 **Hessian 信息**（类似 GPTQ），按"量化这个方向对 loss 的影响"来排序。

2. **理论保证的局限**: Theorem 4.2 最小化的是**量化误差范数**，但量化误差范数小 ≠ 模型精度影响小。这是一个有用的近似，但不是完美的。

3. **固定 r=d/8**: 不同层的方差分布差异大，per-layer adaptive rank 可能更优。

---

## 二、相关工作详解

### 2.1 权重量化方法

| 方法 | 核心思想 | 局限 |
|------|----------|------|
| **GPTQ** (Frantar et al., 2022) | Hessian 引导的逐列舍入，最小化量化误差 | 仅处理权重，不处理激活 |
| **AWQ** (Lin et al., 2024c) | 通道级 scaling，保护显著权重 | 同上 |
| **QuIP/QuIP#** (Chee et al., 2024) | 自适应舍入 + 随机旋转 | 需要 incoherence processing |
| **AQLM** (Egiazarian et al., 2024) | 多码本量化 | 解码开销大 |

### 2.2 权重-激活联合量化方法

| 方法 | 核心思想 | 局限 |
|------|----------|------|
| **SmoothQuant** (Xiao et al., 2023) | 将激活难度通过 scaling 转移到权重 | 4-bit 激活仍然困难 |
| **QuaRot** (Ashkboos et al., 2024c) | Hadamard 旋转消除 outlier，均匀 4-bit 量化 | 依赖 Hadamard 矩阵存在性（维度必须是 2 的幂），在 Qwen2.5 上崩溃 |
| **SpinQuant** (Liu et al., 2024b) | 学习旋转矩阵（需梯度优化），均匀 4-bit | 需要训练，成本约为 ResQ 的 3 倍 |
| **DuQuant** (Lin et al., 2024b) | 双重旋转增强低精度鲁棒性 | 计算开销较大 |

### 2.3 KV Cache 量化方法

| 方法 | 核心思想 |
|------|----------|
| **KIVI** (Liu et al., 2024a) | 非对称 2-bit KV cache 量化 |
| **KVQuant** (Hooper et al., 2024) | 非均匀量化 + 重要性感知精度 |
| **GEAR** (Kang et al., 2024) | 全 token 量化 + 低秩误差补偿 |

### 2.4 低秩分解方法

| 方法 | 核心思想 | 与 ResQ 的关系 |
|------|----------|---------------|
| **SliceGPT** (Ashkboos et al., 2024a) | PCA 投影权重矩阵做稀疏化 | ResQ 用 PCA 投影激活做量化分组 |
| **ESPACE** (Sakr & Khailany, 2024) | PCA 降维压缩激活 | 高度相关，ResQ 需更清楚区分 |
| **ASVD** (Yuan et al., 2023b) | 激活感知的 SVD 权重分解 | 不同目标但同源技术 |
| **Eigen Attention** (Saxena et al., 2024) | 低秩近似降低 KV cache 内存 | 同一作者的前序工作 |

### 2.5 Outlier 检测型混合精度方法

| 方法 | Outlier 选择策略 | 与 ResQ 的关键区别 |
|------|-----------------|-------------------|
| **LLM.int8()** (Dettmers et al., 2022) | 按绝对值大小选 outlier channel | 启发式，非最优 |
| **QUIK** (Ashkboos et al., 2024b) | ℓ∞ norm 选 outlier channel | ResQ 证明 PCA 选择优于 ℓ∞ norm |
| **Atom** (Zhao et al., 2024) | 通道重排 + 显著性驱动 bit 分配 | 更复杂但未必更优 |

---

## 三、ResQ 方法详解

### 3.1 核心思想

ResQ 将激活投影到正交基上，然后：
1. **PCA 分流（P）**: 按协方差矩阵特征值排序，最高方差的 r=d/8 个方向保 8-bit，其余保 4-bit
2. **随机旋转（R）**: 在每个子空间内部施加随机正交矩阵，利用 CLT 使分布更高斯，抑制残余 outlier

组合矩阵 `U_i = P_i · R_i`，其中 i ∈ {h(高精度), l(低精度)}。

### 3.2 量化公式

激活量化：
```
X_q = Q_L(X · U_l) + Q_H(X · U_h)
```
其中 Q_L 是 4-bit 量化，Q_H 是 8-bit 量化。

权重量化：
```
W_q = Q_L(U_l^T · W) + Q_H(U_h^T · W)
```

层输出：
```
X_q · W_q = Q_L(X·U_l) · Q_L(U_l^T·W) + Q_H(X·U_h) · Q_H(U_h^T·W)
```

关键性质：**低精度和高精度分量之间的交叉乘法为零**（正交性），只需同精度 GEMM kernel。

### 3.3 理论最优性 (Theorem 4.2)

PCA 选择最大特征值方向做高精度子空间，可以最小化量化误差的**上界**：

```
E‖X - X_q‖_F ≤ E‖X‖_F · f(L,d,r) - g(L,H,d,r) · E‖X·P_h‖_F
```

要最小化上界 = 最大化 `‖X·P_h‖_F` = 选协方差矩阵最大特征值对应的特征向量。

**注意**: 证明的是上界最优性，非误差本身最优性；高斯假设依赖随机旋转后的近似。

### 3.4 四种投影矩阵的部署

| 投影 | 位置 | 维度 | 运行时开销 | 说明 |
|------|------|------|-----------|------|
| **U_A** | Block boundaries | d_h × d_h | **零** | 融合到相邻层权重 |
| **U_B** | Value projection | d_head × d_head | **零** | 融合到 o_proj 权重 |
| **U_C** | Key projection | d_head × d_head | **有** | 因 RoPE 无法融合，在线计算（量化到 8-bit 减少开销） |
| **U_D** | FFN down_proj | d_FFN × d_FFN | **有** | 因激活函数无法融合，使用 Hadamard 矩阵 + 快速变换 |

### 3.5 PCA Basis 获取流程

1. 收集校准数据（512 个样本，序列长度 2048）
2. 对每个层的激活计算协方差矩阵 `X^T · X`
3. 特征分解，按特征值升序排列
4. 最后 r 列 → P_h（高精度），前 d-r 列 → P_l（低精度）
5. 生成随机正交矩阵 R_h, R_l
6. 组合 U = P · R

---

## 四、实验结果总结

### 4.1 主要结果 (W/A/KV = 4/4/4)

**Llama 系列**（与最强 baseline SpinQuant 比较）：

| 模型 | FP16 PPL | SpinQuant PPL | ResQ PPL | 改善 |
|------|----------|---------------|----------|------|
| Llama-3-8B | 6.1 | 7.4 | **7.1** | 4% |
| Llama-3-70B | 2.9 | 6.2 | **4.1** | 33% |
| Llama-3.2-1B | 9.7 | 13.1 | **12.0** | 8% |
| Llama-3.2-3B | 7.5 | 9.2 | **9.0** | 2% |

**Qwen2.5 系列**（QuaRot 和 SpinQuant 因架构问题表现极差或缺失）：

| 模型 | FP16 PPL | QuaRot PPL | ResQ PPL |
|------|----------|------------|----------|
| Qwen2.5-3B | 8.1 | 68.8 | **9.4** |
| Qwen2.5-7B | 7.3 | 4035.9 | **8.7** |

### 4.2 Speedup (RTX 3090, 单 decoder block, batch=1)

| 精度 | Prefill Speedup | Decode Speedup |
|------|-----------------|----------------|
| W4A4 (ResQ) | 2.45× | 1.61× |
| W4A4 (Uniform) | 2.79× | 1.86× |

ResQ 因混合精度比均匀 4-bit 慢约 14%，但精度显著更高。

---

## 五、批判性评价

### 5.1 核心优点

1. **方法设计优雅**: PCA + 随机旋转的组合简洁有力，有理论支撑
2. **Training-free**: 无需梯度优化，量化成本仅为 SpinQuant 的 1/3（35 min vs 110 min for 8B）
3. **广泛模型覆盖**: 从 0.5B 到 72B，文本和多模态（Qwen2-VL）
4. **Pareto 可控**: rank r 提供精度-效率连续 tradeoff
5. **理论分析方向正确**: 虽然证明的是上界最优性，但给人信心

### 5.2 关键疑虑

#### 5.2.1 Qwen2.5 上的 Baseline 不公平 (严重度: 高)

QuaRot 在 Qwen2.5 上因 Hadamard 矩阵维度问题而崩溃（PPL 高达 4035.9），SpinQuant 完全缺失 Qwen2.5 结果。这意味着 **ResQ 在 Qwen2.5 上的"碾压式"优势很大程度上来自 baseline 的架构兼容性问题**，而非方法本身的巨大优势。论文虽有脚注标注，但未在正文充分讨论。

#### 5.2.2 系统评估薄弱 (严重度: 高)

- 只测了**单个 decoder block**，不是端到端推理延迟
- 只测了 **batch size = 1**（最有利于 memory-bound 优化的场景）
- 只在 **RTX 3090**（消费级 GPU）上测试，缺少 A100/H100 数据
- 缺少 **throughput (tokens/sec)** 指标
- 在线投影 U_C、U_D 的实际 overhead 未在不同序列长度/batch 大小下详细分析

#### 5.2.3 "Up to 33%" 数字的来源 (严重度: 中)

33% 来自 Llama-3-70B（SpinQuant 6.2 vs ResQ 4.1），但 SpinQuant 在 70B 上的数字异常差（甚至不如 QuaRot 的 5.7），暗示 SpinQuant 在 70B 上可能有实现问题或未充分调优。

#### 5.2.4 理论局限性 (严重度: 低)

Theorem 4.2 假设量化后分布满足高斯假设——这依赖于随机旋转后的近似，存在一定程度的循环论证。但整体思路是正确的。

### 5.3 缺失实验

- [ ] 端到端推理延迟（使用 vLLM/TensorRT-LLM 等框架）
- [ ] 大 batch prefill 场景性能
- [ ] 与 QServe (W4A8KV4) 的系统级对比
- [ ] A100/H100 上的性能数据
- [ ] Coding/instruction-following 任务（HumanEval, MT-Bench）
- [ ] Per-layer adaptive rank 的消融实验

### 5.4 遗漏的相关工作

- **SVDQuant** (Li et al., 2024): 用低秩分量吸收 outlier，思路高度相似，论文引用但未在正文讨论
- **FlatQuant / PreQuant**: Concurrent work，探索 learned/PCA-based rotation for quantization
- **QServe** (Lin et al., 2024d): W4A8KV4 系统级实现，直接竞争者，论文提到但未对比

---

## 六、对我们工作的启示

### 6.1 可复现/可扩展的方向

1. **Per-layer adaptive rank**: 不同层的 activation variance 分布差异大，固定 r=d/8 可能非最优
2. **端到端系统集成**: 将 ResQ 集成到 vLLM/SGLang 等推理框架中做真实 benchmark
3. **与其他量化方法组合**: 论文已展示 ResQ + GPTQ 的组合效果，可进一步探索与 AWQ 等的组合
4. **校准数据敏感性研究**: 当校准数据分布与推理分布有 domain shift 时，PCA 子空间是否稳定

### 6.2 核心代码入口

```
project-resq/
├── fake_quant/
│   ├── get_basis.py          # PCA basis 计算（Phase 1）
│   ├── optimize_rotation.py  # 旋转矩阵优化（Phase 2）
│   ├── ptq.py                # 量化评估（Phase 3）
│   ├── collect_activations.py # 激活收集
│   ├── 0_get_basis.sh        # Step 0: 获取 PCA basis
│   ├── 1_optimize_rotation.sh # Step 1: 优化旋转
│   ├── 2_eval_ptq.sh         # Step 2: 评估量化模型
│   └── 4_collect_act.sh      # 收集激活统计
```

---

## 七、总结

**一句话评价**: ResQ 用 PCA 分解取代启发式 outlier detection 来选择混合精度量化的高/低精度分量，并在每个子空间内施加随机旋转，是一个**方法优雅、理论有支撑、training-free** 的 4-bit W/A/KV 量化方案。

**核心 Insight**: 在选择哪些 channel 保持高精度时，基于协方差特征值排序（PCA）优于基于 ℓ∞ norm 的 outlier 检测，因为前者捕获了全局方差结构而非局部极端值。

**对 ICML 2025 来说**: 合理 accept。方法简洁，实验广泛（虽有瑕疵），真正的 selling point 是 training-free 且比 SpinQuant 更轻量。在 Llama 系列上对 SpinQuant 的优势是增量的（2-33%），在 Qwen2.5 上的大幅优势主要来自 baseline 的架构兼容性问题。
