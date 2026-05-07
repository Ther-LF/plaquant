# ASPLOS 2025/2026 Activation+Weight Quantization 相关工作调研

> 调研时间：2026-04-17
> 目标：投稿 ASPLOS 2027（deadline 预计 2026 年 9-10 月）

## 一、ASPLOS 2025（3 篇直接相关）

### 1. COMET: Towards Practical W4A4KV4 LLMs Serving ⭐最强竞品

- **arXiv**: [2410.12168](https://arxiv.org/abs/2410.12168)
- **作者**: Lian Liu, Long Cheng, Haimeng Ren 等（ICT CAS, ShanghaiTech）
- **核心做法**:
  - FMPQ（Fine-grained Mixed-Precision Quantization）：大部分 activation 压到 4-bit，outlier channel 保留 8-bit
  - 自定义 W4Ax 混合精度 CUDA kernel，INT4 Tensor Core
  - 混合精度 data layout + GPU software pipeline + SM-level 负载均衡
- **结果**: kernel 2.88× over cuBLAS，端到端 2.02× over TensorRT-LLM（A100-80G）
- **和我们的关系**: 最直接的竞品。也做 activation mixed-precision + custom kernel，但按 channel 级别分 outlier，我们按 PCA 维度分

### 2. ABQ-LLM: Arbitrary-Bit Quantized Inference Acceleration

- **arXiv**: [2408.08554](https://arxiv.org/abs/2408.08554)
- **作者**: Chao Zeng 等（ByteDance）
- **代码**: [github.com/bytedance/ABQ-LLM](https://github.com/bytedance/ABQ-LLM)
- **核心做法**:
  - 支持任意位宽组合（W2A8, W6A6 等）
  - BTC（Binary TensorCore）分解：把任意精度 matmul 拆成 binary TC 操作
  - Distribution correction 处理 activation 分布偏移
- **结果**: 1.6× over SmoothQuant，2.7× 内存压缩
- **和我们的关系**: 方法论不同（bit-level 分解 vs PCA-based 分组），但都做 W+A quant + custom kernel

### 3. GANQ: GPU-Adaptive Non-Uniform Quantization

- **arXiv**: [2501.12956](https://arxiv.org/abs/2501.12956)
- **作者**: Pengxiang Zhao, Xiaoming Yuan
- **核心做法**:
  - 非均匀量化 + LUT-based mpGEMM（lookup table 混合精度 GEMM）
  - 主要做 weight-only（3/4-bit），GPU-adaptive 优化
- **结果**: 2.57× on RTX 4090
- **和我们的关系**: Weight-only，不直接竞争，但 LUT-based kernel 技术可参考

### 其他 ASPLOS 2025 相关但不直接竞争

| 论文 | arXiv | 做法 | 备注 |
|------|-------|------|------|
| P3-LLM | 2511.06838 | W4A8 + NPU-PIM 硬件设计 | 不是 GPU kernel |
| Fast On-device LLM (NPU) | 2407.05858 | NPU 量化推理 + shadow outlier | 移动端 NPU，非 GPU |
| MVQ | — | Vector quantization | 非整数量化 |

---

## 二、ASPLOS 2026（无直接竞品）

从 ASPLOS 2026 完整 program（89 篇，10.6% 录取率）中筛选，**没有做 activation+weight quant + custom GEMM kernel 的工作**：

| 论文 | 做法 | 为什么不直接竞争 |
|------|------|------------------|
| **Tilus** (NVIDIA) `2504.12984` | Low-precision GPU 编程语言，支持 1-8 bit | Compiler/DSL 工作，不是量化算法。但可作为我们 kernel 的工具 |
| SNIP `2602.01410` | FP4 混合精度 LLM **training** | 做 training 不做 inference |
| oFFN | Outlier-aware structured FFN pruning | 是 pruning 不是 quantization |
| MoE-APEX | Adaptive precision expert offloading | MoE offloading，非 GEMM kernel |
| BitRed | Bit-level sparsity, RISC-V 加速器 | 硬件设计，非 GPU |

**→ ASPLOS 2026 在 activation+weight quantization 这个 track 上是空的，对我们投 ASPLOS 2027 是好消息。**

注：Tilus (NVIDIA) 已开源 [github.com/NVIDIA/tilus](https://github.com/NVIDIA/tilus)，1.75× faster than Triton，可考虑用来写我们的 kernel。

---

## 二.5、重要 arXiv Preprint（可能投 ASPLOS 2027 或其他顶会）

### MixLLM: LLM Quantization with Global Mixed-Precision between Output-Features ⭐潜在竞品

- **arXiv**: [2412.14590](https://arxiv.org/abs/2412.14590)（2024.12）
- **作者**: Zhen Zheng, Xiaonan Song, Chuanjie Liu（Microsoft）
- **核心做法**:
  - Global mixed-precision **W4A8**：不同 output channel 分配不同 weight 位宽（4-bit vs 8-bit）
  - 基于 Fisher information 的 salience metric 决定分配
  - Two-step dequantization：先 partial dequant → INT8 TC MMA → 再 scale
  - Fast Int-to-Float conversion fused into MMA accumulator
  - `vsub4` 向量化 zero-point 减法
- **和我们的关系**: 方法论不同（他们按 output channel salience，我们按 PCA variance），但目标相似。**如果投顶会需要注意**。

### FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization ⭐潜在竞品

- **arXiv**: [2505.20839](https://arxiv.org/abs/2505.20839)（2025.05）
- **作者**: Samsung SDS
- **核心做法**:
  - **W-INT4 / A-FP8**，CUTLASS-based INT4×FP8 GEMM kernel
  - LUT-based INT4→FP8 conversion in register
  - RoPE-aware outlier smoothing for KV cache
  - Three-stage pipelined FlashAttention-3 (Hopper)
- **结果**: 1.68× faster FFN, 1.26× faster prefill vs QServe
- **和我们的关系**: 也用 CUTLASS on Hopper，也处理 outlier，但用 FP8 而非 INT8。**技术路线接近，需关注**。

### COMET 开源状态 ❌

论文声称 "we provide an open-source W4Ax kernel"，但**实际未开源**：
- GitHub 上搜不到任何相关 repo
- 论文和 arXiv 页面都没有 code link
- 这对我们是好消息：reviewer 无法直接跑 COMET 对比，我们只需和论文数字比

---

## 三、其他顶会 2024-2025

| 工作 | 会议 | arXiv | 做法 | 和我们的关系 |
|------|------|-------|------|-------------|
| **QServe** | MLSys'25 | 2405.04532 | W4A8KV4，CUDA cores（非 INT4 TC） | activation 统一 8-bit，不做 mixed |
| **Atom** | MLSys'24 | 2310.19102 | 早期 W4A4，mixed-precision activation | COMET 的前身 |
| **QuaRot** | NeurIPS'24 | 2404.00456 | Rotation + quantization | 和 ResQ 的 methodology 相似（都用旋转变换） |
| **M-ANT** | HPCA'25 | — | Mathematically adaptive INT4 | 数据表示层面优化 |

---

## 四、和 ResQ（我们）的差异化定位

### 竞品对比矩阵

| 维度 | COMET（最强竞品） | ABQ-LLM | QServe | **ResQ（我们）** |
|------|-------------------|---------|--------|-----------------|
| **Outlier 检测方法** | Channel-level magnitude | Distribution correction | 无 | **Per-head PCA eigenvalue** |
| **分组依据** | 按 channel 值域统计 | 按 bit 分解 | 不分组 | **按 PCA variance 维度分组** |
| **旋转/变换** | 无 | 无 | 无 | **PCA rotation fused into weight** |
| **量化精度** | W4A4 + W4A8 | 任意位宽 | W4A8 | W4A4(main) + W8A8(high) |
| **Kernel 类型** | Custom W4Ax CUDA | BTC decomposition | CUDA cores | **CUTLASS INT8/INT4 TC (Hopper)** |
| **目标硬件** | A100 (Ampere) | A100 (Ampere) | A100 | **H20/Hopper (SM90)** |
| **TC 利用** | INT4 TC (A100) | Binary TC 分解 | 不用 TC | **INT8/INT4 TC (Hopper TMA)** |

### 我们的核心 Novelty（vs COMET）

1. **PCA-based per-head 维度分析**：不是简单按 channel magnitude 分 outlier，而是对每个 attention head 做 eigenvalue decomposition，按方差维度分组 → 更精细、有理论依据
2. **Rotation fused into weight**：PCA basis matrix 直接乘进 v_proj/o_proj weight → 推理时零额外计算开销
3. **Hopper-native kernel**：基于 CUTLASS 3.x 的 TMA + Warp-specialized + INT8/INT4 TC kernel，不是 Ampere 时代的 kernel
4. **Per-group activation + grouped GEMM**：o_proj 使用 CUTLASS PtrArray grouped GEMM 实现 per-group activation scale，单次 kernel launch

### 需要重点超越的指标

- COMET kernel: 2.88× over cuBLAS（A100 INT4 TC）→ 我们需要在 Hopper 上达到更高加速比
- COMET 端到端: 2.02× over TensorRT-LLM → 需要展示完整的 serving throughput 对比
- 精度: 需要在多个 model + benchmark 上展示 PCA-based 分组比 channel-level FMPQ 更好

---

## 五、投稿策略建议

1. **ASPLOS 2027 deadline 预计 2026 年 9-10 月**，时间充裕
2. ASPLOS 2026 在这个 track 上是空的 → 有空间
3. 核心 story：**PCA-guided mixed-precision 比 channel-level 更好** + **Hopper-native fused kernel 比 Ampere kernel 更快**
4. 需要补充的实验：
   - 与 COMET 的直接对比（accuracy + throughput）
   - 与 ABQ-LLM 的对比
   - 多模型（LLaMA-2-7B/13B/70B, LLaMA-3）
   - 多 benchmark（WikiText, C4, MMLU, LongBench 等）
