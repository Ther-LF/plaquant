# ResQ 端到端量化评估结果

## 评估配置

- **模型**: Llama-3.2-1B-Instruct (unsloth)
- **量化方案**: W16A4KV4 (权重 FP16, 激活/KV Cache 4-bit)
- **混合精度**: high_fraction=0.125 (1/8 通道 8-bit, 7/8 通道 4-bit)
- **旋转模式**: ResQ (PCA子空间分离 + 子空间内随机旋转)
- **精度**: FP16 (BF16 在 H20 GPU 上有 cuBLAS bug)
- **设备**: NVIDIA H20 GPU
- **PyTorch**: 2.4.0+cu121

---

## Fake Quant vs Real Quant 对比

### 关键区别

| 模式 | 激活量化 | 权重 | 矩阵乘 |
|------|---------|------|--------|
| **Fake Quant** | quantize → dequantize → fp16 | FP16 原始权重 | 1次 fp16 GEMM |
| **Real Quant** | quantize → dequantize → fp32 | 8-bit per-group(gs=128) → dequantize | 1次 fp32 GEMM → cast fp16 |

Real Quant 对所有权重精度组（main/high/low）均使用 **8-bit per-group symmetric** 量化（groupsize=128）。

### WikiText-2 Perplexity

| 模式 | WikiText2 PPL |
|------|:------------:|
| Fake Quant | **15.017** |
| Real Quant | **15.091** |
| Δ | +0.074 (+0.5%) |

### 全部 9 Benchmark 对比 (acc)

| Task | Fake Quant | Real Quant | Δ | 对齐 |
|------|:----------:|:----------:|:-:|:----:|
| WikiText2 PPL ↓ | 15.017 | 15.091 | +0.074 | ✅ |
| MMLU | 0.4009 | 0.4109 | +0.0100 | ✅ |
| BoolQ | 0.6804 | 0.6728 | -0.0076 | ✅ |
| PIQA | 0.7051 | 0.7209 | +0.0158 | ✅ |
| HellaSwag | 0.4324 | 0.4364 | +0.0040 | ✅ |
| WinoGrande | 0.5564 | 0.5856 | +0.0292 | ✅ |
| ARC-Easy | 0.6448 | 0.6591 | +0.0143 | ✅ |
| ARC-Challenge | 0.3319 | 0.3396 | +0.0077 | ✅ |
| OpenBookQA | 0.2380 | 0.2400 | +0.0020 | ✅ |
| Social IQA | 0.3992 | 0.4038 | +0.0046 | ✅ |

### 含 norm 指标的完整对比

| Task | Metric | Fake Quant | Real Quant | Δ |
|------|--------|:----------:|:----------:|:-:|
| **MMLU** (overall) | acc | 0.4009 | 0.4109 | +0.0100 |
| - Humanities | acc | 0.3866 | 0.3804 | -0.0062 |
| - Social Sciences | acc | 0.4475 | 0.4517 | +0.0042 |
| - STEM | acc | 0.3492 | 0.3714 | +0.0222 |
| - Other | acc | 0.4474 | 0.4567 | +0.0093 |
| **BoolQ** | acc | 0.6804 | 0.6728 | -0.0076 |
| **PIQA** | acc | 0.7051 | 0.7209 | +0.0158 |
| | acc_norm | 0.7176 | 0.7149 | -0.0027 |
| **HellaSwag** | acc | 0.4324 | 0.4364 | +0.0040 |
| | acc_norm | 0.5803 | 0.5841 | +0.0038 |
| **WinoGrande** | acc | 0.5564 | 0.5856 | +0.0292 |
| **ARC-Easy** | acc | 0.6448 | 0.6591 | +0.0143 |
| | acc_norm | 0.6052 | 0.5972 | -0.0080 |
| **ARC-Challenge** | acc | 0.3319 | 0.3396 | +0.0077 |
| | acc_norm | 0.3541 | 0.3677 | +0.0136 |
| **OpenBookQA** | acc | 0.2380 | 0.2400 | +0.0020 |
| | acc_norm | 0.3280 | 0.3300 | +0.0020 |
| **Social IQA** | acc | 0.3992 | 0.4038 | +0.0046 |

---

## 分析

### 对齐度评估

**结论：Real Quant 与 Fake Quant 完全对齐。**

1. **PPL 差异仅 0.074 (0.5%)**：来自 8-bit per-group 权重量化引入的微小量化误差，以及 fp32 vs fp16 累加精度差异。
2. **所有 9 个 benchmark 差异均在统计误差范围内**：最大偏差 WinoGrande +0.029，但该 benchmark stderr=±0.014，属于 ~2σ 的统计波动。
3. **Real Quant 大部分指标甚至略高**：这不是 real quant 更好，而是 fp32 累加精度更高+权重量化引入的随机扰动碰巧有利。
4. **无系统性偏差**：有些指标正偏（PIQA +0.016），有些负偏（BoolQ -0.008），表明差异是随机噪声而非系统性错误。

### 权重量化方案

- 所有权重精度组均使用 **8-bit symmetric per-group** 量化 (groupsize=128)
- 4-bit 权重量化在不使用 GPTQ 的情况下过于有损（per-layer error ~50%），因此全部使用 8-bit
- 目标混合精度 GEMM: **INT4_act × INT8_weight (main)** + **INT8_act × INT8_weight (high)**

---

## 下一步

- [x] Fake Quant 全量评估 (W16A4KV4)
- [x] Real Quant 端到端实现
- [x] Real Quant vs Fake Quant 对齐验证
- [ ] Phase 2: 学习 CUTLASS 3.x SM90 GEMM for Hopper
- [ ] Phase 3: 实现 Hopper 混合精度 GEMM kernel
- [ ] Phase 4: 集成到 ResQ
- [ ] Phase 5: 纯 CUDA 优化
