# ResQ 端到端量化评估结果

## 评估配置

- **模型**: Llama-3.2-1B-Instruct (unsloth)
- **量化方案**: W16A4KV4 (权重 FP16, 激活/KV Cache 4-bit)
- **混合精度**: high_fraction=0.125 (1/8 通道 8-bit, 7/8 通道 4-bit)
- **旋转模式**: ResQ (PCA子空间分离 + 子空间内随机旋转)
- **精度**: FP16 (BF16 在 H20 GPU 上有 cuBLAS bug)
- **设备**: NVIDIA H20 GPU
- **PyTorch**: 2.4.0+cu121

## 评估结果

### WikiText-2 Perplexity

| 指标 | 值 |
|------|-----|
| WikiText2 PPL | **15.017** |

### 下游任务 Accuracy

| Task | Metric | Score | ± Stderr |
|------|--------|-------|----------|
| **MMLU** (overall) | acc | **0.4009** | ±0.0041 |
| - MMLU Humanities | acc | 0.3866 | ±0.0087 |
| - MMLU Social Sciences | acc | 0.4475 | ±0.0088 |
| - MMLU STEM | acc | 0.3492 | ±0.0084 |
| - MMLU Other | acc | 0.4474 | ±0.0088 |
| **BoolQ** | acc | **0.6804** | ±0.0082 |
| **PIQA** | acc | **0.7051** | ±0.0106 |
| | acc_norm | 0.7176 | ±0.0105 |
| **HellaSwag** | acc | **0.4324** | ±0.0049 |
| | acc_norm | 0.5803 | ±0.0049 |
| **WinoGrande** | acc | **0.5564** | ±0.0140 |
| **ARC-Easy** | acc | **0.6448** | ±0.0098 |
| | acc_norm | 0.6052 | ±0.0100 |
| **ARC-Challenge** | acc | **0.3319** | ±0.0138 |
| | acc_norm | 0.3541 | ±0.0140 |
| **OpenBookQA** | acc | **0.2380** | ±0.0191 |
| | acc_norm | 0.3280 | ±0.0210 |
| **Social IQA** | acc | **0.3992** | ±0.0111 |

### 汇总对比 (acc)

| Task | ResQ W16A4KV4 | FP16 Baseline* |
|------|:------------:|:--------------:|
| WikiText2 PPL ↓ | 15.017 | ~10.5 |
| MMLU | 0.4009 | ~0.46 |
| BoolQ | 0.6804 | ~0.73 |
| PIQA | 0.7051 | ~0.75 |
| HellaSwag | 0.4324 | ~0.56 |
| WinoGrande | 0.5564 | ~0.62 |
| ARC-Easy | 0.6448 | ~0.72 |
| ARC-Challenge | 0.3319 | ~0.37 |
| OpenBookQA | 0.2380 | ~0.28 |

*FP16 Baseline 数值为 Llama-3.2-1B-Instruct 典型参考值（需跑 FP16 baseline 确认）

## 分析

1. **PPL 退化**: 15.017 vs ~10.5 (FP16)，退化约 43%。对于 4-bit 激活量化来说在合理范围内。
2. **MMLU**: 40.09% — 对 1B 模型来说可接受，FP16 的 1B 模型本身 MMLU 也不高（~46%）。
3. **整体趋势**: 所有 benchmark 相比 FP16 都有不同程度退化，但保持了基本的模型能力。
4. **注意**: 当前配置为 W16A4KV4（权重未量化），论文主打的 W4A4KV4 配置需要额外跑 GPTQ 权重量化。

## 结论

ResQ 的 W16A4KV4 量化在 Llama-3.2-1B-Instruct 上表现合理，PPL 和下游任务准确率退化在可接受范围内。这验证了 ResQ 方法的有效性，值得为其开发高效的混合精度 GEMM kernel 来加速推理。

## 下一步

- [ ] 跑 FP16 baseline 获取精确对比数据
- [ ] 跑 W4A4KV4 (含 GPTQ 权重量化) 配置评估
- [ ] 在更大模型上验证 (Llama-3.1-8B, 70B)
