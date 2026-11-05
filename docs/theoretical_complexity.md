# 理论计算复杂度对比(覆盖不可运行的方法)

> 2026-11-05。纯算术 + 已测硬件峰值(`hardware_format_peak.md`),无需 GPU。
> 目的:把在 Blackwell 上跑不起来的方法(INT4 系、ASIC-only、无 kernel)也放到同一把尺上比。

## 指标定义

- **BOPs(Bit Operations)**= MACs × bit_w × bit_a,量化/混合精度论文的标准硬件无关计算量指标(HAWQ 等)。同一模型同一 GEMM,MACs 相同,所以 **相对 BOPs = (bit_w × bit_a) / (16×16)**。
- **effective bits/weight** 含 micro-scale / 残差通道 / 高精度比例的均摊。
- **datapath gate(关键)**:理论 BOPs 只有在硬件**有对应数据通路**时才成立。SM120 实测(`hardware_format_peak.md`):NVFP4/MXFP8/INT8 原生;**INT4 无通路,ptxas 降级为 INT8 指令(s4→IMMA.S8,零 S4 SASS)** → INT4 的"真实计算"是 8-bit。

## 理论 vs 门后真实(W4A4 为主)

| 方法 | W-bit | A-bit | 理论 rel-BOPs | SM120 通路 | 门后真实 rel-BOPs | 备注 |
|---|---|---|---|---|---|---|
| FP16 | 16 | 16 | 256 | ✅ | 256 | 基准 |
| FP8 W8A8 | 8.5 | 8 | ~68 | ✅ | ~68 | — |
| **NVFP4 W4A4** | 4.5 | 4.5 | ~20 | ✅ 原生 | ~20 | 格式天花板 |
| **ProMix r=1/8** | 4.97 | ~4.5(混) | ~22 | ✅ 单launch融合 | ~22 | 高精度段 MXFP8 半速已含 |
| ARCQuant | 5.12 | 4.5 | ~23 | ✅ | ~23 + 残差K | K 扩 S 列的额外 MAC |
| SharQ | ~4.5 | 4.5 | ~20 | ✅ | **~40** | 稀疏+稠密**双 GEMM ≈2×** 工作量 |
| MicroMix | ~5.2 | ~4.5(混) | ~24 | ✅ | ~24 | 分段 3-GEMM 的调度开销另计 |
| GPTQ/AWQ W4A16 | 4 | 16 | ~64 | ✅ | ~64 | 仅权重,激活 FP16(不省 compute,只省带宽) |
| **QuaRot / Atom / ResQ (INT4 W4A4)** | 4 | 4 | **~16(纸面最低)** | ❌ **无 INT4 通路** | **~64(降 INT8)** | 纸面最便宜,Blackwell 实际 = FP8 量级;实测 kernel 慢 FP16 4-7× |
| FGMP (NVFP4+FP8) | ~4.6 | 4.5 | ~21 | ❌ **ASIC-only** | N/A | 自认软件实现不可行 |
| Oltron (INT4+fp8/12) | 混 | 混 | 低 | ❌ **ASIC(TSMC-28nm 仿真)** | N/A | 定制硅,无 GPU 路径 |

## 内存(bits/weight + KV)

| 方法 | bits/weight | KV | 权重显存 vs FP16 |
|---|---|---|---|
| FP16 | 16 | 16 | 1× |
| NVFP4 / ProMix(r=1/8) | 4.50 / 4.97 | 16(或 KV4=5.4) | **3.5× / 3.2×** 压缩 |
| INT4 系 | 4.0 | KV4≈4-4.5 | 4× 压缩(但 compute 不省,见上) |
| W4A16 | 4.0 | 16 | 4× 压缩(纯带宽收益) |

## 结论(论文"理论复杂度"小节)

1. **纸面 BOPs ≈ bit 的函数,W4A4 各家理论上都挤在 ~16-24**(差异来自 micro-scale/残差/高精度比例的零点几 bit)——"实际上就是看多少 bit"这个直觉在**理论层面成立**。
2. **但"看 bit"必须乘硬件 datapath 门**:INT4 纸面 16(最低)在 Blackwell 实为 ~64(降 INT8),SharQ 纸面 4.5bit 实为 ~2× 工作量,FGMP/Oltron 根本没有 GPU 路径。**理论便宜 ≠ 真实便宜。**
3. 这把尺让**不可运行的方法也能诚实入表**:INT4 系(理论 16 / 真实 64 / 无快 kernel)、FGMP/Oltron(ASIC-only)—— 它们在"理论 BOPs"列看着有竞争力,在"门后真实"列现原形。ProMix/NVFP4/MXFP8 是**理论与真实一致**的少数(格式有原生通路)。

> 配图建议:两根柱子并排(理论 rel-BOPs vs 门后真实 rel-BOPs),INT4 系那根"理论矮/真实高"的落差最醒目 —— 一张图说清"FP4 时代 INT4 的理论优势是纸面的"。
