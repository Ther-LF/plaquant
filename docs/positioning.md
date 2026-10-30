# ProMix — 定位 / positioning(advisor 讨论用)

> 2026-10-30。基于 round-28~ 的完整实验 + 9 篇 related work 全文读 + vLLM 可行性 spike。
> 诚实版:能 claim 什么、不能 claim 什么、与竞品的真实差异、以及证据强度。

## 一句话 thesis

**ProMix 是唯一把整条 W4A4 精度-性能 Pareto 做成"一个旋钮、细粒度、每点可部署"的连续曲线的方法;所有竞品都是这条曲线上的一个固定点。**

> 一个硬件对齐、细粒度(64-通道环)、单旋钮(平均高精度预算 B)的 W4A4 精度-性能连续前沿,由单 launch 的 NVFP4+MXFP8 双格式融合 kernel 让曲线上任意点都能一次部署 —— 竞品给你一个点,我们给你整条前沿。

## 为什么是一条"曲线"而不是一个点(四件套)

| 组件 | 作用 | 竞品 |
|---|---|---|
| 单旋钮预算 B | 一个参数扫出整条曲线(纯NVFP4 4.5b → 膝盖 → 纯MXFP8 8.5b) | 固定配方 = 一个点 |
| 细粒度 64-环(PCA 排序后) | 曲线平滑连续、硬件对齐(K_high%32、K_low%64) | per-layer / 固定1/8,粒度粗 |
| 率失真分配(实测 ΔD) | 曲线上每点都是该比特下最优精度 | 固定比例 / 阈值 |
| 单 launch 双格式融合 kernel | 曲线上任意混合比例都一次 launch 部署 | 分段多 GEMM / 残差 / ASIC |

## 诚实证据

**精度(gap-to-FP16,wikitext,越低越好)**:ProMix 8B **+0.46** @~4.5–5.0bit,是 NVFP4 簇里最好的之一且比特更少 —— MicroMix +0.48@5.5b、SharQ +0.48、ARCQuant +0.63。C4 同号(非 wikitext artifact)。**zero-shot 看不出分配增益(任务噪声内)—— 不 claim 0-shot 收益。**

**曲线数据**:12 预算 × {1B/3B/8B/13B} × {PPL + C4 + 真实 prefill/decode 延迟}(42+ 点);悬崖 = outlier 集中(top-1/32 环 = 42% ΔD),平滑 S 坡,膝盖 ~1/96(实测细采样)。

**性能(诚实)**:同弱栈 e2e W4A4/BF16 = 1.29×(8B b16);**格式全摆幅 GEMM-级 ~1.3×(我们形状,TC 没喂饱)/ 2.05×(16384³ 喂满)**;隔离对比我们融合 GEMM vs bf16 = **2.35–2.6× geomean(随 M→3.1×)**、vs 朴素 2-launch 同算法 = **1.3–1.7×**(融合的独立价值)。**上限 2–2.87×(vLLM 整栈证)——我们 1.29× 的差距是研究 harness 工程,不是格式/算法**;要在同强栈拿公平数 → vLLM 集成(进行中)。

## 能 claim / 不能 claim(scope 到组合)

- **不能单独 claim**:"误差驱动自适应比特预算"(MicroMix/FGMP/SHMQ 占了)、"比方差更好的子空间准则"(CoQuant 占了)、"outlier/残差补偿融进 GEMM"(MosaicQuant/SharQ/ARCQuant 占了)。
- **可防御的 novelty = 完整组合**:W4A4 + PCA 旋转分区 + NVFP4/MXFP8 **异构格式** + **单 launch 双格式融合主循环** + 真 Blackwell 实测输出误差 + 一个旋钮扫出的可调曲线。**这个组合没人做过。**
- **真·护城河 = 单 launch 异构融合 kernel**(别人:MicroMix 分段 3-GEMM、SharQ/ARCQuant 残差+标准 GEMM、FGMP ASIC)。

## 竞品(全文读过)

- **必比 head-to-head(可在 Blackwell 跑)**:ARCQuant(纯NVFP4+残差,反 mixed-precision/反 rotation)、MicroMix(MXFP4/6/8 分段)、SharQ(全NVFP4 稀疏+残差)—— 全是同一实验室。
- **消融对象**:CoQuant(加权-PCA 子空间,证我们"比方差好"要小心)。
- **仅引用**:FGMP(ASIC)、Oltron(ASIC INT4+fp8/12)、SHMQ(INT W4A8,激活 8-bit)、DualQuant/dMX/MosaicQuant(INT 或 Ada)。

## 必答的 reviewer 攻击:旋转伤 FP4

**ARCQuant + DualQuant + SharQ 三篇都攻击旋转**(DualQuant 示 QuaRot 在 NVFP4 上 PPL>1000)。我们的回应:**PCA 旋转是按方差排序把 outlier 集中到少数方向(再分区),和他们批的 Hadamard "把 outlier 摊开抬高局部块动态范围"相反**;且旋转对我们只是小内部件(实测值 0.02–0.17),主卖点是曲线+kernel 不是旋转。**建议补一个"无主-Hadamard 变体"实验正面回应。**

## 已知 limitation(诚实列出)

1. per-layer 误差分配增益小、≤3B 有感、随规模→0(8B/13B 持平)、下游看不出 —— 它是"安全精化 + 让曲线每点最优",不是精度 headline。
2. 同弱栈性能只 1.29×(工程稀释);真实竞争力待 vLLM 集成坐实。
3. decode:纯 NVFP4 path 走 prefill GEMM,无 decode-GEMV 变体(待补)。

## money 图

一条 ProMix 曲线(精度 × 性能,扫 B)+ ARCQuant/MicroMix/SharQ 作为点 → 曲线压点 / 支配。精度轴现在可画(统一 regime),性能轴等 vLLM 集成(同强栈)。
