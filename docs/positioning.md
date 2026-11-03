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

**性能(FINAL,2026-11-01,vLLM 0.32 同强栈实测;完整表 `docs/vllm_final_fiveway.md`)**:R1-R5 性能 loop 把 8B decode pm/nv 从 0.52 → **0.79-0.82**(b8/b16),大 batch 终表进一步到 **0.85-0.93**(b128-512,两模型);prefill pm/nv 0.54-0.57 → **0.58-0.68**。对 BF16:decode **1.09-1.68×**、prefill **1.26-1.82×** 全点赢;对 weight-only W4A16-Marlin:prefill 全点 1.3-2.0×、decode b≥128 1.03-1.24×、QServe 协议 c256 吞吐 1.17×(3B)/1.34×(8B) 赢(只让 b1-64 小 batch decode);对 FP8 W8A8:大 batch 持平(0.90-1.09)。**精度在 loop 期间净改善**(decode m16 marlin 路由 −0.32 PPL,3B 8.047;served 口径 8.370/6.640 仍是最准 W4A4)。**结构地板已分解**(r5 §3c):剩余 vs NVFP4 的差距 = 双段 mixed GEMM(1.33× vs 单段 fp4)+ act-MSE 量化(精度换的)+ down-proj 在线 Hadamard(ResQ 旋转的固有税)+ 插件 host 间隙;shipped 配置离分解地板 ~4%,无 ≥2% 的未试杠杆。旧"工程稀释 1.29×"叙事作废——现在是同强栈、同协议、可复现的 serving 数。

## 能 claim / 不能 claim(scope 到组合)

- **不能单独 claim**:"误差驱动自适应比特预算"(MicroMix/FGMP/SHMQ 占了)、"比方差更好的子空间准则"(CoQuant 占了)、"outlier/残差补偿融进 GEMM"(MosaicQuant/SharQ/ARCQuant 占了)。
- **可防御的 novelty = 完整组合**:W4A4 + PCA 旋转分区 + NVFP4/MXFP8 **异构格式** + **单 launch 双格式融合主循环** + 真 Blackwell 实测输出误差 + 一个旋钮扫出的可调曲线。**这个组合没人做过。**
- **真·护城河 = 单 launch 异构融合 kernel**(别人:MicroMix 分段 3-GEMM、SharQ/ARCQuant 残差+标准 GEMM、FGMP ASIC)。

## 竞品(全文读过)

- **必比 head-to-head(可在 Blackwell 跑)**:ARCQuant(纯NVFP4+残差,反 mixed-precision/反 rotation)、MicroMix(MXFP4/6/8 分段)、SharQ(全NVFP4 稀疏+残差)—— 全是同一实验室。
  **已实测(2026-11-01,`docs/vllm_final_fiveway.md` §6)**:SharQ 用它自己的 vLLM 集成在同栈/同卡/同协议跑通 —— 精度打平(8.385 vs 我们 8.370),serving 我们在工作区间全面赢(decode b128 1.14×、qserve c256 1.40×、prefill b128 2.41×;它只赢 b1 小 batch)。MicroMix 用它自己的 e2e bench 在同卡复现其论文 prefill 2.00×,但 decode as-shipped 慢 FP16 4.4×,其管线 3B PPL 9.045 明显逊于我们;fp_quant/Quartet(vLLM 原生)RTN 精度 9.62-10.25,是速度点不是精度对手,且 as-shipped 在 SM120 上根本跑不了(wheel 只带 sm_100 cubin)。
- **消融对象**:CoQuant(加权-PCA 子空间,证我们"比方差好"要小心)。
- **仅引用**:FGMP(ASIC)、Oltron(ASIC INT4+fp8/12)、SHMQ(INT W4A8,激活 8-bit)、DualQuant/dMX/MosaicQuant(INT 或 Ada)。

## 必答的 reviewer 攻击:旋转伤 FP4

**ARCQuant + DualQuant + SharQ 三篇都攻击旋转**(DualQuant 示 QuaRot 在 NVFP4 上 PPL>1000)。但他们攻击的是**全宽 Hadamard(把 outlier 摊到所有通道、抬高每个 block 的局部动态范围、破坏块隔离)**。**ProMix 用的不是这个:**
- **主旋转是 PCA(U)= 按方差"集中"outlier,不是"摊开"** —— 和攻击的方向相反;
- **环 Hadamard 是局部 64-宽块对角**(只在环内混,能量留在环内不跨 tensor 外溢),且 **outlier 重的环被分配到 MXFP8**(更大 range)再兜一层 —— 与全宽 Hadamard 根本不同;环本身是载重件(ring128/256 的 overturn 靠它,不能去掉);
- 唯一的全宽 Hadamard 只在 down_proj 一处(ResQ 遗留,intermediate 太大做不了 PCA)。

**回应 = 概念区分(局部环+PCA集中 vs 全宽摊开)+ 现成实证**:ProMix gap +0.46 比"反旋转"的 ARCQuant +0.63 还准,且不像 QuaRot 在 NVFP4 上发散 → 我们的旋转在 block-scaled FP4 下活得好。**无需额外消融实验**(环是必要件,不消融;全宽 Hadamard 我们本就几乎不用)。

## 已知 limitation(诚实列出)

1. per-layer 误差分配增益小、≤3B 有感、随规模→0(8B/13B 持平)、下游看不出 —— 它是"安全精化 + 让曲线每点最优",不是精度 headline。
2. ~~同弱栈性能只 1.29×~~ 已被 vLLM 集成 + R1-R5 loop 取代(FINAL 表);真实剩余 limitation:vs 纯 NVFP4 仍 0.85-0.93(大 batch decode)/0.58-0.68(prefill),为双段结构+act-MSE+Hadamard 的结构税(r5 §3c 已分解到地板)。
3. ~~无 decode-GEMV 变体~~ 已补齐:M=1 GEMV(字节级门控)+ M=2..48 marlin W4A16 低-M 路由 + MXFP8-marlin 高段伴侣(R4/R5),decode 精度还顺手 −0.32 PPL。
4. 小 batch decode(b1-64)输 weight-only Marlin 0.66-0.85×——latency-bound 单用户区间,诚实列出。

## money 图

一条 ProMix 曲线(精度 × 性能,扫 B)+ ARCQuant/MicroMix/SharQ 作为点 → 曲线压点 / 支配。精度轴现在可画(统一 regime),性能轴已有同强栈 serving 终表 + **实测 r-sweep 曲线**(`docs/vllm_final_fiveway.md` §4:3B 6 点 / 8B 4 点,每点 PPL+吞吐+gate;全曲线精度压 NVFP4、吞吐 0.85-0.93,serving 代价对 r 近乎平坦 —— "曲线压点"现在两条轴都是实测)。

## down_proj 叙事防御(2026-11-03,显式对照已钉死)

**潜在质疑**:"down_proj 用在线 Hadamard+全FP4,不符合 PCA 分区叙事。"
**回应框架(两种变换、一个率失真原则)**:变换的目的是让"该花高精度的地方"被实测 ΔD 看见 —— hidden 流 PCA 可行 → **集中** outlier → 分区捕获(top-1/32 环=42% ΔD);intermediate 流 PCA 不可行 → Hadamard **摊平** outlier → ΔD 荒漠(<1%)→ 全 FP4 即最优。分配器按同一准则裁决两者。
**显式对照(3B,`docs/rd_down_fp4_control.md`)**:强塞 down_proj 1/8 FP8(等预算、挑其最高 ΔD 环)→ wiki +0.008 / **C4 +0.056(= 整个分配收益的 78%)更差**;白送 +35% 预算强塞 → 符号不一致(噪声),买不到。
**论文句**:"Forcing high precision onto the Hadamard-flattened down_proj buys no accuracy even with extra budget and costs accuracy at matched budget — all-FP4 down_proj is measured-optimal, not a simplification."
**顺带的精细论点(反打 ARCQuant/DualQuant 的 anti-rotation)**:Hadamard 摊开对"要分区的流"有害(所以那里用 PCA 集中),对"统一 FP4 的流"恰好正确(摊平后 FP4 够用)—— 旋转好坏取决于下游是分区还是统一量化。
