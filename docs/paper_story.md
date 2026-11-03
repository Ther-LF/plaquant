# ProMix 论文总纲 —— 故事、方法、数据、图表(写作交接文档)

> 2026-11-03。给论文写作者的自包含交接:整个故事线、方法细节、全部关键数字、图表清单、相关工作定位、诚实边界。
> 每个数字的出处见 `docs/paper_evidence_index.md`(claim → 文件 → 数字);深入某块读对应报告。
> 硬件:NVIDIA RTX Pro 6000 Blackwell(SM120, 96GB)。模型:Llama-3.2-1B/3B、Llama-3-8B(+13B、Qwen3-8B 泛化)。

---

## 0. 一句话 thesis(论文的魂)

**ProMix 是第一个把 W4A4 的精度-性能权衡做成"一个旋钮、细粒度、每点可部署"的连续前沿的方法 —— 由一个单 launch 的 NVFP4+MXFP8 双格式融合 kernel 支撑;竞品给你一个固定的点,我们给你整条曲线,而且曲线上的精度点全面压住所有可运行的 W4A4 同行。**

标题方向候选(写作者定):
- "ProMix: A Tunable Accuracy-Performance Frontier for W4A4 LLM Serving on Blackwell"
- 关键词:mixed-precision / rate-distortion allocation / dual-format fused GEMM / NVFP4+MXFP8 / serving

---

## 1. 背景与动机

1. **W4A4(权重+激活都 4-bit)是吞吐的终极档**,但激活 outlier 让它精度很难看。经典解法两类:旋转(QuaRot/SpinQuant)和混合精度(Atom/QUIK/ResQ 留一撮高精度通道)。
2. **硬件拐点:Blackwell 原生 FP4**。我们实测(`hardware_format_peak.md`):NVFP4 1469 TFLOPS vs INT8 814 vs FP16 406;**INT4 在 Blackwell 零数据通路**(s4 被 ptxas 降成 INT8 指令,SASS 证据)→ **所有 INT4 系的 W4A4 前作(QuaRot/Atom/ResQ/QServe-W4)在这代硬件上没有可行的快实现**(QuaRot kernel 比 FP16 慢 4-7×)。格式的时代切换本身就是动机。
3. **现有 FP4 方法都是"单点"**:vLLM-NVFP4 纯 FP4(精度差);ARCQuant 固定残差通道;MicroMix 固定阈值通道混合;SharQ 固定稀疏+残差。**没人提供可调的精度-性能旋钮,更没人有支撑任意混合比例的单 launch kernel。**
4. 部署现实:不同场景要不同的点(有人要精度、有人要吞吐)。**一个方法应该交付前沿,而不是一个点。**

## 2. 核心观察(三个,全部实测)

**O1 — 量化误差高度集中在少数方向**:PCA 旋转后,实测每环升级收益 ΔD=MSE(FP4)−MSE(FP8) 与方差完美同序(ρ=0.9999);**top-1/32 的 64-通道环扛全模型 42% 的 ΔD(≈均匀的 13 倍)**。纯 FP4 是精度悬崖(3B 10.91 vs bf16 7.82),但一点点高精度就把它救回来:恢复曲线是**平滑陡 S 坡,膝盖在 ~1/96**(1/256→1/32 恢复 13%→86%)。(`rd_accpuzzle_report.md`,谱图 `rd_deltaD_spectrum.html`)

**O2 — 固定比例(如 ResQ 的 1/8)浪费预算**:uniform-1/8 把 ~41% 的高精度名额花在近零价值的环上(拔掉那些环 PPL 只动 −0.0024);尤其 down_proj 是 **ΔD 荒漠**(35-44% 的环、<1% 的 ΔD —— 其在线 Hadamard 把 outlier 摊平了)。(`rd_whydecay_report.md`)

**O3 — 混合精度可以在硬件上近乎免费**:我们的单 launch 双格式融合 GEMM 使"带一段 MXFP8"的边际延迟很小(serving 曲线对 r 近乎全平,r=1/32→1/4 decode b256 只动 18.1→19.1ms)—— **这直接证伪了 ARCQuant 的核心前提("双格式混合破坏硬件流水线,所以别混")**。(`vllm_final_fiveway.md` §4)

## 3. 方法(ProMix,四个组件)

### 3.1 旋转:PCA 集中 + 环块对角(零在线开销)
- `T = U @ R1`:**U = PCA 基**(wikitext-512 校准协方差特征分解,全局按方差排序)—— 把 outlier **集中**到少数方向(与 Hadamard 的"摊开"相反,这是对"旋转伤 block-scaled FP4"攻击的回应);**R1 = 块对角环旋转**(每 512 通道一个 Hadamard 块;ring512 实测最优)。
- 整条 residual stream 的旋转 **在 embedding 处 bake 一次、全模型共享、零在线开销**。
- **例外:down_proj**(intermediate 维度太大,PCA 不可行)用在线 Hadamard **摊平** outlier → 全 FP4。**两种变换、一个原则**:让"该花高精度的地方"被实测误差看见(PCA 使之集中可见;Hadamard 使之消失、故不用花)。显式对照钉死(§6.4)。

### 3.2 率失真分配(单旋钮 B)
- **分配单元 = 64-通道环**(R1 混合原子 + 硬件对齐 K_high%32/K_low%64 的最细粒度)。
- 每环实测两个失真:D_fp4、D_fp8(用部署管道的真量化器,离线校准一遍);**升级收益 ΔD = D_fp4 − D_fp8**。
- **全局单阈值分配**:所有层所有环按 ΔD 从高到低排,取 top-(B×总环数) 为 MXFP8,其余 NVFP4。θ 阈值由预算 B 反推(一一对应);B 即用户唯一旋钮(或由精度目标/膝盖规则自动选)。
- 嵌套/对齐自动满足;每层的 r_i 自然浮动(outlier 重的层多拿、down_proj 拿 0)。

### 3.3 量化器(精度配方,全部消融转正)
- 权重:GPTQ(每模型最优校准样本 1024/512/512)+ **两级 scale**(per-tensor FP32 g × per-block E4M3/UE8M0 micro,NEAREST+micro-MSE,depth7 仅 3B/8B);g 搭在 GEMM epilogue alpha 上零成本。
- 激活:**块内 MSE scale 搜索**(E4M3 格点,下 3 步 + **上 2 步** —— 上向是本轮新发现,−0.032)。
- decode 相位自适应:M 小走 **W4A16**(免激活量化,精度反升 −0.32)。

### 3.4 系统(两层)
- **kernel:单 launch 双格式融合 GEMM** —— NVFP4 相(block16)+ MXFP8 相(block32)共享 FP32 寄存器累加器,一次 launch 零中间显存;达 CUTLASS 张量核平价(alu/tc≈1);配 per-shape/per-M config dispatch、split-K/stream-K(跨双相 union k-domain)、Marlin 式小 M 路由(M=2..48)+ MXFP8-marlin 高段伴随、decode GEMV(M=1)。
- **serving:vLLM 插件**(custom quant method,不 fork;51+51 CUDA graphs;合并 QKV/gate_up 的 shared-g 重量化)。**数值对齐三级 gate 方法论**(可作为次要贡献写):T0 packed-buffer torch.equal / T1 逐层 bit-exact / T2 logits+PPL|Δ|≤0.1,随 checkpoint 发布参考。

## 4. 主结果(三张表 + 一条曲线)

### 4.1 精度(我们协议,全部 bridge 逐位验证;新锚点)
| | 1B | 3B | 8B |
|---|---|---|---|
| bf16 | 13.16 | 7.82 | 6.14 |
| **ProMix** | **14.17** | **8.22** | **6.56** |
| SharQ | 14.79 | 8.39 | 6.68 |
| ARCQuant | 15.22 | 8.54 | 6.81 |
| vLLM-NVFP4 | 15.65 | 8.63 | 6.87 |
| MicroMix | — | 9.05 | 7.39 |
| 老四样(QuaRot/SpinQuant/Atom/ResQ) | 16.1-22.1 | 8.71-10.7 | 7.06-8.65 |

C4 同号;跨族 Qwen3-8B 成立;校准鲁棒(wiki↔C4 交叉 ≤1%,排名不变)。

### 4.2 kernel(M=8192 同台,全对手真跑)
ProMix vs:fp16 **2.3-3.1×**、vLLM-NVFP4 1.0-1.53×、SharQ 1.12-1.51×、MicroMix 1.4-1.85×、**ARCQuant 平手**(但精度我们赢 → Pareto 支配);融合本身 vs 同算法朴素 2-launch:**1.3×(→1.7× 随 M)**。

### 4.3 serving(vLLM 0.32 同栈五方,b16-512,QServe 式方法论)
- vs **BF16**:decode 1.09-1.68×、prefill 1.26-1.82×,全点赢;
- vs **W4A16-Marlin(weight-only)**:prefill 1.3-2.0×、decode b≥128 1.03-1.24×、QServe 协议 c256 **1.17×/1.34×**(诚实:b1-64 decode 输);
- vs **FP8 W8A8**:大 batch 持平(0.90-1.09)—— **4.6-5.4bit 打平 8.5bit**;
- vs **纯 NVFP4**:0.85-0.93(decode 大 batch)—— 差距=精度的结构价格(已分解到地板:双段 GEMM+act-MSE+Hadamard,离地板 ~4%);
- **论文方法 e2e**:SharQ(其自带 vLLM 集成、同栈):精度平、serving 我们 prefill 2.41×/c256 1.40×;MicroMix 同卡复现其论文 prefill 数但 decode 慢 FP16 4.4×;Quartet as-shipped 不可运行。

### 4.4 money 图:r-sweep 曲线(两轴实测)
3B 6 点(r=1/32..1/4)+ 8B 4 点,每点 gate:**曲线上每个点的精度都压 NVFP4**(连 4.62bit 的 r=1/32 都赢:8B 6.817<6.87),serving 代价对 r 近乎平坦 → **"接近纯 FP4 的吞吐 + 全程更高的精度 + 一个旋钮"**。单点竞品(NVFP4/ARCQuant/SharQ/W4A16/FP8)散布为点,被曲线压住/夹住。

## 5. 机制与消融(reviewer 弹药,全部实测)

1. **命门 gate**:实测-MSE 分配 8.31 < uniform 8.39 < 方差代理 8.47(同预算)—— 方差代理反而比 uniform 差(漏 outlier/clipping),实测失真是正确信号。
2. **环宽 ε/gain 分解**(机制图):S=64→512,旋转代价 ε 与分配增益 gain 的消长;宽环抹花方差尾伤 uniform,**实测分配免疫**(按真实 ΔD 重排)→ 净收益单调扩大。
3. **增益随规模衰减的机理**(1B −0.30 → 13B ≈0):两段式 —— 小→中是 headroom 收缩(gain/headroom 恒 ~6%);中→大是**每环 PPL 敏感度塌 6.2×**(模型冗余把局部 MSE 与 loss 解耦;flip-16 实验)。上游 ΔD 结构 scale-invariant(排除"能量变平"假设)。**诚实定位:per-layer 分配是小模型上的安全精化,不是精度 headline。**
4. **down_proj 全 FP4 显式对照**:强塞 1/8 FP8 等预算更差(+0.056 C4 = 分配收益的 78%),白送 +35% 预算也买不到 → 实测最优非简化。
5. **loss 加权 ΔD**:修复 8B 的 mispick 73% 但不翻正;3B 反伤 → 不 ship,诊断价值(证明局部 MSE≠loss,但冗余是根本)。
6. **校准鲁棒**:敏感度在 GPTQ(+0.068)>基U(+0.043)>我们的分配(≈0);U 子空间跨分布重叠 0.66-0.69 → **PCA 捕捉模型固有 outlier 结构,非数据集特性**。
7. **4 个性能负收益诚实记录**(had_pack、1字节scale、norm-fuse、act_order)+ vLLM compile-cache 地雷(out-of-tree pass 不进 cache key)。

## 6. Related Work 定位(全文读过 9+ 篇,`rd_related_deepread.md`)

- **不能单独 claim**:误差驱动自适应预算(MicroMix/FGMP/SHMQ)、比方差好的子空间(CoQuant —— 但其主张在它自己管道里未复现,我们有数据)、补偿融进 GEMM(MosaicQuant/SharQ/ARCQuant)。
- **可防御 novelty = 完整组合**:W4A4 + PCA 集中旋转分区 + NVFP4/MXFP8 异构 + **单 launch 双格式融合主循环** + 真 Blackwell serving + 单旋钮曲线。**组合无人做过;护城河是 kernel。**
- **正面交锋**:ARCQuant 前提("混合破坏流水线")被我们 kernel 证伪;anti-rotation 三篇(ARCQuant/DualQuant/SharQ)攻击的是全宽 Hadamard 摊开,我们是 PCA 集中+局部环(§5.4 对照佐证)。
- 仅引用:Oltron(ASIC)、SHMQ(INT W8A8)、FGMP(ASIC)、dMX/MosaicQuant/DualQuant(INT/Ada)。
- 口径纪律:**W4A4(激活也量化)为主对比组;weight-only(GPTQ/AWQ)只做参考,绝不混入 peer 对比**。

## 7. 诚实 limitations(主动写,别让 reviewer 先说)

1. per-layer 分配增益随规模→0(8B+ 持平),是"安全精化+曲线每点最优"不是精度主引擎;0-shot 下游看不出分配增益(任务噪声内)。
2. vs 纯 NVFP4 serving 仍 0.85-0.93(decode 大 batch)/0.6-0.68(prefill)—— 结构税(已分解),是精度的价格。
3. 小 batch(b1-64)decode 输 weight-only Marlin(latency-bound 单用户区间)。
4. KV cache 量化行待补(16ch<32block 修复进行中);旋转训练未用(随机正交已近最优,1B 实证)。

## 8. 图表清单(建议)

| 图/表 | 内容 | 数据 |
|---|---|---|
| **Fig.1(teaser)** | accuracy×throughput 散点:ProMix 曲线压住全部单点 | `vllm_final_fiveway.md` §4 + r14 修正 |
| Fig.2 | 方法总览:PCA 集中→环→ΔD 实测→单阈值→双格式融合 kernel | 示意图 |
| Fig.3 | 排序 ΔD 谱(实测,log-log)+ 悬崖恢复 S 坡 | `rd_deltaD_spectrum.html`、accpuzzle |
| Fig.4 | ε/gain 环宽分解(机制) | rd_ron |
| Fig.5 | kernel 架构:单 launch 双相共享累加器 + 路由(GEMV/marlin/split-K/GEMM) | 示意 + perf rounds |
| Tab.1 | 精度主表(4.1,含全部 peer) | evidence index §1 |
| Tab.2 | serving 五方(4.3) | fiveway |
| Tab.3 | kernel M=8192 同台 | rd_baselines_m8192 |
| Tab.4 | 消融(配方各杠杆 + down_proj 对照 + 校准 2×2) | acctune/downctl/calib |
| Tab.5 | 跨论文相对比(QServe 口径) | fiveway 跨论文节 |

## 9. 写作提示

- 数字出处一律查 `paper_evidence_index.md`;协议/桥接数写法见 `baselines_on_hardware.md`;
- 叙事顺序建议:硬件拐点(INT4 死、FP4 生)→ 单点方法的局限 → 三个观察 → 方法 → 曲线结果 → 机制 → 系统;
- "数值对齐 gate 方法论"(T0/T1/T2 + 签名 ΔPPL)可写成 systems 小节,是可复现性卖点;
- 所有"诚实负收益/负结果"保留(4 个 kernel 负收益、loss 加权、CoQuant 未复现、小 batch 输 Marlin)—— 它们是信誉资产;
- 进行中(写作时留占位):ring512→serving 导出(serving 表精度列将更新)、KV4 行。
