# 论文正式实验矩阵(evaluation 章 spec)

> 2026-11-03 定稿(与用户确认:加 32B/70B 规模点;领域任务暂不补;"bias 实验"= 系统组件逐项收益分解)。
> 状态标记:✅ 已有(出处见 `paper_evidence_index.md`)/ 🔄 进行中 / ❌ 待跑(本 spec 启动)。

## 块 1:精度主实验 —— ProMix 比例曲线(2026-11-20 重定义,方案 A)

> **重定义**:旧版是"发货点 r=1/8 的 ProMix vs 对手"单点对比表(误把默认配置当成实验)。
> 现在:精度主实验 = **ProMix 沿高精度比例 r 的精度曲线**;对手各用自己固定配置,是曲线旁的
> **参照散点/横线**。**不再有单点对比表**。

**实验本体**:ProMix 高精度比例 r 的精度曲线。每个「模型 × 指标」画 **两条**(两种 KV 配置):
- **W4A4 / KV16** —— 发货默认,KV 不量化
- **W4A4KV4** —— KV 4-bit,用于和 INT4-KV4 对手(ResQ/QuaRot/Atom/SpinQuant)**同-KV 公平比**
  (KV 位宽:option(a) 每头高段凑整到 32 ≈ 5.44 b/v@head128;见 `rd_kv4_fix.md`)

**自变量 r**:均匀 11 点 `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0`
(r=0 = 纯 NVFP4 左锚点;r=1 = 全高精度 MXFP8)。
**坏点规则(铁律)**:某 r 若不能块对齐/整除、或量化/eval 报错 → **跳过该点 + 记录原因(跟踪表 C) +
汇报**,绝不让一个坏点卡住整条曲线(先例:融合 qkv 在某些 r 产生 mixed K-split 断言失败)。

**指标(4,= W4A4 对手并集,已核够)**:WikiText-2 PPL · C4 PPL ·
0-shot 6 任务(piqa/arc_e/arc_c/hellaswag/winogrande/lambada)均值 · MMLU 5-shot。
领域任务(GSM8K/HumanEval):W4A4 量化论文基本不用 → 不做。

**模型(7)**:Llama-3.2-1B · Llama-3.2-3B · Llama-3-8B · Llama-3-70B ·
Qwen3-8B · Qwen3-14B · Qwen3-32B(含 Llama↔Qwen 跨架构泛化)。

**对手(曲线旁参照,各自固定配置,需跑齐全 7 模型)**:
- **W4A4 同类**:SharQ · ARCQuant · NVFP4 · MicroMix · CoQuant
- **INT4-KV4**(对 ProMix-KV4 曲线比;Blackwell 无 INT4 通路 → s4 降 INT8,带 caveat):
  QuaRot · SpinQuant · Atom · ResQ(ResQ 取 W4A4KV4 版本以同-KV)
- **参照组(别类)**:bf16 上界 · GPTQ/AWQ W4A16 · FP8 W8A8

**协议**:fake-quant 算法级;ring512 + measured-ΔD 分配;校准 WikiText-2 512 samples/seqlen2048/
seed0;PPL seqlen2048;lm-eval 0.4.5;硬件 RTX Pro 6000 SM120。driver = `rd_mmlu.py --budget_frac r
--c4 --mmlu --zeroshot`(一次量化出全 4 指标)。

> **规模 caveat(原 E2)**:70B 走 GPTQ 逐层 offload(96GB 单卡);70B 下游(0-shot/MMLU)需先跑通
> offload-lm-eval 通路(目前只有 NVFP4 代理)。serving 的 70B 口径见块 2。

### 跟踪表 A — ProMix 曲线进度(做一点填一点;格式 `KV16 n/11 · KV4 n/11`;⬜ = 未启动)
| 模型 | wiki | C4 | 0-shot | MMLU |
|---|---|---|---|---|
| Llama-3.2-1B  | ⬜ | ⬜ | ⬜ | ⬜ |
| Llama-3.2-3B  | ⬜ | ⬜ | ⬜ | ⬜ |
| Llama-3-8B    | ⬜ | ⬜ | ⬜ | ⬜ |
| Llama-3-70B   | ⬜ | ⬜ | ⬜ | ⬜ |
| Qwen3-8B      | ⬜ | ⬜ | ⬜ | ⬜ |
| Qwen3-14B     | ⬜ | ⬜ | ⬜ | ⬜ |
| Qwen3-32B     | ⬜ | ⬜ | ⬜ | ⬜ |

### 跟踪表 B — 对手覆盖(每格 = 该对手×模型 的 4 指标;✅全测 / 部分列指标 / ⬜待跑 / — 不适用)
| 对手 | 1B | 3B | 8B | 70B | Qwen3-8B | Qwen3-14B | Qwen3-32B |
|---|---|---|---|---|---|---|---|
| SharQ     | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| ARCQuant  | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| NVFP4     | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MicroMix  | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| CoQuant   | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| QuaRot    | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| SpinQuant | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Atom      | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| ResQ      | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| bf16      | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| GPTQ/AWQ W4A16 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| FP8 W8A8  | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### 跟踪表 C — 跳过记录(坏点 + 原因)
| 模型 | KV | r | 指标 | 原因 |
|---|---|---|---|---|
| (空) | | | | |

## 块 2:端到端 serving —— 同栈 vLLM(QServe 式,2026-11-20 重定义)

> 诚实对标 = **同一推理栈(vLLM 0.32, RTX Pro 6000 SM120)下真跑**。torch-only / 只测单 block /
> 无 vLLM kernel 的对手**不进 serving**(他们的 headline 虚高,见 `quant_e2e_methodology_survey.md`),
> 改用跨论文表(自报加速比)+ BOPs。

**比较对象(分档)**:
- **同栈真跑(serving 同台)**:BF16 · **FP8 W8A8** · NVFP4 W4A4 · **ProMix W4A4** · SharQ(自带 vLLM 集成)
- **weight-only 一档(不同计算类,单列标注"只省带宽不省 compute")**:GPTQ-Marlin W4A16 · AWQ-Marlin W4A16
- **论文方法 e2e 同卡复现**:MicroMix(可跑但 decode ~4.4× 慢,诚实列)
- **跨论文相对表(自报加速比,相对参照,不同台)**:QServe 1.2–2.4× · Atom 2.53× · AWQ · QuaRot …(口径见 survey)
- **跑不了 serving → BOPs(理论 + 门后真实, `theoretical_complexity.md`)**:
  ARCQuant(自家 CUTLASS 非 vLLM) · CoQuant · QuaRot/Atom/ResQ/SpinQuant(INT4 无通路) · FGMP/Oltron(ASIC)

**测试方案(统一协议)**:
- **batch 网格**:`b1 / 16 / 32 / 64 / 128 / 256 / 512`(bs=1 latency-bound → 大 batch compute-bound)
- **负载(ISL/OSL)**:prefill-heavy **1024/128** · decode-heavy **128/1024** · 纯 prefill TTFT **L2048** ·
  QServe 连续批处理 **1024/512@并发**(c64/128/256) · 长上下文 **2048/4096**(fp8-KV)
- **记录指标**:**TTFT**(prefill 延迟) · **TPOT**(decode 每 token) · **gen tok/s**(吞吐) ·
  **并发吞吐** · **容量**(KV GiB / 最大可服 batch)

**实验**:
1. 同栈多方 × 上面全网格 —— 大体 ✅(`vllm_final_fiveway`);补缺:batch512、8B 长上下文、ShareGPT 真实分布
2. **ProMix serving r-sweep 曲线** —— 与块1 **同 11 点均匀网格**(0→1),每点过 T2 gate;看 serving 代价随 r 的平坦度(卖点:接近纯 FP4 吞吐 + 全程更高精度 + 一个旋钮)
3. 跨论文相对表 + BOPs(非同栈对手)
4. **70B**:量化单卡 vs BF16-TP2 诚实口径(插件无 TP)
5. 维持决定:**不做"单层比较"档**

**模型范围**:全 7 个(3B/8B 主;1B sanity;70B 带 TP caveat;Qwen3-8B/14B/32B 取决于 vLLM 插件 Qwen 支持——**待验**,Llama 路径现成)

### 跟踪表 D — serving 进度(⬜ 未启动 / 🔄 部分 / ✅ 齐)
| 模型 | 同栈多方(全网格) | ProMix r-sweep(11点) | 已知缺格 |
|---|---|---|---|
| Llama-3.2-1B | ⬜ | ⬜ | — |
| Llama-3.2-3B | 🔄(缺 b512/ShareGPT) | ⬜ | b512, ShareGPT |
| Llama-3-8B   | 🔄(缺 b512/长上下文/ShareGPT) | ⬜ | b512, 长上下文, ShareGPT |
| Llama-3-70B  | ⬜(量化单卡 vs BF16-TP2) | ⬜ | TP caveat |
| Qwen3-8B     | ⬜ | ⬜ | 插件 Qwen 支持待验 |
| Qwen3-14B    | ⬜ | ⬜ | 同上 |
| Qwen3-32B    | ⬜ | ⬜ | 同上 |

## 块 3:算子级

| 实验 | 状态 |
|---|---|
| M=8192 全对手算子同台 + 融合价值 | ✅ |
| **算子 Pareto 曲线(终版 kernel 的 延迟-vs-r 全曲线 0→1,对手算子为点)** | ❌ E4(旧 ratio-curve 是优化前 kernel,需终版复测)|

## 块 4:机制/内部分析

全部 ✅(ΔD谱、三方gate、ε/gain、悬崖、衰减机理、down_proj 对照、校准2×2、loss加权、负收益)。
**新增(用户定)E5 —— 系统组件逐项收益分解(cumulative build-up 表,"bias 实验")** ✅ 已完成(`docs/rd_buildup_ablation.md`,feat/rd-buildup):
- **精度侧**(同协议 3B 为主、1B/8B 复核):base(RTN 纯NVFP4 无旋转)→ +PCA U → +ring512 R1 → +GPTQ → +两级 weight scale → +act-MSE(d3u2)→ +measured 分配 → +decode-W4A16 —— **每步一行,累计 PPL**,一张表看清每个组件买了多少;
- **性能侧**(vLLM serving,3B/8B b128):朴素 2-launch → 单 launch 融合 → +per-shape dispatch → +split-K → +GEMV/marlin 路由 → +pack/gather/cast 融合 —— 每步累计 TPOT/吞吐。
- 素材大部分已散落(acctune 逐杠杆/perf rounds),但要**同一基底一次性重跑成干净的累计表**(消融表是论文核心,不能拼凑)。
- **误差棒 E6**:headline 数字(3B/8B 的 ProMix 主点 + gate 三方)跑 3 个校准种子,报 mean±std。✅ 已完成(同上:3B 8.2158±0.0020 / 8B 6.5668±0.0026,gate 三方排序 3 seed 全稳)。

## 块 5:配方消融

✅(acctune 逐杠杆表)。E5 的 build-up 表是它的"累计"视图。

---

## 待跑清单(负责/车道/预估)

| ID | 实验 | GPU 车道 | 预估 |
|---|---|---|---|
| **E0(新,块1重定义 2026-11-20)** | **精度主实验 = ProMix 比例曲线**(7模型×4指标×{KV16,KV4}×11点);旧 E1/E2 的单点精度被此取代为曲线campaign;对手需跑齐全模型(表B) | 见下 | 🔄 **开跑中**(表A/B/C 跟踪)|
| E1(旧) | ⚠️ 已并入 E0:旧版是发货点 r=1/8 单点(`rd_mmlu_report.md` 3B MMLU 54.12 等)——作为曲线 r=1/8 的历史锚点保留,但精度主实验以 E0 曲线为准 | 4 | 并入 E0 |
| E2(旧) | ⚠️ 已并入 E0:70B/32B 的单点(`rd_scale_report.md`)并入 E0 的模型维度(70B 曲线 + Qwen3-32B);serving 的 70B 口径仍在块2 | 0-2 | 并入 E0 |
| E4 | ✅ `rd_operator_pareto.md`(算子 Pareto,块3)| 6 | 完成 |
| E5 | 组件累计消融(精度 build-up + 性能 build-up) ✅ `docs/rd_buildup_ablation.md` | 5(精度)/7(性能) | 完成 |
| E6 | 3-seed 误差棒(headline 点) ✅ 同上 | 搭 E5 车道 | 完成 |
| E3 | 8B serving r-sweep 补密(块2,非精度)5/6,第6点撞 qkv mixed-K-split bug 待修 | 低优 | 🔄 5/6 |
| (KV4) | 机制已通 `rd_kv4_fix.md`;KV4 精度曲线并入 E0(每格第二条曲线)| 3 | 并入 E0 |
