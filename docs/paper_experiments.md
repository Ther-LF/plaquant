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

## 块 2:端到端 serving

| 实验 | 状态 |
|---|---|
| 五方同栈(b16-512)+ 论文方法 e2e(SharQ/MicroMix)+ 跨论文表 | ✅ |
| e2e Pareto(r-sweep 曲线) | ✅(3B 6 点);**8B 补密至 6 点** ❌ E3(低优)|
| 32B/70B serving 点 | ❌ E2 附带(带 TP caveat)|

**决定:不做"单层比较"档** —— 无 serving 的对手已在算子级同台(比单 block 口径更强)。

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
| E1 | ✅ `rd_mmlu_report.md`(3B ProMix MMLU 第一 54.12;8B 平 SharQ;0-shot 1B/3B 噪声内、8B +0.33)| 4 | 完成 |
| E2 | 32B/70B ✅ `rd_scale_report.md`(70B 全矩阵;32B arch no-go;领先 70B 翻转+0.045 诚实;单卡量化70B 完胜双卡BF16 3.4-4.2×/GPU;ring512 分配增益 70B 回归) | 0-2 | 完成 |
| E4 | ✅ `rd_operator_pareto.md` | 6 | 完成 |
| E5 | 组件累计消融(精度 build-up + 性能 build-up) ✅ `docs/rd_buildup_ablation.md` | 5(精度)/7(性能) | 完成 |
| E6 | 3-seed 误差棒(headline 点) ✅ 同上 | 搭 E5 车道 | 完成 |
| E3 | 8B r-sweep 补密 | 低优,E2 后 | — |
| (KV4) | ✅ `rd_kv4_fix.md` | 3 | 完成 |
