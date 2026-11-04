# 论文正式实验矩阵(evaluation 章 spec)

> 2026-11-03 定稿(与用户确认:加 32B/70B 规模点;领域任务暂不补;"bias 实验"= 系统组件逐项收益分解)。
> 状态标记:✅ 已有(出处见 `paper_evidence_index.md`)/ 🔄 进行中 / ❌ 待跑(本 spec 启动)。

## 块 1:精度主实验

| 实验 | 状态 |
|---|---|
| W4A4 全同行同协议 PPL(wiki+C4,1B/3B/8B/13B) | ✅ |
| 0-shot 6 任务(旧锚点) | ✅;**新配方(ring512)0-shot** ❌ E1 |
| **MMLU 5-shot**(ProMix 新配方 + ARCQuant/SharQ/NVFP4/BF16) | ❌ E1 |
| 领域任务(HumanEval/GSM8K) | 暂不做(用户定) |
| KV4 公平行 | 🔄(feat/rd-kv4)|
| **规模扩展:32B + 70B**(用户定,"B 太小") | ❌ E2 |

**E2 规模扩展细则**:候选 = Qwen3-32B(已缓存!先验 head_dim 兼容性,Qwen3-8B plugin 已通)+ Llama-3.3-70B 或 Llama-2-70B(Llama 架构稳,需下载 ~140GB,先报可行性)。精度管线优先(GPTQ 逐层可在 96GB 单卡+offload 做);测 {bf16 bridge, uniform-1/8, measured-1/8, NVFP4} wiki+C4。serving:量化后 70B(~40GB)单卡可服;**BF16 70B 锚点需 TP=2 而我们插件未做 TP** → 70B serving 先给"量化单卡 vs BF16-TP2"的诚实口径或只报精度,TP 支持另立项。

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
| E1 | MMLU 5-shot + 新配方 0-shot ✅ `rd_mmlu_report.md`(3B ProMix MMLU 第一 54.12;8B 平 SharQ;0-shot 1B/3B 噪声内、8B +0.33)| 4 | 完成 |
| E2 | 32B/70B 规模扩展(可行性→精度→serving-带注) | 0-2 | 2-3 天(含下载)|
| E4 | 算子 Pareto 终版复测(延迟 vs r 全曲线 + 对手点) | 6 | 半天 |
| E5 | 组件累计消融(精度 build-up + 性能 build-up) ✅ `docs/rd_buildup_ablation.md` | 5(精度)/7(性能) | 完成 |
| E6 | 3-seed 误差棒(headline 点) ✅ 同上 | 搭 E5 车道 | 完成 |
| E3 | 8B r-sweep 补密 | 低优,E2 后 | — |
| (KV4) | 🔄 进行中 | 3 | — |
