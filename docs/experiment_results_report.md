# AdaMix (ProMix) — 实验结果总报告

日期：2026-11-28 ｜ 状态：全部实验线收口 ｜ 协议权威：`paper_experiments.md`（+GB10 serving spec / motivation 任务书×4）
数据可溯源：`dashboard_data.json`（每值带 source）· 可视化 `results_dashboard.html` · 时间线 `results_tracker.md`

---

## 1. 块1 · 精度比例曲线（7 模型 × 双 KV × 11 点 × 4 指标）

**覆盖**：Llama-3.2-1B / 3.2-3B / 3-8B / 3-70B + Qwen3-8B / 14B / 32B，KV16+KV4 双曲线，wiki/C4/MMLU/0-shot 四指标（70B KV16 另有 0.05 密网格 20 点；70B/32B 下游为 r=1/8 单点 + lim25 协议）。

### 关键数字（wiki PPL，KV16，r=0 → r=0.1 → r=1/8 → r=1.0 ｜ bf16 锚）
| 模型 | r=0 | r=0.1 | r=1.0 | bf16 |
|---|---|---|---|---|
| Llama-3.2-1B | 15.68 | 14.86 | ~13.9 | 13.16 |
| Llama-3.2-3B | 9.37 | 8.26 | ~8.0 | 7.82 |
| Llama-3-8B | 7.36 | 6.61 | ~6.3 | 6.14 |
| **Llama-3-70B** | **29.41（悬崖）** | **6.06** | 5.76 | 5.32 |
| Qwen3-8B | 10.92 | 9.90 | ~9.8 | 9.73 |
| Qwen3-14B | 9.77 | 8.85 | 8.73 | 8.65 |
| Qwen3-32B | 8.51 | 7.79 | 7.69 | 7.61 |

### 核心发现
1. **70B r=0 悬崖（混合精度必要性的规模涌现证据）**：wiki 29.41 / C4 41.67 双数据集确认；r=0.1 即恢复 6.06；KV4 复现同形（30.21）；小模型 r=0 不崩（+1~2.5）→ 规模涌现
2. **KV4 近零代价**：全模型全 r 段 KV4 比 KV16 差 ≤0.02-0.09（70B 仅 +0.008~0.02）
3. **跨架构泛化**：Qwen 三尺寸曲线形态与 Llama 一致，r≥0.1 全程压 NVFP4 对手（8B 9.90<10.07、14B 8.85<8.90、32B 7.79<8.00）
4. 下游随 r 平稳（lim25 噪声带内），70B 下游 r=1/8：MMLU 77.33 / 0-shot 78.0（排序随 wiki）

数据：`docs/rd_metricsweep_data/` · `docs/rd_acctune_data/densesweep/`

---

## 2. Related Work 覆盖矩阵（13 方法 × 7 模型 × 4 指标，全部收口）

**实测格 ~190+**（wiki 全 / C4 全 / 下游全，分五批补齐）；缺格全部定性入档（missing 台账）。

### 精确等效位宽（含 scale/metadata，推导 `effective_bits.md`）
QuaRot/SpinQuant 4.005 · Atom 4.17-4.30 · **NVFP4 4.500** · ResQ/CoQuant 4.51 · **ProMix 4.5+3.75r** · ARCQuant 5.12-5.22 · MicroMix 5.18-5.23 · **SharQ 6.125**（激活实为 7.5bit 双流）· FP8 8.006 · GPTQ/AWQ W4.125/A16（weight-only tier）

### 代表值（8B wiki ｜ MMLU）
bf16 6.14｜65.3 → FP8 6.19｜65.2 → GPTQ 6.53｜62.6 → AWQ 6.73｜63.3 → SharQ 6.68｜62.9 → ARC 6.81｜62.2 → NVFP4 6.87｜62.3 → ResQ 7.06｜59.9 → CoQuant 7.35｜50.3* → MicroMix 7.39｜59.1 → Atom 7.57｜56.8 → SpinQuant 7.88｜55.4 → QuaRot 8.65｜50.2
（四件套四指标跨方法排序完全单调一致 = 管线可信）

### 定性闭环（论文证据，非缺测）
- **SharQ/ARCQuant/MicroMix/CoQuant/INT4 四件套 × Qwen：不可迁移**（SharQ 崩 wiki 42804；ARC 缺 Llama 专用 reorder 工件；MicroMix kernel 硬拒 1B K=2048）
- 70B 下游 7 方法齐：bf16 80.63 > FP8 80.35 > AWQ 79.79 > GPTQ 79.09 > SharQ 78.53 > ARC 77.47 ≈ ProMix r=1/8 77.33（MMLU 严格随 wiki 排序）
- 注记项：AWQ-1B wiki 重对齐 14.53→15.16（旧值管线错位）；QuaRot/ResQ-1B 存 torch 版本敏感双值

数据：`docs/rd_newacc_backfill/`（~113 格逐格 JSON 含协议/gate/raw）

---

## 3. 块2 · Serving（vLLM 同栈实测）

### Pro6000（SM120，vLLM 0.32，graphs ON）
1B 14 点 / 3B 21 点 / 8B 16 点 r-sweep。**TPOT r=0→1 增幅（口径已核准 2026-11-28）**：
- QServe 连续批处理（1024/512）**b256**：1B +3.8% / 3B +1.5% / 8B +4.5%（旧 +3.7% 为同口径早期数据的舍入）
- decode-heavy（128/1024）b256：1B +3.9% / 3B +2.7% / 8B +9.4%；b16 小批最敏感（3B +41% / 8B +46%，launch-bound）
论文采用 QServe b256 口径并标注 batch；"精度旋钮近乎免费"的表述限定在饱和批（b128-256）。TTFT 随 r 单调升，趋势干净。
数据：`docs/vllm_final_data/pm*_{main,qserve}.json`

### GB10（SM121，vLLM 0.33，**插件零代码适配**）
- 跨设备数值对齐：同配方包 vLLM PPL vs Pro6000 Δ≤0.012（r0.5 仅 0.0008）→ **同一 checkpoint + 同一插件跨 SM120/121 部署，数值逐点对齐**
- 7 点十进制网格（表 D-GB10）：TPOT b128 57-68ms、TTFT L2048 14.1→17.8s 单调、QServe 3.3-3.5k tok/s；b128 为吞吐甜点（b256 饱和）；趋势与 Pro6000 同构（GB10 decode 斜率更缓=带宽主导）
- FK kernel serving 接线修复（plugin.py 原只在 eval 生效）+ nsys 路由证据；FK prefill 高 r 净收益 -2~-3%（与 GEMM gap 提升自洽）
- 方法论入档：机器重启对 decode 影响 +20-64% ≫ kernel 效应，对比用同机基线隔离

数据：`docs/vllm_final_data/gb10v2_*`（48 文件）｜ spec：`paper_experiments_gb10_serving.md`

---

## 4. 块3 · 算子（双平台，gap-to-theory 验收）

理论线 t(r)=t_FP4·(1−r)+t_FP8·r（本机 cuBLASLt kernel 级 baseline 两端插值，backend 均有 profiler 证据）。

| 平台 | gap 中位 | 最差 | 说明 |
|---|---|---|---|
| Pro6000（shipping）| 1.109 | 1.889 | 18 shapes × 2M × 11r |
| GB10 调优前 | 1.080 | 1.519 | 同套件 |
| **GB10 KDA-4 最终（v4）** | **1.028** | 1.209 | r=0 列 1.011（含 sub-theory 点 0.940）|

- **单 launch 融合**：profiler 复验（CodesignFusedKernel×1/call），双 phase 共享寄存器累加器；vs bf16 中位 2.4×、vs 纯 FP8 1.3-1.6×（精度相当区间）
- 方阵套件（4096³-16384³，双平台）：GB10 16384³ gap≈0.99 — 大 K 已到理论线，剩余肉在短 K
- **KDA 四轮调优**（log `kda_gb10_log.md` Round 1-17）：castfold / GB10 窗口表 / K_low+K_high 128-pad / swizzle v1-v2 / fk_smallk+fk_spg（ForgeKernel 进化）/ walk 翻案；cosine 全程零回归
- **NCU 定界**（Round 15 普查）：全场 1 block/SM、~20% occupancy、tensor-pipe issue-latency bound — 带宽不是墙；剩余肉 = MX-down mainloop（~1.1-1.25×）等三处，全落地约 median 1.01（记 future work）
- ForgeKernel：SM121 全链路跑通（hw_kb 开树 + 2 条 certified 机制），F3 进化 kernel 合入，SASS recon 报告

数据：`docs/rd_oppareto_data/` ｜ kernel 详解：`kernel_optimizations.md`

---

## 5. Motivation（§3.1-3.3，四任务 530 配置）

详见 `motivation_results_summary.md`。一句话版：
- **M31**（202）：通道选择重要（RND 差 2+ PPL、seed 方差到 11）；MA-across 灾难（未归一化分数饿死 down_proj 且层均误差假性下降）
- **M31-ext**（120+154）：悬崖区解析（r=0.02 即 14.33→8.22）；**逐通道格式感知 DD 反例**（连 act_mse 都输）→ 跨层分数必须"同量纲+按打包单元聚合" = measured-ΔD 以 ring 为单位的设计理由
- **M32**（48）：整组旋转层误差降但 PPL 低 r 升/高 r 降 → 误差落点＞大小 → ring512 块对角粒度
- **M33-ext**（160）：naive 2-launch 全程拱起 1.14-1.67×（最差恒在 r=1/16）；四类系统税与 §4.3 融合消除项 1-to-1；decode 下 +在线旋转 2.35× BF16

---

## 6. 质量纪律（贯穿全部结果）

- 每个新落点 sanity-gate：单调性 / >bf16 / 无 NaN / KV4≥KV16 / 量级合理
- 对手每格先 wiki gate（复现偏差 <0.05）再测新指标；管线零漂移多次验证（逐位/3-4 位小数复现）
- kernel 每轮：cosine 逐点不降（多数 bit-exact）+ 单 launch profiler 复验 + 入账计时无 co-tenant
- 不利结果全部如实入档（M32 交叉、DD 反例、gate_up 例外、0-shot 噪声注记、机器状态效应等）

## 7. 已知边界 / Future Work

FK·spg 窗口按 serving 合并形状重测 ｜ MX-dominant down mainloop（70B 尾 gap 1.209）｜ Qwen×INT4 四件套（低优）｜ FP-Quant/QuTLASS 无 runner ｜ serving backlog（Qwen/70B serving、ShareGPT）｜ 32B 下游全曲线（现 ppl 全+下游单点协议）
