# 块2-GB10 · Spark (GB10/SM121) Serving 实验 Spec

日期：2026-11-27 ｜ 上位 spec：`docs/paper_experiments.md` 块2（协议权威，本文只定义 GB10 差异项与跟踪表）
执行基建：`docs/vllm_final_scripts/gb10_{env,export_point,serve_point}.sh` ｜ 集成验证记录见 results_tracker "GB10 vLLM 集成收官"

## 1. 目标与叙事

跨代部署故事的 serving 落点：**同一量化 checkpoint 配方 + 同一 vLLM 插件**（零代码适配）在 SM121 桌面超算上的端到端表现。
两个论点：① "精度旋钮近乎免费"（TPOT 随 r 平坦）在带宽受限平台同样成立；② kernel 调优（KDA-1→4，GEMM gap 1.08→1.028）在 serving 端到端的兑现量。
不做跨平台绝对吞吐对比叙事（GB10 LPDDR ~273GB/s vs Pro6000 GDDR7 ~1.8TB/s，平台差 ~14× 属硬件事实，只作 context 注记）。

## 2. 协议（引用块2，仅列 GB10 差异）

| 项 | Pro6000（块2）| GB10 本 spec |
|---|---|---|
| vLLM | 0.32 | **0.33**（插件零适配，见 tracker）|
| 模型 | 1B/3B/8B | **3B 主**（本机有旋转产物；1B 可选扩展）|
| batch | b16-256，b256 主 | b16/64/**128 主**，b256 仅验证（实测 decode b128 后饱和：b128 ~1.8k > b256 ~1.1k tok/s）|
| 导出配方 | serve_point.sh 3b 同款（ring512+measured, gptq512, percdamp .025, WMSE7, ACT_UP2）| 同左，本机导出 |
| 其余 | graphs ON · fp16 · mem-util 0.90 · max_num_seqs 512 · workloads: decode 128/1024 · prefill 1024/128 + L2048 · QServe 1024/512 | 完全相同 |
| kernel 栈 | shipping | **KDA-4 最终版**（FK 窗口 v3 + fk_spg + K_high pad + swizzle v2；fk_spg 带 M%128 守卫，serving 实际路由需 profiler 抽证）|

## 3. 实验矩阵

- **主网格（与 Pro6000 serving 主网格同构）**：r ∈ **{0, 0.1, 0.2, 0.5, 0.8, 1.0}** × 3 workloads ×（b16/64/128 主 + b256 验证）
- **调优前后对比组**：调优前旧点 `gb10_pm3b_r{0,0.125,0.5,1.0}`（2026-11-27 前旧 kernel）vs 新 kernel 同 r 重测；0.125 点与 Pro6000 8B 加密点同位
- 每点先过 gate_t2（dPPL 与集成验证记录一致 ±0.005）再入账

## 4. 跟踪表 D-GB10（做一个填一个）

| r | 导出包 | gate dPPL | TPOT b128 (ms) | TTFT L2048 b128 (ms) | QServe tok/s | 状态 |
|---|---|---|---|---|---|---|
| 0.0 | ✅(已有) | ⬜ 复验 | ⬜ | ⬜ | ⬜ | 🔄 |
| 0.1 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| 0.2 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| 0.5 | ✅(已有) | ⬜ 复验 | ⬜ | ⬜ | ⬜ | 🔄 |
| 0.8 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| 1.0 | ✅(已有) | ⬜ 复验 | ⬜ | ⬜ | ⬜ | ⬜ |

### 表 D-GB10b · 调优前后对比（同 r，旧 kernel 数据=gb10_pm3b_*）

| r | TPOT 旧→新 | TTFT 旧→新 | QServe 旧→新 | 备注 |
|---|---|---|---|---|
| 0.0 | 66.9→⬜ | 14291→⬜ | 481→⬜ | 旧值为调优前 b128 |
| 0.125 | 69.9→⬜ | 15103→⬜ | 488→⬜ | 对照点（非主网格）|
| 0.5 | 68.9→⬜ | 16468→⬜ | 475→⬜ | |
| 1.0 | 70.9→⬜ | 18298→⬜ | 472→⬜ | |

### 表 D-GB10c · 坏点记录
| r | workload | 原因 |
|---|---|---|
| (空) | | |

## 5. 质量 Gate

1. gate_t2 dPPL ±0.005（vs 集成验证记录）；2. co-tenant=0（开测前后各验一次）；3. 趋势同构性：TTFT 随 r 单调升、decode 近平（与 Pro6000 同构，背离要查）；4. fk_spg/castfold 在 serving 下的实际 dispatch 路由 profiler 抽一次证据；5. 坏点 skip+记录不阻塞。

## 6. 数据位置

- 新数据：`docs/vllm_final_data/gb10v2_pm3b_r*_{main,qserve}.json`（gb10**v2** 前缀 = 调优后 kernel）
- 调优前旧点：`docs/vllm_final_data/gb10_pm3b_r*_{main,qserve}.json`（保留，仅对比用）
- gate：`gate_t2_gb10v2_pm3b_r*.json` ｜ 导出 meta：`gb10v2_pm3b_r*_export_meta.json`
