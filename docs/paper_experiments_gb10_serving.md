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
| 0.0 | ✅ | ✅逐位 | 63.1 | 14094 | 3491 | ✅ |
| 0.1 | ✅新导 | -0.0043 | 57.0 | 14808 | 3435 | ✅ |
| 0.2 | ✅新导 | -0.0019 | 63.0 | 15101 | 3419 | ✅ |
| 0.5 | ✅ | ✅ | 68.0 | 16440 | 3333 | ✅ |
| 0.8 | ✅新导 | +0.0015 | 57.6 | 16974 | 3388 | ✅(boot3) |
| 1.0 | ✅ | ✅ | 65.5 | 17759 | 3361 | ✅(boot3) |

### 表 D-GB10b · 调优前后对比（同 r，旧 kernel 数据=gb10_pm3b_*）

| r | TPOT 旧→新 | TTFT 旧→新 | QServe 旧→新 | 备注 |
|---|---|---|---|---|
| — | 见注 | L2048: r0.8 17342→16974(-2.1%) r1.0 18185→17759(-2.3%) | — | **对比口径修正**: 旧 gb10_pm3b 与新数据间隔两次整机重启, decode/吞吐差异(+20-64%)由机器状态主导不可归因 kernel; 调优收益以同机 nofk 基线隔离: prefill 高 r 净 -2~-3%(与 GEMM gap 1.08→1.028 定量自洽), decode 设计上 FK 不触及 |

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


## 7. 执行后发现（2026-11-28 入档）
- **plugin FK 接线修复**: plugin.py 原未传 cd_fk/cd_fk_spg 给 MixedQuantLinear——此前 serving 一直跑 castfold 老栈; 修复后 nsys 证据: fk_smallk 在 o_proj prefill chunk 命中(GPU 2.8-3.2%), castfold 仍 36%。
- **fk_spg 在 3B serving 不命中**: 窗口表按 eval 拆分形状测, serving 合并形状(qkv 5120/gate_up 16384/down 8192)不在表; decode M≤128 低于全部窗口下界 → 后续项: 按 serving 合并形状重测窗口表。
- GB10 decode 指标跨 engine 实例波动 ±9-18%(实例内 ±0.05%), decode 按实例级不稳定对待; prefill/qserve 稳定 ±1%。b256 饱和确认, b128 甜点。
- 趋势与 Pro6000 同构(TPOT/TTFT 随 r 单调升, GB10 斜率更缓)。

## 8. P0-1 · 同机基线对照组（gb10v3，2026-11-28 入档）

协议：与 gb10v2 完全同栈（vLLM 0.33 · graphs ON · prefix caching OFF · greedy+ignore_eos · 1 warmup + median of 3 · mem-util 0.90 · max_num_seqs 512）。main b16/64/128 + prefill L2048；qserve 1024/512 加测 **c256**。
**同 boot 硬约束达成**：boot_id `de09c9e5-681a-4c8a-b28b-795500f07782`，7 个 serve 点 + T2 gate 于 07:21–11:19 CST 连续跑完，无重启（session log 每点记 boot id）。

Checkpoint 溯源（与 Pro6000 同 checkpoint）：
- FP8：Spark 本地按同配方 data-free 重建（`gb10v3_make_fp8.py`），**全部 450 tensor 与 Pro6000 checkpoint sha256 逐字节一致**（需 fp32 除法 + 直接 RNE cast 才能复现 ct-0.19 行为）。
- GPTQ/NVFP4：直连管道传输，model.safetensors 两端 sha256 一致。BF16 = 同 snapshot hash（d444645…）。

锚点校验：gb10v3_pm3b_r0.125 vs gb10v2 同点 — main 全部 15 指标 ±1.2%，qserve ±3%（门 ±5% 过）；gate_t2 PPL 8.29126（dPPL +2.6e-4，与 gb10v2 gate 逐位一致）。

### 表 D-GB10d · 3B 五方法同机对照（decode TPOT=128/1024；QServe=total tok/s）

| 方法 | TPOT b16 (ms) | TPOT b128 (ms) | TTFT L2048 b128 (s) | QServe c64 | c128 | c256 | vs BF16 (TPOT b128 / QServe c256) | vs FP8 |
|---|---|---|---|---|---|---|---|---|
| BF16 | 34.0 | 72.2 | 21.18 | 2274 | 1983 | 1368 | 1.00x / 1.00x | 0.81x / 0.42x |
| FP8-W8A8 | 21.8 | 58.4 | 24.86 | 2782 | 3138 | 3249 | 1.24x / 2.38x | 1.00x / 1.00x |
| GPTQ-W4A16 | 15.7 | 56.3 | 24.87 | 3004 | 3189 | 3165 | 1.28x / 2.31x | 1.04x / 0.97x |
| NVFP4-W4A4 | 16.8 | 53.1 | 11.30 | 3302 | 3637 | 3733 | 1.36x / 2.73x | 1.10x / 1.15x |
| AdaMix r0.0 | 16.8 | 56.2 | 14.09 | 3156 | 3495 | 3610 | 1.28x / 2.64x | 1.04x / 1.11x |
| **AdaMix r0.125** | 18.8 | 62.9 | 14.94 | 3131 | 3482 | 3599 | 1.15x / **2.63x** | 0.93x / **1.11x** |
| AdaMix r1.0 | 22.4 | 60.7 | 17.68 | 2852 | 3248 | 3403 | 1.19x / 2.49x | 0.96x / 1.05x |

### 发现
- **"什么时候该用 AdaMix 而不是 FP8"**：GB10 带宽受限下，低批 decode（b16 TPOT 21.8→18.8ms，**-14%/1.16x**）与高并发连续批处理（qserve c256 **+11%**）AdaMix r0.125 均稳定快于 FP8；prefill TTFT 14.9s vs FP8 24.9s（**1.66x**，FP8 prefill 带 quant 开销反比 BF16 21.2s 慢）。即 ~5bit 的带宽红利在统一 LPDDR 平台全负载段兑现。
- **NVFP4 可跑**：SM121 vLLM 0.33 经 `FlashInferCutlassNvFp4LinearKernel` 正常运行（非 runnability 缺口）；吞吐略高于 AdaMix（c256 +3.7%），但精度不可比（NVFP4 W4A4 PPL 显著劣于 AdaMix ~5bit，见质量册）。
- **BF16 qserve 非单调**：c64 2274 → c128 1983 → c256 1368 total tok/s，3 runs 一致（±3%），无 preemption、KV 充裕（954k tokens）——BF16 大批 decode 在 GB10 上实测塌陷，量化方法均无此现象；BF16 行按实测入账。
- **8B 未跑**：Spark 外网带宽实测 ~0.4MB/s（hf-mirror/pypi ~KB/s 级），8B 四基线 + BF16 base ≈ 35GB ≈ 24h+ 传输，且本机无 AdaMix-8B 导出包；3B 组先行入账，8B 待带宽窗口。
- 数据：`docs/vllm_final_data/gb10v3_{pm3b_r0.0,pm3b_r0.125,pm3b_r1.0,bf3b,fp83b,gptq3b,nv3b}_{main,qserve}.json` + `gate_t2_gb10v3_pm3b_r0.125.json` + `gb10v3_session.log`（boot id/时间戳）；脚本 `docs/vllm_final_scripts/gb10v3_{serve_point,session}.sh`、`gb10v3_make_fp8.py`。
