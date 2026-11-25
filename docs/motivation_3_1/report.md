# M31 Report — Motivation §3.1 通道级精度分配实验

日期: 2026-11-25 ｜ 任务书: `docs/paper_motivation_3_1_experiments.md` ｜ 数据: `docs/motivation_3_1/{runs.csv,layer_errors.csv,allocations.jsonl,data/}` ｜ 图: `motivation-allocation-{within,across,diag}.{pdf,png}`

## 1. 完成度与验收 gate

- **矩阵完成**: 2 模型 × 99 逻辑配置 = 198/198（+4 浮点 controls = 202 行 runs.csv），无缺点、无静默插值。坏点（若干 OOM，均为共卡冲突）全部重试成功，记录于 failures.csv。
- **端点一致性 gate（任务书 §9.5）**: PASS。原始表示下 RND(3 seeds)/SS/MA × within/across 在 r=0 与 r=1 的 wiki PPL 各只有一个 distinct 值（3B: 14.3309/7.8537; 8B: 7.7818/6.1835）— 端点 mask 结构相同，物理去重复用（逻辑映射保留于各 run json 的 physical_run_id）。
- **无量化等价性 gate**: PASS（selfcheck_3b.json）。置换机械 ΔPPL=-2.4e-5，PCA 变换 ΔPPL=+2.3e-5（vs 共同浮点参考 7.8118）；分段路由单元检验 bit-exact。浮点 controls 的 fp16 数值底噪 E≈7e-8（两表示一致）。
- **E_l 无 NaN**: 198 配置全部 nan_error_layers=[]。近零 ENERGY 层: 无。
- **单调性**: within 曲线全部大体单调；across 的 MA 分支显著非单调（见 §3，经核验为策略性质而非实现错误）。

## 2. 问题 A（层内选哪些通道重要吗？）— 数据回答

Wiki PPL（C4 定性相同，见 runs.csv/图A）:

| | r=0.1 | r=0.5 | r=0.9 | r=0 (纯NVFP4) | r=1 (纯MXFP8) |
|---|---|---|---|---|---|
| 3B SS within | 8.122 | 7.981 | 7.872 | 14.331 | 7.854 |
| 3B MA within | 8.124 | 7.979 | 7.873 | 14.331 | 7.854 |
| 3B AV within (PCA表示) | 8.378 | 8.130 | 8.023 | **10.649** | 7.996 |
| 3B RND s0/s1/s2 | 10.49/19.77/8.85 | ~8.03 | ~7.88 | 14.331 | 7.854 |

(3B float 7.812)。8B（float 6.135）: SS/MA@0.1 ≈6.48, AV 6.62, RND s0/s1/s2 @0.1 = 7.68/6.56/7.61; r=0 raw 7.782 vs PCA 7.932。

观察（不外推为普适结论）:
1. **选择确实重要**: RND 在低比例区显著劣于有分数策略且 seed 间波动巨大（3B r=0.1 跨 seed 8.85–19.77；8B r=0.1 6.56–7.68），r≳0.5 后趋同。
2. SS 与 MA 在同一原始表示上几乎不可区分（≤0.005 PPL），层误差亦然。
3. **AV 携带表示差异**（任务书 §1 已声明 A 不是纯排序消融）: 在 3B，PCA 表示大幅改善纯 NVFP4 端点（10.65 vs 14.33）但在所有混合点与纯 MXFP8 端点上劣于原始表示的 SS/MA（差 ~0.15–0.25 PPL）；**在 8B 连 r=0 端点也更差**（7.93 vs 7.78）。两类指标同向: AV 的层误差 mean 在混合点亦更高（如 3B r=0.1: 0.00507 vs SS 0.00474）。
4. 局部误差与 PPL 大体同向但不成比例（r=0 的 E 均值仅约 0.007–0.009，PPL 却从 7.8 退到 14.3 (3B)）。

## 3. 问题 B（跨层分配重要吗？）— 数据回答

同一分数、同一总预算 H(r) 下 within → across:

- **SS**: across ≈ within（3B 全程差 <0.04 PPL，略差于 within 为主；8B 同）。
- **AV**: across 在低比例略优于 within（3B r=0.1: 8.296 vs 8.378; 8B r=0.1: 6.586 vs 6.620），高比例持平。幅度小（≤0.08 PPL）。
- **MA**: across 发生**灾难性失败**: 3B r=0.1–0.6 PPL 15.7–18.1（高于 r=0 的 14.3）；8B r=0.1–0.7 为 7.0–7.5（劣于 within 约 0.6–1.1）。
  - 机理（allocations.jsonl / 图C 可核验）: 原始分数未做层归一化（任务书 §6 规定），MA 的 mean-abs 在 hidden-stream 投影上远大于 down_proj 输入，全局 top-chunk 把预算全部给 q/k/v/gate/up（入选比例最高 0.83），down_proj 几乎为 0（3B r=0.2 平均 0.1%）；down_proj 的 E_l 居全网最高（~0.013）。r≥0.7 预算溢出到 down_proj 后曲线骤然恢复。
  - 指标背离: 该区间 **E_l 层均值反而低于 r=0**（0.0045 vs 0.0070），PPL 却更差 — 等权层均值掩盖少数关键层（down_proj）的持续高误差；layer_errors.csv 保留全部逐层值与 per-type 统计。
- 结论性观察（限本协议范围）: 跨层自由度本身收益有限（SS/AV 下 |Δ|≤0.08 PPL），但对分数跨层可比性极其敏感；未归一化分数可使 across 远劣于 within。

## 4. 偏差 / 局限（如实记录）

- o_proj 在所有配置固定 FP16（排除原因见 inventory §5）→ 绝对 PPL 不可与主实验块1 曲线直接对比；对照内部自洽。
- allocations.jsonl 不逐 run 落盘完整 stable_selected_channel_ids（体积原因）: 存 h+ids_sha256+score_summary；完整 ids 可由已提交 scores/*.pt + 确定性规则重建并校验 sha（`rd_m31.py alloc --dump`）。
- RND 的 PPL 带 = 3 seeds min/max；层误差带 = 先 seed 平均每层 E 再层间 std（§7.3 口径，图注写明）。
- 两条 lane 并发偶发重复物理运行（结果一致，保留先完成者）。
- 评估器为等长窗：窗均 NLL 均值 == 总 token NLL 均值（任务书 §7.1 数学等价），NLL sum/token 数存于各 run json。
- GPTQ 配方冻结为 metricsweep 主曲线协议（nsamples=128, percdamp=0.01, 默认 MSE env），非 rd_acctune 加强版 — 见 manifest.json。

## 5. 两类指标的初步观察（供论文叙事，不写预设结论）

- 相同精度比例下，层内通道选择影响显著（RND 的劣化与 seed 方差），但同一原始表示上"用哪个量级分数"（SS vs MA）几乎无差别。
- 表示（PCA vs 原始）的影响大于排序指标本身，且方向依赖比例与模型（3B r=0 受益、混合点受损；8B 全程受损）。
- 跨层分配不是免费改进: 同分数下至多小幅收益（AV 低比例 -0.08 PPL），对层间尺度未对齐的分数（MA）是灾难。
