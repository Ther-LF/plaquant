# M31 对齐与执行指导 — Motivation §3.1 ↔ 主实验 campaign

日期：2026-11-24 ｜ 任务书：`docs/paper_motivation_3_1_experiments.md`（协议权威，本文不复述、不覆盖）
本文回答：任务书落到我们现有基建上**怎么跑**——复用什么、新建什么、在哪跑、怎么跟踪。冲突时以任务书为准。

## 1. 与主实验的关系（一句话）

主实验（paper_experiments.md 块1）回答"ProMix 整法有多好"（ring512+measured+GPTQ 全配方曲线）；
M31 回答"为什么要这样设计"的第一层：**选哪些通道重要吗（A）、跨层分配重要吗（B）**。
两者共享：11 点均匀 r 网格、NVFP4/MXFP8 真格式 fake-quant、wiki/C4 PPL 协议（seqlen2048、C4 256 窗）、WikiText 512×2048 seed0 校准。
两者隔离：M31 以**通道**为精度单位（非 ring）、固定 KV16、不跑 MMLU/0-shot、输出到 `docs/motivation_3_1/`，绝不写主实验数据目录。

## 2. 基建映射（复用 vs 新建）

| 需求 | 复用 | 新建 |
|---|---|---|
| runner | eval（evaluate_ppl/get_c4）、quant_utils 的 nvfp4/mxfp8 fake-quant 与 block-scale、loader/校准 hook 参考 rd_validity | `promix/scripts/rd_m31.py`：--model --strategy {rnd,ss,ma,av} --scope {within,across} --ratio r --assignment_seed；内含分数收集、可行配额 F_l、打包、GPTQ（冻结主配方超参）、teacher 层误差 hook |
| launcher | metricsweep.sh 的模式：幂等 complete() 检查、坏点 skip→failures.csv、逐点 git commit、MASTER_PORT=29500+GPU | `promix/scripts/m31sweep.sh`：按任务书 §9 顺序排点（不是盲扫网格） |
| 产出 | — | 任务书 §8 全套：inventory.md / manifest.json / runs.csv / layer_errors.csv / allocations.jsonl / failures.csv / report.md / plot.py，全部在 `docs/motivation_3_1/` |

**明确不做**：改 quant_utils 成每通道 scale（任务书 §5 禁止）；用 `--ring_size 1` 冒充通道级（任务书 §9.2 点名禁止）；复用块1 曲线充当策略曲线（§2 点名禁止）。

## 3. 资源排班（Pro6000，与主实验并行）

- **精度实验可共卡**（campaign 纪律：只有 serving 必须独占）。3B fp16 ~7GB、8B ~17GB，96GB 卡一张可塞 3-4 条 lane。
- 排班：GPU4-7 上铺 **4-6 条 M31 lane**（如 GPU4: 3B-A策略×2 lane；GPU5: 8B-A×2；GPU6: 3B-B+RND seeds；GPU7: 8B-B），各 lane 按策略×scope 切分互不重叠的配置子集，MASTER_PORT 区分。也允许与 32B/70B 共卡蹭空显存，但优先用空卡。
- **不抢**：GPU2（70B-KV4 链显存紧）；GPU0/1/3 的 32B 任务重启后若显存余量 >30GB 也可共卡。
- 单点成本（ppl+层误差，无下游）：3B ~10-15min、8B ~25-35min。198 逻辑配置去重后，6 lane 并行 ≈ **<1 天**收官。

## 4. 执行顺序（= 任务书 §9，加落地细节）

1. **inventory.md**（开跑前返回）：metricsweep worktree 的 commit/dirty、GPU 实况、模型 revision（/home/luofan/models/Llama-3.2-3B、hf_cache 的 Meta-Llama-3-8B 用 resolve_paths 同源）、"原始表示"路径核验（关键陷阱：loader/rotation 是否默认融了 PCA——RND/SS/MA 必须拿到真正未旋转的表示）。
2. 实现 rd_m31.py + 单元自检：无量化等价性（关掉量化后 PPL==浮点基线到容差）、W/X 重排配对、配额 F_l 合法性、Across 的配额约束最大分数选择（任务书 §6 的前缀增量法）。
3. 1B/3B 小样本验证 → **3B seed0 跑 r=0/0.1/0.5/1 四点审查**（两类指标 + 文件格式都对了再放量）。
4. 放量：3B 全 11 点×7 策略配置 → 8B 同 → RND seeds 1/2 → 图 A/B/C。
5. 每个坏点进 failures.csv 继续，不中断矩阵。

## 5. 跟踪表 M31（进度填这里，配合主实验表 A 的习惯）

| 模型 | A: RND/SS/MA/AV ×11 (within) | B: SS/MA/AV ×11 (across) | RND seeds1/2 | 层误差+图 |
|---|---|---|---|---|
| Llama-3.2-3B | ⬜ | ⬜ | ⬜ | ⬜ |
| Llama-3-8B | ⬜ | ⬜ | ⬜ | ⬜ |

验收 gate（每批落点必查，同主实验 sanity 纪律）：r↑→PPL↓ 大体单调（策略差异允许交叉）；r=0/1 端点上 RND/SS/MA 数值一致（同一纯格式 fake-quant，差异超容差=实现 bug，任务书 §9.5）；E_l 无 NaN、ENERGY 近零层单列；不利结果照报。

## 6. 执行方式

由独立 sub-agent 实现并驱动（主 loop 每 tick 顺带 harvest + 查表），结果逐点 commit 到 metricsweep worktree 随主流水 harvest。实验结论回填论文前人工复审（任务书 §9.7）。
