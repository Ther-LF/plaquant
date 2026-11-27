# Motivation §3.1 补充实验任务书（M31-ext）：格式感知跨层误差准则与低预算加密网格

- **日期：** 2026-11-27
- **对应论文：** AdaMix（仓库 `promix/`），Section 3.1 Motivation（Fig. 3）
- **前置依赖：** 已完成的 M31 实验（`docs/motivation_3_1/`，任务书见 `docs/paper_motivation_3_1_experiments.md`）
- **目标读者：** 实验执行者 / 实验 Agent（可直接按本任务书复用 `rd_m31.py` 扩展运行）
- **隔离原则：** 所有新产物写入独立目录 `docs/motivation_3_1_ext/`，**严禁覆盖或修改** `docs/motivation_3_1/` 中既有文件。

---

## 1. 为什么要补做 M31-ext（动机与边界）

既有 M31 在原始（未旋转）通道表示上已证明两件事：
1. **层内与跨层/跨投影误差极度非均匀**：在 $r=0\to 1$ 全区间，`down_proj / up_proj / v_proj` 的输出相对误差 $E_\ell$ 持续远高于 `q_proj / k_proj`（甚至 `down_proj` 在 $r=0.8$ 时仍高于 `q/k_proj` 在 $r=0.0$ 的误差）；
2. **未校准跨层启发式会灾难性崩坏**：原始 $L_1$ 幅值均值 `MA (across)` 在 $r=0.1\text{–}0.6$ 饿死 `layers.1.mlp.down_proj`（分配 0 通道），导致 3B WikiText-2 PPL 飙升至 18.15（劣于纯 NVFP4 的 14.33）；而 $L_2$ 平方均值 `SS (across)` 虽避免了崩溃（8.14），但仅与 `within`（8.12）打平且层均 $E_\ell$ 反而略升（`4.97` vs. `4.74 × 10⁻³`）。

然而，**从审稿人视角看，既有 M31 存在两个缺口**：
1. **缺少“同量纲、感知目标格式误差”的跨层正向对照**：M31 只测了与格式无关的激活幅值启发式（`MA`、`SS`），没有在**同一原始表示、同一通道粒度**下检验“直接按 NVFP4 vs. MXFP8 真实量化误差降幅（`DD`）做跨层分配”是否能真正避免崩溃、压低全局量化误差，并在低预算下优于固定每层分配；
2. **超低预算悬崖区采样过粗**：在原始表示 3B 上，$r=0.0$（14.33 PPL）直接跳到 $r=0.1$（8.12 PPL），跳过了 **6.2 PPL 的陡峭悬崖区**（到 $r=0.1$ 时 `within` 距纯 MXFP8 的 7.85 只剩 0.27 PPL 空间）。同时，若要补密低预算点，**所有被对比的策略必须补齐同一套 $r$ 网格**，保证横轴数据量与预算完全对齐。

### 严格反循环边界（Anti-Circularity）
本补充实验属于 **Section 3 Motivation（理想化/格式感知对照研究）**，**绝不引入 Section 4 AdaMix 方法组件**：
- **不使用** PCA 特征基 $U$、**不使用**共享通道组（ring512/64）、**不使用**组内正交/Hadamard 旋转、**不使用** AdaMix 融合 kernel；
- 全部在 M31 相同的**原始未旋转通道表示**上运行，仅改变“通道打分准则”与“$r$ 采样密度”。

---

## 2. 冻结协议（必须与 `docs/motivation_3_1/manifest.json` 逐项一致）

为保证与既有 `docs/motivation_3_1/` 无缝拼接，以下条件**严格锁定，禁止改动**：

| 项目 | 固定设置（与 M31 一致） |
|---|---|
| 主模型 | `Llama-3.2-3B`（P0 必跑）、`Llama-3-8B`（P1 必跑）；`Llama-3.2-1B`（P2 可选，因 1B headroom 更大） |
| 候选投影（eligible） | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `o_proj` / `lm_head` / KV | `o_proj` 保持 FP16 且排除出分配集合；`lm_head` FP16；KV16 |
| 表示 | 原始未旋转表示（`representation = raw`，同 M31 共同浮点预处理，无 PCA、无随机旋转） |
| 量化格式 | 主格式 `NVFP4 (blk16, two-level E4M3, act-MSE search steps=3 fast)`；高精格式 `MXFP8 (blk32, E8M0, weight-A2)` |
| GPTQ 配方 | `nsamples=128`, `percdamp=0.01`, `w_clip=true`, `act_order=false`, `PROMIX_ACT_MSE=1`, `PROMIX_WEIGHT_MSE=1`, `PROMIX_WEIGHT_2LEVEL=1` |
| 校准与打分集 | WikiText-2 train `512×2048 seed0`；通道打分使用前 **2 个窗口（4096 tokens）**，所有投影严格取相同非 padding token 位置 |
| 局部误差诊断集 | `32×2048` held-out WikiText-2 train 窗口（`seed=31001`，与 512 fit 窗口零重叠，同 M31） |
| 评估集 | WikiText-2 test（full, seqlen=2048）与 C4 val（256 窗口, seed=0） |
| 打包与可行配额 | `F_l = {h : 0 <= h <= d_l, h % 32 == 0, (d_l - h) % 16 == 0}`；分区内按稳定原始通道 ID 升序打包 `[unselected asc | selected asc]`，打包后重算真实 block scale |

---

## 3. 新增打分准则定义：`DD`（Format-Aware Quantization Error Reduction）

在既有原始表示的 `RND`、`MA`、`SS` 之外，新增一个**格式感知、跨层同量纲可比**的通道打分准则 `DD`：

### 3.1 主准则：`DD`（逐通道实测 NVFP4 − MXFP8 激活量化 MSE 降幅）
对每个 eligible 线性层 $\ell$，在相同的 fit 激活张量 $X_\ell \in \mathbb{R}^{N \times d_\ell}$（$N=4096$ tokens，原始未旋转坐标）上，分别调用部署同款的 `fake_quantize_nvfp4`（block-16）与 `fake_quantize_mxfp8`（block-32）各一次，得到 $\widehat{X}_\ell^{(\mathrm{fp4})}$ 与 $\widehat{X}_\ell^{(\mathrm{fp8})}$。定义第 $j$ 个通道的格式误差降幅分数：

$$s_{\ell, j}^{\mathrm{DD}} = \frac{1}{N} \sum_{t=1}^{N} \Bigl[ \bigl(X_{\ell,tj} - \widehat{X}_{\ell,tj}^{(\mathrm{fp4})}\bigr)^2 - \bigl(X_{\ell,tj} - \widehat{X}_{\ell,tj}^{(\mathrm{fp8})}\bigr)^2 \Bigr]$$

- **物理含义与跨层可比性**：所有层、所有通道的 $s_{\ell,j}^{\mathrm{DD}}$ 均具有完全相同的物理单位（每元素平方激活量化误差的绝对减少量），且直接反映 NVFP4（E2M1, blk16）与 MXFP8（E4M3, blk32）在该通道上的真实失真差，而非激活幅值代理。
- **实现注意点**：
  - 使用 `float64` 跨 token 累加 $(X - \widehat{X})^2$，最后除以 $N$；
  - 若极个别通道因块取整产生微小负值（如 `-1e-9`），保存原始有符号分数至 `scores/*.pt`，排序时保留原值（或记录是否出现负值比例，不静默截断，方便诊断）；
  - 与 `SS`/`MA` 保持**完全相同的 32-chunk 前缀贪心跨层求解器**（见 M31 §6）及**完全相同的打包顺序** `[unselected asc | selected asc]`；
  - 选完并打包后，**必须按最终打包布局重算真实 NVFP4/MXFP8 block scale 并运行冻结 scale GPTQ**（严禁把选前逐通道误差直接相加当作最终结果）。

### 3.2 可选诊断变体：`DD_out`（权重范数加权的输出误差代理，P2 可选）
仅作为可选附加对照（不替代主准则 `DD`）：
$$s_{\ell, j}^{\mathrm{DD\_out}} = s_{\ell, j}^{\mathrm{DD}} \cdot \frac{\|W_{\ell, :, j}\|_2^2}{m_\ell}$$
用于检验“将激活量化 MSE 映射为线性层输出 MSE（乘以对应权重列平均平方范数）”是否对跨投影（如 `down_proj` vs `q_proj`）的 PPL 更有帮助。若算力充裕可一并跑 3B 对比；若优先控时长，**优先完成主准则 `DD`**。

---

## 4. 统一加密预算网格与精确运行矩阵（保证所有策略数据量完全对齐）

### 4.1 统一 15 点目标比例网格
将原有 11 点粗网格与 **4 个超低预算加密点** 合并为统一 15 点网格：
$$r_{\text{target}} \in \{0.00,\ \mathbf{0.02,\ 0.04,\ 0.06,\ 0.08},\ 0.10,\ 0.20,\ 0.30,\ 0.40,\ 0.50,\ 0.60,\ 0.70,\ 0.80,\ 0.90,\ 1.00\}$$

**块对齐可行性预检（已核实，4 个新点在各模型上均为互异的 32 倍数，零舍入碰撞）：**
- **Llama-3.2-3B**（`d=3072`, `d_ff=8192`）：
  - `r=0.02` → `h = 64 / 160`（actual ≈ `2.00%`）
  - `r=0.04` → `h = 128 / 320`（actual ≈ `4.00%`）
  - `r=0.06` → `h = 192 / 480`（actual ≈ `6.00%`）
  - `r=0.08` → `h = 256 / 640`（actual ≈ `8.00%`）
- **Llama-3-8B**（`d=4096`, `d_ff=14336`）：
  - `r=0.02` → `h = 96 / 288`
  - `r=0.04` → `h = 160 / 576`
  - `r=0.06` → `h = 256 / 864`
  - `r=0.08` → `h = 320 / 1152`
- **Llama-3.2-1B**（可选，`d=2048`, `d_ff=8192`）：
  - `r=0.02` → `h = 32 / 160`；`r=0.04` → `h = 96 / 320`；`r=0.06` → `h = 128 / 480`；`r=0.08` → `h = 160 / 640`

### 4.2 精确运行清单（最大化复用已有 M31，消除冗余）

> **关键原则**：每个新加入的低预算点 `{0.02, 0.04, 0.06, 0.08}` 上，**所有对比策略必须成套补齐**，使合并后 15 个点上每条曲线都有完全一致的样本数和相同总预算 $H(r)$。

#### Part A — 加密点补齐（4 个新比例：`0.02, 0.04, 0.06, 0.08`）
每个模型需运行以下全部 7 组配置（共 `4 × 9 = 36` 次逻辑配置/模型）：

| 策略 | Scope | Seeds | 点数 | 运行次数/模型 | 说明 |
|---|---|---|---|---|---|
| `RND` | `within` | `0, 1, 2` | 4 | 12 | 复用 M31 同一 `md5` 固定随机优先级排列前缀 |
| `MA` | `within` | `0` | 4 | 4 | 复用 M31 `scores/{3b,8b}_raw.pt` |
| `SS` | `within` | `0` | 4 | 4 | 复用 M31 `scores/{3b,8b}_raw.pt` |
| `MA` | `across` | `0` | 4 | 4 | 复用 M31 `scores/{3b,8b}_raw.pt` |
| `SS` | `across` | `0` | 4 | 4 | 复用 M31 `scores/{3b,8b}_raw.pt` |
| **`DD`** | **`within`** | `0` | 4 | 4 | 新增格式感知准则（层内对照） |
| **`DD`** | **`across`** | `0` | 4 | 4 | **核心新增：格式感知跨层分配** |

#### Part B — 粗网格补齐（9 个内部点：`0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`）
原有 `RND/MA/SS` 在这 9 个点与端点 `0.0/1.0` 已全部存在于 `docs/motivation_3_1/`，**无需重跑**；仅需补跑新增策略 `DD`（共 `9 × 2 = 18` 次逻辑配置/模型）：

| 策略 | Scope | Seeds | 点数 | 运行次数/模型 | 说明 |
|---|---|---|---|---|---|
| **`DD`** | **`within`** | `0` | 9 | 9 | 补齐 0.1–0.9 全曲线 |
| **`DD`** | **`across`** | `0` | 9 | 9 | **补齐 0.1–0.9 全曲线** |
| 端点 `0.0, 1.0` | `within/across` | `0` | 2 | 0（复用+校验） | 验证与 M31 原始端点 `r=0/1` 位级一致后逻辑映射复用 |

**总工作量汇总：**
- **P0（Llama-3.2-3B）**：`36 + 18 = 54` 次运行（单卡 SM120 预计约 15–25 分钟，可优先交付回填 Fig. 3）；
- **P1（Llama-3-8B）**：`36 + 18 = 54` 次运行（用于正文双模型稳健性验证，避免单模型孤证）；
- **P2（可选，Llama-3.2-1B）**：仅当 3B/8B 在低预算 PPL 空间仍很紧、需要检查更大 headroom 模型表现时再启动（否则可跳过）。

---

## 5. 新增核心诊断指标（除原有 PPL 与 $E_\ell$ 之外，务必记录！）

既有 M31 记录了 `wiki_ppl`、`c4_ppl` 与逐层输出误差 `relative_squared_error` ($E_\ell$)。为直接检验跨层同量纲准则在**全局量化失真**上的真实改善，**M31-ext 在 held-out 诊断集（32×2048 窗口，同 teacher 输入 $X_\ell$）上新增记录以下激活端量化指标**：

### 5.1 逐层新增字段（写入 `docs/motivation_3_1_ext/layer_errors.csv`）
在原有 9 列基础上追加 4 列（均用 `float64` 流式累加，按**最终打包后的真实量化路径**测得）：
- `act_SSE`：$\sum_{\text{windows}} \|X_\ell - \widehat{X}_\ell\|_F^2$（按最终打包分区经真实 NVFP4/MXFP8 fake-quant 后的激活平方误差总和）
- `act_ENERGY`：$\sum_{\text{windows}} \|X_\ell\|_F^2$
- `act_MSE`：`act_SSE / (n_tokens * d_l)`（该层每元素平均激活量化 MSE）
- `act_rel_err`：`act_SSE / (act_ENERGY + 1e-12)`（该层激活相对平方误差）

### 5.2 全网汇总字段（写入 `docs/motivation_3_1_ext/runs.csv`）
在原有列基础上追加 3 列：
- `act_mse_global`：$\frac{\sum_\ell \text{act\_SSE}_\ell}{\sum_\ell (n_{\text{tokens},\ell} \cdot d_\ell)}$（**全网总激活量化 MSE**，与论文 Section 4.2 Theorem 1 目标函数严格同口径！）
- `act_sse_total`：$\sum_\ell \text{act\_SSE}_\ell$
- `out_rel_err_weighted`：$\frac{\sum_\ell \text{SSE}_\ell}{\sum_\ell \text{ENERGY}_\ell}$（能量加权全网输出误差，与等权 `error_mean_layers` 互为对照）

### 5.3 既有 M31 粗网格点的 `act_MSE` 轻量回填（无需重跑 GPTQ！）
由于 `act_SSE / act_MSE` **只依赖输入激活 $X_\ell$ 和通道选择 mask（与权重/GPTQ 完全无关！）**，对既有 `docs/motivation_3_1/` 中已跑完的 9 个粗网格点（`MA/SS within & across`，以及 `r=0.0/1.0`）：
- 只需写一个轻量回填脚本（从 `docs/motivation_3_1/allocations.jsonl` + `scores/*_raw.pt` 重建确定性 mask，在 32 个 held-out 诊断窗口上跑一次纯激活 fake-quant 前向，**不跑 GPTQ、不跑 PPL eval**），即可在 **~1–2 分钟内**把全部 15 个点上 `MA/SS/DD` 的 `act_mse_global` 与逐层 `act_MSE` 补齐！
- 回填结果单独存入 `docs/motivation_3_1_ext/act_mse_backfill.csv`（严禁覆盖原目录）。

---

## 6. 验收 Gate 与质量检查（执行完必过）

1. **端点一致性 Gate**：在 `3B` 和 `8B` 上用新 runner 重跑一次 `r=0.0` 和 `r=1.0` 抽检，验证 `wiki_ppl`、`c4_ppl` 和 `error_mean_layers` 与 `docs/motivation_3_1/runs.csv` 误差 $\le 10^{-4}$（确认代码环境零漂移）。
2. **配额精确守恒 Gate**：对每个新比例 $r \in \{0.02, 0.04, 0.06, 0.08\}$ 及所有粗网格点，验证同一模型同一 $r$ 下 `within` 与 `across` 的总 FP8 通道数 $H(r) = \sum_\ell h_\ell$ **精确相等**（`H_across == H_within`）。
3. **无 NaN / 无静默插值 Gate**：所有完成配置 `nan_error_layers == []`；任何失败配置写入 `failures.csv` 并重试。
4. **反作弊与如实报告铁律**：
   - **不预设 `DD (across)` 在 PPL 上一定大幅击败 `within`**；如实汇报三组指标：
     1. 全网激活量化误差 `act_mse_global`（预期 `DD (across)` 相对 `Fixed per-layer`、`MA (across)`、`SS (across)` 全程显著更低）；
     2. 低预算悬崖区 `r = 0.02–0.08` 及全区间 `0.1–0.9` 的 WikiText-2 / C4 PPL；
     3. 各投影类型（`down/up/v/gate/q/k`）及 `layers.1.mlp.down_proj` 在 `MA` / `SS` / `DD` 下的实际 FP8 分配比例（验证 `DD (across)` 是否自动保护关键层且把预算优先分向高误差投影）。

---

## 7. 交付物清单（全部输出至 `docs/motivation_3_1_ext/`）

1. `manifest.json`：记录 commit、模型路径/SHA、`DD` 打分定义、15 点网格、与 `docs/motivation_3_1/manifest.json` 的一致性校验结果；
2. `scores/{3b,8b}_dd_raw.pt`：逐层每通道的 `DD` 分数张量（含符号与统计摘要）；
3. `runs.csv`：全部 54（3B）+ 54（8B）次新运行结果（含新增 `act_mse_global`、`act_sse_total`、`out_rel_err_weighted`）；
4. `layer_errors.csv`：逐层输出误差 + 逐层激活量化误差（含新增 4 列）；
5. `allocations.jsonl`：每层分配的 $h_\ell$、`ids_sha256` 与分数摘要；
6. `act_mse_backfill.csv`：对既有 `docs/motivation_3_1/` 粗网格点轻量回填的激活量化 MSE（不重跑 GPTQ）；
7. `merged_runs.csv` 与 `merged_layer_errors.csv`：经 Gate 1–3 校验后，将 `docs/motivation_3_1/` 与 `docs/motivation_3_1_ext/` 合并成完整 15 点对齐表（供论文 `plot_motivation_figures.py` 直接读取出图）；
8. `report.md`：简明结果汇总表（3B/8B 在 `r=0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.5` 下 `RND / MA / SS / DD` 的 within vs. across 对比：PPL、`act_mse_global`、`error_mean_layers`、以及 `down_proj` / `layers.1.mlp.down_proj` 分配占比）。

---

## 8. 数据回来后，论文 Fig. 3 的直接落图方式

待 `docs/motivation_3_1_ext/merged_runs.csv` 与 `merged_layer_errors.csv` 返回后，论文 **Fig. 3** 将使用完全对齐的 15 点网格（$r \in [0, 1]$，含 `0.02–0.08` 悬崖区）直接绘制：
- **Panel (a) Within-layer selection ($r = 0 \to 1$)**：`Targeted` vs. `Random` 在含 `0.02–0.08` 悬崖区的 PPL 对比；
- **Panel (b) Cross-projection error ($r = 0 \to 1$)**：6 类投影的 $E_\ell(r)$ 曲线（清晰呈现 `down/up/v` 全程远高于 `q/k`）；
- **Panel (c) Cross-layer criterion ($r = 0 \to 1$)**：同图对比 `Fixed per-layer` vs. `Across: L1 abs` vs. `Across: L2 energy` vs. **`Across: Format-error (DD)`**（按实测结果展示其对全局量化误差 `act_mse_global` 与/或低预算 PPL 的正向改善，彻底闭环“跨层同量纲 + 感知目标格式误差”的核心论点）。
