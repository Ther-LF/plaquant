# M31 Inventory — Motivation §3.1 通道级精度分配实验

日期: 2026-11-24 ｜ 任务书: `docs/paper_motivation_3_1_experiments.md` ｜ 对齐: `docs/motivation_3_1/alignment.md`

## 1. 代码与环境

- Worktree: `/home/luofan/plaquant-metricsweep` (branch `feat/rd-metricsweep`), HEAD `ce24b42`。
- Dirty 文件（其他 agent 的，M31 不动、不覆盖）:
  - `promix/quantize/quant_utils.py`: `_ceil_to_value` 分块化（bit-identical OOM fix）
  - `promix/scripts/metricsweep.sh`: 增加 ppl/full 模式
  - untracked: `promix/configs/qwen3-*-kv4.yaml` 等
- Python: `/home/luofan/baselines/venv/bin/python` — torch 2.13.0+cu130, transformers 5.17.0, lm_eval 可用。
- GPU 实况（开工时）: GPU0/1/3/4/5/6/7 空闲, GPU2 占用 47GB（70B 链）。M31 只用 **GPU4-7**。
- MASTER_PORT: 每 lane 29560+lane（见 m31sweep.sh），torch.distributed nccl 单 rank（fuse_basis 的 barrier 需要）。
- HF: `HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache`（wikitext/c4 已缓存，metricsweep 同源）。

## 2. 模型

| key | 路径 | hidden | intermediate | layers | 备注 |
|---|---|---|---|---|---|
| 3b | /home/luofan/models/Llama-3.2-3B | 3072 | 8192 | 28 | 本地 snapshot（与主实验同源, rd_ron.resolve_paths） |
| 8b | /home/luofan/models/llama-3-8b | 4096 | 14336 | 32 | 同上 |

tokenizer 取自各模型目录（同一 revision）。tie_word_embeddings 由 loader 解开并 clone lm_head（float 等价变换，所有配置一致）。

## 3. "原始表示"路径核验（任务书 §3.3 关键陷阱）

已逐文件核验，**不存在隐式 PCA/旋转加载**：

- `promix/models/loader.py::load_model`: 仅 untie embeddings + `.cuda()`，无任何旋转。
- PCA/旋转只经显式调用 `fuse_basis_to_model`（内含 `rotate_*`）和 `rearrange_columns` 进入；`setup_down_proj_hadamard`（down_proj 在线 Hadamard）也是显式调用。
- 因此 **原始表示 = `load_model` + `fuse_layer_norms`**（RMSNorm scale 融进相邻线性层的 float 等价变换，披露为共同浮点预处理），其余一概不加。RND/SS/MA 在此表示上收集分数并直接 fake-quant 原始通道（选中通道经列置换打包到 [main|high] 段，置换同时作用于激活 (`_column_order` + `reorder_before_quant`) 与权重列 — 数学等价，自检覆盖）。

## 4. AV（PCA）表示路径

- 复用已验证构件: `fuse_basis_to_model(identity_main_rotation=True)`（T = U_attn 纯 PCA，**关闭 R1**，无 §3.2 的组内随机旋转）+ `rearrange_columns`(o_proj per_head) + `setup_down_proj_hadamard` + `install_column_order_hooks`。
- Basis 构造（`promix/quantize/basis.py`）如实记录: 累计的是 **未中心化 second moment X^T X**（fp64, 对角 1% damping, `eigh` 升序排序）— 不是中心化 covariance。按任务书 §4，AV 的主分数不用特征值，而是在该表示 Z=XU 上实测的**中心化方差**（float64 累计, 负值夹零）。
- down_proj 例外（披露）: intermediate 维无 PCA 基（ResQ 本身对 down_proj 用固定在线 Hadamard）。AV 的 down_proj 表示 = 在线 Hadamard 后坐标，分数 = 该坐标下实测中心化方差。manifest/图注记录。
- 复用 artifacts（与主实验同一文件，冻结）:
  - `rotation/U-wikitext-512-Llama-3.2-3B.bin` (2026-10-12), `rotation/U-wikitext-512-llama-3-8b.bin`
  - `rotation/R-high-0.125-low-0.0-sparse-0.0-{Llama-3.2-3B,llama-3-8b}.bin`（identity_main 下 R1 不用；R2 仅进 v 输出/o 输入 — o_proj 不在 eligible 集合，不影响被测投影）

## 5. Eligible 集合与统一配置

- Eligible（分配 + 量化 + 层误差）: 每 block 的 `q/k/v/gate/up/down_proj`（3B: 28×6=168, 8B: 32×6=192 个线性层）。
- **o_proj 排除并在所有配置固定 FP16（权重+激活）**。原因: AV 表示下 o_proj 输入须走独立 per-head PCA 路径，原始表示下无对应物 — 任一量化方式都会让各策略的 o_proj 表示不同，污染对照（任务书 §3.3 允许排除并记录）。lm_head/embedding/norm FP16，KV16。注意：因此本实验绝对 PPL 不可与主实验曲线(块1)直接比较，只做 M31 内部对照。
- q/k/v（及 gate/up）共享输入但 **mask 独立**（fake-quant 分开执行，激活/权重列严格配对）；SS/MA/AV 下共享输入自然给出相同 mask，RND 每投影独立优先级（记录于 manifest）。
- 可行配额 `F_l = {h: h%32==0, (d-h)%16==0, 0<=h<=d}`。所有 d ∈ {3072, 8192, 4096, 14336} 均为 32 的倍数 → F_l = {0,32,...,d}，非空；无 padding。
- 纯端点: h=0 → 纯 NVFP4 单格式分支; h=d → 纯 MXFP8 单格式分支（激活）；权重端 GPTQ 对应整段单 quantizer（空段有 guard，已核验 `_weight_two_level_nvfp4_eff` 的 numel==0 分支）。

## 6. 量化配方（冻结 = 主实验 metricsweep 曲线协议）

- 真格式 fake-quant: `fake_quantize_nvfp4`(blk16, E4M3 scale 两级, ACT MSE search steps=3 fast) / `fake_quantize_mxfp8`(blk32, E8M0)。精度分区变更后 scale 按最终打包布局在 forward/GPTQ-freeze 时重算（不是预量化误差相加）。**不改每通道 scale**，不用 ring。
- GPTQ: wikitext-512-seed0 序列前 128 窗（`calibration.nsamples=128`, 与 512 窗 fit 序列同 seed 前缀 — 子集映射明确）, percdamp=0.01, w_clip(mse)=true, act_order=false, frozen block scales（spec §11 路径）。
- 环境 knobs 冻结为默认: PROMIX_ACT_MSE=1(steps 3, fast), PROMIX_WEIGHT_MSE=1(steps 3), PROMIX_WEIGHT_2LEVEL=1 — 与 metricsweep 主曲线相同（**不是** rd_acctune 的 STEPS_UP=2/WEIGHT 7 变体；主 sweep 实际没设这些 env）。所有策略/scope/r 同配方，不逐策略调优。
- per-Linear K_high 注入复用已验证机制: `module._edriven_khigh` + GPTQ.__init__ patch（ablation_acc_per_bit/rd_validity 同款），新增 `_m31_skip`（o_proj 权重不量化）。

## 7. 数据协议

- Fit/选择: wikitext-2 train, random.seed(0) 的 512 窗序列（与主实验一致）。分数收集用该序列**前 2 窗 = 4096 tokens**（无 padding, 所有投影同 token 位置，单次前向同时喂所有 hook → 上限一致）。窗口偏移记录在 scores 文件 meta。
- Held-out 诊断集: 32 窗 × 2048，random.seed(31001) 抽取并**拒绝与 512 fit 窗区间重叠**；固定，不参与任何调参。
- PPL: wiki test 全量（~141 窗×2048, 等长窗 → evaluator 的窗均 NLL 均值 == 总 token NLL 均值，数学等价于任务书 §7.1 的累计式）；C4 validation 256 窗 seed0（现有协议）。记录 NLL、token 数、input_ids sha256。
- Random: assignment_seed ∈ {0,1,2} 仅改通道优先级（每模型×投影×seed 一个固定 randperm，所有 r 用同一排列前缀）。

## 8. 需要新建（已随本 commit 实现）

- `promix/scripts/rd_m31.py`: collect / selfcheck / control / run / aggregate 子命令（分数收集、F_l 配额、within/across 前缀增量配额法、打包、GPTQ、PPL、teacher 层误差 hook、幂等+物理去重）。
- `promix/scripts/m31sweep.sh`: lane launcher（幂等、坏点进 failures.csv 不中断、逐点 flock+commit）。
- `docs/motivation_3_1/{data,scores}/` 输出目录；manifest.json / runs.csv / layer_errors.csv / allocations.jsonl / failures.csv / plot.py / report.md 按任务书 §8。
- 已知偏差（report.md 同步记录）: allocations.jsonl 存 per-module h + 选择规则 + ids sha256 + score_summary；完整 stable_selected_channel_ids 不逐 run 落盘（8B 全矩阵约数百 MB），由已提交的每通道分数张量 (`scores/*.pt`) + 确定性选择规则可精确重建（`rd_m31.py alloc --dump` 可导出任意 run 的完整 ids 并校验 sha）。

## 9. 不触碰

- `docs/rd_metricsweep_data/`（主实验数据）、其他 agent 的 dirty 文件、GPU2 与 GPU0/1/3 任务。
