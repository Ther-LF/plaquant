# AdaMix (ProMix) — P0 论文补测实验 Spec（ACM TACO 审稿防线）

> **目的**：补齐 ACM TACO 审稿人几乎必问的 3 项高性价比硬证据（预计总卡时 1–2 小时），并顺带核对 2 项零卡时的数据口径。完成后即可直接更新论文 §5/§6 并收口定稿。

---

## 任务 P0-1：DGX Spark GB10（SM121）端到端 Serving 同机基线对照组

### 为什么必补
目前 `docs/vllm_final_data/gb10*` 只有 AdaMix 自己的 7 点 $r$-sweep，缺少同机基线锚点。审稿人必然追问：*“在 PRO 6000 上你们跟 FP8 吞吐打平，那什么时候该用 AdaMix 而不是 FP8？”* GB10 的统一 LPDDR 带宽受限环境正好是 ~5 bit 相对 8-bit FP8 / 16-bit BF16 体现带宽与显存红利的绝佳场景。

### 实验配置
- **硬件与软件栈**：DGX Spark GB10（SM121），vLLM 0.33（与已有 `gb10v2_pm3b_*` 完全同栈：CUDA Graphs + `torch.compile` ON，prefix caching OFF，greedy + `ignore_eos`，1 warmup + median of 3）。
- **同机状态强约束**：**必须在同一次开机状态内连续跑完**（不要跨重启），同时重跑 1 个 `AdaMix (r=0.125)` 锚点与旧 `gb10v2_pm3b_r0.125_*` 对齐，消除重启状态漂移（文档已记录跨重启 decode 漂移可达 20–64%）。
- **模型**：
  - **必跑**：`Llama-3.2-3B`
  - **选跑（强烈推荐，若显存与时间允许）**：`Llama-3-8B`（8B 在 LPDDR 带宽下对 BF16/FP8 的差距会更炸裂！）
- **对比方法（4 个基线 + 1 个同会话 AdaMix 锚点）**：
  1. `BF16`
  2. `FP8 W8A8`（`llm-compressor FP8_DYNAMIC`，与 PRO 6000 同 checkpoint/同配方）
  3. `GPTQ-Marlin W4A16`（与 PRO 6000 同 checkpoint）
  4. `NVFP4 W4A4`（FlashInfer/CUTLASS SM121，若 SM121 vLLM 0.33 不支持则如实记录具体报错作为 runnability 证据）
  5. `AdaMix W4A4 (r=0.125, ring512)`（同会话锚点；如能顺带给 `r=0.0` 和 `r=1.0` 更佳）
- **测试负载（与表 D-GB10 完全一致）**：
  - `main`：prefill-heavy `1024/128`、decode-heavy `128/1024`（batch `1, 16, 32, 64, 128, 256`）、prefill-only `L2048`（batch `1, 16, 32, 64, 128`）
  - `qserve`：连续批处理 `1024/512`（concurrency `64, 128, 256`）

### 交付物与验收门控
- **原始文件**：保存到 `docs/vllm_final_data/gb10v3_{bf,fp8,gptq,nv,pm_r0.125}{3b,8b}_{main,qserve}.json`（含开机时间戳/boot id 或同 session 记录）。
- **汇总表**：更新/补充 `Table D-GB10`（含各方法的 decode `b16/b128` TPOT、prefill `L2048 b128` TTFT、`qserve c128/c256` tok/s，以及相对 BF16 / FP8 的加速比）。
- **验收**：同会话 `pm_r0.125` 与已过门控的 T2 PPL 保持一致，无 OOM / NaN / 跨重启混拼。

---

## 任务 P0-2：离线准备耗时（Calibration / Export Time）与实测显存占用表

### 为什么必补
审稿人对带 PCA + 全局 $\Delta D$ + GPTQ + MSE scale 搜索的 PTQ 系统 100% 会问两个问题：（1）离线准备 3B/8B/70B 到底要多久（分钟级还是小时级）？（2）~4.97 bit 实际落地后比 BF16 / FP8 省了多少权重显存，以及 decode-GEMV 副本占了多少额外空间？

### 实验配置（RTX PRO 6000 SM120）
- **模型**：`Llama-3.2-1B`、`Llama-3.2-3B`、`Llama-3-8B`、`Llama-3-70B`（固定 `r=1/8`，`ring512`，主部署配方）。
- **子项 A · 离线端到端耗时拆解（wall-clock 秒/分钟，单卡；70B 标注 offload 方式）**：
  1. `RMSNorm fold + PCA basis (U, R)` 拟合耗时
  2. `全局 ΔD 采集 + 排序分配` 耗时
  3. `Frozen-block-scale GPTQ + 两级 scale + weight-MSE` 量化耗时
  4. `权重预打包与 Checkpoint 导出` 耗时
  5. **总计（Total）**，并附对比参考（如 GPTQ W4A16 或 NVFP4 的校准耗时，如有日志直接抄录即可）
- **子项 B · 权重存储与 GPU 显存占用（3B & 8B，可选 70B）**：
  - 统计并对比 `BF16`、`FP8 W8A8`、`GPTQ-Marlin W4A16`、`NVFP4 W4A4`、`AdaMix (r=1/8)`：
    1. **量化线性层权重 + scale 纯字节数（单副本 fused pack，单位 GB）**——验证 `4.97 bit` 理论压缩比（相对 FP8 省 ~38%，相对 BF16 省 ~69%）
    2. **整模型磁盘 checkpoint 大小（GB）**（标明 `embed/lm_head` 为 FP16）
    3. **vLLM 运行时模型权重显存（GB）**：分别给出 **单副本（关闭低批 GEMV 副本，或纯 fused 权重大小）** 与 **默认含低批 decode-GEMV/Marlin 副本** 的对比（说清空间换时间的精确代价，单位 GB）。

### 交付物与验收门控
- **输出文件**：`docs/rd_overhead_memory.md`（附原始计时日志/统计 JSON）。
- **验收**：单副本线性层权重字节数与 `4.5 + 3.75r` 公式误差 `< 1%`；70B 注明是否使用 CPU offload。

---

## 任务 P0-3：组宽度 $b$（`ring_size`）敏感度消融

### 为什么必补
论文精度扫用 $b=64$、实际部署用 $b=512$，审稿人必然问：*“为什么选 $b=512$？从 32/64 放大到 512 损失了多少精度、换来了多少算子/系统收益？”* 目前缺少控制变量的单轴扫描表。

### 实验配置（RTX PRO 6000 SM120）
- **固定变量**：模型 `Llama-3.2-3B` 和 `Llama-3-8B`（两者均能被 512 整除），固定全局预算 **`r = 1/8`（`0.125`）**、完整 shipped 精度配方（PCA + 组内 Sylvester--Hadamard + 全局实测 $\Delta D$ + GPTQ + 两级 scale + `act-MSE d3u2`，KV16）。
- **扫描变量**：`ring_size` $b \in \{32, 64, 128, 256, 512\}$（5 个点，均为 $\operatorname{lcm}(16, 32)=32$ 的 2 的幂倍数）。
- **测量指标**：
  1. **精度**：WikiText-2 PPL、C4 PPL（与 `rd_buildup_ablation.md` 完全同协议；`b=512` 应逐位复现已有锚点 `3B: 8.2180 / 11.2549`，`8B: 6.5642 / 9.5639`）。
  2. **分配与系统代价**：
     - 各 $b$ 下非单调层数（需要在线 group gather $\Pi_\ell \neq I$ 的层占比）
     - 代表形状（如 8B `gate_up` & `down`, `M=8192`）的 `qpfm` pack/gather 延迟与融合 GEMM 延迟（如有 `b < 128` 触及 128-pad 惩罚则直接量化体现，正好解释为何选 `b=512`！）

### 交付物与验收门控
- **输出文件**：`docs/rd_ringsize_ablation.md` + 原始 JSON（`docs/rd_ringsize_data/`）。
- **验收**：`b=512` 行逐位复现 `rd_buildup_ablation.md` 锚点（误差 `< 0.001`）；5 个点的实测 `avg_fp8` 一并记录（验证边界 guard 引起的实际 $r$ 偏移）。

---

## 附：2 项零卡时核对（直接查已有文件，5 分钟搞定）

1. **核对已有 ShareGPT 数据（`docs/vllm_final_data/sgpt_*.json`，16 份文件已在盘里）**
   - 确认 `sgpt_{bf,fp8,gptq,awq,nv,pm,pmr0.0625,pmr0.25}{3b,8b}.json` 是否为与 `vllm_final_fiveway.md` 同协议的干净实测数据，并给出一张 3B/8B 汇总表（请求吞吐 / TTFT / TPOT）。确认干净后我直接写入论文 §6.4，补齐真实变长请求流量证据。
2. **对齐 `experiment_results_report.md` §3 的 TPOT 增幅口径**
   - 总报告 §3 写 Pro6000 `r=0→1` TPOT 增幅为 `1B +3.7% / 3B +1.5% / 8B +4.5%`，而 `dashboard_data.json` 中 `b256` 对应 `r=0→1` 为 `1B: 7.27→7.56 (+3.9%)`、`3B: 18.89→19.41 (+2.8%)`、`8B: 24.85→27.19 (+9.4%)`。
   - 请确认 `+1.5% / +4.5%` 具体对应的 batch（例如 `b128` 还是 `r=0.03125→0.25`），论文将统一采用明确标注 batch 的精确值。

---

### 顺手选做（P1，有空闲卡再跑，不阻塞投稿）
- `Qwen3-8B` 补跑 `FP8 W8A8` + `GPTQ W4A16` 两个参考锚 + `Qwen3-32B` 补跑 `r=1/8` 单点 MMLU（填满 Table 1 右半区）。
- `Llama-3-8B` 在 4K / 8K 上下文窗口下测 1 组 `KV16 vs KV4` WikiText-2/PG-19 PPL（验证长上下文 KV4 无累积崩塌）。
