# P0-2 · 离线准备耗时拆解 + 实测权重存储/显存占用

> 任务书：`docs/p0_supplemental_experiments_spec.md` · 任务 P0-2。
> 硬件：RTX PRO 6000 Blackwell Server Edition（SM120，96 GB），ICT90 容器 `luofan-plaquant`。
> 配方：主部署配方 r=1/8 — `--substrate ring --ring_size 512 --alloc measured --high_fraction 0.125`
> + frozen-scale GPTQ + 两级 scale + weight-MSE + act-MSE d3u2（`PROMIX_ACT_MSE_STEPS_UP=2`），
> 与 `gb10_export_point.sh` / `serve_point.sh`（common.sh size_knobs）逐项同款：
> 1B GPTQ n=1024（wmse3）；3B n=512 + percdamp 0.025 + wmse7；8B n=512 + wmse7；70B n=512（E2 up2+g512 配方，CPU offload）。
> 原始数据：`docs/rd_overhead_data/`（计时 JSON、safetensors 字节统计、vLLM 载入显存、70B 日志出处）。

---

## 表 1 · 子项 A：离线端到端耗时拆解（wall-clock，单卡）

| 阶段 | Llama-3.2-1B | Llama-3.2-3B | Llama-3-8B | Llama-3-70B（日志抄录） |
|---|---|---|---|---|
| ① RMSNorm fold + PCA basis (U,R) 拟合（512×2048 wikitext） | **925 s** (15.4 min) | **3,155 s** (52.6 min) | **6,081 s** (101.4 min) | **≈58,000 s (≈16.1 h)**，协方差采集 80 层 16:02:08（~717 s/层），FP32 CPU 常驻 + 逐层上 GPU |
| ② 全局 ΔD 采集 + 排序分配（ring512 基底构建 + 4096 tok 采集 + 分配） | **47 s** | **83 s** | **142 s** | **1,763 s** (29.4 min) |
| ③ frozen-scale GPTQ + 两级 scale + weight/act-MSE（含折叠/旋转 + 模型加载） | **412 s** | **654 s** | **1,286 s** | **10,509 s** (2.92 h) |
| ④ 权重预打包 + checkpoint 导出 | **3.9 s** | **5.9 s** | **12.4 s** | 未导出（插件无 TP，70B 不走 vLLM 导出）；按 8B 打包吞吐折算 GPU 侧 ≈2 min，加 43 GB 落盘估 **≈5–10 min**（估算，未实测） |
| **Total（①+②+③+④）** | **≈23.1 min** | **≈65.0 min** (3,898 s) | **≈125.4 min** (7,521 s) | **≈19.6 h**（①–③实测=70,272 s） |

- 1B/3B/8B 为本次实测（2026-11-27，GPU 0–3，`docs/rd_overhead_data/time_export_{1b,3b,8b}.json`、`time_basis.jsonl`）。计时驱动与 `promix.vllm_integ.export` 主路径逐行同构（见方法学注记 M1）。
- 70B 从既有日志抄录：①`rdscale/70b_basis.log`；②③`rdscale/70b_up2.log`（shipping E2 配方 `70b_ring512_measured_up2_g512`，r=0.125；交叉核对 `70b_ring512_measured.log` 1795 s / `densesweep/logs/acc_70b_r0.10.log` 10,520 s，同量级）。全部 70B 阶段均为 **CPU 常驻权重 + 逐层 GPU 流式（PROMIX_LOAD_DEVICE=cpu）单卡 offload**。出处与原文行见 `rd_overhead_data/p02_70b_provenance.json`。
- 一次性成本（不计入离线准备）：CUDA 扩展 JIT 编译 ~430 s（进程级缓存后 0 s）；本表也不含 T2 gate 的参考 PPL 评测与 `t2_reference.pt` 快照（质量门控产物，非部署必需）。

**对比参考（llm-compressor，同机，wikitext 256×2048 校准；`rd_overhead_data/baseline_quant_times.json`）**

| 方法 | 1B | 3B | 8B | 70B |
|---|---|---|---|---|
| GPTQ W4A16 (g128) | 108 s | 196 s | 457 s | — |
| NVFP4 W4A4 | 80 s | 139 s | 206 s | 1,537 s |
| AWQ W4A16 | 460 s | 328 s | 538 s | — |
| FP8 W8A8 (dynamic) | 20 s | — | — | — |

解读：AdaMix 离线准备在 ②+③+④（给定基底后的量化+导出）与 GPTQ 同量级（3B 12.4 min vs 3.3 min；8B 24 min vs 7.6 min）；大头是一次性的 PCA basis 拟合（①），其成本按模型只需支付一次，可横跨所有 r 预算与 ring_size 复用（本仓库全部 r-sweep/消融共用同一份 U/R 工件）。

---

## 表 2 · 子项 B：权重存储与 GPU 显存（Llama-3.2-3B / Llama-3-8B）

字节统计直接来自各 checkpoint 的 safetensors header（`rd_overhead_data/bytes_*.json`）；GB = 1e9 B，GiB = 2^30 B。
“量化线性层权重+scale” = 全部 decoder 线性层（q/k/v/o/gate/up/down）的权重 + 全部 scale 张量（单副本 fused pack；AdaMix 另有 column_order gather 索引，单列注明）。

### 3B

| 方法 | 线性层权重+scale (GB) | 实测 bits/weight | 磁盘 ckpt (GB, safetensors) | vLLM 运行时权重 (GiB) |
|---|---|---|---|---|
| BF16 | 5.637 | 16.00 | 6.43 | 6.04 |
| FP8 W8A8 (llmc FP8_DYNAMIC) | 2.820 | 8.00 | 3.61 | 3.45 |
| GPTQ-Marlin W4A16 (g128) | 1.453 | 4.125 | 2.24 | 2.14 |
| NVFP4 W4A4 | 1.585 | 4.50 | 2.37 | 2.27 |
| **AdaMix r=1/8（单副本）** | **1.763** | **5.005** | **3.34** | **3.18**（`PROMIX_DECODE_GEMV=0`） |
| AdaMix r=1/8（默认，含 decode 副本） | 同上（运行时派生，不占磁盘） | — | 同上 | **5.48** = 3.18 + GEMV sf8 副本 0.63 + Marlin lo/hi 副本 1.67 |

### 8B

| 方法 | 线性层权重+scale (GB) | 实测 bits/weight | 磁盘 ckpt (GB, safetensors) | vLLM 运行时权重 (GiB) |
|---|---|---|---|---|
| BF16 | 13.959 | 16.00 | 16.06 | 14.96 |
| FP8 W8A8 | 6.982 | 8.00 | 9.08 | 8.46 |
| GPTQ-Marlin W4A16 (g128) | 3.599 | 4.125 | 5.70 | 5.32 |
| NVFP4 W4A4 | 3.926 | 4.50 | 6.03 | 5.62 |
| **AdaMix r=1/8（单副本）** | **4.398** | **5.041** | **6.50** | **6.09**（`PROMIX_DECODE_GEMV=0`） |
| AdaMix r=1/8（默认，含 decode 副本） | 同上 | — | 同上 | **11.94** = 6.09 + GEMV sf8 副本 1.56 + Marlin lo/hi 副本 4.29 |

### 70B（可选项，部分实测）

| 方法 | 线性层权重+scale (GB) | 磁盘 ckpt (GB) |
|---|---|---|
| BF16 | 136.90 | 141.11 |
| NVFP4 W4A4 | 38.50 (4.50 bit) | 42.71 |
| AdaMix r=1/8（估算，未导出） | ≈43.1（按 4.5+3.75·r̄，r̄≈0.14 取 8B 口径折算） | — |

**压缩收益（单副本线性层字节）**：AdaMix vs BF16 −68.7% (3B) / −68.5% (8B)；vs FP8 −37.5% (3B) / −37.0% (8B) —— 与论文 “相对 FP8 省 ~38% / 相对 BF16 省 ~69%” 的口径一致。

### 公式对账（验收门控）

单副本字节数 vs 比特公式（NVFP4 段 4+8/16=4.5 bit/元素，MXFP8 段 8+8/32=8.25 bit/元素，每线性层另 +4 B 全局 scale）：

| 模型 | 实测 bits/weight | r̄（字节加权 FP8 通道占比 ΣN·K_high/ΣN·K） | 公式 4.5+3.75·r̄ | 偏差 | vs 名义 r=1/8 的 4.96875 |
|---|---|---|---|---|---|
| 1B | 5.1021 | 0.1606 | 5.1021 | **+0.000%** | +2.68% |
| 3B | 5.0050 | 0.1347 | 5.0050 | **+0.000%** | +0.73% |
| 8B | 5.0409 | 0.1442 | 5.0409 | **+0.000%** | +1.45% |

- 公式在**实测 r̄** 下逐字节精确成立（exact-bytes 预测偏差 +0.0000%，见 `rd_overhead_data` 校验脚本输出），**验收 <1% 通过**。
- r̄ 高于名义 1/8 的原因是**口径差**而非超预算：全局 ΔD 分配按 **ring 条数**恰好花掉 12.5% 预算（export 日志 `ALLOC ... frac=0.1250`），但它把 FP8 ring 几乎全部从 down_proj（r_eff=0）移到 q/k/v/gate/up（r_eff≈0.18–0.23，o_proj 固定 1/8）；按**字节（N·K 加权）**折算时宽 N 的投影权重更大，故 r̄_bytes=0.135–0.161 > 0.125。序列化字节与 4.5+3.75r 公式本身零偏差。
- AdaMix 另含 column_order gather 索引（int32）：3B 0.34 MB / 8B 0.52 MB，可忽略。

---

## 方法学注记

- **M1 计时驱动**：`p02_export_timed.py`（远端 `/home/luofan/p02_tmp/`，与 `promix/vllm_integ/export.py` 的 measured 路径逐行同构：`rd_ron.resolve_paths` → `rv.build_base`+`rv.collect_stats`+`rv.measured_alloc`+fused-group unify → `rv.build_eval` → `pack_all_targets_shared_g`+`export_checkpoint`），每阶段 `time.time()` 包夹并 `torch.cuda.synchronize()`。跳过 per-proj/shared-g 参考 PPL 评测与 T2 快照（gate 产物）。阶段②含 tokenizer+calib 装载（1B 16 s/3B 16 s/8B 16 s，已计入）。
- **M2 basis 计时**：`promix.quantize.basis --nsamples 512`，输出到独立目录（不覆盖生产工件），FP32 模型 CPU 常驻、逐层上 GPU（70B 同机制，即其 offload 方式）。拆分：协方差采集 1B 890 s / 3B 3,118 s / 8B 6,032 s（tqdm 实测），特征分解各 <5 s，其余为 FP32 模型装载与落盘。
- **M3 字节统计**：`p02_bytes.py` 解析 safetensors header 的 `data_offsets`（零加载、零舍入）。磁盘列取 safetensors 之和（70B BF16 目录含 ~20 GB 残留 `.incomplete` 下载缓存，已剔除）。
- **M4 embed/lm_head**：各量化 ckpt 的 embed_tokens/lm_head 均未量化。AdaMix 导出为 FP16 且强制 untie（rotation 融合需要，1B/3B 原模型 tie），故 3B 多存一份 0.79 GB lm_head；BF16/FP8/GPTQ/NVFP4 的 3B ckpt 保持 tie（单表）。8B 原生 untied，各方法同口径。
- **M5 vLLM 运行时权重显存**：vLLM 0.32（baselines venv + promix 插件 entry-point），`LLM(..., enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.90)`，取 worker 日志 `Model loading took X GiB`（在 `process_weights_after_loading` 之后测量，故包含 AdaMix 运行时派生的 decode-GEMV/Marlin 副本）。单副本档 = `PROMIX_DECODE_GEMV=0`（同时关闭 GEMV sf8 与 Marlin lo/hi 副本，恢复纯 fused-GEMM 路径）；分解档 `PROMIX_MARLIN=0`（GEMV on / Marlin off）用于把副本代价拆成 GEMV 与 Marlin 两项。交叉核对：单副本档 GiB 与 safetensors 线性层+embed/lm_head 字节之和一致（3B 3.18 vs 3.11，8B 6.09 vs 6.05，差为 vLLM 分配器余量）。
- **M6 70B 来源**：全部抄录自既有日志（非本次实测），行号与原文见 `rd_overhead_data/p02_70b_provenance.json`；70B 阶段④未发生过（70B 只做 fake-quant/offload 评测，不出 vLLM ckpt），表中给出明确标注的估算值。
- **M7 对比参考耗时**：llm-compressor 日志 `baselines/logs/quant_*.log` 的 `DONE ... elapsed` 行与 `rdscale/70b_nvfp4_quant.log` 的 `saved ... secs 1537`，设备同机单卡。
