# P0 补测实验结果总结（ACM TACO 审稿防线）

日期：2026-11-28 ｜ spec：`docs/p0_supplemental_experiments_spec.md` ｜ 状态：**四项全部收官**
原始数据：`docs/vllm_final_data/gb10v3_*` · `docs/rd_overhead_memory.md`(+`rd_overhead_data/`) · `docs/rd_ringsize_ablation.md`(+`rd_ringsize_data/`)

---

## P0-1 · GB10 同机基线对照（表 D-GB10d，已写入 `paper_experiments_gb10_serving.md` §8）

**执行质量**：boot_id de09c9e5 单会话 4h 连续（无跨重启混拼）；AdaMix 锚点与 gb10v2 旧点 main 15 指标 ±1.2%、gate dPPL 逐位；checkpoint 严格同源（FP8 data-free 复刻 450 tensor **sha256 与 Pro6000 逐字节一致**，GPTQ/NVFP4 直传 sha 一致）。

### 结果（Llama-3.2-3B，decode TPOT b16/b128 ms · TTFT L2048 s · qserve c128/c256 tok/s）

| 方法 | TPOT b16 | TPOT b128 | TTFT L2048 | qserve c128 | qserve c256 |
|---|---|---|---|---|---|
| BF16 | 34.0 | 72.2 | 21.18 | 1983 | 1368* |
| FP8 W8A8 | 21.8 | 58.4 | 24.86 | 3138 | 3249 |
| GPTQ-Marlin W4A16 | 15.7 | 56.3 | 24.87 | 3189 | 3165 |
| NVFP4 W4A4 | 16.8 | 53.1 | 11.30 | 3637 | 3733 |
| **AdaMix r=1/8** | **18.8** | **62.9** | **14.94** | **3482** | **3599** |

*BF16 qserve 非单调塌陷（c64 2274→c256 1368，3 runs ±3% 一致）已注记。

### 审稿人问题的数据回答
1. **"何时该用 AdaMix 而不是 FP8？"** — 带宽受限平台（GB10 统一 LPDDR）上 AdaMix 全负载段胜 FP8：**decode b16 1.16×、qserve c256 1.11×、prefill TTFT 1.66×**（FP8 在线 quant 开销使其 prefill 甚至慢于 BF16）。配合 Pro6000 上与 FP8 打平 → "~5bit 的带宽红利随平台带宽收紧而放大"。
2. **NVFP4 在 SM121 可跑**（FlashInfer CUTLASS kernel），吞吐比 AdaMix 高 4-10%，但 W4A4 精度档不可比（32B wiki 8.00 vs 我们 7.69）→ 论文叙事："同速差精度 / 同精度差速"。
3. 8B 基线未跑（Spark 外网 0.4MB/s，35GB ckpt 传输不可行）— spec 选跑项，不阻塞；等带宽窗口可补。

---

## P0-2 · 离线准备耗时 + 显存占用（`rd_overhead_memory.md`）

### 表1 · 离线端到端耗时（Pro6000 单卡，r=1/8 主配方）
| 阶段 | 1B | 3B | 8B | 70B（日志抄录，offload）|
|---|---|---|---|---|
| ① fold+PCA basis | 925s | 3155s | 6081s | ≈16.1h |
| ② 全局 ΔD 采集+分配 | 47s | 83s | 142s | 29min |
| ③ GPTQ+两级scale+MSE | 412s | 654s | 1286s | 2.9h |
| ④ 打包导出 | 4s | 6s | 12s | 估 5-10min（注明估算）|
| **Total** | **23min** | **65min** | **2.1h** | **≈19.6h** |

对比参考（抄录）：llm-compressor GPTQ 108-457s、NVFP4 80s-26min — 我们贵在 PCA+ΔD+MSE 搜索，一次成本换整条 r 部署曲线。

### 表2 · 存储/显存（GB；vLLM 为 GiB）
| | 线性层权重+scale | 磁盘 ckpt | vLLM 运行时 |
|---|---|---|---|
| 3B BF16 / FP8 / **AdaMix** | 5.64 / 2.82 / **1.76** | 6.43 / 3.61 / 3.34 | 6.04 / 3.45 / **单副本 3.18**（默认含 GEMV+Marlin 副本 5.48）|
| 8B BF16 / FP8 / **AdaMix** | 13.96 / 6.98 / **4.40** | 16.06 / 9.08 / 6.50 | 14.96 / 8.46 / **单副本 6.09**（默认 11.94）|

压缩收益：**vs BF16 −68.7%、vs FP8 −37.5%**（与论文 ~69%/~38% 口径一致）；decode 副本的空间换时间代价精确入档。

### 验收与口径发现
- exact-bytes 与公式 4.5+3.75·r̄ 偏差 **0.000%**（<1% gate 过）
- **论文引用注意**：字节加权 r̄（3B 0.135 / 8B 0.144）> 名义 0.125 — ΔD 把 FP8 ring 分给宽 N 投影（down=0、qkv/gate/up 0.18-0.23），ring 条数仍恰 12.5% 非超预算。**等效位宽引用 5.005(3B) / 5.041(8B) 并注明 r̄ 定义**。

---

## P0-3 · ring_size（组宽 b）敏感度消融（`rd_ringsize_ablation.md`）

**Gate**：b=512 float 逐位复现锚点（3B 8.2180/11.2549、8B 6.5642/9.5639）。

| b | 3B wiki/C4 | 8B wiki/C4 | 非单调层(在线gather) | 系统注记 |
|---|---|---|---|---|
| 32 | 8.294/11.350 | 6.587/9.601 | 3B 10.1% / 8B 14.6% | 需在线 gather |
| 64 | 8.296/11.387 | 6.565/9.594 | 0% | K_high=832(mod128=64) 半 k-tile 惩罚 2.7% |
| 128 | 8.225/11.278 | 6.577/9.613 | 0% | |
| 256 | 8.241/11.297 | 6.567/9.601 | 0% | |
| **512** | **8.218/11.255** | **6.564/9.564** | **0%** | K_high 天然 128 对齐；元数据缩 16× |

**结论（审稿人问"为何 b=512"）**：**没有取舍 — b=512 同时是精度 argmax 和系统复杂度 argmin**。b≤64 精度反而差（最多 +0.078 wiki，≈15× 噪声底）且触发对齐惩罚与在线 gather。注记：精度-b 非严格单调（中段交叉在噪声内）；avg_fp8 全点精确 0.1250 零偏移。

---

## 附 · 两项零卡时核对

1. **TPOT 增幅口径锁定**：论文旧值 "+3.7/+1.5/+4.5%" 来源 = **QServe(1024/512) b256 TPOT**（实算 +3.8/+1.5/+4.5 逐一吻合）。decode-heavy b256 为 +3.9/+2.7/+9.4；b16 小批 +41-46%（launch-bound）。**论文统一 QServe b256 口径并标注 batch**；"近乎免费"限定饱和批。
2. **ShareGPT 16 份数据确认干净**（vLLM bench_serving，11-06 同 session，512 req 全完成零失败）：吞吐 NVFP4 > FP8 > AdaMix(r1/16≈r1/8) > GPTQ/AWQ > BF16；AdaMix r=1/8 比 BF16 +17%(3B)/+43%(8B) — 可直接入 §6.4。

---

## 总结

P0 spec 四项全部收官，TACO 审稿防线闭合：GB10 同机基线回答了"何时用 AdaMix"、耗时/显存表回答了"离线代价与落地收益"、ring_size 消融回答了"为何 b=512"、两项核对统一了论文口径。唯一遗留（8B GB10 基线）为选跑项不阻塞投稿。§5/§6 可收口定稿。
