# B1 优化第二十五轮 — 报告（dense-r 公平 best-config 全程扫描：co-design GEMM vs 固定 FP4/FP8 端点）

> 全程 remote-only（10.208.130.90；`luofan-plaquant` 跑；RTX PRO 6000 Blackwell **SM120**，188 SM）。锁频 **2415 MHz**（空闲 GPU1；跑前 `-lgc 2430`，实测稳定 2415）。跑完 `-rgc` 解锁。
> **GEMM-only**——计时区只有 GEMM kernel，输入全部预打包（NVFP4/MXFP8 codes + 预交错 SF），无 quant/pack、无 e2e。
> **纯测量，零新核逻辑**：复用 round-24 已编译的 best-config 二进制（`presf_cfg_*`、`cd_cfg24_*`，torch 扩展缓存命中，**无重编译**）。PROTECTED 文件全未动；本轮唯一新增 `r25_fair_dense.py`（run-only 扫描脚本，#include 同一套受保护 mainloop 头）。
> 正确性硬门：每个测量点先过 cosine 再计时；**全程最低 cosine = 0.999998**（co-design 的 bit-exact 水平，round-24 同级）。

---

## 一句话结论

1. **全 17 点 dense-r 都测到了**（r = 0, 1/16, …, 16/16）。`Kt∈{2048,8192}` 下 r=k/16 **全部精确落在合法 K-split**（Kh=k·Kt/16 必是 32 的倍数，Kl=Kt−Kh 必是 64 的倍数；脚本内 `assert` 逐点确认）。唯一测不到的是 **r=1（Kl=0）**：dual kernel 无 NVFP4 phase 不能跑，记 **N/A**；**FP8 端点就是 r=1 的值**（CSV cfg 列标 `NA(=FP8_endpoint)`）。每 cell 16 个可测 r + 1 个 N/A = 17 行，共 **204 行**。
2. **co-design 随 r 单调退化**（r↑→FP8 占比↑→有效吞吐↓），是干净的从 FP4 端点到 FP8 端点的下降曲线；但**它相对「FP4→FP8 两独立 best-config 发射的线性混合参考线」的位置分两类**：
   - **大 M / 计算受限 cell（qkv M≥4096、down M≥2048）**：co-design **全程压在线性参考线之上**（单发 whole-K + 硬件 block-scaled 共享 accumulator 赢过「跑两次各自最优核再相加」的朴素线性 blend）。最猛在 **down·M8192**，全 r 高出线性参考 **+9%…+58%**；down·M4096 +9…+52%；qkv·M8192 +9…+34%。
   - **小/中 M、或 N 很大的 gate（qkv/gate/down·M512、gate 全 M、qkv·M2048）**：co-design **贴着或略低于线性参考线**（持久单波在 CTA 不足时摊不开 ramp，退回端点水平）。gate 偏负最多（gate·M4096 低 −3…−21%）——与 round-23「gate N 大、co-design 在中 M 占用不足」一致。
3. **%-of-peak（eff_peak(r)=peak_nvfp4/(1+r)，peak_nvfp4=188·clk·4096）随 r 普遍缓升**：有效峰值随 r 收缩得比实测吞吐更快，故 %-peak 爬坡；**最高 ~62%（down·M8192, r≈0.94）**，最低小 M ~6–9%。这重申 round-23/24 的「SM120 block-scaled pipeline 天花板 ~60–78%」。

---

## Part 1 — dense-r 公平表（co-design TFLOPS + %-peak，全 17 r）vs 固定 FP4/FP8 best-config 端点

端点为 **r-无关的 per-shape×M 固定值**，本轮同场重测复核（与 round-24 一致，差异在时钟 2415 vs 2332 的线性缩放内）：FP4 端点 = `{128x128_auto, 128x128_pp, 256x128_s3, 128x64_auto}` 取最优；FP8 端点 = `{128x128_auto, 128x128_pp, 128x64_auto}` 取最优。co-design best config 取 round-24 dispatch：qkv/down·M512 → `128×64 NV4/MX2`，其余 → `128×128 NV4/MX2`（gate 全程 128×128）。

### 每 cell 概览（TFLOPS）：FP4/FP8 端点、co-design 端值/中点/全程范围、相对线性参考

| cell | FP4ep | FP8ep | cd@r=0 | cd@r=½ | cd range (min…max) | cd vs 线性参考（16 可测 r） |
|--|--|--|--|--|--|--|
| qkv·512 | 104.9 | 94.1 | 101.9 | 101.5 | 93.5 … 109.9 | 12/16 在线上，gap **[−3%, +5%]** |
| qkv·2048 | 307.8 | 240.6 | 293.2 | 258.6 | 227.8 … 304.0 | 1/16，**[−9%, 0%]** |
| qkv·4096 | 442.6 | 328.2 | 513.8 | 413.1 | 348.6 … 513.8 | **16/16**，**[+3%, +16%]** |
| qkv·8192 | 534.9 | 382.7 | 718.2 | 515.4 | 437.0 … 718.2 | **16/16**，**[+9%, +34%]** |
| gate·512 | 324.8 | 245.4 | 319.2 | 259.4 | 231.9 … 319.2 | 0/16，**[−10%, −2%]** |
| gate·2048 | 727.2 | 468.0 | 719.2 | 524.8 | 438.1 … 719.2 | 0/16，**[−15%, −1%]** |
| gate·4096 | 929.8 | 587.4 | 905.5 | 648.3 | 526.3 … 905.5 | 0/16，**[−21%, −3%]** |
| gate·8192 | 989.9 | 591.0 | 998.5 | 693.6 | 570.8 … 998.5 | 1/16，**[−12%, +1%]** |
| down·512 | 311.0 | 237.9 | 309.1 | 262.8 | 226.3 … 309.1 | 0/16，**[−9%, 0%]** |
| down·2048 | 524.0 | 355.6 | 651.7 | 485.2 | 389.9 … 651.7 | **16/16**，**[+4%, +24%]** |
| down·4096 | 642.4 | 434.5 | 974.6 | 683.6 | 545.7 … 974.6 | **16/16**，**[+9%, +52%]** |
| down·8192 | 709.4 | 458.4 | 1117.9 | 761.7 | 594.7 … 1117.9 | **16/16**，**[+9%, +58%]** |

> cd@r=0 = 纯 NVFP4 单发（`single_nvfp4_gemm`，whole-K）；cd@r=½ 为 Kh=Kl 的双格式。「线性参考」= `fp4_tf·(1−r) + fp8_tf·r`（两端点 best-config 各自独立发射后的朴素 TFLOPS 线性插值）。

### 代表 %-of-peak 轨迹（cd_best_pct_peak，eff_peak(r)=peak_nvfp4/(1+r)）

| cell | r=0 | r=¼ | r=½ | r=¾ | r=15/16 |
|--|--|--|--|--|--|
| qkv·8192 | 38.6 | 38.4 | 41.6 | 44.2 | 45.5 |
| gate·8192 | 53.7 | 53.1 | 55.9 | 58.3 | 59.5 |
| down·8192 | 60.1 | 59.9 | 61.4 | 61.4 | 62.0 |
| down·4096 | 52.4 | 53.1 | 55.1 | 55.3 | 56.9 |

> %-peak 对 r 缓升（有效峰值收缩快于实测吞吐），峰顶 ~62% @ down·M8192，重申 SM120 block-scaled pipeline 天花板。

---

## Part 2 — 诚实边界 / 方法

- **端点是固定单值**：每 (shape,M) 一次性同场交错（min-of-50，median warmup 15）测全部可用纯配置取最优，作为 17 行共享的 FP4/FP8 端点；co-design 每个可测 r 独立与两端点 best-config 交错（同一时钟/竞争窗，r23_partA 法）取 min。
- **r=1 不可测**是 dual kernel 结构性的（Kl=0→无 NVFP4 phase），不是测量失败；**该点 = FP8 端点**，CSV 已显式标注，报告与线性参考分析均以此为右端。
- **cd best config r-无关**沿用 round-24 的实测结论（r{0,⅛,½} 内配置最优一致）；本轮不重跑 config grid，直接用 round-24 dispatch 表的 per-cell best config 跑全 17 r。
- **时钟 2415 MHz**（锁 2430、功率封顶实测稳 2415），略高于 round-24 的 2332，故本轮绝对 TFLOPS/%-peak 比 round-24 整体高 ~3.5%（线性缩放），**趋势、winner、vs-线性参考的符号完全一致**。
- **cosine 全程 ≥ 0.999998**（co-design bit-exact 水平）；纯端点 0.999999。无静默略过，无 cosine 失败。
- **mainloop 逻辑零改动**：复用受保护 co-design/纯端点 mainloop 头与 round-24 config-only 扩展；本轮仅新增 run-only 扫描脚本。

## 产物

- **数据**：`r25_fair_dense.csv`（204 行 = 12 cell × 17 r；列：shape, M, N, Kt, r, Kl, Kh, clk, cd_best_us, cd_best_tflops, cd_best_pct_peak, cd_best_cfg, fp4_best_tflops, fp8_best_tflops）。
- **脚本**：`r25_fair_dense.py`（run-only dense-r 扫描；reload 缓存扩展无重编译）。日志 `logs/_r25_dense.log`（判定行 `R25_FAIR_DENSE_ALL_COSINE_PASS`、`R25_MIN_COSINE=0.999998`、`R25_FAIR_DENSE_DONE`）。
- `candidates.jsonl`：+1 条（dense-r 公平扫描 → co-design 全 r 相对线性参考的 above/below 分区图景）。
- **GPU 已释放**（GPU1 `-rgc` 解锁）。
