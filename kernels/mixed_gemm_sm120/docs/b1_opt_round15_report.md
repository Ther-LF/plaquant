# B1 优化第十五轮 — 报告（MEASUREMENT round：双格式 co-design kernel 的 ncu TC + 端到端延迟 A/B → SHIP/NO-SHIP）

> 全程 remote-only（10.208.130.90；luofan-plaquant 建/跑延迟，luofan-ncu 跑 ncu；RTX PRO 6000 Blackwell **SM120**）。
> 本轮无新 kernel 功能：只收割 round-14 被 infra 卡住的两个测量——(1) 直接 ncu 双格式 co-design kernel 的 TC；
> (2) co-design vs 出厂 `b1_tma_ext.cu` 的端到端延迟 A/B——并据数据做 ship/no-ship 判定。
> **不碰任何 kernel 源（b1_tma_ext.cu / codesign_* / 受保护文件全部零改动）；只加测量 harness（`r15_lat.py`/`prof_r15dual.py`/`run_r15_tc.sh`/`agg_r15.py`）。**
>
> **本轮结论（干净判定）：双格式 co-design kernel 在 SM120 上 bit-exact（cosine=0.999999，24/24 cell）且每相位 TC 继承 CUTLASS 水平
> （alu/tc 0.68-1.15≈1，TCact 54-81%），但端到端**比出厂 `b1_tma_ext.cu` 慢 19-42%**（production r⅛：qkv +31%、down +42%、gate ±1%）。
> → **NO-SHIP：co-design 记为「higher-TC-but-not-faster」，出厂 `b1_tma_ext.cu` 保持生产 kernel。** 相位屏障串行是症结（见 Part 3）。**

---

## Part 0 — 本轮状态与 round-14 遗留

round-14 构造了双格式共享累加器 co-design kernel 并 gate 过 bit-exact（cos_core=0.999997/cos_full=0.999999），
但两项测量被远程机 CPU 过载卡死：`_cdfused_gate.log=EXIT_137`（clean build 被杀）、`_cdlat.log=0 字节`（latency 从未产出）。

**本轮机器已缓解**：128 核、load ~14、无饥饿——codesign_fused_ext 单 TU 编译 **~3 分钟**完成（round-14 的「40min 拿 69s CPU」是当时并发重编译导致的调度饥饿，非常态）。
两个 detached 作业（GPU 1 跑延迟、GPU 3 跑 ncu）各自 clean build + 测量，全部干净落盘。
出厂 `b1_tma_ext.so`（round-12 产，md5 与本地 committed 源一致、未被 round-14 触碰）直接复用（touch 缓存跳过重编）。

---

## Part 1 — 双格式 co-design kernel 的 ncu TC（deliverable #1，`logs/tcR15/`，`agg_r15.py`）

round-11 imix 指令集 + **两个 TC rollup**（`...pct_of_peak_sustained_active`=TCact / `...elapsed`=TCela）+ alu/tc + lsu/tc，
直接打在**双格式 fused_gemm 的 CUTLASS device_kernel** 上（`-k regex:device_kernel -c1 -s5`；每次 fused_gemm 只有 1 个 device_kernel，
SFA scatter 是另名 kernel 不被匹配），3 shape × M{2048,8192} × r{⅛,½}。

| cell (r=⅛) | **DUAL** TCact / TCela / **alu/tc** / lsu/tc / dur_us | round-13 单格式 PoC（=CUTLASS 端点）TCact / alu/tc | ΔTCact(dual−PoC) |
|--|--|--|--|
| qkv M8192  | 62.4 / 53.9 / **1.02** / 0.83 / 105.5 | 69.9 / 1.16 | **−7.5pp** |
| qkv M2048  | 54.4 / 36.2 / **1.15** / 0.84 / 38.5  | 59.0 / 1.26 | −4.6pp |
| down M8192 | 80.9 / 72.5 / **0.68** / 0.74 / 313.2 | 86.6 / 0.80 | −5.7pp |
| down M2048 | 76.9 / 51.5 / **0.72** / 0.74 / 112.1 | 81.1 / 0.83 | −4.2pp |
| gate M8192 | 64.7 / 62.5 / **0.99** / 0.82 / 353.2 | 73.1 / 1.13 | −8.4pp |
| gate M2048 | 62.5 / 54.1 / **1.02** / 0.83 / 105.1 | 70.1 / 1.15 | −7.6pp |

| cell (r=½) | **DUAL** TCact / TCela / **alu/tc** / lsu/tc / dur_us |
|--|--|
| qkv M8192  | 66.2 / 58.0 / **0.95** / 0.96 / 130.7 |
| qkv M2048  | 60.3 / 39.8 / **1.05** / 0.97 / 47.3  |
| down M8192 | 80.9 / 72.6 / **0.70** / 0.90 / 410.0 |
| down M2048 | 78.4 / 52.2 / **0.72** / 0.90 / 146.7 |
| gate M8192 | 67.0 / 64.6 / **0.93** / 0.96 / 453.1 |
| gate M2048 | 66.5 / 58.7 / **0.95** / 0.96 / 129.2 |

regs=168（全 cell）。

**发现（证据，非猜测）：**
1. **alu/tc = 0.68-1.15 ≈ 1**（目标达成）——双格式核逐字继承 STAGE-1 的两个 stock block-scaled collective（硬件 `mma.sync...block_scale` 直接吃 SF），
   fork 只加了 `if constexpr(ClearAccum)` 守卫 + glue 管线，没碰 MMA/SF 消费路径。对照出厂手写 consumer 的 alu/tc≈3.5 —— **低 ~3-4×**，确认 TC 质量成立。
2. **TCact：down 落在 CUTLASS 68-86% 带内（76.9-80.9%）；qkv/gate 略低于带（54.4-67.0%）**。
3. **round-14 的唯一未验变量（相位屏障串行对聚合 ACTIVE 的影响）现在量化了：双格式比单格式 PoC 低 4.2-8.4pp**
   （M8192 三 shape −5.7/−7.5/−8.4pp）。根因就是 correctness-first 的相位屏障——NVFP4 相位 drain 完 union smem 才让 MXFP8 相位覆写，**无跨相位 overlap**，
   相位切换处 pipeline 空窗压低了聚合 TC 占空比。这是 round-14 预判的代价，实测为个位数 pp。
4. 确认目标（「alu/tc≈1 + ACTIVE 在/近 CUTLASS 带」）：**alu/tc 达成；ACTIVE down 带内、qkv/gate 近带（差 4-8pp 的相位串行税）**。

---

## Part 2 — 端到端延迟 A/B（deliverable #2，`r15_lat.csv`）：SHIP/NO-SHIP

**co-design `fused_gemm`（双格式共享累加器，plain epilogue）** vs **出厂 `b1_tma_ext.b1_fused_tma_dual`
（round-12 生产配置：SFILV=0 + SFASYNC(sfa=1) + KPIPE per-shape，full s_x·s_w epilogue）**，同数据同 GPU，warm，median(50, 10 warmup)。
3 shape × M{2048,8192} × r{0,⅛,½,1}。excess vs `t_ideal(r)=t_fp4·(1−r)+t_fp8·r`（同 session 的 CUTLASS pure-presf 端点）。
**先 re-confirm 双格式 cosine（硬门，施加可分离 s_x·s_w）再计时**——全 cos=0.999999，`R15_LAT_ALL_COSINE_PASS`。

绝对 us（cd=co-design，ship=出厂）+ excess + cd/ship 比值：

| shape | M | r | cos_cd | cd_us | ship_us | ideal_us | ex_cd | ex_ship | **cd/ship** |
|--|--|--|--|--|--|--|--|--|--|
| qkv  | 2048 | 0    | 0.999999 | 58.4  | 54.7  | 58.0  | +0.6%  | −5.7%  | 1.07 |
| qkv  | 2048 | ⅛    | 0.999999 | 71.8  | 55.4  | 60.2  | +19.3% | −8.1%  | **1.30** |
| qkv  | 2048 | ½    | 0.999999 | 77.8  | 63.7  | 66.9  | +16.3% | −4.8%  | 1.22 |
| qkv  | 2048 | 1    | —        | n/a   | 75.2  | 75.8  | —      | −0.8%  | — |
| qkv  | 8192 | 0    | 0.999999 | 137.1 | 116.6 | 134.2 | +2.2%  | −13.1% | 1.18 |
| qkv  | 8192 | ⅛    | 0.999999 | 155.6 | 118.7 | 140.8 | +10.5% | −15.7% | **1.31** |
| qkv  | 8192 | ½    | 0.999999 | 176.4 | 144.0 | 160.7 | +9.8%  | −10.4% | 1.22 |
| qkv  | 8192 | 1    | —        | n/a   | 178.3 | 187.1 | —      | −4.7%  | — |
| down | 2048 | 0    | 0.999999 | 144.1 | 120.1 | 141.8 | +1.6%  | −15.3% | 1.20 |
| down | 2048 | ⅛    | 0.999999 | 158.4 | 121.9 | 150.1 | +5.5%  | −18.8% | **1.30** |
| down | 2048 | ½    | 0.999999 | 185.5 | 155.4 | 174.9 | +6.0%  | −11.2% | 1.19 |
| down | 2048 | 1    | —        | n/a   | 199.3 | 208.1 | —      | −4.2%  | — |
| down | 8192 | 0    | 0.999999 | 412.8 | 312.8 | 408.5 | +1.1%  | −23.4% | 1.32 |
| down | 8192 | ⅛    | 0.999999 | 455.5 | 320.4 | 435.2 | +4.7%  | −26.4% | **1.42** |
| down | 8192 | ½    | 0.999999 | 537.0 | 419.6 | 515.5 | +4.2%  | −18.6% | 1.28 |
| down | 8192 | 1    | —        | n/a   | 559.8 | 622.6 | —      | −10.1% | — |
| gate | 2048 | 0    | 0.999999 | 103.6 | 118.2 | 99.3  | +4.3%  | +19.0% | **0.88** |
| gate | 2048 | ⅛    | 0.999999 | 124.9 | 120.7 | 106.2 | +17.6% | +13.7% | 1.03 |
| gate | 2048 | ½    | 0.999999 | 145.1 | 146.5 | 126.8 | +14.5% | +15.6% | 0.99 |
| gate | 2048 | 1    | —        | n/a   | 180.0 | 154.2 | —      | +16.7% | — |
| gate | 8192 | 0    | 0.999999 | 320.0 | 358.4 | 313.2 | +2.2%  | +14.4% | **0.89** |
| gate | 8192 | ⅛    | 0.999999 | 366.9 | 372.0 | 336.7 | +9.0%  | +10.5% | 0.99 |
| gate | 8192 | ½    | 0.999999 | 448.4 | 447.7 | 407.0 | +10.2% | +10.0% | 1.00 |
| gate | 8192 | 1    | —        | n/a   | 563.5 | 500.9 | —      | +12.5% | — |

说明：
- r=0 的 co-design 用 `single_nvfp4_gemm`（STAGE-1 纯 NVFP4 PoC 核，非双相位核）；r=1（纯 MXFP8，K_low=0）co-design 无暴露路径，记 n/a（出厂核照常计时）。
- excess 诚实 caveat：co-design 走 plain epilogue（少做外层 s_x·s_w），出厂核做 full epilogue。GEMM mainloop 主导、epilogue 占比极小——
  **即便把这点差额全算给出厂核，co-design 在 qkv/down r⅛ 仍慢 30-42%，结论不翻盘**。

**读数：**
1. **production 点 r=⅛（最关键）**：co-design **慢**——qkv M8192 +31%、down M8192 **+42%**、gate M8192 −1.4%（噪声内）。M2048 同向。
2. **excess vs t_ideal(r)**：出厂核在 qkv/down **低于** CUTLASS 2-launch 包络 8-26%（r⅛ down M8192 −26.4% ≈ 任务引用的 −29%，run 间抖动），
   co-design 全部**高于**包络（+4.7% ~ +19.3%）。即出厂核不但更快，还赢过 CUTLASS 2-launch 理想线；co-design 连理想线都没压过。
3. gate shape co-design ≈ 持平（r⅛/½ 比值 0.99-1.03）；仅退化 r=0（纯 NVFP4 PoC 核）比出厂双核跑退化 case 快 11-12%（出厂双核在 r=0 带着双相位结构的空转开销）。

---

## Part 3 — 为什么更高 TC 没转成延迟（「higher-TC-but-not-faster」的机理）

co-design 每相位 TC 更优（alu/tc≈1 vs 出厂 ~3.5），但端到端更慢，三条叠加：
1. **相位屏障串行（主因）**：correctness-first 的 `phase_bar1/2` 让 NVFP4 相位完全 drain union smem 后 MXFP8 相位才开始——
   **零跨相位 overlap**。Part 1 实测聚合 ACTIVE 因此比单格式 PoC 低 4-8pp，且相位切换处的 pipeline 空窗直接计入 wall-clock。
2. **union smem 压浅 stage**：SM120 每 block 仅 101376 B opt-in smem，强制两相位 union，NV=4/MX=2 stages（< 单格式满环），prefetch 深度受限。
3. **出厂核的 whole-K overlap + 单 launch**：出厂手写核虽 alu/tc≈3.5，但保持**单一融合 consumer**、全 K 跨块 overlap + SFASYNC 预取 + 单 launch，
   在我方 K-size（Kl 1792-7168 / Kh 256-1024）上把内存/计算叠满——所以它甚至压过 t_ideal(r)（省掉第二次 launch + 中间写）。

co-design 的 TC 优势是「**每相位**占空比」，而延迟由「**聚合** wall-clock（含相位空窗）」决定；在当前相位串行结构下，前者赢、后者输。

---

## Part 4 — 判定

| 判据 | 目标 | 实测 | 判定 |
|--|--|--|--|
| 双格式 cosine | =1.0 | 0.999999（24/24 cell，含 r=0 PoC） | ✔ |
| 双格式 alu/tc | ≈1 | 0.68-1.15 | ✔（继承 STAGE-1；vs 手写 3.5 低 3-4×） |
| 双格式 ACTIVE | 在/近 CUTLASS 带 | down 76.9-80.9%（带内）；qkv/gate 54.4-67.0%（近带，相位串行 −4-8pp） | ◑ |
| 相位串行对聚合 ACTIVE 影响 | 量化 | **−4.2 ~ −8.4pp vs 单格式 PoC**（round-14 未决问题已答） | ✔ |
| 延迟 ≤ 出厂核（ship 条件） | ≤1.0× | qkv/down r⅛ **1.30-1.42×（慢）**；gate r⅛/½ 0.99-1.03×（持平） | ✘ |

### **SHIP / NO-SHIP 判定：NO-SHIP**
双格式共享累加器 co-design kernel **bit-exact 成立、每相位 TC 达 CUTLASS 水平**，但端到端在 production r⅛ 比出厂 `b1_tma_ext.cu`
**慢 30-42%（qkv/down）、持平（gate）**，且 excess 全部在 t_ideal(r) 之上（出厂核在其之下）。
更高的 TC 被相位屏障串行 + union 浅 stage 的开销吃掉——典型的「higher-TC-but-not-faster」。

**→ 保持出厂 `b1_tma_ext.cu`（SFILV=0 + SFASYNC + KPIPE per-shape dispatch）为生产 kernel。**
co-design 记为已验证的「更高 TC 但当前结构不更快」方向，**不上位、不回退**。

### 诚实边界 / 若要让 co-design 翻盘（future work，本轮不做）
1. r=1（纯 MXFP8）co-design 无暴露路径，未计时（退化 case，出厂核覆盖）。
2. co-design plain epilogue vs 出厂 full epilogue 的差额已论证不影响结论（mainloop 主导，30-42% gap 远大于 epilogue 占比）。
3. **唯一可能翻盘的改法**：去掉相位屏障串行、做**跨相位 overlap**（双缓冲 union 或 non-union + 更激进 stage 裁剪，让 NVFP4/MXFP8 相位在时间上重叠）——
   把「每相位 TC 优势」变成「聚合 wall-clock 优势」。这是对 kernel 结构的实质改动，须另起一轮并重新 gate bit-exact。

---

## 产物
- **新增测量 harness（无 kernel 改动）**：`r15_lat.py`（延迟 A/B + cosine 门）、`prof_r15dual.py`（双格式 ncu runner）、
  `run_r15_tc.sh`（ncu imix + 两 TC rollup）、`agg_r15.py`（聚合）。
- 数据：`r15_lat.csv`（24 cell，cos=0.999999，含 cd/ship 比值与双向 excess）；`logs/tcR15/`（12 ncu captures，CDd × shape × M × r{⅛,½}）；
  日志 `logs/_r15_lat.log`（`R15_LAT_ALL_COSINE_PASS`+`R15_LAT_DONE`）、`logs/_r15_tc.log`（`TC_R15_DONE`）。
- **未改动（保护）**：`b1_tma_ext.cu` / `codesign_fused_ext.cu` / `codesign_fused_kernel.hpp` / `codesign_sm120_bs_mma.hpp` / `mixed_gemm.py` / `pure_presf_ext.cu` 等。
- 复用：出厂 `b1_tma_ext.so`（md5 校验 = committed 源）跳过重编；codesign_fused_ext clean build ~3min（机器已缓解）。
- `candidates.jsonl`：+2 条（双格式 TC 实测继承 + 延迟 A/B NO-SHIP）。GPU 1/3 已释放。
