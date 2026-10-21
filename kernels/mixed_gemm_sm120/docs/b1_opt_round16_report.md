# B1 优化第十六轮 — 报告（DIAGNOSTIC round：解决「更高 TC 却更慢」的悖论 — metric error vs 不公平 A/B vs kernel 问题）

> 全程 remote-only（10.208.130.90；luofan-plaquant 建/跑，luofan-ncu 跑 ncu；RTX PRO 6000 Blackwell **SM120**）。
> 本轮**无任何 kernel 逻辑改动**（b1_tma_ext.cu / codesign_*.cu / 受保护文件全部零改动）——只加测量 harness
> （`run_r16_tc.sh`+`prof_r15dual.py`/`prof_tc.py` 复用、`agg_r16.py`、`run_r16_trace2.sh`+`agg_r16_trace2.py`）。
> 正确性：计时对象全程复用 round-15 已 gate 的 bit-exact 路径（cos=0.999999），本轮不重新引入未验核。
>
> ## 一句话结论（verdict）
> **悖论 = 不公平的 A/B（测量假象），不是 metric-read error，也不是 GEMM kernel 的问题。**
> round-15 的 ncu TC（两个 rollup，`-k device_kernel`）量的是**纯 GEMM**；端到端延迟 A/B 量的是 **GEMM + 每次调用的
> 激活-SF scatter + 每次 alloc/initialize 开销**。前者 co-design 在**两个 rollup 上都赢**出厂核、GEMM 本体在**同一时钟锁**下
> **快 13-18%**；后者把 co-design 的 wrapper 开销（`cd_fill_sf_kernel` 两发 + phys_SFA/workspace alloc + can_implement/initialize，
> 出厂核**一样都不付**）全算进去，才显得慢 30-42%。把比较对齐成「GEMM vs GEMM」，悖论消失。

---

## Part 0 — 悖论复述与三条待验路径

项目负责人指出的矛盾：round-15 说 co-design 的 TC **ACTIVE** 已达 CUTLASS 带（down 76.9-80.9%、alu/tc≈1），
但端到端比出厂 `b1_tma_ext.cu` 慢 30-42%。compute-bound GEMM 里 TC 占空比应决定延迟 → 矛盾。待判：
(a) **metric 读错**（ELAPSED≠ACTIVE：round-15 用 ACTIVE 宣称「TC matched」，而决定延迟的是 ELAPSED，出厂核可能 ELAPSED 更高）；
(b) **真 kernel 问题**（相位串行空窗 / memory-bound）。

本轮把三件事一次性量清：
1. **ELAPSED vs ACTIVE，两个 kernel 并排**，同一 ncu 会话（同 base-clock 锁），GEMM-only，含**绝对周期**。
2. **roofline**（DRAM%/L2/L1）确认 compute-bound 前提。
3. **A/B 公平性**：端到端 wall-clock 的构成分解（scatter / alloc / GEMM）。

**关键方法学修正（本轮最重要的一条）**：round-15 的 ncu 用 `-k regex:device_kernel` 只抓到 CUTLASS GEMM device_kernel；
co-design 的 `cd_fused_gemm` 每次调用还会 launch **两发 `cd_fill_sf_kernel`**（把激活 SF 从 natural layout scatter 进 CUTLASS 交错 atom buffer）
+ 两个 `phys_SFA` alloc + workspace alloc + `can_implement`/`initialize`——**这些全部不被 `device_kernel` regex 匹配，所以 TC 测量把它们排除了**，
而端到端 wall-clock 把它们**全算进去**。出厂手写核直接吃 natural-layout SF（软件广播，alu/tc≈3.5），**没有任何 per-call scatter**。
→ round-15 其实在比「GEMM-only（ncu）」对「GEMM+scatter+host（wall-clock）」，是**不对等比较**。

---

## Part 1 — ELAPSED vs ACTIVE：两个 kernel 并排，同一 base-clock 锁（deliverable 核心，`logs/tcR16/`，`agg_r16.py`）

同一脚本、同一 ncu（默认锁 base clock，两核时钟一致）、GEMM-only（CD=`device_kernel`，SHIP=`b1_tma_dual_kernel`
出厂生产配置 SFILV=0+SFASYNC(sfa1)+KPIPE，per-shape tile/sw/pers），3 shape × M{2048,8192} × r{⅛,½}。
TC 两个 rollup + **绝对周期**（`sm__pipe_tensor_cycles_active.sum`=tcyc / `sm__cycles_elapsed.sum`=ecyc）+ DRAM%/L2/L1 + dur。

**r = 1/8（生产点）：**

| cell | CD **TCela** / TCact | SHIP **TCela** / TCact | CD dur_us | SHIP dur_us | **CD/SHIP dur** | CD DRAM% | SHIP DRAM% |
|--|--|--|--|--|--|--|--|
| qkv  M8192 | **54.2** / 62.3 | **44.1** / 49.6 | 104.4 | 127.0 | **0.82** | 8.5 | 6.5 |
| qkv  M2048 | **35.8** / 54.3 | **31.0** / 46.6 | 38.8  | 44.9  | **0.86** | 8.5 | 7.3 |
| down M8192 | **72.2** / 80.7 | **60.2** / 66.7 | 311.8 | 371.7 | **0.84** | 10.5 | 8.9 |
| down M2048 | **51.5** / 76.9 | **44.2** / 65.6 | 112.7 | 129.5 | **0.87** | 11.6 | 10.1 |
| gate M8192 | **62.2** / 64.6 | **52.2** / 54.0 | 355.8 | 421.0 | **0.85** | 19.4 | 16.4 |
| gate M2048 | **53.6** / 62.7 | **45.8** / 51.8 | 106.2 | 123.9 | **0.86** | 7.9 | 6.7 |

**r = 1/2：**

| cell | CD **TCela** / TCact | SHIP **TCela** / TCact | CD dur_us | SHIP dur_us | **CD/SHIP dur** |
|--|--|--|--|--|--|
| qkv  M8192 | **58.4** / 66.4 | **48.9** / 54.8 | 128.9 | 153.5 | **0.84** |
| down M8192 | **72.4** / 80.7 | **59.5** / 65.8 | 410.9 | 499.1 | **0.82** |
| gate M8192 | **64.7** / 67.0 | **55.6** / 56.8 | 454.6 | 525.6 | **0.86** |
| qkv  M2048 | 39.2 / 60.2 | 33.9 / 51.5 | 47.8  | 55.4  | 0.86 |
| down M2048 | 52.5 / 78.3 | 43.8 / 64.8 | 145.9 | 173.6 | 0.84 |
| gate M2048 | 58.7 / 66.2 | 49.0 / 55.3 | 129.2 | 154.3 | 0.84 |

alu/tc：CD 0.68-1.15（继承 stock block-scaled MMA，硬件吃 SF）；SHIP **3.78-4.28**（手写软件 SF 广播）。regs 均 168。

### 读数（直接否掉 metric-error 假设）
1. **co-design 在两个 rollup 上都比出厂核高**，不是 round-15 悖论假设的「CD 高-ACTIVE/低-ELAPSED、SHIP 低-ACTIVE/高-ELAPSED」。
   SHIP 的 **ELAPSED 恒 ≤ 其 ACTIVE**（44-60% < CD 的 54-72%），**出厂核不可能在 ELAPSED 上赢**。→ **(a) metric-read error 被数据否定**。
2. **同一 base-clock 下，co-design GEMM dur 比出厂 GEMM 快 13-18%**（每个 cell，CD/SHIP dur=0.82-0.87）——与「CD 的 ELAPSED 更高」自洽
   （ELAPSED 高 → 同样 TC 工作量下 wall-clock 更短）。
3. **DRAM% 两核都 < 20%、L2 hit 90-95%**：两个 GEMM 都 **compute-bound**（负责人的前提成立），co-design **不是** memory-bound。
   → **(b) 的 memory-bound 子假设被否定**；相位空窗子假设见 Part 2/3（影响是个位数 pp，且 co-design GEMM 仍快于出厂）。

---

## Part 2 — 周期账：latency = TC_active_cycles / ELAPSED，直接算给你看（`agg_r16.py` 的 tcyc/ecyc 列）

compute-bound GEMM 的 TC-active 周期由 FLOPs 固定；两核做**同一个 GEMM**（同 M/N/Kl/Kh、同 fp4-Kl+fp8-Kh 数学）。实测 **tcyc 两核逐 cell 完全相同**：

| cell (r⅛) | **tcyc(M)**（两核相同） | CD **ecyc(M)** | SHIP **ecyc(M)** | CD/SHIP ecyc | = dur 比 |
|--|--|--|--|--|--|
| qkv  M8192 | 75.5  | 34.8  | 42.8  | 0.81 | 0.82 |
| down M8192 | 302.0 | 104.6 | 125.5 | 0.83 | 0.84 |
| gate M8192 | 302.0 | 121.3 | 144.5 | 0.84 | 0.85 |

**tcyc 相同 = 两核的张量管线工作量一样**（同样的 MMA 周期）；但 co-design 用**更少的 elapsed 周期**把它跑完
（ELAPSED 更高 → TC 占空比更满）。`latency ∝ ecyc`，ecyc 比 ≈ dur 比 ≈ ELAPSED 的倒数比——**完全一致，GEMM 本体没有任何悖论**。
矛盾只在把「GEMM-only 的 TC」对「GEMM+scatter+host 的 wall-clock」硬比时才出现。

---

## Part 3 — A/B 公平性 + wall-clock 分解：悖论的真正来源（`logs/traceR16b/`，`agg_r16_trace2.py`）

抓 co-design `fused_gemm` 每次调用的**全部 device kernel**（不加 `-k` 过滤），按名字切分
`cd_fill_sf_kernel`（激活-SF scatter）vs CUTLASS `device_kernel`（GEMM），稳态取中位数（base clock）：

| cell (r⅛) | scat_lo (NVFP4, M×Kl 线程) | scat_hi (MXFP8, M×Kh) | **scat_tot** | gemm | **scat占device%** | device_tot(base) | 对照 SHIP gemm(base) |
|--|--|--|--|--|--|--|--|
| down M8192 | **213.6** | 32.8 | **246.4** | 312.3 | **44.1%** | 558.7 | 371.7 |
| qkv  M8192 | 55.7 | 10.2 | 65.9 | 105.6 | 38.4% | 171.5 | 127.0 |
| gate M8192 | 55.7 | 10.2 | 66.0 | 355.7 | 15.7% | 421.7 | 421.0 |
| down M2048 | 55.7 | 10.3 | 66.0 | 112.6 | 37.0% | 178.7 | 129.5 |
| qkv  M2048 | 16.0 | 4.5 | 20.5 | 39.1 | 34.4% | 59.6 | — |
| gate M2048 | 16.1 | 4.6 | 20.7 | 104.8 | 16.5% | 125.5 | — |

出厂核 trace：**只有 `b1_tma_dual_kernel` 一发，零 scatter**（confirmed）。

### 这把 round-15 的端到端 A/B（`r15_lat.csv`）逐 shape 解释得严丝合缝
- **scatter 随 M×Kl 线性涨**：down(Kl7168) 213us ≈ qkv/gate(Kl1792) 56us 的 **~3.8×**（7168/1792=4×），M2048 再减半。
- **每 shape 的 wall-clock 退化 = scatter 占 device 的比例**：
  - down（scatter 44% of device）→ wall-clock **最差 +42%**；
  - qkv（GEMM 小，同样 56us scatter 占 38%）→ **+31%**；
  - gate（GEMM 大 N8192，56us scatter 只占 16%）→ **≈ 持平（-1.4%）**。
  round-15 观察到的「down 最差、gate 持平」**正是 scatter/GEMM 比例**，不是相位空窗。
- 粗对账（boost 时钟，用出厂核 wall≈其 GEMM 推 boost 因子 ~0.86）：down M8192 r⅛，co-design GEMM@boost≈269us、
  scatter@boost≈212us → 器件总 ~480us，加少量 host ≈ 实测 wall 455.5us。**scatter 几乎吃掉全部 135us 的 gap**。

### 相位屏障串行（round-15 归的「主因」）其实是次要的
co-design down r⅛ M8192 的 **GEMM ELAPSED=72.2%**，对照同形状 CUTLASS 纯-NVFP4 端点 ELAPSED≈78.8%（tcIMIX，跨会话参考）
——相位串行 + MXFP8 半速只压了 ~6pp；但 co-design GEMM 仍比出厂核 GEMM（60.2%）**高 12pp、快 16%**。
**相位空窗是真实但个位数 pp 的小税，不是端到端慢的原因**；round-15 Part 3 把它当「主因」是误判。

---

## Part 4 — 判定：是哪一类？

| 待判 | 本轮数据 | 结论 |
|--|--|--|
| (a) metric-read error（ELAPSED≠ACTIVE，SHIP 赢 ELAPSED） | co-design 在 **ELAPSED 与 ACTIVE 两个 rollup 上都高于**出厂核；SHIP ELAPSED≤其 ACTIVE<CD | **否**（出厂核不可能赢 ELAPSED） |
| (b)-memory | 两核 GEMM DRAM<20%、L2 90-95% | **否**（都 compute-bound） |
| (b)-相位空窗 | co-design GEMM ELAPSED 比纯 CUTLASS 低 ~6pp，但仍高于出厂核 12pp、GEMM 快 16% | **真实但次要**（非端到端慢的主因） |
| **不公平 A/B（wrapper 开销）** | 每次调用 2 发 `cd_fill_sf` scatter（66-246us，占 device 16-44%）+ alloc/initialize，**全被 TC 测量排除、全进 wall-clock**；出厂核零 scatter | **是，主因** |

### **悖论归因：不公平的 A/B（测量假象）**
- **ncu TC 没抓错**（两个 rollup 都对、co-design GEMM 本体确实更高占空比、确实更快）；
- **是「GEMM-only TC」对「GEMM+scatter+host wall-clock」的不对等比较**造成了假矛盾。
- round-15 的 **「TC matched（on ACTIVE）」是对的但口径偏窄**——更完整的说法是：**co-design GEMM 在 ELAPSED 与 ACTIVE 上都超过出厂核，
  且同锁下 GEMM 快 13-18%**；round-15 把端到端慢归因于「相位屏障串行」是**误判**，真正吃掉收益的是**未 hoist 的 per-call 激活-SF scatter + per-call setup**。

### 修正后的收敛图（ELAPSED 口径，r⅛ M8192，GEMM-only）
```
TC ELAPSED:   出厂手写核 44-60%   <   co-design 54-72%   <   CUTLASS 纯NVFP4 端点 ~63-79%(参考)
GEMM 延迟(同锁): co-design 比出厂核快 13-18%（ecyc 比 0.81-0.84）
端到端(wall):   co-design 慢 30-42%（qkv/down）/ 持平(gate) ——全部来自 per-call scatter(16-44% device)+ setup，出厂核不付
```

### 若把 A/B 做公平（会不会翻盘）
会。**GEMM-only 的公平 A/B（同 base-clock ncu）里 co-design 已经快 13-18%**。端到端要翻盘只需把 wrapper 开销拿掉——
**全部可在 harness/wrapper 层做，不碰 mainloop**：
1. **把激活-SF scatter 从 per-call 挪走 / 融合**：`cd_fill_sf_kernel` 现在是 M×K 线程、1-byte 非合并 scattered write 的独立核
   （down M8192 Kl7168 = 58.7M 线程 → 213us）。可 (i) 让量化/pack 核**直接产出 CUTLASS 交错 atom 布局**（零额外核），
   或 (ii) 融进前一个 pack kernel，或 (iii) 若激活 SF 在多次 GEMM 间复用则 hoist 到循环外（本 A/B 已对权重 SF 这么做 = `prepack_sf_*`）。
2. **缓存 workspace / 跳过 per-call `initialize`+`can_implement`**：持久化 GemmUniversalAdapter 的 workspace 与 args，避免每次 alloc+host 构造。
   （本轮因受保护文件不可改 `codesign_fused_ext.cu`，未在源码层 hoist；上述为公平化方案，留给下一轮实现并重新 gate bit-exact。）

**诚实边界**：scatter 本身是 co-design「采纳 stock block-scaled collective（要求交错 SF atom 布局）」的真实代价，不是纯粹的 harness bug——
但它是**可摊销/可融合**的 wrapper 代价，不是 GEMM mainloop 的计算缺陷。出厂核把 SF 处理放在 GEMM 内（alu/tc 3.5、单发），
co-design 放在 GEMM 外（alu/tc 1、GEMM 快但多一发 scatter）——这是设计权衡，当前实现把权衡做在了低效的独立 scatter 核上。

---

## Part 5 — 对 round-15 结论的修正（honest reporting）

| round-15 原话 | 本轮修正 |
|--|--|
| 「每相位 TC 继承 CUTLASS（TCact 54-81%）」→ 用 **ACTIVE** 宣称 matched | 口径偏窄但不算错：co-design **ELAPSED 也高于出厂核**；完整结论是两个 rollup 都赢 + GEMM 同锁快 13-18% |
| 「端到端慢的**主因是相位屏障串行**」 | **误判**：相位空窗只压 ~6pp ELAPSED，且 co-design GEMM 仍比出厂核快；**主因是 per-call 激活-SF scatter + setup**（占 device 16-44%，被 TC 测量排除） |
| 「higher-TC-but-not-faster」 | 更准确：**higher-TC-and-faster-GEMM, but-slower-end-to-end-due-to-un-hoisted-SF-scatter**。NO-SHIP 对**当前 wrapper**仍成立，但 blocker 可在 wrapper 层修、不碰 mainloop |

**ship/no-ship 的净判定**：当前 `codesign_fused_ext` 端到端 NO-SHIP 仍成立（wrapper 带着 scatter+setup）；
但**根因已从「kernel 结构（相位串行）」改正为「wrapper 的 per-call SF scatter/setup」**，且这是**可修复的测量/封装问题**，
不是 GEMM 的计算问题——GEMM 本体已经赢出厂核。下一轮的正确动作是**公平化 wrapper（hoist/融合 SF scatter + 缓存 workspace）后重测**，预期翻盘为 SHIP。

---

## 产物
- **新增测量 harness（零 kernel 改动）**：`run_r16_tc.sh`（两核同会话 ncu：两 rollup + 绝对周期 + DRAM/L2/L1 + dur）、
  `agg_r16.py`（聚合）、`run_r16_trace2.sh`（全-kernel trace，切 scatter/GEMM）、`agg_r16_trace2.py`（分解）。
  复用 `prof_r15dual.py`（CD runner）/`prof_tc.py`（SHIP runner）。
- 数据：`logs/tcR16/`（24 captures，CD×SHIP×shape×M×r⅛/½，含 tcyc/ecyc/DRAM/L2）；`logs/traceR16b/`（9 captures，device 分解）；
  对账 wall-clock 用 round-15 的 `r15_lat.csv`（已 gate cos=0.999999）。
- **未改动（保护）**：`b1_tma_ext.cu` / `codesign_fused_ext.cu` / `codesign_fused_kernel.hpp` / `codesign_sm120_bs_mma.hpp` /
  `mixed_gemm.py` / `pure_presf_ext.cu` 等。
- 复用已缓存 .so（两核 `ninja: no work to do`，无 40min 重编）。
- `candidates.jsonl`：+3 条（metric-error 否定；GEMM 同锁 co-design 快 13-18% + tcyc 相同 ecyc 更少；scatter 分解=悖论主因）。GPU 2/3 已释放。
