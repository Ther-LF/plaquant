# B1 优化第十七轮 — 报告（公平化 wrapper 后重测 → 真正的 SHIP/NO-SHIP）

> 全程 remote-only（10.208.130.90；luofan-plaquant 建/跑；RTX PRO 6000 Blackwell **SM120**，GPU 2 空闲、无 co-tenant）。
> 本轮**无任何 mainloop 逻辑改动**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / 受保护文件全部零改动。
> 只改 wrapper/loader/pack 层（`codesign_fused_ext.cu` 的 SF-interleave + workspace 封装），再加公平 A/B harness（`r17_lat.py`）。
> 正确性：buffer 级 bit-exact 探针 + cosine 硬门 **24/24 cell = 0.999999** 先过，再计时。
>
> ## 一句话结论（verdict）
> **把 round-16 诊断出的「不公平 wrapper 开销」拿掉后，公平 A/B 翻盘为 **条件 SHIP（按 shape 分派）**。**
> 用高效 activation-SF interleave（1 线程/SF 字节，取代旧的 VEC=16/32× 冗余 scatter）+ 缓存 workspace 后，
> co-design 的**真实每次调用延迟**在**大 GEMM 预填充 shape 上已快过出厂核**：down·8192 **0.95×**（快 5%）、gate·8192 **0.94×**（快 6%）
> ——这两条从 round-15 的 1.42× / 0.99× 直接翻过来。GEMM-only 天花板（activation SF 预交错）在 6 个 shape 里**赢 5 个**（0.92–0.97×），
> 只输 qkv·2048。残余的 per-call interleave（~9–20us）仍让 qkv 与 M2048 小 shape 略慢，**出厂核在这些 shape 保持最优**。
> **诚实双向**：co-design 不是全盘碾压，但也绝非 round-15 的全盘 NO-SHIP——它**赢大 shape、输小 shape**，应按 shape 分派。

---

## Part 0 — 本轮依据（round-16 的诊断）

round-16 把「更高 TC 却更慢」的悖论定性为**不公平的 A/B**（测量假象）：
- co-design 的 GEMM 本体在同锁下**已快 13–18%**（两个 TC rollup 都赢、tcyc 逐 cell 相同、ecyc 更少）；
- 端到端慢只因 wrapper 每次调用付了出厂核**不付**的代价——**两发 `cd_fill_sf_kernel` 激活-SF scatter**（占 device 16–44%）+ workspace alloc/initialize/can_implement。
- round-16 明确：这些是**可摊销/可融合的 wrapper 代价，不是 GEMM 的计算缺陷**，修在 wrapper 层即可、不碰 mainloop。

本轮就做这件事，然后重测，做真正的 ship/no-ship 判定。

---

## Part 1 — 消除了什么（wrapper 层，零 mainloop 改动）

### 1. 权重-SF interleave → load 时一次（已摊销到零 per-call）
`prepack_sf_fp4_b/fp8_b` 本来就由 caller 在计时循环**外**调用一次（`r15_lat.py`/`r17_lat.py` 都如此）——权重 SF 是静态的，交错只做一次。本轮顺手把它也换成高效 kernel（免费）。**per-call 权重-SF 代价 = 0**，确认。

### 2. 激活-SF interleave → 从「16/32× 冗余 scatter」改成「单遍、每 SF 字节 1 线程」（核心修复）
**根因（本轮确证）**：旧 `cd_fill_sf_kernel` 按 `(row,k)` 每元素开一个线程（`rows*K`），但 CUTLASS atom 布局里 **VEC（block 内）那一维是 stride-0 broadcast**（`filter_zeros` 把它折叠，故 phys buffer 只有 `rows*K/VEC` 字节）——于是同一个物理字节被**重复写 VEC(=16/32)×**。这正是 round-16 测到的 16–44%-of-device scatter。
**修复**：新 `cd_fill_sf_block_kernel` 按 `(row, block)` 每逻辑 SF 字节开 1 线程（`rows*K/VEC`），每个物理字节**恰写一次**。读是合并的（连续 block → 连续 logical）。
**bit-exact 证明（硬门）**：`torch.equal(prepack_a_fast, prepack_a_slow)` 对 fp4/fp8、三 shape 全 True（`R17_PROBE_OK`）——是**布局置换，不是数值改变**。旧冗余 kernel 保留为 `fused_gemm_slow` / `prepack_*_a_slow` 仅用于测「before」。

**before→after（standalone interleave，含启动+同步，`intlv_slow`→`intlv_fast`，r=⅛，us）：**

| shape M | slow | fast | | **in-stream 真实残余**（cd_full − cd_gemmonly） |
|--|--|--|--|--|
| down 8192 | 231.2 | 54.1 | **4.3×** | 8.6us |
| qkv 8192  | 110.9 | 64.1 | 1.7× | 19.9us |
| gate 8192 | 91.3 | 59.3 | 1.5× | 8.8us |
| down 2048 | 91.4 | 47.3 | 1.9× | 10.9us |
| qkv 2048  | 54.8 | 43.9 | 1.2× | 10.0us |
| gate 2048 | 56.3 | 45.4 | 1.2× | 10.2us |

> standalone 数被**固定的 2×（启动+`cudaStreamSynchronize`）开销**主导（~40–60us 地板，随 size 近乎不变）——这是测量 artifact。
> **真实的 per-call 残余**是 fused_gemm（内部同流、无中间 sync）相对 GEMM-only 天花板的差：**~9–20us**。这才是 co-design 采纳 stock collective 后**真正剩下**的激活-SF 代价。

### 3. Workspace → 缓存/预分配，跳过 per-call `can_implement`
`run_with_cached_ws`：持久化（function-local static）workspace tensor，只增长不每次 alloc/free；`can_implement` 只在首次跑。
`initialize` 仍每次跑（TMA 描述符依赖 per-call 指针，无法省）——但去掉了每次 device alloc + host 校验。此项为 host 小税（round-16 已指出相对 scatter 次要），折进下面的 cd_full。

### 产物（wrapper）
`codesign_fused_ext.cu` 新增：高效 `cd_fill_sf_block_kernel`；`interleave_sfa/sfb`（EFFICIENT 模板）；`prepack_sf_fp4_a/fp8_a`（+`_slow` 对照）；
`fused_gemm`（高效内部 interleave + 缓存 ws，= co-design 真实 per-call）；`fused_gemm_prepacked`（activation SF 预交错 = GEMM-only 天花板）；`fused_gemm_slow`（旧冗余，仅测 before）；`run_with_cached_ws`。**所有 .hpp mainloop 零改动。**

---

## Part 2 — 公平 A/B 重测（`r17_lat.csv`，24 cell，warm，median-of-50，GPU 2 独占同会话）

两核同数据同 GPU；co-design 从**和出厂核同一个** quantizer 输出（natural-layout `SFAl/SFAh`）起步，各付各的真实 per-call 代价：
- **SHIP** `b1_fused_tma_dual`（round-12 生产：SFILV0+SFASYNC+KPIPE per-shape）：原生吃 natural SF，**零 reformat**。
- **cd_full** = co-design 真实 per-call：高效 activation-SF interleave + 缓存 ws + 双格式 GEMM。
- **cd_gemmonly** = 天花板：activation SF 预交错后只计 GEMM（若把 interleave 折进 quant/pack 可达）。
- **cd_slow** = 旧冗余 interleave（= round-14/16 wrapper，before 数）。
- excess vs `t_ideal(r)=t_fp4(1−r)+t_fp8·r`（同会话 CUTLASS pure-presf 端点）。

**生产点 r = 1/8（关键）：**

| shape | M | cos | cd_full | cd_gemmonly | cd_slow | ship | **cd_full/ship** | cd_gemm/ship | cd_slow/ship |
|--|--|--|--|--|--|--|--|--|--|
| qkv  | 2048 | 0.999999 | 72.08  | 62.06  | 72.27  | 57.02  | **1.26** | 1.09 | 1.27 |
| qkv  | 8192 | 0.999999 | 149.70 | 129.82 | 168.66 | 133.94 | **1.12** | 0.97 | 1.26 |
| down | 2048 | 0.999999 | 132.61 | 121.73 | 162.18 | 126.43 | **1.05** | 0.96 | 1.28 |
| down | 8192 | 0.999999 | 300.02 | 291.42 | 446.38 | 314.80 | **0.95** ✅ | 0.93 | 1.42 |
| gate | 2048 | 0.999999 | 128.26 | 118.05 | 128.13 | 124.32 | **1.03** | 0.95 | 1.03 |
| gate | 8192 | 0.999999 | 341.84 | 333.01 | 357.60 | 362.69 | **0.94** ✅ | 0.92 | 0.99 |

**r = 1/2：**

| shape | M | cd_full | cd_gemmonly | ship | cd_full/ship | cd_gemm/ship |
|--|--|--|--|--|--|--|
| qkv  | 8192 | 166.37 | 151.22 | 158.75 | 1.05 | 0.95 |
| down | 8192 | 383.15 | 357.04 | 410.13 | **0.93** ✅ | 0.87 |
| gate | 8192 | 426.18 | 403.10 | 450.62 | **0.95** ✅ | 0.89 |
| qkv  | 2048 | 79.78  | 69.54  | 65.33  | 1.22 | 1.06 |
| down | 2048 | 160.05 | 148.51 | 160.18 | 1.00 | 0.93 |
| gate | 2048 | 149.39 | 138.45 | 150.50 | 0.99 | 0.92 |

**r = 0（退化纯 NVFP4，co-design 走 single_nvfp4 PoC 核）：** cd_full/ship = qkv 0.94/0.94、down 0.92/0.79、gate 0.85/0.83 ——全赢（但非生产点）。
**r = 1（纯 MXFP8，K_low=0）：** co-design 无暴露路径，记 n/a（出厂核覆盖，excess −12%..+22%）。

### 读数
1. **核心修复生效**：cd_slow→cd_full（旧冗余→高效 interleave）在大 shape 砍掉巨量：**down·8192 446→300us（−33%）**、qkv·8192 169→150（−11%）、down·2048 162→133（−18%）。cd_slow/ship 的 down·8192 = **1.42×**，**精确复现 round-15 的 1.42×**（证明 round-15 的慢就是这个冗余 scatter）；修掉后 cd_full/ship = **0.95×**。**round-16 预测的翻盘兑现。**
2. **cd_full（公平真实 per-call）**：**赢** down·8192（0.95×）、gate·8192（0.94×）；**≈持平** down·2048（1.05）、gate·2048（1.03）；**输** qkv·8192（1.12）、qkv·2048（1.26）。
3. **cd_gemmonly（天花板）**：6 个 shape **赢 5 个**（0.92–0.97×），只输 qkv·2048（1.09×）。→ 若把 activation-SF interleave 折进 quant/pack 步，co-design 近乎全胜。
4. **残余 per-call interleave（~9–20us）是唯一还把 cd_full 顶过 ship 的东西**，且只在 GEMM 小的 shape（qkv、M2048）吃亏（interleave 占比高）；大 GEMM（down/gate·8192）里它被摊薄，co-design 净赢。

---

## Part 3 — 残余 activation-SF 代价：诚实边界

co-design 采纳 stock SM120 block-scaled collective，**硬件 `mma.sync...block_scale` 要求交错 atom 布局的 SF**（这正是它 alu/tc≈1、GEMM 更快的来源）。出厂手写核用 natural-layout + 软件广播（alu/tc≈3.5，GEMM 慢但零 reformat）。这是**设计权衡**，不是 bug：
- 权重 SF 静态 → 已摊销到 load 时**零 per-call**。
- 激活 SF 动态 → 本轮把它的 per-call 代价从 16/32× 冗余 scatter（66–246us）**降到单遍（真实残余 ~9–20us in-stream）**。
- **能否再降到零**：只有把交错**折进量化/pack 核**（量化已遍历每个元素产出 SF，可直接按交错布局 emit）。但本 A/B 要求两核**同一输入**（natural SF，出厂核原生消费）——真要折进 quant 就得让量化核**额外**产一份交错布局（或量化两次），对出厂核不公平/有额外成本。故本轮把它如实记为 co-design 的**最小诚实 per-call 残余**，并用 `cd_gemmonly` 给出折进后的天花板。
- 残余是**单遍合并读 + 1-byte 散写**（输出偏移按 atom 布局散列，无法平凡向量化写）。进一步优化方向：预算 per-row 偏移表 / 向量化散写 / 融进 pack，可把 qkv 也翻过来——留作 future work。

---

## Part 4 — 判定

| 判据 | 目标 | 实测 | 判定 |
|--|--|--|--|
| buffer bit-exact（高效 vs 冗余 interleave） | 完全相同 | `torch.equal` 全 True（`R17_PROBE_OK`） | ✔ |
| 双格式 cosine（公平化后重门） | =1.0 | 0.999999（24/24，含 r=0 PoC） | ✔ |
| 权重-SF per-call 代价 | 0 | load 时一次（caller），确认 | ✔ |
| 激活-SF per-call 代价 | 最小化 | 16/32× 冗余 → 单遍；真实残余 ~9–20us（cd_full−cd_gemmonly） | ✔ |
| workspace alloc/can_implement | 缓存/跳过 | 持久 ws + can_implement 仅首次；initialize 仍每次（TMA 必需） | ✔ |
| **延迟 ≤ 出厂核（ship 条件）** | ≤1.0× | **down·8192 0.95×、gate·8192 0.94×（赢）**；down/gate·2048 ≈1.0；qkv 1.12–1.26（输） | **◑ 按 shape** |

### **SHIP / NO-SHIP 判定：条件 SHIP（按 shape 分派）**
公平化 wrapper 后，co-design 的**真实 per-call 延迟已在大 GEMM 预填充 shape 上快过出厂核**（down·8192 −5%、gate·8192 −6%，r=⅛；r=½ 更大 −7%/−5%），
并在 GEMM-only 天花板下**赢 5/6 shape**。这与 round-15「全盘 NO-SHIP、慢 30–42%」是**定性翻转**，直接来自拿掉 round-16 诊断的冗余 scatter。
但残余 ~9–20us 的 per-call 激活-SF interleave 仍让 **GEMM 小的 shape（qkv 全部、M2048）** 略慢，这些 shape 出厂核最优。

### 按 shape 分派建议（生产 r=⅛）
| shape / M | 推荐 kernel | 理由 |
|--|--|--|
| **down · M8192** | **co-design** `fused_gemm` | 0.95× ship（r=½ 0.93×），大 GEMM 摊薄残余 interleave |
| **gate · M8192** | **co-design** `fused_gemm` | 0.94× ship（r=½ 0.95×） |
| qkv · M8192 | 出厂 `b1_tma_ext` | cd_full 1.12×（GEMM-only 0.97× 但残余 interleave 顶过去） |
| qkv · M2048 | 出厂 `b1_tma_ext` | 1.26×（小 GEMM，残余占比最高） |
| down · M2048 | 出厂（≈持平 1.05×） | 边际，默认出厂 |
| gate · M2048 | 出厂（≈持平 1.03×） | 边际，默认出厂 |

> **若后续把 activation-SF interleave 折进量化/pack 核**（消掉残余），cd_gemmonly 表明 co-design 会在 **5/6 shape 全面领先**（0.92–0.97×），届时可对除 qkv·2048 外全部切 co-design。这是把本轮 GEMM 优势完全兑现的下一步。

---

## Part 5 — 对 round-15 / round-16 的收口（诚实双向）

| 先前结论 | 本轮修正 |
|--|--|
| round-15：「co-design 端到端慢 30–42%，全盘 NO-SHIP，相位屏障串行是症结」 | 慢的**主因是 wrapper 冗余 scatter**（round-16 已改正）；拿掉后 **down/gate·8192 反快 5–6%**。相位串行只是个位数 pp 小税。 |
| round-16：「GEMM 本体已快 13–18%，公平化后预期翻盘 SHIP」 | **兑现**：cd_gemmonly 赢 5/6（0.92–0.97×）、cd_full 赢大 shape；但**残余 ~9–20us interleave 让小 shape 仍输** → 结论是**按-shape 条件 SHIP**，不是无条件 SHIP。 |
| 「higher-TC-but-not-faster」 | 更准确：**higher-TC-and-faster-on-big-GEMM-shapes；小 shape 受限于不可零化的 per-call SF-interleave 残余**。 |

**净判定**：co-design **上位用于 down·8192 / gate·8192**（大 M 预填充的两条重 GEMM），其余 shape 保持出厂 `b1_tma_ext`。GEMM 优势已证并在大 shape 兑现为延迟优势；要全面取代出厂核需把激活-SF interleave 折进量化步（future work）。

---

## 产物
- **wrapper/pack 改动（零 mainloop 改动）**：`codesign_fused_ext.cu`（高效 `cd_fill_sf_block_kernel` + `interleave_sfa/sfb` + `prepack_sf_*_a[/slow]` + `fused_gemm`(高效)/`fused_gemm_prepacked`(天花板)/`fused_gemm_slow`(before) + `run_with_cached_ws` workspace 缓存）。
- **公平 A/B harness**：`r17_lat.py`（buffer bit-exact 探针 + 24-cell cosine 门 + cd_full/cd_gemmonly/cd_slow/ship 四路计时 + before→after interleave）。
- 数据：`r17_lat.csv`（24 cell，全 cos=0.999999，含四路延迟 + 双向 excess + interleave before/after）；日志 `logs/_r17_lat.log`（`R17_PROBE_OK` + `R17_LAT_ALL_COSINE_PASS` + `R17_LAT_DONE`）。
- **未改动（保护）**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu` 等 mainloop 文件全部零改动。
- `candidates.jsonl`：+3 条（高效 interleave bit-exact + before→after；公平 cd_full A/B 翻盘；GEMM-only 天花板 5/6 赢 + 残余 interleave 边界）。GPU 2 已释放。
