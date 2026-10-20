# B1 优化第十轮 — 报告（Lever: ping-pong / cooperative 双 math warp-group）

> 全程 remote-only（10.208.130.90, luofan-plaquant / luofan-ncu, RTX PRO 6000 Blackwell SM120）。
> 抗中断：自包含 detached 脚本 + 从磁盘收割。
> 正确性 gate：cosine=1.000000（pp1 全点通过，硬门）。
> 收敛判据（项目负责人）：**TC 占空比**两 rollup 同时逼近 CUTLASS 同 shape；mainloop 段不下陷。
> 本轮成功判据（项目负责人加码）：ncu 上 (1) `wait` stall 显著降；(2) ACTIVE ~55%→CUTLASS ~74%；
> (3) ELAPSED 更贴 ACTIVE（段间空窗被填）。**分层**：cooperative 治段内 MMA-wait；pingpong 治段间 epilogue 重叠。

---

## Part 0 — 先验：前人怎么做（CUTLASS 源码确认，借鉴了谁的什么）

逐条读了 remote `third_party/cutlass`（4.4.2）+ KernelWiki + DeepSeek（CUTLASS issue #2181 权威问答）：

- **Cooperative（本轮实现的 pp1 借鉴对象）** = `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`
  + `sm120_mma_builder.inl` 的 `AtomLayoutMNK = Shape<_4,_2,_1>`（协作时 8 warp = 两个 4-warp math
  warp-group，沿 **M 拆同一个 output tile** 成上下两半）。关键证据（逐行）：
  - line 359「Consumer0 和 Consumer1 collaborate on the **same tile**」；line 362-363 两个 Consumer 角色。
  - line 451-456：两个 Consumer **都是同一条 mainloop pipeline 的 consumer**，`num_consumers = NumMMAThreads`
    （= 两组全部 256 线程都 release/free）。**无 math-order barrier**（issue #2181 原文确认）。
  - line 804-815：每个 consumer warp-group 分配**自己的 accumulator** + 调**自己的 `collective_mainloop.mma()`**。
  → 这正是我方 pp1：两组 4-warp（各 `Shape<2,2,1>`）拆 128×128 的两个 64×128 M-半、各自 FP32 累加器、
    共享 producer ring、**全员 free**、**无序 barrier**。producer 一字未改（整块 128×128 A/B/SF）→ 数据/SF 位级一致。
- **Pingpong（CUTLASS 另一标签）** = `sm90_gemm_tma_warpspecialized_pingpong.hpp`：两组各跑**不同 tile**，
  用 `MathWarpGroupOrderBarrier = OrderedSequenceBarrier<StagesPerMathWarpGroup=2, NumMmaWarpGroups>`
  （line 156-158）强制两组**错相位**，一组 epilogue(CUDA core) 盖另一组 mainloop(TC)。issue #2181 原文：
  「ping-pong 两组在不同 output tile，一组 epilogue 被另一组 mma 掩盖，顺序由 math order barrier 严格保证」。
  → 不同 tile + 序 barrier，与我方「单 tile 挂在一个 ring stage 上」的 producer 结构**不兼容**（见 Part 5）。
- **SM120 cooperative 还带的另外两味**（非本轮 lever，round-11 候选，已确认在源码）：
  (a) **非对称 DMA**：两个 producer warp `LoadMK`(A) / `LoadNK`(B) 跑**两条独立 pipeline**（line 365-369,
  439-459）；我方是**单 producer warp 串行搬 A+B** 一条 ring。(b) `mma()` 里的深度寄存器 K-pipe（= 我方 round-8 KPIPE）。

**结论**：我方「单一 8-warp `Shape<2,4,1>` TiledMMA」唯一能忠实对标 CUTLASS 的是 **cooperative**
（pingpong 要不同 tile，见 Part 5 不可行性）。所以本轮 pp1 = CUTLASS cooperative 字面搬运。

---

## Part 1 — CUTLASS 同 shape TC 端点（收敛靶，沿用 round-9 Part 1 实测，CUTLASS 参考稳定）

| shape·M | fp4 TCela/act | fp8 TCela/act |
|--|--|--|
| qkv·8192 | 61.1 / 68.5 | 67.6 / 75.1 |
| down·8192 | 78.1 / 86.2 | 75.2 / 83.7 |
| gate·8192 | 70.5 / 72.0 | 75.0 / 76.8 |

---

## Part 2 — pp1 实现（cooperative M-split）+ 位级正确性

`b1_tma_ext.cu` 新增模板开关 `int PINGPONG`（默认 0 = round-9 字节等价、受保护）。PINGPONG=1：
- consumer 256 线程按 `tid/128` 分两组；WG0=rows[0,64)，WG1=rows[64,128)。
- 每组一个 `make_tiled_mma(Atom, Layout<Shape<_2,_2,_1>>)`（4 warp，2M×2N，各 warp 32M×64N），
  各自 `partition_fragment_C(…,64×128)` 累加器 + 各自 epilogue 写自己的 M-半。
- A/SFA 用 `local_tile(…, (64,KCH), (wg,0))` 切 M-半（swizzle 布局随 view 偏移，地址正确）；B/SFB 读全 N。
- 两组消费**同一个 ring stage 的同一 chunk g**，全员 `mbar_arrive(free[s])`（NT_C）—— 与 CUTLASS
  `num_consumers=NumMMAThreads` 一致。producer 完全不动 → 数据/SF 位级一致，逐元素 fp4-then-fp8 chunk/kb
  的 FP32 累加顺序不变 → 结果位级一致。
- 曲线盘 pybind 参数 `pingpong`；curated 实例化 km2/st3/tile0/kpipe0 × sw{0,8} × persist{0,1} × sfa{0,1}（与 SFASYNC 正交可组合）。

**正确性（硬门）**：3 shape × M{2048,8192} × r{0,1/8,1/2,1} × sfa{0,1}，**pp1 全部 cosine=1.000000**
（`pingpong_check.py` → `logs/r10_pp_gate.log`，**PINGPONG_GATE_ALL_PASS，PASS=49 FAIL=0**）。

---

## Part 3 — TC 前后（pp0→pp1，ncu，`logs/tcPP/`，`agg_pp.py`；全 sfa=0 隔离 pp 变量，每点位级一致）

| shape·M·Kh | TCela 0→1 | **TCact 0→1** | **wait 0→1** | lsb 0→1 | ssb 0→1 | **regs 0→1** | warp% | dur(us) 0→1 |
|--|--|--|--|--|--|--|--|--|
| qkv·8192·0    | 36.4→34.6 | 40.7→**38.5** (−2.2) | 2.91→**2.56** | 2.50→2.54 | 0.22→0.17 | 139→**144** | 18.1 | 135→143 |
| qkv·8192·256  | 38.8→37.6 | 43.6→42.0 (−1.6) | 2.85→2.54 | 2.38→2.78 | 0.19→0.15 | 139→144 | 18.1 | 143→147 |
| qkv·8192·1024 | 44.6→43.9 | 50.2→49.0 (−1.2) | 2.84→2.55 | 1.57→1.80 | 0.14→0.13 | 139→144 | 18.2 | 167→173 |
| down·8192·0   | 47.8→49.1 | 52.8→**54.3** (+1.5) | 2.95→**2.56** | 1.49→1.36 | 0.25→0.18 | 139→144 | 18.5 | 409→410 |
| down·8192·1024| 51.8→52.2 | 57.4→57.5 (+0.1) | 2.92→2.56 | 1.11→1.09 | 0.20→0.15 | 139→144 | 18.6 | 424→429 |
| down·8192·4096| 53.6→53.6 | 59.2→59.1 (−0.1) | 2.85→2.55 | 1.03→1.00 | 0.22→0.19 | 139→144 | 18.6 | 546→554 |
| gate·8192·0   | 46.7→46.5 | 48.9→48.3 (−0.6) | 2.95→**2.65** | 2.22→1.95 | 0.18→0.14 | 143→**148** | 18.7 | 431→425 |
| gate·8192·256 | 48.9→47.9 | 50.2→49.9 (−0.3) | 2.92→2.62 | 1.88→1.73 | 0.15→0.11 | 143→148 | 18.7 | 449→469 |
| gate·8192·1024| 52.6→51.6 | 54.0→54.0 (0.0) | 2.85→2.58 | 1.47→1.31 | 0.11→0.11 | 143→148 | 18.7 | 555→574 |

（M2048 同趋势：wait 全降 ~0.3，TCact ±1，dur 中性-微亏；完整表见 `logs/tcPP`。）

**机理证据（诚实判读，对着项目负责人的 4 条判据逐条）**：
1. **`wait` 全 18 点从 2.85–2.95 降到 2.54–2.65（−0.3，≈10%）** —— 这是 cooperative 的**真实**作用：
   两组独立累加器 → 调度器多了一份去相关的 MMA 工作，填掉**部分**段内 MMA-依赖空窗。判据(1) ✔（方向对，但幅度小）。
2. **ACTIVE intensity 几乎不动**（qkv 微降 1–2pp，down 微升 0–1.5pp，gate 平）—— **没有**往 CUTLASS 74% 推。判据(2) ✘。
3. **ELAPSED 对 ACTIVE 的差不变**；且该差**最大在 M2048 非持久**（qkv2048 26.1/39.8 差 13.7，down2048 36.5/53.5 差 17）、
   **最小在持久 gate**（gate8192 46.7/48.9 差 **2.2**）—— 与「持久多 tile 有 epilogue 段间空窗」的假设**相反**（见 Part 5）。判据(3) ✘。
4. regs **139→144 / 143→148（+5）**，warp% 不变（18.x），`limit-Reg`/`limit-Smem` 仍 =1（仍 1 CTA/SM）——
   **拆两组没有缓解寄存器**（每线程累加器仍 64 个 FP32；两个 TiledMMA 对象反而多 +5 reg）。回答了项目负责人的问题：**占空比故事未变**。

**净判**：pp1 **位级正确、wait 真降 ~10%，但 ACTIVE 不涨、延迟中性-微亏**。说明：我方本就有 8 个独立 warp
（= 2 个 warp-group 的线程量），把它们**重组**成两个 TiledMMA 组只去相关了**一部分** MMA-依赖空窗（wait 降），
但 (a) 幅度被 +5 reg、两个 mma 对象的额外指令吃掉；(b) **残差主项不是 wait 能填的**（见 Part 4）。
→ **pp1 不是 TC-duty 的赢点，默认保持 OFF**，knob 留作可组合性研究。

---

## Part 4 — 残差主项定位（ncu 证据）+ 为什么 cooperative 不够

- pp1 后 `wait` 仍 **~2.55 地板**。这是 **MMA 数学固定延迟**；在 1 CTA/SM（warp% 18.6% = 9 warp）下，
  即便两组去相关，驻留 warp 总量**没增加**，填不满延迟。CUTLASS 在**同样 1 CTA/SM** 下拿到 74% ACTIVE，
  说明它靠的**不是**更多 warp，而是 **mainloop 指令流里 MMA 占比更高 / 发射更密**。
- 我方 mainloop 每个 kb 的 MMA 之间夹着**非-TC 指令**：`fp4_shift_A/B`（寄存器 shift）、SF 的 zip/partition、
  ldmatrix —— 这些 CUDA-core/MIO 指令占了 issue slot，直接压低 ACTIVE（TC 发一拍、ALU 一拍、再 TC）。
  这与「wait 降了但 ACTIVE 不涨」自洽：空窗被 ALU 指令而非更多 MMA 填了。
- `lsb`（long_scoreboard）在 sfa=0 下仍 1.0–2.5（SFASYNC=0 隔离态；round-9 已证 sfa=1 可把它压到 0.7–1.7），
  `ssb` 平（pp1 没制造新的 shared 等待，反而微降）—— 没有把一个 stall 换成另一个。

---

## Part 5 — 分层第二味（项目负责人加码）：pingpong 段间 epilogue 重叠 —— 证据表明对本 kernel 低价值

项目负责人的洞察：pingpong 两组错相位，WG0 做 tile N 的 epilogue(CUDA core) 时 WG1 做 tile N+1 的 mainloop(TC)，
让两类单元异步并行、填段间空窗。我方对此做了**证据评估**而非盲目实现，结论：**本 kernel 头很小，收益低**。

1. **段间空窗本身就很小**：判据(3) 的 ELAPSED-vs-ACTIVE 差，在**持久多-tile 的 gate**（最该有 epilogue 段间空窗的 shape）
   只有 **~2.2pp**（gate8192 46.7 vs 48.9）；大 gap 反而在 **M2048 非持久 single-tile**（qkv/down 差 14–17pp），
   那是**流水 fill**（每 CTA 只做一个 tile），**不是** epilogue，pingpong 的 epilogue 重叠**治不了 fill**。
2. **我方 epilogue 极轻**：`accum*s_x*s_w → __floats2half2_rn → half2 store`，**无** accumulator promotion
   （issue #2181 明确 cooperative/pingpong 的主要收益之一是盖 fp8 的周期性累加器 promotion —— 我方单 FP32 累加器全程，没有）。
   每线程 epilogue ≈ 64 元素的 乘-转-存，相对 mainloop 的几百条 MMA 是**个位数百分比**。
3. **结构不兼容 + 寄存器墙**：pingpong 要两组各跑**不同 tile**。若每个 4-warp 组独跑一个整 128×128 tile →
   每线程累加器 **翻倍到 128 个 FP32** → 必然 spill（现在 64 个 FP32 就已 139 reg）。退而求其次让每组跑
   **自己的 64×128 tile**（各 64 FP32，寄存器可行）是可做的，但需要 `OrderedSequenceBarrier` 序 barrier +
   ring 要容纳两个 in-flight tile —— 是比 cooperative 更大的重构，且按上面 (1)(2) 其**可填的段间空窗 ≤2pp**，
   投入产出比低。**本轮不实现，列为 round-11 低优先候选**，证据充分。

---

## Part 5.5 — skew 变体（pp2）尝试与诚实判负（为什么 chunk 粒度错相不行）

按项目负责人「让两组独立推进、错峰发射」的指引，先试了一个**自由-非对称 skew**（pp2）：只让滞后组 WG1 free
（free-count=NT_WG），想让 WG0 跑到 producer frontier、两组自然错开。**实测判负**：
- **死锁/NaN**：`pingpong_check.py` 下 pp2+sfa1 在 r=0.125 出 **cos=nan**；pp2 在 down r=0 sfa1 **spin-wait 死锁**
  （GPU 100%、日志 0 增长，已 kill 释放 GPU）。free-asymmetry 的 skew 不稳定（WG0 越过 producer frontier 读到被覆盖的 stage）。
- **即便不死锁也无用**：chunk 粒度的整数错位 = 偏移整数条相同操作 → **微观相位仍对齐**（两组在各自 chunk 内处于相同相对位置，
  MMA-wait 仍同时发生）。要真正去相位必须是**子-chunk / named-barrier** 粒度（即 CUTLASS pingpong 的 OrderedSequenceBarrier），
  而那又要不同 tile（Part 5 不兼容）。→ **pp2 已从 kernel 移除**（默认/shipped 只有 pp0 和位级安全的 pp1）。

---

## Part 6 — per-shape dispatch（round-10 更新）

**无变化**：pp1 不是 TC-duty 赢点（ACTIVE 不涨、延迟中性-微亏），默认 **PINGPONG=0**，沿用 round-9 dispatch。

| 形状 | PINGPONG | SFASYNC | KPIPE | scheduler | cfg | 依据 |
|--|--|--|--|--|--|--|
| 纯 fp4（r=0） | 0 | 1 | 0 | 非持久(qkv/down)/持久(gate) | t0 st3 | round-9 |
| qkv/down 混合（r>0） | 0 | 1 | 1 | 非持久 | t0 st3 sw0 | round-9 |
| gate 混合（r>0） | 0 | 1 | 1 | 持久(P1) | t0 st3 sw8 | round-9 |

PINGPONG=1 保留为可组合 knob（位级安全，与 sfasync/persist 正交），供 round-11 与新 lever 叠加实验。

---

## Part 7 — 收敛判定（TC 占空比，per shape；vs Part 1 CUTLASS 端点）

| shape | TC 现状（pp1 后，M8192 act） | CUTLASS 同 shape act | 判定 | 残差来源（ncu） |
|--|--|--|--|--|
| qkv | 38–49 | 68–75 | **未收敛**（pp1 微降） | ACTIVE intensity：wait 地板 2.55 + mainloop 非-TC 指令占比 |
| gate | 48–54 | 72–77 | **未收敛**（ELAPSED≈ACTIVE，纯半速发射，段间空窗已 ~2pp） | 同上（intensity，非段间空窗） |
| down | 54–59 | 84–86 | **未收敛**（ACTIVE 差最大 ~27pp） | 同上 + fp4 残余 long_scoreboard（sfa=1 可治） |

**全局结论**：cooperative pp1 **做对了、位级安全、`wait` 真降 ~10%**，但**未能抬 ACTIVE / 未改善延迟**——
因为我方本就有 2 个 warp-group 的线程量，重组只去相关了一部分 MMA 空窗（被 +5 reg 与非-TC 指令吃掉），
且**残差主项不是段内 wait、也不是段间 epilogue**（gate 段间空窗仅 ~2pp、epilogue 极轻），而是
**1 CTA/SM 下 mainloop 的「半速发射」—— MMA 之间夹着 fp4_shift / SF-zip / ldmatrix 等非-TC 指令，压低了 TC 发射密度**。

### round-11 下一杠杆（ncu-grounded，按预期收益排序）
1. **降 mainloop 非-TC 指令、抬 MMA 发射密度**（CODEGEN checklist D：把 `fp4_shift_A/B` 的寄存器 shift 外提/消除、
   SF zip/partition 预算到更少指令、ldmatrix 更宽）—— 直接攻 ACTIVE intensity（本轮定位的真残差）。可与 pp1/KPIPE 叠加。
2. **非对称双-producer DMA**（CUTLASS SM120 `LoadMK`/`LoadNK` 两条 pipeline）：我方单 producer warp 串行 TMA A+B
   可能是 producer 侧瓶颈；拆成两个 producer warp / 两条 ring 解耦 A、B 载入。结构改动中等。
3. **cooperative pp1 + KPIPE 叠加**（当前互斥）：两者都治 wait，叠加是否把 wait 从 2.55 再往下压、进而松动 ACTIVE —— 低成本验证。
4. （低优先）pingpong 不同-tile + `OrderedSequenceBarrier`，每组 64×128 tile 避开寄存器翻倍 —— 但 Part 5 证据示段间空窗 ≤2pp，ROI 低。

## 产物
- `b1_tma_ext.cu`：新增 `PINGPONG` 模板开关（默认 0，保护 round-9 字节等价）+ cooperative 双 warp-group consumer
  （两个 `Shape<2,2,1>` TiledMMA、M-split、各自累加器/epilogue、全员 free、无序 barrier）+ `pingpong` pybind 参数
  + curated 实例化。pp2(skew) 已尝试并移除（死锁/NaN，Part 5.5）。
- 正确性：`pingpong_check.py`（`logs/r10_pp_gate.log`，PINGPONG_GATE_ALL_PASS，49/0）。
- TC 前后：`run_tc_pingpong.sh` + `agg_pp.py`（`logs/tcPP/`：TC 两 rollup + wait + lsb + ssb + regs + occupancy limiter + warp% + dur）。
- 未改动（保护）：`b1_fused_ext.cu`/`fused_gemm_ext.cu`/`nvfp4_gemm_ext.cu`/`mxfp8_gemm_ext.cu`/`mixed_gemm.py`/`pure_presf_ext.cu`。
