# B1 优化第十三轮 — 报告（STAGE 1：co-design 重写 de-risk PoC — 整条 CUTLASS block-scaled 流水在我方 harness 上单格式 NVFP4 是否达到 CUTLASS TC）

> 全程 remote-only（10.208.130.90, luofan-plaquant 建/跑, luofan-ncu 跑 ncu, RTX PRO 6000 Blackwell SM120）。
> 抗中断：自包含 detached 脚本 + 从磁盘收割。
> 正确性 gate：cosine=1.000000（6/6 shape×M 全通过，硬门）。
> 本轮 = 项目负责人裁定的「尝试 co-design 重写、追 CUTLASS TC 占空比平价」的 **STAGE 1（去风险 PoC）**：
> 先在 **单格式 NVFP4** 上证明「整条 co-design」能在我方数据契约下达到 CUTLASS 的 **alu/tc≈1 + ACTIVE 68-86%**，
> 作为是否继续双格式共享累加器缝（round-14）的闸门。**不碰出厂 `b1_tma_ext.cu`。**
> **本轮结论（干净判正）：STAGE 1 通过。整条 co-design 在 NVFP4 上达到 CUTLASS TC（alu/tc 0.80-1.26，ACTIVE 59-87%，
> 与 CUTLASS 端点逐指标相同），cosine=1.0；且 Path A 的共享累加器前提（NVFP4 与 MXFP8 累加器 fragment 同构）在编译期证实。→ round-14 继续缝 MXFP8。**

---

## Part 0 — 先验（为什么这轮这么做；round-10→12 的决定性发现）

round-10→12 的收敛结论（跨三轮一致，ncu-grounded）：~20-30pp 的 ACTIVE 差距由 **ALU 发射密度**造成
（我方 alu/tc≈3.5 vs CUTLASS≈1），而这是我方**手写单-warpgroup 双格式 SF-广播 consumer 的内在属性**，
**无法靠零敲碎补 CUTLASS 的单点机制修复**：
- round-10 warp 重组：无帮助；
- round-11 KPIPE register-pipeline hoist：alu/tc 不降反升（3.5→4.0）；B-x4 ldmatrix：LSU 非瓶颈；
- round-12 把 CUTLASS 交错 SF atom 布局嫁接到我方 TV：alu/tc 更糟（3.5→4.3），TCact 暴跌 27pp、延迟 2.5-3×。

**教训**：CUTLASS 的低 alu/tc 是其**整条 co-designed 流水**的性质（warp-specialized TMA producer A/B/SF 同 mbarrier →
交错 SF atom smem → UniversalCopy SF-consumer → cooperative 2-math-warpgroup + **硬件 block-scaled MMA 直接吃 SF**），
**只作为整体成立，不能拆片移植**。本轮 STAGE 1 就是检验「整体采纳」到底能不能在我方 harness/数据契约上兑现 CUTLASS TC。

### 新增参考源（项目负责人指定）：DeepSeek DeepGEMM / DeepInfer（remote `/home/luofan/DeepGEMM`,`/DeepInfer`，已 clone）
读完 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` + `ptx/tcgen05.cuh` + `mma/sm100.cuh`：
- **DeepGEMM 的低 alu/tc 100% 来自「硬件 block-scaled MMA 直接从 TMEM 吃 SF」**：`tcgen05.mma....kind::mxf8f6f4.block_scale [acc],A,B,idesc,[SFA],[SFB],p`——
  SF 作为 **TMEM 地址操作数**喂给张量核，**SF 全程不进寄存器、不产生任何 per-MMA ALU 指令**。SF 经 TMA→smem（独立 barrier）→
  UTCCP(`tcgen05.cp`) smem→TMEM（**每 SF-group 一次**，不是 per-MMA）；布局 massage 甩给**专用 transposer warp**（离开 MMA warp 指令流）；
  所有 SF-id/column 索引是 `#pragma unroll` 下的编译期常量。
- **关键注意**：DeepGEMM **只支持 SM100（B200, tcgen05/TMEM）**，**全仓无 SM120 kernel**（SM120 无 tcgen05/TMEM）。所以不能逐行移植。
  但**原理可移植且与我们本轮直接呼应**：低 alu/tc = 让**硬件 block-scaled MMA 吃 SF**，而不是软件把 SF 广播乘进累加器。
- **对我们的直接意义**：这正是 CUTLASS SM120 block-scaled collective 做的事——它用 SM120 的 warp-level `mma.sync...block_scale`
  原子（SF 以固定 fragment 布局从寄存器喂给硬件），所以 alu/tc≈1。**我方手写 consumer 的 alu/tc=3.5 正是因为做了软件 SF 广播/乘**。
  → 这从第三方独立实现佐证了 round-11/12 的机理定位，也解释了为什么 STAGE 1「整体采纳 CUTLASS collective」必然继承低 alu/tc。

---

## Part 1 — 构造路径抉择：Path A vs Path B（搜 + 原型 + 判定）

项目负责人给了两条路。**本轮选 Path A，并在编译期/源码两级证明其可行**：

### Path A（kernel glue 层双 CollectiveMma 共享累加器）—— **选定，可行**
re-examine round-7「stock CollectiveMma 表达不了双格式」的判定——**在 KERNEL glue 层（而非单个 CollectiveMma 内）**重审：
- **证据①（源码）**：SM120 dense block-scaled cooperative 的 schedule tag `KernelTmaWarpSpecializedCooperativeBlockScaledSm120`
  继承自 `KernelTmaWarpSpecializedCooperative`，故实际被 **SM90 cooperative GemmUniversal glue**
  （`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`）驱动。该 glue 的 consumer 循环：
  ```
  auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));   // line 787，glue 拥有累加器
  collective_mainloop.mma(..., accumulators, ...);                            // line 790/802，累加进 glue 的 fragment
  collective_mainloop.mma_tail(...);                                          // line 814
  collective_epilogue.store(..., accumulators, tiled_mma, ...);               // line 852，epilogue 消费同一 fragment
  ```
  即**累加器属于 glue、传入 mma()、再传给 epilogue**。round-14 只需在 line 790 之后**再调一次第二个（MXFP8）collective 的
  `mma(..., accumulators, ...)`**（K-split：Phase-1 NVFP4 K_low，Phase-2 MXFP8 K_high），然后走一条共享 epilogue。
  **这个「多次 mma() 进同一 accumulators」的结构 stock glue 本身已在用**（line 790 vs 802 是 `if constexpr(IsSm120Family)` 两分支，
  每 tile 一次，进同一 fragment）——round-14 的改动是该模式的自然延伸，不是另造结构。
- **证据②（编译期，本轮硬证）**：round-14 共享累加器的**前提**是两个 collective 的 glue-owned 累加器 fragment 必须**同构**。
  本轮在 `codesign_nvfp4_ext.cu` 里**同时实例化 NVFP4 与 MXFP8 两个 stock CollectiveMma**，对各自
  `partition_fragment_C(TiledMma, take<0,2>(TileShape))` 做 **compile-time `static_assert`**：
  ```
  static_assert(is_same_v<NV::AccumElement, MX::AccumElement>);           // 都是 float
  static_assert(size(NV::AccumFragLayout) == size(MX::AccumFragLayout));  // 都是 64
  static_assert(is_same_v<shape(NV::Frag), shape(MX::Frag)>);             // shape 相同
  ```
  **build 成功 = 三条 static_assert 全部成立**（见 Part 2，运行期 report 打出 NVFP4/MXFP8 accum frag size 都 =64，
  element=float，TileShape MN 都 =(128,128)）。→ **round-14 的共享累加器在类型上安全**，Path A 的共享缝可表达。

### Path B（手写 CuTe co-designed mainloop）—— 本轮不需要，列为回退
Path A 编译期证实可行后，Path B（从头手写一条忠实复刻 CUTLASS co-design 的 CuTe mainloop）**不是本轮必需**。
保留为回退：仅当 round-14 真正缝 MXFP8 时发现两个 stock CollectiveMma 无法在同一 forked glue 里共存
（如 smem/pipeline/named-barrier 资源冲突不可调和），才回退 Path B。**本轮无证据指向需要 Path B。**

---

## Part 2 — STAGE-1 构造（`codesign_nvfp4_ext.cu`，新文件，整条 co-design 采纳）

新文件（**不碰出厂 `b1_tma_ext.cu`；pure_presf_ext.cu 等受保护文件零改动**）：
- 用 CollectiveBuilder 产出 **stock SM120 block-scaled NVFP4 CollectiveMma**（= 整条 co-design：warp-specialized TMA
  producer A/B/SF 同 mbarrier、交错 SF atom smem、UniversalCopy SF-consumer、cooperative atom `Shape<4,2,1>`、
  **硬件 block-scaled `mma.sync` 原子直接吃 SF**）+ stock cooperative epilogue + stock SM90-cooperative GemmUniversal glue
  （拥有累加器）+ GemmUniversalAdapter 发射。config 逐字来自 ex79a / pure_presf_ext.cu。
- **消费我方真实数据契约**（与 `mixed_gemm.py` oracle 完全一致）：NVFP4 E2M1 codes、block-16 UE4M3 scale、2×FP4/byte、
  A=(M,K)/B=(N,K) packed、SF 逻辑 (R,K/16) 由 `codesign_fill_sf_kernel` scatter 进 CUTLASS 交错 atom buffer
  （`Sm1xxBlkScaledConfig::tile_atom_to_shape_SF*`）。weight SF 一次性 prepack（`prepack_sf_fp4_b`）。
- 新增 **Path-A 共享累加器编译期兼容检查**（Part 1 证据②），运行期 `compat_report()` 打印。
- pybind：`codesign_nvfp4_gemm`、`prepack_sf_fp4_b`、`compat_report`。

### 正确性（硬门，6/6，`logs/_codesign_gate.log`）
`codesign_check.py`：3 shape（qkv N2048/Kt2048、down N2048/Kt8192、gate N8192/Kt2048）× M{2048,8192}，
对 NVFP4 oracle 全部 **cosine=0.999999（=1.000000）**，`CODESIGN_GATE_ALL_PASS`。
`compat_report()` 输出：`PATH_A_ACCUM_COMPAT=PASS`，NVFP4/MXFP8 accum frag size 均 64、element=float、TileShape MN (128,128)。

---

## Part 3 — ncu TC 证明（round-11 imix 方法，`logs/tcCODESIGN/`，`agg_codesign.py`）

每管线指令数 + TC 两 rollup，codesign（CD，我方新文件）vs CUTLASS 纯-NVFP4 端点（CUT，pure_presf），同 shape×M。

| cell | CD TCact / TCela / **alu/tc** / lsu/tc / dur_us | CUT TCact / TCela / **alu/tc** / lsu/tc / dur_us | 我方手写 consumer（round-11/12 对照） |
|--|--|--|--|
| qkv M8192 | 69.9 / 61.9 / **1.16** / 0.78 / 85.4 | 69.7 / 61.7 / **1.16** / 0.78 / 83.6 | TCact 42-46, alu/tc 3.6 |
| qkv M2048 | 59.0 / 38.5 / **1.26** / 0.79 / 32.0 | 59.1 / 39.6 / **1.26** / 0.79 / 31.2 | — |
| down M8192 | 86.6 / 78.5 / **0.80** / 0.67 / 261.3 | 86.7 / 78.5 / **0.80** / 0.67 / 262.6 | TCact 54-60, alu/tc 3.0-3.5 |
| down M2048 | 81.1 / 54.9 / **0.83** / 0.67 / 94.8 | 81.3 / 54.6 / **0.83** / 0.67 / 95.3 | — |
| gate M8192 | 73.1 / 71.3 / **1.13** / 0.77 / 280.4 | 73.0 / 71.4 / **1.13** / 0.77 / 277.7 | TCact 48-53, alu/tc 3.5-4.3 |
| gate M2048 | 70.1 / 62.6 / **1.15** / 0.78 / 82.7 | 70.2 / 62.2 / **1.15** / 0.78 / 83.4 | — |

**发现（证据，非猜测）：**
1. **codesign 与 CUTLASS 端点逐指标相同**（TCact Δ<0.2pp，alu/tc 完全一致，dur Δ<1%）——意料之中：codesign **就是**整条 stock collective，
   只是驱动在我方 harness、吃我方数据契约。这正是 STAGE 1 要证的：**我方数据契约（packed A/B/SF + SF scatter）不引入任何 TC 退化**。
2. **alu/tc = 0.80-1.26 ≈ 1**（目标达成）。对照我方手写 consumer 的 3.0-4.3 —— **降了 ~3-4×**。
3. **ACTIVE（TCact）= 59-87%**：M8192 三 shape 全进 **70-87%**（qkv 69.9 / gate 73.1 / down 86.6），落在 CUTLASS 68-86% 带内；
   M2048 qkv/gate 稍低（59-70%）是小-M 尾波效应，且 **CD==CUT**（是该 shape/collective 的性质，不是我方的差距）。
   对照我方手写 consumer 的 ACTIVE 42-59% —— **抬了 ~20-30pp，正好填平 round-10→12 的 gap**。
4. lsu/tc 0.67-0.79（CUTLASS 水平），regs 168。

---

## Part 4 — STAGE-1 判定 + round-14 计划

### STAGE-1 闸门：**通过（co-design 在 NVFP4 上达到 CUTLASS TC）**

| 判据 | 目标 | 实测（codesign，我方 harness + 数据契约） | 判定 |
|--|--|--|--|
| cosine | =1.000000 | 0.999999（6/6） | ✔ |
| alu/tc | ≈1 | 0.80-1.26 | ✔（vs 手写 3.0-4.3） |
| ACTIVE | 68-86% | M8192 69.9-86.6（带内）；M2048 59-81（=CUTLASS） | ✔（M8192 全带内） |
| = CUTLASS 端点 | 逐指标 | TCact Δ<0.2pp、alu/tc 一致、dur Δ<1% | ✔ |
| Path-A 共享累加器前提 | 两格式 accum fragment 同构 | 编译期 static_assert 成立（size64/float/128×128 均同） | ✔ |

**→ co-design 方向未被 refute，反而在 NVFP4 上兑现了 CUTLASS TC**。round-12 证的是「拆片移植」死路；本轮证的是「整体采纳」通路。
二者不矛盾、互补：**唯一能让我方达到 alu/tc≈1 的办法是采纳整条 stock collective（含硬件 block-scaled MMA 吃 SF），而非在手写 consumer 上打补丁**。
DeepGEMM（独立第三方）从 SM100 侧佐证同一机理。

### round-14 计划（Path A，缝 MXFP8 + 共享累加器）
1. **fork** `sm90_gemm_tma_warpspecialized_cooperative.hpp` 到我方 header（`codesign_sm120_kernel.hpp`），在 consumer 循环的
   `mma(..., accumulators, ...)` 之后**再调一次 MXFP8 collective 的 `mma(..., accumulators, ...)`**（Phase-1 NVFP4 K_low → Phase-2 MXFP8 K_high，
   共享同一 `accumulators`，本轮已编译期证实同构），再走**一条共享 epilogue**（融合 `acc·s_x·s_w → fp16`）。
2. **资源共存**（本轮未验，round-14 首验）：两个 stock CollectiveMma 的 smem（A/B/SF ring）、mainloop pipeline、
   NamedBarrier/mbarrier 在同一 forked glue 里的共存/错峰（SMEM 预算、stage 数）。这是 round-14 的真正风险点
   （若不可调和 → 回退 Path B 手写一条 co-designed mainloop）。
3. **gate**：双格式融合 cosine=1.0 + ncu alu/tc≈1 + ACTIVE 对齐 CUTLASS 的 `t_ideal(r)=t_fp4·(1-r)+t_fp8·r` 包络。
4. 与出厂 `b1_tma_ext.cu`（round-12 SFILV=0+KPIPE+SFASYNC，bit-exact、延迟领先）做 A/B——**co-design 版只有在把 TC 优势
   转成端到端延迟优势时才上位**；否则出厂 kernel 保持 fallback。

### 诚实边界（本轮没证的）
- STAGE 1 是**单格式**，**未**真正跑双格式共享累加器的运行时（那是 round-14）。本轮对 Path A 的支撑是
  「源码结构 + 编译期 fragment 同构」两级证据，不是运行时双 mma() 的 cosine。
- `codesign_nvfp4_ext.cu` 走 GemmUniversalAdapter（stock glue），**未**在本轮 fork glue——fork 是 round-14 改 glue 时才必需的动作；
  本轮刻意不 over-build 进双格式缝（项目负责人明确要求）。

---

## 产物
- **新文件 `codesign_nvfp4_ext.cu`**（整条 stock SM120 block-scaled NVFP4 collective + 我方数据契约 + Path-A 共享累加器编译期兼容检查）
  + `codesign_check.py`（loader + cosine gate + compat report）+ `prof_codesign.py`（ncu 单实例 runner）+ `run_tc_codesign.sh`
  （imix ncu，CD vs CUT）+ `agg_codesign.py`（聚合）。
- 正确性：`logs/_codesign_gate.log`（`CODESIGN_GATE_ALL_PASS` 6/6 cosine=0.999999 + `PATH_A_ACCUM_COMPAT=PASS`）。
- TC 证明：`logs/tcCODESIGN/`（12 captures：6 CD + 6 CUT），`agg_codesign.py` 输出（CD==CUT，alu/tc 0.80-1.26，ACTIVE 59-87%）。
- 参考：DeepGEMM/DeepInfer SF-consume 机理分析（Part 0，硬件 block-scaled MMA 吃 SF = 低 alu/tc 之源，佐证 round-11/12 定位）。
- `candidates.jsonl`：+2 条（STAGE-1 co-design TC 达标；Path-A 共享累加器编译期同构）。
- **未改动（保护）**：`b1_tma_ext.cu` / `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` /
  `mixed_gemm.py` / `pure_presf_ext.cu`。GPU 已释放。
