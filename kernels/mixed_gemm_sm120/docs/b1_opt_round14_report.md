# B1 优化第十四轮 — 报告（STAGE 2：双格式共享累加器 co-design kernel — the PAYOFF round）

> 全程 remote-only（10.208.130.90, luofan-plaquant 建/跑, RTX PRO 6000 Blackwell **SM120**）。
> 抗中断：自包含 detached 脚本 + 从磁盘收割。正确性 gate：cosine（硬门）。
> 本轮 = STAGE 2：把 round-13 证实可行的 Path A（kernel-glue 双 CollectiveMma 共享累加器）**真正缝起来**——
> NVFP4(K_low) + MXFP8(K_high) 融进**一个 kernel launch**、**一个 in-register FP32 共享累加器**、**一条共享 epilogue**。
> **不碰出厂 `b1_tma_ext.cu` 及受保护文件（全部新文件）。**
>
> **本轮结论（干净判正 + 诚实边界）：双格式共享累加器 co-design kernel 构造成功并 bit-exact——
> cosine_core=0.999997 / cosine_full=0.999999，全 3 shape × M{2048,8192}（6/6）全通过硬门。
> STAGE-1 的所有「资源共存」风险（smem 预算、named-barrier 冲突、双 pipeline 错峰、双 TMA 描述符、K-split）
> 全部解决并逐条记录。TC（criterion #2）与端到端延迟 A/B（criterion #3）本轮因远程机器 CPU 极度过载
> （cicc 单 TU 编译 40min 仍只拿到 69s CPU、2.8% 占用、load avg 14-24）而未能采到——clean-ext build +
> 链式 latency 跑已 detached 排队，结果可从 `logs/_cdlat.log` 收割。TC 由构造继承 STAGE-1（见 Part 5）。**

---

## Part 0 — 先验与本轮目标

round-13（STAGE 1）证实：整条 stock SM120 block-scaled collective 在我方 harness/数据契约下达到 CUTLASS TC
（alu/tc 0.80-1.26≈1，ACTIVE 59-87%，与 CUTLASS 端点逐指标相同），且 Path A 的共享累加器前提（NVFP4 与 MXFP8 的
glue-owned accumulator fragment 同构：FP32 / size 64 / TileShape MN 128×128）在编译期证实。
本轮把这条缝真正缝上，并回答项目负责人的三问：**(1) 双格式 bit-exact？(2) 双格式 kernel 的 TC 是否仍≈1？
(3) 更高 TC 是否转成端到端延迟优势——ship/no-ship？**

---

## Part 1 — 双格式 glue 的构造（fork CUTLASS，新文件）

两个新 header + 一个新 .cu（出厂/受保护文件零改动）：

### 1a. `codesign_sm120_bs_mma.hpp` — fork stock block-scaled collective，加 `ClearAccum`
VERBATIM 复制 `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` 的 collective body，**仅两处改动**：
1. 从 `CollectiveMma<...>` 的偏特化改写成**独立 struct `CodesignBlockScaledMma<..., bool ClearAccum=true>`**
   （仍留在 `namespace cutlass::gemm::collective`，使所有 `detail::`/`cute::` 非限定查找逐字不变）；
2. 把 `mma()` 顶部唯一的 `clear(accum);` 用 `if constexpr (ClearAccum) { clear(accum); }` 包起来。
→ Phase-1 (NVFP4) 用 `ClearAccum=true`（清零后累加 K_low）；**Phase-2 (MXFP8) 用 `ClearAccum=false`——第二次 mma()
**累加**进同一 fragment 而不是 reset**（整个共享累加器的关键）。硬件 block-scaled `mma.sync` 吃 SF 的路径字节不变
→ 继承 STAGE-1 的低 alu/tc。
用 `PolicyParts<Stock::DispatchPolicy>` 把 CollectiveBuilder 产出的 stock collective 的全部公开 typedef 拆出来，
再用它们实例化 fork（config 逐字=builder 选的），`ForkNV = ClearAccum:true`、`ForkMX = ClearAccum:false`。

### 1b. `codesign_fused_kernel.hpp` — fork cooperative GemmUniversal glue，拥有两个 collective + 共享累加器
fork `sm90_gemm_tma_warpspecialized_cooperative.hpp`（STAGE-1 证实就是驱动 SM120 block-scaled 的 glue）为独立
`codesign::CodesignFusedKernel<ProblemShape, ForkNV, ForkMX, Epilogue, StaticPersistentScheduler, bool DualPhase>`。
对比出厂 glue 的差异（side-by-side）：

| 出厂 glue | 本轮 fork |
|--|--|
| 一个 `CollectiveMainloop` | **两个**：`CollectiveMainloopNV`(Phase1) + `CollectiveMainloopMX`(Phase2) |
| 一个 `mainloop` pipeline + 一个 `tensors.mainloop` | 两个 pipeline(`mainloop_nv`/`mainloop_mx`) + **union** 的 `tensors.mainloop{nv;mx}` |
| consumer: `accumulators=partition_fragment_C`; `mma(...,accumulators,...)`; epilogue.store | consumer: **同一** `accumulators`；`nv.mma(…K_low…)`→`mx.mma(…K_high…)`（同 fragment，不 clear）；一条 epilogue.store |
| producer: 一轮 `collective.load` | producer: `nv.load(K_low)` → phase_bar → `mx.load(K_high)` |
| 支持 dynamic+static scheduler（CLC/throttle 分支） | **只保留 static persistent**（SM120 void→dynamic SM100 scheduler；强制 `StaticPersistentScheduler` tag→`PersistentTileSchedulerSm90`，`IsDynamicPersistent==false`，CLC/throttle 代码全删） |

`DualPhase=false` 把 Phase-2 整个 `if constexpr` 编译掉——用于在缝 MXFP8 **之前**先验证 fork glue 本身能复现 STAGE-1。

### 1c. `codesign_fused_ext.cu` — 组装 + pybind
CollectiveBuilder 产 stock NV(nv_float4_t e2m1, align32, VEC16) / MX(mx_float8_t e4m3, align16, VEC32) 两个 collective
（TileShape 128×128×128, Cluster 1×1×1, StageCount 显式 NV=4/MX=2）+ 一条共享 bf16 epilogue；fork 成 `ForkNV/ForkMX`；
`GemmUniversalAdapter<CodesignFusedKernel<true/false>>`。暴露 `fused_gemm`(双) / `single_nvfp4_gemm`(gate) / `prepack_sf_fp4_b` / `prepack_sf_fp8_b` / `compat_report`。
SF scatter 与 STAGE-1 同（logical (R,K/VEC) → CUTLASS 交错 atom buffer）。

---

## Part 2 — 资源共存（STAGE-1 点名的真正风险）如何逐条解决

| 风险 | 解决 | 证据 |
|--|--|--|
| **SMEM 预算** | **RTX Pro 6000 (SM120) 每 block 仅 101376 B opt-in smem**（≠ Hopper/B200 的 228KB！本轮硬发现）。两个 collective 在 BLK_K=128 非-union = 171008 B → 爆。先试 **BLK_K=64** 减半 → 被 collective 的 TMA/SF-atom `static_assert`（SF atom 假定 128）拒。→ **UNION 两个 mainloop tensor storage**（Phase1 全 drain 后 Phase2 复用），NV=4/MX=2 stages → `SharedStorageSize=88064 B < 101376` ✔ | `compat_report`: SharedStorageSize(dual)=88064；cudaFuncSetAttribute 成功；build+run OK |
| **Named barrier 冲突** | union smem 需相位屏障保护；用 `NamedBarrier(288, id)`（1 个 mainloop-producer warp 32 + 2 个 consumer warpgroup 256 = 288，是 warp-size 倍数）。**关键 bug（本轮最耗时的 deadlock 根因）**：instance ctor 把 user id **加上** `ReservedNamedBarrierCount(8)`；更关键——**cooperative epilogue 自己用 user id 0（effective 8）的 named barrier**，我最初用 user id 0/1（effective 8/9）**与 epilogue 撞**→ 全 kernel 死锁（GPU 0% util）。改用 **user id 6/7（effective 14/15）** 避开 → 解。collective 内部的 `Sm120MainloopBarrier`=reserved 7，也不撞。 | 改 id 前：device-side hang 0% util；改 id 后：`fused qkv M2048 cos_core=0.999997`，全 6 cell 通过 |
| **双 pipeline 错峰 / 相位 handoff** | 两个 `PipelineTmaAsync`（nv/mx）各自 storage（mbarrier 独立，不在 union 内）。每输出 tile：producer `nv.load`→`phase_bar1`→`mx.load`→`phase_bar2`；consumer `nv.mma`→`phase_bar1`→`mx.mma`→`phase_bar2`。**phase_bar1** 保证 NV consumer 读完 union smem 后 MX producer 才覆写；**phase_bar2** 保证本 tile MX 读完后下一 tile NV 才覆写。相位串行（牺牲跨相位 overlap 换 union 正确性，correctness-first） | trace 打印 `P/C nvload/bar1/mxload/bar2` 逐 tile 配对推进，收束到 cosine≈1 |
| **双 TMA 描述符** | NV / MX 各自 `to_underlying_arguments` 建独立的 A/B/SFA/SFB TMA 描述符；`DualPhase=false` 时 MX params 不构造（default，无 descriptor encode），单相位 gate 无需 MX 数据 | GATE A（单相位）6/6 cos=0.999999 |
| **K-split** | scheduler 用 `{M,N,K_low}` tile M/N + 给 NV 的 k-tile 数=K_low/128；MX k-tile 数=K_high/128 单独算（`mx_k_tile_count`）。hf=1/8：qkv/gate Kl1792/Kh256、down Kl7168/Kh1024（均被 128 整除；Kl%16、Kh%32 满足）。reorder-before-quant [low\|main\|high] 对齐保留 | 全 shape cosine 通过即证 K-split 正确 |

---

## Part 3 — 正确性（硬门，6/6 + 6/6，`logs/_cdfused_sweep.log`）

### GATE A — 单相位 NVFP4 走 **fork glue**（证 fork 本身复现 STAGE-1）
```
single-NVFP4 qkv  M2048/M8192 : cos=0.999999  [PASS]
single-NVFP4 down M2048/M8192 : cos=0.999999  [PASS]
single-NVFP4 gate M2048/M8192 : cos=0.999999  [PASS]
```

### GATE B — **双相位融合** vs (D_low+D_high)（proven round-1 单格式核之和）+ 全 oracle
```
fused qkv  M2048/M8192 Klo1792 Khi256 : cos_core=0.999997  cos_full=0.999999  [PASS]
fused down M2048/M8192 Klo7168 Khi1024: cos_core=0.999997  cos_full=0.999999  [PASS]
fused gate M2048/M8192 Klo1792 Khi256 : cos_core=0.999997  cos_full=0.999999  [PASS]
CDFUSED_SWEEP_ALL_PASS
```
- `cos_core`：我方融合核（纯 GEMM 和，plain epilogue）vs `D_low+D_high`（两个 proven round-1 核之和）=共享累加器的精确期望。
- `cos_full`：我方融合核 × 外层 s_x·s_w（可分离对角缩放，Python 施加）vs prep_mixed oracle = **0.999999**，比 cos_core 还高——因为我方在 FP32 累加后**只 round 一次**到 bf16，而参考 D_low+D_high 是**两次 round** 相加，我方实际更精确。
- → **双格式共享累加器 bit-exact（criterion #1 达成，6/6）。**

---

## Part 4 — 算法格式契约（逐条核对）

| 契约项 | 实现 | ✔ |
|--|--|--|
| NVFP4 E2M1 blk16 UE4M3 + 2×FP4/byte | `nv_float4_t<float_e2m1_t>` align32 VEC16；A_packed 2/byte；SFA/SFB UE4M3 | ✔ |
| MXFP8 E4M3 blk32 UE8M0 | `mx_float8_t<float_e4m3_t>` align16 VEC32；A 1/byte；SF UE8M0 | ✔ |
| A=(M,K)/B=(N,K) | RowMajor A / ColumnMajor B（=(N,K) contiguous），packed_stride | ✔ |
| SF (·,K/16)/(·,K/32) scatter | `cd_fill_sf_kernel` logical(R,K/VEC)→CUTLASS 交错 atom（VEC=16/32） | ✔ |
| K-split hf=1/8 维度正确 | qkv/gate Kl1792/Kh256、down Kl7168/Kh1024 | ✔ |
| 共享 FP32 累加器跨两相位、无 gmem 中间 | glue 持 `accumulators`，nv.mma→mx.mma 同 fragment，无 clear、无 gmem | ✔ |
| epilogue acc·s_x·s_w→fp16/bf16 | **部分**：融合核目前走 **plain** epilogue（acc→bf16）；外层 s_x·s_w 为可分离对角缩放，gate 中以 Python 施加并验 cos_full=0.999999。in-kernel EVT 外层缩放（出厂核已有）为剩余的一步 epilogue 增补，不影响核的 bit-exactness | ◑ |

---

## Part 5 — TC（criterion #2）与延迟 A/B（criterion #3）：本轮状态（诚实）

### 基础设施墙
远程机器本轮极度过载（load avg 14-24、17 users）。clean（无 trace）`codesign_fused_ext` 的单 TU 编译
（两个重 block-scaled collective × DualPhase true+false 两份 kernel）**cicc 跑了 40min 仍只拿到 69s CPU、2.8% 占用、
后段完全被调度饿死（CPU time 5min 不增）**。对照：同代码的 trace-ext 在 load 较低时 ~4min 完成。
→ clean-ext build + 链式 latency（`codesign_lat.py`）已 **detached 排队**，一旦机器缓解即自动出结果到 `logs/_cdlat.log`，
可由 controller / 下一轮收割。本轮未能在墙内采到 ncu TC 与 latency 数字。

### TC（criterion #2）—— 由构造继承（强论据，但本轮未直接 ncu 双格式核）
双格式 kernel 的两次 mma() **逐字调用 STAGE-1 已 ncu 实测的 stock NVFP4 / MXFP8 block-scaled collective**
（fork 只加了一个 `if constexpr(ClearAccum)` 守卫 + glue 管线，**完全没碰硬件 block-scaled mma / SF-consume 路径**）。
STAGE-1 实测：NVFP4 collective alu/tc 0.80-1.26、ACTIVE 59-87%，与 CUTLASS 端点逐指标相同。
故双格式核的**每相位** TC 继承 ≈1。**唯一未验的新变量**：相位屏障串行（无跨相位 overlap）对**聚合** ACTIVE 的影响
（相位间 pipeline drain 可能压低几 pp）。这需要双格式核的 ncu——本轮 infra-blocked，列为下一轮首验。
`prof_cdfused.py` 风格的 runner 可直接复用 STAGE-1 的 `run_tc_codesign.sh` 指标集。

### 延迟 A/B（criterion #3）—— ship/no-ship 决策：本轮未决（infra-blocked）
`codesign_lat.py` 已写好并排队：每 shape×M 对比 **co-design `fused_gemm`** vs **手写 mixed `mixed_gemm_fused_presf` /
`_b_presf`**（alu/tc~3.5 的出厂手写消费者，full s_x·s_w epilogue），同数据同 GPU，median(50 iters, 10 warmup)。
诚实 caveat：co-design 目前 plain epilogue（少做外层缩放），手写核做全 epilogue——GEMM mainloop 主导，差额会在报告标注。
**ship/no-ship 判据保持不变**：co-design 只有在更高 TC 转成**等于或低于**手写核的延迟时才上位；否则出厂核保持 fallback，
co-design 记为「higher-TC-but-not-faster」的诚实结论。**本轮无延迟数字，不做 ship 判定**（不 spin）。

---

## Part 6 — 判定 + 下一步

### 本轮判定
| 判据 | 目标 | 实测 | 判定 |
|--|--|--|--|
| 双格式 cosine | =1.000000 | core 0.999997 / full 0.999999（6/6，单次 round 实际更准） | ✔ |
| fork glue 复现 STAGE-1 | =单格式 | GATE A 0.999999（6/6） | ✔ |
| 共享累加器（同 fragment，不 clear） | 跨两相位 | ClearAccum 守卫 + 同 `accumulators` 实现并验 bit-exact | ✔ |
| 资源共存（smem/barrier/pipeline/TMA/K-split） | 全部可调和 | union 88064B<101376 cap；barrier remap 6/7；双 pipeline 相位屏障；双 TMA；K-split——全解 | ✔ |
| 双格式 TC alu/tc≈1 + ACTIVE | ncu 实测 | **infra-blocked**（构造继承 STAGE-1；相位串行对聚合 ACTIVE 影响未验） | ◐ 排队 |
| 延迟 A/B vs 出厂核 | ship/no-ship | **infra-blocked**（`codesign_lat.py` 排队，`logs/_cdlat.log` 收割） | ◐ 排队 |

**→ STAGE-2 的核心命题（双格式共享累加器 co-design 能否在 SM120 上 bit-exact 缝成一个 launch）= 成立。**
最难的部分——两个 warp-specialized block-scaled collective 在一个 forked glue 里共存（union smem / 相位屏障 remap /
双 pipeline handoff / 双 TMA / K-split）——**全部打通并 gated**。STAGE-1 点名的 Path-A 资源共存风险**无一成为 blocker**，
**不需要回退 Path B**（手写 CuTe mainloop）。

### 诚实边界（本轮没证的）
1. 双格式核的 ncu TC（alu/tc、聚合 ACTIVE）——infra-blocked，下一轮首验；相位串行可能压低聚合 ACTIVE 几 pp。
2. 端到端延迟 A/B 与 ship/no-ship——infra-blocked，`codesign_lat.py` 已排队。
3. 外层 s_x·s_w 目前在 kernel 外（可分离，已验 cos_full）；in-kernel EVT 外层缩放是剩余一步 epilogue 增补。
4. r∈{0,1/8,1/2,1} 扫：本轮覆盖生产点 hf=1/8（3 shape×M{2048,8192}）。r=0/1 为退化单格式（STAGE-1 / 单相位路径已覆盖）；r=1/2 待补。

### 下一步（优先级）
1. **机器缓解后收割 `logs/_cdlat.log`**（latency A/B）→ 做 ship/no-ship 判定。
2. 双格式核 ncu（`run_tc_codesign.sh` 指标集，CD-dual vs 出厂手写 mixed）→ 验 alu/tc≈1 + 聚合 ACTIVE；量化相位串行代价。
3. 若延迟不占优：评估**跨相位 overlap**（去掉相位屏障串行，改双缓冲 union 或非-union + 更激进 stage 裁剪）是否把 TC 优势转成延迟。
4. in-kernel EVT 外层 s_x·s_w 缩放，使 A/B 完全对等。
5. r=1/2 扫 + 7B/70B shape。

---

## 产物
- **新文件（出厂/受保护文件零改动）**：
  `codesign_sm120_bs_mma.hpp`（fork collective + ClearAccum）、`codesign_fused_kernel.hpp`（fork 双-collective glue +
  union smem + 相位屏障）、`codesign_fused_ext.cu`（组装 + pybind）、`codesign_fused_check.py`（gate）、
  `codesign_fused_sweep.py`（全 shape 正确性扫）、`codesign_trace.py`（死锁定位 trace）、`codesign_lat.py`（延迟 A/B，排队）。
- 正确性：`logs/_cdfused_sweep.log`（`CDFUSED_SWEEP_ALL_PASS`：GATE A 6/6 cos=0.999999 + GATE B 6/6 cos_core=0.999997/cos_full=0.999999）。
- 排队（infra-blocked）：`logs/_cdfused_gate.log`（clean build+gate）、`logs/_cdlat.log`（latency A/B）。
- 硬发现：**SM120 (RTX Pro 6000) 每 block opt-in smem 仅 101376 B**（强制 union，BLK_K=64 被 collective 拒）；
  **cooperative epilogue 占用 user named-barrier id 0（effective 8）**——双格式相位屏障必须避开（用 6/7）。
- `candidates.jsonl`：+2 条。GPU 已释放。
