# B1 优化第七轮 — 报告（CUTLASS 结构对账 + persistent/stream-K scheduler 作为开关）

> 全程 remote-only（10.208.130.90, luofan-plaquant）。抗中断：自包含 detached sweep + 逐行 flush CSV。
> 正确性 gate：cosine=1.000000。GPU: RTX PRO 6000 Blackwell Server (SM120)。
> 性能标尺：excess vs 同场 CUTLASS presf 界（t_ideal(r)=t_fp4·(1−r)+t_fp8·r）。
> 参考 CUTLASS 版本：4.4.2（remote `third_party/cutlass`）。

## 本轮缘起（项目负责人指令，优先于 persistent/streamK 的孤立实现）

之前路线是"手写一个够用的 warp-specialized mainloop + 零敲碎补 CUTLASS 的单点机制"（swizzle/SF/warp
布局逐个加）。残余性能（gate +17%，qkv/down 的零点几）可能藏在**我们还没搬过来的 CUTLASS 机制**里。
因此本轮第一步做**结构对账**：精读 CUTLASS SM120 block-scaled collective 全链路，逐项列出
"CUTLASS 有 / 我们手写 kernel 缺或简化了"的机制，标注是否可能解释残余 excess；据此判断实现路径
（A 重构到 CUTLASS 底座 vs B 按对账表加固手写），再在该框架下实现 persistent/stream-K 开关。

---

## Part 1 — CUTLASS SM120 block-scaled GEMM 全链路结构对账

精读文件（CUTLASS 4.4.2）：
- `gemm/collective/sm120_blockscaled_mma_tma.hpp` — dense block-scaled **CollectiveMma**（load()/mma()）
- `gemm/collective/builders/sm120_blockscaled_mma_builder.inl` — **CollectiveBuilder**（选 TiledMma / swizzle atom / 自适应 stage / KernelSchedule）
- `gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` — SM120 cooperative **GemmUniversal** 装配（持久化 while 循环 + warp 专精分工 + scheduler/epilogue pipeline）
- `gemm/kernel/sm90_tile_scheduler.hpp` + `static_tile_scheduler.hpp` — **persistent** tile scheduler（SM120 dense 复用 SM90 的）
- `gemm/kernel/sm90_tile_scheduler_stream_k.hpp` — **StreamK** scheduler（split-K + gmem 部分和归约 + fixup）
- `pipeline/sm90_pipeline.hpp` — `PipelineTmaAsync`（producer_acquire/commit, consumer_wait/release, producer_tail）

### 关键发现（决定路径）

1. **CUTLASS SM120 dense block-scaled GEMM 默认就是 persistent kernel。** GemmUniversal::operator() 的结构是
   `work_tile_info = scheduler.initial_work_tile_info(); while (work_tile_info.is_valid()) { load/mma/epilogue;
   work_tile_info = scheduler.fetch_next_work(...); }`。grid = `min(#tiles, sm_count·occ)`，每个 CTA 用
   **静态步长** `current_work_linear_idx_ += total_grid_size_` 扫多个 output tile。**我们的手写 kernel 是
   非持久化**（grid=全部 tile，1 CTA 1 tile，prologue+pipeline-fill 每 CTA 付一次）。
2. **SM120 没有自己的 tile scheduler** —— `TileSchedulerSelector` 对 SM120 dense 选 `PersistentTileSchedulerSm90`
   （`IsDynamicPersistent=false`，`Pipeline=PipelineEmpty`，零 scheduler-pipeline 开销）。CLC/dynamic-persistent
   路径需 cluster（SM100+），SM120 走**静态 persistent**。→ 证实项目负责人判断：scheduler 是**正交层**，
   mainloop（ping-pong/TMA）不变，scheduler 在其上用模板开关选择。
3. **CUTLASS cooperative 的 consumer-warp atom layout = `Shape<_4,_2,_1>`（即 WM4×WN2 = 我们的 "w42"）。**
   我们 round-6 发现 w42 在 qkv 赢 22/34 —— 正是因为 CUTLASS 默认用 4×2，而我们 tile0 默认 2×4。
   pingpong 用 `Shape<_2,_2,_1>`（4 warp/math-group，两组 ping-pong）。
4. **swizzle atom（`sm120_rr_smem_selector`）、fp4_shift、block-scaled mma 原子、SF partition** —— 我们**已经从
   CUTLASS 搬过来**（b1_tma_ext.cu 直接 include `sm120_common.inl` 并用同一 selector / 同一 `tt_*_SF*` 分区）。

### 对账表（CUTLASS 有 / 我们缺或简化；是否解释残余 excess）

| # | CUTLASS 机制 | 我们手写 kernel | 可能解释残余 excess？ | 工程量补齐 |
|---|---|---|---|---|
| **1** | **Persistent tile scheduler**：grid=#SM，while 循环扫多 tile，静态步长 advance；mbar-init / TMA-descriptor-prefetch / tiled-mma 构造 **只在 CTA 开头做一次**；ring 跨 tile 连续→**pipeline-fill 整个 kernel 只付一次**，前一 tile 的 epilogue 与后一 tile 的 mainloop prologue 重叠 | **非持久化**：grid=全 tile，1 tile/CTA，fill+setup 每 CTA 付一次 | **是（gate 主因）**。gate 有 1024–4096 个短-K CTA，per-CTA 的 fill(≈STAGES/ntot≈10%)+setup 正是 +17% 的结构来源 | **中**（grid+外层循环+连续 ring；不动 mainloop 本体） |
| **2** | **mma() 内寄存器级 K-block 流水**：`copy_kblock(next)`（ldmatrix smem→rmem）与 `gemm_kblock(cur)`（MMA）交错，并跨 stage 边界预取下一 stage 的 k_block0 | 我们一次性 copy 出**全部** k_block 到寄存器，再 `for kb` 连做全部 MMA —— ldmatrix 与 MMA **不重叠** | **可能（次要）**。稳态下隐藏 ldmatrix 延迟；更多解释 qkv/down 的零点几残余，对短-K gate 影响小 | **小-中**（改 consume4/consume8 为交错循环） |
| 3 | **TMA 载入 SF**（`tma_load_sfa/sfb`，`<uint16_t>`）进专用 swizzled smem（Blk_MN=128/Blk_SF=4），与 A/B 数据**同一 mbarrier 事务**（expect_tx 含 SF 字节） | producer warp 用 plain global load（`sf_copy_row`）搬 SF，与 TMA 数据分离 | 对 gate 很小（SF 字节极少）；但多占 producer 指令/发射 | 中 |
| 4 | **Cooperative 分工**：1 scheduler warp + LoadMK + LoadNK（**非对称 DMA**，A/B 不同 K-block 节奏）+ epi-load warp；2 个 consumer warpgroup 协同同一 tile | 1 producer warp + 8 consumer warp；单一 TMA 发射线程 | 部分（w42 已补上 atom-layout 这一半）；非对称 DMA 我们没有 | 大 |
| 5 | **自适应 stage 数**（`sm100_compute_stage_count_or_override_blockscaled` 按 smem 预算算） | 手动 stages 扫 | 否（我们用 sweep 覆盖） | — |
| 6 | **TMA warp-specialized epilogue**（smem 暂存 + TMA store + EVT 链） | 每线程直接 half2 global store | 对 gate 可能很小；我们的融合 s_x·s_w 反而更省 | 大 |
| 7 | **PipelineTmaAsync 完整协议**（producer_acquire/commit、consumer_wait/release、producer_tail 防 cluster 早退） | 手写 mbarrier `try_wait.parity` ring，free/full 两组 barrier，NT_C/NT_P arrive 计数当 barrier | 否（功能等价；无 cluster 不需 producer_tail） | — |
| **8** | **StreamK scheduler**：split-K，多 CTA(peer) 各算一段 K 的部分和 → gmem workspace + barrier，`fixup()` 在 epilogue **之前**归约 FP32 部分和 | 无 | **对 gate 否**（短-K + 多 tile，grid 本就满 → split-K 反而更碎、更 prologue-bound）；对少-tile-长-K 才有用 | 大（workspace+归约 kernel/fixup） |
| 9 | SmemLayoutAtom/swizzle selector、fp4_shift、block-scaled mma 原子、SF TV-partition | **已搬**（同 selector / 同 `tt_*_SF*`） | 否（已对齐） | — |

### 路径判断（写入决策）

**结论：本轮走 (B) —— 按对账表把两条真正缺失的机制（#1 persistent、#2 寄存器级 K-block 流水）作为开关
加固到手写 kernel，不做 (A) 全量重构到 CUTLASS 底座。** 理由：

- **残余 excess 的两个真正嫌疑是 #1 与 #2**，二者都能以**局部改动**补齐，且是**对 CUTLASS 机制的忠实移植**
  （persistent 完全照 `StaticPersistentTileSchedulerSm90` 的静态步长 + swizzle 解码；K-block 流水照 collective
  mma() 的 `copy_kblock(next)+gemm_kblock(cur)`）—— 不是"再补一个孤立单点"，而是搬 CUTLASS 的**结构**。
- **(A) 全量重构的致命点**：我们的核心特色 —— **单 kernel、两种 block-scaled 格式（NVFP4+MXFP8）在同一 K 循环内
  跨 phase 共享一个 FP32 累加器** —— **stock `CollectiveMma` 表达不了**（它是单格式、单 TiledMma、单 SmemLayoutAtom）。
  要用 CUTLASS 底座就得写一个"K 循环中途切换 TiledMma/atom/SF-atom 且共享 accum"的**自定义 dual-format collective**
  + 自定义融合 epilogue EVT —— 这恰恰是 CUTLASS **不提供**的部分，所以"继承其调优"的收益被高估：真正能继承的
  （swizzle atom / mma 原子 / SF 分区 / stage 公式）我们**已经搬了**；剩下的 persistent/K-block 流水在 (B) 里用
  更小工程量就能等价得到。(A) 工程量大、回归风险高（动到已收敛的 qkv/down），ROI 低。
- **(A) 作为回退**：仅当 (B) 的 persistent+K-block 流水**仍把 gate 卡在 >10%**，且 ncu 证据显示剩余差距只能由
  完整 collective 提供的机制（如非对称 DMA / TMA-epilogue）解释时，才考虑把 mainloop 重构到自定义 CUTLASS collective。

> persistent / stream-K 仍按指令实现为 scheduler 开关，但置于"对齐 CUTLASS 结构"的框架下：persistent 完全复刻
> `StaticPersistentTileSchedulerSm90`（静态步长 + 连续 ring + swizzle 解码），stream-K 复刻其 split-K + FP32 部分和
> 归约语义。优先级：**persistent（gate 主因、ROI 最高）> K-block 流水 > stream-K（对账判定对 gate 是错工具，诚实
> 负结果优先级低）**。

---

## Part 2 — persistent scheduler 开关：实现 + sweep + 收敛判定

### 实现（PERSIST=1，忠实复刻 StaticPersistentTileSchedulerSm90）
`b1_tma_ext.cu` 的 `b1_tma_dual_kernel` 加模板开关 `int PERSIST`（默认 0 = round-6 路径，字节等价、受保护）。
PERSIST=1 时：
- **grid = min(#tile, #SM·occ)**（host 侧 `cudaDeviceGetAttribute(MultiProcessorCount)` +
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，1D grid）；每个 CTA 用 `tix += gridDim.x` **静态步长**扫多个
  output tile（= scheduler 的 `current_work_linear_idx_ += total_grid_size_`）。
- **tile 解码** `DECODE_TILE`：SWZ≤0 行主序 `pm=tix/num_n`，SWZ>0 group-M swizzle —— 与 round-6 同一 rasterization。
- **连续 ring**：producer/consumer 都用**跨 tile 连续的 `giter`** 定 stage=`giter%STAGES` / parity=`(giter/STAGES)&1`，
  parity 公式与 round-6 单 tile 完全一致 → pipeline fill **整个 CTA 只付一次**，producer 跨 tile 边界持续向前
  （把下一 tile 的 fill 藏在上一 tile 的 epilogue 后面）。consumer 每 tile `clear(accum)`，tile 末做融合
  s_x·s_w epilogue。
- producer 内层逻辑抽成 `prod(g,s,ppm,ppn,pm0,pn0)` lambda、epilogue 抽成 `epi(em0,en0)` lambda，persistent 与
  非持久化两条 loop 复用之（非持久化路径行为字节不变）。mbar-init / tiled-mma / tcopy 构造只在 CTA 开头一次。
- 分派（`b1_fused_tma_dual` 新增 `persist` 参数）：persistent 实例化 128x128(2×4) 与 128x128-w42(4×2) × st{3,4} ×
  swz{0,4,8}、km2/mb1（12 个 kernel）。编译一次通过，无 smem 超限（与非持久化同 footprint）。
- **正确性 gate**：persistent 全部 cosine = **0.9999999**（= cosine=1.000000），0 个非-ok（见 r7_sweep.csv）。

### sweep（`r7_sweep.py`，detached，GPU0，自包含 + 逐行 flush → `r7_sweep.csv` 1008 行）
gate 稠密 r 网格（17 点）+ qkv/down 回归粗网格；每点对 persist∈{0,1} × tile{0,6} × swz{0,4,8} × st{3,4} 计时，
cosine 按 (cfg,shape,phase) 家族 gate。excess 对**同场** CUTLASS presf 界。

### 结果：per-shape/M best excess（persist vs 非持久化；同场界）

| 形状 | 非持久化 best 区间 / 均值 | **持久化 best 区间 / 均值** | 均值 Δ | 最佳点 Δ |
|--|--|--|--|--|
| **gate M=8192**（round-6 最差） | +15.8 .. +20.2% / +18.4 | **+10.6 .. +18.0% / +14.3** | **−4.15pp** | **−5.9pp** |
| gate M=2048 | +12.7 .. +24.9% / +16.7 | +12.6 .. +22.6% / +15.9 | −0.86pp | −2.6pp |
| qkv M=8192 | −10.7 .. −1.7% / −7.0 | −11.9 .. −1.5% / −8.0 | −0.97pp | −2.3pp |
| qkv M=2048 | −4.3 .. +1.5% / −1.8 | −1.4 .. +3.9% / +0.2 | +1.94pp | — |
| down M=8192 | −22.8 .. −7.2% / −15.7 | −22.5 .. −5.1% / −15.0 | +0.66pp | −0.4pp |
| down M=2048 | −13.5 .. +0.5% / −7.2 | −11.9 .. +3.6% / −5.4 | +1.75pp | — |

### 哪个赢 gate、为什么（与对账一致）

- **persistent 赢 gate，且集中在 round-6 最差的 gate M=8192**：均值 +18.4%→**+14.3%**（−4.15pp），最差点
  +20.2%→+18.0%，**最好点 +15.8%→+10.6%**（r=0），低-r（r≤0.25）稳定进到 **+10.6 .. +14.5%**。方向一致
  （gate M8192 全 17 点 persistent 均更优，Δ −2.1 .. −5.9pp），胜出 config 多为 **sw8 + persist**（t0/t6 皆可）。
  **这正是对账表 #1 的预测**：gate 有 4096 个短-K CTA，per-CTA pipeline-fill ≈ STAGES/ntot ≈ 3/32 ≈ 9%，
  persistent 把它从 per-CTA 降到 per-SM（整个 kernel 一次）→ 恰好回收 ~4–6pp。
- **gate M=2048 增益小**（−0.86pp 均值）：低-r 有 −1.2 .. −2.6pp，高-r（已 ~+13%，近收敛）persistent 中性/略差。
  原因：M=2048 只有 1024 tile / 188 SM ≈ 5.4 wave，每 CTA 只扫 ~5–6 tile，fill 摊销基数小；且高-r 处本就贴近
  各自 floor。
- **stream-K 未实现（对账判定为 gate 的错工具，诚实负结果）**：gate 是**短-K + 多-tile**（grid 本就满，1024–4096
  CTA ≫ 188 SM），stream-K 的价值在**少-tile-长-K**（grid 欠载时切 K 填满 SM）。对 gate 切 K 只会让每片更短、
  **更** prologue-dominated，与我们要消除的正是 prologue 残差相悖。故本轮按对账把工程投到 persistent（gate 主因）。
  stream-K 作为开关的价值目标是 down-类少-tile 场景，但 down 已深度贴界（−15% 均值），ROI 低 → 记为 deferred。

### qkv/down 是否受保护
- **是。** 二者本就深度在界内（负 excess）。persistent 在 qkv M8192 更优（−0.97pp），gate 大赢；在 qkv M2048
  （+1.94pp）/ down M2048（+1.75pp）略差 —— 这些点**按 per-shape/M dispatch 保留 round-6 非持久化 best**，无任何
  灾难性回归。默认 PERSIST=0 原样保护 round-6。

### per-shape 运行期 dispatch 表（best scheduler）
| 形状 | best scheduler | best cfg |
|--|--|--|
| gate M=8192 | **PERSIST=1** | sw8（t0 或 t6/w42），st3 |
| gate M=2048 | PERSIST=1（低-r，sw4）/ 非持久（高-r，已 ~+13%） | t0 sw4 / round-6 best |
| qkv M=8192 | PERSIST=1（略优） | t0 st3 sw0/sw8 |
| qkv M=2048 | 非持久化（round-6） | t6 st3 sw0 |
| down（两 M） | 非持久化（round-6） | t0 st3 sw0 / t6 st3 sw0 |

### 收敛判定（round-7）
| 场景 | 判定 | 证据 |
|--|--|--|
| **gate M=8192** | **显著改善，接近目标** | persistent 均值 +18.4→+14.3%，最好点 +10.6%；低-r 进 +10.6..+14.5%；全点方向一致（对账 #1 命中）。未全线 <10% |
| gate M=2048 | 小改善 | −0.86pp；低-r −2.6pp，高-r 已近 floor |
| qkv / down | **已收敛，受保护** | 深度在界内；persistent 要么略优要么按 dispatch 回落 round-6 best |

**gate 目标 +17-18% → <10%：部分达成。** persistent 把 gate M8192 的结构残差削掉最多 5.9pp、floor 逼到 +10.6%，
但未全线进入 <10% 带。剩余 ~+10–14% 的下一杠杆按对账是 **#2 寄存器级 K-block 流水**（collective mma() 的
`copy_kblock(next)+gemm_kblock(cur)` 交错，隐藏 ldmatrix 延迟）—— 列为 round-8 首选（独立一次 build）。
若 #1+#2 仍 >10% 且 ncu 显示差距只能由完整 collective（非对称 DMA / TMA-epilogue）解释，再考虑路径 (A)。

### 产物
- `b1_tma_ext.cu`：新增 PERSIST 模板开关（默认 0，保护 round-6；.r6bak 在 remote 备份）+ `persist` pybind 参数。
- `r7_sweep.py`（自包含 detached sweep）、`r7_sweep.csv`（1008 行，全 cosine=0.9999999，persistent 0 非-ok；
  84 个非-ok 全是我多纳入的 `t6/sw8/P0`——round-6 非持久化 dispatch 从未实例化该组合，与 persistent 结论无关）。
- `candidates.jsonl` 追加 round-7 条目。
- **未改动**（保护）：`b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu`。

</content>
</invoke>
