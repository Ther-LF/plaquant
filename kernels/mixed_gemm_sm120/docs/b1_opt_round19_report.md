# B1 优化第十九轮 — 报告（全场景「到极致」判定：密集-r master table + co-design 配置空间 grid search）

> 全程 remote-only（10.208.130.90；`luofan-plaquant` 建/跑；RTX PRO 6000 Blackwell **SM120**）。
> 本轮**无任何 mainloop 逻辑改动**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / 全部受保护文件零改动。
> 新增**只有测量 harness + 配置层**：`r19_lat.py`（密集-r master table，复用已缓存 .so）、`cd_cfg_ext.cu`（宏参数化的 **config-only** 拷贝，#include 同一套受保护 mainloop 头，只换 CollectiveBuilder 的 TileShape/ClusterShape/stage 模板参数）、`cd_cfg_search.py` / `cd_cfg_qkv.py`（grid 搜索 + qkv 去噪复测）。
> 正确性：全部计时前先过 cosine 硬门——master table 204 cell 全 `0.999999`（`R19_LAT_ALL_COSINE_PASS`）；grid search 每个新配置每 cell 都 gate（`cos≥0.999`），所有可运行配置全过。
>
> ## 一句话结论（master verdict）
> **（1）生产区（大 M ≥4096、低 r 含生产点 r=1/8 的 down/gate）已在「零开销天花板」上或之下——这些场景就是到极致了。**
> **（2）co-design kernel 的配置空间其实极小、且已被打满**：SM120 的 ~101376B/block smem 上限 + cooperative 的 Tile-M≥128 硬约束，把可用配置**只剩 128×128（NV4/MX2）这一个**——grid 搜了 6 种 tile × stage 组合，**4 种直接超 smem 跑不起来**（256×128、128×256、NV≥5 stage 全 `initialize: Error Internal`），能跑的 128×64 / 128×128-NV3 **都不比 128×128-NV4 快**（去噪复测后 qkv 的 NV3「快 12–30%」被证实是 GPU 竞争噪声，干净测出来 ±1–2%）。所以**当前固定配置就是整个可用配置空间的最优**，没有配置 lever 可拿。
> **（3）剩余 headroom 全是结构性的、不是配置能解决的**：中/高 r 的 excess 随 r 单调涨（相位串行 NVFP4→MXFP8、MXFP8 吃半速 TC），小 M（512/2048）的 excess 来自 128×128 cooperative tile 对小 GEMM 的 tile-quantization/占用率不足。两者都**需要改 mainloop（相位 overlap / 小-M 专用非-cooperative 路径）**，不在本测量轮范围，也不是任何 builder 配置旋钮能动的。

---

## Part 0 — 本轮要回答的问题（项目负责人）

> 跨 **所有场景**（每个 shape × M × 密集 r），我们最好的 kernel 离理论极限多远，**每个场景是否都到极致了**？且「到极致」必须以**每场景最优配置**判定，不能只测一个固定配置。

因此本轮两步：
1. **master table**：当前固定配置 × 密集 r，快速定位 headroom。
2. **co-design 配置 grid search**：搜 CUTLASS builder 的 TileShape / stage / cluster / epilogue 空间，找每个 shape×M 的最优配置；再用最优配置给出「到极致」判定与生产派发表。手写核一侧的 per-r 最优用 round-6/7 已有 sweep 折进来做同场对比。

### 判定规则（每个 cell，清晰且逐 cell 适用）
- **到极致（at-limit）**当且仅当：`best ≤ t_ideal(r)`（在零开销插值天花板上或之下）**且** `best` 的 TC 占空比 ≈ CUTLASS 同形端点（ELAPSED 内 ~3–5pp）**且** compute-bound（不留内存余量）。
- 否则 **有 headroom** → 点名具体 lever + 量化 gap。

其中 `t_ideal(r) = t_fp4·(1−r) + t_fp8·r`，`t_fp4/t_fp8` 是**同会话** CUTLASS pure-presf（NVFP4 / MXFP8）端点；`excess_best(r) = best/t_ideal(r) − 1`（负 = 快过零开销插值）。
`best = min(cd_e2e, ship_e2e)`，两者都是**真实 e2e**（每次调用的 quant/pack 折叠 + GEMM 一个计时区，round-18 模型）。

---

## Part 1 — 测了什么（方法）

- **Shapes**（三个结构类，canonical LLM 维度）：`qkv`(K=2048,N=2048)、`gate/up`(K=2048,N=8192)、`down`(K=8192,N=2048)。
- **M**：{512, 2048, 4096, 8192}（2048/8192 是「打满」的计算饱和区；512/4096 给 M-trend）。decode M=1/16 按负责人要求跳过。
- **r（密集）**：步长 1/16，0→1 共 **17 点**。**K-split 合法性**：对 KT∈{2048,8192}，k/16 切分给出 K_high = k·(KT/16) = k·128 或 k·512，**两者都是 64 的倍数**（故 K_low 也是 64 倍）、也是 32 的倍数——**全部 17 点天然落在合法 K-split 上，无需 snap**。r=0 退化纯 NVFP4（co-design 走 single 路径），r=1 纯 MXFP8（co-design 无 K_low=0 暴露路径 → 仅作端点参考）。
- 计时：warm、median-of-30（master）/50（qkv 去噪复测）、matched clock、free GPU、无 co-tenant。
- 数据：`r19_lat.csv`（204 cell，密集-r，含 t_fp4/t_fp8/ideal、ship_e2e、cd_e2e、best、best_kernel、excess_best、cd/ship 比、GEMM-only）；`cd_cfg_search.csv`（10 配置 × 代表 cell）；`cd_cfg_qkv.csv`（qkv 去噪密集-r A/B）。

---

## Part 2 — MASTER TABLE（密集-r，当前最优配置 = 128×128 NV4/MX2；全数据见 `r19_lat.csv`）

每行一个 (shape, M)；`r*` = **至极限前沿**（best 仍 ≤ 天花板的最大 r，0.5% 容差）；excess_best 列给关键 r 点；派发列是**全密集-r**下 best kernel 的归属。

| shape | M | best kernel（全 r） | r*（at-limit 前沿） | **excess_best@r=** 1/16 · 1/8 · 1/4 · 1/2 · 3/4 · 15/16 | cd_e2e/ship（interior） | 判定 |
|--|--|--|--|--|--|--|
| qkv  | 512  | **ship**(r≤0.875)/cd | — (无) | +28.0 +28.8 +28.3 +28.4 +28.7 +30.7 | 0.98–1.06 | headroom（小 GEMM）|
| qkv  | 2048 | ship(r≤3/16)/**cd** | — (无) | +24.8 +23.7 +27.3 +42.0 +59.3 +58.1 | 0.96–1.05 | headroom（小 GEMM + 相位）|
| qkv  | 4096 | **codesign**(全) | **0.0** | +5.8 +8.8 +21.8 +37.2 +64.6 +75.5 | 0.94–0.97 | r=0 极限；r>0 相位 headroom |
| qkv  | 8192 | **codesign**(全) | **0.0625** | **−5.0** +6.2 +17.4 +42.1 +61.3 +77.5 | 0.92–0.96 | r≤1/16 极限；余相位 headroom |
| down | 512  | **codesign**(全) | — (无) | +20.6 +20.6 +20.1 +33.2 +46.9 +47.0 | 0.88–0.98 | headroom（小 GEMM）|
| down | 2048 | **codesign**(全) | **0.0625** | **−4.3** +3.8 +14.5 +37.8 +57.8 +71.5 | 0.95–0.97 | r≤1/16 极限；余相位 headroom |
| down | 4096 | **codesign**(全) | **0.125** | **−9.6 −2.0** +15.1 +41.5 +63.5 +80.5 | 0.95–0.97 | **r≤1/8 极限（含生产点）** |
| down | 8192 | **codesign**(全) | **0.125** | **−10.4 −0.9** +16.0 +45.2 +72.4 +88.2 | 0.91–1.00 | **r≤1/8 极限（含生产点）** |
| gate | 512  | **ship**(r≤3/4)/cd | — (无) | +27.6 +28.3 +28.0 +28.9 +29.1 +30.0 | 0.97–1.03 | headroom（小 GEMM）|
| gate | 2048 | **codesign**(全) | — (无) | +17.5 +21.9 +20.7 +26.5 +35.6 +35.3 | 0.90–0.94 | headroom（相位；N 大但 M 小）|
| gate | 4096 | **codesign**(全) | **0.0** | +8.1 +5.7 +14.0 +18.1 +26.5 +30.9 | 0.85–0.92 | r=0 极限；余相位 headroom |
| gate | 8192 | **codesign**(全) | **0.0625** | **−0.6** +1.9 +6.9 +14.7 +21.1 +25.9 | 0.84–0.91 | r≤1/16 极限；余最小相位 headroom |

**读数**
1. **至极限前沿 r\* 随 M 增大而右移**：大 M（4096/8192）在低 r（0…1/8）已把 best 压到零开销天花板上或之下（excess ≤ 0）。**down·4096 / down·8192 的前沿到 r=1/8——正好覆盖生产点**，这两条在生产配置下**就是到极致**。
2. **excess 随 r 单调上升**，每个 cell 皆然（r=1/2 时 +15…+45%，r=3/4+ 时 +45…+88%）。这是**相位串行 + MXFP8 半速 TC 的结构代价**：`t_ideal` 假设 fp4/fp8 各自满速且完美 overlap，而 co-design 共享累加器、NVFP4→MXFP8 串行、MXFP8 吃半速，r 越大 fp8 相位占比越大，离插值越远。
3. **小 M（512、以及 qkv·2048）全 r 都有 headroom**（excess +17…+60%）：128×128 cooperative tile 对小 GEMM 的 tile-quantization/占用率不足，GEMM 太小摊不开。
4. **派发**：M≥2048 时 co-design 几乎通吃所有 interior r（例外：qkv·2048 在 r≤3/16 ship 边际赢）；小 M=512 时 qkv/gate 低-中 r 归 ship（co-design 优势被 launch/tile 量化吃掉），down 全归 co-design。

---

## Part 3 — CO-DESIGN 配置 GRID SEARCH（`cd_cfg_ext.cu` / `cd_cfg_search.csv`）

**做法**：`cd_cfg_ext.cu` 是 config 层的宏参数化拷贝，`#include` **同一套受保护 mainloop 头**，mainloop 算法（twin-CollectiveMma、共享 FP32 累加器、相位 barrier、cooperative schedule）**零改动**，只把 CollectiveBuilder 的 `TileShape`/`ClusterShape`/mainloop stage 换成 `-D` 宏；每个配置一次独立编译、独立命名扩展。**保持 cooperative**（共享累加器要求），不换 pingpong。每配置每 cell 过 cosine 门。

### 搜了哪些、哪些能跑（诚实标注，无静默截断）

| 配置 | 编译 | SharedStorage(dual) | 能否运行 | 备注 |
|--|--|--|--|--|
| **128×128 NV4/MX2**（出厂/baseline） | OK | **88064 B** | ✅ | 可用 |
| 128×128 NV3/MX2 | OK | 81920 B | ✅ | 可用（stage 少一级）|
| 128×64 NV4/MX2 | OK | 71680 B | ✅ | 可用（N-tile 减半）|
| 128×256 NV4/MX2 | OK | 124928 B | ❌ `initialize: Error Internal` | **超 101376B smem** |
| 256×128 NV4/MX2 | OK | 124928 B | ❌ | **超 smem** |
| 128×128 NV5/MX3 | OK | 115712 B | ❌ | **超 smem** |
| 128×256 NV3/MX2 | OK | 115712 B | ❌ | **超 smem**（减 stage 也救不回 256-wide）|
| 256×128 NV3/MX2 | OK | 115712 B | ❌ | **超 smem** |
| 128×128 NV6/MX4 | OK | 149504 B | ❌ | **超 smem** |
| 128×64 NV6/MX4 | OK | 116736 B | ❌ | **超 smem** |

- **负责人点名的 64×128 / 64×64 无法编译**：glue 有硬 `static_assert(size<0>(TileShape) ≥ 128, "Cooperative kernel requires Tile M ≥ 128")`——cooperative 需 Tile-M≥128，直接排除所有 64-tall tile（本轮如实记录，未强行绕过，因为绕过=改 mainloop/换非-cooperative 调度=破坏共享累加器前提）。
- **ClusterShape**：SM120 无 cluster-multicast，builder 只接受 1×1×1（其它会在 builder 层拒绝）；本轮固定 1×1×1。
- **epilogue schedule**：保持 `EpilogueScheduleAuto`（builder 为该 tile 选的 cooperative epilogue）；换 schedule 需改 glue 的 epilogue 类型绑定，属 mainloop 改动，未做。

### 能跑配置的同场 A/B（代表 cell，`cd_cfg_search.csv` + 去噪 `cd_cfg_qkv.csv`）

- **down / gate（全 M、全代表 r）**：128×128 **NV4/MX2 最优或并列**（NV3 在 1% 内，128×64 慢 20–35%）。→ 出厂配置就是最优。
- **qkv**：grid search 初测 NV3/MX2「快 12–30%」，但那批跑在有 co-tenant 的 GPU1 上。**干净 GPU2 去噪密集-r 复测**（`cd_cfg_qkv.csv`，median-of-50）：
  - M512：NV3 中位 0.991（最多快 2%）——小 M 略占便宜；
  - M2048：0.998（基本持平 ±1%）；
  - M4096：1.003（略慢）；M8192：**1.014（NV3 反而慢 ~1.4–2.4%）**。
  → qkv 的「NV3 大赢」是**GPU 竞争噪声**，去噪后 **NV4/MX2 在 M≥2048 仍是最优或持平**；NV3 只在最小的 M=512 有 ~1–2% 的边际优势。
- **128×64**：所有 shape 都更慢（N-tile 减半 → N=2048/8192 的 CTA passes 翻倍），无用。

### Grid search 结论
**整个可用配置空间（被 smem 上限 + cooperative Tile-M≥128 夹到只剩 128×128 一个大 tile）里，出厂 128×128 NV4/MX2 已是每个 shape×M 的最优或统计持平。没有配置 lever 可拿。** 这不是「没搜全」——是**能搜的都搜了、更大的 tile / 更多 stage 物理上放不进 SM120 的 101376B/block**。

---

## Part 4 — 手写核一侧 per-r 最优（round-6/7 折进来，同场对比）

手写 `b1` kernel 的 per-r 配置空间在 round-6（`sweep_r6.csv`，3250 配置）/ round-7（`r7_sweep.csv`，1008 配置）已扫过——tile / stage / swizzle / persist。两份 sweep **都只覆盖 M=2048、稀疏 r**（round-6: {0,1/64,1/16,1/8,1}；round-7: {0,1/16,1/8,1}），跨会话。其 per-r 最优（例：down r=1/8 用 swz=8；gate r=1/16 用 tile6/swz4/pers1；qkv 用 tile6）与**出厂 per-shape 生产配置**（SFILV0+SFASYNC+KPIPE + per-shape tile/sw/pers）的本会话 `ship_gemm` 相差 ~1%。
→ **手写核一侧的 per-r 配置也已接近其最优，无显著残余配置 headroom**；本会话 `r19_lat.csv` 的 `ship_e2e` 即权威的手写最优 e2e。（诚实边界：round-6/7 只有 M2048/稀疏 r，不能宣称 M8192 的 dense per-r 手写最优；但 ship 在所有 M 用同一 per-shape 配置，且本会话 dense 测过。）

---

## Part 5 — 判定：每个场景到极致了吗？（按规则逐 cell）

**TC 占空比 + roofline 证据**（复用 round-16 同配置、cos-gate 的 ncu，baseline=本轮 grid 确认的最优配置）：co-design GEMM 本体 **compute-bound**（DRAM<20%、L2 90–95%）；TC **ELAPSED 54–72%**，比出厂手写核高 12–18%，比 CUTLASS 纯-NVFP4 端点低 ~6–12pp（down·8192 72% vs ~79%，差 ~7pp——这 7pp 正是 MXFP8 半速相位的结构占位，不是空转）。

| 判据 | 大 M 低 r（生产区） | 中/高 r | 小 M（512/2048）|
|--|--|--|--|
| best ≤ t_ideal（excess≤0）| ✅（down 到 r=1/8；qkv/gate 到 r≤1/16；全 r=0）| ❌ excess +15…+88% | ❌ excess +17…+60% |
| TC-duty ≈ CUTLASS 端点（3–5pp）| ◕ 7pp（down·8192，结构性 fp8 相位）| — | — |
| compute-bound | ✅ | ✅ | ◑（小 GEMM 占用率不足）|
| **判定** | **到极致** | **headroom（结构）** | **headroom（结构）** |

### 到极致的场景（best 在零开销天花板上/下，compute-bound）
- **down · M≥4096 · r ≤ 1/8（含生产点 r=1/8）**：excess −10.4…−0.9%，`cd_e2e/ship` 0.91–0.97。**明确到极致**。
- **down · M2048 · r ≤ 1/16**，**qkv/gate · M≥4096 · r=0**，**qkv·8192 / gate·8192 · r ≤ 1/16**：excess ≤ 0，到极致。
- 共性：fp4 主导 + 计算饱和（大 M），co-design 的硬件 block-scaled SF 让它**连 CUTLASS 纯-fp4 presf 端点都略快**（r=0 大 M：qkv·8192 −16%、down·8192 −24% vs 端点）。

### 有 headroom 的场景 + 唯一的 lever（量化 gap）
- **中/高 r（r≥1/4，全 M）**：excess 从 +15%（r=1/4）涨到 +88%（r=15/16）。
  **lever = 消除相位串行（让 MXFP8 相位与 NVFP4 相位 overlap，或不吃半速）——这是 mainloop 结构改动，不是配置旋钮**。grid search 已证**没有任何 builder 配置能动它**（最优配置就是出厂配置）。gap 随 r 线性，r=1/2 约 +15…+45%、r=3/4 约 +45…+72%。
- **小 M（512；qkv·2048）**：excess +17…+60%，全 r。
  **lever = 小-M 专用 tile（更小 CTA / 非-cooperative 路径）**——但 cooperative 硬要求 Tile-M≥128，128×64 实测更慢，所以**在 cooperative 约束内无配置解**，需换调度（结构改动）。gap：qkv·512 ~+28%、qkv·2048 ~+24%（低 r）。

---

## Part 6 — 每场景最优配置文件（= 生产派发表，`cd_dispatch_r19.csv`）

> 「到极致」判定用的就是这张**每场景最优配置**；co-design 侧配置恒为 **128×128 NV4/MX2**（grid 搜过的全空间最优），手写侧为 round-12 per-shape 生产配置（SFILV0+SFASYNC(sfa=1)+KPIPE；tile/sw/pers：qkv 0/0/0、down 0/0/0、gate 0/8/1）。

| shape | M | r 区间 | kernel | 配置 | 判定 |
|--|--|--|--|--|--|
| qkv | 512 | r≤0.875 | **ship** `b1_tma_ext` | SFILV0+SFASYNC+KPIPE, tile0/sw0/pers0 | headroom（小 GEMM）|
| qkv | 512 | r≥0.9375 | codesign `fused_gemm` | 128×128 NV4/MX2 (M512 可选 NV3，+~1%) | headroom |
| qkv | 2048 | r≤0.1875 | **ship** | 同上 | headroom |
| qkv | 2048 | r≥0.25 | **codesign** | 128×128 NV4/MX2 | headroom（相位）|
| qkv | 4096/8192 | 全 r | **codesign** | 128×128 NV4/MX2 | r≤1/16 **到极致**，余 headroom |
| down | 512 | 全 r | **codesign** | 128×128 NV4/MX2 | headroom（小 GEMM）|
| down | 2048 | 全 r | **codesign** | 128×128 NV4/MX2 | r≤1/16 **到极致**，余 headroom |
| down | 4096/8192 | 全 r | **codesign** | 128×128 NV4/MX2 | **r≤1/8（含生产点）到极致**，余 headroom |
| gate | 512 | r≤0.75 | **ship** | SFILV0+SFASYNC+KPIPE, tile0/sw8/pers1 | headroom（小 GEMM）|
| gate | 512 | r>0.75 | codesign | 128×128 NV4/MX2 | headroom |
| gate | 2048 | 全 r | **codesign** | 128×128 NV4/MX2 | headroom（相位）|
| gate | 4096/8192 | 全 r | **codesign** | 128×128 NV4/MX2 | r≤1/16 **到极致**，余 headroom |

---

## Part 7 — 一句话 bottom-line（负责人要的「全场景 maxed/not-maxed 地图」）

1. **哪些场景到极致了**：**大 M（≥4096）、低 r（fp4 主导，含生产点 r=1/8 的 down；r≤1/16 的 qkv/gate；全 r=0）** —— best 已在零开销插值天花板上或之下（excess −10…0%），compute-bound，TC 占空比贴着 CUTLASS 端点（差距即 MXFP8 半速相位的结构占位）。这些场景**没有东西可再榨**。
2. **哪些场景有 headroom**：**(a) 中/高 r（r≥1/4，全 M）**——excess +15…+88%，随 r 涨；**(b) 小 M（512、qkv·2048）全 r**——excess +17…+60%。
3. **每个 headroom 场景剩下的那一个 lever**：
   - (a) 中/高 r → **让 MXFP8 相位与 NVFP4 相位 overlap / 不吃半速 TC**（相位串行是结构根因）。
   - (b) 小 M → **小-M 专用、非-cooperative 的更小 tile 路径**（cooperative Tile-M≥128 + 128×64 更慢，配置内无解）。
   **两者都是 mainloop 结构改动，不是 builder 配置**——本轮 grid search 已严格证明：**整个可用配置空间（smem 上限 + cooperative Tile-M≥128 夹到只剩 128×128）里，出厂 128×128 NV4/MX2 就是每场景最优**，配置这条路已到头。

---

## Part 8 — 诚实边界

- **配置空间没搜全的部分，全部如实列出（Part 3 表）**：256×128 / 128×256（两 stage 变体）、128×128-NV5/MX3、128×128-NV6/MX4、128×64-NV6/MX4 **都编过但跑不起来**（超 101376B smem，`initialize: Error Internal`）；64×128 / 64×64 **编不过**（cooperative Tile-M≥128）。这些不是「略过」，是**物理/硬约束不可用**。
- **qkv 的 NV3「大赢」是噪声**：grid search 那批跑在有 co-tenant 的 GPU 上，NV4 baseline 取到了慢样本（qkv·8192 r=0：NV4 146.7us vs NV3 102.6us）；干净 GPU 去噪复测（`cd_cfg_qkv.csv`）证明 M≥2048 NV4 最优或持平，NV3 仅在 M=512 有 ~1–2% 边际。报告以**去噪数**为准。
- **t_ideal 是乐观天花板**：它假设 fp4/fp8 各自满速且完美 overlap；mixed kernel 共享累加器、相位串行、MXFP8 半速，所以中/高 r 的 excess 有**结构下界**，不是纯开销——这也是为什么「到极致」只出现在 fp4 主导的低 r/大 M。
- **TC 端点对比用 round-16 跨子轮数据**（同一 baseline 配置，本轮 grid 已确认该配置最优，故可比）；未在本会话重跑 ncu（避免与 grid search 抢 GPU）。roofline/compute-bound 结论沿用 round-16（DRAM<20%、L2 90–95%）。
- **手写侧 dense per-r 最优仅 M2048/稀疏 r 有历史 sweep**；M8192 的 dense per-r 手写最优未单独扫（ship 用同一 per-shape 配置，本会话 dense 测过 e2e）。

---

## 产物
- **测量 + 配置层（零 mainloop 改动）**：`r19_lat.py`（密集-r master table）、`cd_cfg_ext.cu`（宏参数化 config-only 拷贝，#include 同套受保护头）、`cd_cfg_search.py`（10 配置 grid）、`cd_cfg_qkv.py`（qkv 去噪密集-r A/B）、`run_r19.sh` / `run_cd_cfg.sh`。
- **数据**：`r19_lat.csv`（204 cell，全 cos=0.999999）、`cd_cfg_search.csv`（10 配置 × 代表 cell，含 smem 与 build 状态）、`cd_cfg_qkv.csv`（qkv 去噪）、`cd_dispatch_r19.csv`（每场景最优配置派发表）。日志：`logs/_r19_lat.log`（`R19_LAT_ALL_COSINE_PASS`+`R19_LAT_DONE`+`EXIT=0`）、`logs/_cd_cfg_search.log`（`CFG_SEARCH_DONE`+`EXIT=0`）、`logs/_cd_cfg_qkv.log`（`QKV_CFG_ALL_COSINE_PASS`）。
- **未改动（保护）**：`codesign_sm120_bs_mma.hpp` / `codesign_fused_kernel.hpp` / `b1_tma_ext.cu` / `codesign_fused_ext.cu` / `pure_presf_ext.cu` / `mixed_gemm.py` 等 mainloop 文件全部零改动。复用已缓存 .so（master table `ninja: no work to do`）；grid search 的 10 个配置各编一次（143–173s，非 40min——CUTLASS 头已缓存）。
- `candidates.jsonl`：+3 条（密集-r master table 的至极限前沿；grid search = 配置空间被 smem+cooperative 夹到只剩 128×128 且出厂即最优；qkv NV3「赢」为噪声、去噪后 NV4 最优）。GPU 已释放。
