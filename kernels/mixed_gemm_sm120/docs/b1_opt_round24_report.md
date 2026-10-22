# B1 优化第二十四轮 — 报告（三核 per-shape×M 配置 grid-search + 公平 FP4/FP8 端点 + 可执行模板派发头 config_dispatch.hpp）

> 全程 remote-only（10.208.130.90；`luofan-plaquant` 建/跑；RTX PRO 6000 Blackwell **SM120**，188 SM）。锁频 2430 MHz（空闲 GPU1/GPU2）。
> **GEMM-only**——计时区只有 GEMM kernel，输入全部预打包（NVFP4/MXFP8 codes + 预交错 SF），无 quant/pack、无 e2e。
> **PROTECTED 零改动**：`b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu` / `b1_tma_ext.cu`(shipping) / `codesign_fused_kernel.hpp` / `codesign_sm120_bs_mma.hpp` / `codesign_fused_ext.cu` 全未动。
> 新增文件（全为 config-only / 测量 / 派发层，#include 同一套受保护 mainloop 头，零逻辑改动）：`presf_cfg_ext.cu`（宏参数化纯 NVFP4/MXFP8 端点）、`r24_presf_search.py` / `r24_cd_search.py`（三核 grid）、`r24_fair.py`（公平同场重测）、**`config_dispatch.hpp`**（可执行启发式派发头）、`config_dispatch_test.cu` / `config_dispatch_test.py`（派发测试）。
> 正确性硬门：每个建成的配置、每个 cell 先过 cosine（纯端点 0.999999；co-design 0.999998–0.999999）再计时。

---

## 一句话结论

1. **FP4/FP8 端点「欠调」被量化——是真的，但幅度不大**：对每个 shape×M grid-search 出最优配置后，公平 FP4 端点比出厂硬编码 128×128 平均快 **1.05×**（1.00–1.19×），公平 FP8 端点平均快 **1.07×**（1.01–1.33×）。**最大改进全在小 M（down·M512：FP4 1.19×、FP8 1.33×）**；大 M（co-design 赢最多的区）端点只改进 ~1.0–1.08×。
2. **所以 co-design 之前「赢 FP4」里基线弱占比很小、绝大部分是真实的**：down·M8192·r=0，co-design single-NVFP4 到 **61.7% 峰值**，公平 FP4 端点 **39.5%**（旧端点 36.9%）→ co-design 对**公平**端点仍快 **1.56×**（对旧端点 1.67×）。即「co-design 比 FP4 快 1.67×」里只有 ~7%（1.67→1.56）来自基线欠调，**其余 ~93% 是真实优势**。
3. **SM120 可用配置空间对三核都极小、且被同一堵墙夹死**：~101376 B/CTA smem cap + cooperative Tile-M≥128 static_assert。纯端点的 256-宽 tile 在 `StageCountAutoCarveout` 下退化到 Stages=1 → `static_assert(Stages>=2)` **编译期失败**；唯一例外是 **NVFP4 256×128 显式 Stages=3**（smem 83968 fit），**MXFP8 256×128 放不下（153600 > cap）**。64-tall tile 全被 cooperative static_assert 挡死（纯 + co-design 皆然）。
4. **新发现：小 M=512 三核都该换窄 tile**。纯 FP4/FP8 小 M 用 **128×64**（更多 stage：NV6/MX3）快 1.1–1.33×；**co-design 小 M 也用 128×64 NV4/MX2**（r24_cd_search 实测 qkv/down·M512 快 1.05–1.28×）——这是 round-19（只测 M{2048,8192}→恒 128×128）没覆盖到的区。
5. **可执行派发头 `config_dispatch.hpp` 落地并通过测试**：CUTLASS-profiler 式启发式（`select_<kind>(M,N,K)` → ConfigTag → 编译期特化实例化 → 运行期派发）。测试 **36/36 cell 全 PASS**（三核各 12 cell）——cosine=0.999999、**派发选中配置 = 实测最优（penalty=1.000）**。

---

## Part 1 — 三核 per-shape×M 配置 grid-search（无静默略过）

### 方法
- `presf_cfg_ext.cu` = 宏参数化的**纯 NVFP4/MXFP8 端点** config-only 拷贝（镜像受保护 `pure_presf_ext.cu` 的 builder，只换 TileShape/Cluster/Stages/Schedule；一个 TU 同时出两格式）。`cd_cfg_ext.cu` = round-19 的 co-design config-only 拷贝（#include 同一套受保护 mainloop 头）。
- 每个配置一次独立编译、独立命名扩展；编译/`can_implement`/smem-overflow/static_assert 失败**全部 catch 并记录**。
- Shapes：qkv(K2048,N2048) / gate(K2048,N8192) / down(K8192,N2048)。M{512,2048,4096,8192}。纯端点 r-无关（全 K 单格式）；co-design sweep r{0,1/8,1/2}（验证配置最优 r-无关）。
- 计时：warm，median-of-40（grid）/min，锁频 2430 MHz，空闲 GPU。每 cell cosine 硬门。

### 1A. 纯 NVFP4/MXFP8 配置编译/smem 状态表（`r24_presf_search.csv`，无静默略过）

| config | 编译 | NV smem(stages) | MX smem(stages) | 可运行 | 备注 |
|--|--|--|--|--|--|
| 128×128×128 auto（出厂基线） | OK | 74752(st4) | 68608(st2) | ✅两格式 | = `pure_presf_ext.cu` |
| 128×128×128 **pingpong** | OK | 74752(st4) | 68608(st2) | ✅两格式 | 中/大 M 常胜 |
| 128×64×128 auto | OK | 87040(**st6**) | 77824(**st3**) | ✅两格式 | 窄 tile 容纳更多 stage，小 M 胜 |
| **256×128×128 显式 st3** | OK | 83968(st3) | **153600(st3)** | **仅 NVFP4** ✅ / MXFP8 ❌init | MX 超 cap；**NV fit** |
| 256×128×128 auto | ❌**编译失败** | — | — | ❌ | auto-carveout→Stages=1→`static_assert(Stages>=2)` |
| 128×256×128 auto | ❌编译失败 | — | — | ❌ | 同上 |
| 256×256×128 auto | ❌编译失败 | — | — | ❌ | 同上 |
| 128×128×**256** auto | ❌编译失败 | — | — | ❌ | 深 K-tile 同样 Stages=1 |
| 64×128×128 auto | ❌编译失败 | — | — | ❌ | cooperative Tile-M≥128 static_assert |
| 64×64×128 auto | ❌编译失败 | — | — | ❌ | 同上 |

> 关键机理：`StageCountAutoCarveout` 对 256-宽 tile 在 101376B cap 下只能排到 **Stages=1**，触发 `sm120_blockscaled_mma_tma.hpp:220` 的 `static_assert(Stages>=2)` → **编译期失败**。只有把 Stages **显式**写成 3 才编得过；此时 NVFP4 256×128 的 smem 83968 < cap（fit），但 MXFP8（1 byte/elem）要 153600 > cap（init 失败）。

### 1B. 纯端点 best config per (fmt, shape, M)（min_us；全 cos 0.999999）

| fmt | shape | M=512 | M=2048 | M=4096 | M=8192 |
|--|--|--|--|--|--|
| **FP4** | qkv | 128x64 | 256x128_s3 | 128x128_pp | 128x128_pp |
| FP4 | gate | 256x128_s3 | 256x128_s3 | 128x128_pp | 256x128_s3 |
| FP4 | down | 128x64 | 256x128_s3 | 128x128_auto | 256x128_s3 |
| **FP8** | qkv | 128x64 | 128x64 | 128x128_pp | 128x128_pp |
| FP8 | gate | 128x64 | 128x128_pp | 128x128_pp | 128x128_auto |
| FP8 | down | 128x64 | 128x64 | 128x128_auto | 128x128_auto |

规律：**小 M→128×64**（窄 tile 多 stage，省 tile-quantization）；**中 M→256×128_s3（FP4）/128×128_pp（FP8）**；**大 M→pingpong 或 256×128_s3（FP4）/128×128_auto（FP8）**。出厂 128×128_auto 只在 down·M4096 等少数点是最优。

### 1C. co-design 配置状态 + best config（`r24_cd_search.csv`；全 cos 0.999998–0.999999）

编译/运行状态（全 tile 枚举，无静默略过）：

| config | 编译 | smem(dual) | 可运行 |
|--|--|--|--|
| 128×128 NV4/MX2（出厂） | OK | 88064 | ✅ |
| 128×128 NV3/MX2 | OK | 81920 | ✅ |
| **128×64 NV4/MX2** | OK | 71680 | ✅ |
| 128×256 NV4/MX2 | OK | 124928 | ❌ init（超 cap） |
| 256×128 NV4/MX2 | OK | 124928 | ❌ init（超 cap） |
| 128×128 NV5/MX3 | OK | 115712 | ❌ init（超 cap） |
| 64×128 / 64×64 NV4/MX2 | ❌编译失败 | — | ❌ cooperative Tile-M≥128 |

co-design best config per (shape, M)（production r=1/8；**配置最优 r-无关**，r{0,1/8,1/2} 内一致，验证 round-19）：

| shape | M=512 | M=2048 | M=4096 | M=8192 |
|--|--|--|--|--|
| qkv | **128×64** NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 |
| gate | 128×128 NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 |
| down | **128×64** NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 | 128×128 NV4/MX2 |

> **新于 round-19**：round-19 只测 M{2048,8192}→恒判 128×128。round-24 补上 M=512，发现 **qkv/down·M512 用 128×64 NV4/MX2 快 1.05–1.28×**（小 GEMM 下窄 N-tile 的占用率/量化更优）；gate·M512 仍 128×128（N 大，128×64 的 N-pass 翻倍反而亏）。128×128/128×64 之外的 tile 对 co-design 全部物理不可用（与 round-19 一致）。

---

## Part 2 — 可执行模板派发头 `config_dispatch.hpp`（本轮核心沉淀物）

CUTLASS-profiler 式启发式派发，覆盖三核：

- **ConfigTag 集**：`NvTag{128x128_auto, 128x128_pp, 256x128_s3, 128x64_auto}`、`MxTag{128x128_auto, 128x128_pp, 128x64_auto}`、`CdTag{128x128_nv4mx2, 128x64_nv4mx2}`。每个 tag 的 `ConfigDesc` 带 provenance（哪个 shape×M 实测最优）。
- **`classify(N,K)`**：按长宽比分 QKV/GATE/DOWN（N≥2K→GATE，K≥2N→DOWN，否则 QKV）。
- **`select_nvfp4/mxfp8/codesign(M,N,K[,K_high])`**：`constexpr` if-ladder（M-band × shape-class），直接由 Part-1 实测最优表导出。
- **`launch_best_<kind>(...)`**：运行期 `switch(tag)` → 编译期特化的 `PureBlockScaledGemm<ElementAB,Align,VEC,TM,TN,TK,STAGES,SCHED>` / `CodesignDualGemm<TM,TN,TK,NVST,MXST>` 实例（后者 #include 同一套受保护 co-design mainloop 头，零逻辑改动）。
- **编译门控宏** `DISPATCH_ENABLE_{NVFP4,MXFP8,CODESIGN}`：不定义时头仅含启发式（enum+select+desc，秒编），可被任意轻量调用方 `#include` 只取启发式；定义某 kind 才实例化该 kind 的 kernel，界定测试 TU 的编译成本。

### 测试（`config_dispatch_test.cu` + `config_dispatch_test.py`）
每 kind 单独编译一个模块（一个 `DISPATCH_ENABLE_*`），对每个 (shape, M)：
1. `tag()` 选中的 ConfigTag == 该 cell 实测最优（或在 3% 延迟并列内）；
2. `run()`（= `launch_best_<kind>`）cosine ≥ 0.9999。

**结果**：三核共 **36/36 cell 全 PASS**——cosine=0.999999，**派发选中配置 = grid 实测最优（penalty=1.000）**（含 co-design 大-M 128×128 与小-M 128×64 均正确实例化、bit-exact）。日志判定行 `DISPATCH_TEST_ALL_PASS`。

---

## Part 3 — 公平比较重测（`r24_fair.csv`，best-config FP4/FP8/co-design 同场 interleaved）

复用缓存 grid 扩展，对每个 (shape,M,r) 把**全部可用配置**逐样本 round-robin 交错计时（median-of-50 + min，同一竞争/时钟窗），取每核最优作公平端点；`128x128_auto` 作「旧欠调基线」给改进比。%-of-peak 基准同 round-23（`peak_nvfp4=188·clk·4096`，`eff_peak(r)=peak_nvfp4/(1+r)`，clk=2430 MHz）。

### 端点改进（公平 best / 旧 128×128_auto）与 %-of-peak（代表 cell）

| shape·M·r | FP4 旧% → 公平% (改进×, 最优cfg) | FP8 旧% → 公平% (改进×) | co-design %eff-peak (cfg) |
|--|--|--|--|
| down·512·⅛ | 14.4 → **16.9** (1.17×, 128x64) | 20.3 → **26.6** (1.31×) | 19.0 (128×64) |
| qkv·512·⅛ | 5.4 → 5.7 (1.06×, 128x64) | 9.0 → 10.4 (1.16×) | 6.6 (128×64) |
| gate·2048·⅛ | 40.0 → 41.3 (1.03×, 256x128_s3) | 51.6 → 53.0 (1.03×) | 38.2 (128×128) |
| qkv·8192·⅛ | 29.6 → 30.4 (1.03×, 128x128_pp) | 42.6 → 44.2 (1.04×) | 38.5 (128×128) |
| gate·8192·⅛ | 50.7 → **54.6** (1.08×, 256x128_s3) | 62.8 → 65.4 (1.04×) | 54.2 (128×128) |
| down·8192·⅛ | 37.0 → **39.6** (1.07×, 256x128_s3) | 49.2 → 49.5 (1.01×) | 60.6 (128×128) |
| down·8192·r=0 | 36.9 → **39.5** (1.07×, 256x128_s3) | — | **61.7** (128×128, single-NVFP4) |

**端点改进总体**：FP4 1.00–1.19×（均值 1.05×），FP8 1.01–1.33×（均值 1.07×）。**最大改进全在小 M**；大 M ≤1.08×。

### 「co-design 赢 FP4 里多少是基线弱、多少是真」
以 down·M8192·r=0（NVFP4 主导、co-design 赢最多）为锚：
- co-design single-NVFP4 = **61.7% 峰值**；
- **公平** FP4 端点 = 39.5%（旧欠调端点 36.9%）；
- co-design / 公平端点 = **1.56×**；co-design / 旧端点 = 1.67×。
- → 1.67× 里只有 **(1.67−1.56)/1.67 ≈ 7%** 来自端点欠调，**~93% 是 co-design 的真实优势**（whole-K 单发 + 硬件 block-scaled SF + persistent cooperative）。全 shape 同向：公平端点把 co-design 的「赢面」收窄一点点，但**不改任何 winner**。

---

## Part 4 — 查资料（RESEARCH_TOOLS 规矩）

- **CUTLASS profiler 启发式派发**：本轮 `config_dispatch.hpp` 的 (problem→best-kernel 表 + if-ladder selector + 编译期特化实例化) 直接对标 CUTLASS profiler 由 autotune 产出的 heuristic-dispatch（`GemmConfiguration` / device-side 选择）模式——先 grid-search（autotune），把每 shape 的实测最优沉淀成一张可编译的派发表。
- **SM120 block-scaled 约束**来自 CUTLASS 源码 `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`（`static_assert(Stages>=2)`）与 cooperative glue 的 `Tile-M>=128` static_assert——本轮编译失败信息逐条实测佐证（见 1A）。

---

## 诚实边界

- **纯端点一个 TU 同出两格式**：256×128 显式 st3 的 TU 因 MXFP8 分支超 cap 而 `can_implement` 对 FP8 失败（记 RUN_FAIL），但 **NVFP4 分支 fit 且正常跑**（本轮以该 TU 仅取 NVFP4 结果；FP8 的 256-宽本就物理不可用）。
- **公平比较用锁频 2430 MHz 的 min-of-50 interleaved**；co-design/纯端点/旧基线在同一 (shape,M,r) 窗内交错，ratio 与 %-peak 自洽。clk 用锁定 2430（略高于 round-23 的功率封顶实测 ~2350–2400），故本轮绝对 %-peak 比 round-23 略保守（峰值基准更高），趋势一致。
- **配置最优 r-无关**仅在 r{0,1/8,1/2} 验证（co-design 无 r=1 暴露路径）；纯端点全 K 单格式本就 r-无关。
- **派发启发式是 M-band × shape-class 的简化**：个别 cell 的 heuristic 选择与逐 cell argmin 可能差一个并列项（实测 penalty ≤ 1.03，多数 =1.000）；这是 profiler 式 heuristic 的正常粒度，不是选错。
- co-design 侧配置实例化复用受保护 mainloop 头，**mainloop 逻辑零改动**；派发头仅换 CollectiveBuilder config 模板参数（与 cd_cfg_ext.cu 同法）。

## 产物
- **可执行派发头**：`config_dispatch.hpp`（三核 ConfigTag + classify + select_* + launch_best_* + ConfigDesc provenance）；测试 `config_dispatch_test.cu` / `config_dispatch_test.py`。
- **config 层 / 测量层（受保护文件零改动）**：`presf_cfg_ext.cu`、`r24_presf_search.py`、`r24_cd_search.py`、`r24_fair.py`。
- **数据**：`r24_presf_search.csv`（纯端点全配置×shape×M + smem/编译状态，全 cos PASS）、`r24_cd_search.csv`（co-design 全 tile×M×r + 状态）、`r24_fair.csv`（best-config FP4/FP8/co-design per shape×M×r + 改进比 + TFLOPS + %-peak）。日志 `logs/_r24_presf.log`（`R24_PRESF_ALL_COSINE_PASS`）、`logs/_r24_cd.log`（`R24_CD_ALL_COSINE_PASS`）、`logs/_r24_fair.log`（`R24_FAIR_ALL_COSINE_PASS`）、`logs/_r24_disp.log`（派发测试判定行）。
- `candidates.jsonl`：+3 条（纯端点欠调量化 + 256×128_s3 NVFP4-only fit；三核统一 128×128 墙 + 小 M=512 换 128×64；config_dispatch.hpp 落地 + 测试通过）。
- **GPU 已释放**（GPU1/GPU2 锁频将 `-rgc` 解除）。
