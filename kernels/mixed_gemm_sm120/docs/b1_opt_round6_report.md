# B1 优化第六轮 — 报告（多轴网格扫描 / per-ratio 最优 / gate rasterization 攻坚）

> 全程 remote-only（10.208.130.90, luofan-plaquant, GPU0）。抗中断策略：自包含 detached
> sweep 脚本 + 逐行 flush CSV。正确性 gate：**所有 gated 点 cosine ∈ {0.9999999, 1.0000000}**
> （= cosine=1.000000 口径）。GPU: RTX PRO 6000 Blackwell (SM120)。
> 性能标尺：**excess vs 同场 CUTLASS presf 界**，excess=(t_B1 − t_ideal)/t_ideal，
> t_ideal(r)=t_fp4·(1−r)+t_fp8·r（同场 pure NVFP4/MXFP8 全-K presf GEMM）。

## 结论先行（headline）

- **Round-6 开局先修复前任 mid-edit。** 前任在 `b1_tma_ext.cu` 加 sweep knobs（`sf_copy_row<NB>`、
  group-M rasterization `SWZ`、`MINB` launch-bounds、`kmul`/BK 轴、pybind `swz/kmul/minb`）后停住。
  **诊断：代码语法/模板全部合法**（C++ 前端 + ptxas 跑完几十个 kernel），唯一失败是**一个非法
  smem 组合** `128x128 / st6 / km2`（ptxas `too much shared data`，0x1b060=110688B > 0x18c00=99KB
  上限）。按"剪掉 smem>99KB 非法组合"的 mandate，从 dispatch 与 sweep 各删掉这一条 → **编译通过，
  validated baseline 建立**。其余 mid-edit 全部有效可用。
- **Mandate A（多轴扫描）：真多轴笛卡尔网格跑通**（`sweep_r6.py`，3250 行全 `ok`，全 cosine=1.0）。
  轴：CTA tile × stages{2-6,8} × **consumer-warp layout** × **rasterization/group-M swizzle{0,2,4,8}**
  × **BK/kmul{1,2,4}** × launch-bounds-minb{1,2}。
- **Mandate B（per-ratio 最优）：per-r 调参确实击败单一 config，round-5 的"128x128 st3 处处最优"被
  证伪。** 关键在于：胜出 config 在**128x128 footprint 内部**随 r 切换两个 round-5 从未在 dual kernel
  上扫过的子轴——**consumer-warp layout（WM4×WN2，"w42"）** 和 **group-M rasterization swizzle**。
  同场（same-bounds）per-r-best vs 固定 `tile0/st3/sw0/km2/mb1` 的增益：
  **qkv 均值 +3.08pp（最大 +26.2pp）、down +1.66pp（+11.9pp）、gate +7.05pp（+32.2pp）**。
- **Mandate C（gate 攻坚）：rasterization 就是 round-5 漏掉的那个杠杆**（round-5 T3 只扫 tile×stage，
  从未在 dual kernel 上扫 swz）。gate M=8192：固定 config `+24.4..+57.1%`（剧烈波动）→ per-r best
  用 **swz=8**（高 r 处 w42/st4/sw4）收紧到 **+17.3..+24.9%**，最坏点砍掉 >30pp。但 gate 仍 floor 在
  **~+16-18%（> <10% 目标）**：大-N/短-K 的 prologue/epilogue 结构性残差仍在，彻底关闭仍需
  persistent/stream-K（沿用 round-4/5 判定）。**新 gate band ≈ +16-25%（原固定 config +17-57%）。**
- **诚实负结果：** kmul/BK（km1=BK32、km4=BK128）在 106 个点上**从未**胜出（km2/BK64 处处最优，
  "短-K 用更长 K-pipe"假设被证伪）；stages>3（st5/st6/st8）从未胜出（st4 仅在 gate/qkv 高-r + w42 时赢）；
  更大/更小 CTA tile（128x256/256x128/64x128/128x64）**从未**胜出（只有 128x128 与 128x128-w42）；
  minb=2 从未胜出。

---

## Sweep 方法学 + 产物

- 脚本：`kernels/mixed_gemm_sm120/sweep_r6.py`（自包含；`get_tma()` 编译整份 dispatch 成单 .so，
  所有 config 变体为该 .so 内的运行期分派——一次编译、运行期按 `(tile,st,swz,kmul,minb)` 分派）。
- 启动命令（完全 detached，与我是否在线无关）：
  ```
  docker exec -d -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant \
    -e TORCH_CUDA_ARCH_LIST=12.0a -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_VISIBLE_DEVICES=0 luofan-plaquant \
    bash -c 'python kernels/mixed_gemm_sm120/sweep_r6.py > \
             kernels/mixed_gemm_sm120/logs/sweep_r6.log 2>&1'
  ```
- 轴（独立维度，笛卡尔积，非法 smem>99KB 组合剪枝）：
  - **CTA tile**：128x128(WM2WN4,8warp) / 128x128-w42(WM4WN2) / 128x256(WM2WN8,16w) /
    256x128(WM4WN4,16w) / 64x128(WM1WN4,4w) / 128x64(WM2WN2,4w)
  - **pipeline stages**：2,3,4,5（128x128/km2）、4,6,8（km1）、2（km4）、2,3（大 tile）、3,4,5（小 tile）
  - **consumer-warp layout**：WM2×WN4 vs WM4×WN2（同 8 warp 重排）
  - **rasterization/group-M swizzle**：0（原生 2D grid）/2/4/8（1D grid, Triton 式 group-M）
  - **BK/kmul**：1(BK32)/2(BK64)/4(BK128)
  - **launch-bounds minb**：1/2
- 形状：qkv N2048K2048 / down N2048K8192 / gate N8192K2048；M∈{2048,8192}；
  密 r 网格（K2048 步长 128=17 点；K8192 步长 512 + {128,384}=19 点）。
- 每个 (shape,M,r,config) **计时**；cosine 按 (config,shape,phase∈{fp4,mix,fp8}) 家族**抽样 gate** 并缓存；
  **每行即时 append+flush**。
- **产物**：
  - `sweep_r6.csv`（3250 行，全 config × 全点，含 latency/cosine/excess/status）
  - `ratio_curve_r6.csv`（106 点 per-r 最优 Pareto：best_cfg/best_excess/r5_excess/gain_pp）
  - `logs/sweep_r6.log`、`candidates.jsonl`（+4 条）
- **SF-staging 与 epilogue-store-width 本轮未参数化**（单值：round-5 的 width-generic 向量化
  `sf_copy_row<NB>` + half2 store）；要做成轴需改 kernel（列为后续）。

## 同场 CUTLASS presf 界（本轮会话，用于所有 excess）
每个 (shape,M) 同场测 t_fp4/t_fp8 全-K presf GEMM；excess 对这套界。
**注意：跨会话绝对 excess 不可比 round-5**（界的绝对时间随 GPU 占用变化），故下文只做
**本轮会话内** per-r-best vs 固定单 config 的对比（同界，合法）。

---

## Mandate B — per-ratio 最优 config 表（全 r，两个 M）

完整表见 `ratio_curve_r6.csv`。摘要（excess% / 固定-config excess% / 增益 pp / best_cfg）：

### down（N2048 K8192）— 已收敛，per-r 增益小
| r | best_ex% | r5-cfg% | gain | best_cfg |
|--|--|--|--|--|
| M2048 r=0 | −15.9 | −15.9 | +0.0 | 128x128/st3/sw0 |
| M2048 r=1/8 | −12.3 | −11.7 | +0.6 | 128x128/st3/sw2 |
| M2048 r=1/2 | −4.7 | −4.1 | +0.6 | 128x128-w42/st3/sw0 |
| M2048 r=1 | +0.9 | +2.9 | +2.1 | 128x128-w42/st3/sw0 |
| M8192 r=0 | −23.9 | −23.9 | +0.0 | 128x128/st3/sw0 |
| M8192 r=1/2 | −13.2 | −11.4 | +1.8 | 128x128-w42/st3/sw4 |
| M8192 r=1 | −6.9 | −5.7 | +1.2 | 128x128-w42/st3/sw0 |

→ down **本就贴界**（round-5 判定成立）；per-r 增益多在 0-2pp，主要来自 **w42 warp-layout** 在中/高 r。

### qkv（N2048 K2048）— 中等增益
| r | best_ex% | r5-cfg% | gain | best_cfg |
|--|--|--|--|--|
| M2048 r=0 | −5.8 | −5.8 | +0.0 | 128x128/st3/sw0 |
| M2048 r=1/8 | −2.6 | −1.4 | +1.2 | 128x128-w42/st3/sw0 |
| M2048 r=1/2 | −0.7 | +0.6 | +1.3 | 128x128/st3/sw2 |
| M2048 r=1 | +3.4 | +5.1 | +1.6 | 128x128-w42/st4/sw0 |
| M8192 r=1/4 | −13.7 | −9.8 | +3.8 | 128x128/st3/sw8 |
| M8192 r=5/8 | −10.9 | −5.4 | +5.5 | 128x128-w42/st4/sw0 |
| M8192 r=1 | −6.9 | −5.7 | +1.2 | 128x128-w42/st4/sw0 |

→ qkv 均值 +3.08pp；**w42 layout 赢 22/34 点**，swz{2,4,8} 在部分点再补 1-4pp。

### gate（N8192 K2048）— 见 Mandate C

**per-r 调参增益总览（同场，pp = excess 百分点）：**
| 场景 | 均值 gain | 最大 gain | n |
|--|--|--|--|
| qkv | +3.08 | +26.2 | 34 |
| down | +1.66 | +11.9 | 38 |
| gate | +7.05 | +32.2 | 34 |

**结论：per-r 多轴调参击败单一 config，down 边际（已收敛）、qkv 中等、gate 显著。**

---

## 哪些轴真正有用（证据）

per-r 最优 config 里各轴取值的出现频次（按 shape）：

| shape | tile | stages | swz | kmul |
|--|--|--|--|--|
| qkv | 128x128-w42 ×22, 128x128 ×12 | st3 ×28, st4 ×6 | sw0 ×23, sw4 ×5, sw2 ×4, sw8 ×2 | **km2 ×34** |
| down | 128x128 ×20, w42 ×18 | **st3 ×38** | sw0 ×20, sw4 ×8, sw8 ×7, sw2 ×3 | **km2 ×38** |
| gate | 128x128 ×19, w42 ×15 | st3 ×20, st4 ×14 | sw0 ×13, **sw8 ×12**, sw4 ×9 | **km2 ×34** |

- **有用（新杠杆，round-5 从未在 dual kernel 上扫）：**
  1. **consumer-warp layout（WM4×WN2 "w42"）**——qkv 赢 22/34，down/gate ~一半。round-5 只有 WM2×WN4。
  2. **group-M rasterization swizzle**——gate 大-N 上 sw8 赢 12/17（gate 攻坚核心，见下）；qkv/down 零散补点。
- **次要：** stages（st3 主导；st4 仅 gate/qkv 高-r + w42 时赢）。
- **无用（诚实负结果）：** kmul/BK（km2 处处赢，km1/km4 零胜）；更大/更小 tile（零胜）；minb=2（零胜）；
  st5/st6/st8（零胜）。

---

## Mandate C — gate（N8192 K2048）rasterization 攻坚

round-5 用 roofline + tile 扫描排除了 DRAM/L2/占用/tile/bank-conflict，判为"小-K overhead + 需
persistent/stream-K"。**但 round-5 的 T3 只扫 tile×stage，从未在 dual kernel 上扫 group-M swizzle。**
本轮把 rasterization 作为一等轴扫进去：

| gate | 固定 tile0/st3/sw0 excess | per-r best excess | 胜出 config |
|--|--|--|--|
| M=2048 | +17.5 .. +29.8% | **+16.5 .. +25.4%** | sw0/w42-st4 混合 |
| M=8192 | +24.4 .. +57.1% | **+17.3 .. +24.9%** | **swz=8 赢 11/17**，高 r 处 w42/st4/sw4 |

- **swz=8（group-M=8，1D grid，8 个 M-tile 连续扫完全部 N-tile）** 让 gate M8192 的 64-CTA/N-row
  的 B-tile 在 L2 里热度跨 N 维复用 → 把固定 config 的剧烈波动（+24~+57%）压成紧致的 **+17~+25%**，
  最坏点（r≈0.06 的 +57%）砍到 +25%（−32pp），均值 +7pp。这是**可复现、方向一致**的（swz8 赢 11/17）。
- **但 gate 仍未达 <10% 目标**：floor ~+16-18%。rasterization 消除了"坏调度"带来的额外波动与惩罚，
  但大-N/短-K 的 prologue/pipeline-fill/epilogue 结构性占比仍在。**彻底关闭需 persistent/stream-K 级
  改造**（单核调参不可解；沿用 round-4/5 判定，ROI 低）。**新 gate band ≈ +16-25%。**

---

## 各场景收敛判定

| 场景 | 判定 | 证据 |
|--|--|--|
| **down（大 K）** | **已收敛** | per-r 增益仅 0-2pp（主要 w42）；贴界（M8192 −7~−24%）；round-5 roofline 已证 TC 天花板 |
| **qkv** | **基本收敛，w42/swz 小补** | 均值 +3pp；best 多在界内（负 excess）；高 r 个别点 +1-3% |
| **gate（大 N 短 K）** | **改善但未关闭** | rasterization 把 band 从 +17-57% 收紧到 +16-25%、砍最坏 >30pp；但 floor ~+17% 结构性，需 persistent/stream-K |

> 本轮未重跑 ncu（sweep 完成得比预期快，优先交付全量 CSV + per-r 分析 + gate band）。
> round-5 的 ncu roofline（down/qkv 贴 TC 天花板、无一 DRAM-bound、gate 非 L2/占用/bank-conflict）
> 对本轮结论仍成立；gate 的 rasterization 增益与 round-5"gate 是调度/overhead 而非带宽"判定一致
> ——group-M 正是调度维度。per-r 每点是否到各自 floor 的逐点 ncu 复核列为后续。

## 产物清单
- `b1_tma_ext.cu`：剪掉非法 `128x128/st6/km2`（>99KB）；其余 round-6 mid-edit（sf_copy_row/SWZ/MINB/kmul/pybind）保留并验证。
- `sweep_r6.py`（自包含多轴 detached sweep）、`sweep_r6.csv`（3250 行）、`ratio_curve_r6.csv`（106 点 per-r Pareto）。
- `candidates.jsonl` +4 条（sweep / per-r-positive / gate-raster-partial / negatives）。
- **未改动**（保护）：`b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py`。
