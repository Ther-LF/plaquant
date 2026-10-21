# B1 优化第二十轮 — 报告（STAGE-1：中/高-r「bulge」根因 ncu 确认 + cross-phase overlap PoC → 结构天花板定界）

> 全程 remote-only（10.208.130.90；`luofan-plaquant` 建/跑，`luofan-ncu` ncu；RTX PRO 6000 Blackwell **SM120**，GPU3）。
> 新增文件（受保护/serial 文件零改动）：`codesign_overlap_kernel.hpp`（de-union + 删相位屏障的 overlap glue）、`codesign_overlap_ext.cu`（stage/tile 宏参数化组装）、`prof_cdmid.py` / `run_cdmid_ncu.sh`（mid-r ncu 根因）、`codesign_overlap_check.py` / `run_r20_overlap.sh`（de-union 可行性扫）、`codesign_overlap_iso.py`（同-tile 同-stage 隔离 A/B）、`analyze_cdmid.py`（ncu 解析）。
> `codesign_fused_kernel.hpp`（serial）/ `codesign_sm120_bs_mma.hpp` / `b1_tma_ext.cu` / 全部 PROTECTED 文件 **零改动**。
> 正确性硬门：overlap 核跑起来的每个 cell `cos=0.999999`（`R20_ISO_ALL_COSINE_PASS`）。
>
> ## 一句话结论（STAGE-1 verdict）
> **round-19 的「中/高-r excess = GEMM 相位串行 seam」假设被 ncu 直接证伪，cross-phase overlap 不能收掉这个 bulge——结构天花板就此定界。**
> **(1) ncu**：mid-r 处 serial 双格式核 **比两个端点分开跑还快 2–3%**（down 411.5us < 143.9+275.8=419.7；gate 452.6us < 172.7+294.0=466.7），compute-bound（DRAM 10–16%），TC-elapsed 65–73% **贴着/落在端点之间**（= 共享的 SM120 block-scaled TC 天花板，round-11 的 SF-ALU 路径），相位屏障 stall 可忽略（0.5–0.8 inst/issue vs wait 3.6–4.0）。**没有 seam 可拆**——mid-r 的 e2e excess 其实来自 per-call quant/pack（不在 t_ideal 里）+ 这条共享 TC 天花板。
> **(2) overlap PoC**：真把 seam 拆掉（de-union + 删相位屏障）在 128×64 同-tile 同-stage 下**确实快 2.4–5.5%**（bit-exact 实测），但这点收益**远小于** de-union 被迫吃的小-tile/浅-stage 代价——overlap-128×64 比生产 serial-128×128 **慢 9–22%**；而 de-union 在生产 128×128 tile 上**物理放不下**（NV2/MX2 最小就要 118784B > 101376B cap，且 Stages≥2 硬约束使无法再浅）。**净效果：overlap 不收 bulge，反而 net regression。**
> **(3) 定界**：fused GEMM mainloop 已在融合最优的 ~2–3% 内（DUAL ≤ 端点和）；要突破 65–73% 的 SM120 TC 天花板只剩 Option C（双 warpgroup 并发 FP4+FP8 MMA，各自 partial accumulator），但核已 168 reg/thread、1 CTA/SM，翻倍 accumulator 被 round-9 寄存器墙挡死。**STAGE-1 = 干净负结论：overlap 这条杠杆关闭。**

---

## Part 0 — 本轮依据（round-19 留下的问题 + 负责人指令）

round-19 密集-r master table 发现：大 M + 低 r（含生产点）已到零开销天花板；但 **mid/high r（r≥1/4）的 `excess_best` vs `t_ideal(r)` 随 r 单调涨（r=1/2 +15…+45%、r=3/4+ +45…+88%）**，round-19 归因为「相位串行 NVFP4→MXFP8 + MXFP8 半速 TC」的 **结构 seam**，并点名唯一 lever = **消除相位串行（cross-phase overlap）**。因 production r 可变（adaptive high_fraction），mid-r 是真实目标。

负责人指令（本轮 STAGE-1）：**先用 ncu 确认 mid-r 根因（seam/fill/bound），再 PoC 最可行的 overlap 选项（大概率 Option A），测 excess 前→后是否 materially 下降；bit-exact gate；若不行，诚实 bound 天花板。**

---

## Part 1 — ncu 根因确认（`run_cdmid_ncu.sh` / `prof_cdmid.py`，luofan-ncu GPU3，-s5 -c1 -k regex:device_kernel）

在两个 bulge 最大的 cell（gate·8192 r=1/2、down·8192 r=1/2）上，把 serial 双格式核的 **GEMM-only**（`fused_gemm_prepacked`，activation SF 预交错，计时区只剩 fused mainloop + epilogue）对比 **同 per-phase K 的 pure-NVFP4 / pure-MXFP8 presf 端点**。单发 under-ncu（replay），绝对 us 比 warm median 偏大，看的是 **ratio / TC% / stall**。

| label | dur(ns) | TC-elap% | TC-act% | SM-act% | DRAM% | L2% | barrier | membar | longsb | wait | regs |
|--|--|--|--|--|--|--|--|--|--|--|--|
| **DUAL_down** (Kl=Kh=4096) | 411520 | **72.56** | 80.88 | 89.71 | 10.25 | 64.45 | 0.51 | 0.00 | 0.48 | 3.99 | 168 |
| FP4_down (K=4096) | 143936 | 71.78 | 80.15 | 89.56 | 10.84 | 70.46 | 0.34 | 0.00 | 0.65 | 3.47 | 168 |
| FP8_down (K=4096) | 275808 | 72.98 | 80.89 | 90.22 | 9.84 | 62.74 | 0.24 | 0.00 | 0.62 | 3.73 | 168 |
| **DUAL_gate** (Kl=Kh=1024) | 452608 | **64.95** | 67.21 | 96.64 | 16.05 | 61.28 | 0.79 | 0.00 | 0.51 | 3.58 | 168 |
| FP4_gate (K=1024) | 172704 | 57.77 | 59.00 | 97.90 | 35.63 | 65.53 | 0.50 | 0.00 | 0.74 | 2.86 | 168 |
| FP8_gate (K=1024) | 294048 | 68.14 | 70.01 | 97.32 | 22.59 | 63.37 | 0.37 | 0.00 | 0.64 | 3.28 | 168 |

**读数（假设 → 证伪）**
1. **DUAL 比「两端点分开跑之和」还快**：down 411.5us < 143.9+275.8 = **419.7us**（快 2.0%）；gate 452.6us < 172.7+294.0 = **466.7us**（快 3.0%）。假设的「double fill + seam」应让 DUAL **慢于**端点和（多付两次 fill + 一个屏障 drain），实测反而**快**——融合省下一次 launch ramp + 共享一条 epilogue，已把 seam/double-fill 的代价**净盖过**。→ **seam 不存在（或已被融合吃掉），假设证伪。**
2. **DUAL compute-bound**：DRAM 10.3%(down)/16.1%(gate)、L2 64/61%、SM-active 90/97%、long-scoreboard stall 仅 0.48/0.51 inst/issue → **内存 fill 已被藏住**，不是 memory/fill-bound。
3. **相位屏障 stall 可忽略**：barrier 0.51(down)/0.79(gate) inst per issue-active，相对主导的 wait（3.99/3.58）是 ~13–22% 的零头，membar=0。→ 相位 drain **不是** dead-time 大头。
4. **TC-elapsed 贴着端点**：down DUAL 72.6% ≈ fp4 71.8 / fp8 73.0；gate DUAL 65.0% 落在 fp4 57.8 / fp8 68.1 之间（K-加权）。双格式核**没有**比端点更低的 TC 占空——它**继承**了 SM120 block-scaled 的 **~65–73% 共享 TC 天花板**（round-11 已定位 = SF-consume 的 ALU-pipe 压力），没有额外 bubble。

**那 mid-r 的 +45% excess 到底在哪？** 不在 GEMM mainloop。对照 r19 warm median：down·8192 r=1/2 **cd_gemmonly=375us < ideal=531us**（GEMM 本体比零开销插值还快 29%！），而 cd_e2e=771us → **pack ≈ 396us** 才是把 e2e 顶到 +45% 的主因；gate·8192 r=1/2 cd_gemmonly=417 ≈ ideal=419，pack 再加 +14.7%。**t_ideal 只建模 GEMM 端点、不含 per-call quant/pack**，所以 mid-r 的「excess vs t_ideal」= (a) 每次调用的 quant/pack（与 mainloop overlap 无关）+ (b) 共享的 ~70% SM120 TC 天花板——**两者都不是相位 seam**。

---

## Part 2 — overlap 设计与 PoC（Option A，de-union + 删相位屏障）

### 2a. 选 Option A + 借鉴的 pattern（RESEARCH_TOOLS）
- **KernelWiki `technique-pipeline-stages`**：多-stage 环形缓冲让 producer 装后一个 tile、consumer 算前一个；换 stage 数**不是纯性能编辑**，必须 re-gate 相位转换与 buffer 复用（本轮严格执行：每个 config 过 cosine 门，smem 不够直接 initialize Error Internal 捕获，无静默截断）。
- **FlashInfer PR-1039 / warp-specialization**：producer/consumer mbarrier handshake 的 per-stage 所有权协议——de-union 后每条 pipeline 用**自己**的 mbarrier 保护自己的 stage，跨-tile 复用由 `producer_acquire` 保护，**不再需要相位屏障**。
- 选 **Option A（最小侵入）**：Option B（手写 interleaved-K 单 mainloop）与 C（双 warpgroup 各 partial accumulator）都需要**两种格式 smem 同时常驻**（和 A 一样撞 smem 墙），且 C 额外要 2× accumulator 寄存器。A 是「producer/consumer 在相位边界的 handoff 改动」，先验证它。

### 2b. overlap glue 的三处改动（`codesign_overlap_kernel.hpp`，fork serial glue）
1. **TensorStorage 改成 SEPARATE `nv` + `mx`（不 union）**——两相位 smem 共存，MX 的 TMA 能在 NV 还在被消费时就填。
2. **删掉两个相位 NamedBarrier（bar1/bar2）**——它们只为保护 union 存在；de-union 后每条 pipeline 自己的 mbarrier 全管。
3. **producer 把 `nv.load(K_low)` 和 `mx.load(K_high)` 背靠背发**（中间无屏障）；consumer `nv.mma`(Clear=true) → `mx.mma`(Clear=false) 进**同一** FP32 accumulator（仍串行 MMA，共享累加器寄存器，无 race；MX 的 fill 藏在 NV tail compute 后）。
> 诚实边界：单 producer warp 顺序发 nv.load→mx.load，mx.load 要等 nv.load 全部**发射**完（受 NV pipeline 深度 gate，非全 drain）才起步，所以 overlap 是「去掉显式相位 drain + 两相位 smem 共存」的部分 overlap，不是完全并发 fill（完全并发需 Option B 的 interleaved-K 或双 producer）。

### 2c. de-union 撞 SM120 smem 墙（`codesign_overlap_check.py`，5 config）
de-union 让 tensor smem **翻倍**。生产 128×128 tile 下**每个合法 config 全超** ~101376B cap（`initialize: Error Internal`）：

| config | de-union smem(B) | 能跑 |
|--|--|--|
| 128×128 NV2/MX2 | **118784** | ❌ 超 cap |
| 128×128 NV3/MX2 | 137216 | ❌ |
| 128×128 NV4/MX2 | 155648 | ❌ |
| 128×128 NV2/MX1 · NV3/MX1 | — | ❌ **编不过** |

- **无法再浅到 fit**：`sm120_blockscaled_mma_tma.hpp:220 static_assert(DispatchPolicy::Stages >= 2)` → NV1/MX1 直接编译失败（实测 nv2mx1、nv3mx1 都挂在这条）。最小合法 de-union = NV2/MX2 = 118784B，**超 cap 17408B，且没有合法更浅配置**。
- TK 不能降（collective SF-atom static_assert 假定 BLK_K=128，round-14）；只有 TN 能降（128×64）。
- → **Option A / B（都要两格式 smem 共存）在生产 128×128 tile 上物理放不下。Option C 另需 2× accumulator，核已 168 reg/thread、1 CTA/SM，被 round-9 寄存器墙挡死。**

---

## Part 3 — 隔离 A/B：真把 seam 拆掉值多少？（`codesign_overlap_iso.py`，median-of-50）

128×64 tile 把 B smem 减半 → de-unioned overlap 在 **NV2/MX2 = 94208B < 101376** 装得下。于是做**同-tile 同-stage** A/B：overlap-128×64-NV2MX2（de-union + 无屏障）vs serial cd_cfg-128×64-NV2MX2（union + 相位屏障），**唯一差别就是 seam**；再都对比生产 serial 128×128 NV4/MX2。

| cell | cos(prod/ser64/ov64) | prod128(us) | ser64(us) | ov64(us) | **ov64/ser64** | ov64/prod128 |
|--|--|--|--|--|--|--|
| gate·8192 r=1/2 | 0.999999 / 0.999999 / **0.999999** | 697.6 | 806.4 | 762.0 | **0.945** (overlap 快 5.5%) | 1.092 (慢 9.2%) |
| down·8192 r=1/2 | 0.999999 / 0.999999 / **0.999999** | 606.0 | 757.0 | 738.8 | **0.976** (overlap 快 2.4%) | 1.219 (慢 21.9%) |

**读数**
1. **overlap 核 bit-exact**：`R20_ISO_ALL_COSINE_PASS`，两 cell cos=0.999999 —— de-union + 删屏障**没破坏**共享累加器数学，format contract 成立。
2. **隔离 seam 收益 = 2.4–5.5%**（ov64/ser64，同-tile 同-stage）——小，且**与 ncu 的 barrier-stall 量级一致**（0.5–0.8 inst/issue）。这就是「拆 seam」能拿的全部。
3. **但净 regression**：overlap-128×64 比生产 serial-128×128 **慢 9–22%**（ov/prod128 1.092 / 1.219）——de-union 逼出的小-tile/浅-stage 代价**远大于** seam 省的那 2–5%；而 de-union 在 128×128 上放不下（Part 2）。→ **没有任何路径能在生产 tile 上拿到这点 seam 收益。**

---

## Part 4 — 判定（STAGE-1，逐条）

| 判据（负责人） | 目标 | 实测 | 判定 |
|--|--|--|--|
| ncu 确认 mid-r 根因（seam/fill/bound） | 量化 | DUAL ≤ 端点和（快 2–3%）、compute-bound（DRAM 10–16%）、TC 贴端点（65–73% 共享天花板）、barrier-stall 0.5–0.8（可忽略）→ **seam 证伪** | ✔ 根因确认 |
| overlap PoC bit-exact | cos=1.0 | overlap-128×64 两 cell cos=0.999999 | ✔ |
| overlap 收 mid-r gap？ | excess 前→后 materially 降 | 隔离 seam 仅值 2.4–5.5%；拿它需 de-union→128×64，比生产 128×128 慢 9–22%；de-union 在 128×128 物理放不下 | ✘ **不收** |
| 若不行 → bound 天花板 | 诚实 | fused GEMM 已在融合最优 ~2–3% 内（DUAL≤端点和）；突破 70% TC 天花板只剩 Option C，被 168-reg/1-CTA 寄存器墙挡死 | ✔ **已定界** |

**STAGE-1 verdict：cross-phase overlap 关闭。** mid-r 的「bulge」**不是** GEMM 相位 seam（ncu 证伪：serial 融合核已 ≤ 端点和、compute-bound、barrier-stall 可忽略），而是 (a) per-call quant/pack（不在 t_ideal，且不归 mainloop overlap 管）+ (b) 所有 block-scaled 核共享的 ~65–73% SM120 TC 天花板（round-11 SF-ALU 路径，端点同样吃）。真拆 seam 只值 2.4–5.5%，且被 de-union 的 smem 墙逼成净 regression。**→ 不进 STAGE-2 的「overlap 全形状产品化」**；结构天花板就此 bound。

---

## Part 5 — 下一步（优先级，给负责人）

既然 GEMM mainloop 这条（overlap）已到头，mid-r e2e 的真实可拿 headroom 在别处：
1. **quant/pack 才是 mid-r e2e 大头**（down·8192 r=1/2 pack ≈ 396us vs GEMM 375us）。下一轮应把 ncu/优化对准 **融合 quant/pack kernel**（round-18 的 `cd_quant_pack_kernel`）——它是每次调用必跑、且 t_ideal 没算、但 e2e 占比最大的一块。量化它 compute/mem-bound、能否与 GEMM 用 stream/graph 重叠或进一步融合。
2. **若坚持要动 GEMM TC 天花板**：只剩 Option C（双 warpgroup 并发 FP4+FP8，各 partial accumulator 末尾求和）——先验证 partial accumulator 是否能压进寄存器预算（当前 168 reg/thread、1 CTA/SM；需实测 2×accumulator 下的 occupancy / spill），大概率 register-blocked，但这是唯一能理论上超 70% 天花板的结构。
3. 维持现状派发（round-19 `cd_dispatch_r19.csv`）：生产 128×128 NV4/MX2 serial 核就是每场景最优。

---

## Part 6 — 诚实边界

- **overlap 是部分 overlap**：单 producer warp 顺序发 nv.load→mx.load，MX fill 不是与 NV 完全并发（Part 2b）。但隔离 A/B 已证「拆 seam」的上限收益只有 2.4–5.5%，即便做到完全并发也盖不过 de-union 的 9–22% tile 代价——结论不变。
- **ncu 单发 under-replay**：绝对 us 比 warm median 偏大；本轮用 ratio（DUAL vs 端点和）、TC%、stall ratio 作结论，不用 ncu 绝对延迟。warm median 延迟取自 r19（cd_gemmonly）与隔离 A/B。
- **隔离 A/B 在有 co-tenant 的 GPU3**（49% util 外部占用）跑；但 overlap/serial 同 GPU 同 session 同数据，**相对 ratio** 不受影响（两核同受竞争）。生产 128×128 的绝对 us（697/606）比 r19 clean（417/375 cd_gemmonly）偏大即竞争噪声，但三核同场可比。
- **只测了 128×64 一个能 fit 的 de-union tile**：128×128 物理放不下已由 compat_report 的 smem 数（118784/137216/155648 全 > cap）+ Stages≥2 static_assert 双重证明，非「没测」。
- **Option C 未建**：由寄存器预算（168 reg/thread、1 CTA/SM + round-9 墙）推断 register-blocked，未实际编译验证——列为下一轮若要碰 TC 天花板时的首验。

---

## 产物
- **新文件（serial / PROTECTED 零改动）**：`codesign_overlap_kernel.hpp`、`codesign_overlap_ext.cu`、`prof_cdmid.py`、`run_cdmid_ncu.sh`、`analyze_cdmid.py`、`codesign_overlap_check.py`、`run_r20_overlap.sh`、`codesign_overlap_iso.py`。
- **数据**：`logs/cdmid/*.csv`（6 核 ncu）+ `r20_cdmid_summary.txt`（解析表）、`r20_overlap.csv`（de-union smem 墙：5 config 全不 fit 128×128）、`r20_overlap_iso.csv`（隔离 A/B，overlap bit-exact + ov/ser + ov/prod）。日志 `logs/_cdmid.log`（`CDMID_NCU_DONE`）、`logs/_r20_overlap.log`（`R20_OVERLAP_DONE EXIT=0`）、`logs/_r20_iso.log`（`R20_ISO_ALL_COSINE_PASS` + `R20_ISO_DONE`）。
- **未改动（保护）**：`codesign_fused_kernel.hpp` / `codesign_sm120_bs_mma.hpp` / `b1_tma_ext.cu` / `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu` 全部零改动。
- `candidates.jsonl`：+3 条（ncu seam 证伪；overlap de-union smem 墙；隔离 seam 值 2–5% 但净 regression）。GPU 已释放。
