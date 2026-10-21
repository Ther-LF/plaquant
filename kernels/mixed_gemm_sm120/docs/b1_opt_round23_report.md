# B1 优化第二十三轮 — 报告（GEMM-only 绝对 %-of-peak 定界 + qkv 干净 matched-clock 判定 + Option-C 真建造：寄存器墙证伪、真瓶颈=smem、并发双精度 TC go/no-go）

> 全程 remote-only（10.208.130.90；`luofan-plaquant` 建/跑，`luofan-ncu` ncu；RTX PRO 6000 Blackwell **SM120** Server Edition，188 SM，GDDR7 96GB）。
> **GEMM-only，全程无 quant/pack、无 e2e**——计时区只有 GEMM kernel，输入全部预打包（NVFP4/MXFP8 codes + 预交错 SF）。
> 新增文件（受保护/shipping 文件零改动）：`r23_partA.py`（干净 matched-clock、interleaved、%-peak harness）、`codesign_optc_kernel.hpp`（Option-C 双累加器 fork）、`codesign_optc_ext.cu`、`r23_optc.py`（寄存器/cosine/latency 探针 + smem introspection）、`prof_optc.py`/`run_r23_ncu.sh`（ncu TC-duty + 寄存器 + 占用）。
> **PROTECTED 零改动**：`b1_tma_ext.cu`(shipping) / `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py` / `pure_presf_ext.cu` / `codesign_fused_kernel.hpp` / `codesign_sm120_bs_mma.hpp` / `codesign_fused_ext.cu` 全部未动。
> 正确性硬门：所有计时 kernel cosine ≥ 0.999（Part A 全 PASS；Option-C 双累加器 cos=0.999998–0.999999，bit-exact）。

---

## 一句话结论

1. **绝对 %-of-peak（真天花板）**：我们**从未如此定界过**——即使最好的 cell（down·M8192，NVFP4 主导）也只到 **~63% of dense NVFP4 硬件峰值**；生产点（M≥4096, r=1/8）约 **50–60%**；CUTLASS 纯 block-scaled presf 端点更低（NVFP4 ~36–50%、MXFP8 ~47–64%）。**离硬件峰值还差 37–64%**——这才是真天花板，远低于之前只对比 CUTLASS/t_ideal 得出的「到极致」。gap 的根因有二：(a) block-scaled MMA 的 tensor-pipe 占空比上限 ~71–78%（ncu 实测，见下），(b) block-scaled MMA 每个 tensor-active cycle 的 retire 率低于 naive dense FP4 率。
2. **qkv matched-clock 判定（settled）**：干净免 GPU + 锁频 + interleaved 计时下，**shipping GEMM 在 M≤4096 确实赢 co-design**（ship/cd = 0.87（M512）… 0.92（M2048）… 0.97（M4096）），**但在 M=8192 co-design 反超**（ship/cd 1.05–1.07）。即 qkv 的「shipping 在 M≤4096 胜出」被干净复测**证实**，交叉点在 M=8192。down/gate 则 co-design 在 M≥2048 通吃。
3. **Option-C（并发双 warpgroup，各自 partial accumulator）—— 真建造的结论**：
   - **寄存器墙被证伪**：双累加器（128×128，2×64=128 floats/thread）**实测 fit**——ptxas 168 reg/thread（与 baseline 相同，受 384-thread/1-CTA launch-bound 封顶），仅 trivial spill（24B store/48B load vs baseline 4B/16B），**1 CTA/SM 维持、cosine bit-exact、latency 仅 +0.2…+3%**。round-20「2× accumulator 被寄存器墙挡死」的**推断（从未建造）不成立**。
   - **真瓶颈 = SMEM**：真并发需两格式 tensor smem **同时常驻（de-union）= 141312 B（NV 73728 + MX 67584）≫ SM120 的 101376 B cap**（本轮编译期实测），在生产 128×128 tile 上**物理放不下**。这与 round-20 的 de-union smem 墙一致，且给出 Option-C 专属的确切数字。
   - **并发双精度 TC go/no-go = NO-GO**：ncu 实测**纯单格式 block-scaled kernel 已经用满 2 个 warpgroup（256 线程）却仍只有 71–78% TC-elapsed**（fp4_down 78.0、fp8_down 75.5、fp4_gate 71.2、fp8_gate 75.6）——**~73–78% 是 block-scaled 算子管线（operand/SF staging）的硬天花板，不是 issue-width 限制**。并发双精度用的是同样 2 个 warpgroup、共享同一条 operand/SF 管线 → 只会**争用**不会**填泡**，无法超过单格式已达到的天花板。round-20 的 isolated de-union overlap（codesign_overlap，128×64）已实测拆 seam 仅值 2.4–5.5%、净 regression 9–22%。**结论：并发双精度 issue 无法突破 ~78% 的 NVFP4 管线天花板；且在生产 128×128 tile 上 smem-blocked。**

---

## Part A — GEMM-only，干净 matched-clock，绝对 %-of-peak

### 方法
- **干净**：抓到真正空闲的 GPU1（0% util、4 MiB），`nvidia-smi -lgc 2430,2430` 锁频（300W Server Edition 实跑功率封顶 ~2347–2400 MHz，见每 cell `clk_mhz`）。
- **matched-clock + interleaved**：每个 (shape,M,r) 把 co-design / shipping / pure-NVFP4 / pure-MXFP8 四核做**逐样本 round-robin 交错**计时（median-of-60 + min），四核经历**同一竞争/时钟分布** → ratio 与 min-absolute 都干净（修正了 r19 在竞争盒上端点与 kernel 分窗计时导致的偏差）。
- **计时对象（GEMM-only）**：co-design=`fused_gemm_prepacked`（SFA 预交错）；shipping=`b1_fused_tma_dual`（natural SF 单发，无 scatter）；pure=`pure_presf` NVFP4/MXFP8（weight SF 预交错）。r=0 用 co-design single-NVFP4 path。

### 硬件峰值（derive + 来源）
- **来源 1（datasheet，权威）**：HPE QuickSpecs「NVIDIA RTX PRO 6000 Blackwell **Server Edition** 96GB PCIe」= **1.6 PFLOP FP4 TC、811 TFLOPS FP8 TC、406 TFLOPS FP16**（全 dense，每降一档精度正好 2×）。
- **来源 2（per-SM，交叉校验）**：Blackwell 5th-gen tensor 每 SM 每 clock dense = **FP16 1024 / FP8(MXFP8) 2048 / FP4(NVFP4) 4096 FLOP**（对 RTX 5090 FP16 419 TF@170SM,2.41GHz 反推 = 1022≈1024，吻合）。datasheet 1600/811/406 ↔ ~2.07 GHz nominal。
- **本轮用的峰值（按每 cell 实测 clk）**：`peak_nvfp4 = 188·clk·4096`，`peak_mxfp8 = 188·clk·2048`。实测 clk ~2347–2400 MHz → **peak_NVFP4 ≈ 1808–1848 TFLOPS，peak_MXFP8 ≈ 904–924 TFLOPS**。混合比 r 的等效峰值 `eff_peak(r)=peak_nvfp4/(1+r)`（r 比例的 K 走 MXFP8 半速；r=0→NVFP4、r=1→MXFP8）。

### 结果表（achieved TFLOPS · %-of-eff-peak；fp4%/fp8% 对各自纯格式峰值）

| shape | M | r | co-design TF · %pk | shipping TF · %pk | pure-NVFP4 %pk | pure-MXFP8 %pk | winner | ship/cd | 离峰 gap |
|--|--|--|--|--|--|--|--|--|--|
| qkv | 512 | 1/8 | 98 · 5.9 | 113 · 6.9 | 5.3 | 8.9 | **ship** | 0.865 | ~93% |
| qkv | 2048 | 1/8 | 287 · 17.6 | 311 · 19.0 | 16.5 | 25.3 | **ship** | 0.923 | ~81% |
| qkv | 4096 | 1/8 | 462 · 28.5 | 479 · 29.5 | 23.8 | 35.0 | **ship** | 0.965 | ~70% |
| qkv | 8192 | 1/8 | 609 · 37.3 | 580 · 35.5 | 28.2 | 40.1 | **codesign** | 1.051 | ~63% |
| qkv | 8192 | 1/2 | 514 · 42.2 | 480 · 39.4 | 28.6 | 40.4 | **codesign** | 1.071 | ~58% |
| gate | 2048 | 1/8 | 608 · 37.4 | 578 · 35.5 | 37.9 | 49.2 | codesign | 1.053 | ~63% |
| gate | 4096 | 1/2 | 673 · 55.9 | 605 · 50.2 | 49.9 | 61.5 | codesign | 1.113 | ~44% |
| gate | 8192 | 1/8 | 864 · 53.5 | 761 · 47.1 | 49.8 | 62.2 | codesign | 1.137 | ~47% |
| gate | 8192 | 1/2 | 695 · 57.0 | 614 · 50.3 | 49.4 | 61.8 | codesign | 1.132 | ~43% |
| down | 2048 | 1/8 | 582 · 35.6 | 572 · 34.9 | 27.2 | 36.8 | codesign | 1.018 | ~64% |
| down | 4096 | 1/8 | 862 · 52.8 | 816 · 50.0 | 33.8 | 46.8 | codesign | 1.056 | ~47% |
| down | 8192 | 1/8 | **982 · 60.4** | 865 · 53.1 | 36.4 | 48.7 | codesign | 1.136 | ~40% |
| down | 8192 | 1/2 | 764 · 63.2 | 659 · 54.5 | 36.7 | 49.1 | codesign | 1.158 | ~37% |
| down | 8192 | **r=0** | **1173 · 63.5** (single-NVFP4) | — | 36.4 | — | — | — | **~37%** |

（全 48 cell 见 `r23_partA.csv`；小 M=512 全 shape 都 5–21% peak = 小 GEMM 延迟/占用率受限。）

### Part A 读数
1. **真天花板 gap 巨大**：最好的 cell（down·M8192）才 ~63% of dense NVFP4 峰值 → **离硬件天花板差 37%**；生产点 50–60%；小 M 掉到 5–21%。之前轮次说「到极致」只是对 CUTLASS/t_ideal 而言，**对硬件峰值从来不是**。
2. **纯 CUTLASS block-scaled presf 端点只有 36–50%（NVFP4）/47–64%（MXFP8）of 各自峰值**——连 CUTLASS 自己的 block-scaled kernel 都远离 dense 峰值。我们的 co-design single-NVFP4（63.5%）反而**比 pure-presf 端点快 ~1.7×**（与 round-13「co-design 比端点快 16–24%」同向，本会话 persistent+cooperative 配置差距更大）。
3. **%-peak 随 M 单调升**（down 512→8192：14%→63%）——大 M 才摊得开 128×128 cooperative tile。
4. **TC-duty vs FLOP-%-peak 的差**（见 Part C ncu）：pure-NVFP4 presf TC-elapsed 78% 但 FLOP 只有 36% of 峰值 → **block-scaled MMA 每 tensor-active cycle 的 retire 率 ~46% of naive dense**（duty 78% × ~46% ≈ 36%）。这是 block-scaled 算子的内在特性，是 FLOP-%-peak 低的第二个根因。

### qkv matched-clock 判定（负责人点名）
> 「干净 matched clock 下，shipping GEMM 是否真的在 M≤4096 赢 co-design，赢多少？」

**是，证实**。ship/cd（min-latency，interleaved 同场）：

| qkv M | 512 | 2048 | 4096 | 8192 |
|--|--|--|--|--|
| ship/cd (r=1/8) | **0.865** | **0.923** | **0.965** | 1.051 |
| ship/cd (r=1/2) | 0.876 | 0.924 | 0.992 | 1.071 |

shipping 在 **M≤4096 赢 3.5–13.5%**（M 越小赢越多），**交叉点在 M=8192**（co-design 反超 5–7%）。根因：co-design 是 persistent 单 wave（waves=1.00，188 CTA 一波），small/mid-M qkv（N=2048 小）CTA 数不足以摊开 persistent 的 ramp，而 shipping 多-wave（waves=5.45）在小 GEMM 上 ramp 更快；大 M co-design 的 whole-K + 高 TC 才体现。down/gate（N 大或 K 大）则 co-design 在 M≥2048 通吃。

---

## Part B — Option-C 真建造（并发双精度，各自 partial accumulator）

### 建了什么
`codesign_optc_kernel.hpp` = fork `codesign_fused_kernel.hpp`，加 `TwoAccum` 模板路径：NV 相位累加进 `accumulators`、MX 相位累加进**独立的** `accMX2`（各自 ClearAccum=true），epilogue 前求和。**这隔离了「2× partial accumulator」的寄存器成本**（仍是 256-thread cooperative 串行 issue——寄存器问题与 issue 问题正交，本 fork 回答寄存器问题；并发 issue 问题见下的 smem/天花板论证）。bit-exact：accNV+accMX == shared-accumulate。

### B1. 寄存器成本（128×128，2× accumulator）—— 实测 fit，证伪 round-20

| kernel | regs/thread | occ-limit | spill | 1 CTA/SM | latency vs base | cosine |
|--|--|--|--|--|--|--|
| baseline（shared 1 accum，64 f/thr） | **168** | registers | 4B st / 16B ld | ✅ | 1.00 | — |
| **Option-C（2 accum，128 f/thr）** | **168** | registers | **24B st / 48B ld** | ✅ | **1.009 / 0.993 / 0.990**（down/gate/qkv）| **0.999998–0.999999** |

- **双累加器 fit**：ptxas 两者都 168 reg/thread（受 `MaxThreadsPerBlock=384 × 1 CTA/SM` launch-bound 封顶 ≈ 65536/384≈170），Option-C 仅多 **trivial spill（24B/48B）**，**1 CTA/SM 维持、latency 几乎不变、bit-exact**。
- **→ round-20「2× accumulator 在 128×128 被 64K-reg 墙挡死、Option-C register-blocked」的推断（从未建造）不成立**。baseline 的 168-reg 预算对第二个 64-float accumulator 有足够 slack（block-scaled MMA 的 operand 从 smem 取、不长期占寄存器）。
- 更小 per-WG tile（64×128，各半 accumulator=64 f/thr）寄存器成本更低、更不是问题——寄存器**从来不是** Option-C 的真瓶颈。

### B2. 真瓶颈 = SMEM（并发需两格式 smem 同时常驻）
编译期实测（`compat_report`，128×128 NV4/MX2）：

| | tensor smem |
|--|--|
| NV collective | 73728 B |
| MX collective | 67584 B |
| **union（串行，当前 co-design）** | **73728 B** ✅ (< 101376) |
| **de-union（并发 Option-C 必须）** | **141312 B** ❌ **≫ 101376 B SM120 cap** |

- 真并发（WG0 发 NV、WG1 发 MX，同时）要求两格式 tensor smem **同时常驻** → 141312 B，**超 cap 39936 B**，在生产 128×128 tile 上**物理放不下**。
- 这正是 round-20 的 de-union smem 墙（118784 B@NV2/MX2 也超），本轮给出 Option-C 专属确切数（141312 B@生产 NV4/MX2）。唯一 fit 的并发变体是 **64×128 de-union**（round-20 实测 94208 B < cap），但 round-20 已证它比生产 128×128 **慢 9–22%**。

### B3. 并发双精度 TC go/no-go —— ncu 实证 NO-GO
ncu（luofan-ncu，GPU4，-c1 -s10，`device_kernel`/`b1_tma` regex）TC-duty + 寄存器 + 占用（代表 cell）：

| label (M8192) | TCela | TCact | SMact | DRAM | regs | waves | dur_us |
|--|--|--|--|--|--|--|--|
| **fp4_down** (pure NVFP4, 256-thr/2WG) | **78.0** | 86.6 | 90.0 | 11.4 | 168 | 5.45 | 260.8 |
| **fp8_down** (pure MXFP8, 2WG) | **75.5** | 83.6 | 90.2 | 10.6 | 168 | 5.45 | 526.9 |
| base_down (serial dual, shared accum) | 72.6 | 81.0 | 89.6 | 10.5 | 168 | 1.00 | 313.4 |
| optc_down (serial dual, **2 accum**) | 72.0 | 80.5 | 89.5 | 10.4 | 168 | 1.00 | 315.6 |
| ship_down | 60.1 | 66.6 | 90.2 | 8.7 | 168 | 5.45 | 374.2 |
| **fp4_gate** (pure NVFP4, 2WG) | **71.2** | 72.7 | 98.0 | 24.0 | 168 | 21.8 | 280.8 |
| **fp8_gate** (pure MXFP8, 2WG) | **75.6** | 77.2 | 97.9 | 14.8 | 168 | 21.8 | 523.6 |
| base_gate (serial dual) | 64.4 | 67.1 | 96.1 | 16.0 | 168 | 1.00 | 455.7 |
| optc_gate (2 accum) | 62.6 | 64.8 | 96.7 | 15.4 | 168 | 1.00 | 470.7 |
| base_qkv / optc_qkv / ship_qkv | 54.4 / 53.0 / 44.7 | 62.4/60.8/50.1 | 87.2 | 7.9 | 168 | 1/1/5.45 | 105/107/126 |

**go/no-go 论证（全部实测）**：
1. **~73–78% 是 block-scaled 管线的硬天花板，不是 issue-width 限制**：**纯单格式 kernel 已经用满 2 个 warpgroup（256 线程）**，fp4_down 仍只有 78.0%、fp8_down 75.5%、fp4_gate 71.2%。再多/再拆 warpgroup 不会超过单格式已达到的占空比。
2. **串行 dual 已经贴着这条天花板**：base_down 72.6% ≈ down 的 K-加权单格式天花板（0.875·78+0.125·75.5≈77.7%）内 ~5pp（round-20 实测 barrier-stall 可忽略，这 5pp 是 fill seam 不是 barrier）。
3. **并发用同样 2 WG、共享同一条 operand/SF 管线** → NV 的泡被 MX 填？不能——两格式**争用同一条 SF/operand staging 管线**（round-11 已定位 block-scaled 的 duty 上限来自这条路径）。即便理想拆掉那 ~5pp fill seam，也只到 ~77.7%，**仍 ≤ 纯 NVFP4 的 78% 管线天花板，不是「突破」**。
4. **round-20 的 isolated de-union overlap（codesign_overlap，128×64，bit-exact）已实测拆 seam 仅值 2.4–5.5%、净 regression 9–22%**。
5. **生产 128×128 上并发 smem-blocked（141312 B > cap）**。

**→ NO-GO**：并发双精度 issue **无法突破 ~78% 的 NVFP4 block-scaled 管线天花板**（单格式 2-WG 已定义该天花板），且在生产 tile 上 smem-blocked；寄存器不是瓶颈（已证伪），但移除它也不解决问题，因为天花板是共享的 operand/SF 管线 + 128×128 de-union smem 限制。

---

## Part C — 对负责人几个问题的直接回答

1. **我们绝对的 %-of-peak 是多少？离真天花板多远？** 最好 ~63%（down·M8192）、生产点 50–60%、小 M 5–21% of **dense NVFP4 硬件峰值（~1.8 PFLOP@实测 clk，datasheet 1.6 PF@nominal）**；离硬件天花板 **37–95%**。纯 CUTLASS block-scaled 端点更低（36–50%）。两条根因：block-scaled MMA 的 ~71–78% tensor-duty 上限 + ~46% per-active-cycle retire 率。
2. **qkv：干净 matched-clock 下 shipping 真的在 M≤4096 赢吗？** 真的，赢 3.5–13.5%（min-latency、interleaved、锁频、免 GPU），交叉点 M=8192（co-design 反超 5–7%）。
3. **Option-C 能不能超过 ~70% 天花板？** 不能（NO-GO，见 Part B3）。**但本轮纠正了 round-20 的一个错误推断**：Option-C 不是 register-blocked（双累加器实测 fit，168 reg、bit-exact、baseline latency）；真瓶颈是 **de-union smem（141312 B > 101376 cap）** + block-scaled 管线的 ~78% 硬天花板。

---

## 诚实边界
- **%-peak 用 min-of-60（interleaved，锁频免 GPU）** 作 absolute，以逼近无竞争延迟；median 亦在 CSV。clk 按每 cell 实测（~2347–2400 MHz，300W 功率封顶，非锁定的 2430）。
- **pure-presf 端点是一个基础 CollectiveBuilder 配置**，非最优 NVFP4 kernel（我们的 co-design single 比它快 ~1.7×）；它作为**一个** CUTLASS block-scaled 参照，不是「CUTLASS 最快」。TC-duty（ncu）与 FLOP-%-peak（latency）量的是不同东西，差值 = block-scaled MMA 的 sub-dense retire 率，本报告两者都给。
- **Option-C 双累加器 fork 是串行 issue**（256-thread cooperative），它回答的是**寄存器可行性**，不是并发 issue 本身；并发 issue 的 TC 结论由「单格式 2-WG 已达 71–78% 天花板 + de-union smem 墙 + round-20 overlap 实测 2.4–5.5%」三条实证支撑，**未**从头建造完整并发核（在生产 128×128 上 smem-blocked，无 bit-exact 落地意义；唯一 fit 的 64×128 变体 round-20 已建过 = codesign_overlap）。标注清楚：并发完整核 = not-built（smem-blocked），其余为实测。
- ncu 默认锁 base clock、单发 replay；TC%/寄存器/占用是 ratio/静态量，对 replay 稳健；绝对 dur 偏大，不用于 latency 结论（latency 用 Part A warm median/min）。

## 产物
- **数据**：`r23_partA.csv`（48 cell GEMM-only %-peak，全 cos PASS）；`logs/r23ncu/*.csv`（13 核 ncu：TC-duty + 寄存器 + 占用）；`logs/_r23_optc*.log`（Option-C 寄存器/cosine/latency + smem introspection）。
- **新文件（PROTECTED 零改动）**：`r23_partA.py`、`codesign_optc_kernel.hpp`、`codesign_optc_ext.cu`、`r23_optc.py`、`prof_optc.py`、`run_r23_ncu.sh`。
- `candidates.jsonl`：+3 条（绝对 %-peak 定界；qkv matched-clock 判定；Option-C 寄存器墙证伪 + smem 真瓶颈 + 并发 TC NO-GO）。
- **GPU 已释放**（GPU1/GPU3 锁频已 `-rgc` 解除）。
