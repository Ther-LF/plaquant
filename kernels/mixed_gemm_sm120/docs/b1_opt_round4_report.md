# B1 优化第四轮 — 报告 (双相 TMA 尝试 / 占用率 / 大 N 栅格化 / 加密曲线)

> 全程 remote-only + detached+落盘。正确性 gate:**所有测量点 cosine=1.000000**
> (TMA phase-2 全 shape/stage/tile/swz;双相 TMA 的 fp4 路径编译失败,见 T1;
> scoreboard 全 shape/M;加密比例曲线 54 点全部)。rel≈2.1e-4(纯 fp16 round)。
> GPU:RTX PRO 6000 Blackwell (SM120),GPU1 空闲 (685MiB/0%),co-tenant 在 0/2/3/4
> 保留显存但 0% util。

## 结论先行(headline)

- **T2(占用率)— 诚实负结果 + 修正 round-3 判断:** round-3 说"smem-bound,省 ~2KB 可上
  2 CTA/SM"。实测 **smemPerSM=102400、regsPerSM=65536**:phase-2 st3 kernel smem=50736
  (2×=101472 < 102400,**容量上已能放下 2 块**),真正卡在 **regs=121(>113)+ 每块
  ~1KB driver 保留**。强行 `__launch_bounds__(,2)` 把 regs 压到 96 但**溢出到 local
  (localB=72)且占用仍=1**(因 smem+保留 2×超限),latency **+63%**。换 **128×64 tile**
  确实拿到**真·2 CTA/SM(regs=78、无溢出、maxBlocks=2)**,但 down_r1 **+75% / gate +68%
  / qkv +41%**——算术强度/MMA 效率损失远大于占用收益。**结论:占用率不是本 kernel 的
  杠杆;128×128 st3 @ 1 CTA/SM 已是最优工作点。**
- **T3(大 N 栅格化)— 诚实负结果 + ncu 证据:** 给 TMA kernel 加 Triton 式 group-M CTA
  重排(SWZ∈{2,4,8},1D grid 内解码),gate_r1 latency **swz0 +32.2% vs swz8 +32.2%
  (st3,噪声内)**。ncu:**L2 命中率 swz0=93.69% / swz8=93.66%,DRAM 均 21.73MB,时长
  228.7 vs 226.8us**——**gate-class 根本不是 L2 复用受限**(命中率已 93.7% 无余量),栅格化
  无效。gate 的 56% band 是结构性(wave 量化 / TMA 在大 N 本就高于 fp8 ceiling),与 L2 正交。
- **T1(双相 TMA)— 硬卡,精确定位 blocker:** 双相 kernel 已成型(两相共用同一 uint8
  SW swizzle smem 环 + 生产者按相位选 low/high TMA 描述符 + 消费者 consume4/consume8 切换);
  **纯 fp8 相(r=1)路径可编译**。**fp4 相(SU4 ldmatrix)编译失败**:`SM100_SU4_DU8x16_x4_LDSM_N`
  "src/dst failed to vectorize into registers"。**根因已查明**(见 T1 节):dst 已正确是
  `subbyte_iterator<uint4_t>`、swizzle 已是 SW128(KMUL=4),但 SU4 需要 collective 的
  **整-BK `tile_to_shape` smem 布局**,而本 kernel 的**单-chunk 环形切片**(每 stage 128B)
  破坏了 SU4 的寄存器向量化模式。给出下一轮的两条精确修法。
- **T4:加密比例曲线落地(负责人硬要求):** `ratio_curve.csv` 覆盖为加密版 **54 点**
  (qkv/gate 各 17 点步长 128;down 20 点步长 512 + 低端 128/256/384),每点 cosine gate +
  同场 presf bounds + t_ideal 插值 + excess_pct,**按产品派发**(nonTMA auto 为主,纯 fp8
  大 K 点并入 TMA 候选取 min)。scoreboard 刷新确认 down_3B M=256 band **-9%**(塌陷已修)。

---

## T1 — 双相 TMA(fp4 Phase-1 也走 TMA)

### 关键简化(本轮发现,降低了 round-3 的预估风险)
查 CUTLASS SM120 blockscaled builder(`sm120_blockscaled_mma_builder.inl` / `sm120_common.inl`):
对 **F8f6f4 路径**(我们的 `SM120_16x8x64_TN_VS` fp4 / `SM120_16x8x32_TN_VS` fp8 都属此),
`SmemAllocTypeA = uint8_t`(**不是** unpacksmem;unpacksmem 是另一条 narrow mxf4nvf4 collective),
fp4 的 ldmatrix = `sm120_rr_smem_copy_selector_A<...,UseF8f6f4=true>` → **`SM100_SU4_DU8x16_x4_LDSM_N`**
(读 uint8-packed smem、解包成 fp4 寄存器)。**两相的 TMA 字节数/row/chunk 完全相同(64B@KMUL2
或 128B@KMUL4),smem swizzle 布局完全相同** → union smem 天然成立,TMA 描述符仅 base/K-extent
不同,无需复杂的 phase-boundary drain,生产者按相位选描述符即可。

### 已落地(可编译、r=1 纯 fp8 相验证通过)
`b1_tma_dual_kernel`:4 个 TMA 描述符(a/b × low/high),共用 uint8 SW smem 环;生产者
`g<n1` 走 low、否则 high;消费者 consume4(AtomNV+fp4 SF)/consume8(AtomMX,与已验证
phase-2 一致)共享 FP32 累加器;SF plain-store(fp4 NBLK4=4·KMUL、fp8 NBLK8=KMUL)。

### blocker(精确)
fp4 消费 `copy(SU4_copy, partition_S(sAs_uint8), retile_D(partition_fragment_A))`:
- **dst 正确**:`subbyte_iterator<uint4_t>`,fragment 维度/类型已对(units 可和解,
  partition_fragment_A(uint8 smem) 产出 fp4 fragment)。
- **SW128 已正确**:KMUL=4→128B chunk→`Swizzle<3,4,3>`(SW128,SU4 需要的宽度)。
- **仍失败**:`copy_traits.hpp` static_assert "src/dst failed to vectorize into registers.
  Layout is incompatible with this CopyOp."。SrcLayout = 单-chunk 的 swizzle 切片
  `(((4,4),2),4,2)`,与 SU4 的 TV `((4,8,2,4),(8,2,2),(1,1))` 不相容——SU4 要求的是
  collective 那样 **对整个 BK 做 `tile_to_shape(SW128_atom,(BM,BK_full,STAGES))`** 后、
  按 pipeline **整段** ldmatrix 的布局;我们每 stage 只放 1 个 128B chunk,K 方向切得太碎,
  SU4 无法向量化成 128-bit 载入。

### 下一轮两条精确修法(择一)
1. **低风险、复用已验证路径:** TMA 目标用 **SW_NONE(非 swizzle)** 的 fp4 smem 布局(TMA
   支持 SW_NONE),消费端照搬**已 cosine=1.0 的非-TMA `consume4`**(`uint4_t` view +
   `SM75_U32x4_LDSM_N` + ROW_B pad),绕开 SU4/swizzle 一致性问题。代价:非 swizzle 可能
   有 bank conflict,但非-TMA 版证明可行且正确。
2. **对齐 collective:** fp4 相 smem 改为 `tile_to_shape(SW128_atom,(BM,BK_full,STAGES))`
   整-BK 布局(不是单 chunk),每 stage 一次性 ldmatrix 整个 BK,再喂 KMUL 段 MMA——满足
   SU4 向量化。代价:每 stage smem 更大,需配合更小 tile/更浅 stage 以不超 99KB。

**判断:** T1 本轮未落地,但**机制全部打通、blocker 精确定位、修法明确**。rules 允许
"T1 stalls hard → ship T2+T3";本轮按此交付,并把 T1 收敛到一个确定的下轮起点。

---

## T2 — 占用率(2 CTA/SM via smem/reg)

### Ground truth(cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor)
`[dev] smemPerSM=102400 regsPerSM=65536`

| 变体 (phase-2 kernel) | smem/block | regs | localB | maxBlocks/SM |
|---|---|---|---|---|
| 128×128 st3 (prod) | 50736 | 121 | 0 | **1** |
| 128×128 st3 **minb=2** | 50736 | 96 | **72(溢出)** | **1** |
| 128×64 st3 | 38064 | 78 | 0 | **2** ✅ |
| 128×64 st4 | 50752 | 78 | 0 | 1 |

### 诊断(修正 round-3)
2×50736=101472 < 102400 → **smem 容量不是瓶颈**;卡在 **regs 121>113**(2 块需
≤65536/(288×2)=113)**+ 每块 ~1KB driver 保留**(2×(50736+1024)=103520>102400)。
round-3 "省 2KB smem" 的前提**被推翻**。

### latency(excess over 同场 pure-fp8 presf bound,median/50)
| 场景 | 128×128 st3 (1块) | 128×128 minb2 (reg压96+溢出) | 128×64 st3 (**真2块**) | 128×64 st4 |
|---|---|---|---|---|
| down_r1 | **+22.7%** | +63.4% | +75.1% | +35.7% |
| gate_r1 | **+32.3%** | +79.2% | +67.9% | +71.5% |
| qkv_r1  | **+13.8%** | +43.0% | +41.2% | +25.0% |

**结论:** 两条上 2 CTA/SM 的路(reg-cap 溢出 / 小 tile)都**显著更慢**。本 kernel 在
128×128 compute-bound 工作点上,占用率(20.7%)不是性能杠杆——拉高占用反而牺牲算术强度。
**占用率这条线判负,收敛。**

---

## T3 — 大 N 结构性(gate band 56%)

给 TMA phase-2 kernel 加编译期 `SWZ`(group-M Triton 式 1D-grid 重排,`if constexpr`
保证 SWZ=0 产品路径零开销、regs 回到 121)。

### latency(excess over pure-fp8 bound)
| 场景 | st | swz0 | swz2 | swz4 | swz8 |
|---|---|---|---|---|---|
| gate_r1 | 3 | +32.2% | +32.9% | +32.5% | +32.2% |
| gate_r1 | 4 | +38.5% | +38.4% | +38.3% | +33.8% |
| down_r1 | 3 | +24.5% | +24.7% | +24.7% | +24.9% |

### ncu(gate_r1 st3,swz0 vs swz8)
| 指标 | swz0 | swz8 |
|---|---|---|
| lts__t_sector_hit_rate.pct | **93.69** | **93.66** |
| dram__bytes.sum | 21.73 MB | 21.73 MB |
| gpu__time_duration | 228.70 us | 226.82 us |

**结论:** L2 命中率已 93.7%、DRAM 零变化、latency 噪声内——**gate-class 不是 L2 复用
受限**,CTA 栅格化无效。56% band 的结构根因在别处(wave 量化:gate N=8192→1024 CTA/
188 SM≈5.45 waves 的尾波;以及 TMA 在大 N 本就高于 fp8 ceiling,round-3 已记)。
栅格化这条线判负;SWZ 开关保留但默认关闭(零开销)。

---

## T4 — 刷新

### 加密比例曲线(`ratio_curve.csv`,覆盖版,54 点,全 cosine=1.0)
密度(本轮起每轮保持):qkv/gate(K_total=2048)步长 128 → 17 点;down(K_total=8192)
步长 512 → 17 点 + 低端 128/256/384 → 20 点。派发:nonTMA auto 为主,纯 fp8 大 K 点并入
TMA 候选取 min(反映产品形态)。

excess over t_ideal(=t_fp4·(1−r)+t_fp8·r)节选:

| 场景 | r=0 | 生产点 | r=0.5 | r=1 |
|---|---|---|---|---|
| qkv_class (N2048) | −5.4% | r=1/8:**−5.7%** | −3.9% | +0.8% |
| down_class (N2048) | −3.3% | r=1/8:**+5.3%** | +19.2% | **+19.9%(TMA_mx_st3)** |
| gate_class (N8192) | +27.5% | r=1/8:+22.9% | +19.0% | +21.5% |

- qkv 全程 −7~+4%(融合开销≈0)。down 低-r 生产区(r≤1/4)+3.6~7%,高-r 升到 +40%+;
  **r=1 端点 TMA 派发接管**(0.2514ms vs nonTMA 外推 ~0.30),印证 round-3 的 TMA 价值点。
- gate 全程 +19~+27%(大 N 结构性,T3 已证非 L2)。

### Scoreboard(b1_fused auto,M=2048 band;0%=fp4 floor,100%=fp8 ceiling)
| shape | band | shape | band |
|---|---|---|---|
| q_proj_1B | **−9%** | down_proj_1B | 29% |
| o_proj_1B | −14% | q_proj_3B | 37% |
| gate_proj_1B | 56% | down_proj_3B | 50% |
| up_proj_1B | 56% | | |

M=256 全 shape band<0(派发健康);**down_3B M=256 band −9%**(负责人关注的塌陷已修,
round-1 为 +232%)。decode M=1/16 继续 launch-light 碾压 collective。

### 产品派发策略(host dispatch policy)
| regime | 变体 | 依据 |
|---|---|---|
| decode / 小 M (M≤16) / 小 K | **非-TMA b1_fused auto**(轻 kernel) | TMA 描述符开销摊不开;b1 0.38-0.49× presf |
| 中 M (M∈[64,512]) | **非-TMA cfg10/6/7**(round-2 派发) | M=256 全 band<0 |
| 大 M 饱和 + 大 K + 纯 fp8 (r=1 down-class) | **TMA phase-2 (st3)** | round-3:+41%→+20%;本轮曲线 r=1 取 min 命中 |
| 大 M 混合 (0<r<1) | **非-TMA b1_fused auto** | 双相 TMA 未落地(T1);非-TMA 覆盖 |
| 大 N (gate/up) | 非-TMA auto | TMA 大 N 更差(T3);栅格化无效 |

---

## 收敛评估(还剩什么,值不值得)

1. **双相 TMA(T1)— 唯一能动"混合-r 生产区"的线:** 本轮未落地但 blocker 精确、修法明确
   (SW_NONE+SM75+uint4 复用非-TMA consume4,或对齐 collective 整-BK tile_to_shape)。
   **这是下轮唯一值得主攻的点**——它才是 down-class 生产 r=1/8 真正用途的加速路径。
2. **占用率(T2)判负、栅格化(T3)判负:** 两条 round-3 列出的"叠加杠杆"经 ground-truth +
   ncu 证伪,**从后续计划中剔除**,避免浪费迭代。这本身是本轮最大的"收敛"价值。
3. **gate/up 56% band:** 既非 L2(T3 ncu)也非占用(T2),剩 wave 量化 / 大 N TMA ceiling;
   需要 persistent/stream-K 级改造,ROI 低,暂缓。
4. **判断:** 产品形态(非-TMA auto + 中 M cfg10 + r=1 TMA)已在 qkv/down 低-r 生产区
   做到 **−7~+7% excess**,接近理想插值。**B1 的 1-CTA/SM 128×128 compute-bound 工作点
   基本收敛;** 唯一的实质增量是双相 TMA 覆盖混合-r,留作下轮确定起点。

## 产物
- `b1_tma_ext.cu`:新增 T2 `tile`(128×64)/`minb`(launch_bounds 2) + T3 编译期 `SWZ`
  group-M 重排(默认 0,零开销)+ TMA_ATTRS 占用探针;**双相 fp4 代码本轮移除**(编译失败,
  blocker 记于本报告,保持文件可编译);phase-2 路径未回归(cosine=1.0)。
- `t2_probe.py` / `t3_probe.py` / `prof_t3.py`(占用/栅格化/ncu driver);
- `ratio_curve.py`(加密版 + best-dispatch);`ratio_curve.csv`(覆盖,54 点);
- `scoreboard_r4.py`;`benchmark.csv` +144 行(b1_r4)。
- **未改动** `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` /
  `mxfp8_gemm_ext.cu` / `mixed_gemm.py`。
- profiles:`r4_t3_ncu.log`(L2 证据);attrs 全在 `r4_t2*.log`。
