# B1 优化第三轮 — 报告 (TMA 流水化)

> 全程 remote-only + detached+落盘。正确性 gate:**所有测量点 cosine=1.000000**
> (TMA kernel 全 shape/M/stage,scoreboard 全 shape/M,比例曲线端点)。
> rel≈2.1e-4(纯 fp16 round,与 round-2 一致)。

## 结论先行(headline)

按负责人指令 **USE TMA**,本轮把 B1 的 Phase-2(MXFP8)gmem→smem 加载从浅
cp.async 改为 **cp.async.bulk.tensor(TMA)+ mbarrier expect_tx 多级流水 +
swizzled smem(UMMA::Layout_K_SW64_Atom<uint8>)**,consumer 侧 ldmatrix 不变
(SM75_U32x4_LDSM_N,正是 SM120 collective 为 8-bit 配对的 atom)。

- **ncu baseline(down r=1 纯 MXFP8, M=2048 N=2048 K=8192):** 之前以为 long_scoreboard
  主导,实测 **wait 2.608 / math_pipe_throttle 1.682 / long_scoreboard 1.675 共同
  主导**,占用率仅 20.7% warps active、1.36 waves/SM,**91KB smem → 1 CTA/SM
  (smem-capacity-bound)**,125 regs/thread,TC 仅 30% of peak。内存延迟+同步等待在低
  占用下饿死 tensor core。
- **TMA 落地结果(同场):down-class r=1 超额 cp.async +41.2% → TMA +19.9%**
  (vs 纯 fp8 presf bound 0.2089ms:cp.async 0.2950ms → TMA st3 0.2505ms,**-15%**)。
  **达成任务目标 "r=1 44%→<20%"。**
- **诚实负结果:** TMA 只在**大 K、计算饱和、内存受限**的 r=1 场景赢;小 K / 小 M /
  大 N 场景 TMA 的 per-stage 描述符开销摊不开,反而更慢(见下表)。
- **r=1/2 目标(22%→<10%)未达成**:r=1/2 需要**双相** TMA(Phase-1 NVFP4 也走 TMA),
  本轮只做了 Phase-2;phase-2-only kernel 覆盖不到 r=1/2(K_low>0)。这是下一轮的明确
  落点,plumbing 已验证。

---

## T1 — TMA 流水化

### T1.1 ncu baseline(before-evidence)

`profile/b1_round3/down_r1_mxfp8_before.ncu-rep`(b1_ws_kernel cfg10 4-deep):

| 指标 | 值 |
|---|---|
| 时长 | 330.6 us |
| warps_active (% peak) | 20.71 |
| SM/TC throughput (% peak elapsed) | 30.23 |
| smem/block | 91.2 KB → 1 CTA/SM |
| regs/thread | 125 |
| **wait** (stall/issue) | **2.608** |
| **math_pipe_throttle** | **1.682** |
| **long_scoreboard** | **1.675** |
| mio_throttle | 0.350 |
| not_selected / selected | 0.452 / 1.004 |

诊断:**占用瓶颈是 smem(91KB → 单 CTA/SM = 10 warps,仅 ~20% peak)**;在此低并行度下,
内存加载延迟(long_scoreboard)+ MMA/同步等待(wait + math_throttle)无法互相掩盖,
TC 只跑到 30%。TMA 的价值点:bulk 传输把加载从 LSU/MIO issue slot 卸载(减 mio_throttle),
硬件描述符驱动的异步传输更好地掩盖加载延迟(减 long_scoreboard),并降低 load 路径的
寄存器/指令压力。

### T1.2 TMA 转换(descriptor / barrier gotchas — 给下一位工程师)

**先做 plumbing 探针**(`tma_probe_ext.cu` + `tma_probe.py`):单 CTA 用 TMA 把一个
(128,64) uint8 B-tile 载入 swizzled smem,再读回比对——**bit-exact PASS**。这步 2 分钟
就确认了 TMA 在我们 JIT extension 里能编能跑,强烈建议任何 TMA 工作先做。

落地要点 / 坑:
1. **include:** `cute/atom/copy_traits_sm90_tma.hpp`(SM90_TMA_LOAD / make_tma_copy)+
   `cutlass/gemm/collective/builders/sm120_common.inl`(拿 `detail::sm120_rr_smem_selector`
   选 swizzle atom)。
2. **swizzle atom:** SM120 collective 对 8-bit K-major 用 `UMMA::Layout_K_SW128_Atom<uint8>`;
   我们每 chunk 的 K=64(KMUL=2),64%128≠0 → 退到 **SW64**(编出来是 `Swizzle<2,4,3>`)。
   smem tensor 用 `tile_to_shape(SmemAtom, (BM,KCH,STAGES))` 构造;**consumer 的 ldmatrix 直接
   在这个 swizzled layout 上 partition_S/retile_D,cute 自动算 swizzle 地址**——这是整个方案
   能复用现有 consume8 的关键,consumer 代码几乎不动。
3. **host 侧 descriptor:** `make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, SmemLayout(_,_,0),
   make_shape(BM,KCH), _1{})`,把返回的 TMA 对象 **按值 `CUTE_GRID_CONSTANT` 传进 kernel**。
4. **partition rank 坑(踩到):** 不要 partition 整个多-chunk/多-stage(rank-3)张量再去
   索引——会 `static assert "Mismatched Ranks"`。要像探针那样**在循环里对单个 (BM,KCH) tile
   和单个 stage 切片 `sA(_,_,s)` 分别 partition_S/partition_D**。
5. **mbarrier:** 一个 elected producer 线程做 `mbarrier.arrive.expect_tx`(字节数=A+B tile
   bytes)+ 两条 `copy(tma.with(mbar[s]), gtile, stile)`;其余 producer 线程做普通 arrive
   释放各自写的 plain SF 字节。`init count = NT_P`(expect_tx 算 thread0 的那一个 arrive)。
6. **SF 仍走 plain store**(MXFP8 phase SF 极小,KMUL 字节/行/chunk);SF 的 TMA(uint16 +
   专用 swizzled layout)是最脏的部分,本轮**刻意不碰**,留给需要时再做。
7. **static smem 上限:** SM120 单 kernel static `__shared__` 实测上限 **0x18c00 = 101376 bytes**。
   128×128 tile、KMUL=2、swizzled A+B 每 stage 16384B:STAGES=5 ≈ 84KB(可),**STAGES=6
   溢出(101984B)编译 error "uses too much shared data"**。

### T1.3 结果(correctness-gated,median/50)

正确性:TMA kernel 在 down_r1 / qkv_r1 / gate_r1 / down_r1@M256 / down_r1@M16,
stages∈{3,4,5} 全部 **cosine=1.000000**。

超额%(over 纯 fp8 presf bound,同场):

| 场景 (K_low=0 纯 MXFP8) | bound ms | cp.async WS | **TMA (best)** |
|---|---|---|---|
| **down_r1** M2048 N2048 K8192 | 0.2089 | 0.2950 **+41.2%** | **0.2505 (st3) +19.9%** ✅ |
| qkv_r1 M2048 N2048 K2048 | 0.0783 | 0.0788 +0.7% | 0.0832 +6.3% ✗ |
| gate_r1 M2048 N8192 K2048 | 0.1550 | 0.1881 +21.3% | 0.1976 +27.4% ✗ |
| down_r1 M256 N2048 K8192 | 0.0964 | 0.1070 +11.0% | 0.1318 +36.8% ✗ |
| down_r1 M16 N2048 K8192 | 0.0971 | 0.0814 -16.1% | 0.1164 +19.9% ✗ |

**TMA 赢的充要条件:大 K(chunk 多,描述符开销摊得开)+ 计算饱和(M=2048)+ 内存受限。**
小 K(K=2048 只有 32 chunks)、大 N(gate 是 tile/rasterization 问题非加载延迟)、中/小 M
(占用低、每个小 grid 的 TMA setup 占比高)TMA 都不划算——诚实记录,不作一刀切派发。

### T1.1-after ncu(after-evidence,同一 kernel 同一 shape)

`profile/b1_round3/down_r1_mxfp8_tma_after.ncu-rep`(b1_tma_mx_kernel st3):

| 指标 | cp.async (before) | **TMA (after)** | Δ |
|---|---|---|---|
| 时长 us | 330.6 | **295.1** | **-11%** |
| smem/block | 91.2 KB | 51.8 KB | — |
| SM/TC throughput % | 30.23 | **33.38** | +10% rel |
| **long_scoreboard** | 1.675 | **1.203** | **-28%** |
| **mio_throttle** | 0.350 | **0.062** | **-82%** |
| lg_throttle | 2.96e-5 | 0 | — |
| short_scoreboard | 0.128 | 0.072 | -44% |
| wait | 2.608 | 2.305 | -12% |
| math_pipe_throttle | 1.682 | 1.730 | +3% |
| warps_active % | 20.71 | 18.59 | — |

**机制确认(正是负责人预判):TMA 把 Phase-2 加载延迟掩盖掉了——long_scoreboard -28%,
mio_throttle -82%(bulk 传输不再占 LSU/MIO issue slot)。** TC 利用率随之 30→33%。
注意占用仍是 1 CTA/SM:TMA st3 的 51.8KB,2×51.8=103.5KB 仍 >101.4KB 上限,**差一点点**;
再省 ~2KB/CTA 就能上 2 CTA/SM —— 这是叠加在 TMA 之上的下一个明确杠杆。

### T1.4 Phase-1(NVFP4)TMA — 未做

本轮只做 Phase-2。Phase-1 的 TMA 需要 `float_e2m1_unpacksmem_t` + `SM100_SU4_DU8x16_x4_LDSM_N`
(FP4 专用解包 ldmatrix),与 uint8 路径不同,风险更高;且**双相 union smem + phase 边界
流水正确性**才是 r=1/2 和生产 r=1/8 真正需要的。plumbing 已验证,留作下一轮。

---

## T2 — M=256 cfg10 派发(已在 round-2 落地,本轮确认)

candidates `B1-R2-T3-midM-dispatch` 已把 M≤256 的派发写进 host(N≥4096→cfg10,
2048≤N<4096→K≥4096?cfg6:cfg7,N<2048→cfg3)。本轮 scoreboard 复测确认 **M=256 全 shape
band 均 <100%**:

| shape | round-1 band | **round-3 M=256 band** |
|---|---|---|
| q_proj_3B | 162% ❌ | **-26%** ✅ |
| down_proj_3B | 232% ❌ | **-10%** ✅ |
| gate_proj_1B | 76% | **-53%** ✅ |
| down_proj_1B | — | **-7%** ✅ |
| q/k/v/o_proj_1B | — | -154 / -178 / -172 / -162% ✅ |

T2 确认达标,无需新改动。

---

## T3 — 刷新成绩单(scoreboard,quiet GPU 0:co-tenant 83GB 保留显存但 0% util)

正确性 cosine=1.000000 全 shape/M。band%(0%=纯FP4 floor,100%=纯FP8 ceiling;presf 同场 bound):

| shape (M=2048) | lat ms | band% | vs presf | 实现 TFLOPS | % of fp4-bound |
|---|---|---|---|---|---|
| q_proj_1B   N2048 | 0.0498 | -4% | 0.62x | 345 | 101% |
| o_proj_1B   N2048 | 0.0490 | -8% | 0.60x | 350 | 103% |
| down_proj_1B N2048 | 0.1532 | 35% | 0.89x | 448 | 85% |
| q_proj_3B   N3072 | 0.0935 | 34% | 0.79x | 414 | 86% |
| down_proj_3B N3072 | 0.2150 | 55% | 1.00x | 479 | 76% |
| gate_proj_1B N8192 | 0.1241 | 58% | 0.74x | 553 | 75% |
| up_proj_1B   N8192 | 0.1251 | 59% | 0.75x | 549 | 75% |
| k/v_proj_1B N512 | 0.028 | -150% | 0.49x | — (小N) | — |

(TFLOPS=2·M·N·K_total/lat;% of fp4-bound = 同场纯 NVFP4 presf 的实现 TFLOPS 为分母,
即同硬件经验上界。profiled 点的硬件 SOL:down r=1 TC 30-33% of peak。)

- **decode(M=1/16):band -170%~-246%,b1 ≈ presf 的 0.38-0.49x**——轻量 kernel launch
  优势,碾压 collective(按指令不作卖点)。
- **M=2048:** q/o_proj 已到或低于 FP4 floor(-4~-8%);down 系列 35-55%;**gate/up ~58%
  仍是大 N 结构性弱点**(round-2 已定性为 tile/rasterization,非融合/非加载);k/v(N=512)
  深负。全部 **≤ 纯 FP8 ceiling**(band<100%),无 round-1 的"不及格四兄弟"。
- vs presf(rungB):M=2048 全部 ≤1.00x(down_3B 持平),decode/中档 0.38-0.89x,全面更快。
- 三条比例曲线:mixed-r 路径 kernel 本轮未改(`ratio_curve.csv` round-2 仍有效);本轮的
  TMA 改进落在 **r=1 端点**(down-class +43.6%→+19.9%)。

---

## 收敛评估(what's left,值不值得继续)

1. **双相 TMA(最高价值,下一轮主攻):** Phase-1 NVFP4 也走 TMA + union swizzled smem +
   phase 边界流水。这是打 r=1/2(目标 <10%,现 +22%)和**生产 r=1/8**(down-class 真正用途)
   的唯一路径。plumbing(descriptor/mbarrier/swizzle/ldmatrix)本轮已全部验证,FP4 还需
   `unpacksmem_t`+`SM100_SU4` ldmatrix。
2. **2 CTA/SM:** TMA st3 的 51.8KB 差 ~2KB 就能双 CTA/SM(占用从 20%→~40%,直击 wait/
   long_scoreboard 主导的低占用根因)。缩 SF buffer / 降一点 pad 即可试。
3. **gate/up 大 N(band ~58%):** round-2 诊断的 CTA rasterization/swizzle for L2 reuse
   on N=8192,与 TMA 正交,是 M=2048 剩余最大头。
4. 判断:**TMA 这条线证明有效且机制清晰(long_scoreboard/mio_throttle 显著下降),
   值得继续**——但增量应投向双相 TMA(覆盖生产 r)而非 phase-2-only 的更多调参。

## 产物
- `tma_probe_ext.cu` / `tma_probe.py`(TMA plumbing 探针,bit-exact PASS)
- `b1_tma_ext.cu` / `b1_tma.py`(Phase-2 MXFP8 TMA kernel + validator/bench)
- `prof_tma.py`(TMA kernel ncu driver);`scoreboard_r3.py`
- `profile/b1_round3/down_r1_mxfp8_before.ncu-rep` / `...tma_after.ncu-rep`(before/after)
- `benchmark.csv` +144 行(b1_r3 scoreboard);`candidates.jsonl` +本轮条目
- **未改动** `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` /
  `mxfp8_gemm_ext.cu` / `mixed_gemm.py`(验证路径完好无回归)
