# B1 优化第五轮 — 报告 (双相 TMA 诊断并修复 / roofline / 密集 Pareto 曲线 / tile 扫描)

> 全程 remote-only + detached+落盘。正确性 gate:**所有测量点 cosine=1.000000**（rel≈2.1e-4，纯 fp16 round）。
> GPU: RTX PRO 6000 Blackwell (SM120)，GPU0/GPU1 空闲。
> 性能标尺:**excess vs CUTLASS**，即 excess_pct = (t_B1 - t_ideal)/t_ideal，
> t_ideal(r) = t_fp4·(1-r) + t_fp8·r，t_fp4/t_fp8 = 同场 pure NVFP4/MXFP8 全-K
> presf GEMM(= CUTLASS block-scaled 上/下界)。

## 结论先行 (headline)

- **T1 — 双相 TMA(recipe B)从"全程慢 2.7x"修到"全程 ≤ CUTLASS 界、多数点在界内":**
  round-5 开局双 TMA 对比 cp.async 全程慢(r=0 +175%、r=1 +54% vs CUTLASS ideal)。
  ncu 精确定位两个根因并逐一修复(每步单独 cosine+ncu 验证):
  1. **SW_NONE flat smem → SW64 swizzle**:recipe B 为绕开 round-4 的 SU4 编译阻塞用了
     非 swizzle 环,导致 **ldmatrix shared-load bank conflicts 18.9M–37.8M**(ncu
     `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`),MIO throttle 拉满、TC 饿死
     (13–36%)。改用与"零冲突 phase-2 TMA kernel"相同的 SW64 swizzle 环:对 swizzle 后的
     uint8 环做 `recast<uint4_t>` + `SM75_U32x4_LDSM_N`(**用 SM75 而非 SU4,绕开 round-4
     阻塞,编译通过**)。bank conflict → 0,**fp8 腿 r=1 +54%→+18%**。
  2. **生产者标量 SF-scatter → 向量化**:fp4 相每 chunk 由 1 个 producer warp 做
     2048 字节的 **1-byte 标量 global gather**,把 `long_scoreboard` 顶到 4.42–5.18
     卡在关键路径。改成**每行一次 coalesced uint64(fp4)/uint16(fp8)**、load/store 分离
     拿 MLP:long_scoreboard 4.42→1.45,global requests 1.25M→0.33M,**fp4 腿 r=0
     +141%→-10%**。
  - **最终 down-class excess vs CUTLASS(M2048 N2048 K8192):r=0 -7.6%、r=1/8 -6.7%、
    r=1/2 -1.9%、r=1 +4.1%**;**全部 ≤5%,多数在界内(负 excess)**。
  - **判负的假设(ncu 证伪):** occupancy/寄存器不是杠杆(双 TMA 与达标的 phase-2 TMA
    occupancy 同为 18.6%、1 CTA/SM,regs 128≤134);mbarrier-wait 不是瓶颈(st_mbar=0、
    st_barrier≈0,胜出的 phase-2 TMA 的 st_wait 也是 ~2.5);**SM120 的 CUTLASS collective
    本身也不用 setmaxnreg**(grep 无命中——那是 Hopper warpgroup 的手段)。
- **泛化(密集曲线):qkv 与 down 两个 M(2048/8192)全部点 ≤5% vs CUTLASS**(down M=8192
  甚至 -5~-18%,大幅低于界)。**唯一不达标是 gate(大 N=8192、小 K=2048):+11~+32%,
  且 cp.async 同样 +15~+36%** —— 是**两套手写 kernel 共有的大-N/小-K 结构性低效**,非
  相位混合(与 round-4 gate 56%-band 一致)。
- **roofline:这些 kernel 都不是 DRAM 带宽受限**——achieved DRAM BW 仅 **5.5–8.6% of peak**
  (88–137 GB/s vs GDDR7 ~1.8TB/s)。瓶颈在片上:**TC 利用率(36–57%)+ 内存 pipe SOL
  (L1/L2/shared,cp.async 达 70–82%)**。所以"吃更多 DRAM 带宽"这条杠杆在这里不适用;
  down/qkv 已贴着 CUTLASS 界(同样的 TC 天花板),属实际 roofline 收敛。gate 的 20% 缺口
  是 128×128 tile 在大 N/小 K 上的 wave/overhead 低效(非 DRAM、非 L2、非 bank conflict)。

---

## T1 — 双相 TMA 诊断与修复

### ncu 证据(down-class r=1/2 与端点,M2048 N2048 K8192,--set full)

三个版本在 r=0(纯 fp4)下的机制演进:

| 指标 | 原始 SW_NONE | +SW64 swizzle | +向量化 SF (FINAL) |
|---|---|---|---|
| 时长 | 535 us | 480 us | **145 us** |
| TC active % | 13.2 | 14.8 | **52.2** |
| shared-ld bank conflicts | **18.9M** | 0 | 0 |
| st_mio throttle | 1.86 | 0.04 | 0.08 |
| st_long_scoreboard | 5.18 | 4.42 | **1.45** |
| global load requests | 1.25M | 1.25M | **0.33M** |
| regs | 129 | 128 | 128 |
| occupancy % | 18.7 | 18.7 | 18.7 |

r=1(纯 fp8)apples-to-apples 对比:dual(SW_NONE)393us/37.8M 冲突/TC36% vs 胜出的
phase-2 TMA 299us/0 冲突/TC49% —— 同一 fp8 workload,唯一差异是 smem 布局,证明 swizzle
是 fp8 腿的根因。

### 修法(代码:`b1_tma_ext.cu` `b1_tma_dual_kernel` + `launch_tma_dual`)
1. `SmemLayoutFlat` → `SmemLayoutA_t/B_t`(SW64,CHB=64B 对 fp4/fp8 相同,一个 atom 两用);
   consume4 用 `recast<ElemFP4>(sA(_,_,s))`,consume8 直接用 uint8 环;TMA 描述符按 swizzle 布局建。
2. producer SF 填充:fp4 每行 1×uint64、fp8 每行 1×uint16,gmem→reg 与 reg→smem 分离;
   sSFA/sSFB 对齐提到 alignas(16)。字节布局与原标量版逐字节一致(cosine 不变)。

---

## roofline 分析(ncu,GDDR7 peak ~1.8TB/s)

| 场景点 | 时长 | DRAM BW % peak | achieved GB/s | TC act % | MEM pipe SOL % | 判定 |
|---|---|---|---|---|---|---|
| down fp4 r=0 (FINAL) | 145us | 8.2 | 131 | 52.2 | 36.6 | 片上/TC-bound |
| down r=1/2 (FINAL)   | 195us | 8.6 | 137 | 57.1 | 36.0 | 片上/TC-bound |
| down cp r=1/2        | 232us | 7.2 | 115 | 46.6 | 69.6 | mem-pipe(L2/sh) |
| down dual r=1        | 306us | 7.3 | 116 | 47.8 | 42.7 | 片上/TC-bound |
| gate dual r=0        | 133us | 5.7 | 91  | 42.2 | 42.6 | overhead/TC |
| gate dual r=1        | 211us | 6.8 | 108 | 53.0 | 42.7 | overhead/TC |
| gate cp r=0          | 135us | 20.5| 327 | 39.9 | 82.3 | mem-pipe |

**结论:无一是 DRAM 带宽受限(≤20% peak)。** down/qkv 的 TC 48–57% 已贴 CUTLASS 界
(CUTLASS fp4 在这些 shape 本身也到不了 100% TC),属 roofline 收敛。gate 的真实瓶颈是
小-K(2048)下 prologue/pipeline-fill/epilogue 的 overhead 占比高 + 大-N 下 128×128 tile 的
利用率不足(TC 42–53%),既非 DRAM、也非 L2(round-4 已证 93.7% 命中无余量)、也非 bank
conflict(0)。

---

## 密集 Pareto 曲线(`ratio_curve_r5.csv`,全 cosine=1.0,画图用)

M ∈ {2048, 8192} × {qkv(N2048K2048,17点)、down(N2048K8192,19点)、gate(N8192K2048,17点)}。
每点:dual-TMA(best st3/4/5)、cp.async、同场 CUTLASS presf 界、excess vs ideal、best 派发。

excess vs CUTLASS ideal 摘要(best variant):

| 场景 / M | r=0 | 生产 r≈1/8 | r=1/2 | r=1 | 达标? |
|---|---|---|---|---|---|
| qkv  M=2048 | -3.5% | -5.2% | -4.3% | +1.2% | ✅ 全 ≤5% |
| qkv  M=8192 | -10.5% | -8.8% | -6.6% | -4.5% | ✅ |
| down M=2048 | -7.6% | -6.7% | -1.9% | +4.1% | ✅ |
| down M=8192 | -16.4% | -15.9% | -11.6% | -6.6% | ✅ 大幅低于界 |
| gate M=2048 | +18.3% | +18.9% | +11.4% | +13.6% | ❌ 结构性 |
| gate M=8192 | +32.5% | +30.8% | +19.9% | +23.8% | ❌ 结构性 |

**dual-TMA 是 down 全程、qkv/gate 高-r 的 best variant;cp.async 仅在 qkv/gate 低-r(小K)
胜出。** gate 两个 M 都是唯一不达标,且 cp.async 在 gate 同样 +15~+36%(两套 kernel 共有)。

## T3 — tile × stage 网格扫描(gate/down,M∈{2048,8192})

脚本 `tile_sweep.py`,tile ∈ {128×128, 128×256(BN=256,WN=8), 256×128(BM=256,WM=4)}
× st,全 cosine-gate。大 tile 受 99KB 静态 smem 限制只能 st3。

excess vs CUTLASS ideal(best st):

| 场景 / M / r | 128×128 | 128×256 | 256×128 |
|---|---|---|---|
| gate M2048 r=0 | **+24.0%** | +70.3% | +69.5% |
| gate M8192 r=0 | **+35.7%** | +71.6% | +72.0% |
| gate M8192 r=1 | **+26.2%** | +41.6% | +41.7% |
| down M2048 r=0 | **-13.0%** | +20.6% | +20.6% |
| down M8192 r=0 | **-19.0%** | +25.4% | +25.6% |

**结论(诚实负结果):128×128 st3 在所有 shape 都最优;更大 CTA tile 显著更差**(smem 限制
只能 st3 + 占用下降 + 算术强度未补偿)。**tile/占用不是 gate 的杠杆**(与 round-4 占用负结果
一致)。gate 大-N 不是被 tile-count/wave 量化卡(更大 tile 本应减 wave 却更慢)。

### gate 20% 缺口的收敛判定
- **非 DRAM 带宽**(roofline 5.7–6.8% peak)、**非 L2**(round-4:93.7% 命中无余量)、
  **非 bank conflict**(0)、**非占用/tile**(本轮扫描 + round-4 均负)。
- 真实根因:**小 K(2048)→ prologue/pipeline-fill/epilogue overhead 占比高**(down 的大 K
  把 mainloop 摊薄到界内,gate 做不到);CUTLASS 在 gate 领先 ~20% 源于其 persistent/
  stream-K 调度 + cooperative TMA-store epilogue 对小-K/大-N 的摊销。**追平需 persistent/
  stream-K 级改造,ROI 低**(round-4 已记),本轮以证据确认后暂缓。

---

## 产品派发策略(round-5 更新)

| regime | 变体 | 依据 |
|---|---|---|
| **down-class 全 r(大 K)** | **dual-TMA st3**(tile 128×128) | excess -16~+4%,全程 best |
| **qkv/gate 高 r (r≥~0.4)** | **dual-TMA st3** | 高 r best variant |
| qkv/gate 低 r(小 K,r≲0.3) | **cp.async auto** | 小 K dual 略逊、cp.async 近界 |
| decode / 小 M (M≤16) | **cp.async auto**(轻 kernel) | launch-light |
| gate 大 N | 两者皆 +20%(结构性) | 暂无单核修法,待 persistent/stream-K |

**dual-TMA 已成为 down + 高-r 的生产主路径**(round-4 时它还全程判负/未落地)。

## 收敛评估

1. **T1 双相 TMA:从"全程 2.7x 慢"修到"down/qkv 全 M 全 r ≤5% vs CUTLASS、多数在界内"** —
   本轮最大成果。两个根因(swizzle、SF 向量化)均 ncu 实证、单独验证、已 commit。
2. **down/qkv 已贴 roofline**(TC 48–57%、DRAM 仅 7%,与 CUTLASS 同 TC 天花板):**收敛**。
3. **gate(大 N、小 K)+11~+32%:** 本轮用 roofline + tile 扫描排除了 DRAM/L2/占用/tile/
   bank-conflict,定位为**小-K overhead + 需 persistent/stream-K**,**ROI 低,判为已知结构性
   残差**,暂缓。
4. **判断:B1 单核混合 GEMM 在 qkv/down(含 decode→prefill 全 M)已对标 CUTLASS 收敛;
   gate 大-N 是唯一剩余结构性缺口,非单核调参可解。**

## 产物
- `b1_tma_ext.cu`:dual-TMA 修复(SW64 swizzle + 向量化 SF)+ tile 参数(T3 扫描,128×128 默认);
  **未改动** `b1_fused_ext.cu` / `fused_gemm_ext.cu` / `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` / `mixed_gemm.py`。
- `ratio_curve_r5.csv`(密集 Pareto,M∈{2048,8192},全 cosine=1.0)。
- `ratio_curve_r5.py` / `tile_sweep.py` / `gen_check.py` / `prof_dual.py`(扫描+ncu driver)。
- profiles:`profile/b1_round5/*.ncu-rep`(bank-conflict、roofline、gate 证据)。
