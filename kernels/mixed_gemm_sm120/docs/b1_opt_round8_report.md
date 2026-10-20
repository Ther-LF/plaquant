# B1 优化第八轮 — 报告（TC 占空比体检 + register K-block 流水 KPIPE）

> 全程 remote-only（10.208.130.90, luofan-plaquant / luofan-ncu）。抗中断：自包含 detached + 逐行 flush。
> 正确性 gate：cosine=1.000000（全点通过）。GPU: RTX PRO 6000 Blackwell (SM120)。
> **本轮判据升级（项目负责人）**：收敛的唯一判据是 **TC 占空比**，不再是 excess。两个 rollup 都要看：
> - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`（ELAPSED = TC 活跃占整个 kernel 时长）
> - `...pct_of_peak_sustained_active`（ACTIVE = 活跃期间相对峰值的发射强度）
> 真·打满 = 两者都逼近 100% 或 CUTLASS 同 shape 水平。参考 CUTLASS 4.4.2（remote）。

---

## Part 0 — 先验：前人怎么做（每个杠杆的 prior art）

- **T1 register K-block 流水**：直接移植 **CUTLASS `sm120_blockscaled_mma_tma.hpp` 的 `CollectiveMma::mma()`** 结构
  —— 持久寄存器 fragment（K_BLOCK_MAX=KMUL 个 slot），prologue `copy_kblock(0)`，稳态循环 `copy_kblock(k+1)`
  （ldmatrix smem→rmem）**先于** `gemm_kblock(k)`（MMA），并在 chunk 边界（`k_block==K_BLOCK_MAX-1`）释放当前
  smem stage + 预取**下一 stage 的 k_block0**（跨 stage 寄存器预取）。这与 **LeetCUDA(xlite-dev) 的
  `ldmatrix(k+1) ⊗ MMA(k)` 寄存器乒乓**是同一模式。per-kb 的 `fp4_shift_A/B` 调用也照搬 CUTLASS 的
  `fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block))`（line 733/834）。

---

## Part 1 — TC 占空比体检表（round-7 kernel，全 shape × M × r）

ncu（luofan-ncu，`-c 1 -s 5`，单 kernel 实例，warm）。per-shape round-7 best cfg（qkv/down: t0 st3 sw0 非持久；
gate: t0 st3 sw8 持久）。CUTLASS presf 纯端点（r=0 fp4 / r=1 fp8）作同 shape 对标。

| shape·M | 我们 TCela%（r=0→1/8→1/2→1） | 我们 TCact% | CUTLASS fp4/fp8 TCela | CUTLASS fp4/fp8 TCact |
|--|--|--|--|--|
| qkv·2048 | 25.5 / 23.7 / 31.1 / 31.9 | 38.3 / 35.3 / 46.3 / 48.8 | 36.7 / 46.4 | 56.8 / 67.6 |
| qkv·8192 | 35.5 / 38.9 / 42.0 / 45.8 | 39.8 / 43.3 / 47.4 / 51.4 | 57.6 / 65.8 | 65.8 / 74.1 |
| gate·2048 | 39.5 / 41.4 / 45.0 / 49.2 | 45.2 / 46.7 / 50.8 / 54.7 | 58.8 / 65.9 | 67.3 / 74.5 |
| gate·8192 | 46.1 / 48.3 / 52.1 / 54.5 | 47.9 / 50.0 / 53.4 / 55.9 | 70.2 / 74.2 | 72.2 / 76.6 |
| down·2048 | 36.2 / — / 38.8 / 39.6 | 52.5 / — / 56.9 / 58.5 | 52.8 / 55.2 | 80.4 / 81.6 |
| down·8192 | 47.1 / 50.1 / 52.8 / 55.0 | 52.3 / 55.6 / 58.4 / 61.0 | 77.8 / 75.5 | 86.2 / 83.5 |

**结论（round-7「qkv/down 已收敛」判定作废，与项目负责人一致）**：
1. **没有任何 shape 的 TC 打满**。我们 25–55% ELAPSED / 38–64% ACTIVE，**全线比 CUTLASS 低 15–25pp**。
2. **同一纯 NVFP4 数学（r=0）**：gate·8192 我们 46.1% vs CUTLASS 71.1%（24pp 差）—— 纯 mainloop 质量差距。
3. **两 rollup 的分型**：
   - gate·8192：ELAPSED≈ACTIVE（~47≈~49）→ TC 几乎全程活跃但**只半速发射**（intensity 问题，非空窗）。
   - qkv/down·2048：ELAPSED≪ACTIVE（如 25.5 vs 38.3）→ **既有空窗又半速**。
4. **主导 stall 全程是 `wait`≈2.8–2.95**（MMA 固定延迟依赖）+ fp4 段的 `long_scoreboard`（producer 全局 SF load）。
   —— `wait` 正是 register K-block 流水（T1）要治的；long_scoreboard 指向 SF-via-TMA（对账 #3，未做）。

产物：`logs/tc/*.csv`（ELAPSED+stall）、`logs/tcA/*.csv`（ELAPSED+ACTIVE）、`agg_tc.py`。

---

## Part 2 — T1：register K-block 流水（KPIPE 开关）

### 实现
`b1_tma_dual_kernel` 加模板开关 `int KPIPE`（默认 0 = round-7 路径，受保护、byte 等价）。KPIPE=1 时 consumer 主循环
重构为 `run_tile_kpipe(giter0)`：**按 phase 分别流水**（fp4 段 / fp8 段各自 uniform atom），每段 prologue 载 kb0，
稳态 `load(kb+1)` 先于 `mma(kb)`，chunk 边界 `mbar_arrive(free[sc])` 释放当前 stage + `mbar_wait(full[next])`
+ `load(next,0)` 预取下一 chunk 的 kb0。连续 `giter` 保持 mbar ring parity 与非流水路径完全一致（producer 不动）。
持久/非持久两条路径都走 KPIPE。相格盘 seam 只有 1 次非重叠 ldmatrix（round-7 实测 seam≤0，可忽略）。

- **正确性**：3 shape × M{2048,8192} × r{0,1/8,1/2,1} × kp{0,1}，**全部 cosine=1.000000**（`logs/kpipe_gate.log`）。
- **编译**：curated 实例化（km2 st3，tile{0,6}×sw{0,8}×persist{0,1}，8 个 KPIPE kernel），一次通过。

### TC 前后对比（ncu，kp0→kp1，contention-robust；`logs/tcK/*.csv`，`agg_kpipe.py`）

| shape·M | Kh | TCela kp0→kp1 | TCact kp0→kp1 | wait kp0→kp1 | dur kp0→kp1(us) |
|--|--|--|--|--|--|
| qkv·8192 | 2048(r1) | 46.7→49.2 (+2.5) | 52.3→55.2 (+2.9) | 2.79→2.57 | 215.8→205.1 |
| down·8192 | 8192(r1) | 55.5→57.9 (+2.4) | 61.3→64.0 (+2.8) | 2.80→2.56 | 708.8→681.7 |
| gate·8192 | 2048(r1) | 53.2→56.6 (+3.4) | 56.2→58.7 (+2.6) | 2.80→2.62 | 742.0→693.1 |
| gate·8192 | 1024(r.5) | 52.4→54.7 (+2.3) | 53.8→56.1 (+2.3) | 2.86→2.58 | 561.1→539.6 |
| gate·8192 | 0(r0) | 46.5→45.4 (−1.1) | 48.6→47.7 (−0.9) | 2.95→2.51 | 424.1→433.5 |

**机理证据（验收核心）**：**`wait` stall 全点从 2.8–2.95 降到 2.5–2.6** —— register 流水确实把 ldmatrix 叠进了
MMA、削掉了 MMA 依赖空窗，正是设计目的。TC 两 rollup 随 **fp8 占比** 上升：
- 纯 fp4（r=0）：中性或微降（gate·8192 −1.1pp）。fp4 段 KMUL=2、MMA atom K=64（大块、单次重叠占比小），且其
  主导 stall 是 long_scoreboard 不是 wait → register 流水收益小，额外指令/寄存器压力反而微亏。**→ r=0 应关 KPIPE。**
- 混合 / fp8 重（r≥1/8）：稳定 +1~3.4pp TC、−3~7% 延迟。fp8 段 MMA atom K=32（块小、次数多）→ 重叠收益大。

### 诚实判定
T1 **有效且 bit-exact**，但增益**温和**（+1–3.4pp TC），**未闭合到 CUTLASS 水平**（kp1 后 ACTIVE 仍 ~55% vs
CUTLASS ~74%）。它削掉了 `wait`（MMA 依赖）这一类空窗，但剩余差距来自别处（见 Part 3）。

---

## Part 3 — 剩余 TC 差距的定位（ncu 证据）+ 下一杠杆

KPIPE 后仍有 ~15–20pp ACTIVE 差距（vs CUTLASS）。证据指向两条**未做**的杠杆：
1. **fp4 段 `long_scoreboard`（~1.5–2.9）= producer 的全局 SF load**（`sf_copy_row` 从 gmem 读 SFAl/SFBl）。
   CUTLASS 用 **TMA 异步载 SF**（对账 #3，`tma_load_sfa/sfb`，与 A/B 同一 mbarrier 事务）→ 不占 issue slot、
   不产生 long_scoreboard。这是 fp4 段（低 r）TC 低的主因，KPIPE 治不了。**→ round-9 首选：SF 走 TMA。**
2. **ACTIVE intensity 仍 ~55% = 发射节奏/occupancy**。我们 1 CTA/SM、8 consumer warp + 1 producer warp（producer
   抢 issue）。`wait`≈2.5 仍在 → 单 warp-group 藏不住 MMA 延迟。CUTLASS cooperative 用 2 个 math warpgroup +
   异步 DMA warp。**→ 提 occupancy / 更多 warp 藏延迟**（降 smem 让 2 CTA/SM，或 ping-pong 双 math group）。

**mainloop 段无下陷判据（layer 2）**：gate·8192 ELAPSED≈ACTIVE → TC 几乎无空窗（mainloop 不掉零），问题是
intensity；qkv/down·2048 ELAPSED≪ACTIVE → mainloop 内确有 per-chunk/fill 空窗，KPIPE 已把 ELAPSED 往 ACTIVE 推
（qkv·2048 r1 33.1→34.8 ela）。（CLI PM-sampling 时间线导出受限；以两 rollup 的差值作 mainloop 空窗的等价度量。）

---

## Part 3.5 — 延迟 / excess 前后（clean GPU，无 co-tenant；`r8_kpipe_lat.csv`）

excess vs 同场 CUTLASS presf 界 ideal(r)=t_fp4·(1−r)+t_fp8·r。kp0→kp1 delta 是 KPIPE 的净贡献。

| shape·M | r | excess kp0→kp1 | speedup | shape·M | r | excess kp0→kp1 | speedup |
|--|--|--|--|--|--|--|--|
| gate·8192 | 0 | +12.6→+14.3 (+1.7) | −1.5% | gate·8192 | 1/2 | **+15.7→+9.8 (−6.0)** | +5.1% |
| gate·8192 | 1/8 | +14.7→+12.6 (−2.1) | +1.8% | gate·8192 | 1 | **+18.0→+10.2 (−7.8)** | +6.6% |
| qkv·8192 | 1/2 | −5.6→−9.0 (−3.4) | +3.6% | qkv·8192 | 1 | −0.4→−4.0 (−3.5) | +3.5% |
| down·8192 | 1/2 | −12.9→−14.9 (−2.0) | +2.3% | down·8192 | 1 | −6.4→−10.6 (−4.2) | +4.4% |
| down·2048 | 1 | +2.5→−2.6 (−5.2) | +5.0% | qkv·2048 | 1 | +3.6→−0.1 (−3.7) | +3.6% |

**KPIPE 把延迟 excess 在 r>0 普遍压低 2–7.8pp**（r=0 微升 0.3–3.2pp → r=0 关 KPIPE）。**gate·8192 的 fp8-重端
从 +18%/+16% 降到 +10.2%/+9.8%（r=1/r=0.5 接近或进入 <10%）** —— 叠在 round-7 persist 之上的真实增益。

---

## Part 4 — per-shape dispatch（round-8 更新）

| 形状 | KPIPE | scheduler | cfg | 依据 |
|--|--|--|--|--|
| 纯 fp4（r=0，任意 shape） | **0** | 非持久（qkv/down）/ 持久（gate） | t0/t6 st3 | KPIPE 在 r=0 微降 |
| qkv / down 混合·fp8（r>0） | **1** | 非持久 | t0 st3 sw0 | +1~3pp TC |
| gate 混合·fp8（r>0） | **1** | 持久(P1) | t0 st3 sw8 | +2~3.4pp TC，dur −3~7% |

---

## Part 5 — 收敛判定（TC 占空比，per shape）

判据（项目负责人）：mainloop 段 TC 全程贴峰、无下陷，且 ELAPSED/ACTIVE 都逼近 CUTLASS 同 shape 或峰值。

| shape | TC 现状（kp1 后） | CUTLASS 同 shape | 判定 | 剩余空窗来源（ncu） |
|--|--|--|--|--|
| qkv | ela 36–49 / act 41–55 | 58–66 / 66–74 | **未收敛** | wait 2.5（残）+ occupancy；fp4 段 long_scoreboard |
| gate | ela 45–57 / act 48–59 | 59–74 / 67–77 | **未收敛**（最接近，mainloop 无空窗但半速） | ACTIVE intensity（发射节奏/occupancy） |
| down | ela 36–58 / act 52–64 | 53–78 / 80–86 | **未收敛** | ACTIVE 差最大；occupancy + fp4 long_scoreboard |

**全局结论：没有任何 shape 的 TC 打满；round-7「qkv/down 已收敛」判定作废。** T1（register K-block 流水）是
正确方向的一步 —— bit-exact、`wait` stall 2.9→2.5、TC +1~3.4pp、fp8-重端延迟 −3~7% —— 但**单靠它远不足以贴峰**。
按 ncu 证据，闭合剩余 15–20pp 的两条杠杆（均未做，留 round-9）：
1. **SF 走 TMA**（对账 #3）——治 fp4 段 long_scoreboard（producer 全局 SF load），低-r TC 低的主因。
2. **提 occupancy / 多 warp-group**——治 ACTIVE intensity（半速发射），gate 的主残差。

## 产物
- `b1_tma_ext.cu`：新增 `KPIPE` 模板开关（默认 0，保护 round-7 byte 等价）+ `kpipe` pybind 参数 + curated 实例化。
- 体检：`prof_tc.py`、`run_tc_ncu.sh`（ELAPSED+stall）、`run_tc_active.sh`（+ACTIVE）、`agg_tc.py`、`logs/tc*`。
- KPIPE 验证/测量：`kpipe_check.py`（cosine 全过）、`run_tc_kpipe.sh`+`agg_kpipe.py`（TC 前后）、
  `r8_kpipe_lat.py`+`r8_kpipe_lat.csv`（延迟/excess 前后，clean GPU）。
- 未改动（保护）：`b1_fused_ext.cu`/`fused_gemm_ext.cu`/`nvfp4_gemm_ext.cu`/`mxfp8_gemm_ext.cu`/`mixed_gemm.py`/`pure_presf_ext.cu`。
