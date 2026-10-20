# B1 优化第九轮 — 报告（Lever 1: SF via async / Lever 2: occupancy 定位）

> 全程 remote-only（10.208.130.90, luofan-plaquant / luofan-ncu, RTX PRO 6000 Blackwell SM120）。
> 抗中断：自包含 detached 脚本 + 逐行 flush CSV + 从磁盘收割。
> 正确性 gate：cosine=1.000000（全点通过，硬门）。
> 收敛判据（项目负责人）：**TC 占空比**两 rollup 同时逼近 CUTLASS 同 shape；mainloop 段不下陷。
> - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`（ELAPSED）
> - `...pct_of_peak_sustained_active`（ACTIVE）

---

## Part 0 — 先验：前人怎么做（每个杠杆的 prior art）

- **Lever 1（SF 异步化）借鉴了谁的什么**：CUTLASS `sm120_blockscaled_mma_tma.hpp` 的 `load()`
  把 A/B/**SFA/SFB 四个 TMA 全部 `.with(*tma_barrier)` 挂在同一个 producer barrier 上**，
  `TmaTransactionBytes = SF 字节 + A/B 字节`（line 255-262，一次 expect_tx 声明合并字节）。
  SF 的 gmem 用交错 atom 布局 `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA` + `make_tma_copy<uint16_t>`
  （line 309-377）。**字面照搬 cp.async.bulk.tensor 被对齐卡住**（见 Part 2）；
  我方改用 **SM80 cp.async + `cp.async.mbarrier.arrive.noinc`** 把 SF 异步化挂到同一 mbarrier
  —— 该 PTX 模式来自 CUDA Programming Guide / PTX ISA（“arrive-on from prior cp.async ops；
  .noinc ⇒ init count 已计入”），以及 LeetCUDA(xlite-dev) “多次异步搬运共享一个 mbarrier、
  expect_tx 一次声明总字节” 的写法（DeepSeek web search 命中）。
- **Lever 2（occupancy）借鉴了谁的什么**：KernelWiki `hw-2sm-cooperative`（SM100 2-SM cooperative，
  **不适用 SM120**，无 tcgen05/cluster）、flashinfer PR-2460（SM120 FP4 GEMM tile configs + streamK，
  RTX PRO 6000 上 0.124→0.069ms —— 靠 **多 CTA/tile 调度**填 SM，而非降 smem）；
  CUTLASS SM120 collective 的 cooperative = 2 个 math warp-group。本轮用这些做 occupancy 的对照诊断。

---

## Part 1 — CUTLASS 同 shape TC 端点（本轮实测，收敛靶，`logs/tcC2/`）

`ncu -c 1 -s 0 -k regex:device_kernel`，CUTLASS 4.4.2 presf 纯端点（r=0 纯 NVFP4 / r=1 纯 MXFP8）。
与 round-8 表一致（CUTLASS 参考稳定）。

| shape·M | fp4 TCela/act | fp8 TCela/act |
|--|--|--|
| qkv·2048 | 37.1 / 55.9 | 46.2 / 67.2 |
| qkv·8192 | 61.1 / 68.5 | 67.6 / 75.1 |
| gate·2048 | 60.8 / 68.8 | 67.4 / 75.4 |
| gate·8192 | 70.5 / 72.0 | 75.0 / 76.8 |
| down·2048 | 54.0 / 80.5 | 55.6 / 80.9 |
| down·8192 | 78.1 / 86.2 | 75.2 / 83.7 |

---

## Part 2 — Lever 1（PRIMARY）：SF 异步化（SFASYNC 开关）

### 设计与为何不是字面 cp.async.bulk.tensor
round-8 定位：fp4 段的 `long_scoreboard`(~1.5–2.9) = producer 的**全局 SF load**（`sf_copy_row`
从 gmem 读 SFAl/SFBl 到寄存器再 st.shared，st 依赖 ld → 长记分牌空窗）。CUTLASS 的解法是 SF 走 TMA。

**字面 TMA 被对齐卡住**：我方 SF 的 gmem 是 plain 连续 `uint8[R, K/16]`(fp4) / `uint8[R, K/32]`(fp8)
（`quantize_nvfp4/mxfp8` 的 `sf_f8.view(R, K//BLOCK)`）。2D TMA 要求 gmem 行 stride 是 16B 的倍数；
**小-Kh fp8（Kh=256 → 行 stride = 256/32 = 8B）< 16B**，描述符非法。这正是 CUTLASS 必须把 SF 重排成
交错 atom 布局（`tile_atom_to_shape_SFA`）的原因 —— 但重排会**改变 SF 的 smem 布局**，牵动 consumer 的
`tt_partition_fragment_SFA` + consume4/consume8 + KPIPE 全链路，对 bit-exact 硬门是高回归风险。

**本轮取法（SFASYNC）**：保持 **SF 的 smem 布局与 round-8 字节完全相同**（consumer 一行不改 → bit-exact 天然成立），
只把 fp4 段 producer 的 SF 搬运从 `ld.global + st.shared` 换成 **SM80 cp.async（8B/行，NBLK4=8）**，
完成度挂到**同一个 mbarrier**：
- thread0 用非到达形 `mbarrier.expect_tx`（只记 A/B 的 TX 字节，不计到达）+ A/B TMA；
- 全 32 producer 线程发 cp.async SF 后各发 `cp.async.mbarrier.arrive.noinc` —— 32 次到达（init count=32），
  每次都被各自 cp.async 组完成门控；mbarrier 翻转 = A/B TMA 字节到齐 **且** 32 个 SF cp.async 完成。
- fp8 段（SF 仅 2B/行，cp.async 最小 4B 不支持）保持 round-8 plain 路径。
模板开关 `int SFASYNC`（默认 0 = round-8 byte 等价、受保护），curated 实例化 km2/st3/tile0 × sw{0,8} ×
persist{0,1} × kpipe{0,1}；与 KPIPE/PERSIST 正交可组合。

- **正确性**：3 shape × M{2048,8192} × r{0,1/8,1/2,1} × kpipe{0,1} × sfa{0,1}，**全部 cosine=1.000000**
  （`logs/r9_sfa_gate.log`，SFASYNC_GATE_ALL_PASS）。

### TC 前后（ncu，sfa0→sfa1，per-shape 最佳 kpipe：r=0→kp0 / r>0→kp1；`logs/tcS/`，`agg_sfasync.py`）

| shape·M·Kh | TCela sfa0→1 | TCact sfa0→1 | lsb 0→1 | wait 0→1 | dur 0→1 (us) |
|--|--|--|--|--|--|
| qkv·2048·0 | 26.6→28.3 (+1.7) | 40.0→43.1 (+3.1) | 2.90→2.16 | 2.90→2.83 | 47.1→44.6 |
| qkv·2048·256 | 27.9→31.2 (+3.3) | 41.5→47.1 (+5.6) | 2.54→1.95 | 2.53→2.52 | 50.9→45.6 |
| qkv·2048·1024 | 32.7→34.5 (+1.8) | 49.9→53.0 (+3.1) | 1.77→1.56 | 2.56→2.55 | 57.7→55.2 |
| down·2048·0 | 36.9→39.6 (+2.7) | 54.5→59.3 (+4.8) | 1.46→0.87 | 2.95→2.87 | 137.0→129.4 |
| down·2048·1024 | 36.4→44.2 (**+7.8**) | 52.5→65.7 (**+13.2**) | 1.74→0.77 | 2.51→2.53 | 159.1→132.4 |
| down·2048·4096 | 40.8→43.7 (+3.0) | 60.1→64.5 (+4.4) | 1.18→0.92 | 2.55→2.55 | 187.4→176.2 |
| gate·2048·0 | 40.2→42.2 (+2.0) | 46.3→48.0 (+1.7) | 2.47→1.93 | 2.93→2.86 | 126.1→121.5 |
| gate·2048·256 | 42.5→45.7 (+3.2) | 48.8→52.4 (+3.7) | 1.98→1.51 | 2.52→2.51 | 134.7→127.1 |
| gate·2048·1024 | 47.8→48.7 (+0.9) | 54.5→55.6 (+1.0) | 1.42→1.26 | 2.58→2.57 | 159.0→157.3 |
| qkv·8192·0 | 38.6→40.9 (+2.4) | 43.1→45.9 (+2.8) | 2.43→1.80 | 2.90→2.83 | 130.9→123.5 |
| qkv·8192·256 | 40.7→45.3 (+4.6) | 45.2→50.9 (+5.7) | 2.26→1.61 | 2.52→2.53 | 138.8→128.4 |
| qkv·8192·1024 | 47.3→49.3 (+2.1) | 53.2→55.3 (+2.1) | 1.52→1.33 | 2.56→2.55 | 161.7→156.5 |
| down·8192·0 | 49.0→54.1 (+5.1) | 54.4→60.1 (+5.6) | 1.42→0.78 | 2.95→2.87 | 401.7→368.8 |
| down·8192·1024 | 53.9→60.3 (**+6.4**) | 59.4→66.8 (**+7.4**) | 1.23→0.72 | 2.54→2.53 | 413.3→373.3 |
| down·8192·4096 | 55.2→59.3 (+4.1) | 61.0→65.6 (+4.6) | 1.09→0.82 | 2.55→2.56 | 538.6→501.8 |
| gate·8192·0 | 46.9→48.1 (+1.2) | 48.9→49.7 (+0.8) | 2.21→1.75 | 2.95→2.87 | 422.3→409.3 |
| gate·8192·256 | 50.4→52.6 (+2.2) | 52.3→54.2 (+1.9) | 1.71→1.39 | 2.54→2.52 | 436.6→422.9 |
| gate·8192·1024 | 55.2→55.2 (+0.0) | 56.5→56.9 (+0.4) | 1.29→1.15 | 2.58→2.57 | 532.4→538.7 |

**机理证据（验收核心）**：`long_scoreboard` **全 18 点下降**（这是本杠杆要治的全局 SF-load 空窗），
TC 两 rollup **无一处回退**，增益随 **fp4 占比** 上升（fp4-heavy / 混合最大：down·2048·Kh1024 TCact **+13.2pp**，
down·8192·Kh1024 +7.4pp，qkv·8192·Kh256 +5.7pp；纯 fp4 r=0 也 +0.8~5.6pp）。`wait` 基本不动（那是 KPIPE 治的 MMA
依赖，正交）。gate 的 fp8-最重端（r=0.5）近中性（+0.0~0.4pp，fp4 SF 本就少），但不倒亏。

### 净赚三指标（项目负责人指定；`logs/bc/`，`agg_bc.py`）—— 没把全局空窗换成 shared 冲突

| cell (M8192) | lsb 0→1（全局，应降） | ssb 0→1（shared 等待，别涨） | bank-conflict **LD** 0→1（必须 ~0） | bank-conflict **ST** 0→1（附赠） | TCact 0→1 |
|--|--|--|--|--|--|
| down·Kh0 | 1.48→0.79 ✅ | 0.25→0.20 ✅ | 0→0 ✅ | 425128→**0** | 54.1→59.9 |
| down·Kh1024 | 1.27→0.73 ✅ | 0.11→0.10 ✅ | 0→0 ✅ | 416520→10271 | 58.8→66.5 |
| qkv·Kh256 | 2.28→1.67 ✅ | 0.12→0.12 ✅ | 0→0 ✅ | 97077→8823 | 43.5→49.9 |
| gate·Kh1024 | 1.30→1.17 ✅ | 0.21→0.21 ✅ | 0→0 ✅ | 270848→43249 | 56.4→56.7 |

**判定：净赚成立。** (1) 全局 `long_scoreboard` 降；(2) `short_scoreboard` 平（没把全局空窗换成 shared 等待）；
(3) SF-read bank conflict **恒 0**（因为 SF 的 smem 布局与 consumer 读路径一字未改）。**附赠**：旧 plain-store 路径
（32 线程 st.shared.64 到 row*8B 的 strided 地址）本身制造了 **几十万次 store bank conflict**，cp.async 写入把它
从 42.5 万 → 0/1 万。一句话：long_scoreboard 降、bank conflict 与 short_scoreboard 没涨，净赚（还顺手清了 store 冲突）。

### 延迟 / excess 前后（clean GPU，无 co-tenant；`r9_sfa_lat.csv`）

excess vs 同场 CUTLASS presf 界 ideal(r)=t_fp4·(1−r)+t_fp8·r；sfa0→sfa1 是 Lever-1 净贡献（叠在 round-8 KPIPE 之上）。

| shape·M | r | excess sfa0→1 | speedup | shape·M | r | excess sfa0→1 | speedup |
|--|--|--|--|--|--|--|--|
| down·8192 | 1/8 | −21.6→−29.0 | **+9.5%** | down·8192 | 1/2 | −11.7→−20.2 | **+9.7%** |
| down·8192 | 1 | −5.2→−11.6 | +6.8% | down·2048 | 1/8 | −11.7→−17.1 | +6.1% |
| gate·8192 | 1/8 | +18.2→+9.6 | **+7.2%** | gate·8192 | 1/2 | +17.7→+11.8 | +5.0% |
| qkv·8192 | 1/8 | −11.3→−15.1 | +4.3% | down·2048 | 1/2 | −6.0→−9.9 | +4.2% |
| qkv·8192 | 1/2 | −8.4→−10.3 | +2.0% | gate·2048 | 1/8 | +22.4→+17.8 | +3.8% |

**Lever-1 在 r>0 普遍压低延迟 2–9.7%**（混合/ fp8-重端最大），纯 r=0 微赚 0.5–1.8%（SF 空窗也在那）。
gate·8192 fp8-重端 excess 从 round-8 的 +10% 级进一步降到 +9.6%(r=1/8) / +11.8%(r=1/2)。

---

## Part 3 — Lever 2（SECONDARY）：occupancy / 多 warp 藏 MMA 延迟（ncu 定位，诚实判负）

round-8 残差：ACTIVE intensity ~55% vs CUTLASS ~74%，`wait`≈2.5 不降 → 单 math warp-group 藏不住 MMA 延迟。
项目负责人给的两条路：降 smem 到 2 CTA/SM，或 ping-pong 双 math warp-group。本轮用 ncu 把这条路**量清楚**。

### 证据（`logs/occ/` + `logs/occ4/`，`agg_occ.py`）

| 配置（down·M8192·r0.5） | regs/thr | smem(B) | limit-Smem | limit-Reg | warp% | TCela | TCact | dur(us) |
|--|--|--|--|--|--|--|--|--|
| 128×128 st2 | 139 | 36896 | **1** | **1** | 18.6 | 47.9 | 53.0 | 607.8 |
| 128×128 st3（当前最佳） | 139 | 55344 | 1 | 1 | 18.6 | 53.2 | 58.8 | 550.1 |
| 128×128 st4 | 139 | 73792 | 1 | 1 | 18.6 | 42.9 | 47.4 | 681.6 |
| 128×128 st5 | 139 | 92240 | 1 | 1 | 18.6 | 42.7 | 47.1 | 683.4 |
| 64×128 st3（小 tile） | 137 | 41520 | **2** | **2** | 20.2 | 32.6 | 35.3 | 894.2 |
| 128×64 st3（小 tile） | 139 | 41520 | **2** | **2** | 20.3 | 30.7 | 33.4 | 943.8 |

**两条「降资源换 2 CTA/SM」的路都判负（有据）：**
1. **降 smem 无效**：128×128 即使在 st2（36.9KB，smem 单独算足够 2 CTA/SM）仍 **limit-Reg=1** —— 真正的
   occupancy 墙是**寄存器**（139/thread × 288 thread = 40032 regs/block；2 block = 80064 > SM 的 65536）。
   128×128 FP32 累加器（每线程巨型 fragment）把寄存器顶到 139 → 任何 stage 数都只 1 CTA/SM。
   并且 st4/st5 的 TC **更差**（更多 buffer 在 occupancy=1 时藏不了延迟，反而更慢）→ 证明瓶颈是藏延迟不是流水深度。
2. **小 tile 换 2 CTA/SM 净亏**：64×128 / 128×64 在 st3 确实到 **2 CTA/SM**（limit-Reg=2、limit-Smem=2），
   但 warp% 只从 18.6→20.2（边际），而 TC **腰斩**（TCela 53.2→32.6）、延迟 **暴涨 60%**（550→894us）——
   半个 tile 的算术强度损失远大于那点 occupancy 增益。

**Lever-2 诚实判定**：「降资源到 2 CTA/SM」在本 kernel 是**死路**。残余 15–20pp ACTIVE 差距的正解是
**保持 128×128 + 1 CTA/SM、把 consumer 拆成 2 个 math warp-group 做 ping-pong**（一组 MMA 延迟被另一组发射掩盖），
这是实质重构，**ncu 证据充分，留 round-10**。

---

## Part 4 — mainloop 段无下陷判据（layer 2）

PM-sampling 时间线 CLI 导不干净，沿用 round-8 的等价度量：ELAPSED-vs-ACTIVE 差 = mainloop 空窗。
SFASYNC 后多数 shape 的 ELAPSED 更贴近 ACTIVE（如 down·8192·Kh1024 53.9→60.3 ela，ACTIVE 59.4→66.8，
差从 5.5→6.5；qkv·2048·Kh256 ela 27.9→31.2），即削掉了 producer SF 空窗把 mainloop 往贴峰推。
gate·8192 ELAPSED≈ACTIVE（无空窗、半速发射）—— 其残差是 intensity（Lever 2），非 mainloop 下陷。

---

## Part 5 — per-shape dispatch（round-9 更新）

| 形状 | SFASYNC | KPIPE | scheduler | cfg | 依据 |
|--|--|--|--|--|--|
| 纯 fp4（r=0，任意 shape） | **1** | 0 | 非持久(qkv/down)/持久(gate) | t0 st3 | SFASYNC 治 fp4 SF 空窗，r=0 也 +0.8~5.6pp；KPIPE r=0 微降 |
| qkv / down 混合·fp8（r>0） | **1** | 1 | 非持久 | t0 st3 sw0 | SFASYNC +2~13pp TC、−2~9.7% 延迟；KPIPE +1~3pp |
| gate 混合·fp8（r>0） | **1** | 1 | 持久(P1) | t0 st3 sw8 | SFASYNC +0~3.7pp（r 小更大）；KPIPE +2~3.4pp |

SFASYNC 全 shape 全 r 开（无回退点；r=1 纯 fp8 时自动走 plain 路径，零影响）。

---

## Part 6 — 收敛判定（TC 占空比，per shape；vs Part 1 CUTLASS 端点）

| shape | TC 现状（sfa1+kpipe 后，M8192） | CUTLASS 同 shape | 判定 | 残差来源（ncu） |
|--|--|--|--|--|
| qkv | ela 40–49 / act 46–55 | 61–68 / 68–75 | **未收敛**（差 ~15–20pp act） | occupancy(1 CTA/SM, warp 18.6%, wait 2.5) |
| gate | ela 48–55 / act 50–57 | 70–75 / 72–77 | **未收敛**（最接近，mainloop 无空窗但半速） | ACTIVE intensity = occupancy |
| down | ela 54–60 / act 60–67 | 75–78 / 84–86 | **未收敛**（ACTIVE 差最大 ~18pp） | occupancy + fp4 残余 long_scoreboard |

**全局结论**：Lever 1（SF 异步化）是**本轮的实打实一步** —— bit-exact、`long_scoreboard` 全降、bank conflict/
short_scoreboard 不涨（还顺手清了 42 万次 store 冲突）、TC 两 rollup +0~13.2pp、r>0 延迟 −2~9.7%。
但**仍未贴峰**：剩余 15–20pp ACTIVE 差距经 ncu 定位为 **occupancy（寄存器墙 → 1 CTA/SM → 单 warp-group
藏不住 MMA wait≈2.5）**，且 Lever-2 的「降资源换 2 CTA/SM」已被实测判负。

### round-10 下一杠杆（ncu-grounded）
1. **ping-pong 双 math warp-group**（保持 128×128 / 1 CTA/SM，consumer 拆 2 组交替发射掩盖彼此 MMA 延迟）——
   治 ACTIVE intensity（本轮 Lever-2 证明这是唯一可行的 occupancy 路）。CUTLASS SM120 cooperative(2 warpgroup) / 
   TRT-LLM pingpong 可借鉴。
2. **fp4 段残余 long_scoreboard 收尾**：SFASYNC 后 fp4 的 lsb 仍 0.7–1.7 → 若 ping-pong 后仍是瓶颈，再评估
   SF 交错 atom 布局 + 真 TMA（需 host 重排 + consumer SF-read 适配，大改，bit-exact 风险，按收益决定是否做）。

## 产物
- `b1_tma_ext.cu`：新增 `SFASYNC` 模板开关（默认 0，保护 round-8 byte 等价）+ cp.async/mbarrier 辅助
  （`mbar_expect_noarrive`/`cp_async_mbar_arrive`/`cp_async8`）+ `sfasync` pybind 参数 + curated 实例化（LDS 块）。
- 正确性：`sfasync_check.py`（cosine 全过，`logs/r9_sfa_gate.log`）。
- TC 前后：`run_tc_sfasync.sh` + `agg_sfasync.py`（`logs/tcS/`）。
- 净赚三指标：`run_bc.sh` + `agg_bc.py`（`logs/bc/`：lsb↓ / ssb平 / bcLD=0 / bcST↓）。
- occupancy 定位：`run_occ_ncu.sh` + `run_occ_tile4.sh` + `agg_occ.py`（`logs/occ/`,`logs/occ4/`）。
- CUTLASS 端点：`run_cutlass_tc2.sh` + `agg_cut.py`（`logs/tcC2/`）。
- 延迟/excess：`r9_sfa_lat.py` + `r9_sfa_lat.csv`（clean GPU）。
- 未改动（保护）：`b1_fused_ext.cu`/`fused_gemm_ext.cu`/`nvfp4_gemm_ext.cu`/`mxfp8_gemm_ext.cu`/`mixed_gemm.py`/`pure_presf_ext.cu`。
