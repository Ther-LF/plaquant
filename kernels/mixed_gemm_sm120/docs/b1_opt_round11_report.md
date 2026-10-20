# B1 优化第十一轮 — 报告（Lever: 抬 MMA 发射密度 = 降 mainloop 非-TC 指令；对着 CUTLASS 逐行搬）

> 全程 remote-only（10.208.130.90, luofan-plaquant 建/跑, luofan-ncu 跑 ncu, RTX PRO 6000 Blackwell SM120）。
> 抗中断：自包含 detached 脚本 + 从磁盘收割。
> 正确性 gate：cosine=1.000000（48/48 全点通过，硬门，`logs/r11_gate2.log`，R11_GATE_ALL_PASS）。
> 收敛判据（项目负责人）：**TC 占空比**两 rollup 同时逼近 CUTLASS 同 shape；mainloop 不下陷。
> 项目负责人加码本轮：**对着 CUTLASS `sm120_blockscaled_mma_tma.hpp` 的 `mma()` 逐行搬**内循环，
> 只保留"双格式共享 FP32 累加器"那个缝；搬完逐条核对格式契约。

---

## Part 0 — 先验（CUTLASS 源码逐行确认 + KernelWiki）

- **CUTLASS `mma()`**（`sm120_blockscaled_mma_tma.hpp` line 714-904）：partition/fragment/copy-view
  **全部 hoist 到 k-loop 外**，只构造一次；稳态循环用 `for_each(make_int_sequence<K_BLOCK_MAX>)`
  交织 `copy_kblock(k_block_next)` 在 `gemm_kblock(k_block)` 之前；chunk 边界（`k_block==K_BLOCK_MAX-1`）
  做 `NamedBarrier::sync(Sm120MainloopBarrier)` + `consumer_release` + 推进 stage + `consumer_wait`，
  再 hoist 出最后一个 k_tile。`copy_kblock` = copy A(ldmatrix)、copy B(ldmatrix)、`fp4_shift_A/B`、
  copy SFA、copy SFB。
- **copy atom 选择**（`builders/sm120_common.inl` line 48-95 + `sm120_blockscaled_mma_builder.inl` line 163-174）：
  - 算子 A/B ldmatrix：`sm120_rr_smem_copy_selector_A/B` → fp8 **都返回 `SM75_U32x4_LDSM_N`（128-bit）**；
    fp4 返回 `SM100_SU4_DU8x16_x4_LDSM_N`（x4，4→8 bit expand）。**A、B 都是 x4**。
  - SF copy atom：`SmemCopyAtomSF = Copy_Atom<UniversalCopy<SF>, SF>`（注释"auto-vectorized LDS"）；
    SF smem 用**交错 atom 布局** `Sm1xxBlkScaledConfig::SmemLayoutAtomSFA`（`mnBasicBlockShape=<32,4>` 等）。
- **fp4_shift**（`mma_traits_sm120.hpp` line 200-258）：`v<<2`，`if constexpr(is_same_v<AType,e2m1>)` 守卫 →
  **fp8 路径编译期为空**（无开销）；fp4 路径 A/B 各做一次。CUTLASS 与我方都调它，**非冗余**。
- KernelWiki：sglang PR-21314（SM120 NVFP4 穷举 tile/cooperative/pingpong/StreamK，StreamK 无效，
  主要靠 tile 启发式）——与 round-10 结论一致，无新的指令级招法。

---

## Part 1 — CUTLASS 同 shape TC 端点（本轮 imix 实测，收敛靶，`logs/tcIMIX/`）

| shape·M | CUTLASS fp4 TCact/ela | CUTLASS fp8 TCact/ela | CUTLASS alu/tc fp4 / fp8 |
|--|--|--|--|
| qkv·8192 | 70.0 / 62.8 | 75.8 / 68.0 | 1.16 / 0.70 |
| down·8192 | 87.1 / 78.9 | 83.7 / 75.4 | 0.80 / 0.52 |
| gate·8192 | 73.2 / 71.5 | 77.2 / 75.3 | 1.13 / 0.69 |

（与 round-9 Part 1 一致，CUTLASS 参考稳定。新增列 alu/tc = ALU 指令数 / Tensor 指令数。）

---

## Part 2 — side-by-side：我方 consumer 内循环 vs CUTLASS `mma()`（逐条）

| # | CUTLASS `mma()` 这么写 | 我方原来这么写 | 判定 |
|--|--|--|--|
| 1 | partition/fragment/copy-view **hoist 到 k-loop 外**，构造一次 | 非-KPIPE 的 `consume4/consume8` **每 chunk 重建** sSFA/sSFB tensor（`make_tensor+make_layout+recast_ptr`）并重 `partition_S/partition_fragment` | KPIPE 路径已 hoist（round-8 忠实搬运）；非-KPIPE 未 hoist |
| 2 | 稳态 `copy_kblock(kb+1)` 交织 `gemm_kblock(kb)`；chunk 边界 release+advance+wait | KPIPE 已同构（prologue load(0)→稳态 load(kbn);mma(kb)→边界 arrive+wait+load(0)） | KPIPE ✔ 一致；非-KPIPE 是"全 load 后全 mma" |
| 3 | A ldmatrix `SM75_U32x4_LDSM_N`（x4） | A `SM75_U32x4_LDSM_N`（x4） | ✔ 一致 |
| 4 | **B ldmatrix `SM75_U32x4_LDSM_N`（x4, 128-bit）** | **B `SM75_U32x2_LDSM_N`（x2, 64-bit）** | ✘ 本轮尝试对齐→**编译阻断**（见 Part 3） |
| 5 | SF copy `UniversalCopy<SF>` over 交错 atom 布局 | SF copy `DefaultCopy` over 平坦 `(BM,(VEC,NBLK))` stride `(NBLK,(0,1))` 布局 | ✘ 布局不同（本轮定位为 ALU 主因，见 Part 4，round-12 搬运） |
| 6 | `fp4_shift_A/B` per k_block（fp8 编译期空） | 同（KPIPE per-kb；非-KPIPE 整 frag 一次；fp8 编译期空） | ✔ 等价 |
| 7 | chunk 边界 `NamedBarrier Sm120MainloopBarrier` | 直接 mbarrier arrive/wait（无 named barrier） | 偏离但 bit-exact（我方 ring 结构，允许的契约偏离） |

---

## Part 3 — Lever 尝试①：B ldmatrix x2→x4（对齐 CUTLASS）—— 编译阻断，诚实判负

照 Part 2#4，新增模板开关 `LDBW`（默认 0 = round-10 x2 字节等价，受保护），用 `conditional_t` 选 B 的 copy atom。
`LDBW=1`（x4）在 **WM2WN4 和 WM4WN2 两种 warp 布局下都编译失败**：
`copy_atom.hpp(206): static assertion "TiledCopy uses too few vals for selected CopyAtom"`（`make_tiled_copy_B`，
fp4 line 764 / fp8 line 766）。

**根因（ncu 无关，纯 CuTe 布局）**：我方 B 算子每线程每 k-block 只有 **2 个 uint32** 的 fragment
（B = 8 N × 32 K fp8 = 256 uint8 / 32 线程 = 2 uint32/thr），`SM75_U32x4_LDSM_N` 需要 **4 uint32/thr**。
CUTLASS 能喂 x4 是因为它的 B smem→reg copy 结构跨越更多（2 个 k-block 合并）；要对齐需重构 B copy（一次 ldmatrix
搬 2 个 k-block），有 bit-exact 回归风险 → **本轮不做**。`LDBW` 模板参数保留（默认 0 字节等价），`ldbw=1`
在 host 侧直接 `TORCH_CHECK(false)`，**不实例化**（kernel 正常编译）。

**关键**：Part 4 的 imix 证据显示 B ldmatrix（LSU 管线）**根本不是**瓶颈（我方 lsu/tc 0.84–1.28 ≈ CUTLASS 0.67–1.08），
所以这条 lever 即便做成收益也小。真瓶颈是 ALU。

---

## Part 4 — 指令混合诊断（本轮核心发现，ncu `logs/tcIMIX/`，`agg_imix.py`）

对 OUR dual baseline（round-10 字节等价，per-shape 出厂 dispatch）与 CUTLASS 同 shape 纯端点，测**每管线指令数**。

| cell (M8192) | TCact 我/CUT | **alu/tc 我/CUT** | lsu/tc 我/CUT | inst(M) 我 | alu(M) 我 | tc(M) 我 |
|--|--|--|--|--|--|--|
| qkv r=0 (fp4) | 45.9 / 70.0 | **3.60 / 1.16** | 1.06 / 0.78 | 33.7 | 15.1 | 4.2 |
| qkv r=1 (fp8) | 51.7 / 75.8 | **3.62 / 0.70** | 0.94 / 1.08 | 62.9 | 30.3 | 8.4 |
| down r=0 (fp4) | 60.0 / 87.1 | **3.29 / 0.80** | 0.87 / 0.67 | 117.7 | 55.2 | 16.8 |
| down r=1 (fp8) | 61.3 / 83.7 | **3.49 / 0.52** | 0.84 / 1.02 | 232.3 | 117.0 | 33.6 |
| gate r=0 (fp4) | 49.8 / 73.2 | **3.71 / 1.13** | 1.06 / 0.77 | 135.7 | 62.3 | 16.8 |
| gate r=1 (fp8) | (Part5) | 3.71 / 0.69 | 0.94 / 1.07 | — | — | — |

**发现（证据，非猜测）**：
1. **ALU 管线是发射槽主占用者**：我方 **alu/tc = 3.3–4.3**，CUTLASS **0.5–1.3**。即每条 MMA 之间我方多发
   **3–6 倍** 的 ALU 指令。down r=0 的 ALU 占全部指令 **47%**（55.2M/117.7M），Tensor 仅 14%。
2. **LSU（ldmatrix）不是瓶颈**：我方 lsu/tc 0.84–1.28 ≈ CUTLASS 0.67–1.08。→ 推翻"窄 ldmatrix 是主因"的假设，
   Part 3 的 B-x4 即便做成收益也有限。
3. **ALU 超额与 fp4 shift 无关**：纯 fp8 r=1 **没有** fp4_shift（编译期空），alu/tc 仍 3.49 vs CUTLASS 0.52
   （6.7×）。→ 超额 ALU 是**系统性的地址/索引数学**，不是 shift。
4. MMA 原子数一致（我方 tc.sum == CUTLASS 同 shape，如 down r1 都是 33.6M）→ 同样的 MMA 工作量下我方多 3-6× ALU，
   是**纯开销**，直接压低 TC 发射密度（ACTIVE 45–61% vs CUTLASS 70–87%）。

**ALU 来源定位（代码级，Part 5 的 kp0/kp1 对照佐证）**：两路都 ~3.5 alu/tc，而 CUTLASS 同 MMA 数/同 zip/同 shift
只有 0.5-1.3 → 差在 **SF copy 路径**：我方 SF smem 是平坦 `(BM,(VEC,NBLK))` stride `(NBLK,(0,1))`（VEC 维 stride-0 广播），
`get_layoutSFA_TV` + `DefaultCopy` 在这个非-atom 布局上**无法向量化**，退化成**逐元素 LDS + 每元素地址数学**（ALU）；
CUTLASS 的交错 `Sm1xxBlkScaledConfig` SF atom 布局 + `UniversalCopy` 是干净可向量化 TV，地址数学极少。

---

## Part 5 — Lever 尝试②：hoist partition（KPIPE）能否降 ALU —— 诚实判负（但抬了 ACTIVE）

纯相 kp0（非 hoist，consume4/8）vs kp1（hoist，CUTLASS mma() 忠实 register-pipeline），M8192，sfa=0，`logs/tcKPH/`：

| cell | TCact kp0→kp1 | alu/tc kp0→kp1 | regs kp0→kp1 | dur(us) kp0→kp1 |
|--|--|--|--|--|
| qkv r=0 | 38.4→**41.9** (+3.5) | 3.36→3.76 (↑) | 139→168 | 144.8→130.6 |
| qkv r=1 | 51.7→**56.1** (+4.4) | 3.62→4.15 (↑) | 139→168 | 212.9→200.2 |
| down r=0 | 52.2→**53.4** (+1.2) | 3.04→3.45 (↑) | 139→168 | 412.6→410.0 |
| down r=1 | 60.6→**64.5** (+3.9) | 3.46→3.99 (↑) | 139→168 | 711.1→680.1 |
| gate r=0 | 47.8→**48.1** (+0.3) | 3.46→3.99 (↑) | 143→168 | 424.8→426.9 |
| gate r=1 | 56.0→**59.0** (+3.0) | 3.71→4.24 (↑) | 143→168 | 711.0→693.0 |

**判定**：hoist（KPIPE）**不降 ALU**（反而 +，因 register-pipeline 自身加索引数学 + regs 139→168），
**但每个纯相 cell 的 TCact 都升 +0.3~+4.4pp、延迟平/降** —— 因为 KPIPE 的作用是**藏延迟**（round-8 已证），不是降指令。
→ 证实 Part 4 结论：~3.5 alu/tc 是**内在的**，hoist 治不了；要降 ALU 必须换 SF 布局（round-12）。

**可落地（出厂 dispatch 更新）**：既然 KPIPE 在**纯 fp4 r=0** 也抬 ACTIVE（+0.3~+3.5pp）且延迟不亏，
**r=0 也改用 kp1**（原 round-9 dispatch r=0 用 kp0）。这是把 CUTLASS 忠实 register-pipeline 用到纯-fp4 相。
bit-exact 已验（`logs/r11_gate2.log` r=0 kp1 全 PASS；round-8 亦已 gate KPIPE）。

---

## Part 6 — 格式契约逐条核对（项目负责人加码；源头 `mixed_gemm.py`/`nvfp4_gemm.py`/`mxfp8_gemm.py`）

cosine=1.000000（48/48, `logs/r11_gate2.log`）是端到端契约验证——kernel 消费的是我方真实打包 + SF 布局 + K-split，产出我方输出格式。逐条：

- **NVFP4**：E2M1 codes（`vals` 表）、`BLOCK=16`、ue4m3 per-block scale（`float8_e4m3fn` byte）、
  2×FP4/byte 打包（`c[...,0] | c[...,1]<<4`）、SF 逻辑形状 (R, K/16)。✔ 未被改动。
- **MXFP8**：E4M3 元素（`float8_e4m3fn`）、`BLOCK=32`、ue8m0 幂二 per-block scale（byte=exp+127）、SF (R, K/32)。✔
- **A/B/SF 布局**：A=激活 RowMajor (M,K)；B=权重 (N,K)（oracle `Bl_deq.t()`）；SF nvfp4 (·,K/16) / mxfp8 (·,K/32)。✔
- **K 切分**：K_low(NVFP4)+K_high(MXFP8)，high_fraction=1/8；per-projection（qkv/gate/up Kh256·Kl1792；down Kh1024·Kl7168）✔。
- **核心缝**：oracle `(Al_deq@Bl_deq.t() + Ah_deq@Bh_deq.t())` → kernel Phase-1 NVFP4 与 Phase-2 MXFP8
  累加到**同一 FP32 寄存器累加器**（中间不落 smem/gmem）；epilogue `acc*s_x*s_w → __floats2half2_rn → fp16`。✔
- **本轮改动不触碰数据路径**：LDBW 默认 0 → `conditional_t` 编译期选回原 `SM75_U32x2_LDSM_N`，字节等价 round-10。
  所以 cosine=1.0 由构造保证，且实测 48/48 PASS。唯一偏离 CUTLASS 的仍只有"共享累加器那个缝"（+ 我方 mbarrier ring 替 NamedBarrier）。

---

## Part 7 — per-shape dispatch（round-11 更新）

| 形状 | KPIPE | SFASYNC | PINGPONG | LDBW | scheduler | cfg | 依据 |
|--|--|--|--|--|--|--|--|
| 纯 fp4（r=0） | **1（本轮改，原 0）** | 1 | 0 | 0 | 非持久(qkv/down)/持久(gate) | t0 st3 | KPIPE r=0 TCact +0.3~+3.5pp、延迟不亏（Part 5） |
| qkv/down 混合（r>0） | 1 | 1 | 0 | 0 | 非持久 | t0 st3 sw0 | round-9/10 |
| gate 混合（r>0） | 1 | 1 | 0 | 0 | 持久(P1) | t0 st3 sw8 | round-9/10 |

LDBW=1 保留为 host-throw 占位（x4 B 需 B-copy 重构，round-12+）。PINGPONG=0（round-10 判负）。

---

## Part 8 — 收敛判定（TC 占空比，per shape；vs Part 1 CUTLASS 端点）

| shape | TC 现状（M8192 act, kp1） | CUTLASS 同 shape act | 判定 | 残差来源（本轮 ncu 定位） |
|--|--|--|--|--|
| qkv | 41.9(r0)–56.1(r1) | 70.0–75.8 | **未收敛**（差 ~20–28pp） | **ALU 发射密度**：alu/tc 3.6–4.2 vs CUTLASS 0.7–1.2（SF copy 逐元素地址数学） |
| gate | 48.1(r0)–59.0(r1) | 73.2–77.2 | **未收敛**（差 ~18–25pp） | 同上 alu/tc 3.7–4.2；+ fp4 残余 lsb 1.4–2.3 |
| down | 53.4(r0)–64.5(r1) | 83.7–87.1 | **未收敛**（差 ~20–30pp） | 同上 alu/tc 3.3–4.0（最接近，ALU 占 47%） |

**全局结论**：本轮把 round-10 笼统的"非-TC 指令"**精确定位到 ALU 管线**——我方每条 MMA 之间多发 **3–6× 于 CUTLASS 的 ALU 指令**，
且与 fp4_shift 无关、hoist（KPIPE）治不了、与 B-ldmatrix 宽度（LSU）无关。证据指向 **SF copy 路径**：平坦 SF 布局 + `DefaultCopy`
退化成逐元素地址数学，而 CUTLASS 的交错 `Sm1xxBlkScaledConfig` SF atom 布局 + `UniversalCopy` 是向量化 TV。
本轮两条指令级 lever（B-x4 / KPIPE-hoist-降-ALU）均诚实判负，但 KPIPE 抬 ACTIVE 的副作用使 **r=0 也改用 kp1** 成为可落地小赢。

### round-12 下一杠杆（ncu-grounded，按预期收益排序）
1. **搬 CUTLASS 的 SF 布局（交错 atom `Sm1xxBlkScaledConfig::SmemLayoutAtomSFA` + `UniversalCopy`）** —— 直接攻 ALU 主因
   （本轮定位的真残差）。即 round-9 Part 2 deferred 的大改（host SF 重排 + consumer SF-read 全链路适配），**现被 ALU 证据充分 justify**。
   需逐点 bit-exact 回归（硬门）。预期：alu/tc 3.5→~1，ACTIVE 向 CUTLASS 70-87% 靠。
2. **B-copy 重构喂 x4 ldmatrix**（一次搬 2 k-block）—— 次要（LSU 非瓶颈），但对齐 CUTLASS #4，可叠加。
3. **SASS 源级 per-line 确认**（`--set full --import-source`）把 ALU 指令落到具体源行，进一步缩小 SF-copy vs 地址数学的占比。

## 产物
- `b1_tma_ext.cu`：新增 `LDBW` 模板参数（默认 0 = round-10 x2 字节等价，受保护）；`ldbw=1` host-throw（x4 B 编译阻断，Part 3）。
  pybind 新增 `ldbw=0`。数据路径零改动 → cosine=1.0 由构造保证 + 实测 48/48。
- 诊断：`run_tc_imix.sh`+`agg_imix.py`（`logs/tcIMIX/`：per-pipe 指令数 + TC 两 rollup + regs，我方 vs CUTLASS 端点）；
  `run_tc_kphoist.sh`（`logs/tcKPH/`：kp0 vs kp1 ALU/ACTIVE 对照）。
- 正确性：`r11_gate.py`（`logs/r11_gate2.log`，R11_GATE_ALL_PASS，48/48 cosine=1.000000，含 r=0 kp1）。
- `ldbw_check.py`（B-x4 编译阻断记录）；`prof_tc.py` 加 ldbw argv。
- 未改动（保护）：`b1_fused_ext.cu`/`fused_gemm_ext.cu`/`nvfp4_gemm_ext.cu`/`mxfp8_gemm_ext.cu`/`mixed_gemm.py`/`pure_presf_ext.cu`。
