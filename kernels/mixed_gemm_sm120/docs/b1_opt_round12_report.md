# B1 优化第十二轮 — 报告（Lever: 搬 CUTLASS 交错 SF atom 布局 + UniversalCopy，直攻 round-11 的 ALU 主因）

> 全程 remote-only（10.208.130.90, luofan-plaquant 建/跑, luofan-ncu 跑 ncu, RTX PRO 6000 Blackwell SM120）。
> 抗中断：自包含 detached 脚本 + 从磁盘收割。
> 正确性 gate：cosine=1.000000（SFILV=1 全 48/48 通过 + SFILV=0 回退全 48/48 通过，硬门）。
> 收敛判据（项目负责人）：**TC 占空比**两 rollup 同时逼近 CUTLASS 同 shape；mainloop 不下陷。
> **本轮结论（诚实判负）：交错 SF atom 布局 bit-exact 但 TC 严重回退，直接 REFUTE round-11 的 ALU 假设。SFILV 保持默认 0（round-11 flat 出厂路径不变）。**

---

## Part 0 — 先验（CUTLASS 源码逐行确认）

- **交错 SF atom 来源**：`sm120_blockscaled_mma_builder.inl` line 178-209 内联计算 `SmemLayoutAtomSFA/SFB`
  （而非 SM100 的 `deduce_smem_layoutSFA`）；`detail/sm100_blockscaled_layout.hpp` 的
  `Sm1xxBlockScaledConfig`：`Blk_MN=128`、`Blk_SF=4`、`mnBasicBlockShape=<32,4>` stride `<16,4>`、
  `kBasicBlockShape=<SFVecSize,MMA_NSF>` stride `<0,1>`、`Blk_Elems=512`。
- **SF copy atom**：`SmemCopyAtomSF = Copy_Atom<UniversalCopy<SF>, SF>`（注释 "auto-vectorized LDS"）。
- **CUTLASS 消费路径**（`sm120_blockscaled_mma_tma.hpp` line 731-840）：
  `sSFA = make_tensor(smem, SmemLayoutSFA)` → `partition_fragment_SFA(sSFA(_,_,0), mma)` →
  `make_tiled_copy_impl(SmemCopyAtomSFA, get_layoutSFA_TV(mma), ...)` →
  `partition_S(as_position_independent_swizzle_tensor(sSFA))` → `copy(tCsSFA(_,_,kb), tCrSFA_copy_view)`。
  **关键**：CUTLASS 的 SF gmem 本身就是 atom 布局（`tile_atom_to_shape_SFA`），TMA gmem-atom→smem-atom 1:1，
  消费端再读 smem-atom —— **全链路 co-designed**。

---

## Part 1 — 我方交错布局的构造（host 产 + consumer 读，逐条对照 CUTLASS）

把 fp4（NVFP4）相的 SF smem 从 flat 改成 CUTLASS 交错 atom。**逐 stride 核对与 CUTLASS 一致**：

| 维度 | CUTLASS SM120 builder（NV: SFVec16, MMA_NSF4, Tile M128 K128） | 我方 `SFILV_NVr` | 判定 |
|--|--|--|--|
| M (128行) | `(1,32,4):(512,16,4)` → offset = (m%32)*16 + (m/32)*4 | `(32,4):(16,4)` 同 offset（前导1贡献0） | ✔ 一致 |
| K (128元素) | `((16,4),1,2):((0,1),4,512)` → offset = (e/16%4)*1 + (e/64)*512, vec=e%16 stride0 | `(16,(4,2)):(0,(1,512))` 同 offset | ✔ 一致 |
| cosize | 1024 B | 1024 B（== flat SFA_ST=BM*NBLK4） | ✔ 无 smem 预算变化 |

- **Host/producer 侧**（NON-protected `b1_tma_ext.cu` producer warp，不碰 `mixed_gemm.py`）：
  `Tensor dA = make_tensor(make_smem_ptr(dSFA), SFILV_NVw{}); dA(row,b) = SFgmem[row, c*8+b]`。
  逻辑 (row,block) 的 SF 值不变，仅物理字节序改成 atom 交错。**量化器输出的逻辑 SF (R,K/16) 不变**，
  交错步骤放在 kernel producer（非保护文件），`mixed_gemm.py` 零改动。
- **Consumer 侧**：`sSFA = make_tensor(ptr, SFILV_NVr{})` → `tt_partition_fragment_SFA(sSFA, thr_nv)` →
  `Copy_Atom<UniversalCopy<ElemSF_NV>, ElemSF_NV>`（替 `DefaultCopy`）。非-KPIPE `consume4` 与
  KPIPE `run_tile_kpipe` fp4 相都搬了。
- **fp8/MX 相保持 flat**：MX 每 chunk 只 2 个 SF block（NBLK8=KMUL=2）< CUTLASS `Blk_SF=4`，
  atom 无法整切（KMUL=2 下 `TileK/SFVec/Blk_SF = 64/32/4 = 0.5` 非整数）。**诚实 blocker**，留 flat。
- **知识点 knob**：`SFILV`（默认 0 = round-11 flat 字节等价，受保护）。SFILV 用 plain scatter producer
  （与 SFASYNC cp.async 的连续 8B 搬运不兼容），故 SFILV=1 走 sfa=0。

---

## Part 2 — 正确性（硬门，48/48，`logs/r12_sfilv_gate.log`）

`sfilv_check.py`：3 shape（qkv/down/gate）× M{2048,8192} × r{0,0.125,0.5,1} × kp{0,1}，**全部 cosine=1.000000**
（`SFILV_GATE_ALL_PASS`）。SFILV=0 回退用 `r11_gate.py` 复验，仍 **48/48 cosine=1.000000**（`R11_GATE_ALL_PASS`）——
我方 producer/consumer 重构（producer else-if 链、consume4 重排、KPIPE sfA/sfB if-constexpr）**未破坏 round-11 出厂路径**。

**格式契约逐条核对**（SFILV 只改 SF 物理字节序，由 cosine=1.0 端到端保证）：
- NVFP4：E2M1 codes、BLOCK=16、ue4m3 per-block scale、2×FP4/byte、SF 逻辑 (R,K/16) ✔ 未改
- MXFP8：E4M3、BLOCK=32、ue8m0 per-block scale、SF (R,K/32) ✔ 未改（且 MX 相仍走 flat）
- A=(M,K)/B=(N,K)；K 切分 high_fraction=1/8（qkv/gate/up Kh256·Kl1792；down Kh1024·Kl7168）✔
- reorder-before-quant [low|main|high] block 对齐 ✔；共享 FP32[4] 寄存器累加器（NVFP4→MXFP8，无中间落 smem/gmem）✔
- epilogue out=acc·s_x·s_w→fp16 ✔。唯一对 CUTLASS 的偏离仍只有双格式共享累加器缝 + mbarrier ring。

---

## Part 3 — 指令混合 before/after（本轮核心证据，ncu `logs/tcSFILV/`，`agg_sfilv.py`）

i0 = flat（round-11，sfa=0）；i1 = interleaved（SFILV=1，sfa=0）；**同 producer、同 kpipe，仅 SF smem 布局 + copy atom 差异**。
CUTLASS 同 shape fp4 端点取自 `logs/tcIMIX`。

| cell (M8192) | TCact i0→i1 | alu/tc i0→i1 | lsu/tc i0→i1 | lsb i0→i1 | inst(M) i0→i1 | dur_us i0→i1 | CUTLASS fp4 TCact / alu/tc |
|--|--|--|--|--|--|--|--|
| qkv r0 kp0 | **43.4→16.5** | 3.36→**3.79** | 1.06→1.41 | 2.34→**5.52** | 32.9→50.8 | 130.7→**326.2** | 70.4 / 1.16 |
| qkv r0 kp1 | 42.8→16.6 | 3.76→4.07 | 1.31→1.41 | 2.48→5.48 | 36.6→52.1 | 132.8→324.8 | 70.4 / 1.16 |
| down r0 kp0 | **54.3→17.8** | 3.04→**3.48** | 0.88→1.22 | 1.47→**5.23** | 114→188 | 402.9→**1202.9** | 87.1 / 0.80 |
| down r0 kp1 | 54.3→18.1 | 3.45→3.76 | 1.13→1.22 | 1.59→5.17 | 130→192 | 405.6→1185.6 | 87.1 / 0.80 |
| gate r0 kp0 | 48.8→18.6 | 3.46→3.90 | 1.06→1.41 | 2.21→5.85 | 131→190 | 420.5→1065.0 | 73.1 / 1.13 |
| gate r0 kp1 | 48.1→18.9 | 3.99→4.29 | 1.31→1.41 | 2.11→5.56 | 146→198 | 425.1→1048.5 | 73.1 / 1.13 |
| qkv r0.125 kp1 | 45.0→31.1 | 3.88→4.13 | 1.29→1.36 | 2.22→3.08 | 41→47 | 139.2→198.4 | — |
| down r0.125 kp1 | 59.2→28.9 | 3.58→3.82 | 1.12→1.19 | 1.24→3.29 | 146→182 | 419.6→834.0 | — |
| gate r0.125 kp1 | 52.5→36.6 | 4.06→4.29 | 1.28→1.36 | 1.71→2.63 | 163→183 | 438.3→618.7 | — |

**发现（证据，非猜测）—— 与 round-11 预期完全相反：**
1. **TC 两 rollup 不升反而暴跌**：纯 fp4 TCact 43-54 → 16-19（−27pp），ELAPSED 同步跌。目标是 → CUTLASS 70-87，现在反向。
2. **alu/tc 不降反升**：3.0-4.0 → 3.5-4.3（全部 ↑）。round-11 预期 → ~1，实测向上。
3. **long_scoreboard 爆炸**：纯 fp4 lsb 1.5-2.3 → 5.2-5.9（×2-3.5），lsu/tc 也升（1.06→1.41）。总指令 +54%。
4. **延迟 2.5-3×**：qkv 131→326us，down 403→1203us。

**机理定位（ncu-grounded）**：把 CUTLASS 的 atom smem 布局嫁接到**我方自研的 `get_layoutSFA_TV` TV 划分 + UniversalCopy**，
**不但没向量化，反而让 consumer 的 SF LDS 变成散射/高延迟**：atom 的 32×4 M-交错（stride 16,4）+ 第二个 MMA 的
K-stride=512，使一个 warp 的 SF 读跨越整个 1024B buffer（局部性崩坏）→ long_scoreboard 爆炸、地址数学更多
（alu/tc↑、inst +54%），而非更少。CUTLASS 的低 alu/tc 来自它**完整 co-designed 的 TMA gmem-atom→smem-atom→
专属 SF-consumer 流水**，**无法只搬 smem 布局那一块就移植到我方 TV**。

---

## Part 4 — 与 round-11 两个 lever 的交叉综合（ALU 主因是内在的，非 SF 布局）

round-11 提名了两条治 ALU 残差的路；**本轮把第二条（交错布局）也实测判负**：

| lever | bit-exact? | alu/tc 效果 | TC 效果 | 判定 |
|--|--|--|--|--|
| round-11 KPIPE hoist | ✔ | **+0.4~0.5（升）** | TCact +0.3~4.4（藏延迟副作用） | 不降 ALU；但抬 TC → 已出厂 |
| round-12 交错 SF atom | ✔ | **+0.3~0.5（升）** | **TCact −27pp、dur 2.5-3×** | 判负，SFILV 默认 0 |

**两条独立的 bit-exact 实验都把 ALU 抬高** → **~3.5 的 alu/tc 是我方自研「双格式共享累加器 + 逐 MMA 的 SF 广播」
consumer 的内在属性，不是 flat SF smem 布局的产物**。flat 布局在三者里对 TC 占空比反而是最好的。
CUTLASS 的低 alu/tc 是其整条 warp-specialized TMA SF 流水 + MMA fragment handling 的性质，不是 SF smem atom 单点能移植的。

---

## Part 5 — per-shape dispatch（round-12：维持 round-11，SFILV 默认 0）

| 形状 | KPIPE | SFASYNC | SFILV | PINGPONG | LDBW | scheduler | cfg | 依据 |
|--|--|--|--|--|--|--|--|--|
| 纯 fp4（r=0） | 1 | 1 | **0** | 0 | 0 | 非持久(qkv/down)/持久(gate) | t0 st3 | SFILV=1 判负（Part 3） |
| qkv/down 混合（r>0） | 1 | 1 | **0** | 0 | 0 | 非持久 | t0 st3 sw0 | 同上 |
| gate 混合（r>0） | 1 | 1 | **0** | 0 | 0 | 持久(P1) | t0 st3 sw8 | 同上 |

SFILV=1 保留为 knob（bit-exact，可复现判负），默认 0 = round-11 出厂路径。SFILV 与 SFASYNC 互斥（scatter vs cp.async）。
**SF-via-TMA bonus（round-9 原意）**：交错布局本是让 SF 可 TMA 的前提，但既然交错布局本身在我方 consumer 上判负，
SF-via-TMA 失去价值（即便 TMA 把 SF 搬进交错 smem，consumer 读仍散射）——**不再追**。

---

## Part 6 — 收敛判定（TC 占空比，per shape；vs CUTLASS 端点）

| shape | TC 现状（M8192 flat，仍 round-11 出厂） | CUTLASS 同 shape | 判定 | 本轮对残差的新认识 |
|--|--|--|--|--|
| qkv | act 42-45 | 70-76 | **未收敛**（差 ~25-30pp） | ALU 残差**内在**，非 SF 布局；交错布局判负 |
| gate | act 48-53 | 73-77 | **未收敛**（差 ~20-25pp） | 同上 |
| down | act 54-59 | 84-87 | **未收敛**（差 ~25-30pp） | 同上 |

**全局结论**：本轮把项目负责人指定的"THE convergence shot"（交错 SF atom + UniversalCopy）**完整 bit-exact 落地并实测**，
结果是**决定性判负**：它不降 ALU、反而让 TC 占空比暴跌 27pp、延迟 2.5-3×。结合 round-11 KPIPE 同向判负，
**证明 ~3.5 alu/tc 的残差是我方单-warpgroup 自研 consumer（双格式共享累加器 + 逐-MMA SF 广播 `make_zip_tensor`）的内在
开销，不是 SF smem 布局能治的**。这关闭了 round-9→round-12 一直悬而未决的"换 SF 布局"方向。

### round-13 下一杠杆（ncu-grounded，按预期收益排序）
1. **停止攻 SF 读路径**，转攻 ALU 的真实源头：逐-MMA 的 `make_zip_tensor(tCrA, tCrSFA)` SF 广播 + 共享累加器 fragment 的
   per-MMA 索引/广播数学（我们被允许保留的"缝"，但它正是 per-MMA ALU 的来源）。可评估：把 SF 预广播进寄存器一次、
   或用 `filter_zeros` 消除 stride-0 vec 的冗余广播读，减少每 MMA 的 ALU，而**不动** smem 布局（已证 smem 布局方向死路）。
2. **转攻正交的 occupancy/pingpong 轴**（round-10 deferred）：既然单-warpgroup 自研 consumer 的 alu/tc 结构上到不了 CUTLASS，
   TC 占空比的提升应走"双 math warp-group 交替发射掩盖彼此 MMA 延迟"（128×128 / 1 CTA/SM），而非继续榨 SF 读。
3. **SASS 源级确认**（`--set full --import-source`）把交错布局那 +18M 指令落到具体源行，确证散射 LDS 的来源
   （为 lever 1 的 `filter_zeros` 方案提供精确靶点）。

## 产物
- `b1_tma_ext.cu`：新增 `SFILV` 模板 knob（默认 0 = round-11 flat 字节等价，受保护）+ 交错 NV SF 布局 typedef
  （`SFILV_M/_Kw/_NVw/_Kr/_NVr`，逐 stride 对齐 CUTLASS SmemLayoutAtomSFA）+ producer scatter + consume4/KPIPE
  interleaved 读（UniversalCopy）+ pybind `sfilv` 参数。SFILV=0 数据路径零改动 → cosine=1.0 由构造保证 + 实测 48/48。
- 正确性：`sfilv_check.py`（`logs/r12_sfilv_gate.log`，`SFILV_GATE_ALL_PASS` 48/48 cosine=1.000000）；
  `r11_gate.py` 复验 SFILV=0 回退（`R11_GATE_ALL_PASS` 48/48）。
- 指令混合 before/after：`run_tc_sfilv.sh` + `agg_sfilv.py`（`logs/tcSFILV/`：i0 flat vs i1 interleaved，TC 两 rollup +
  alu/lsu/tc + lsb + inst + dur + regs）；基线 `logs/tcIMIX/`（CUTLASS 端点 + round-11 基线，`agg_imix.py`）。
- `prof_tc.py` 加 `sfilv` argv（第 16 位）。
- `candidates.jsonl`：+2 条（SFILV 交错布局判负；ALU 内在性综合判定）。
- 未改动（保护）：`b1_fused_ext.cu`/`fused_gemm_ext.cu`/`nvfp4_gemm_ext.cu`/`mxfp8_gemm_ext.cu`/`mixed_gemm.py`/`pure_presf_ext.cu`。
